# 架构深入分析（2026-11-24）

本文是对当前代码库的一次结构性分析，不是路线图。所有数字都来自这一轮在工作区里跑出的命令，命令附在每节末尾，便于复算。

口径说明：**代码行数只统计非测试文件**（`ls internal/*/*.go cmd/orbix/*.go | grep -v '_test.go$'`），测试行数为 `*_test.go` 之和。函数计数用 `go/ast` 遍历 `*ast.FuncDecl`，方法计入、测试文件不计入。

## 一、规模与形状

| 包                  | 代码行    | 函数（含方法） | 顶层函数 | 测试行    | 测试/代码 |
| ------------------- | --------- | -------------- | -------- | --------- | --------- |
| `internal/tokenize` | 4811      | 137            | 108      | 6945      | 1.44      |
| `internal/llama`    | 4686      | 121            | 39       | 12191     | 2.60      |
| `internal/server`   | 4033      | 89             | 39       | 9628      | 2.39      |
| `cmd/orbix`         | 3429      | 72             | 69       | 7000      | 2.04      |
| `internal/chat`     | 2333      | 67             | 32       | 3800      | 1.63      |
| `internal/engine`   | 1693      | 36             | 13       | 5322      | 3.14      |
| `internal/gguf`     | 1459      | 49             | 12       | 2827      | 1.94      |
| `internal/quant`    | 873       | 27             | 20       | 3055      | 3.50      |
| `internal/gguftest` | 828       | 22             | 14       | 372       | 0.45      |
| `internal/otel`     | 784       | 35             | 14       | 1639      | 2.09      |
| `internal/kernel`   | 667       | 28             | 20       | 1835      | 2.75      |
| `internal/f16`      | 154       | 4              | 4        | 380       | 2.47      |
| **合计**            | **25750** | **687**        | **384**  | **54994** | **2.14**  |

三个观察：

1. **仓库的重心在 token 与提示词，不在算子。** `tokenize` + `chat` 合计 7144 行，是 `kernel`（667 行）的 10.7 倍；`llama` 里真正属于"数学生成"的部分（attention、feedForward、rope、logprob）约 900 行。这与 README 的自述一致——这是学习工程，难点被有意放在"文件格式与模型格式"上。
2. **测试密度随"契约有多硬"上升，不随代码量上升。** `quant`（3.50）、`engine`（3.14）、`kernel`（2.75）最高，因为它们和外部参考逐值比对；`tokenize`（1.44）与 `chat`（1.63）最低，因为 `pre.go` 是 34 个预分词族的表驱动实现，单族覆盖成本高。`gguftest`（0.45）是测试夹具包，这个比值没有信息量。
3. **`cmd/orbix` 的 69 个顶层函数里只有 6 个是命令入口**，其余是渲染、校验与错误解释。CLI 不是薄壳，它持有真实的业务规则（见第五节）。

复算命令：

```bash
go run /tmp/count2.go            # 每包行数与函数数（AST）
grep -c "func Test" internal/*/*_test.go cmd/orbix/*_test.go
```

## 二、依赖图：分层是真的，渗透是真的

`go list` 给出的实际内部导入边：

```text
cmd/orbix      → engine, gguf, llama, otel, server, tokenize
internal/server → engine, llama, otel, tokenize
internal/engine → chat, gguf, llama, otel
internal/llama  → gguf, kernel, quant
internal/quant  → f16, gguf
internal/tokenize → gguf
internal/chat, kernel, f16, otel → （无内部依赖）
```

无环，且 kernet 层不依赖任何上层。**但 graph 里有两个不该出现在上层的东西**：

- `internal/server` 直接导入 `llama`，用到 9 个符号：`Greedy`、`Temperature`、`Biased`、`Sampler`、`TokenBias`、`TokenLogprobs`、`Alternative`、`ErrPoolExhausted`、`Config` 的字段。其中 `ErrPoolExhausted` 是**存储层错误穿过调度层**：`server.go:1680` 和 `:1989` 用 `errors.Is(err, llama.ErrPoolExhausted)` 把它翻成 400/503，而 `engine` 自己也在 `engine.go:891` 做同样的判断。一个错误类型需要两个上层分别识别，说明"池耗尽"没有在 engine 边界上被翻译成领域错误。
- `internal/engine` 的 `Request` 结构体字段类型是 `llama.Sampler`，`ChatRenderer` 住在 engine 包里却只依赖 `chat` 与 `gguf`。把渲染器放进调度包，使它成为唯一一个"engine 里不做调度"的文件（209 行）。

复算命令：

```bash
go list -f '{{.ImportPath}}: {{join .Imports " "}}' ./internal/... ./cmd/...
grep -rho "llama\.[A-Z][A-Za-z]*" internal/server/*.go | sort | uniq -c
```

## 三、最大的结构问题：一次前向被写了三遍

`internal/llama` 有**三条各自独立、逐行重复**的前向实现：

| 路径         | 入口                                                            | 规模         | 使用方                                                               |
| ------------ | --------------------------------------------------------------- | ------------ | -------------------------------------------------------------------- |
| 单 token     | `Forward` → `attention`/`feedForward`/`finish`                  | 77+30+20 行  | 只有 `Model.Generate`，而 `Generate` 只有 `cmd/orbix run` 一个调用方 |
| 批量 prefill | `Prefill` → `attentionBatch`/`feedForwardBatch`/`finishBatch`   | 133+28+25 行 | engine 的 prefill 阶段                                               |
| 批量 decode  | `DecodeStep` → `decodeBatch.attention`/`.feedForward`/`.finish` | 84+33+22 行  | engine 的 decode 阶段                                                |

`feedForward` 三个版本的核心体逐字相同：

```text
MatMulTransposed(gate, xn, wGate) → MatMulTransposed(up, xn, wUp)
→ activateGate(gate) → MulInPlace(up, gate) → MatMulTransposed(attnOut, up, wDown)
→ postNormRows(attnOut, ffnPostNorm) → AddInPlace(x, attnOut)
```

差别只有两点：单 token 版用 `MatVec`，两个批量版用 `MatMulTransposed`；归一化一个走 `kernel.RMSNorm` 循环、一个走 `normRows`。`moeRowsFromBatch` 与 `moeRowsFromDecode` 也是逐字相同的两个适配函数（各 8 行）。

代价不是行数，是**改一处必须记得改三处**，而仓库的文档已经记录了被这件事咬过的证据：

- `llama/doc.go` 写着 Qwen3 的 QK 归一化"应用在四个地方——单 token、批量 prefill、批量 decode，以及复用提示词重算的 cached query——每一处都有对照参考的测试"。
- `docs/models.md` 记录过一个真实缺陷：Gemma 2 的 output softcap "被加到了单 token 和批量 decode 的投影上，但没加到 prefill 的，于是每次回复的第一个 token 来自另一个分布"。这正是三份实现的必然产物。
- `addBias` 有三个版本（`attention.go:294`、`batch.go:241`、`batch.go:743`）；`finish` 有三个版本，每个都要记得调 `kernel.Softcap(b.logits, cfg.FinalSoftcap)`。

**更正（当轮复查时发现，上一轮的建议是错的）。** 这里原本写着"单 token 路径只剩 `run` 一个用户，把 `run` 迁到 engine 就能删掉这条腿约 190 行，把扇出从 3 降到 2"。前半句关于**生产**调用点是事实，结论却错了：单 token 路径不是遗留物，而是整个包唯一的**独立参考实现**。

- `internal/llama/reference_test.go` 与 `reference_real_test.go`（后者是 numpy 交叉验证，六个 checkpoint 门禁）都是**逐 token 调 `Forward`**，并把每层的 residual 抓出来定位分叉点。它们是这个包唯一与外部实现对齐的证据。
- `TestBatchMatchesSequential`、`TestQwen2PrefillMatchesForward`、`TestQwen3PrefillMatchesPerTokenForward`、`TestDecodeStepMatchesForward`、`TestDecodeStepContinuesLikeForward`、`TestPagedPrefillMatchesContiguous`、`TestMoEDecodeMatchesPrefill` 全部以 `Forward` 为 `want`。`qwen3_test.go:144` 的注释把理由写得很清楚：numpy 参考只走单 token，所以它**看不见**批量路径漏掉一步；批量路径的正确性靠与 `Forward` 对齐来保证。
- 实测：35 处 `Forward(cache, …)` 调用点是这些断言的参考侧。

所以删掉它会同时抽掉三层地基——numpy 对齐、prefill 对齐、decode 对齐——换来 190 行。**这个交换不成立，建议作废。**

真正的问题因此换了个形状：三份实现里，**两份是被验的、一份是验的**。合理的做法不是删掉参考，而是把"三份"变成"一份可直接读的标量实现 + 一份批量实现"，并让批量实现与标量实现逐位对齐。那是一次真正的重构（要把 `batch` 与 `decodeBatch` 的 scratch 合并、把 `MatVec` 与 `MatMulTransposed` 的差别收进一个参数），不是删代码。是否值得做，取决于"三份"已经造成的缺陷率——按 `docs/models.md` 的记录是至少一次真实缺陷（Gemma 2 softcap），这个证据支持做，但不支持用"删掉一条腿"的方式做。

复算命令：

```bash
grep -n "^func (m \*Model) Forward\|^func (m \*Model) Prefill\|^func (m \*Model) DecodeStep" internal/llama/*.go
grep -rn "\.Forward(\|\.Generate(" --include=*.go internal cmd | grep -v _test
```

## 四、entry point 的重复：三次打开模型、五处手写 pump 循环

### 4.1 一次 `serve` 打开模型三次

```text
cmd/orbix/serve.go:241   gguf.Open          （--prefix-cache 校验 --block-size）
cmd/orbix/serve.go:256   gguf.Open          （tokenizer + chat 模板）
engine.New → llama.NewWithOptions → LoadWeights → quant.OpenLoader → gguf.Open + os.Open   （权重）
```

三次解析 GGUF 目录、两次真正的 `os.Open`。`gguf.Open` 在 `gguf.go:84` 读完目录就立刻 `Close`，所以重复打开不是资源泄漏，是**三次同样的解析工作**和**三处可能各自失败的入口**。`serve` 里那段为了提前校验 `--block-size` 而先开一次文件的注释自己写着"the model is opened first"，是把"参数校验要先于加载"的要求翻译成了"多开一次"。

### 4.2 每个命令各写一遍装配

`run`、`chat`、`bench`、`serve` 四处的装配序列几乎相同，但各自手写：`gguf.Open` → `tokenize.New` → `llama.LoadConfig`（其中三处）→ `engine.New` → `NewChatRenderer`（两处）。同一条规则——"`--block-size` 大于模型上下文时拒绝"——在 `serve.go:249`、`bench.go:136`、`chat.go:187` 写了三遍，连错误消息的措辞都几乎一样。`defaultBlockSize = 16` 在 `llama/paged.go:132` 有定义，`chat.go:162` 又硬编码了一次 `blockSize = 16`。

### 4.3 五个 pump 循环

`engine.Pump` 必须由单一 goroutine 驱动，于是每个调用方都自己写循环：

```text
internal/server/server.go:245       带 ctx/stop 与 1ms<<idle 退避
cmd/orbix/chat.go:548               drain-then-pump，带最后的 dec.Flush()
cmd/orbix/bench.go:240              ↑ 独立 goroutine drain + for eng.Pump() > 0
cmd/orbix/bench_suite.go:378/464/566   三处几乎相同
```

这不是"重复几行"的问题：`server` 那份处理了取消、退避和关闭；`bench` 那几份假定 `chans` 会在 Pump 返回零之前关闭；`chat` 那份多了一个 `Flush()`（丢弃跨 token 的半个字符会改变输出）。这些差异**没有任何测试交叉覆盖**——测 `bench` 的循环不会发现 `server` 的退避写错。一个 `engine.Drive(ctx, ...)` 之类的统一驱动函数可以同时收掉五个循环和它们的差异，但这是 API 变更，属于下一轮的决定，不在本轮。

复算命令：

```bash
grep -rn "eng.Pump()" --include=*.go internal cmd | grep -v _test
grep -rn "gguf.Open" cmd/orbix/*.go
```

## 五、`cmd/orbix` 是第四个"逻辑层"

CLI 层 3429 行里有大量不属于"解析参数并打印"的内容：

- **窗口与预算的算术**：`server.go:1402`、`chat.go:520`、`run.go:189` 各自实现"预算超过剩余窗口就截断"，`run` 还额外判断 `cache.Len() == cache.Cap()` 来决定是否打印提示。
- **错误消息的领域解释**：`chat.go:441` 的 `explainChatError` 把 `ErrPoolExhausted`、`errContextFull` 翻译成带建议的段落，并在同一会话里"解释一次，之后只打印数字"。
- **启动横幅**：`serve.go` 的 `poolCapacityWarning`、`poolReservationWarning`、`reservationWarning`、`contiguousMemoryNote` 四个函数共 120 行，把配置翻译成告诫。

这些是有价值的 UX，但它们的输入（`ContextLen`、`PoolBlocks`、`PoolBlockSize`、`errContextFull`）全部来自 engine/llama，所以**每加一个缓存布局或一条限制，就要回到三个命令里各改一次**。`engine.Stats` 已经把池字段聚合好了，而 `chat`/`run` 仍在通过 `eng.Model().Config()`、`pool.Blocks()`、`cache.Cap()` 自己拼装同一个判断。

## 六、并发模型：一处状态机，多处手写

- **engine 的核心是正确的**：`Submit` 把队列检查与插入放在同一个临界区（注释记录了过去 TOCTOU 的实测：MaxActive 1 + MaxQueued 2 时 12 个并发提交全部被接受）；`Stats()` 在一个锁内取快照；`Pump` 用 `pumping` 标志而不是锁来保证排他。
- **`server` 是并发原语最集中的文件**：`server.go` 里 9 处 `sync.Once`/`sync.WaitGroup`/`atomic.Uint64`，另有 3 个 `chan struct{}`，而整个 engine 只有 2 个 `sync.Mutex`（`engine.go:260` 的调度状态、`chat.go:83` 的渲染原因）。竞争点分散在"请求计数、拒绝计数、id 生成、模板告警一次性、pump 生命周期"五件事上，各用各的机制。engine 少而 server 多，方向是对的——调度器没有启动任何 goroutine（`grep "go func\|wg.Go" internal/engine` 为空），计数与生命周期归网络边界——但五种机制没有共同的抽象，新增一个计数器就要重新选一次工具。
- **每个 reader 一个 2 秒放弃窗口**（`engine.go:865` 的 `abandonGrace`）是全局常量，不可配置，也不在 `engine.Options` 里。文档里把它写成契约（"drain as you go"），但部署方无法按自己的慢客户端调它。

## 七、公共 API 面：宽、且有一部分只为测试存在

`internal/llama` 导出 20 个类型/函数，其中这些在生产代码里**零调用**：

| 符号                      | 生产调用点 | 说明                                                |
| ------------------------- | ---------- | --------------------------------------------------- |
| `PagedKVCache.BlocksHeld` | 0          | 只在测试里用                                        |
| `BlockPool.RefCount`      | 0          | 只在测试里用                                        |
| `BlockPool.Positions`     | 0          | 只在测试里用                                        |
| `Model.Logits`            | 0          | 文档注释说的"nil before the first pass"行为只为测试 |

这一节写下的当天就处理了第一个：`PrefixIndex.Key` 曾整个仓库零调用（包括测试），注释却说"so a caller can pass it as the next block's parent"，而实际实现是包内未导出的 `chainRoot`（`paged.go:439`）。它已删除，`Remember` 那段依赖它的注释同时改成描述真实契约——parent 是链上前缀的 key，由 `rememberFullBlock` 折叠得出。覆盖率里那条 `prefix.go:201 Key 0.0%` 随之消失，`internal/llama` 仍是 88.4%（这个数字里的分母小了，比例没动，说明删掉的正是从未执行的语句）。

`Model.Forward`/`Generate` 则相反：有调用方（`run`），但那个调用方让整条单 token 路径活着（见第三节）。删它需要先迁移 `run`。

覆盖率快照（`go test -cover ./...`）：engine 95.8%、kernel 99.0%、quant 98.0%、f16 98.1%、tokenize 96.7%、server 94.9%、gguf 94.7%、chat 93.0%、otel 91.5%、gguftest 88.6%、llama 88.4%、cmd/orbix 88.2%。`llama` 与 `cmd` 的低点集中在 `cmdServe`（50.4%，模型加载后的编排难以在测试里跑完整）；删掉 `Key` 之后 `llama` 已无 0% 的函数，最低是 `rememberPrefix`/`feedForwardBatch`/`feedForward`/`sameTokens` 的 66.7%。

## 八、门禁与验证现状

| 门禁                       | 本轮结果                                   |
| -------------------------- | ------------------------------------------ |
| `go vet ./...`             | 通过                                       |
| `go test -count=1 ./...`   | 通过，24 秒；engine 占 22.7 秒             |
| `just lint`                | 通过，0 issues（自动选中 v2.15.0）         |
| `just docs`（rumdl）       | 通过，10 个文件（含本文）                  |
| `just hooks`（prek）       | 通过                                       |
| `just cross`（5 目标）     | 通过                                       |
| `just check` 整体          | **通过**（本文写完后修好了下面两条）       |

初测时 `just check` 在这台干净的树上还是红的，两个原因都不是代码：

1. **`just lint` 只看了 `PATH` 上的第一个 `golangci-lint`**。那一个是 mise 装的 v1.64.8，而 `.golangci.yml` 是 v2 配置，于是门槛报告"工具缺失"——同时一个能用的 v2.15.0 就排在 `PATH` 第三位。这比不检查更糟：它把一个可用的环境判成不可用，而错误信息指向版本号，不指向"还有别的候选"。
   现在改成用 `type -ap` 枚举全部候选、逐个探测、取第一个 major 为 2 的；一个都没有时把每个候选的版本和路径都列出来。三条路径都实测过：正常（选中 v2）、只有 v1（列出 `1  /…/golangci-lint` 并退出 1）、完全没有（提示安装）。

   写这段踩了两个 bash 与 just 的交互坑，都记在 justfile 注释里：字符串里的字面换行会**结束 recipe**（just 按缩进读正文），而想用 `$(printf '%s\n' …)` 绕开时，命令替换会**吃掉末尾换行**，最后一行和下一句粘在一起。最终用 bash 数组。

2. **`scripts/jinja_diff.py` 的期望表过期**：`inline_if` 被列在"应当拒绝"里，而 `internal/chat` 在 `6da2e26` 之后已经支持内联条件（Gemma 3 模板需要它）。门槛因此长期出口 1，报的是一个**已经实现**的构造——一个不随功能更新的期望表，会把真实的回归藏在它后面。
   现在 `inline_if` 移入 SUBSET 并扩成五个用例（两个分支、字段条件、与比较/连接的优先级、嵌套），空出的"应拒绝"位置换成文档确实承诺拒绝的短式 `a if cond`。结果：**598 例，588 与 Jinja2 一致，10 例按设计拒绝，exit 0**。

覆盖率与 checkpoint 门禁：65 处 `t.Skip` 分布在 24 个测试文件里，由 11 个环境变量控制——`MODEL`、`QWEN2_MODEL`、`QWEN3_MODEL`、`GEMMA_MODEL`、`GEMMA2_MODEL`、`GEMMA3_MODEL`、`MISTRAL_MODEL`、`MIXTRAL_MODEL`、`MODEL_KQUANT`、`CODER_MODEL`、`ORBIX_REFERENCE_DIR`。本机没有任何 `.gguf`，所以这一轮**所有与真实权重相关的结论都没有被复测**；CI 的 skipped-tests 报告行存在的意义就是这个。

## 九、结论与建议的下一步

**本轮已做**（三项小时级的，全部有实测验证）：

1. ~~**删 `PrefixIndex.Key`**~~ — 已删，`Remember` 的注释改成描述真实契约，`internal/llama` 里再无 0% 的函数。
2. ~~**修 `jinja_diff.py` 的期望表**~~ — 已修，门槛从 exit 1 回到 exit 0（598 例）。顺带把内联条件补成五个用例，并明确写进注释：期望表不随功能更新，是让门槛长期报假回归的原因。
3. ~~**修 `just lint` 的候选解析**~~ — 已修，`just check` 在这台机器上首次全绿。

**其余建议项**（半天级，有测试面扩张）：

4. ~~**把 `run` 迁到 engine**~~ — **已作废**，理由见第三节的更正：单 token 路径是 numpy 参考与两条批量路径的对齐基准，删它等于抽掉三层验证。替代方向是"标量参考 + 批量实现"的合并，见下。
5. **合一个 `cmd/orbix` 的装配函数**（`openModel`/`newEngine`）：消掉三次打开与三份 `--block-size` 校验。
6. ~~**在 `engine` 边界上翻译 `ErrPoolExhausted`**~~ — 已做，见下。

**本轮做掉的第 6 项，连同它的代价一起记下来。**

`engine` 现在有一个 `ErrPoolExhausted` 哨兵，`poolAwareErr` 在 prefill 与 decode 两处把存储层的错误翻译成
`poolExhaustedError`：它 `Unwrap() []error` 成 `{engine 哨兵, 原始原因}`，所以 `errors.Is` 对两者都成立，
而 `Error()` 直接返回原始原因的文本——客户端看到的仍是 `llama: block pool exhausted (4 blocks of 4 positions)`，
块数与块大小都在。`internal/server` 的两处判断和 `cmd/orbix` 的两处判断都改到 engine 哨兵上。

两个必须说清的限制：

- **`server` 并没有因此不再导入 `llama`。** 它还用 `Greedy`、`Temperature`、`Biased`、`Sampler`、
  `TokenBias`、`TokenLogprobs`、`Alternative` 七个符号（采样器与 logprob 类型）。这次消除的是
  **"用存储层错误做调度判断"** 这一条，不是整个 import 边。上一轮把目标写成"让 server 不再导入 llama"
  是不准确的，实际能做到的是这条边不再承载一个跨层语义。

- **这处改动无法被测试区分，我没有假装它可以。** 注入验证的结果是：把 `chat.go` 的判断改回
  `llama.ErrPoolExhausted`，`TestExplainChatError` 仍然通过——因为在真实错误上两条链都成立，
  `errors.Is` 两种写法都是 true。这是**行为等价的接口收窄**，价值在于依赖方向而不是输出。
  新加的 `TestPoolExhaustionIsRecognisableWithoutTheCachePackage` 能区分三件事，都做过注入验证：

    - `Unwrap()` 只返回原因、不返回 engine 哨兵 → 红（"a caller would have to import
      internal/llama to recognise the cause"）；
    - 错误文本被改成 engine 哨兵的措辞（`"engine: block pool exhausted: llama: ..."`）→ 红。
      这一条最重要：它锁住的正是**我第一版写错的那个形态**——用 `fmt.Errorf("%w: %v", ...)` 包装，
      `errors.Is` 两条链都成立，但客户端看到的句子多了一层前缀。上一轮的断言只查 `errors.Is`，
      对文本完全无感；
    - 把判断改回 `llama.ErrPoolExhausted` → **不红**，因为真实错误上两条链都成立。这一条无从
      测试，也不该假装能测。

第 5 项（合并 CLI 装配）仍未做，与第 6 项互不依赖。

## 复算清单

```bash
go test -count=1 ./...                                    # 24s
go test -cover ./...
golangci-lint run                                         # 需 v2
just docs && just cross
uv run --offline --with jinja2 python3 scripts/jinja_diff.py   # exit 1（inline_if）
```

```bash
# 依赖边
go list -f '{{.ImportPath}}: {{join .Imports " "}}' ./internal/... ./cmd/...
# 三份前向
grep -n "^func (m \*Model) Forward\|^func (m \*Model) Prefill\|^func (m \*Model) DecodeStep\|^func (b \*decodeBatch)" internal/llama/*.go
# 三次打开
grep -rn "gguf.Open\|OpenLoader" --include=*.go internal cmd | grep -v _test
# 五个 pump
grep -rn "eng.Pump()" --include=*.go internal cmd | grep -v _test
# 零调用导出
grep -rn "\.Key(\|\.BlocksHeld(\|\.RefCount(\|\.Positions(\|\.Logits(" --include=*.go . | grep -v _test
```
