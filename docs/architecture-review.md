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

**重构后的同一张表**（三项重构提交之后实测，与本表上下对照可以看出改动的规模）：

| 包                | 代码行 | 函数（含方法） | 变化                                                                                         |
| ----------------- | ------ | -------------- | -------------------------------------------------------------------------------------------- |
| `internal/llama`  | 4666   | 123            | -20 行；函数 +2（新增 `embedRow`/`addBiasTo`/`outputWeight`/`capLogits`，删掉 6 个重复函数） |
| `cmd/orbix`       | 3457   | 74             | +28 行；`model_files.go` 是新增的装配层，`chat.go` 净减 80 行                                |
| `internal/engine` | 1789   | 40             | +96 行；`drive.go` 与它的测试                                                                |
| `internal/gguf`   | 1488   | 51             | +29 行；`OpenCount`                                                                          |
| `internal/server` | 4038   | 89             | +5 行注释，判断改到 engine 哨兵                                                              |

行数变化不大，因为这次重构换掉的不是代码量而是**同一件事写了几遍**：`cmd/orbix` 少了三处重复装配、`llama` 少了六份重复函数、`engine` 多了两个共享入口。判断收益要看重复度而不是行数。

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
- `addBias` 有三个版本、`embed` 有三个版本、`finish` 有三个版本，每个都要记得调 `kernel.Softcap`。

**这一节列出的重复已经处理掉了四分之三。** 三份 `embed`、三份 `addBias`、三份"选 tied 权重"和三份 softcap 合并成 `embedRow`、`addBiasTo`、`outputWeight`、`capLogits`，`internal/llama` 从 88.4% 覆盖率升到 88.7%（删掉的是重复语句）。

两处注入验证证明这个合并不是只在行数上有益：把 `capLogits` 从 prefill 路径删掉，`TestFinalSoftcapBoundsTheLogits` 立刻红——复现的正是 `docs/models.md` 记录的那个历史缺陷；把 `embedRow` 的边界从 `token < 0 || token >= VocabSize` 放松成只查上界，`TestEmbedRejectsNegativeToken` 以 `slice bounds out of range [-32:]` panic——正是那条注释解释了三遍的失败。

`feedForward` 的三份里，**两个批量版已经合并**（`6d554da`）。它们原本是同一段 25 行写了两遍：`batch` 与 `decodeBatch` 的字段名和布局完全一致，差别只是哪个结构体装着切片。`ffnRows`（原本叫 `moeRows`，MoE 路径已经在用它做同一件事）现在是两者共同的参数，`feedForwardRows` 是唯一实现，两个调用点各变成三行。

这次合并没有牺牲对齐保证：`TestDecodeStepMatchesForward` 与 `TestBatchMatchesSequential` 都用 `!=` 直接比对逐 token 参考，不是容差。把 gate 激活从共享实现里删掉，**15 个测试同时变红并横跨两条路径**——合并前，prefill 那份里的同一个缺陷会让所有 decode 测试保持绿色，那正是 `docs/models.md` 记录的 Gemma 2 softcap 缺陷的形状。

`attention` 的三份仍然**没有合并，这是判断而不是遗漏**：

- `attentionBatch` 写一个 cache、按位置顺序、**并行**处理位置；
- `decodeBatch.attention` 读每个序列自己的 cache、在自己的末尾、因每序列只有一个位置而**串行**；
- `attentionCached` 是第三种形状——它只为 cache 已有内容的提示词重算 query，连 key 和 value 都不投影。

把三种压进一个函数需要把 mask、cache 归属和并行性都变成参数，而结果会比现在任何一份都更难与参考核对。单 token 的 `feedForward` 同理保持独立：它是另外两条路径的参考。

顺带合并的一处：attention 与 feed-forward 的子层归一化本来是四份（prefill 一份、decode 两份），现在统一走 `normRowWise`（`bceb10c`）。注入错误的 epsilon 会让 11 个测试变红。

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

## 四、entry point 的重复（已处理，见本节末尾的结果）

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

这不是"重复几行"的问题：`server` 那份处理了取消、退避和关闭；`bench` 那几份假定 `chans` 会在 Pump 返回零之前关闭；`chat` 那份多了一个 `Flush()`（丢弃跨 token 的半个字符会改变输出）。这些差异**没有任何测试交叉覆盖**——测 `bench` 的循环不会发现 `server` 的退避写错。

### 4.4 处理结果

三项都做了，改动分别是 `9831e01`、`d67b949`、`da9ec9d`。结论与原先的预测有出入，按实际测量写：

- **`serve` 是 3 → 2，不是 3 → 1。** `openModel` 把"校验用的那次"和"tokenizer 那次"合成一次，于是元数据只解析一次；但 `engine.New → llama.LoadWeights → quant.OpenLoader` 还要再解析一次并另开一个 `os.Open`，因为它读张量字节需要一个定位在数据段的 reader，而元数据要同时可用。这是结构性的，不是遗漏。`bench` 与 `chat` 本来就在 2，它们的变化是"检查搬进 `planCache`、硬编码的 16 换掉"，不是"少解析一次"。
- **`chat` 的 `cfgForBlocks` 是死路径。** 它在文件没写 `context_length` 时回退 2048，而 `openModel` 先调 `llama.LoadConfig`，后者把该键当必需——所以在当前结构下那个分支不可达。删掉，它的测试改成断言用户真能看到的拒绝。
- **`--block-size` 的三处检查条件确实不一致**：`serve` 要求 `blocks>0` 且 `block-size>0`，`bench` 只要求 `block-size>0`，`chat` 无条件（因为它默认开复用）。现在统一在 `planCache`，按"是否真的会有池 + 是否真的会复用"判断——这才是这个检查真正的主题。
- **`engine.Drive` 收掉了四处，不是五处。** 命令行侧四个循环共用同一个骨架（每序列一个 drainer → pump 到零 → join），差异只在 drainer 里的计数。`internal/server` 那份**保留**：它要轮询、空闲退避、响应 ctx 与关闭信号，是另一种循环，硬套 `Drive` 会把它需要的三件事塞进参数。`chat` 那份是**同步** drain（不能阻塞，否则与 pump 互锁），也不是同一形状。

回归被测试钉住而不是写在注释里。`TestDriveDrainsBeyondTheEventBuffer` 要 200 token 对着 64 事件的缓冲；把 `Drive` 换回"先 pump 后读"，它恰好收到 64 并报出缺的 136——正是 `bench` 修复前产生的数字。

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

| 符号                      | 生产调用点 | 测试调用点 | 真实性质          |
| ------------------------- | ---------- | ---------- | ----------------- |
| `PagedKVCache.BlocksHeld` | 0          | 4          | 块核算的观察口    |
| `BlockPool.RefCount`      | 0          | 15         | 引用计数的观察口  |
| `BlockPool.Positions`     | 0          | 1          | 池容量的观察口    |
| `Model.Logits`            | 0          | 2          | 前向结果的读取口  |

**这一节最初把"零生产调用"当成了缺陷，这是错的，复查后保留全部四个。** 判断要分两步：
`PrefixIndex.Key` 是**整个仓库零调用，连测试都没有**，注释还描述了一个不存在的行为，那是死代码；
上面这四个**有测试在依赖**，删掉它们等于删掉对应的测试能力——`RefCount` 撑起 prefix eviction 的
引用计数断言，`BlocksHeld` 撑起"释放后块全部归还"和"驱逐前缀后尾部仍被持有"这两条，而那正是
块泄漏仅有的两个发生点，泄漏在生成结果里不可见，要等池在几十个请求之后枯竭。

`Model.Logits` 是唯一有歧义的：它本身是前向输出的唯一读取口，但注释花了四行讲"第一次前向之前是
VocabSize 个零而不是 nil"——那个陷阱只对测试有意义。注释留着是对的，它是给测试读者的警告。

所以处理方式是**把观察口的性质写在代码上**，而不是删掉它们：`Positions` 与 `BlocksHeld` 的注释现在
各自写明了为什么没有生产调用方、以及它们守住的是哪个缺陷。`RefCount` 原本就有"for tests and
diagnostics"。

真正被处理掉的是 `PrefixIndex.Key`：曾整个仓库零调用（包括测试），注释却说"so a caller can pass it
as the next block's parent"，而实际实现是包内未导出的 `chainRoot`（`paged.go:439`）。它已删除，
`Remember` 那段依赖它的注释同时改成描述真实契约——parent 是链上前缀的 key，由 `rememberFullBlock`
折叠得出。覆盖率里那条 `prefix.go:201 Key 0.0%` 随之消失，`internal/llama` 仍是 88.4%（分母小了，
比例没动，说明删掉的正是从未执行的语句）。

`Model.Forward`/`Generate` 则相反：有调用方（`run`），但那个调用方让整条单 token 路径活着（见第三节）。删它需要先迁移 `run`。

### 用扫描器找死导出符号，结论是零个

试过一次自动化：遍历所有 `.go`，对每个导出的函数/方法统计它在**非测试**文件里被 `SelectorExpr`
引用的次数，为 0 就报出来。25 个结果，逐个查下来**没有一个是能删的**：

| 误报来源     | 例子                                               | 真实情况                                                                                         |
| ------------ | -------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| 同包内调用   | `kernel.MatMul`、`kernel.Dot`、`llama.LoadWeights` | 脚本只数跨包的 `x.Name`，漏掉包内直接调用                                                        |
| 接口实现     | `MarshalJSON`、`UnmarshalJSON`、`Unwrap`           | 通过接口调用，永远不会出现字面的 `SelectorExpr`                                                  |
| 测试对拍基准 | `MatMul`/`MatVec`/`Dot`                            | `ops.go:106` 原话是 "This is the serial reference"；生产走 `Runner` 的并行版，串行版就是拿来比的 |
| 测试观察口   | `RefCount`、`BlocksHeld`、`Positions`              | 见上表                                                                                           |

`MatMul` 的例子最典型：整个仓库只有它的定义、`checkMatMul`（校验形状）和 `ops_test.go` 三处提到它，
一个纯粹的"零调用"信号。而它的文档写着"Tests compare against this one with a tolerance"，删掉它
等于删掉并行实现的正确性基准。

**结论：这类扫描能列出待查清单，不能给出结论。** 判断一个导出符号该不该删，要问的是"删掉之后
谁的断言失去依据"，而不是"有多少个 `x.Name`"。这条和 AGENTS.md 里"判定不可达之前先看函数的参数"
是同一个道理：静态计数看到的是引用图，不是用途。

覆盖率快照（`go test -cover ./...`，2026-11 末复测）：kernel 99.0%、quant 98.0%、f16 98.1%、tokenize 96.7%、server 94.9%、gguf 94.8%、engine 94.7%、chat 93.2%、otel 91.5%、llama 89.2%、gguftest 88.6%、cmd/orbix 87.8%。`internal/arch` 没有可测代码——它只有测试。`llama` 与 `cmd` 的低点集中在 `cmdServe`（50.4%，模型加载后的编排难以在测试里跑完整）；删掉 `Key` 之后 `llama` 已无 0% 的函数，最低是 `rememberPrefix`/`feedForwardBatch`/`feedForward`/`sameTokens` 的 66.7%。

## 八、门禁与验证现状

| 门禁                                     | 本轮结果                                        |
| ---------------------------------------- | ----------------------------------------------- |
| `go vet ./...`                           | 通过                                            |
| `go test -count=1 ./...`                 | 通过，24 秒；engine 占 22.7 秒                  |
| `just lint`                              | 通过，0 issues（自动选中 v2.15.0）              |
| `just docs`（rumdl）                     | 通过，10 个文件（含本文）                       |
| `just hooks`（prek）                     | 通过                                            |
| `just cross`（5 目标）                   | 通过                                            |
| `just check` 整体                        | **通过**（本文写完后修好了下面两条）            |
| `just test-race`                         | 通过，无 `DATA RACE`；server 47.1 s，f16 54.6 s |
| `just isolate`（`-count=2 -shuffle=on`） | 通过，13 个包全绿，无顺序依赖、无状态泄漏       |
| `just FUZZTIME=10s fuzz`                 | 通过，6 个 target 无崩溃                        |

后三条是 CI 的独立 job 或辅助配方，`just check` 不跑它们，所以容易被忘。**2026-11 末一次完整跑**的结果记在这里，因为此前没有任何地方写过它们跑过——一个"CI 会跑"的门禁和一个"实际跑过"的门禁不是一回事。

fuzz 的 seed 语料只有两个文件，都在 `internal/gguf/testdata/fuzz/FuzzRead/`，其余五个 target 一个都没有。这不是缺口：`go test` 会重放已有的 seed（实测 `FuzzRead/seed#0` 到 `seed#4` 都执行），而当前 fuzz 没发现崩溃，所以没有新语料落盘。语料薄说明还没有崩溃被查出来过，不说明检查没跑。

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
5. ~~**合一个 `cmd/orbix` 的装配函数**（`openModel`/`planCache`）~~ — 已做，见 4.4。
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

第 5 项也已做完。三项重构（`9831e01`、`d67b949`、`da9ec9d`）覆盖了本节原本列的 5 与 6，以及第三节"三份实现"里逐字重复的四分之三。

**没做的，以及为什么。**

- `attention` 的三份没有合并，`feedForward` 的单 token 版也没有并入批量实现。前者的差别是驱动方式（写一个 cache 并行 / 读各自 cache 串行 / 只重算 query），后者是参考实现必须保持独立。`feedForward` 的两个批量版**已合并**，见第三节。
- `run` 仍走 `llama.Model` + contiguous cache，没有迁到 engine。这不是被排除，而是**待定**：迁过去会改变它测的东西（它现在直接测 `llama` 的生成路径），也可能改变 `--stats` 里那几个数字的含义。要做的话应当先明确它到底该测什么。
- ~~`internal/server` 仍导入 `internal/llama`~~ — **已切掉**（`5619da2`）。`Sampler`、`Alternative`、`TokenBias` 现在声明在 `internal/engine`；`llama` 的实现结构化地满足 `Sampler`，所以没有牺牲能力。三个构造函数（`NewBiased`/`Greedy`/`NewTemperature`/`TokenLogprob`）转调 `llama`，没有重新实现——第一版我给 `Biased` 写了个拷贝 logits 的版本，那会改变一个原实现用整段注释解释过的每 token 成本。
  这条边现在由 `internal/server/imports_test.go` 强制：它解析包内每个 `.go` 文件（含测试），命中 `internal/llama` 就失败并给出原因。用 grep 不行——那个包里有好几处注释引用 `llama.cpp` 作为 FIM 与模板分词的对拍参考，那些应当留下。

## 十、函数级失控：本轮按当下代码做的分解

第九节的三项是小时的清理，第三节到第七节是结构问题。这件事两者都不是，却是最影响阅读的：
**18 个函数在 100 行以上**（两轮合计从 21 降到 18，且降幅远小于被拆开的函数数——见本节末尾；
用 AST 按 `FuncDecl` 的起止行数，不用括号计数——注释里的花括号会把
统计骗成几千行；722 个函数，59 个在 60 行以上）。本轮按当下代码逐个处理，每改一个都核对错误字符串
与字段赋值是否逐字保留。

| 函数                      | 之前 | 之后 | 拿出来的部分                                                |
| ------------------------- | ---- | ---- | ----------------------------------------------------------- |
| `llama.LoadConfig`        | 315  | 45   | 六个阶段方法 + `configParser`（`get` 一个必填正整数读取器） |
| `server.generate`         | 282  | 189  | `resolvePrompt`（预算校验、按端点编码、窗口检查）           |
| `cli.cmdBench`            | 279  | 234  | `validateBenchFlags`（59 行，9 条错误串逐字一致）           |
| `cli.cmdServe`            | 311  | 238  | `validateServeFlags`（94 行，9 条错误串逐字一致）           |
| `server.blockingResponse` | 201  | 131  | `choiceObject`（68 行，两种 schema / echo / stop 截断）     |
| `server.streamResponse`   | 162  | 90   | `eventWriter`（帧头、对象名、usage 标记、编码、flush）      |

`cmdServe` 与 `cmdBench` 仍在 200 行以上：剩下的部分是"装配 + 运行"，是一条连续流程，再切只会
把一步拆成两半而不减少读者要在脑中持有的东西。

**仍未处理、>=100 行的十二个**（2026-11 末 AST 实测，731 个函数）：`tokenize.preFamilyOf` 271、
`chat.evalExpr` 242、`cli.cmdServe` 238、`cli.cmdBench` 234、`server.generate` 189、
`server.streamOneChoice` 157、`tokenize.loadPreTokenization` 157、`cli.cmdRun` 147、
`cli.chatSession` 135、`llama.attentionBatch` 133、`server.blockingResponse` 131、
`tokenize.splitKimiK2` 123。

逐个判断如下，而不是笼统地"改动面大"：

- **`preFamilyOf`（271）是数据不是逻辑。** 34 个 `case`，每个返回一个 `preRule` 字面量，注释是那个
  家族的预分词正则原文。改成 `map[preType]preRule` 同样 271 行，注释还得逐条贴，而且 `(preRule, bool)`
  的 `bool` 正是"这个家族是否已知"的判定，表驱动能保留它——所以改与不改都不改变读者要读的量。
  **行数是数据密度，不是复杂度。**
- **`evalExpr`（242）的骨架是分派。** 20 个 case 里 15 个只有 2–20 行"求子节点再合并"，拆成 20 个
  方法只增加间接层。有独立规则的那一个已经拆出去了（`evalArithmetic`，见下节）。
- **`cmdServe`/`cmdBench`/`cmdRun`/`chatSession`** 剩下的都是连续装配流程，只跑一次、只失败一次。
- **`LoadWeights`（134）与 `attentionBatch`（133）** 我在第三节判断过不该合。
- **`generate`/`streamOneChoice`/`blockingResponse`/`loadPreTokenization`** 是前几轮拆到一半的产物，
  再拆一层会把一步分成两半而不减少要同时握住的东西。

这一轮只处理了"拆开后能给出一个可命名的单元"的那种。

`LoadConfig` 的六段是 `dimensions`、`headGeometry`、`rotation`、`experts`、`window`（含
`localLayers`）、`vocabulary`。`dimensions` 顺手改成了表驱动；把 `block_count` 映射到错误的字段
会让 117 个测试变红，这是那张表能改的前提。

### 这一轮最重要的两个发现

**一、有些改动在测试层面不可观测，写测试会得到一条通过但无判别力的用例。**

`generate` 里把预算截到窗口的那一行（`ContextLen - len(promptTokens)`）就是。把它改成
`ContextLen`，响应完全不变——实测两种实现都返回 `completion=253`、`finish_reason="length"`，
因为 `engine.decode` 自己也会在窗口处停。这行代码改变的是 `usage` 报告的数字，不是模型生成的东西。

我先写了断言 `completion_tokens == 253` 的测试，注入之后它**照样通过**，于是把它删了，改成在代码
注释里写清这行为什么存在、以及为什么没有测试。留一条为错的理由通过的测试，比没有测试更糟。

**二、注入验证必须先确认构建成功。**

两次给 `validateBenchFlags` 注入 `if false` 都显示"测试没抓住"，实际上那次编辑让 `ignored`
成了未使用变量，**包根本没编译过**，`go test` 跑的是上一次的测试二进制。改成能编译的
`if !usedBySuite[f.Name] && false` 之后，测试立刻红。

此后每次注入都先 `go build`。这条和 AGENTS.md 里"注意同一段文本可能出现在多处"是同一类错误：
验证手段本身失效时报出的"通过"没有意义。

### 第二轮：五个函数，以及"数量没降"这个结果

| 函数                     | 之前 | 之后 | 拿出来的部分                                          |
| ------------------------ | ---- | ---- | ----------------------------------------------------- |
| `server.streamOneChoice` | 161  | 157  | `stopGate`（与上游对齐的文本保持状态机）              |
| `cli.chatSession`        | 178  | 135  | `openChatSession`（51 行，开模型、定容量、打 banner） |
| `cli.cmdRun`             | 175  | 147  | `validateRunFlags`（41 行，12 条错误串逐字一致）      |

**`streamOneChoice` 只少了 4 行，而且 >=100 行的函数总数仍是 18。** 这两个数字都对，原因是新增的
小函数（`stopGate` 的 `push`/`flush`、`openChatSession`、`validateRunFlags`）本身不短，而 `chatSession`
与 `cmdRun` 仍在一百行以上。把一行数当成进度指标是错的：这一轮的收益是**`stopGate` 第一次让流式路径里
的一段逻辑可以脱离引擎测试**，不是行数。

`stopGate` 五个测试覆盖：部分匹配被扣留、普通文本不延迟、无匹配时尾部最终释放、匹配后丢弃停止串及其后、
空停止列表立即释放。注入验证：去掉扣留逻辑红三个，去掉 `sent` 推进红一个。

第三个注入找出了一处死代码：`flush` 里 `if g.sent >= len(g.seen) { return "" }` 拿掉后行为不变，
因为真正保证幂等的是 `g.sent = len(g.seen)`。guard 已删，注释写明哪一行才是性质所在。

### 第三轮：`tokenize.New` 与"总数仍是 18"

| 函数           | 之前 | 之后 | 拿出来的部分                                        |
| -------------- | ---- | ---- | --------------------------------------------------- |
| `tokenize.New` | 274  | 31   | `loadTokenMetadata`(98)、`loadPreTokenization`(157) |

切分点是文件本身换命名空间的地方：前半段读 `tokenizer.ggml.*` 里关于单个 token 的事实（字符串、分数、
类型、特殊 id、字节级探测），后半段读文本怎么切块（pre-tokenizer 家族、merges、前缀空格）。

预分词器那个 `switch` 没有拆：它是查表，注释记录了每个名字属于 llama.cpp 的哪个分组、以及某个别名
为什么是精确的，拆开会让这些理由与它们解释的分支分离。

**这一轮又把函数总数从 18 保持在 18。** 三拆分后 `New` 少了 243 行，但两个新函数一个 98、一个 157，
前者没超线、后者超了。三轮下来的净结果是 21 → 18。这个指标对"拆成两个中等函数"这类改动不敏感，
它只对"拆成一个短函数加几个短函数"敏感。

### 一次真实的破坏，以及为什么测试没发现

`tokens` → `t.tokens` 的正则替换**也打到了两条错误消息**：

```text
tokenize: %d scores for %d t.tokens
tokenize: %d token types for %d t.tokens
```

`just check` 全绿，`internal/tokenize` 全部测试通过。原因是那两条的测试只断言了数字：

```go
for _, want := range []string{"1", "3"} {
    if !strings.Contains(err.Error(), want) { ... }
}
```

数字对，句子错，断言成立。这正是 AGENTS.md 那条"断言消息里的数据，不要断言模板文字"的镜像：
断言了数据，模板文字就可以被自由改坏。

处理方式是把两条消息整体比较（`err.Error() != "tokenize: 1 scores for 3 tokens"`），然后把同样的
损坏注入回去确认会红——它红了，报的正是 `error = "tokenize: 1 scores for 3 t.tokens"`。

这个破坏不来自重构的逻辑，来自重构的**手法**。批量文本替换在移动代码时会连字符串一起改，而字符串
里恰好有同名标识符。

### 第四轮：`evalExpr` 里唯一有独立规则的那个 case

`evalExpr` 的 20 个 case 里，19 个是"求子节点再合并"，唯一的例外是 `arithExpr`：43 行，包含减法的
溢出检查、模零拒绝、负数模拒绝，每条都带实测注释。它现在是一个独立函数 `evalArithmetic(op, left,
right)`，`evalExpr` 从 275 行降到 242。

判断依据不是行数而是**有没有自己的规则**。`indexExpr` 的"括号按操作数分两种含义"也有规则，但它与
两次递归求值耦合在一起，提取后剩 10 行；`sliceExpr` 的越界钳制只有 8 行且已有测试。这两个没动，
`arithExpr` 动了，因为它的规则有 43 行且几乎没测过——整个包里只有一条 `{{ 1 % 0 }}`。

补的六个测试覆盖了注释记录但无测试的两个真实缺陷：

- `MinInt64 - 1` 曾渲染成 `9223372036854775807`，`MaxInt64 - (-1)` 反向回绕。两个方向都断言，
  外加边界本身（`MaxInt64 - 0` 必须仍然通过），因为一端写了守卫通常漏另一端。
- 取模的负操作数被拒绝而不是计算，因为 Go 的截断除法与 Jinja 的地板除法在这些操作数上结论不同。
  三种符号组合都覆盖。

注入验证：关掉溢出比较红两个测试，关掉负数检查按名字红掉 divisor 那条。

### 第五轮：`LoadWeights` 终于降到线下，以及一个测不到的注释

`LoadWeights` 134 行里塞着三件互不相关的事：读 rope 频率张量、推导张量名清单、逐层装配。名清单先拆
（`tensorNames`，33 行），rope 张量后拆（`loadRopeFreqs`，23 行），**134 → 78**，四轮以来第一次让
"≥100 行"的总数从 18 掉到 17。

规模不是这轮的重点。重点是 `tensorNames` 此前**一个测试都没有**：所有现有测试都通过 `NewWithOptions`
加载真实文件，而名字清单是 `LoadConfig` 与文件之间的契约——它决定了"缺张量"是被命名报错还是在第一次
前向时空指针崩溃。现在五个测试直接覆盖它：四个可选组各自增减对应的张量、专家分支是**替换**稠密 FFN
而不是追加、没有重名、数量随层数线性增长、每个层内张量都是 `blk.<n>.<name>` 形状。

其中"没有重名"那条是永远不会自己报错的缺陷：重复的名字会让加载器把同一个张量读两遍、第二次覆盖第一次，
唯一的症状是内存占用看起来不对。

顺带补上 `rope_freqs` 的**零值**检查。它有注释和实测记录（`freqs {1, 0}` 时 `cos[pos=1][1]` 是 NaN），
兄弟分支（读取、长度、NaN）各有子测试，唯独这条没有。它是 `checkFinite` 覆盖不到的那类：零是有限值。
关掉检查后测试红，报的正是注释警告的那句话。

四个注入全部先 build 再跑，全部命中。

### 第六轮：把"零引用"当作待查清单，而不是结论

前一节记录过用扫描器找死导出符号，25 个结果零个可删。这一轮用同样的方法查 `preRule` 的 40 个规则
字段：统计每个字段在 `pre_test.go` 里的字面引用，**12 个为 0**。

逐个注入验证（把 `field: true` 改成 `field: false`，先 build 再跑）：

| 字段                                                   | 设置点 | 变红测试数 |
| ------------------------------------------------------ | ------ | ---------- |
| `newlineChunk`                                         | 19     | 14         |
| `contractionSuffix`                                    | 4      | 5          |
| `deepseekLLM`                                          | 1      | 4          |
| `digitsFromRight`                                      | 3      | 4          |
| `cjk`                                                  | 3      | 1          |
| `kimiK2`、`asian`、`leadingNewlinePass`、`newlineRuns` | 各 1   | 各 1       |

**12 个里 11 个被测试守着**——静态扫描是误报，因为字段通过 family 测试间接覆盖，测试里不会出现
字段名本身。这条和上一节的结论一致：扫描器给待查清单，不给结论。

剩下两个 `digitsRightBoundary` / `digitsRightDecimal` 是唯一真的例外，而原因在 history 里：
`git log -S` 显示它们为 tiny_aya 引入，又被 `85e9e22` 删掉——那次提交把这个家族拆成它两个表达式
本来的两趟，字段却留了下来。**从此没有任何 family 设过它们，也没有任何测试构造过。**

所以两个调用点的实参恒为 `(false, false)`。现在字面上就是这样：`splitDigitsFromRight(text, false,
false)`。分支保留（注释里有实测："123a" 在尾部 `\b` 下是一个 chunk），但文档不再假装它们活着。

覆盖率确认了"死"而不是"没测"：`splitDigitsFromRight` 83.7%，未覆盖的正是那三个实参门后的三行。

### 顺带补的一个有注释无测试的函数

查"哪些 family 测试最薄"时发现 `preWhitespace`：34 个家族里，16 个有从 llama.cpp 表达式生成的
参考用例，其余有手写表，唯独它以 `preRule{whitespaceOnly: true}` 一行交给 `splitWhitespaceRuns`，
而那个函数的注释记着一个真实缺陷（丢掉分隔符把 `"a  b"` 变成 `["a","b"]`），**零测试**。

补两条，各查一件事：分块形状（连续空格是一个 chunk、tab 与空格同属一类），以及把所有 chunk 拼回去
必须等于输入。第二个不是冗余——注入"丢弃分隔符"让第一条红三个用例而第二条全绿（拼起来仍然对，只是
切在错的位置）；注入"丢弃尾部"让两条都红。

### 第七轮：`attentionBatch` 的核心正确性已经锁住了

上一轮说要去查 `attentionBatch`（133 行）——那是我列出的最后一个"有大段逻辑但没系统验证"的函数。
查完的结论是：**它的两个核心保证已经被 10 和 23 个测试锁住，不需要额外补测。**

**每个位置用自己的 rope 表项。** 把 `pos := startPos + i` 改成 `pos := startPos`（每行都用第一个
位置），**10 个测试变红**，首个是 `TestBatchMatchesSequentialTokens`——那个逐 token 对拍的测试。原因
是参考实现（单 token 前向）也是按位置调 `rope.apply`，批量版用错位置，两边就不等了。

**因果 mask。** mask 的强制点在 `m2 := pos + 1`：每行只向 cache 请求到自己位置为止的 span，所以一行
看不到后面的位置。把它改成 `m2 := pos + 1 + n`，**23 个测试变红**。

这两个数字说明一件事：`TestBatchMatchesSequential` 这类对拍测试**不是只查数值近似**，它锁住的是整条
路径的结构性质。这也是为什么第三节那个"不要删单 token 参考实现"的判断是对的——它是对拍的另一端。

真正缺的只有并发缓冲的算术，本轮补上了：

| 提取           | 行数     | 新测试                | 注入命中                                                          |
| -------------- | -------- | --------------------- | ----------------------------------------------------------------- |
| `scoreBuffers` | 分配逻辑 | 4 个（含 6 个子用例） | 去掉 `min` → 3 个用例红；固定宽度 → 报出宽度；共用切片 → 报出别名 |

`scoreBuffers` 的注释记着真实缺陷：`Parallel` 把 n 个位置分给 `min(workers, n)` 个 goroutine，
所以高于这个数的缓冲没有任何人索引——"a deliberately large --workers on a long-context model
reserved gigabytes of score buffers per layer and made a single request tens of times slower"。
请求仍然正确，只是慢，所以它在输出里看不出来。

其中"切片互不别名"那条是计数型测试会漏的：每个条目都指向第一个，长度对、并发错，race detector
会报而错误结果可能不报。

### 第八轮：文档里有一条错的，而且它说了谎的方式很具体

上一轮把依赖方向变成了门禁。这一轮先用同一个方法查另一条只写在散文里的承诺——"每个包都有
`doc.go`，讲清这一层负责什么"。

十二个包全都有 `doc.go`。但顺着读下来发现 **`docs/architecture.md` 的阅读顺序是错的**：

```text
文档: gguf(1) f16/quant(2) kernel(3) tokenize(4) llama(5) engine(6) chat(7) server(8)
实际: engine → chat
```

文档把 `engine` 排在第 6、`chat` 排在第 7，而 `internal/engine` 导入 `internal/chat`。它自己是
这么解释这个顺序的："the concepts arrive in the order a request travels through them"——那是一个
依赖顺序，所以按它读，读者会在遇到渲染器之前先遇到调用它的调度器。

改成 `chat(5) llama(6) engine(7)`，并加了一条守卫：
`TestArchitectureDocAgreesWithTheLayerTable` 读那张表，对每一对有序的包断言两件事——层级表同意这个
先后，且**前面的包不导入后面的包**。把旧顺序改回去，它同时报出两条：

```text
the layer table has them the other way round
internal/engine imports internal/chat -- a reader meets the caller first
```

两个细节让这条守卫值得写：只比相对顺序而不比编号（文档把 f16 和 quant 放在同一行，因为读者是一起
遇到的）；同名的包出现在两行时取最早的（重复链接不该看起来像矛盾）。

顺带补了 `internal/engine/doc.go`：它描述了调度轮次、两种缓存布局和 pump 规则，却**没提**那三个
为了让 server 不导入 llama 而存在的类型（`Sampler`、`Alternative`、`TokenBias`），也没提
`ErrPoolExhausted` 的边界和 `Drive`。`Drive` 那段带的数字是查过的：读端先于 pump 会在 64 个事件处
卡住（缓冲大小），实测 `--max-tokens` 65/100/200 每次都只生成 64 个 token 且不打印 finish 行。

还查了其余文档的失效引用：所有 `doc.go` 里反引号引用的符号都能在对应包里找到（0 个失效），
`docs/cli.md` 覆盖了全部实际命令。

### 第九轮：flag 全表也没有门禁

上一轮查了 `doc.go` 与阅读顺序。这一轮查另一条"文档承诺"：`docs/configuration.md` 开头写着
"Every setting is a flag today"，然后列了 44 个 flag。

**核对结果是完全一致**——两个方向、44 个 flag、零差异。但那次核对是我手工做的，没有任何测试会
因为漂移而失败。

现在有了：

| 检查                                     | 抓什么                        | 注入验证                        |
| ---------------------------------------- | ----------------------------- | ------------------------------- |
| `TestConfigurationDocListsEveryFlag`     | 命令接受但文档没列的 flag     | 把 `--shutdown-grace` 改名 → 红 |
| `TestConfigurationDocNamesOnlyRealFlags` | 文档列了但没有命令接受的 flag | 编造 `--drain-timeout` → 红     |

第二个方向**故意比能做的更弱**。按命令归因是显然的下一步，但它不成立：文档的表按**功能**分节，
"Paged cache, on serve, chat and bench" 是一张覆盖三个命令的表。我按 `##`/`###` 分节 + "以冒号
结尾且提到命令名的句子"做归因，**报了 11 个 flag 归错命令，全是误报**。于是退到并集检查：文档
提到的每个 flag 必须被某个命令接受。它抓不住"这个 flag 属于别的命令"，但抓得住文档侧改名和代码侧
删 flag——这两个才是"表承诺了没人能传的东西"的实际形态。

实现上跑 `go run ./cmd/orbix <cmd> -h` 而不是把 flag 注册抽成函数。我试过抽成
`defineServeFlags` 并改好了 `serve.go`（44 行改动），然后**撤回**了：抽出来后没有消费者，四个命令
都要为同一个测试改一遍。实测四次调用共 1.2 秒，够便宜。

守卫里 `if len(union) < 25` 的 25 是照实测的 26 留的余量，不是猜的。

### 第十轮：端到端跑了一遍，以及一次"重复"的复核

前九轮改的多是结构，`just check` 只证明单元与集成测试通过。这一轮用合成 fixture 起真服务跑了一遍：

```bash
# 合成一个 185 KiB 的完整模型，起服务
./bin/orbix serve --model /tmp/e2e/tiny.gguf --addr 127.0.0.1:19871 --blocks 64 --block-size 16
```

| 路径                                   | 结果                                                           |
| -------------------------------------- | -------------------------------------------------------------- |
| `/healthz`                             | `{"model":"tiny","status":"ok"}`                               |
| `/v1/completions` 非流式               | `finish_reason: length`，`completion_tokens` 与请求一致        |
| `/v1/chat/completions`                 | 正常，`prompt_tokens` 随模板变化                               |
| SSE 流式 + `[DONE]`                    | 逐 chunk 到达                                                  |
| `stream_options.include_usage`         | usage 块携带完整计数                                           |
| `chat_template_kwargs`（无模板的模型） | 400，`param: chat_template_kwargs`，消息说明模型没有可评估模板 |
| `max_tokens: -5`                       | 400，`param: max_tokens`，`code: invalid_value`                |
| `/metrics`                             | `orbix_tokens_generated_total 14`、`orbix_kv_pool_free 64`     |
| SIGTERM                                | 日志 `shutting down`，端口释放                                 |

顺带一个自己的错误：`nohup ... &` 记下的 `$!` 是**子 shell** 的 pid，kill 它不会停服务——第一次测关闭时得到
`STILL UP`，看起来像缺陷。按端口取 pid 后正常。这正是 AGENTS.md 里那条不要按名字找进程的同一类。

**重复代码复核**：用一个按函数切分的 8 行窗口散列扫全仓，跨函数重复 5 处，逐个查过**无一该合并**：

| 位置                                  | 为什么不是重复                                                                           |
| ------------------------------------- | ---------------------------------------------------------------------------------------- |
| `gguf.intAs` / `uintAs`               | 有符号与无符号的定义域不同，注释写明为什么分开；合并会让 uint64 超过 MaxInt64 的值被误拒 |
| `llama.attention` / `forward`         | 前几轮已判定：差别在驱动方式而非算术                                                     |
| `llama.newBatch` / `newDecodeBatch`   | 字段相似，但 scores 的类型不同（`[]float32` vs `[][]float32`）                           |
| `bench.runRequests` / `runSequential` | 套件要对比的就是并发与串行两种测量                                                       |
| `cmdBench` / `cmdServe`               | 滑窗假阳性，实际内容不同                                                                 |

API 面也复核了：`engine` 的导出签名里只出现 `Sampler`（`Greedy()` 返回它）和 `*gguf.File`（渲染器要读文件里的模板）；`server` 只出现 `*engine.Engine` 与 `*tokenize.Tokenizer`，都是它该接收的。没有下层实现类型泄漏到上层签名里。

### 第九轮：四个怀疑，四次证伪

这一轮没有改代码，因为查的四件事都经得起检验。记录在这里是为了让后来者不必重查。

**「`best_of` 失败时每个 channel 起一个 goroutine，是无界的。」** 不是。
`internal/server/server.go:1525` 那个循环确实每 channel 起一个 goroutine，但 `maxChoices = 8`
（`server.go:852`）且 `maxBestOf = maxChoices`（`bestof.go:14`），两个字段都在 `invalid` 表里校验，
所以循环最多 8 次。另外两处 `go func()`：`streamChoices` 的 `wg.Wait` 一个，`serve.go:323` 的
`httpSrv.Serve` 一个。三处都有界。

**「`preFamilyOf` 的 271 行里应该有可合并的重复。」** 没有。34 个 `case` 全部返回 `preRule`
字面量（非字面量返回 0 个），按空白归一化后是 **33 条不同的规则**，只有 `preQwen2` 与
`preBailingMoE` 共享一条。这不是可以压缩的逻辑，是 34 条各自独立的数据。

**「`bench` 的表格用 `\t` 分隔，而 `docs/cli.md` 的样例是空格对齐。」** 一致。
`bench_suite.go:215` 是 `tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)`，`\t` 展开成至少两空格的列对齐，
文档样例就是它输出的样子。

**「`/stats` 的 `kv pool:` 行没有测试。」** 有。`chat_test.go:1288` 断言了它的四个组成部分：
`kv pool:`、`of 4 blocks free`、`16 positions each`、`64 total`。

四件事里最有价值的是第一条：**"每个 X 起一个 goroutine"看起来像无界，但上界由请求校验决定，
而不是由循环本身决定。** 判断并发是否安全，要看的是那个校验，不是循环的形状。

### `run` 仍然不迁到 engine

第九节把这件事列为待定。本轮看了它到底要什么，结论是**不迁**，理由比上一轮具体：

- 它现在用 `llama.NewKVCache`，`--stats` 报 `cache: N/M positions, S reserved`，三个数都来自
  contiguous cache 的 `Len()`/`Cap()`/`Bytes()`。engine 只有一个共享的 `BlockPool`
  （`Blocks()`/`BlockSize()`/`Bytes()`），语义不同——报出来的不会是同一个量。
- 它还靠 `cache.Len() == cache.Cap()` 判断"是不是窗口用尽"，据此打印预算被截断的提示。paged
  pool 没有等价的判断。
- 迁过去会让 `run` 测的不再是 `llama` 的生成路径，而它现在正是那条路径唯一的生产使用者。剔掉它
  之后，`llama.Generate` 只剩测试在用，那等于把一个有生产调用方的接口变成测试专用接口。

要做的话，先得回答"`run --stats` 应该报什么"，那是产品决定，不是重构决定。

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
