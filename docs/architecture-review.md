# 架构深入分析（2026-11-24）

本文是对当前代码库的一次结构性分析，不是路线图。所有数字都来自这一轮在工作区里跑出的命令，命令附在每节末尾，便于复算。

口径说明：**代码行数只统计非测试文件**（`ls internal/*/*.go cmd/orbix/*.go | grep -v '_test.go$'`），测试行数为 `*_test.go` 之和。函数计数用 `go/ast` 遍历 `*ast.FuncDecl`，方法计入、测试文件不计入。

## 一、规模与形状

| 包                  | 代码行    | 函数（含方法） | 顶层函数 | 测试行    | 测试/代码 |
| ------------------- | --------- | -------------- | -------- | --------- | --------- |
| `internal/server`   | 5040      | 109            | 51       | 12377     | 2.46      |
| `internal/tokenize` | 4843      | 138            | 109      | 7073      | 1.46      |
| `internal/llama`    | 4820      | 135            | 40       | 12844     | 2.66      |
| `cmd/orbix`         | 3890      | 85             | 81       | 8602      | 2.21      |
| `internal/chat`     | 3151      | 78             | 40       | 4645      | 1.47      |
| `internal/engine`   | 2079      | 47             | 20       | 6092      | 2.93      |
| `internal/gguf`     | 1490      | 51             | 14       | 2968      | 1.99      |
| `internal/quant`    | 904       | 27             | 20       | 3510      | 3.88      |
| `internal/gguftest` | 817       | 22             | 14       | 372       | 0.46      |
| `internal/otel`     | 789       | 35             | 14       | 1681      | 2.13      |
| `internal/kernel`   | 705       | 28             | 20       | 2017      | 2.86      |
| `internal/f16`      | 154       | 4              | 4        | 380       | 2.47      |
| `internal/arch`     | 14        | 0              | 0        | 909       | 64.93     |
| **合计**            | **28696** | **759**        | **427**  | **63470** | **2.21**  |

**重构后的同一张表**（三项重构提交之后实测，是历史快照，不是当前值——上面的表才是）：

| 包                | 代码行 | 函数（含方法） | 变化                                                                                         |
| ----------------- | ------ | -------------- | -------------------------------------------------------------------------------------------- |
| `internal/llama`  | 4666   | 123            | -20 行；函数 +2（新增 `embedRow`/`addBiasTo`/`outputWeight`/`capLogits`，删掉 6 个重复函数） |
| `cmd/orbix`       | 3457   | 74             | +28 行；`model_files.go` 是新增的装配层，`chat.go` 净减 80 行                                |
| `internal/engine` | 1789   | 40             | +96 行；`drive.go` 与它的测试                                                                |
| `internal/gguf`   | 1488   | 51             | +29 行；`OpenCount`                                                                          |
| `internal/server` | 4038   | 89             | +5 行注释，判断改到 engine 哨兵                                                              |

行数变化不大，因为这次重构换掉的不是代码量而是**同一件事写了几遍**：`cmd/orbix` 少了三处重复装配、`llama` 少了六份重复函数、`engine` 多了两个共享入口。判断收益要看重复度而不是行数。

三个观察：

1. **仓库的重心在 token 与提示词，不在算子。** `tokenize` + `chat` 合计 7435 行，是 `kernel`（667 行）的 11.1 倍；`llama` 里真正属于"数学生成"的部分（attention、feedForward、rope、logprob）约 900 行。这与 README 的自述一致——这是学习工程，难点被有意放在"文件格式与模型格式"上。
2. **测试密度随"契约有多硬"上升，不随代码量上升。** `quant`（3.61）、`engine`（2.96）、`kernel`（2.90）最高，因为它们和外部参考逐值比对；`tokenize`（1.46）与 `chat`（1.59）最低，因为 `pre.go` 是 34 个预分词族的表驱动实现，单族覆盖成本高。`gguftest`（0.46）是测试夹具包，这个比值没有信息量。
3. **`cmd/orbix` 的 69 个顶层函数里只有 6 个是命令入口**，其余是渲染、校验与错误解释。CLI 不是薄壳，它持有真实的业务规则（见第五节）。

复算命令：

```bash
go run ./scripts/count.go -table   # 上面那张表的原始数据，可直接对照
grep -c "func Test" internal/*/*_test.go cmd/orbix/*_test.go
```

## 二、依赖图：分层是真的，曾经的两处渗透已修掉

`go list` 给出的实际内部导入边：

```text
cmd/orbix      → engine, gguf, llama, otel, server, tokenize
internal/server → engine, otel, tokenize
internal/engine → chat, gguf, llama, otel
internal/llama  → gguf, kernel, quant
internal/quant  → f16, gguf
internal/tokenize → gguf
internal/chat, kernel, f16, otel → （无内部依赖）
```

无环，且 kernel 层不依赖任何上层。

**这一节最初写的两个"渗透"都已经修掉了，记录保留是因为它们解释了现在为什么是这个形状：**

- `internal/server` 曾经直接导入 `llama`，用到 9 个符号（`Greedy`、`Temperature`、`Biased`、`Sampler`、
  `TokenBias`、`TokenLogprobs`、`Alternative`、`ErrPoolExhausted` 和 `Config` 的字段）。其中
  `ErrPoolExhausted` 是存储层错误穿过调度层：`server` 用 `errors.Is(err, llama.ErrPoolExhausted)` 把它
  翻成 400/503，而 `engine` 自己也做同样的判断——一个错误类型需要两个上层分别识别，说明"池耗尽"没有
  在 engine 边界上被翻译成领域错误。

  现在这些类型都住在 `internal/engine`（`Sampler`、`Alternative`、`TokenBias`、`Engine.ErrPoolExhausted`），
  `engine.ErrPoolExhausted` 的注释记着这段来历，`poolAwareErr` 把存储层的错误转成能匹配该 sentinel 的形状。
  `internal/server/imports_test.go` 拒绝任何文件（含测试）导入 `internal/llama`，所以这条边不会悄悄回来。

- `internal/engine` 的 `Request.Sampler` 字段类型曾是 `llama.Sampler`，现在是 engine 自己的 `Sampler`。
  `ChatRenderer` 仍住在 engine 包里（`chat.go`，246 行），是唯一一个"engine 里不做调度"的文件——那个
  位置保留着，理由在文件头部：渲染器要读 checkpoint 的模板，而它只依赖 `chat` 与 `gguf`，不构成向上依赖。

复算命令：

```bash
go list -f '{{.ImportPath}}: {{join .Imports " "}}' ./internal/... ./cmd/...
# 现在应该输出 0：server 不导入 llama。
go list -f '{{join .Imports "\n"}}' ./internal/server | grep -c "internal/llama"

# 上一步用 grep 扫 "llama.某导出名" 是行不通的：doc.go 和测试文件里的**注释**会命中，
# 看起来像渗透，实际是解释这件事为什么不能发生。
grep -rn "llama\.[A-Z][A-Za-z]*" internal/server/*.go
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

覆盖率快照（`go test -cover ./...`，2026-11-26 复测）：kernel 99.0%、f16 98.1%、quant 98.0%、tokenize 96.8%、engine 96.0%、server 95.7%、gguf 94.8%、chat 92.2%、otel 92.9%、llama 89.9%、gguftest 89.1%、cmd/orbix 88.0%。`internal/arch` 没有可测代码——它只有测试。`llama` 与 `cmd` 的低点集中在 `cmdServe`（39.4%，模型加载后的编排难以在测试里跑完整）；删掉 `Key` 之后 `llama` 已无 0% 的函数，最低是 `rememberPrefix`/`feedForwardBatch`/`feedForward`/`sameTokens` 的 66.7%。

**这一行是快照，不是断言。** 它没有守卫，因为它随每次改动变化，守它的测试会在无关提交上变红。复算只要一条命令，数字对不上时信命令。写下这一版时 `chat` 从 93.4% 降到 92.2%：`2605ec3` 给该包加了 Qwen3 需要的求值与解析代码（测得 91.3%），同一次复算又发现 `evalStringMethod` 里一段没人能走到的 `join` 分支，删掉后（`65c0a61`）回到 92.2%。**这两个百分点不是同一把尺子上的加减**：91.3% 与 92.2% 各自是在当时的测试集上测的。
删 `join` 这一步单独的账是分母少 11 条语句、已覆盖的只少 4 条（1073/1170 → 1069/1159，
两组剖面都在当前测试集上跑），即真正删掉的是 7 条永远不及的语句。`cmdServe` 从 39.2% 到
39.4% 同理。

这一行上一版写的是 2026-11 末的一组数，逐项复算后全部偏低（engine 94.7→96.0、server 94.9→95.7、otel 91.5→92.9），差额来自这段时间新增的测试。**`cmdServe` 是唯一反向的一项：上一版写 50.4%，实测 39.2%**，连续两次 `go tool cover -func` 都是这个数。同段的结论（`cmd/orbix` 的低点集中在 `cmdServe`）仍然成立，错的只是那个百分点。

## 八、门禁与验证现状

| 门禁                                     | 本轮结果                                                                                                        |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `go vet ./...`                           | 通过                                                                                                            |
| `go test -count=1 ./...`                 | 通过，24 秒；engine 占 22.7 秒                                                                                  |
| `just lint`                              | 通过，0 issues（自动选中 v2.15.0）                                                                              |
| `just docs`（rumdl）                     | 通过，10 个文件（含本文）                                                                                       |
| `just hooks`（prek）                     | 通过                                                                                                            |
| `just cross`（5 目标）                   | 通过                                                                                                            |
| `just check` 整体                        | **通过**（本文写完后修好了下面两条）                                                                            |
| `just test-race`                         | 通过，无 `DATA RACE`；server 47.1 s，f16 54.6 s                                                                 |
| `just isolate`（`-count=2 -shuffle=on`） | 通过，13 个包全绿，无顺序依赖、无状态泄漏                                                                       |
| `just FUZZTIME=10s fuzz`                 | 通过，9 个 target 无崩溃（2026-11 末复算：3 个在 tokenize，其余 chat、gguf、kernel、llama、quant、server 各一） |

后三条是 CI 的独立 job 或辅助配方，`just check` 不跑它们，所以容易被忘。**2026-11 末一次完整跑**的结果记在这里，因为此前没有任何地方写过它们跑过——一个"CI 会跑"的门禁和一个"实际跑过"的门禁不是一回事。

fuzz 的 seed 语料只有两个文件，都在 `internal/gguf/testdata/fuzz/FuzzRead/`，其余八个 target 一个都没有。这不是缺口：`go test` 会重放已有的 seed（实测 `FuzzRead/seed#0` 到 `seed#4` 都执行），而当前 fuzz 没发现崩溃，所以没有新语料落盘。语料薄说明还没有崩溃被查出来过，不说明检查没跑。

初测时 `just check` 在这台干净的树上还是红的，两个原因都不是代码：

1. **`just lint` 只看了 `PATH` 上的第一个 `golangci-lint`**。那一个是 mise 装的 v1.64.8，而 `.golangci.yml` 是 v2 配置，于是门槛报告"工具缺失"——同时一个能用的 v2.15.0 就排在 `PATH` 第三位。这比不检查更糟：它把一个可用的环境判成不可用，而错误信息指向版本号，不指向"还有别的候选"。
   现在改成用 `type -ap` 枚举全部候选、逐个探测、取第一个 major 为 2 的；一个都没有时把每个候选的版本和路径都列出来。三条路径都实测过：正常（选中 v2）、只有 v1（列出 `1  /…/golangci-lint` 并退出 1）、完全没有（提示安装）。

   写这段踩了两个 bash 与 just 的交互坑，都记在 justfile 注释里：字符串里的字面换行会**结束 recipe**（just 按缩进读正文），而想用 `$(printf '%s\n' …)` 绕开时，命令替换会**吃掉末尾换行**，最后一行和下一句粘在一起。最终用 bash 数组。

2. **`scripts/jinja_diff.py` 的期望表过期**：`inline_if` 被列在"应当拒绝"里，而 `internal/chat` 在 `6da2e26` 之后已经支持内联条件（Gemma 3 模板需要它）。门槛因此长期出口 1，报的是一个**已经实现**的构造——一个不随功能更新的期望表，会把真实的回归藏在它后面。
   现在 `inline_if` 移入 SUBSET 并扩成五个用例（两个分支、字段条件、与比较/连接的优先级、嵌套），空出的"应拒绝"位置换成文档确实承诺拒绝的短式 `a if cond`。结果：**598 例，588 与 Jinja2 一致，10 例按设计拒绝，exit 0**（此后补齐 `is none`、`is string` 的缺陷与 `-`、`%` 的对照，扩到 712 例、700 一致、2 例按设计发散；再后来 Qwen3 模板带来六个构造，语料随之补到 908 例、896 一致，数字见第十节后的复算清单）。

   这个坑此后又踩了一次，而且是同一条路径的镜像：`namespace()` 在 `2605ec3` 之后已经实现，却仍留在 `UNSUPPORTED` 表里，门槛从那一刻起报的是一个**已经渲染成功**的构造。上一段自己写着"期望表不随功能更新，是让门槛长期报假回归的原因"，而它没有阻止第二次——因为它只说了要更新，没有让"没更新"变得可见。

   真正修掉这一点的是给新构造逐个补上语料条目，并各做一次注入。四种改动当时**全都不让门槛变红**：`<` 改成 `<=`、切片的负起点当成 0、`split` 按首字符切、`namespace` 写回当前帧。原因是旧语料里没有相等边界、负索引、多字符分隔符，也没有 loop 之外的字段赋值。补上这些用例后再注入，四种都被抓住。语料条目是对照的判别力所在，不是例数。

覆盖率与 checkpoint 门禁：74 处 `t.Skip` 分布在 29 个测试文件里（2026-11-25 复算；上一版写 73 和 28，写下时是 65 和 24），由 **14** 个环境变量控制——`MODEL`、`QWEN2_MODEL`、`QWEN3_MODEL`、`GEMMA_MODEL`、`GEMMA2_MODEL`、`GEMMA3_MODEL`、`MISTRAL_MODEL`、`MIXTRAL_MODEL`、`MODEL_KQUANT`、`MODEL_Q4_1`、`MODEL_Q5_0`、`MODEL_Q5_1`、`MODEL_IQ4_NL`、`CODER_MODEL`。本机没有任何 `.gguf`，所以这一轮**所有与真实权重相关的结论都没有被复测**；CI 的 skipped-tests 报告行存在的意义就是这个。
`ORBIX_REFERENCE_DIR` 与 `ORBIX_PREDUMP`、`ORBIX_QUANTDUMP`、`ORBIX_ROPEDUMP` 不算在这 14 个里：它们
不是 checkpoint 门禁，而是让跨实现对照脚本拿到输入或 Go 侧输出的开关，没有它们时那几个测试自己跳过。

这一句原先写「11 个」，列出的名单也少了 `MODEL_Q4_1`、`MODEL_Q5_0`、`MODEL_Q5_1`、`MODEL_IQ4_NL`。
四个都读不到 `os.Getenv("...")` 的字面量——它们是 `internal/quant/formats_real_test.go` 那张表的行，
由 `os.Getenv(g.env)` 间接读取——所以只 grep `Getenv` 会漏掉，而 `docs/models.md` 的对照表一直是对的
（它由 `TestModelsDocGateTableMatchesTheCode` 双向校验）。**同一份事实写在两份文件里，只有一份有守卫，
另一份就烂了。** 这份文档里的数字没有守卫，这也是它第二次在该节上出错。

**后续（2026-11 同日）：上面「本机没有任何 `.gguf`」是那一轮的事实，不是当前状态。** 本机有
Hugging Face checkpoint 目录，用仓库自己的转换器就能造出 GGUF，于是真实权重这条路不再依赖下载：

`/models/Qwen2.5-0.5B-Instruct`（`model_type` 为 `qwen2`，494,032,768 个参数，
`tie_word_embeddings: true`，所以安全张量里没有 `lm_head.weight`）。

```bash
uv run --offline --with numpy python3 scripts/hf_to_gguf.py \
  /models/Qwen2.5-0.5B-Instruct /tmp/orbixreal/qwen2-f32.gguf
# loaded 290 tensors / wrote 290 mapped tensors / vocab 151665 / 1979340608 bytes
```

`QWEN2_MODEL` 指向它之后，与真实权重相关的结论**这一轮全部复测过**，每条都有输出：

| 检查           | 命令                                                                                  | 结果                                                           |
| -------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| 端到端生成     | `orbix run --model … "The capital of France is"`                                      | `Paris. It is the largest city in Europe…`                     |
| 服务端 chat    | `curl` 打 `/v1/chat/completions`                                                      | `content: "Paris"`，`finish_reason: stop`，`prompt_tokens: 44` |
| 模板是真的在用 | `serve` 启动横幅                                                                      | `chat: model template`                                         |
| 独立解析器对照 | `MODEL=… just xcheck`                                                                 | `290 tensors compared; all agree on type, shape and SHA-256`   |
| numpy 参考     | `QWEN2_MODEL=… go test ./internal/llama/ -run TestMatchesNumPyReferenceOnRealWeights` | `qwen2` PASS，3 层 residual 全一致                             |
| 架构门禁       | `QWEN2_MODEL=… go test ./internal/llama/ -run Qwen2`                                  | 6 个 PASS（bias、NeoX rope、prefill 对齐等）                   |
| 真实模板对照   | `uv run --with jinja2 --with gguf python3 scripts/jinja_diff.py --template-file …`    | `18 cases compared, 18 agree with Jinja2`                      |
| 竞态混合负载   | `MODEL=… just stress`                                                                 | `ok -- no races, no 5xx, active returned to 0, 3 cancelled`    |

两条数字值得单独记，因为它们以前只有推断：

- **`orbix run --stats` 报 `params: 494032768`**，与 safetensors 头里逐张量求和的参数数一致。
  转换器写 F32，290 个张量全部 F32，`model inspect` 报 `1.84 GiB`，峰值 RSS 实测 2,787,124 kB
  （约 2.66 GiB）——即 `docs/configuration.md` 说的「常驻内存跟参数走，不跟文件大小走」在真实
  checkpoint 上成立，且扩张比 1.4x 是因为这份文件本来就是 F32。
- **转换出的词表是 151665 个 piece，而 `config.json` 声明 `vocab_size: 151936`。** 差 271 个来自
  `tokenizer.json` 的 `added_tokens`：最大 id 151664，中间没被定义的位置由转换器补成
  `<unused{i}>`。生成的 token 落进这些洞里的概率极低，但 `logits` 的宽度是 151936 而不是词表长度，
  所以两者不是同一个数——`model inspect` 与 `run --stats` 都报 151936，读的人容易以为词表就是那么长。

竞态那一条要看清它证明了什么：`just stress` 的 3 次 cancel 来自排队中被挂断的客户端，判据是
日志无 `DATA RACE`、无 5xx、`orbix_requests_active` 归零，脚本另外要求取消数非零，否则报失败而
不是 ok。这台机器 `load average` 在 10–47 之间波动（128 核，`nproc`），所以**这一轮没有做任何
计时类结论**；18.5 tok/s 是单次观测，不作为性能结论引用。

## 九、结论与建议的下一步

**本轮已做**（三项小时级的，全部有实测验证）：

1. ~~**删 `PrefixIndex.Key`**~~ — 已删，`Remember` 的注释改成描述真实契约，`internal/llama` 里再无 0% 的函数。
2. ~~**修 `jinja_diff.py` 的期望表**~~ — 已修，门槛从 exit 1 回到 exit 0（当时 598 例，后来 712 例，Qwen3 的构造补进来之后是 908 例）。顺带把内联条件补成五个用例，并明确写进注释：期望表不随功能更新，是让门槛长期报假回归的原因。**这条修正自己又复发了一次**：`namespace()` 实现了但没从 `UNSUPPORTED` 移出，门槛在 `2605ec3` 之后一直是红的而无人发现。第二次的教训与第一次不同——文档里写着"要更新期望表"没有用，有用的是每个新构造都配一条语料、每一条都做一次注入验证，以及把它接进 CI。
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
**19 个函数在 100 行以上**（2026-11-26 复算；本轮拆分前是 18，详见本节末尾的清单；
用 AST 按 `FuncDecl` 的起止行数，不用括号计数——注释里的花括号会把
统计骗成几千行；762 个函数）。本轮按当下代码逐个处理，每改一个都核对错误字符串
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

**现存 19 个 >=100 行函数**（`go run ./scripts/count.go -long` 实测，762 个函数）：`chat.evalExpr`
316、`cli.cmdServe` 275、`tokenize.preFamilyOf` 271、`cli.cmdBench` 240、`server.streamOneChoice`
205、`server.generate` 199、`cli.runTo` 177、`tokenize.loadPreTokenization` 157、`server.blockingResponse`
141、`cli.chatSession` 135、`llama.attentionBatch` 126、`tokenize.splitKimiK2` 123、`engine.Submit`
122、`server.handleChat` 122、`gguftest.kvs` 119、`server.resolvePrompt` 108、`gguf.Read` 107、
`chat.lex` 103、`cli.validateServeFlags` 101。

这一份上一版写的是「十九个」，逐项复算后变的不只是总数：

- `chat.parsePostfix` 上一版新进入清单（113 行）。它后来被拆成 `parsePostfix` 与
  `parseBracket`——加 `[a:b:c]` 都要落在这里，两个函数各管一段比一个函数管三种形式清楚——
  于是它降回 100 行以下、离开清单，而 `chat.evalExpr` 从 310 涨到 316。**这条值得留着**：它是
  这份清单最常见的一次真实变化，加一个语法形态会长在一个函数上，拆开就消失。
- `chat.evalExpr` 从 271 涨到 310，`cli.cmdServe` 从 261 涨到 275，`server.generate` 从 192
  涨到 199，`cli.runTo` 从 165 涨到 177，`server.blockingResponse` 从 131 涨到 141；其余几项
  未变。函数总数从 751 涨到 759。
- 上一版漏过 `server.handleChat`（122 行），它和表里列出的 `engine.Submit` 一样长；
- `cli.cmdRun` 已改名为 `cli.runTo`（`cmdRun` 现在只是把 `os.Stdout` 传进去的一行）。它一度是
  174 行，因为流式输出直接写在函数体里；把那段移到 `generateStreaming` 之后是 165，仍比改写前的
  147 长，多出来的是新增的窗口预算提示与写入错误检查；
- `cli.validateServeFlags` 是新进入清单的一项，101 行。

`validateServeFlags` 进入清单是这套拆分的必然结果：从 `cmdServe` 切出来的部分自己也长到了
一百行。这不说明拆分做错了，但说明「>=100 行」是个会自我复发的阈值，对它的计数要当作体检
指标而不是任务清单。

这份清单原先写的是"十二个"，因为生成它的脚本只打印前 12 行（`if i >= 12 { break }`），
而总数行写的是真实值 17。两个数字都在同一份输出里，被截断的那个进到了文档。补上的五个是
`Submit`、`kvs`、`resolvePrompt`、`Read`、`lex`。

复算用 `go run ./scripts/count.go -long`：计数行和列表来自同一个切片，不会再各说各话。

**而且现在有守卫了。** `TestArchitectureReviewLongFunctionCountsMatchTheCode` 跑同一条命令，把这份
文档里的两组数字（本节开头的「N 个函数在 100 行以上 / M 个函数」和清单上方的「现存 N 个 / M 个函数」）
以及整份名单逐项对上。这条测试存在的原因是这份清单已经悄悄过期两次，而每次都是靠人重新跑一遍才
发现的；文档里写「数字可复算」并不能让数字保持正确，让某个测试去复算才行。

它读文档的数字而不只是比名单，是因为两者会各自错：删掉一个名字而数字不变时，只比数字的守卫看不
出来；数字改了而名字没改时，只比名单的守卫看不出来。所有四种情况都用注入验证过——改左边的数字、
改右边的数字、从名单里删一个、往名单里加一个不在清单里的函数——每种都会红。把 `HEAD` 上那份旧
文档放回去，它同时报出三条：`751 → 759`、`19 → 20`、以及 `parsePostfix` 没被列出。

逐个判断如下，而不是笼统地"改动面大"：

- **`preFamilyOf`（271）是数据不是逻辑。** 34 个 `case`，每个返回一个 `preRule` 字面量，注释是那个
  家族的预分词正则原文。改成 `map[preType]preRule` 同样 271 行，注释还得逐条贴，而且 `(preRule, bool)`
  的 `bool` 正是"这个家族是否已知"的判定，表驱动能保留它——所以改与不改都不改变读者要读的量。
  **行数是数据密度，不是复杂度。**
- **`evalExpr`（310）的骨架是分派。** 20 个 case 里 15 个只有 2–20 行"求子节点再合并"，拆成 20 个
  方法只增加间接层。有独立规则的那一个已经拆出去了（`evalArithmetic`，见下节）。它这一轮又长了
  39 行，全部是 Qwen3 模板要的构造；下一个该从这里拿出去的是字符串方法（`strip` 族与 `split`），
  它们已经有自己的 `evalStringMethod`，而分派那一支仍留在 `evalExpr` 里。
- **`chat.parsePostfix`（原 113，现不在清单里）** 处理 `.field`（含方法调用）、`|filter`（含
  关键字参数）和方括号。加 `[a:b:c]` 时它被拆开：`parseBracket` 用一个扫描器读三种方括号形式
  （索引 / 切片 / 带步长），`parsePostfix` 只负责把后缀串起来。**这是这份清单里唯一一次
  "拆开真的让两个函数都变短"的例子**——因为切开的边界正好是一次后缀的完整解析，而不是把一步
  分成两半。
- **`cmdServe`/`cmdBench`/`cmdRun`/`chatSession`** 剩下的都是连续装配流程，只跑一次、只失败一次。
- **`attentionBatch`（126）** 我在第三节判断过不该合。`LoadWeights` 当时还是 134 行，现在是 78，
  已经不在本表里了。
- **`generate`/`streamOneChoice`/`blockingResponse`/`loadPreTokenization`** 是前几轮拆到一半的产物，
  再拆一层会把一步分成两半而不减少要同时握住的东西。
- **补上的五个**：`engine.Submit`（122）与 `server.resolvePrompt`（108）都是"校验 + 组装"的连续
  流程，每一步的失败要带自己的参数名，拆开会让那个名字离校验两跳远；`gguf.Read`（107）与
  `chat.lex`（103）是各自格式的单一状态机，分段的边界只在函数内成立；`gguftest.kvs`（119）是
  测试夹具的元数据表，和数据一样不可压缩（同 `preFamilyOf`）。

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

### 第十一轮：四个怀疑，四次证伪

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

### 第十二轮：默认值与守卫，用注入量了一遍

`LoadConfig` 给六个键提供默认值，`workers()`、`SoftcapValue`、`dataStart` 各有一处守卫。方法
是同一个：把值改错或把守卫放松，跑整包，数失败的测试。目的是分清「有测试盯着」和「只是看起来
合理」。

| 改动                                 | 失败的测试 |
| ------------------------------------ | ---------- |
| `HeadDim = Hidden/Heads` → `Hidden`  | 130        |
| `RopeScale = 1` → `2`                | 9          |
| `RopeBase = 10000` → `1000`          | 8          |
| `EmbedScale = sqrt(Hidden)` → `1`    | 6          |
| `RopeDim = HeadDim` → `HeadDim/2`    | 3          |
| `KVHeads = Heads` → `1`              | 1          |
| `RMSNormEps = 1e-5` → `1e-9`         | **0**      |
| `workers()` 的 `n <= 0` → `n == 0`   | **0**      |
| `SoftcapValue` 的 `c <= 0` → `c < 0` | 13         |
| `dataStart` 去掉 `nbytes < 0`        | 1          |

两处 0 是真实的空白，都补了测试：

- **`RMSNormEps`** 的默认值只在文件**不写** `attention.layer_norm_rms_epsilon` 时生效，而每个
  fixture 都写了它。缺失该键合法（llama.cpp 同样当可选），且这个值决定 RMSNorm 里怎么防除零，
  写错是静默的数值偏差而不是崩溃。
- **`workers()` 的负数**：三层下游守卫（`scoreBuffers` 把 slots 夹到 1、`split` 对 `parts <= 1`
  退化、`Runner.Workers` 对 nil 返回 1）都能兜住崩溃，但兜不住「悄悄变成串行」。CLI 会照报调用方
  给的值，而实际 fan-out 是 1。

`KVHeads` 只有 1 是合理的：它只在文件省略 `head_count_kv` 时生效，而那个省略很罕见。

排除掉的一处：`RMSNorm` 里 `n == 0` 提前返回，删掉后零测试失败——但那是因为 `for i := range n`
在 `n=0` 时本就不执行，删掉只是多算一次 NaN 再丢弃。它是微优化不是正确性守卫，不必为它写测试。

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
uv run --offline --with jinja2 python3 scripts/jinja_diff.py   # 908 cases, 896 agree, 2 diverge by design, exit 0

# 跨实现对照：三个脚本各自从上游写期望值，不复述 Go 的读法
ORBIX_PREDUMP=/tmp/pre.json go test ./internal/tokenize/ -run TestDumpPreSplit
uv run --offline --with regex python3 scripts/pre_xcheck.py /tmp/pre.json   # 2 families, 104 inputs

ORBIX_QUANTDUMP=/tmp/q.json go test ./internal/quant/ -run TestDumpDequant
uv run --offline python3 scripts/quant_xcheck.py /tmp/q.json               # 14 formats

ORBIX_ROPEDUMP=/tmp/rope.json go test ./internal/llama/ -run TestDumpRope
uv run --offline --with numpy python3 scripts/rope_xcheck.py /tmp/rope.json # 6 cases, 2048 values

uv run --offline --with gguf python3 scripts/fixture_xcheck.py   # 18 keys, 21 tensors
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
