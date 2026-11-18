# AGENTS.md — Orbix 仓库约定

本仓是纯 Go 的 LLM 推理引擎，从 GGUF 到 OpenAI 兼容服务逐层实现。这里是给 agent 的仓库级补充；跨仓的个人习惯见 `~/.agents/AGENTS.md`，冲突时本文件优先。

## 硬约束

- **NEVER** 引入 CGo。`CGO_ENABLED=0` 是唯一支持的构建方式，`import "C"` 直接拒绝。
- **NEVER** 为了对齐 llama.cpp 的数值而引入 C 依赖；参考实现必须能在纯 Go 下读懂。
- 依赖方向只能向下：kernel 不 import scheduler，scheduler 不 import server。
- 模型文件（`*.gguf`）不进仓库，测试用合成 fixture。

## 入口

命令一律走 Makefile，不另起入口：

```bash
make build   # 编译到 bin/orbix，注入版本信息
make test    # go test -count=1 ./...
make test-race # 竞态检测（单独覆盖 CGO_ENABLED=1，仅测试用）
make lint    # golangci-lint run
make check   # 本地门槛：vet + test + lint + gofmt
make docs    # Markdown 门槛：rumdl check .
make fuzz    # 跑全部 fuzz target，FUZZTIME 控制每个的时长（默认 30s）
make bench   # 基准测试
make cross   # 交叉编译到每个声称支持的平台（CGO_ENABLED=0）
```

配置以单一事实源维护：Go 用 `go.mod`，lint 用 `.golangci.yml`，Markdown 用 `.rumdl.toml`，CI 用 `.github/workflows/ci.yml`。

CI 的 Go 相关 job 调用 Make 目标，不另写 `go build` 命令；两处例外各有原因：docs job 用 rumdl 官方 action（它是 linter 本身，不是安装器），test-darwin job 直接跑 `go test ./... -count=1`，因为 `make test` 固定 `CGO_ENABLED=0` 而对 macOS 原生测试没有额外价值，后者也不需要 `make` 里的其他变量。五个 job 各自验证一条 README 声明——gate（格式/vet/测试/lint）、cross-build（5 个平台的 CGO_ENABLED=0 构建 + 产物自报 `cgo: false`）、test-darwin（唯一非 Linux 的可执行测试环境）、test-race（需要 cgo，`make test-race` 单独覆盖）、docs（`make docs`）。

本地跑 `make check` 绿了，CI 的 gate 就应该绿；docs、cross、test-race、test-darwin 是需要额外工具或平台的独立门槛，各自有对应的 Make 目标，动到 Markdown 就加跑 `make docs`。这条分工是因为 rumdl 不在 Go 工具链里：把 `docs` 塞进 `check` 会让只装了 Go 的贡献者无法通过本地门槛。

`make fuzz` 同理不并进 `check`：它按时间跑而不是按用例跑，默认 30 秒一个 target，全套要几分钟。但 **seed corpus 由 `make test` 跑**——`go test` 会重放 `testdata/fuzz/` 里的每个文件，所以一个修好的崩溃会一直防住回归，不需要记得跑 fuzz。

## 目录

```text
cmd/orbix/          CLI 入口：参数解析、输出渲染
internal/gguf/      GGUF 容器解析：头部、元数据、张量目录、张量数据
internal/f16/       binary16 与 float32 互转（量化 scale 的基础）
internal/quant/     反量化：F32/F16/BF16/Q4_0/Q8_0 + 张量加载器
internal/kernel/    参考算子：matmul / RMSNorm / softmax / SiLU
internal/tokenize/  SentencePiece 分词，对齐 llama.cpp 的 SPM 实现
internal/llama/     Llama 架构：超参数校验、RoPE、KV cache、前向推理、采样
internal/engine/    调度：批量 prefill、批量 decode、每序列独立 logits 与 KV cache
internal/server/    HTTP：OpenAI 兼容 API、SSE 流式、断连取消、Prometheus /metrics
internal/otel/      OTLP/HTTP+JSON 追踪导出，不依赖 SDK
internal/gguftest/  合成 GGUF fixture：几 KB 的完整可跑模型，供各包测试使用
internal/<层>/      后续层按 README 的阅读顺序添加
```

## 分层与阅读顺序

README 承诺每层都有 `doc.go` 解释概念和设计取舍。新增包时必须写 `doc.go`，讲清"这一层在推理里负责什么"，而不是复述代码。

顺序：GGUF → kernels → KV cache → scheduler → server。新代码顺着这个顺序加，上层可以依赖下层。

## Go 版本与现代化写法

`go.mod` 声明 `go 1.27.0`，工具链低于 1.27 无法构建。用不上旧写法的地方就用新特性：

- **泛型方法**（Go 1.27 新增）：替代 `(T, bool)` 一族的访问器。见 `gguf.File.Get[T]` / `GetArray[T]`，由目标类型选择转换，越界或类型不符返回 `false`。
- **`iter.Seq` / `iter.Seq2`**：遍历用迭代器。`File.All()` 按 key 排序产出 `(string, Value)`，`File.Tensors()` 按文件顺序产出 `TensorInfo`；需要 `len` 时用 `TensorCount()` / `MetadataCount()`，不要为了 `len` 退回返回切片。
- **`slices` / `maps`**：排序、查找、转换用泛型版本。`slices.Sorted(maps.Keys(m))` 替掉"收集 key + sort.Strings"，`slices.SortFunc` 替掉 `sort.Slice`。
- **`strings.CutLast`**（Go 1.27 新增）：切最后一个分隔符用 `CutLast`，不要手算索引。
- **`sync.WaitGroup.Go`**（Go 1.25）：并发任务用 `wg.Go(fn)`，不需要 `wg.Add(1)` + `defer wg.Done()`。并发必须有界，用带缓冲 channel 当信号量，不做无界 fan-out。
- 能用的还有 `min` / `max` 内建、`for range N`、泛型容器。

新增普通访问器时先想想能否用 `Get[T]`；只有需要"缺失时返回默认值"这类语义时才另写方法，并在注释里说明原因。

## 代码风格

- 注释、日志、报错信息用英文，只解释非显而易见的约束和原因，不复述签名。
- 错误返回带上下文：`fmt.Errorf("gguf: read %q: %w", key, err)`，不吞异常。
- 数值常量（量化块大小、字节数）必须在测试里用独立的算术复算一遍，防止抄错后只在推理出垃圾输出时才暴露。
- 解析器遇到未知枚举值要如实报告类型，不猜、不静默跳过。

## 测试

- 单元测试不依赖网络和真实模型文件。需要能跑推理的测试用 `internal/gguftest` 构造 fixture：一个几 KB、两层的完整 GGUF，由同一个 reader 解析，`Prefill` 能跑通。
- fixture 的两个属性按需选择，不要靠猜 token 数：`NoEOS` 让生成跑满预算（默认 EOS 在随机权重下何时命中是偶然的），`ContextLen` 决定长请求能否装下。断言写相对关系（等待多久、轮次多少），不写某个 checkpoint 上的绝对值。
- 测试依赖执行时长时用等待时间的比值，不用「谁先返回」：短请求即使排队也必然先完成，用完成顺序会把正常调度判成失败。
- 确实要真实权重的测试（例如与 llama.cpp 逐 token 比对 logits）保留 `MODEL` 门禁，只用于本地交叉验证。
- 解析器的错误路径要测：截断、幻数错误、版本不符、越界张量。
- 覆盖率按**包**看，不按仓库看：`go test -cover ./pkg/` 只 instrument `pkg` 自己，所以别的包的测试调用过来的代码在这个数字里仍然是 0%。`Float32s` 显示 0% 而实际被 `tokenize` 的测试大量调用，追这个数字会浪费一轮。判断真缺口要问「这个函数有没有生产调用方」，不是「这个数字是不是 0」。
- 覆盖率可能依赖**环境变量**，但只在这两处：设齐四个 `*_MODEL` 后 `internal/engine` 从 85.5% 升到 89.1%、`internal/llama` 从 82.9% 升到 83.3%，其余九个包一个字节都不变。原因是只有 engine 的追踪测试和 llama 的权重测试需要真实 checkpoint，别的包的门控测试跑的是同一条代码路径，跳过了也不影响哪些语句被执行。
- 所以追低覆盖率之前先问「这个包的数会不会因为我没设环境变量而偏低」，答案在上述两个包里是「会」，别处是「不会」。`engine.finish` 是这个陷阱的唯一实例：无 `MODEL` 时 47.4%、有 checkpoint 时 84.2%，因为 `TestTracingRecordsRequestSpans` 被跳过就把整条追踪分支抹掉了——我为这个数字追过两轮，它从来不是缺口。
- 剩下那些设了 checkpoint 仍不着陆的行，多半是**防御性分支**：`engine.finish` 里 `cache.Reset()` 失败那条只在引用计数被破坏时触发，注释自己写着「a pool bug」。不为它伪造内部状态去凑覆盖率——不可达的错误路径正说明它所在的层是对的。
- 验证一条回归测试有没有判别力，要靠改动被测代码后重跑，不是靠读代码推断。改动时注意**同文本多处出现**：`moeRowsFromBatch` 和 `moeRowsFromDecode` 的构造体逐字相同，用 Python `str.replace` 不加限定会打到第一个，于是测出来「注入不生效」而实际是没打到目标。改完用 `git diff` 确认命中位置。
- 真实模型的交叉验证用独立参考实现（`uv run --with gguf`）解析同一文件，逐张量比对 SHA-256。
- `make test-race` 只覆盖测试里出现的并发。手动验证用 `CGO_ENABLED=1 go build -race` 出一个竞态版二进制，跑 `serve --max-active N`（配 `--blocks --block-size --prefix-cache` 开共享缓存），再打并发混合负载：非流式、SSE 流式、以及用 `timeout` 中途取消的请求。取消路径最容易暴露竞态，因为它要同时回收 KV block 和 goroutine。判据是日志里没有 `DATA RACE`，且 `orbix_requests_active` 归零——队列泄漏和竞态一样会让这个数停在非零。

## 文档

- README 是门面和索引，权威正文放在各包 `doc.go`。
- README 里承诺的 CLI 命令和 Make 目标必须真的可用；改接口时同步更新 README。
- Markdown 过 `rumdl check`，表格用对齐样式。

## 依赖

外部依赖可用，逐个判断，不是禁令。`go.mod` 目前没有 require 行，这是结论不是前提：每一处被考虑过的依赖都记录在下面的表里，包括没被采用的。

判断标准：**这个库替我们承担的复杂度，是否大于它自身的复杂度**。四条参照：

1. 标准库已经覆盖的，用标准库。`flag`、`encoding/json`、`net/http`、`crypto/rand` 覆盖了 CLI 参数、GGUF 中的 JSON、HTTP 服务端和追踪 id。
2. 库的能力要有实际去处。viper 的价值在多格式配置文件和配置来源合并；本项目只有 `ORBIX_MODEL` 一个环境变量、没有配置文件，引入它等于让库承担它无从发挥的工作。
3. 规模匹配。6 个命令、约 2400 行的 CLI 不需要命令树框架；如果命令数快速增长，这个判断要重新做。
4. 维护状态看**默认分支的最新提交**，不看 `pushed_at`（它会被无关推送刷新）。2026-11 核对：

   | 库 | 最新提交 | 判断 |
   | --- | --- | --- |
   | `urfave/cli` | 2026-11（v3.14.0） | 活跃 |
   | `kong` | 2026-10 | 活跃 |
   | `spf13/cobra` | 2026-07 | 四个月无提交，放缓 |
   | `spf13/viper` | 2025-10 | 十三个月无提交，停滞 |

   cobra 与 viper 都是成熟稳定的库，低提交量本身不等于不可用；但「最近在动」要求这两者时，urfave/cli 和 kong 是更实际的选择。

| 依赖 | 状态 | 理由 |
| ---- | ---- | ---- |
| （无） | — | 见上 |
| `viper` | 不采用 | 只有 1 个环境变量、0 个配置文件，没有多来源合并的需求 |
| `cobra` / `urfave/cli` / `kong` | 不采用（当前规模） | 6 个命令用 `flag` 加 `cmd/orbix/flags.go` 的共用注册已经够；命令数增长后重新评估 |
| OpenTelemetry SDK | 不采用 | 只用到一个 protobuf 消息的 JSON 编码，`internal/otel` 自行实现且已被真实 collector 验证 |
| `log/slog` | **采用（标准库）** | 服务端访问日志。要的只是分级结构化记录和可换 handler，`slog` 就是；配置见 `serve --log-format text\|json\|off` |
| `urfave/cli` + `Koanf` | 不采用 | Koanf 的价值在多来源配置合并，本项目 0 个配置文件、1 个环境变量；CLI 规模见上 |
| `Kong` + `cleanenv` | 不采用 | cleanenv 默认分支最近提交 2025-09，已 14 个月未动 |
| `go-playground/validator` | 不采用 | 26 个 flag 的校验是十几行显式判断，引入反射驱动的校验器换不到等价的可读性 |

新增依赖时：在 `go.mod` 之外同步本表，写明替我们承担的复杂度；**NEVER** 为了少写几十行样板而引入一个框架。
