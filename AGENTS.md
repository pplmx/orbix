# AGENTS.md — Orbix 仓库约定

本仓是纯 Go 的 LLM 推理引擎，从 GGUF 到 OpenAI 兼容服务逐层实现。这里是给 agent 的仓库级补充；跨仓的个人习惯见 `~/.agents/AGENTS.md`，冲突时本文件优先。

## 硬约束

- **NEVER** 引入 CGo。`CGO_ENABLED=0` 是唯一支持的构建方式，`import "C"` 直接拒绝。
- **NEVER** 为了对齐 llama.cpp 的数值而引入 C 依赖；参考实现必须能在纯 Go 下读懂。
- 依赖方向只能向下：kernel 不 import scheduler，scheduler 不 import server。这条由 `internal/arch/layers_test.go` 强制——它遍历每个 internal 包的**非测试**文件，遇到层级表判定为反向或同层的边就失败。新包必须加进 `layer` 表，否则守卫报「not in the layer table」而不是静默跳过；某个边确实需要例外时写进 `allowed` 并给出理由。
- 模型文件（`*.gguf`）不进仓库，测试用合成 fixture。

## 入口

命令一律走 justfile，不另起入口：

```bash
just build   # 编译到 bin/orbix，注入版本信息
just test    # go test -count=1 ./...
just test-race # 竞态检测（单独覆盖 CGO_ENABLED=1，仅测试用）
just lint    # golangci-lint run
just check   # 本地门槛：vet + test + lint + gofmt
just docs    # Markdown 门槛：rumdl check .
just fuzz    # 跑全部 fuzz target，FUZZTIME 控制每个的时长（默认 30s）
just bench   # 基准测试
just cross   # 交叉编译到每个声称支持的平台（CGO_ENABLED=0）
just hooks   # 跑 prek 的 pre-commit 钩子（非 Go 文件的空白、换行、YAML/TOML/JSON 语法）
just commits # 按 Conventional Commits 校验某段提交信息；改范围用 `just RANGE=base..HEAD commits`
just release-preview # 看下一次 release 会是什么版本、changelog 写什么，不改任何东西
just release # 升版本、写 CHANGELOG.md、打 tag（tag 推到远端后由 release.yml 发布产物）
```

上面是门槛和发布入口。另外十七个是辅助配方，不进门槛，但会用得上（`just --list` 里的第 29 条是 `default`，即不带配方时等价于 `just check`）：

```bash
just fmt     # gofmt -w，修 check 报出来的格式问题
just vet     # go vet，check 的子集；改 PKG 收紧范围：just PKG=./internal/gguf vet
just clean   # 删 bin/、.rumdl_cache、.cache 和 *.test/*.out/*.prof；不碰 go build 的全局缓存
just isolate # 测试顺序依赖与状态泄漏：-count=2 -shuffle=on；改测试辅助或包级变量后跑
just stress  # 竞态版二进制跑混合负载（非流式/SSE/中途取消）；要 MODEL=<真实 gguf>，验 active 归零
just xcheck  # 用 Python 的 gguf 包独立解析同一文件，逐张量比对类型、维度和 SHA-256；要 MODEL= 和 just build
             # MODEL 不必是真实 checkpoint：just fixture 造的合成文件同样能跑（39 个张量全对）
just jinja-diff # 用真实 Jinja2 渲染 chat 模板与本包的输出逐字比对；不需要 checkpoint，语料在脚本里
just xcheck-pre # 预分词器与 GPT-2 / Llama 3 的上游正则对照；不需要 checkpoint
just xcheck-quant # 反量化与 ggml quantize.c 的块布局对照，14 种格式、14750 个值（脚本末尾会打印总数）
just xcheck-rope # RoPE 与 numpy 按 RoFormer 定义算出的旋转对照，6 组参数
just xcheck-fixture # 用 Python gguf 包解析 gguftest 的合成 fixture，比对元数据与张量目录
just docker  # 本地构建容器镜像并打印里面的版本号；发布镜像是 build-push-action 走 buildx 出的双架构
             # 清单，这个配方只建本机架构——本地复现双架构要模拟器，换不到什么。它不推任何地方，
             # 用途是确认 Dockerfile 还能建、且镜像里的 version 与当前 checkout 一致。
just skips   # 数一次运行里实际跳过了多少个测试，CI 每轮打印的就是这个数。
             # 它先打一行 `gates set: N of 14`，再打 `skipped: M`：一个数字单独看没有意义，
             # 只有配上当时设了几个检查点才知道它说的是什么。实测（2026-11-28）：
             # 0 个 → 27；4 个（gemma2 Q3_K_L 充当 MODEL/GEMMA2_MODEL/MODEL_KQUANT、
             # gemma3 270M 充当 GEMMA3_MODEL）→ 20（那一档测于那六条还带门禁的时候，现在更低）。
             # **27 是「没有任何真实权重」的数，不是仓库的常数**：它随 gated 测试增删而变。
             # 本会话先涨到 34（加了两条 gemma 门禁测试），随后六条改用 fixture 兜底
             # （`TestTCPResetStopsGeneration`、`TestTensorDecodesTheSameTwice`、
             # `TestRoundTripEveryRune`、`TestRoundTripEveryRuneByteLevel`、
             # `TestRoundTripText`、`TestRoundTripTextByteLevel`）掉到 28，再一条
             # （`TestDocsExplainTheirVersionMarkers`）改成按 tag 列表核对后掉到 27。
             # 5 个和 6 个那两档这一轮没能复算——手头只有 gemma2 与 gemma3 两个文件，用它们
             # 冒充 MODEL/QWEN2_MODEL 等会让 suite 失败，而 `just skips` 在失败时不报数字
             # （这是它该有的行为）。要更新那两档得先凑齐六个真文件。2026-11-26 的读数是
             # 5 个 → 13、6 个 → 10；最后剩下的那些要的是这台机器没有的东西：IQ4_NL/Q4_1/
             # Q5_0/Q5_1 四种格式、gemma-1、mistral-v2、一个 MoE 文件、三个只在设了
             # ORBIX_*DUMP 时执行的 dump 测试，以及四个需要别的真实文件的（prefix cache 复用、
             # 真实权重矩阵统计、gemma3 的 prompt token ids、FIM 的 prompt ids）。
             # 这些数字都没有守卫，换机器就要重数。4 个档一度不可测，因为 gemma3 的两个
             # reference fixture 每次带门禁都报红，`just skips` 就只肯说「这轮是坏的」；
             # 把那两个 fixture 标成记录已被取代的读数之后才拿到数字——一份「已知失效但仍红」
             # 的测试会让整条配方失去读数，代价不止是噪声。
             # 它与 architecture-review.md 的「74 处 t.Skip」不是一回事：那是代码里的跳过点，
             # 这里是运行时真正跳过的。后者少于前者，因为有的跳过点在子测试里、有的在同一轮
             # 被别的路径覆盖。suite 失败时这条配方会先打印计数再以非零退出。
just coverage # 按函数看覆盖率，每个包都被 instrumented——这是回答「这段代码没人用吗」的那条。
             # `go test -cover ./pkg` 只 instrument 自己，被别的包的测试调用的函数会显示 0%，
             # 本节「覆盖率按包看」那段讲的就是这个坑；这条配方给出另一个数。实测引擎各包
             # 已无 0% 函数；剩下的六个是三个 `main`（测试调不到）和 scripts/count.go 的四个
             # 辅助函数（那个包没有测试文件，它是开发工具不是引擎的一层）。
             # 比裸的 -cover 慢（约 50s vs 35s），所以不进 check。
just mutate file pkg # 把一个文件里的 `return true`/`return false` 逐个翻过来、重编、重跑，
             # 报出哪些翻转测试察觉不到。与 `just coverage` 回答的是两个问题：那条只说明某行
             # 没跑到，这条说明跑到的时候测试能不能分辨对错。本会话用它找到过一个真缺陷——
             # checkModel 的 `return false` 有覆盖，翻成 true 之后套件仍全绿，而响应体变成了
             # 错误信封后接一个完整 model 对象。存活不等于缺陷：拒绝路径的返回值常在写出响应
             # 之后才被读（变异等价），注释里写明不可达的分支是已记录的取舍。每个位点各跑一遍
             # 全套，二十个位点的文件就是二十遍，目标选小一点。
             # 扫过的文件与存活数（2026-11-28，同一台机器）：`server/logprobs.go` 4/4 捕获、
             # `server/server.go` 20 个位点 7 个存活、`engine/engine.go` 12 个里 5 个存活、
             # `llama/prefix.go` 6 个里 1 个、`gguf/scalar.go` 4/4、`chat/parse.go` 4/4、
             # `tokenize/pre.go` 14/14。全是 100% 的那四个不是偶然：`scalar.go` 是泛型 switch
             # 的每种越界、`parse.go` 的 `isIdentifier` 每个字符类、`pre.go` 的 34 个预分词族，
             # 每条都有一条用例盯着。存活的那些逐个读过，都在拒绝路径或不可达分支上。
             # **捕获率不是质量分**：`isIdentifier` 4/4 捕获，而我读它时两次以为 `default:
             # return true` 是 bug（它其实是「前面都合法，到此可以收尾了」），实测返回才对。
             # 变异只回答「改动会被发现吗」，不回答「当前值对不对」。
just check-guard doc pattern with guard # 把文档里的一处说法改错，报出守护它的测试红不红。
             # 与 `just mutate` 是一对：那条问测试能不能分辨代码改动，这条问守卫能不能分辨
             # 文档写错。两者都需要，因为守卫可以还在、还绿、却已经不看原来那件事了——本会话
             # 出现六次，而每次都是手工改坏文档再看，所以二十多条文档守卫里只有三条被这样验过。
             # **最危险的是静默空操作**：pattern 匹配不到时改动等于没做，守卫当然还是绿，报告
             # 就会说「这条守卫没牙」而实际什么都没测。实测踩过：替换「14 个门禁」报 GREEN，
             # 因为文档里写的是「由 **14** 个环境变量控制」。所以 pattern 落空时配方以退出码 2
             # 拒绝继续，不改完就看结果。
just fixture # 生成一个够大的合成 checkpoint，让 `just stress` 有东西可打断（默认落 .cache/）
just         # 不带配方等价于 just check
```

配置以单一事实源维护：Go 用 `go.mod`，lint 用 `.golangci.yml`，Markdown 用 `.rumdl.toml`，提交钩子用 `.pre-commit-config.yaml`，行尾与二进制标记用 `.gitattributes`，编辑器约定用 `.editorconfig`，命令入口用 `justfile`，版本与 changelog 用 `.cz.toml`，CI 用 `.github/workflows/ci.yml` 和 `.github/workflows/release.yml`。

CI 的 Go 相关 job 调用 just 配方，不另写 `go build` 命令；三处例外各有原因：docs job 用 rumdl 官方 action（它是 linter 本身，不是安装器），test-darwin job 直接跑 `go test ./... -count=1`（因为 `just test` 固定 `CGO_ENABLED=0` 而对 macOS 原生测试没有额外价值，后者也不需要 `justfile` 里的其他变量），docker job 手写 `docker build` 而不用 `just docker`（本地配方只建本机架构并读一次版本号；CI 还要验镜像声明了非 root 的 65534、容器真能转 healthy，那是 sha 与 tag 的差）。第三处此前写成两处，是因为改这句话时只数了 Go 代码里那两条路。十三个 job 各自验证一件事——gate（格式/vet/测试/lint）、cross-build（5 个平台的 CGO_ENABLED=0 构建；编译后读回产物里的 `build\tCGO_ENABLED=0` 标记，所以「没带上 cgo」也被验了——不能运行的目标也能这样查，标记在文件里）、test-darwin（唯一非 Linux 的可执行测试环境）、test-race（需要 cgo，`just test-race` 单独覆盖）、isolate（`just isolate`，`-shuffle=on -count=2` 抓 gate 抓不到的两种失败：依赖前一条测试留下状态的，和不幂等的；配方自己不并进 `check`，因为 shuffle 失败给的是种子不是 bug，而 gate 要回答是或否）、stress（`just fixture` 造出够大的合成 checkpoint，再用竞态版二进制跑混合负载；四条判据是 DATA RACE、5xx、active 未归零、以及「什么都没被取消」——最后一条防止在什么都没测到时报 ok）、prek（`just hooks`，补 `check` 看不见的非 Go 文件格式问题）、commits（`just commits`，按提交范围校验信息格式）、docs（`just docs`）、docker（构建 `Dockerfile` 并真的启动容器等它 healthy——`scratch` 镜像没有 shell，`HEALTHCHECK` 跑的就是 `/orbix healthcheck`，所以这个 job 同时验证了镜像能建和那个命令在镜像里能用；静态读 Dockerfile 只能确认 ARG 到 ldflags 的接线和 `.dockerignore` 的排除项，确认不了 `FROM golang:1.27` 这个 tag 能不能拉到）、xcheck（`just xcheck`，用 Python 的 gguf 包独立解析同一个文件，逐张量比对类型、维度和 SHA-256；用它是因为这是唯一一个不是本项目写的实现对读，与自己的 fixture 构造器互相印证的 reader 遮不住问题）、jinja-diff（`just jinja-diff`，用真实 Jinja2 渲染 `internal/chat` 的模板子集并逐字比对；加它是因为这个门槛之前只靠手工跑，`2605ec3` 把期望表弄红之后整整一版都没人发现）、xcheck-parts（`just xcheck-pre`/`-quant`/`-rope`/`-fixture`，把预分词器、反量化、RoPE 和合成 fixture 各自对着仓库外的实现比一遍；四个之前也只在本地跑过，合在一个 job 里是因为每个都只要一两秒，分拆只多出四份 checkout 与 setup）。

本地跑 `just check` 绿了，CI 的 gate 就应该绿；hooks、docs、cross、test-race、test-darwin 是需要额外工具或平台的独立门槛，各自有对应的 just 配方，动到非 Go 文件就加跑 `just hooks`，动到 Markdown 也加跑 `just docs`。

这条分工是因为 rumdl 和 prek 都不在 Go 工具链里：把 `docs` 或 `hooks` 塞进 `check` 会让只装了 Go 的贡献者无法通过本地门槛。

提交钩子还要 `prek install` 装一次才生效——它写 `.git/hooks/` 下的 shim，仓库里不落任何东西。装之前 `just hooks` 仍可手动跑，装之后每次 commit 自动跑，两条路径执行的是同一份 `.pre-commit-config.yaml`。

**`just hooks` 会改写文件。** 其中的 `rumdl` 与 `rumdl-fmt` 带 `--fix`，`end-of-file-fixer` 补末尾换行，`mixed-line-ending` 统一成 LF：发现问题时它们直接改工作区，然后以非零码退出，并打印 `files were modified by this hook`。这与 `just docs` 相反——后者跑的是 `rumdl check .`，只报告不改文件。所以「`just docs` 红了」之后跑 `just hooks`，文件可能已经被改好，需要重新 `git add` 再看。把改动逐条看一眼再决定要不要，比盲信自动修复安全：`rumdl fmt` 会重排标题空行和表格对齐，不全是纯格式的小事。

提交信息格式同理但机制不同：`commitizen` 钩子只检查**写下的那一条**，而且只在装了钩子的机器上跑；`just commits` 用 `cz check --rev-range` 检查**一整段**，CI 的 commits job 用它兜底。`prek run --all-files` 不会触发 commit-msg 阶段的钩子（它需要一个 message 文件），所以这两条路不能互相替代。

`just fuzz` 同理不并进 `check`：它按时间跑而不是按用例跑，默认 30 秒一个 target，全套要几分钟。但 **seed corpus 由 `just test` 跑**——`go test` 会重放 `testdata/fuzz/` 里的每个文件，所以一个修好的崩溃会一直防住回归，不需要记得跑 fuzz。

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
- 覆盖率基本不依赖**环境变量**：2026-11 测过，设齐四个 `*_MODEL` 后 `internal/engine` 仍是 93.5%，`internal/llama` 从 85.5% 升到 86.0%，`internal/tokenize` 从 96.4% 升到 96.7%（`QWEN2_MODEL` 才动，`MODEL` 不动）。这几个数是写下时的快照：2026-11 末复测已是 95.8%、88.4%、96.7%，结论不变，但绝对值随测试增加而上升，别把它们当基准，其余九个包一个字节都不变。llama 的 0.5 个百分点全部落在 `addBias`、`addBiasBatch`、`finishBatch`、`finish`、`LoadConfig` 这五处，tokenize 的 0.3 落在 `looksByteLevel` 和 `drainPending`。
- 盯覆盖率之前先跑一遍上面这条对照：这个包的数字可能是低在没设 checkpoint 上，也可能只是防御性分支。两边都对不上时，先找有没有生产调用方，再决定要不要补测试。
- 剩下那些设了 checkpoint 仍不着陆的行，多半是**防御性分支**：`engine.finish` 里 `cache.Reset()` 失败那条只在引用计数被破坏时触发，注释自己写着「a pool bug」。不为它伪造内部状态去凑覆盖率——不可达的错误路径正说明它所在的层是对的。
- 判定「不可达」之前先看**函数的参数**，而不是凭调用链推断。2026-11 有两处被判死、后来又测通了：`PagedKVCache.rememberFullBlock` 的 token 切片和 `sequence.remaining()` 的预算都是从参数算的，同包测试直接构造就能触发。真正够不到的是**只从内部状态派生**的条件（`cache.Reset()` 的失败、`paramError.Error()` 这种没有调用方的接口方法），不是「正常路径不会传这种值」。
- 验证一条回归测试有没有判别力，要靠改动被测代码后重跑，不是靠读代码推断。改动时注意**同文本多处出现**：`moeRowsFromBatch` 和 `moeRowsFromDecode` 的构造体逐字相同，用 Python `str.replace` 不加限定会打到第一个，于是测出来「注入不生效」而实际是没打到目标。改完用 `git diff` 确认命中位置。
- **注入之后先 `go build`，再跑测试。** 2026-11 两次给 `validateBenchFlags` 注入 `if false`，都得出「测试没抓住」的结论，实际上那次编辑让 `ignored` 成了未使用变量，**包没有编译过**，`go test` 跑的是上一次的二进制，报的是旧结果。改成能编译的 `if !usedBySuite[f.Name] && false` 后测试立刻变红。一次注入的可信度不超过它的构建结果。
- **把注入写成脚本批量跑时，脚本自己判读输出也要验证。** 2026-11 想一次审完 `generateParams` 的 24 个字段，三轮脚本都是错的：(1) 用 `grep -c "^--- FAIL"` 计数，子测试的 FAIL 行是缩进的，只有子测试失败时数出 0；(2) 把 `timeout` 的退出码当成「零失败」，因为超时的那次跑没有任何输出可数——`maxTokens: 0` 会让一条 pool-refusal 测试永久阻塞（那是注入造出的非法状态，真实请求里 `max_tokens: 0` 让 `maxTokensGiven` 为 false 而走默认值）；(3) 从正则里取类型判空值，取到带空格和逗号的一段，查表失败就静默跳过，报出一个不存在的字段。三条都是手工重跑同一个注入、读输出才发现的。判据：脚本报「某字段零覆盖」时，先手跑一次确认那个注入真的让包变红了。
- **有些改动在测试层面根本不可观测，这时不要写一条「通过但无判别力」的测试。** `server.generate` 里把预算截到窗口那行（`ContextLen - len(promptTokens)`）就是：改成 `ContextLen` 后响应完全不变，实测两种实现都返回 `completion=253`、`finish_reason="length"`，因为 `engine.decode` 自己也在窗口处停——这行改变的是 `usage` 报告的数字，不是模型生成的东西。先写了 `completion_tokens == 253` 的断言，注入后照样通过，于是删掉它，改为在代码注释里写清这行为什么存在、为什么没有测试。留一条为错的理由通过的测试比没有更糟。
- 统计结构体字段的使用率时，`Field:` 和 `x.Field =` 是两种写法，只 grep 前一种会漏掉用赋值写入的地方。2026-11 查 `tinyConfig` 的死字段时，`ropeFreqs` 和 `tied` 都显示 0，实际各有 4 处和 1 处赋值。同理，字段也可能通过位置初始化写入（`T{1, 2, 3}`），那时连名字都不出现。
- **注入执行了、也改变了代码，但改变的量本身不可观测，是第三种「绿」。** 2026-11 给 `RMSNorm` 的八处调用点做注入，两轮下来三处一直是绿的，追下去发现是两个不同的原因，都不是测试弱：
  (1) 把 `eps` 改成 0——`eps` 是 1e-5 而 `mean` 在 1 附近，差三个数量级，改完结果一模一样；
  (2) 把 `layer.ffnNorm` 换成 `layer.attnNorm`——`gguftest` 把所有 norm 权重都填成 1（有意为之，注释说明了随机 norm 会让残差流饱和或衰减），于是两个不同的 norm 在 fixture 里是同一个数。
  判据：一条注入在**断言层面**没有变化时，先问「这两个值在这个输入下会不同吗」，再问「测试有没有看见它」。验证方式是把注入后的中间量打出来看一眼，而不是只看测试的红绿。为此改 `gguftest` 让 norm 接近 1 但各不相同，验证过要付代价：`output_norm[0] = 1.06, want 1` 这类断言和 NumPy 参考数值都会跟着变，三个包变红——所以这是取舍，不是缺陷。
- **注入的分配可能被编译器优化掉，于是「注入没生效」和「测试没抓住」看起来一样。** 2026-11 想验证一条分配次数的断言，往 `MatVec` 里加 `_ = make([]byte, 64)`，测试照常通过，我据此认为断言没有判别力。实际上那行被完全消除了——没有被使用的 `make` 不产生任何效果。换成写入包级变量 `sink = make(...)` 之后，分配数从 23 变成 38，测试立刻变红。判据：注入的东西必须能被观测到才算注入成功，`_ = ...` 不是。
- **覆盖率看不到配置面：一个字段绑死在另一个字段上，行跑到了，但造不出有意义的输入。** 2026-11-28 连续两轮各找到一个，都出在 `internal/gguftest` 这个 fixture builder 上：(1) `Model.HeadDim` 写入了 `<arch>.attention.key_length`，但 `attn_q.weight` 和 `attn_output.weight` 的形状仍硬编码成 `{h, h}`，于是设了 `HeadDim=16` 的文件自相矛盾，加载器拒绝「attn_q.weight has shape [64 64], want [64 128]」——这个字段在整棵树里零处设置，所以从没被发现；(2) `rope.dimension_count` 被无条件写成 `m.headDim()`，而 `llama.Config.RopeDim` 的注释写着「usually but not always equal to HeadDim」，于是「只旋转一部分维度」这条加载器路径永远造不出来。两处的行都在覆盖率里执行过，`headDim()` 还显示 66.7%，那个 66.7% 只说明有个分支没跑，不说明这个字段没人用。判据：**问「builder 能不能表达真实文件的这种配置」，不要只问「这行走过没有」**。具体的两个查法：列出资深字段逐个 grep 测试里有没有设置过；把 builder 写入的元数据 key 与加载器读取的 key 求差集。第二问在这棵树上结果是 5 个 key（`sliding_window_pattern`、`rope.freq_base`、`rope.freq_base_swa`、`rope.scale_linear`、`rope.scaling.type`）builder 写不了——但这 5 个都由 `config_test.go` 和 `gemma2_test.go` 直接构造 kv 列表覆盖了，所以是「builder 表达不了、测试仍覆盖到」的情形，不是缺口。区分这两者要看测试，不能只看差集。
- **「这个测试没有断言」要先查三种写法，AST 只看函数体会漏掉全部三种。** 2026-11-28 写了个脚本扫 `Test*` 函数的函数体，找 `t.Error`/`t.Fatal` 调用，报出五个「零断言」的测试。五个都有断言，分布在三个脚本看不到的地方：(1) **helper 里** —— `TestNonASCIILatinSurvivesDecode` 的整个函数体只有一行 `latin1SurvivesDecode(t, realTokenizer(t))`；(2) **子测试里** —— `TestMatchesNumPyReferenceOnRealWeights` 在 `t.Run` 循环中断言，父函数体里一个都没有；(3) **隐式断言** —— `TestStartPumpIsIdempotent` 没有 `t.Error`，它的断言是「在 `-race` 下跑并发请求而不报 DATA RACE」，注释写明了去掉守卫会产生四份竞态报告；`TestSoftmaxEmpty` 只调用两次 `Softmax`，断言是「不 panic」。所以「无断言」这个结论要么跟进 helper 与子测试，要么改写判据为「这个测试在实现被改坏时会红吗」——后者只能靠注入回答，不能靠静态扫描。
- **用模型输出验证功能之前，先确认那个模型有能力展示它。** 2026-11-28 手工验 `orbix chat` 的多轮记忆，用 gemma-3-270m 问「记住 BANANA」再问「我刚才让你记什么」，它答「我不记得你要我记的词」。读起来像历史没传过去。读代码是相反的：`chat.go:373` 每轮都用 `chatTurns(system, history, user)` 从整个历史重建 prompt，`chatTurns` 的顺序也是 system → history → 新 user，全对。换成 gemma-2-2b 问同样的问题，它答「4271」。所以缺陷在模型不在代码，而 270M 那种规模记不住一个词是它该有的表现。判据：一个「功能没生效」的结论要落笔时，先问这个探针需不需要被测对象**本身**有某种能力；需要的话换一个能展示它的模型重做一次，或者换一个不依赖模型能力的探针（`/history` 打印的轮次、`/stats` 的数字、`engine.Turn` 切片的内容）。同一条也适用于反向：270m 能正确执行 `--system`（答出了 ZORBLAX），说明模板和系统消息链路是通的，而这一条不依赖它记住任何东西。
- **「零生产调用」是问题而不是结论，先查三种正常解释。** 2026-11-28 用 AST 扫出五个没有非测试调用点的导出函数（`kernel.MatMul`、`kernel.Dot`、`f16.FromFloat32`、`gguf.OpenCount`、`gguftest.Path`），五个都正常，没有一个是死代码：(1) **测试对拍基准** —— `MatMul`/`MatVec`/`Dot` 是生产并行版的串行参照，`docs/architecture-review.md` 的审计表已经逐条记过；(2) **注释写明 for tests** —— `OpenCount` 的文档第一句就是「It is exported for tests」；(3) **被测试辅助调用** —— `FromFloat32` 的零调用是**扫描方式造成的**：我的 AST 统计把 `_test.go` 排除在「生产」之外，而 `internal/quant/dequant_test.go` 的 `encodeF16` 正调着它。第三种最容易误判，因为它读起来和死代码一模一样。判据：删任何导出 API 之前，先 `grep -rn "<name>(" --include=*.go`（**不带**路径过滤、**包括**测试文件），三种解释都排除掉才算孤儿。
- **下结论之前先确认搜索范围覆盖了被测对象。** 2026-11-28 同一轮里犯了两次，两次的结论都是「不存在」，两次都错：(1) 在 `for c in ...; do go run ./cmd/orbix $c -h; done 2>/dev/null | grep ...` 里，`2>/dev/null` 把整个循环的输出一起丢掉了，管道收到零行，于是十四个 flag 全部判成「不是 flag」；(2) 用 `grep -rn "func TestX" --include=*_test.go internal/` 说某个测试不存在，而它定义在 `cmd/orbix/flagsdocs_test.go`——`internal/` 这个范围本来就不包含它。两次都不是解析错误，是范围错误，而范围错误的输出和「确实没有」长得一模一样。判据：一条「不存在」的结论落笔前，说得出这条命令看过哪些路径、多少个文件；说不出来就换一条不带 `2>/dev/null`、不带窄目录的重新跑。
- **先怀疑检查工具本身，再怀疑被测代码。** 2026-11 两处：`just skips` 以 `|| true` 结尾，测试失败时仍 exit 0；`cmd | head` 之后 `$?` 读的是 `head` 的状态，于是 `orbix run` 少 prompt 的退出码被读成 0 而实际是 1。两次都差点被当成产品缺陷去修。一个返回 0 的命令不等于它检查的东西通过了，尤其是当它被 `||`、管道或 `2>/dev/null` 包过的时候。
- **性能问题的根因常在调用方，不在被改的那几行。** 2026-11 两次：`Model.Forward` 每 token 分配 23 次，来自 `splitRows` 的闭包逃逸，而注释说的是 scratch buffer；`PagedKVCache.spans()` 每次分配两个切片，把切片挂到 cache 上复用看起来是明显的修法，结果 6 个测试同时报「values cover 3 positions, want 1」——批量路径从 `m.run.Parallel` 里并发调它，共享 scratch 会互相覆盖。两处的约束都不在被改的函数里。改这类代码前先找**所有**调用点，包括并行的那几处。
- **靠调度器竞争才走到的分支，宁可写注释也不要写会闪断的测试。** `engine.emit` 的 select 有三条臂：写进缓冲区、cancel 通道关闭、`abandonGrace` 到点。前两条都有测试，第三条测了两轮都够不到——缓冲区满之后关闭 cancel，只有**下一次** emit 才看得见这个关闭，而那时两秒的 deadline 往往已经赢了 select，选中哪条臂取决于运行时调度。试了 200ms、500ms、1s、3s 四个延迟，1s 那次测试通过但分支仍未覆盖，3s 那次 deadline 先到、测试直接失败。换成「断言耗时 < abandonGrace」也一样，那只是把竞争挪了个位置。最终删掉测试，在代码里写清哪条臂没测、四个延迟各测出什么。**一条时灵时不灵的测试比没有更糟**：覆盖率数字看起来可信，实际是随机的。
- 逐字段合并的测试辅助函数会静默丢字段。`newTestServerConfig` 用一串 `if extra.X != 0 { conf.X = extra.X }` 合并调用方的 `Config`，这种写法是有意的（测试设一个参数不必重述模型名），但列表漏了 `TopP` 和 `ContextLen`——设了这两个字段的测试拿到的是零值，没有任何提示。检查方式是把辅助函数处理的字段与被合并类型的字段列表对照一遍。
- 批量文本替换会改到字符串字面量里的同名标识符。2026-11 把 `tokens` 换成 `t.tokens` 来移动代码，同时把两条错误消息改成了 `"1 scores for 3 t.tokens"`；`just check` 全绿，因为那两条测试只断言数字。移动代码时用带上下文的替换，或改完 `grep` 一遍被替换词在字符串里的出现。
- 断言错误消息时，断言**消息里的数据**，不要断言它周围的模板文字。`strings.Contains(err, "tokens")` 对 `"prompt is too many tokens"` 成立而没有任何尺寸；`strings.Contains(err, "does not implement")` 是模板前缀，与它后面拼进去的原因无关。2026-11 连着修了四处这类：上下文溢出只断言 `"tokens"` 没断言两个数字，413 只断言 `"too large"` 没断言字节上限，形状字段拒绝只断言字段名没断言原因，词表并行数组的两条长度检查则一条都没测。每条都是把断言换成消息里的具体值（数字、上限、原因短语、两个计数）之后才抓得住对应变异。
- 真实模型的交叉验证用独立参考实现（`uv run --with gguf`）解析同一文件，逐张量比对 SHA-256。
- `internal/chat/testdata/*.jinja` 是 checkpoint 里 `tokenizer.chat_template` 的**逐字节副本**，不是源文本。标签外的文本会被渲染，所以多一个换行就进了 prompt。`.gitattributes` 标了 `-text`，`.pre-commit-config.yaml` 把三个空白钩子排除在该路径外，`TestTemplateAssetsAreExact` 用长度与 SHA-256 兜底。新增模板资产时沿用这三处，不要为了让钩子通过而裁剪文件。
- `just test-race` 只覆盖测试里出现的并发。整机验证用 `just MODEL=<真实 gguf> stress`：它编出竞态版二进制，起 `serve`（`--max-active 1` + 共享缓存），再打混合负载——非流式、SSE 流式，以及排队中被客户端放弃的请求。取消路径最容易暴露竞态，因为它要同时回收 KV block 和 goroutine。判据是日志里没有 `DATA RACE`、没有 5xx、且 `orbix_requests_active` 归零。
- 那个脚本要求取消/放弃计数非零，否则以失败退出：小 fixture 生成只要几毫秒，取消窗口容易错过，脚本就会在什么都没测到的情况下报 ok。第一次写它时消息里的模型名取错了字段，所有请求都是 404、从未进入引擎，而脚本仍报告「no request was cancelled」——测试没跑起来和测试通过，输出必须能区分。
- **验证取消路径时，先确认请求真的在跑。** 2026-11-28 想测「中途取消会不会回收资源」，三次尝试三次都没测到取消，每次都看起来像引擎的问题：(1) `timeout 0.08 curl …` 在连接建立前就杀掉，`orbix_requests_submitted_total` 一动不动，请求从未进入引擎；(2) 换成 `max_tokens: 300`，那个池只有 `--blocks 8 --block-size 16`，请求要 20 个 block，直接 400 拒绝，也没进取消流程；(3) 换成能装下的 100 token，可小 fixture 只要 72 ms 一个请求，`timeout 0.3` 到点时它早已生成完，`length` 加一而 `cancel` 仍是 0。真正的读数来自第四次：池扩到 64 个 block、`max_tokens: 800`（实测 1.37 s）、`timeout 0.3`，这才拿到 `cancel=1` 且 `active` 归零。**判据：一次「取消没生效」的结论，先看 `submitted` 有没有加、再看 `finished_by_reason` 里是哪个 reason 加了一**——三个计数器分别对应「没发出去」「被拒了」「跑完了」，它们的静默和缺陷的静默是同一副样子。
- **脚本里的每个时间上限，都是按当时用过的那种文件标定的，换一种就全不成立。** 2026-11 用 gemma-2-2b-it（1.5 GB）跑 `just MODEL=... stress`，三处上限同时不够，每次失败都报成引擎缺陷：(1) 等服务就绪只给 120 s，而 race 版冷读这个文件要 155 s——日志是空的，因为加载完成前什么都不打印；(2) holder 请求给 300 s，它单独跑 12.9 s，但脚本自己排了三个 120-token 客户端在 `--max-active 1` 后面，实测 414.9 s，于是 503 且 duration 正好 300038 ms，读起来像「服务器返回 5xx」；(3) `orbix_requests_active` 只轮询 10 s，而排队那三个是在 holder 结束后才开始被推进的，实测 10 s 时仍是 1、20 s 才归零，于是报「a sequence leaked」。三处都改成按最坏规模留余量，而不是继续用 fixture 的数字。判据：看到脚本报「服务没起来」「返回 5xx」「资源泄漏」时，先用 curl 手工复现同一组请求并读 metrics，再决定是不是引擎的问题——这三条里有两条的真相是 harness 先放弃了。
- **判定「这里没有测试」之前，先搜完所有测试文件，再搜整个仓库。** 上一轮加了一条「warm prefill 与 cold 的结果要相同」的测试，理由是 `prefix_test.go` 里只测了跳过工作。`batch_test.go:1381` 的 `TestLogitsForCachedMatchesAFreshPrefill` 早就在测同一件事，而且更强——用带 attention bias 的 Qwen2，那才是 `logitsForCached` 与前向真正不同的地方，注释里还附了「去掉 bias 后 351/363 个 logit 不同」的实测。我写的版本用默认 llama、没有 bias，测的是更容易的那个。两个证据确认是重复：加上和删掉它，`logitsForCached` 的覆盖率都是 71.4%；同一个「pos 减一」的注入，已有的两条测试都能抓住。**方法是先 `grep -rn \"func Test\" <包>` 把测试名列全，按被测函数名和断言的性质筛，不要只搜一个词就下结论。**

## 文档
- README 是门面和索引，权威正文放在各包 `doc.go`。
- README 里承诺的 CLI 命令和 just 配方必须真的可用；改接口时同步更新 README。
- Markdown 过 `rumdl check`，表格用对齐样式。

### 数字要有人复算，否则会静默过期

文档里的**稳定事实**（有几个 job、几条门禁、几处 skip、长函数清单）必须有测试复算，**快照**（覆盖率、吞吐、内存）不设守卫但要写明它是快照、以及复算命令。

这条不是洁癖：2026-11 一次会话里，同一份 `docs/architecture-review.md` 有三处独立过期——长函数清单（19→20，漏了 `parsePostfix`）、门禁数（11→14，四个变量在表格里而 `grep Getenv` 搜不到）、覆盖率（chat 93.4%→92.2%）。外加 `AGENTS.md` 自己的 job 数（9→10→11）和 `scripts/jinja_diff.py` 的期望表。**每一处都是「没人跑那条命令」，不是「跑错了」。**

守在哪一层有讲究：

- 能跑工具的，就调工具。`TestArchitectureReviewLongFunctionCountsMatchTheCode` 直接 `go run ./scripts/count.go -long`，文档的复算命令与守卫是同一条，不会各说各话。
- 数字和名单都要守。只比数字时，删掉一个名字数字不变；只比名单时，数字改了名字没改。两者都查才抓得住。
- 同一事实写在两份文档里，两份都要守。`docs/models.md` 的门禁表一直有守卫，`architecture-review.md` 那句话当时没有，于是后者烂了三次；现在两边都有（`TestArchitectureReviewStatesTheSameGateCount`、`TestArchitectureReviewStatesTheSkipCounts`），补守卫时顺手把「这份文档里的数字没有守卫」那句也改成了事实。
- 守卫只覆盖能机械判定的部分。哪些函数该拆、哪个 job 验证什么，是判断，写进注释让人读，别假装能测。
- 守卫要在文档被**改写**时报「这条测试读不出那个数了」，而不只是报数字不符——否则下次重构文档时，守卫自己先失效。

### 守卫自己会失效，而且失效时不出声

上面那条讲的是数字过期。更麻烦的是守卫过期：它还在跑、还是绿的，但它已经不看原来那件事了。2026-11-28 一轮会话里出现六次，每次都是靠**故意把文档改错再看守卫红不红**发现的：

- **前提变了就自己跳过。** `TestDocsExplainTheirVersionMarkers` 原来在「仓库没有 tag」时检查文档里的版本号是计划，写完直接 `if len(tags) > 0 { t.Skip(...) }`。这个仓打了三个 tag，于是它从第一个 tag 起就不再检查，而两份文档一直写着「no release has been tagged yet」。跳过信息里甚至写着「the markers are now history and could be checked against the tag list」——它知道自己该改成什么，但选择了不做。**前提变化时应该改成新的检查，不是关掉旧的。**
- **只守了行，没守合计。** 尺寸表逐行比对代码，`**合计**` 行因为不带包名被 `sizeTableRows` 跳过。改一行之后合计停在上一次的读数上，而所有行都是对的。`TestArchitectureReviewSizeTableTotalsAreSummed` 补上之后立刻又红了一次——那次是补它的这个测试让 `internal/arch` 自己的行变长。
- **只守名字，不守参数。** `TestEveryRecipeIsDocumented` 只查配方是否被提到，`justMention` 只抓第一个词，把参数写成大写 `FILE`/`PKG` 也算「提到了 mutate」，而实际参数是位置参数 `file`/`pkg`，照抄得到 `no such file`。
- **守卫里写了恒真的条件。** 上一条的守卫第一版用 `strings.CutSuffix(arg, "")` 判断参数是不是 `NAME=value`，而它永远返回 `(arg, true)`，于是所有比较都被 `continue` 跳过——守卫对刚写下的错误文本报绿。
- **测量方法挡住了被测对象。** `go test -cover ./pkg` 只 instrument 自己，被别的包调用的函数显示 0%。`llama` 的 `DefaultBlockSize` 就是如此，而「`llama` 已无 0% 函数」这句话正是用这个测量得出的。`just coverage` 用 `-coverpkg=./...` 给出另一个数。
- **数字从没被任何一个工具算出来。** `AGENTS.md` 说 xcheck-quant 对照 17770 个值，脚本只打印逐格式的行和一个不含总数的汇总，实测是 14750。那个数是写的人自己加的。

判据：**每加一条守卫，立刻把文档改成它该抓的错误值，确认它真的红。** 这跟代码注入是同一条规则，但守卫有个额外陷阱——它对错误值绿有两种原因（没判别力，或那条分支根本没走到），而代码注入至少还能靠 `git diff` 确认打到了目标。这一步花不到一分钟，上面六次里没有一次是靠读代码发现的。

还要给守卫留退路说明：**一条对着错误值仍然绿的守卫，必须写清为什么**（分支不可达、等价变异、或已在别处覆盖）。`internal/engine` 的 `emit` 就是例子——`just mutate` 让 12 个布尔返回里的 5 个存活，逐个读过之后都在 cancel/deadline 臂上，而那段代码的注释早就写明「The cancel arm of that select is the one branch here without a test」并记了四个延迟各测出什么。存活的位点不等于缺陷，但**没读过的存活位点**一定是没查过的角落。

设计守则时同时想清楚它**不该**覆盖什么。覆盖率快照就不该有守卫：它每次都变，守卫会在无关提交上变红，然后被习惯性跳过——那比没有守卫更糟。

## 依赖

外部依赖可用，逐个判断，不是禁令。`go.mod` 目前没有 require 行，这是结论不是前提：每一处被考虑过的依赖都记录在下面的表里，包括没被采用的。

判断标准：**这个库替我们承担的复杂度，是否大于它自身的复杂度**。四条参照：

1. 标准库已经覆盖的，用标准库。`flag`、`encoding/json`、`net/http`、`crypto/rand` 覆盖了 CLI 参数、GGUF 中的 JSON、HTTP 服务端和追踪 id。
2. 库的能力要有实际去处。viper 的价值在多格式配置文件和配置来源合并；本项目只有 `ORBIX_MODEL` 一个环境变量、没有配置文件，引入它等于让库承担它无从发挥的工作。
3. 规模匹配。**8** 个命令（`run`/`bench`/`chat`/`serve`/`healthcheck`/`model`/`version`/`help`）、**4031** 行非测试代码的 CLI 不需要命令树框架；`serve` 的 13 个 flag 是单命令最多的一处。命令数或单命令参数数显著增长时重新评估。这一句原来写「6 个命令、约 3400 行」，与同句末尾的 3607 行自相矛盾；命令数漏了 `healthcheck` 和 `model`，而 `model` 还有 `inspect` 子命令。（2026-11-26 复算：`orbix serve -h | grep -cE '^  -[a-z]'` 是 13，未变。行数从记录下的 3607 涨到 3951 —— `ls cmd/orbix/*.go | grep -v _test` 有 **14** 个文件，`model_files.go` 是后来加的；`cgo_enabled.go` 在 `CGO_ENABLED=0` 下不编译，排除它之后是 13 个文件 3951 行，全算上是 3957 行。（2026-11-28 复算：`inspect` 的超参数表补了若干 key，行数与上一版记录的 3907/3913 相比各增 23；同日再把 `--workers` 的帮助文本从一段内部测量记录换成它实际保证的行为，又增 5 行；下一轮发现那段替换文本点名了 `--max-active`，而 `run` 与 `chat` 都没有这个 flag，改成不点名任何 flag 后再增 6 行。这两个数都没有守卫，靠人复算，所以复算日期写在这里。2026-11-28 又加了 9 行：`healthcheck` 的 `--addr` 拒绝不带 scheme 的值时给出可操作的消息，13 个文件口径变成 3960。同轮再加 `bench --repeat` 的 59 行，变成 4019。再把 `run` 与 `chat` 各自手写的三个采样 flag 与四段校验收进 `registerSamplingFlags`，变成 4031。前面的正文数字里有守卫，这一句里的没有——`TestAgentDocCLIScaleMatchesTheTree` 只查正文那个。）
4. 维护状态看**默认分支的最新提交**，不看 `pushed_at`（它会被无关推送刷新）。2026-11 核对：

   | 库 | 最新提交 | 判断 |
   | --- | --- | --- |
   | `urfave/cli` | 2026-11（v3.14.0） | 活跃 |
   | `kong` | 2026-11 | 活跃 |
   | `spf13/cobra` | 2026-07 | 四个月无提交，放缓 |
   | `spf13/viper` | 2025-10 | 十三个月无提交，停滞 |

   cobra 与 viper 都是成熟稳定的库，低提交量本身不等于不可用；但「最近在动」要求这两者时，urfave/cli 和 kong 是更实际的选择。

| 依赖 | 状态 | 理由 |
| ---- | ---- | ---- |
| （无） | — | 见上 |
| `viper` | 不采用 | 只有 1 个环境变量、0 个配置文件，没有多来源合并的需求 |
| `cobra` / `urfave/cli` / `kong` | 不采用（当前规模） | 8 个命令用 `flag` 加 `cmd/orbix/flags.go` 的共用注册已经够；命令数或单命令参数数增长后重新评估 |
| OpenTelemetry SDK | 不采用 | 只用到一个 protobuf 消息的 JSON 编码，`internal/otel` 自行实现且已被真实 collector 验证 |
| `log/slog` | **采用（标准库）** | 服务端访问日志。要的只是分级结构化记录和可换 handler，`slog` 就是；配置见 `serve --log-format text\|json\|off` |
| `urfave/cli` + `Koanf` | 不采用 | Koanf 的价值在多来源配置合并，本项目 0 个配置文件、1 个环境变量；CLI 规模见上 |
| `Kong` + `cleanenv` | 不采用 | cleanenv 默认分支最近提交 2025-09，已 14 个月未动 |
| `go-playground/validator` | 不采用 | flag 的校验是十几行显式判断，引入反射驱动的校验器换不到等价的可读性 |

新增依赖时：在 `go.mod` 之外同步本表，写明替我们承担的复杂度；**NEVER** 为了少写几十行样板而引入一个框架。

## 下载真实 checkpoint

`dl` 是 `aria2c -c -x 16 -s 16 -k 1M` 的封装。**不要写「它比 curl 快 N 倍」这类结论**：同一个 18.2 MiB 文件在本机测了四轮，`dl` 两次分别是约 12 s 和 257 s，`curl` 两次是 32.8 s 和 133 s，同一时段 `curl` 甚至更快。链路抖动比工具差异大得多，任何单次对比都不能支撑速度结论。

可复现的是这两条：

- **`-c` 断点续传。** `timeout 6 dl` 中断后留 `.aria2` 控制文件，再跑一次接着下完，最终 19,077,344 字节与完整文件一致。`curl` 不带 `-C -` 时中断即全废，1820 万字节重头开始——下载中断时这是实质差别。
- **多连接能吃到不稳定链路。** `dl` 的 `CN:` 会从 16 掉到 2–7（HF 的 CDN 限制并发），但掉到 0 之后还能恢复；curl 单连接遇到 `CONNECT tunnel failed, response 503` 就直接失败，什么都没下到。

所以用 `dl` 的理由是**中断可续、失败可重试**，不是速度。需要速度结论时先跑同条件下的对照，别引用历史数字。

两个坑：

- **`-o` 的路径是相对 cwd 的。** `dl -o /tmp/x.gguf URL` 落在 `$PWD/tmp/x.gguf`，不是 `/tmp/x.gguf`——实测确认。用 `dl -d /tmp -o x.gguf`，或先 `cd` 到目标目录。
- **`-q` 会把进度条一起吞掉**，看不到卡住还是在下。调试时去掉 `-q`，让它打印 `CN:`/`DL:` 那一行。

HF 的 `models/` 路径会 302 到 `models-moved/` 再到 CDN，`dl` 的重定向日志很长但不代表出错；判断是否成功看最后的 `STATUS:OK` 和文件字节数。

## 判断一条测量有没有意义

同一轮里出现过三次「数字有了，结论错了」，都是**基线选错**，不是算术错：

- 量化 matvec 的 1.68x 加速，基线把「解码」算进了每次调用；真实设计只在加载时解码一次，换成按次成本后是**慢 4.14x**，方向相反。
- `--block-size` 对复用的测量，所有请求打同一个 server，计数器把前面 prompt 的复用带了进来；每行起独立 server 后结论反了。
- `-cover` 只看某个包，而该包的函数被别的包的测试调用时仍显示 0%。

写结论前先问：这个数字是**被比较的两边各自在什么条件下跑出来的**。条件不同就先改成同条件。测量工具本身的输出也要看——`golangci-lint` 因为 PATH 里是 v1 而失败，看起来像代码问题，实际是环境。

**结论对，理由不一定对，而改措辞验不出后者。** 2026-11 给 `cmdServe` 的 `select` 写「为什么这两个 case 只有一个进得去」，连续四轮都在改理由：

1. 「唯一入口是进程信号」——读者看得到另一个 case，立刻会问。
2. 「`Shutdown` 只有信号分支调用」——两处都调，事实就错了。
3. 「两处都调，但顺序不同」——顺序根本不是关键。
4. 写了个独立程序实测：从外部关掉 listener 后 `Serve` 返回的是 `use of closed network connection` 而不是 `ErrServerClosed`，于是那条路会走到 `return err` 而不是 `return nil`。真正的理由是**listener 在本函数内创建、从不外传**，外面关不掉它。

四轮的结论一直是「信号是唯一入口」，从没变过；变的是理由，而每一轮我只核对了结论是否还成立。**判据：当你要为一个「为什么不可达」的解释改第三版措辞时，停下写个最小程序去测，不要继续换说法。** 那次实验连一分钟都不到，推翻了三版论证。

同一条规则适用于核对**第三方**的事实。`uv run --offline --with <pkg>` 用的是本地缓存那一版，它可能落后上游几个月，所以它缺一个名字并不证明上游没有。2026-11 查 `docs/models.md` 里「gguf-py 的 arch 枚举只剩 `mistral3`、`mistral4`、`mistral-large4`」这句：本地 gguf 0.19.0 的 `MODEL_ARCH_NAMES` 只有前两个，看起来是文档多写了一个；而 `raw.githubusercontent.com` 上的 `constants.py` 三个都在，且那个文件的最后一次提交就是三天前。文档是对的，缓存是旧的。

要否定一句关于上游的话，读上游的源码，不要读本地装的包。反过来，要跑代码验证某个行为，本地那份才是要用的——两者回答的是不同问题。

**调参常量的注释要能被一条 benchmark 复现，而不只是记录一次当时的测量。** 2026-11 连着两轮各找到一处：
`kernel.minParallelWork` 的注释说「27k multiply-adds 时打平、8k 时慢 1.4x」，而常量是 32768，benchmark 的最小规模是 82944——**它声称的证据覆盖不到自己**；`kernel.workPerWorker` 说「41k 时两个 worker 赢 1.18x、十六个会输」，仓库里没有任何 benchmark 变动过 worker 数。

两处都补了 benchmark，实测都对上了（1.18x 分毫不差），但补的过程中发现更值得记的事：`parts()` 的守卫让 41k 永远只分 2 份，所以「十六个会输」在新代码里**无法复现**——它是这个常量存在的理由，不是它的当前行为。benchmark 名字里写 `workers=16` 而实际跑 2 份，读的人会以为那是 fan-out 对比。

判据：注释里出现「faster/slower/Nx」这类比较时，问一句「跑哪条命令能看到这个数字」。答不上来就补 benchmark，补完再看它测的是不是你以为的那件事。

**补了 benchmark 还不够，样本量要够。** 2026-11 用 `-benchtime=50x` 跑一个新写的 benchmark，从 34452 ns 和 29127 ns 算出「两个 worker 快 1.18x」，写进了注释。换成 `-benchtime=2s` 连跑三次，两个 worker 反而慢 20–40%，方向相反。

`-benchtime=1x` 不可用的证据更直接：同一个 serial 用例连跑三次得到 76833、75611、435647 ns——**6 倍方差**。单次测量测的是冷启动和调度偶然，不是稳态。`just bench` 默认 `-benchtime=1s` 是对的；为求快改成 `20x`、`50x` 时，拿到的是能算出一个数、但不足以支撑结论的东西。

2026-11-28 又碰到一次，同一个对比在两种参数下差出 19%：`just bench` 的默认值（`-benchtime=1s -count=1`）报 sequential 1741.96 ms、batched 342.38 ms，**5.09x**；换成 `docs/performance.md` 记录的 `-benchtime=3s -count=3` 取中位数，得到 1922.26 与 280.98，**6.84x**。文档里写的是 6.30x，与后者只差 8.6%，与前者差 19%。**默认参数给出的那个数才是错的**——一次测量恰好落在 batched 偏慢、sequential 偏快的那一侧。所以「默认值是对的」只表示它不会像 `1x` 那样荒谬，不表示它够用来反驳文档里的数字。要核对一个已记录的值，用记录它的那条命令。

同一台机器上比较两个实现时，两次都要用同一个足够长的 benchtime，并重复至少三次看方向是否稳定。跨机器比较更危险：`kernel.workPerWorker` 的 1.18x 测于 EPYC 7742，在多核共享缓存方式不同的机器上会反向。写下这类数字时标明机器，否则读的人会以为它是这个 shape 的属性。

**跑 benchmark 之前先看机器闲不闲。** 2026-11 在一台 128 核机器上反复测 `Runner` 的并行路径，三次结果都对不上：某个尺寸一次是 parallel 快 22%，另一次慢 29%，再测又变成持平。查下去发现 `uptime` 是 131.87、CPU 空闲 0.1%——机器满载，任何涉及多个 goroutine 的比较测的都是抢不抢得到核，不是 fan-out 本身。

这一条推翻的结论不止一个：此前两轮里写的「两个 worker 快 1.18x」「两个 worker 慢 20–40%」都出自这台机器，方向互相矛盾，而现在知道两次都不可信。真正能保留的只有不依赖计时的那部分（`parts()` 返回几）。

判据：并行相关的结果出现方向反复时，先 `uptime` 和 `nproc`，再看代码。单线程 benchmark 在满载机器上相对可用——它测的是同一个核的吞吐；多线程的结果则只在空闲时才有意义。

**重构前先量一次它要改善的那个指标，而不是先相信结构更干净。** 2026-11-28 的会话里，`internal/server` 的流式路径有两层闸门（`stopGate` 与 `toolCallGate`）各自扣住文本、各自记账，读起来确实乱。于是把一个按字节区间记账的 `tokenLedger` 写了出来，替掉两套 `held`。

它构建通过、既有测试全绿、注释也写得更清楚——然后拿同一组差分矩阵（6 个回复 × 7 个 stop 列表 × 2 种结束原因）去比：**旧代码 6 处不一致，新代码 10 处**。新结构没有解决旧问题，还在「stop 命中与工具闸门途中」这个组合上多弄丢了几条 logprob 条目。如果只看「测试绿了、代码更对称」就提交，这是一次净退步。

判据：**先跑一遍能数出不一致数量的差分探测，再动结构**。数量是 6 还是 10 是可比较的，而「更清晰」不可比较。同一次会话里同一个方法抓到了两个真实缺陷（`x y z` 丢尾部、标记对不成调用时闸门与解析器判断不一），也是它证明了这个重构不划算。

配套的两条：

- **差分探测的判据要写成等式，不要写成印象。** 「流式日志条目拼起来必须等于客户端收到的文本」能直接数出不一致；「流式看起来没问题」什么都数不出来。
- **被数据否掉的方向要连注释一起撤。** 那次重构留下的 `proseSpan` 说明、`tokenLedger` 的类型注释都写得挺像回事，留着会让下一个人以为它在用。`git checkout` 整个文件比逐段删安全。

## 临时文件与后台进程

- **不要用 `pkill -f` 清服务器进程。** 它匹配的是完整命令行，而发起清理的那个 shell 自己的命令行里就含要匹配的字符串——`bin/orbix serve` 也好、`bin/orbix ser[v]e` 也好，只要 bash 的 `-c` 参数里出现过这些字面量，`pkill` 就会把自己一起杀掉，后面的命令一条都不执行。带方括号的写法不解决问题：那是给正则留的转义，而 shell 命令行里的字面量是 `serve`，匹配依然成立。这条在本会话里踩了十几次。

  按监听端口找 pid，不按名字：

  ```bash
  ss -ltnp 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u |
  while read p; do
      case "$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)" in
          *orbix*) kill "$p" ;;
      esac
  done
  ```

  或者起服务时把 pid 记下来（`nohup ... & echo $! > /tmp/orbix.pid`），清理时 `kill "$(cat /tmp/orbix.pid)"`。
- 起 server 用 `nohup ... &` 加 `sleep` 加 `/healthz` 轮询，不要用固定长 sleep。端口用 `--addr 127.0.0.1:19xxx` 手工编号，`--blocks`/`--block-size` 按测试意图显式给。
- 注入验证（改代码看测试是否失败）之后**立刻复原**，并在同一条命令里跑 `git status --short` 确认工作区干净。用 `cp file /tmp/file.bak` 保存原件，不要靠 `git stash`。

## 注入验证要确认打到了目标

验证一条测试有没有判别力，靠改动被测代码后重跑，不靠读代码推断。改的时候注意**同一段文本可能出现多处**：`str.replace` 不加限定会命中第一个，于是测出「注入不生效」，实际是没打到目标。改完先 `git diff` 看命中位置再跑测试。

反过来的情况也发生过：测试和被测代码引**同一个常量**，把常量改成错误值两边一起变，测试照常通过。断言要写字面值（`want 2048`），不写 `want defaultBlockContext`。

## 验证 CI 之前，先在一个干净 checkout 里跑

`.cache/`、`bin/`、编译产物这些东西在开发机上早已存在，所以「本机能跑」证明不了 CI 能跑。2026-11 给 `just stress` 加了 CI job，本地验证通过；但那台机器的 `.cache/` 从会话开始前就在，而全新 clone 里没有它，`go build -o .cache/orbix-race` 会直接失败。

查法很短：

```bash
git clone . /tmp/ci && cd /tmp/ci && <the recipe>
```

`build` 和 `fixture` 本来就有 `mkdir -p`，`stress` 漏了——不是因为谁知道要做，而是因为写它的时候目录恰好存在。**同一个仓库里同类配方要一起看**：只检查自己正在改的那个，会漏掉它和邻居的差别。

2026-11-28 把 ci.yml 的十三个 job 逐个核对了一遍，十一个通过、两个这台机器验不了。在全新 `git clone` 里跑的是九个：`check`、`cross`、`isolate`、`docs`、`race`（`just test-race`）、`stress`、`xcheck`、`jinja-diff`、`xcheck-parts`（后者的四个配方 `xcheck-pre`/`-quant`/`-rope`/`-fixture` 各跑一次）。其中 `stress` 用的是 `just fixture` 造出的 `.cache/stress.gguf`，输出 `no races, no 5xx, active returned to 0, 3 cancelled`——取消计数非零，所以它确实测到了取消路径，不是「什么都没测到就说 ok」；`xcheck` 比对了 39 个张量。另两个在主仓跑：`prek`（`just hooks`，未改写任何文件）与 `commits`（`just commits`）。验不了的两个是 `docker`（这台机器没装 docker）和 `test-darwin`（需要 macOS）。

这条记录的用途是「上一次真跑过是什么时候、当时绿不绿」，不是「它们永远不会坏」——没有守卫，换机器就要重跑。写之前算错过一次：先写了「十二个全部通过」，而 13 减去两个验不了是 11，多出来的那个是我把 `prek` 和 `commits` 也算进了「在干净 clone 里跑」的那一组。
