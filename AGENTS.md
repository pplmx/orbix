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

上面是门槛和发布入口。另外九个是辅助配方，不进门槛，但会用得上：

```bash
just fmt     # gofmt -w，修 check 报出来的格式问题
just vet     # go vet，check 的子集；改 PKG 收紧范围：just PKG=./internal/gguf vet
just clean   # 删 bin/、.rumdl_cache、.cache 和 *.test/*.out/*.prof；不碰 go build 的全局缓存
just isolate # 测试顺序依赖与状态泄漏：-count=2 -shuffle=on；改测试辅助或包级变量后跑
just stress  # 竞态版二进制跑混合负载（非流式/SSE/中途取消）；要 MODEL=<真实 gguf>，验 active 归零
just xcheck  # 用 Python 的 gguf 包独立解析同一文件，逐张量比对类型、维度和 SHA-256；要 MODEL= 和 just build
just skips   # 数一下这棵树跳过多少个测试，CI 每轮打印的就是这个数
just fixture # 生成一个够大的合成 checkpoint，让 `just stress` 有东西可打断（默认落 .cache/）
just         # 不带配方等价于 just check
```

配置以单一事实源维护：Go 用 `go.mod`，lint 用 `.golangci.yml`，Markdown 用 `.rumdl.toml`，提交钩子用 `.pre-commit-config.yaml`，行尾与二进制标记用 `.gitattributes`，编辑器约定用 `.editorconfig`，命令入口用 `justfile`，版本与 changelog 用 `.cz.toml`，CI 用 `.github/workflows/ci.yml` 和 `.github/workflows/release.yml`。

CI 的 Go 相关 job 调用 just 配方，不另写 `go build` 命令；两处例外各有原因：docs job 用 rumdl 官方 action（它是 linter 本身，不是安装器），test-darwin job 直接跑 `go test ./... -count=1`，因为 `just test` 固定 `CGO_ENABLED=0` 而对 macOS 原生测试没有额外价值，后者也不需要 `justfile` 里的其他变量。八个 job 各自验证一件事——gate（格式/vet/测试/lint）、cross-build（5 个平台的 CGO_ENABLED=0 构建；编译后读回产物里的 `build\tCGO_ENABLED=0` 标记，所以「没带上 cgo」也被验了——不能运行的目标也能这样查，标记在文件里）、test-darwin（唯一非 Linux 的可执行测试环境）、test-race（需要 cgo，`just test-race` 单独覆盖）、stress（`just fixture` 造出够大的合成 checkpoint，再用竞态版二进制跑混合负载；四条判据是 DATA RACE、5xx、active 未归零、以及「什么都没被取消」——最后一条防止在什么都没测到时报 ok）、prek（`just hooks`，补 `check` 看不见的非 Go 文件格式问题）、commits（`just commits`，按提交范围校验信息格式）、docs（`just docs`）。

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
- **靠调度器竞争才走到的分支，宁可写注释也不要写会闪断的测试。** `engine.emit` 的 select 有三条臂：写进缓冲区、cancel 通道关闭、`abandonGrace` 到点。前两条都有测试，第三条测了两轮都够不到——缓冲区满之后关闭 cancel，只有**下一次** emit 才看得见这个关闭，而那时两秒的 deadline 往往已经赢了 select，选中哪条臂取决于运行时调度。试了 200ms、500ms、1s、3s 四个延迟，1s 那次测试通过但分支仍未覆盖，3s 那次 deadline 先到、测试直接失败。换成「断言耗时 < abandonGrace」也一样，那只是把竞争挪了个位置。最终删掉测试，在代码里写清哪条臂没测、四个延迟各测出什么。**一条时灵时不灵的测试比没有更糟**：覆盖率数字看起来可信，实际是随机的。
- 逐字段合并的测试辅助函数会静默丢字段。`newTestServerConfig` 用一串 `if extra.X != 0 { conf.X = extra.X }` 合并调用方的 `Config`，这种写法是有意的（测试设一个参数不必重述模型名），但列表漏了 `TopP` 和 `ContextLen`——设了这两个字段的测试拿到的是零值，没有任何提示。检查方式是把辅助函数处理的字段与被合并类型的字段列表对照一遍。
- 批量文本替换会改到字符串字面量里的同名标识符。2026-11 把 `tokens` 换成 `t.tokens` 来移动代码，同时把两条错误消息改成了 `"1 scores for 3 t.tokens"`；`just check` 全绿，因为那两条测试只断言数字。移动代码时用带上下文的替换，或改完 `grep` 一遍被替换词在字符串里的出现。
- 断言错误消息时，断言**消息里的数据**，不要断言它周围的模板文字。`strings.Contains(err, "tokens")` 对 `"prompt is too many tokens"` 成立而没有任何尺寸；`strings.Contains(err, "does not implement")` 是模板前缀，与它后面拼进去的原因无关。2026-11 连着修了四处这类：上下文溢出只断言 `"tokens"` 没断言两个数字，413 只断言 `"too large"` 没断言字节上限，形状字段拒绝只断言字段名没断言原因，词表并行数组的两条长度检查则一条都没测。每条都是把断言换成消息里的具体值（数字、上限、原因短语、两个计数）之后才抓得住对应变异。
- 真实模型的交叉验证用独立参考实现（`uv run --with gguf`）解析同一文件，逐张量比对 SHA-256。
- `internal/chat/testdata/*.jinja` 是 checkpoint 里 `tokenizer.chat_template` 的**逐字节副本**，不是源文本。标签外的文本会被渲染，所以多一个换行就进了 prompt。`.gitattributes` 标了 `-text`，`.pre-commit-config.yaml` 把三个空白钩子排除在该路径外，`TestTemplateAssetsAreExact` 用长度与 SHA-256 兜底。新增模板资产时沿用这三处，不要为了让钩子通过而裁剪文件。
- `just test-race` 只覆盖测试里出现的并发。整机验证用 `just MODEL=<真实 gguf> stress`：它编出竞态版二进制，起 `serve`（`--max-active 1` + 共享缓存），再打混合负载——非流式、SSE 流式，以及排队中被客户端放弃的请求。取消路径最容易暴露竞态，因为它要同时回收 KV block 和 goroutine。判据是日志里没有 `DATA RACE`、没有 5xx、且 `orbix_requests_active` 归零。
- 那个脚本要求取消/放弃计数非零，否则以失败退出：小 fixture 生成只要几毫秒，取消窗口容易错过，脚本就会在什么都没测到的情况下报 ok。第一次写它时消息里的模型名取错了字段，所有请求都是 404、从未进入引擎，而脚本仍报告「no request was cancelled」——测试没跑起来和测试通过，输出必须能区分。
- **判定「这里没有测试」之前，先搜完所有测试文件，再搜整个仓库。** 上一轮加了一条「warm prefill 与 cold 的结果要相同」的测试，理由是 `prefix_test.go` 里只测了跳过工作。`batch_test.go:1381` 的 `TestLogitsForCachedMatchesAFreshPrefill` 早就在测同一件事，而且更强——用带 attention bias 的 Qwen2，那才是 `logitsForCached` 与前向真正不同的地方，注释里还附了「去掉 bias 后 351/363 个 logit 不同」的实测。我写的版本用默认 llama、没有 bias，测的是更容易的那个。两个证据确认是重复：加上和删掉它，`logitsForCached` 的覆盖率都是 71.4%；同一个「pos 减一」的注入，已有的两条测试都能抓住。**方法是先 `grep -rn \"func Test\" <包>` 把测试名列全，按被测函数名和断言的性质筛，不要只搜一个词就下结论。**

## 文档
- README 是门面和索引，权威正文放在各包 `doc.go`。
- README 里承诺的 CLI 命令和 just 配方必须真的可用；改接口时同步更新 README。
- Markdown 过 `rumdl check`，表格用对齐样式。

## 依赖

外部依赖可用，逐个判断，不是禁令。`go.mod` 目前没有 require 行，这是结论不是前提：每一处被考虑过的依赖都记录在下面的表里，包括没被采用的。

判断标准：**这个库替我们承担的复杂度，是否大于它自身的复杂度**。四条参照：

1. 标准库已经覆盖的，用标准库。`flag`、`encoding/json`、`net/http`、`crypto/rand` 覆盖了 CLI 参数、GGUF 中的 JSON、HTTP 服务端和追踪 id。
2. 库的能力要有实际去处。viper 的价值在多格式配置文件和配置来源合并；本项目只有 `ORBIX_MODEL` 一个环境变量、没有配置文件，引入它等于让库承担它无从发挥的工作。
3. 规模匹配。6 个命令、约 3400 行非测试代码的 CLI 不需要命令树框架；`serve` 的 13 个 flag 是单命令最多的一处。命令数或单命令参数数显著增长时重新评估。（2026-11 复核：`orbix serve -h | grep -cE '^  -[a-z]'` 是 13，未变。行数在同一时期从 3429 涨到 3607 —— `ls cmd/orbix/*.go | grep -v _test` 有 13 个文件，但 `cgo_enabled.go` 在 `CGO_ENABLED=0` 下不编译，所以按实际构建算是 12 个文件 3607 行；把两个 build-tag 文件都算上是 3613 行，只算 `cgo_disabled.go` 那个则是 3601 行。2026-11 末复算。）
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
| `cobra` / `urfave/cli` / `kong` | 不采用（当前规模） | 6 个命令用 `flag` 加 `cmd/orbix/flags.go` 的共用注册已经够；命令数或单命令参数数增长后重新评估 |
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
