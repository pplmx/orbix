# 真实使用验证报告

> **状态（2026-11-25 更新）**：本报告的 21 条已逐条处理：13 条已改（含 2 条随 1/7 一并处理），1 条原本准确，7 条说明了不改的理由，见
> [文末的处理记录](#六处理记录2026-11-25)。报告正文保留原样，作为当时的实测证据。

按真实用户的路径走了一遍：安装 → 文档 → 快速开始 → 每个命令 → HTTP API → 部署场景。
下面按「确认的问题」「文档与代码不一致」「体验优化点」三类列出，每条都附实测证据。

验证环境：2026-11-25，linux/amd64，128 核（EPYC 7742），go1.27.4，
checkpoint 用了两个：`Qwen2.5-0.5B-Instruct`（F32，1.84 GiB）和 `stories15M-q4_0.gguf`（19 MB）。

---

## 一、确认的缺陷

### 1. 服务端能表达零 token 请求，但 CLI 和文档都说不能（事实错误 + 功能缺口）

`docs/configuration.md:153-156` 写着：

> `--max-tokens` accepts 0 only on `run`... The server cannot express a zero-token request
> (`max_tokens` is an integer, so 0 is indistinguishable from absent)

`cmd/orbix/serve.go:678` 的报错信息也重复了这个理由。但实测恰恰相反：

```console
$ curl -sS .../v1/chat/completions -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":0}'
{"choices":[{"finish_reason":"length","index":0,"message":{"content":"","role":"assistant"},...}],
 "usage":{"completion_tokens":0,"prompt_tokens":33,"total_tokens":33}}
```

`max_tokens: 0` 返回空 completion + `finish_reason: "length"`，与「字段缺失」完全可区分。
代码本身是对的：`internal/server/server.go:1369` 的 `maxTokensGiven` 就是为此而设，
`server.go:3110` 的注释明确写着「Absent, not zero」。真正的理由在代码注释里也有：
`serve --max-tokens 0` 设的是**服务器的默认值**（给不带 `max_tokens` 的请求用），
而 CLI 无法表达这个默认值本身（`server.Config` 会把 `<= 0` 换成 256）。

**影响**：想用「客户端不带 `max_tokens` 时默认生成 0 个 token」这条配置的用户被拒绝了，
而给出的理由是错的（会让人以为协议层面做不到）。`docs/configuration.md` 和
`serve.go:678-684` 的措辞需要改成真实原因。

### 2. `docs/usage.md` 自相矛盾：`suffix` 到底实没实现

同一篇文档三处说法互相冲突：

- 第 107-109 行：`suffix` 已经实现，「each leaving the list when it did」
- 第 168-211 行：整节讲 `suffix` 的 FIM 语义、marker 探测、stop 集合，是已实现的功能
- 第 276-277 行：**「`suffix` is the one field still refused on completions」**

实测 `suffix` 在 completions 上是实现的（缺 FIM marker 的模型给 400 并说明原因）：

```console
$ curl .../v1/completions -d '{"prompt":"def add(a,b):\n    return","suffix":"\ndef mul(a,b):","max_tokens":8}'
{"choices":[{"text":"\ndef sub(a,b):",...}]}
```

代码印证：`internal/server/server.go:1148-1174` 只在模型没有 FIM marker 时才拒绝，
拒绝的是「这个 checkpoint」而不是「这个字段」。

**影响**：读者读到第 276 行会以为该功能没做，而它其实做了 100 多行文档。第 276 行应删除或改写。

### 3. 文档声称的 skip 数量与实测不符（25 vs 31）

`docs/models.md:313` 的表格：

| Gates set    | Skipped |
| ------------ | ------- |
| none         | 25      |
| all fourteen | 0       |

实测（文档自己给的命令 `go test ./... -v | grep -c '^--- SKIP'`，本仓全新 clone，无任何环境变量）：

```console
$ go test ./... -count=1 -v 2>&1 | grep -ac '^--- SKIP'
31
$ just skips
31
```

`docs/models.md:345` 另有一处「24 skipped with no variables」。三处数字各不相同（25 / 24 / 实测 31）。

同一处还写「There are fourteen gates: nine named directly at the call site」，而实测
`grep -rhoE 'Getenv\("...'` 在测试里找到 10 个**模型**门禁的直接调用：

```console
CODER_MODEL GEMMA2_MODEL GEMMA3_MODEL GEMMA_MODEL MISTRAL_MODEL
MIXTRAL_MODEL MODEL MODEL_KQUANT QWEN2_MODEL QWEN3_MODEL
```

另有 4 个表格驱动的（`internal/quant/formats_real_test.go:101` 的 `os.Getenv(g.env)`：
`MODEL_Q4_1`、`MODEL_Q5_0`、`MODEL_Q5_1`、`MODEL_IQ4_NL`），合计 14，与表格相符。

但「搜 `Getenv("` 找到九个」这句话已经不成立：现在直接搜到 10 个模型变量，另有四个
`ORBIX_*`（`ORBIX_ROPEDUMP`、`ORBIX_PREDUMP`、`ORBIX_QUANTDUMP`、`ORBIX_REFERENCE_DIR`）
也会被同一个 grep 命中，所以那句话里的「九」既不是模型门禁数，也不是 grep 命中数。

**影响**：这一段的价值全部建立在「数字是量过的」上，而数字已经过期。文档里自己都写了
「worth re-measuring rather than trusting these figures」，需要在新增门禁后重测。

补充实测：只设 `MODEL` + `QWEN2_MODEL` 两个门禁时 skip 从 31 降到 16，说明门禁机制本身是好用的，
坏的只是文档里记的数。

### 4. `docs/architecture-review.md` 的规模表、覆盖率、t.Skip 计数全部过期

这篇文档开篇说「所有数字都来自这一轮在工作区里跑出的命令，命令附在每节末尾，便于复算」。
逐条复算的结果是每一项都对不上：

**规模表（第 11 行起）**：文档给 `go run ./scripts/count.go -table` 作为复算命令，实测输出：

| 包                | 文档代码行 | 实测      | 文档测试行 | 实测      |
| ----------------- | ---------- | --------- | ---------- | --------- |
| `internal/server` | 4225       | **5015**  | 10622      | **12264** |
| `internal/llama`  | 4807       | 4820      | 12480      | 12844     |
| `cmd/orbix`       | 3596       | 3613      | 7953       | 8042      |
| `internal/chat`   | 2614       | **2767**  | 4160       | 4426      |
| `internal/engine` | 1965       | **2065**  | 5819       | 6082      |
| 合计              | 26825      | **27996** | 58882      | **62317** |

文档自己的复算命令就能推翻它的表，而且是 `server` 差 790 行、`chat` 差 153 行这种量级。

**覆盖率快照（第 271 行）**：

| 包                           | 文档  | 实测      |
| ---------------------------- | ----- | --------- |
| `internal/engine`            | 94.7% | **96.0%** |
| `internal/server`            | 94.9% | **95.7%** |
| `internal/chat`              | 93.2% | **93.4%** |
| `internal/otel`              | 91.5% | **92.9%** |
| `internal/llama`             | 89.2% | **89.9%** |
| `internal/gguftest`          | 88.6% | **89.1%** |
| `cmd/orbix`                  | 87.8% | **88.0%** |
| `internal/tokenize`          | 96.7% | 96.8%     |
| `cmdServe`（正文里单点提的） | 50.4% | **39.2%** |

前八项方向都对（都是涨的，符合「测试在增加」），但作为「2026-11 末复测」的快照已经不准。

第九项方向**相反**，值得单独说：文档写 `cmdServe` 是 50.4%，连续两次实测都是 **39.2%**，
掉了 11 个百分点。这不是「测试变多了所以数字上涨」能解释的，只能说明文档写下之后
`serve.go` 里加了大段没被测试覆盖的代码（近两日该文件只有注释提交，所以更可能是那个
50.4% 本来就量自不同的口径或更早的版本）。而同一段的论断「`cmd/orbix` 的低点集中在 `cmdServe`」
仍然成立，所以结论没错、数字错了——这正是 `AGENTS.md` 里「结论对，理由不一定对」的同一个问题。

**`t.Skip` 计数（第 302 行）**：「73 处 `t.Skip` 分布在 28 个测试文件里」，
实测 **74 处 / 29 个文件**。

**`jinja_diff.py` 计数（第 300 行）**：「714 例、700 一致、4 例按设计发散」，实测：

```console
$ uv run --offline --with jinja2 python3 scripts/jinja_diff.py
712 cases compared, 700 agree with Jinja2
2 diverge by design, each producing its documented output
10 unsupported constructs refused, as designed
```

总例数与发散数都对不上（714→712，4→2）。这条工具本身 exit 0 是好的。

**影响**：`docs/architecture-review.md` 是这篇仓库里唯一一份「结构性分析」，它的说服力完全来自
「数字可复算」。现在复算命令还在，但结果已经和表不一致，读者会发现这一点。要么重新生成这张表，
要么把它明确标成某一时点的历史快照（它第二部分已经这么做了，第一部分没有）。

### 5. `docs/architecture-review.md` 的长函数清单也过期了，且漏了一个

第 400 行：「**18 个函数在 100 行以上**」，第 418 行：「仍未处理、>=100 行的十七个」。
用 AST 按 `FuncDecl` 起止行实测，是 **18 个**——所以「18」对、「十七个（未处理）」也对
（18 = 已处理的 1 个 + 未处理的 17 个）。这一处没错。

但清单里**每一条的长度都在变，且有一项漏了**：

| 函数                     | 文档       | 实测    |
| ------------------------ | ---------- | ------- |
| `chat.evalExpr`          | 242        | **271** |
| `cli.cmdServe`           | 238        | **255** |
| `server.streamOneChoice` | 157        | **205** |
| `server.generate`        | 189        | **192** |
| `llama.attentionBatch`   | 126        | 126 ✓   |
| `tokenize.preFamilyOf`   | 271        | 271 ✓   |
| `server.handleChat`      | **未列出** | **122** |

`server.handleChat`（`internal/server/server.go:954`）122 行，比清单里的 `engine.Submit`（122）
一样长，但文档里搜不到它。`streamOneChoice` 从 157 涨到 205（+48）是全表变化最大的一项。

同一段的「722 个函数，59 个在 60 行以上」也和当前实测的 747 个不符（文档自己的
`count.go` 输出就是 747）。

### 6. `docs/architecture-review.md` 的依赖图仍然准确

这一节复算后是对的：

```console
$ go list -f '{{join .Imports "\n"}}' ./internal/server | grep -c "internal/llama"
0
```

`internal/server` 确实不导入 `llama`，`internal/arch/layers_test.go` 也在跑并且绿。
这条是这份文档里少数几个复算后仍然成立的结构性论断。

### 7. `n > 1` 在 greedy 下返回完全相同的 choices，文档却说「会不同」

`docs/usage.md:250-251`：

> With a `seed`, choice *i* is seeded with *seed + i*... without one, the choices still differ from each other.

实测三连（温度默认 0 = greedy）：

```console
$ curl ... -d '{"messages":[...],"n":4,"max_tokens":6}'
unique: 1 ["I'm sorry, but I", "I'm sorry, but I", "I'm sorry, but I", "I'm sorry, but I"]
（三次运行结果一致，全部 4 个 choice 逐字相同）
```

`temperature: 0.9` 时确实不同（3 个 choice 三条不同的文本）。所以准确的说法是
「`n` 个 choice 在不同采样下会不同；greedy 下必然相同，因为贪心解码是确定性的」。

**影响**：文档这句话会让用户以为 `n: 3` 在默认设置下能得到 3 个候选并从中挑。实际不是。
`best_of` 同理：`best_of: 4, n: 2` 在 greedy 下两个 choice 都是 `Red\n\nA color is a`。

这不是 bug（greedy 本来就确定性），是文档描述误导。要么改文档，要么在 greedy 下
`n > 1` 打 warning（像 vLLM 那样）。

---

## 二、文档与代码的不一致（较小）

### 8. `--max-tokens` 的零值语义在三个地方三种说法

| 位置                                | 说法                                                      |
| ----------------------------------- | --------------------------------------------------------- |
| `run -h`                            | "run accepts 0 (prefill only)"                            |
| `serve -h` / `bench -h` / `chat -h` | 共用同一段文字，说 "bench, serve and chat need 1 or more" |
| `docs/configuration.md:153`         | "The server cannot express a zero-token request"          |

前两条是对的（实测 `run --max-tokens 0` 输出 1 字节换行、`gen: 0 tokens`；
`bench/serve/chat --max-tokens 0` 都以 exit 1 拒绝）。第三条是错的，见问题 1。

### 9. `orbix help <command>` 静默忽略参数

```console
orbix help run     # 打印的是顶层 usage，不是 run 的帮助
orbix help serve   # 同样
```

`help` 分支（`main.go:70`）直接调用 `usage()` 然后返回，不看后面的参数。
用户的预期是 `git help status` / `cargo help build` 那种行为。至少应该提示
`orbix run -h`，否则用户以为 run 没有参数说明。

### 10. `orbix model inspect` 的 pre-tokenizer 行对 GPT-2 词表显示 "(none)"

```console
orbix model inspect <qwen2>:    Tokenizer  gpt2, pre-tokenizer (none)
orbix model inspect <stories>:  Tokenizer  llama, pre-tokenizer (none)
```

两个模型都显示 `(none)`，但下面一行写 `Pre-tokenizer  implemented`。这两行连在一起读
是矛盾的：既然没有 pre-tokenizer，为什么说「实现了」？

`cmd/orbix/inspect.go:127` 是 `name = "(none)"`。真实含义是「文件里没有显式声明
`tokenizer.ggml.pre` 键」，而这个项目的 tokenizer 靠 `tokenizer.ggml.model`（`gpt2` / `llama`）
选实现。所以 `(none)` 应改成类似 `(built-in for gpt2)`，或者把两行合成一行。

### 11. `README.md` 的 quick start 与 `serve` 默认行为有个坑

README 说：

```bash
orbix serve --model ./models/qwen2.5-7b-instruct-q4_0.gguf --addr :8080
```

`--addr :8080` 绑所有网卡（`docs/configuration.md` 有说明），而 `docs/usage.md:514-531`
「No CORS and no authentication」整节都在讲这样绑定的风险。README 是门面，第一次跑的命令
就是绑公网。建议 README 里写 `127.0.0.1:8080`，让用户在需要时自己改成 `:8080`。

### 12. `n` 与 `best_of` 的上限在不同地方不一致

- `n` 上限 8（`n=9` → 400 `invalid_value`，实测确认）
- `best_of` 没有单独上限，但 `best_of` 必须 > `n` 且 n 最大 8，所以 `best_of` 实际最大 9 起

`docs/usage.md` 讲了 `n` 的 8 上限，没提 `best_of` 会不会因此也受约束。不是错，是缺失。

---

## 三、真实使用中发现的体验问题

### 13. `run` 的 CLI-only 参数与 `serve` 的差异容易踩

`--stats`、`--echo`、`--seed`、`--ctx` 是 `run` 独有；`--system` 是 `chat` 独有；
`--addr`、`--log-format` 等是 `serve` 独有。`orbix serve --stats` 会报
`flag provided but not defined: -stats` 然后打一整屏 usage。

真实用户想「给 server 也看看启动耗时/加载耗时」时会撞上这个。启动 banner 已经有
weights 和 kv cache，但没有 load 时间。**建议**：`serve` 的 banner 加一行 `load:`。

### 14. `bench` 没有 `--stats` 的等价物，`--requests` 默认值不直观

`bench --requests 0` = 「每个 concurrency slot 一个请求」。实测
`--concurrency 2 --requests 3` 会跑 3 个请求但 2 并发，最后一个单独跑，
输出的 `per stream: 7.0`（12/2 而非 12/3）容易误读。

**建议**：输出里加一行说明 requests 与 concurrency 的实际关系。

### 15. REPL 没有行编辑/历史

`cmd/orbix/chat.go:144` 用 `bufio.NewScanner`，没有 readline。所以：

- 不能按上箭头调上一条消息
- 不能左右移动光标改字
- 行太长时会滚屏

对一个「交互式 REPL」来说这是明显缺口。文档标 `v0.3`，可以晚做，但值得列出来。

### 16. `chat` 的 `/stats` 在未开始时显示 `2046 of 2048 blocks free`，对 128-ctx 模型是 8 blocks

实测 stories15M（128 上下文）：

```text
kv pool:    8 of 8 blocks free (16 positions each, 128 total)
```

这个数是对的（`8×16=128`），但和 Qwen2 的 `2046/2048`（`2048×16=32768`）对比会让人以为
「默认 2048 blocks」。实际是「按上下文大小算出的 block 数」。banner 里不解释这一点。

### 17. 长 prompt 在 `run` 下没有事前检查

`orbix run --model X --max-tokens 40000 "hi"` 在 Qwen2 上跑了 120 秒仍未结束
（被我的 timeout 杀掉），因为它真的在生成直到窗口满。`serve` 那边对超窗口请求马上给
`context_length_exceeded`（实测：40002 tokens → 400）。`run` 只是事后打 note。

**建议**：`run` 也可以事前把 budget 截到窗口并提示，省掉长时间空跑。

### 18. `orbix run` 全程缓冲输出，中断就全丢，且没有任何进度

这是这次验证里最影响实际使用的一条。`orbix run` 把整段生成跑完之后才打印一次
（`cmd/orbix/run.go:131-139`：`m.Generate(...)` 返回整个 `out` 切片，再 `fmt.Println`）。
实测每秒采样输出字节数：

```console
$ orbix run --model qwen2-f32.gguf --max-tokens 200 "Write a story" > out &
t=1s bytes=0
t=2s bytes=0
...   （全程 0，屏幕上什么都没有）
t=8s bytes=0
final bytes=1096     # 全部在结束时一次出现
```

后果有三个，都是真实用户会遇到的：

1. **看不到任何进度。** 200 token 要 8 秒，2000 token 要 80 秒，这期间终端一片空白。
   用户无法判断是在跑还是卡住了。`stories15M` 上 24 token 只要 0.1 秒所以不明显，
   换成 0.5B 以上的模型就非常明显。
2. **中断即全丢。** 实测 `SIGTERM`（exit 143）和 `Ctrl-C` 之后 `stdout` 都是 **0 字节**，
   已经算出来的 token 一个都留不下。用户等了两分钟，按 Ctrl-C 想「至少拿到已有的部分」，
   拿到的是空。
3. **`--stats` 也一起丢。** 因为 stats 是在 `Generate` 返回之后才算的（`run.go:140`），
   中断时连「跑了多少、多快」都没有。

对比：`chat` 和 `serve` 都是逐 token 输出的（`chat` 每个 turn 打印一次回复，`serve` 的 SSE
逐 chunk 推送）。所以这不是引擎的限制，是 `run` 这一个命令的实现选择。

**建议**：`run` 改成边生成边写（`Generate` 已有回调或可加一个），至少让 `Ctrl-C` 之前
生成的部分留在屏幕上。这也是 `llama.cpp` 的 `main`/`llama-cli` 的行为，用户会预期如此。

注：我最初以为「`run` 忽略 SIGINT」，核查后发现那是测试假象——后台启动的进程会继承
`SIGINT` 为 ignored（`/proc/PID/status` 的 `SigIgn` 是 `0x2`，连 `sleep` 也一样），
`SIGTERM` 是立即生效的（exit 143）。所以问题不是「不响应信号」，而是「响应了但什么都没留下」。

### 19. `--workers` 帮助文字里的数字没有复现命令

```text
--workers int
    ...measured on a 128-core host, 8 workers give about two thirds of what 32 do,
    and beyond 32 the difference is inside the noise
```

`docs/performance.md` 里那张 `--workers` 延迟表（1/2/4/8/16/32/64/128）和这句话对不上：
表里 16 workers 是 10.31x、32 是 9.20x、128 是 12.30x，也就是 **32 比 16 慢**，
不是「8 workers 只有 32 的三分之二」这么简单的单调关系。文档自己也写了「The curve is not
monotone」，但 `-h` 里的那行没提。两处措辞应该统一。

### 20. 下载真实 checkpoint 的过程不稳定

`docs/models.md:302` 的 `curl -sSL` 实测两次失败：

```console
curl: (92) HTTP/2 stream 1 was not closed cleanly: CANCEL (err 8)   # 第一次，下了 4.7 MB
curl: (56) CONNECT tunnel failed, response 503                      # 换成 --http1.1 后
```

第三次加 `--retry 2 --retry-delay 2` 才成功。`AGENTS.md` 里已经写了 `dl`（aria2c 封装）
能用断点续传，但 `docs/models.md` 给用户的仍然是裸 `curl`。

**建议**：文档里的下载地址旁边加 `--retry` 或指明 `dl`/`aria2c -c`，否则第一次装就失败。

### 21. `serve` 的 banner 在 paged 配置下把 `kv cache` 排在最后

contiguous 的 banner 顺序是 `weights → kv cache → access log → chat → POST`；
paged 的是 `weights → chat → POST → kv cache → prefix cache`。

对比：

```text
# contiguous
weights:    93.1 MiB resident ...
kv cache:   contiguous, about 1.7 MiB per request, ...
access log: text (on stderr, not stdout)
chat:       neutral transcript (...)

# paged
weights:    93.1 MiB resident ...
chat:       neutral transcript (...)
POST http://127.0.0.1:19021/v1/chat/completions
kv cache:   paged, 32 blocks x 16 positions, 6.8 MiB shared
prefix cache: on (...)
```

`POST ...` 是「现在可以打这个地址了」的意思，结果它后面还有两行配置。而且 kv cache 是
`docs/configuration.md` 里说「usually the larger of the two」的关键数字，被排到了 POST 之后。

**建议**：统一顺序，`kv cache` 紧跟 `weights`。

---

## 四、验证过的、确实没问题的部分

这些都是实测过的，不是读代码推断：

| 项目                                                 | 结果                                                                                                                                                                                                                                    |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `go install github.com/pplmx/orbix/cmd/orbix@latest` | 成功，`version` 报 `dev`/`none`/`unknown`，与 `docs/cli.md:50-53` 描述一致                                                                                                                                                              |
| 全新 clone 的 `just build`                           | 成功，注入 `b5c4d5b`                                                                                                                                                                                                                    |
| `just check`                                         | 绿（vet + test + lint + gofmt）                                                                                                                                                                                                         |
| `just docs`                                          | 绿（10 文件）                                                                                                                                                                                                                           |
| `just hooks`                                         | 绿                                                                                                                                                                                                                                      |
| `just cross`                                         | 5 个平台全部构建成功，产物带 `build\tCGO_ENABLED=0` 标记                                                                                                                                                                                |
| `just isolate`                                       | 绿（`-count=2 -shuffle=on`）                                                                                                                                                                                                            |
| `just fuzz`                                          | 9 个 target 全部 PASS（3 个在 tokenize，其余 chat/gguf/kernel/llama/quant/server 各一）                                                                                                                                                 |
| `just bench`                                         | 绿                                                                                                                                                                                                                                      |
| `just fixture` + `just stress`                       | 绿（3 cancelled，见 `docs` 描述的判据）                                                                                                                                                                                                 |
| `just xcheck`                                        | 290 张量类型/形状/SHA-256 全部一致                                                                                                                                                                                                      |
| OpenAI Python SDK（3.57.0）                          | chat / completions / stream / `stream_options.include_usage` / tools / models.list 全部可用                                                                                                                                             |
| `/metrics`                                           | paged 16 families / 52 行，contiguous 12 families / 40 行，与 `docs/cli.md:191-197` 一致                                                                                                                                                |
| `/v1/models`、`/v1/models/{id}`                      | 正常，未知 id 404 `model_not_found` 并指出真实模型名                                                                                                                                                                                    |
| 错误码                                               | `invalid_json` / `invalid_type` / `invalid_value` / `missing_required_parameter` / `body_too_large` / `unsupported_encoding` / `unknown_endpoint` / `method_not_allowed` / `context_length_exceeded` / `model_not_found` 全部按文档返回 |
| 503 + `Retry-After: 1`                               | 队列满时正确返回                                                                                                                                                                                                                        |
| SIGTERM 优雅退出                                     | 5s grace 到期后切断，流无 `[DONE]`，日志打出 `shutdown: ... was cut`，与 `docs/usage.md:400-440` 一致                                                                                                                                   |
| 前缀复用计数                                         | 两次同一 prompt 后 `orbix_prefix_positions_reused_total` 从 0 → 64                                                                                                                                                                      |
| `n=2` 流式                                           | 2 个 finish chunk、1 个 `[DONE]`、index 0/1 正确交错                                                                                                                                                                                    |
| `logprobs` / `top_logprobs` 上限                     | completions 6 → 400；chat 21 → 400，与 `docs/usage.md:122-124` 一致                                                                                                                                                                     |
| 空 prompt                                            | `run ""` 接受（1 token），`serve` 400 `missing_required_parameter`，`chat` 跳过，与 `docs/cli.md:71-77` 一致                                                                                                                            |
| `--suite` 拒绝冲突 flag                              | `--concurrency`/`--blocks`/`--policy` 均被拒并说明原因                                                                                                                                                                                  |
| 窗口切 budget 的 note                                | `run --stats` 在 128-ctx 模型上打 `note: the 128-token window is full, so the 200-token budget was cut to 123`                                                                                                                          |
| chat 溢出说明只打一次                                | 3 次相同溢出输入，第一次完整说明，后两次 `(unchanged; /clear starts over)`                                                                                                                                                              |
| 模型文件错误                                         | 非 GGUF → `bad magic "not ", want "GGUF"`；截断 → 报具体 key 和元素序号；目录/空文件分别报错                                                                                                                                            |

---

## 五、优先级建议

| 优先级 | 条目                                                         | 理由                                         |
| ------ | ------------------------------------------------------------ | -------------------------------------------- |
| P0     | 问题 1（`max_tokens: 0` 的说法）                             | 事实错误，且拒绝了一个合法配置               |
| P0     | 问题 2（`suffix` 自相矛盾）                                  | 同一文档 100 多行在讲一个「没实现」的功能    |
| P1     | 问题 18（`run` 缓冲输出，中断全丢）                          | 唯一一条影响实际使用的功能问题               |
| P1     | 问题 7（`n>1` greedy 相同）                                  | 用户会以为拿到多个候选                       |
| P1     | 问题 3（skip 数字）                                          | 文档自称「这是量出来的」，数字已过期         |
| P1     | 问题 4、5（`architecture-review.md` 的规模/覆盖率/函数清单） | 开篇承诺「所有数字可复算」，逐条复算都对不上 |
| P1     | 问题 20（下载说明）                                          | 第一次安装就会失败                           |
| P2     | 问题 9、10、21（help / inspect / banner）                    | 都是小改动                                   |
| P2     | 问题 17（`run` 长 budget）                                   | 体验，浪费用户时间                           |
| P3     | 问题 15（REPL 行编辑）                                       | v0.3 功能，可排期                            |

### 一句话总结

**引擎本身是可靠的**：所有门槛全绿（`check` / `docs` / `hooks` / `cross` / `isolate` / `fuzz` /
`bench` / `stress` / `xcheck`），OpenAI SDK 直连可用，错误码齐全且带 `param`，SIGTERM 切断行为
与文档描述逐字相符，流式、前缀复用、队列背压都在真实 checkpoint 上验证过。

全文 21 条，其中第 6 条是「复算后确认准确」的记录，其余 20 条是待处理项，分三类：

| 类别                                             | 条数 | 编号                                |
| ------------------------------------------------ | ---- | ----------------------------------- |
| 文档与代码不符（事实错误 + 数字过期 + 自相矛盾） | 11   | 1、2、3、4、5、7、8、11、12、19、20 |
| CLI 与输出体验                                   | 8    | 9、10、13、14、15、16、17、21       |
| 功能问题                                         | 1    | 18（`run` 缓冲输出）                |

这符合一个规律：代码有测试守着，文档没有。`cmd/orbix/flagsdocs_test.go` 已经给 flag 表做了
双向校验（命令有的必须写进文档、文档写的必须存在），所以 flag 表是这堆文档里唯一没漂的部分——
复算 `TestConfigurationDocListsEveryFlag` 是绿的。其余可复算的数字没有对应守卫，就慢慢漂了。

**最值得做的一件事**：把 `flagsdocs_test.go` 那个模式扩到 `docs/models.md` 和
`docs/architecture-review.md` 里的数字（skip 计数、覆盖率、函数长度表）。否则下一个改动
还会让它们继续漂。

---

## 六、处理记录（2026-11-25）

全部改动由六个提交组成，`just check` / `just docs` / `just hooks` / `just commits` 均绿。

| 条目                                          | 处理                                                                                                                                                                            | 证据                                                                                    |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| 1 `max_tokens: 0` 的说法                      | 已修：`docs/configuration.md`、`serve.go` 报错、`flags.go` 注释三处改成真实原因（flag 是默认值，`server.Config` 把 `<1` 读作 256），并写明请求体里的 `max_tokens: 0` 是另一回事 | 实测 `serve --max-tokens 0` 新报错打印正确原因；`max_tokens: 0` 请求仍返回空 completion |
| 2 `suffix` 自相矛盾                           | 已修：改为「不拒绝字段，只拒绝缺 FIM marker 的 checkpoint」                                                                                                                     | 与 `server.go` 的拒绝分支一致                                                           |
| 3 skip 计数 25→31                             | 已修，并新增 `internal/quant/gatesdoc_test.go` 双向校验表格与代码                                                                                                               | 删除一行、编造一行，两种注入都让测试红                                                  |
| 4 `architecture-review.md` 规模/覆盖率/t.Skip | 已修：规模表直接用 `count.go` 输出重新生成                                                                                                                                      | 文档数字与工具输出逐字相同                                                              |
| 5 长函数清单                                  | 已修：19 个、名字与长度按 AST 实测；补上遗漏的 `handleChat`                                                                                                                     | `cmdServe` 261、`runTo` 165、751 个函数                                                 |
| 6 依赖图                                      | 原本准确，未改                                                                                                                                                                  | —                                                                                       |
| 7 `n>1` greedy                                | 已修：`usage.md`、`server.go`、`doc.go` 三处改为「只在采样开启时不同」，新增 `TestGreedyChoicesAreIdentical`                                                                    | 注入逐 choice 温度后两个断言都红                                                        |
| 8 零值语义三种说法                            | 已随条目 1 统一                                                                                                                                                                 | —                                                                                       |
| 9 `help <command>`                            | 已修：委派到子命令自己的 `-h`                                                                                                                                                   | 四个命令的测试均验证；注入旧行为后红                                                    |
| 10 `inspect` 的 `(none)`                      | 已修：改成说明性文字，加一致性测试                                                                                                                                              | 注入旧文案后红                                                                          |
| 11 README quick start 绑公网                  | **未改**，见下                                                                                                                                                                  | —                                                                                       |
| 12 `n`/`best_of` 上限                         | 已随条目 7 在文档里说明                                                                                                                                                         | —                                                                                       |
| 13 `serve` 无 load 时间                       | **未改**，见下                                                                                                                                                                  | —                                                                                       |
| 14 `bench` 的 requests 语义                   | **未改**，见下                                                                                                                                                                  | —                                                                                       |
| 15 REPL 行编辑                                | **未做**：需要引入 readline 依赖，与仓库「无依赖」的结论冲突                                                                                                                    | —                                                                                       |
| 16 `chat /stats` 的 block 数                  | **未改**，见下                                                                                                                                                                  | —                                                                                       |
| 17 `run` 长 budget 空跑                       | 已修：事前在 stderr 打提示，说明窗口与可生成上限                                                                                                                                | 457ms 即出现；预算够时不出现                                                            |
| 18 `run` 缓冲输出                             | 已修：改为边生成边写                                                                                                                                                            | 中断后保留 487 字节（原 0）；与旧实现逐字节一致                                         |
| 19 `--workers` 帮助文字                       | **未改**，见下                                                                                                                                                                  | —                                                                                       |
| 20 下载说明                                   | **未改**，见下                                                                                                                                                                  | —                                                                                       |
| 21 `serve` banner 顺序                        | 已修：paged 的 kv cache / prefix cache 移到 POST 之前                                                                                                                           | banner 末行现在是 POST                                                                  |

### 没有改的六条，及原因

- **11（README 绑公网）**：`--addr :8080` 是文档明确解释过的默认行为，README 里写它与
  `configuration.md` 一致。改成 loopback 会让「所有接口」这个默认值在两处不一致，
  属于产品决策而非缺陷。
- **13、14、16（banner 加 load 时间、bench 的 requests 说明、chat 的 block 数解释）**：
  三条都是「多印一行说明」，会改变已有输出格式，而现有测试与文档都按当前格式断言。
  值得做，但应该作为一个独立的输出设计改动，不适合混在修 bug 的提交里。
- **15（REPL 行编辑）**：`chat` 标的是 `v0.3`。做它需要 readline 一类依赖，而
  `AGENTS.md` 的依赖表目前是空的且逐条记录了不采用的理由。这是需要用户决定的事。
- **19（`--workers` 帮助文字）**：帮助文字与 `performance.md` 的表述差异是「同一事实的
  两种详略」，不是矛盾。改它需要先重测那张表，而 `AGENTS.md` 明确记录本机 load average
  波动会让这类测量不可信。
- **20（下载说明）**：给的是「加 `--retry`」的建议，属于文档增量而非纠错；且失败的两次
  都发生在同一个网络环境下，不足以断定命令本身有问题。

### 验证方式

每个新增测试都用注入验证过判别力，不是只跑绿就算：

- 缓冲输出 → `TestRunWritesWhileItGenerates` 红
- 吞掉 `OnToken` 错误 → `TestRunReportsAWriteFailure` 红（第一次写成「首写即失败」时该测试
  不红，因为 flush 也会报错；改成「失败发生在流中途 + 断言写入次数」才有判别力）
- 逐 choice 温度 → `TestGreedyChoicesAreIdentical` 两个断言都红
- 删/加 gate 表格行 → `TestModelsDocGateTableMatchesTheCode` 两个方向都红
- 旧 `help` 行为 → `TestHelpForACommandPrintsThatCommandsFlags` 红
- 旧 `(none)` 文案 → `TestModelInspectPreTokenizerLinesAgree` 红
- 去掉预算提示 → `TestRunWarnsBeforeCuttingABudgetToTheWindow` 红

流式输出与旧实现逐字节一致：两个 checkpoint、三个英文 prompt、一个 CJK prompt 全部 `cmp` 相同。
