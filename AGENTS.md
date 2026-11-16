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
make test    # go test ./...
make lint    # golangci-lint run
make check   # 本地门槛：vet + test + gofmt
make bench   # 基准测试
```

配置以单一事实源维护：Go 用 `go.mod`，lint 用 `.golangci.yml`，Markdown 用 `.rumdl.toml`。

## 目录

```text
cmd/orbix/         CLI 入口：参数解析、输出渲染
internal/gguf/     GGUF 容器解析：头部、元数据、张量目录、张量数据
internal/<层>/     后续层按 README 的阅读顺序添加
```

## 分层与阅读顺序

README 承诺每层都有 `doc.go` 解释概念和设计取舍。新增包时必须写 `doc.go`，讲清"这一层在推理里负责什么"，而不是复述代码。

顺序：GGUF → kernels → KV cache → scheduler → server。新代码顺着这个顺序加，上层可以依赖下层。

## 代码风格

- 注释、日志、报错信息用英文，只解释非显而易见的约束和原因，不复述签名。
- 错误返回带上下文：`fmt.Errorf("gguf: read %q: %w", key, err)`，不吞异常。
- 数值常量（量化块大小、字节数）必须在测试里用独立的算术复算一遍，防止抄错后只在推理出垃圾输出时才暴露。
- 解析器遇到未知枚举值要如实报告类型，不猜、不静默跳过。

## 测试

- 单元测试不依赖网络和真实模型文件；GGUF 用 `builder_test.go` 里手写的字节构造器生成 fixture。
- 解析器的错误路径要测：截断、幻数错误、版本不符、越界张量。
- 真实模型只用于本地交叉验证，不写进 CI。交叉验证的做法：用独立的参考实现（`uv run --with gguf`）解析同一文件，逐张量比对 SHA-256。

## 文档

- README 是门面和索引，权威正文放在各包 `doc.go`。
- README 里承诺的 CLI 命令和 Make 目标必须真的可用；改接口时同步更新 README。
- Markdown 过 `rumdl check`，表格用对齐样式。
