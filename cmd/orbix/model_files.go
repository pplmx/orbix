package main

import (
	"fmt"

	"github.com/pplmx/orbix/internal/engine"
	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/llama"
	"github.com/pplmx/orbix/internal/tokenize"
)

// modelFiles is one model, opened once.
//
// Every command needs three things from the same file: the metadata directory (for the
// tokenizer and the chat template), the config (for the context window and the vocabulary
// size), and the weights (which engine.New loads). Before this type each command opened the
// file for each purpose separately: `serve` called gguf.Open twice, then engine.New reached
// through llama.NewWithOptions into quant.OpenLoader for a third parse of the same directory.
//
// The parse is not free -- measured on the 553 MiB Llama 3.2 file, gguf.Open alone takes 1.47 s
// -- and more importantly three openings are three places the file can fail or change between
// reads. One handle makes the command's view of the model consistent, and makes "which context
// length is this command using?" a question with one answer.
//
// The weights are still loaded separately inside engine.New, because that is where the model
// lives and moving it would put weight loading in the CLI. What this saves is the metadata
// parsing and the config derivation, which is the part that was being repeated.
type modelFiles struct {
	path string
	file *gguf.File
	cfg  llama.Config

	tok      *tokenize.Tokenizer
	renderer *engine.ChatRenderer
}

// openModel opens a model once and derives everything the commands share from that one read.
//
// The tokenizer and the chat renderer are built here rather than by each caller because every
// command wants both, and the order matters: the renderer reads tokenizer.chat_template out of
// the same directory the tokenizer was built from, so building them from one *gguf.File is what
// guarantees they agree about the vocabulary.
func openModel(path string) (*modelFiles, error) {
	f, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	cfg, err := llama.LoadConfig(f)
	if err != nil {
		return nil, err
	}
	tok, err := tokenize.New(f)
	if err != nil {
		return nil, err
	}
	renderer, err := engine.NewChatRenderer(f)
	if err != nil {
		return nil, err
	}
	// Probe the template once, here, so a caller that prints a banner can trust it.
	//
	// UsesModelTemplate reads renderReason, which only a failed render writes, so without this
	// probe the banner said "the model's own chat template" for a template that could never
	// render -- and then printed the correction after the first answer, leaving two lines that
	// disagreed. Measured with a template of "{{ messages[99].content }}": banner first,
	// correction after the reply.
	//
	// A real render rather than a check, because a render is what records the reason. It costs
	// one template evaluation at startup, which every command can afford.
	renderer.Render([]engine.Turn{{Role: "user", Text: "hi"}})
	return &modelFiles{path: path, file: f, cfg: cfg, tok: tok, renderer: renderer}, nil
}

// ChatFormat describes which transcript the renderer will produce, for a banner.
//
// It is one expression rather than a branch at each call site because serve and chat print the
// same line and used to build it separately.
func (m *modelFiles) ChatFormat() string {
	if m.renderer.UsesModelTemplate() {
		return "the model's own chat template"
	}
	return fmt.Sprintf("a neutral transcript: %s", m.renderer.FallbackReason())
}

// cachePlan is the paged-cache shape a command settled on, after defaults.
//
// It exists so the two decisions that were duplicated -- how big a block is, and whether that
// block could ever fill -- are made once. Callers differ only in whether a pool is created by
// default (chat does, for prefix reuse) or only when asked (serve and bench).
type cachePlan struct {
	blockSize int
	blocks    int
}

// planCache resolves the cache flags against the model's own context.
//
// The default block size comes from llama rather than from a literal here: chat hardcoded 16
// while llama.BlockPool's own default was also 16, so the two agreed by coincidence and would
// have drifted the moment either changed.
//
// autoBlocks says this command sizes a pool when --blocks is unset. chat does, because a REPL
// wants paging plus prefix reuse on by default; serve and bench do not, because an absent
// --blocks means "keep the contiguous cache".
//
// The block-size check is skipped when no paged cache will exist: with the contiguous layout a
// block never fills because there are no blocks, which is not the failure this guard is about.
// serve and bench used to skip it under slightly different conditions -- serve required
// blocks>0 as well, bench did not -- so the same command line could be refused by one and
// accepted by the other.
func planCache(prefix string, cfg llama.Config, flags cacheFlags, autoBlocks, prefixCache bool) (cachePlan, error) {
	plan := cachePlan{blockSize: *flags.blockSize, blocks: *flags.blocks}
	if plan.blockSize <= 0 {
		plan.blockSize = llama.DefaultBlockSize()
	}
	if plan.blocks <= 0 && autoBlocks {
		// Size a pool for one full context. That is the workload paging plus prefix reuse
		// exists for: every turn re-sends the history, so the pool must hold one conversation
		// and the prefix index must be able to hand back the previous turn's blocks.
		plan.blocks = (cfg.ContextLen + plan.blockSize - 1) / plan.blockSize
	}
	if plan.blocks <= 0 {
		// No pool, so no block can fail to fill.
		return plan, nil
	}
	// A block wider than the context can never be filled, so it is never published to the prefix
	// index and reuse would silently never fire. Measured on the 128-position stories model with
	// --blocks 8 --block-size 1000 --prefix-cache and three identical prompts:
	// orbix_prefix_positions_reused_total stayed at 0 while the startup line said "prefix cache:
	// on"; the same run at --block-size 16 reported 32.
	//
	// Refused only when reuse was asked for. Without --prefix-cache a block wider than the
	// context is just a pool with one block in practice, which serves text correctly -- and
	// refusing it would reject a configuration that works.
	if prefixCache && plan.blockSize > cfg.ContextLen {
		return plan, fmt.Errorf("%s: --block-size %d is larger than this model's %d-position "+
			"context, so no block ever fills and --prefix-cache could never reuse anything; use "+
			"--block-size %d or less", prefix, plan.blockSize, cfg.ContextLen, cfg.ContextLen)
	}
	return plan, nil
}
