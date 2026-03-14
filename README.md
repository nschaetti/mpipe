![mpipe banner](images/banner/banner_readme.png)

# mpipe
[![CI](https://github.com/nschaetti/mpipe/actions/workflows/ci.yml/badge.svg)](https://github.com/nschaetti/mpipe/actions/workflows/ci.yml)

A set of LLM-based command line tools

## `mpipe ask`

Minimal Unix CLI to send one prompt to an LLM provider and print raw text output.

`mpask` is kept as a compatibility alias and supports the same options/behavior as `mpipe ask`.

## `mpipe prompt render`

Render the final composed prompt locally without any API call.

```bash
mpipe prompt render "Explain retries"
echo "Explain retries" | mpipe prompt render --prompt "You are concise"
mpipe prompt render --json --system "You are concise" --prompt "Context" --postprompt "Answer in bullets" "Explain retries"
```

## `mpipe models`

List known models from the local catalog.

```bash
mpipe models
mpipe models --provider fireworks
mpipe models --json
```

`--json` includes `provider`, `id`, `source` (`local`), and `recommended`.

## `mpipe index`

Index a text document into ChromaDB (with optional chunking and metadata).

```bash
mpipe index --file notes.txt --collection docs --embedding-model accounts/fireworks/models/kimi-k2-instruct-0905
mpipe index --document "Hello world" --source "chat://session/42" --collection scratch --embedding-model accounts/fireworks/models/kimi-k2-instruct-0905
mpipe index --file notes.txt --collection docs --chroma-path ./.chroma --embedding-model accounts/fireworks/models/kimi-k2-instruct-0905
```

Source metadata policy:

- `--file`: `source` is auto-filled with the file path (unless `--source` is provided).
- `--document`: `--source` is required.
- `--source` always wins over metadata values.

Provide embeddings via stdin (one vector per line, comma-separated):

```bash
printf "0.1,0.2,0.3\n0.4,0.5,0.6" | mpipe index --file notes.txt --collection docs
```

When embeddings are not provided via stdin, `mpipe index` uses Fireworks embeddings and requires `FIREWORKS_API_KEY`.

Metadata can be passed as JSON and overridden by `--metadata`:

```bash
mpipe index --file notes.txt --metadata-json metadata.json --metadata lang=fr --metadata source=manual --embedding-model accounts/fireworks/models/kimi-k2-instruct-0905
```

ChromaDB connection resolution:

- CLI: `--chroma-url` or `--chroma-host`/`--chroma-port`/`--chroma-scheme`
- Local persistent mode: `--chroma-path <dir>` (or env `CHROMA_PATH`) auto-starts `chroma run` and stores data in that directory
- Env: `CHROMA_URL`, `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_SCHEME`, `CHROMA_PATH`
- Collection: `--collection` or `CHROMA_COLLECTION` (default `mpipe`)

Note: `--chroma-url` cannot be combined with `--chroma-path`.

## `mpipe list`

List entries in a collection.

```bash
mpipe list --collection docs --limit 20
mpipe list --collection docs --offset 20 --limit 20 --json
```

## `mpipe grep`

Classic RAG: retrieve top chunks from ChromaDB, then ask the LLM with those chunks as context.

```bash
mpipe grep --collection docs --embedding-model accounts/fireworks/models/qwen3-embedding-8b --model gpt-4o-mini "What does the retry logic do?"
mpipe grep --collection docs --top-k 8 --embedding-model accounts/fireworks/models/qwen3-embedding-8b --provider fireworks --model accounts/fireworks/models/kimi-k2-instruct-0905 "Resume this document"
```

### Prompt input

- `mpipe ask "question"` uses the CLI argument as prompt.
- `echo "question" | mpipe ask` reads from stdin when no prompt argument is passed.
- If both are present, the argument takes precedence.
- `--prompt "..."` prepends a preprompt before the main prompt.
- `--postprompt "..."` appends text after the main prompt.
- `--system "..."` adds a system message sent before the user message.
- When used, segments are joined as `preprompt\n\nmain_prompt\n\npostprompt` (missing segments are skipped).

When `--system` is provided, `mpipe ask` sends two chat messages: `system` then `user`.

### Provider and model selection

- Provider resolution order: `--provider` > `MP_PROVIDER` > default `openai`
- Supported providers: `openai`, `fireworks`
- Model resolution order: `--model` > `MP_MODEL`
- If no model is provided, `mpipe ask` exits with an explicit error.

### Profiles

- `--profile <name>` loads settings from a config profile.
- No implicit profile is loaded when `--profile` is not provided.
- Config path resolution:
  - `MP_CONFIG` if set
  - otherwise `${XDG_CONFIG_HOME:-~/.config}/mpipe/config.toml`
- Resolution priority for overlapping values is:
  - CLI flags > environment variables > profile > provider defaults > built-in defaults

Example config file:

```toml
[providers.openai.defaults]
timeout = 20
output = "text"

[providers.fireworks.defaults]
timeout = 15
retries = 2
retry_delay = 300
output = "json"

[profiles.fireworks]
provider = "fireworks"
model = "accounts/fireworks/models/kimi-k2-instruct-0905"
temperature = 0.2
timeout = 10
show_usage = true
system = "You are concise"
```

Validate config locally (no API calls):

```bash
mpipe config check
mpipe config check --profile fireworks
```

### Generation options

- `--temperature <float>` (range `[0.0, 2.0]`)
- `--max-tokens <int>` (must be `> 0`)
- `--timeout <secs>` (must be `> 0`)
- `--retries <n>` (number of extra retry attempts)
- `--retry-delay <ms>` (base delay, must be `> 0`)
- Environment fallbacks:
  - `MP_TEMPERATURE`
  - `MP_MAX_TOKENS`
  - `MP_TIMEOUT`
  - `MP_RETRIES`
  - `MP_RETRY_DELAY`

If `temperature`, `max-tokens`, or `timeout` are missing, they are not sent and the provider/client default is used. Retry defaults are `retries=0` and `retry-delay=500ms`.

Retries use exponential backoff: `retry-delay * 2^attempt`, capped at 30 seconds.

### Output format

- `--output <text|json>` controls stdout format (`text` by default)
- `--json` is a shortcut for `--output json`
- `--show-usage` prints token usage and latency on stderr
- `--quiet` suppresses non-critical stderr logs (usage/verbose), while keeping fatal errors visible
- `--save <path>` writes the final stdout payload to a file (overwrite mode)
- `--fail-on-empty` returns an error if the model answer is empty (applies in both text and json modes)

`text` prints only the raw answer.

`json` prints one JSON object with:

- `provider`
- `model`
- `answer`
- `latency_ms`
- `request` (`temperature`, `max_tokens`, `timeout_secs`, `retries`, `retry_delay_ms`)
- `usage` (token counts when available, otherwise `null`)

When `--show-usage` is enabled, `mpipe ask` prints either token usage + latency or `usage: unavailable` to stderr.

### Debug modes

- `--verbose` prints request diagnostics to stderr (provider, endpoint, resolved options, prompt source, message counts)
- `--dry-run` prints the final request payload as JSON to stdout and does not call any API
- `--version` prints version, commit, and build timestamp metadata

When both `--quiet` and `--verbose`/`--show-usage` are set, `--quiet` wins for stderr informational output.

`--dry-run` works without API keys. Authorization is always redacted in dry-run output.

### Shell completion

Generate completions with:

```bash
mpipe completion bash
mpipe completion zsh
mpipe completion fish
```

Example install (bash):

```bash
mpipe completion bash > ~/.local/share/bash-completion/completions/mpipe
```

### API keys

- OpenAI: `OPENAI_API_KEY`
- Fireworks: `FIREWORKS_API_KEY`

If the required key is missing for the selected provider, `mpipe ask` prints an explicit error to stderr and exits with a non-zero code.

### Test commands

Fireworks (recommended test model):

```bash
export MP_PROVIDER=fireworks
export MP_MODEL="accounts/fireworks/models/kimi-k2-instruct-0905"
export FIREWORKS_API_KEY="..."
echo "2+2?" | cargo run --quiet --bin mpipe -- ask
```

Equivalent explicit flags:

```bash
echo "2+2?" | cargo run --quiet --bin mpipe -- ask --provider fireworks --model "accounts/fireworks/models/kimi-k2-instruct-0905"
```

OpenAI example:

```bash
export OPENAI_API_KEY="..."
echo "2+2?" | cargo run --quiet --bin mpipe -- ask --provider openai --model "gpt-4o-mini"
```

## Development

Standard local targets:

```bash
make check
make clippy
make test
```

CI runs the same checks on `push`/`pull_request`:

- `cargo fmt --all -- --check`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo test --all-targets --all-features`
