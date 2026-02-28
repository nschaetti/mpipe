![mpipe banner](images/banner/banner_readme.png)

# mpipe
A set of LLM-based command line tools

## `mpipe ask`

Minimal Unix CLI to send one prompt to an LLM provider and print raw text output.

`mpask` is kept as a compatibility alias and supports the same options/behavior as `mpipe ask`.

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
  - CLI flags > environment variables > profile > built-in defaults

Example config file:

```toml
[profiles.fireworks]
provider = "fireworks"
model = "accounts/fireworks/models/kimi-k2-instruct-0905"
temperature = 0.2
timeout = 15
retries = 2
retry_delay = 300
output = "json"
show_usage = true
system = "You are concise"
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

`--dry-run` works without API keys. Authorization is always redacted in dry-run output.

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
echo "2+2?" | cargo run --quiet -- ask
```

Equivalent explicit flags:

```bash
echo "2+2?" | cargo run --quiet -- ask --provider fireworks --model "accounts/fireworks/models/kimi-k2-instruct-0905"
```

OpenAI example:

```bash
export OPENAI_API_KEY="..."
echo "2+2?" | cargo run --quiet -- ask --provider openai --model "gpt-4o-mini"
```
