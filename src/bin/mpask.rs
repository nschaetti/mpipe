use std::env;
use std::fs;
use std::io::{self, IsTerminal, Read};
use std::path::{Path, PathBuf};
use std::process;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use clap::{Parser, ValueEnum};
use mpipe::config::{self, ProfileConfig};
use mpipe::rchain::provider::{self, AskOptions, ChatMessage, Provider};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(
    name = "mpask",
    about = "Ask a question to an LLM provider",
    disable_version_flag = true
)]
struct Cli {
    #[arg(short = 'V', long = "version", action = clap::ArgAction::SetTrue)]
    version: bool,

    #[arg(long)]
    profile: Option<String>,

    #[arg(long, value_enum)]
    provider: Option<ProviderArg>,

    #[arg(long)]
    model: Option<String>,

    #[arg(long)]
    temperature: Option<f32>,

    #[arg(long = "max-tokens")]
    max_tokens: Option<u32>,

    #[arg(long)]
    timeout: Option<u64>,

    #[arg(long)]
    retries: Option<u32>,

    #[arg(long = "retry-delay")]
    retry_delay: Option<u64>,

    #[arg(long, value_enum)]
    output: Option<OutputFormat>,

    #[arg(long)]
    json: bool,

    #[arg(long)]
    show_usage: bool,

    #[arg(long)]
    verbose: bool,

    #[arg(long)]
    dry_run: bool,

    #[arg(long)]
    fail_on_empty: bool,

    #[arg(long)]
    save: Option<PathBuf>,

    #[arg(long)]
    system: Option<String>,

    #[arg(long)]
    prompt: Option<String>,

    #[arg(long)]
    postprompt: Option<String>,

    input: Option<String>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ProviderArg {
    Openai,
    Fireworks,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
}

impl OutputFormat {
    fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Json => "json",
        }
    }
}

#[derive(Debug, Serialize)]
struct JsonOutput {
    provider: String,
    model: String,
    answer: String,
    latency_ms: u128,
    request: JsonRequest,
    usage: Option<JsonUsage>,
}

#[derive(Debug, Serialize)]
struct JsonRequest {
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    timeout_secs: Option<u64>,
    retries: u32,
    retry_delay_ms: u64,
}

#[derive(Debug, Serialize)]
struct JsonUsage {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    total_tokens: Option<u32>,
}

#[derive(Debug, Serialize)]
struct DryRunOutput {
    dry_run: bool,
    provider: String,
    endpoint: String,
    model: String,
    messages: Vec<ChatMessage>,
    request: JsonRequest,
    output: String,
    show_usage: bool,
    authorization: String,
}

#[derive(Debug)]
struct PromptInput {
    text: String,
    source: PromptSource,
}

#[derive(Debug, Clone, Copy)]
enum PromptSource {
    Argument,
    Stdin,
}

impl PromptSource {
    fn as_str(self) -> &'static str {
        match self {
            Self::Argument => "argument",
            Self::Stdin => "stdin",
        }
    }
}

#[derive(Debug)]
struct UsageData {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    total_tokens: Option<u32>,
}

#[tokio::main]
async fn main() {
    if let Err(err) = run().await {
        eprintln!("{err}");
        process::exit(1);
    }
}

async fn run() -> Result<(), String> {
    let cli = Cli::parse();

    if cli.version {
        println!("{}", render_version());
        return Ok(());
    }

    let profile = resolve_profile(cli.profile.as_deref())?;

    let provider = resolve_provider(cli.provider, &profile)?;
    let model = resolve_model(cli.model, &profile)?;
    let temperature = resolve_temperature(cli.temperature, &profile)?;
    let max_tokens = resolve_max_tokens(cli.max_tokens, &profile)?;
    let timeout_secs = resolve_timeout(cli.timeout, &profile)?;
    let retries = resolve_retries(cli.retries, &profile)?;
    let retry_delay_ms = resolve_retry_delay(cli.retry_delay, &profile)?;
    let output_format = resolve_output_format(cli.output, cli.json, &profile)?;
    let show_usage = resolve_show_usage(cli.show_usage, &profile);
    let system = resolve_system(cli.system, &profile);

    let options = AskOptions {
        temperature,
        max_tokens,
        timeout_secs,
        retries,
        retry_delay_ms,
    };

    let main_prompt = resolve_prompt(cli.input)?;
    let prompt = compose_prompt(
        cli.prompt.as_deref(),
        &main_prompt.text,
        cli.postprompt.as_deref(),
    );
    let messages = build_messages(non_empty(system.as_deref()), &prompt);

    if cli.verbose {
        log_verbose(
            provider,
            &model,
            output_format,
            cli.dry_run,
            show_usage,
            main_prompt.source,
            &messages,
            &options,
        );
    }

    if cli.dry_run {
        let output = DryRunOutput {
            dry_run: true,
            provider: provider.as_str().to_string(),
            endpoint: provider::endpoint(provider).to_string(),
            model,
            messages,
            request: JsonRequest {
                temperature,
                max_tokens,
                timeout_secs,
                retries,
                retry_delay_ms,
            },
            output: output_format.as_str().to_string(),
            show_usage,
            authorization: "Bearer ***REDACTED***".to_string(),
        };
        let rendered = format!(
            "{}\n",
            serde_json::to_string(&output)
                .map_err(|err| format!("Failed to serialize dry-run output: {err}"))?
        );
        print!("{rendered}");
        if let Some(path) = &cli.save {
            write_output(path, &rendered)?;
        }
        if show_usage {
            eprintln!("usage: unavailable latency_ms=0 (dry-run)");
        }
        return Ok(());
    }

    let start = Instant::now();
    let response = provider::ask(provider, &model, &messages, options)
        .await
        .map_err(|err| err.to_string())?;
    let latency_ms = start.elapsed().as_millis();

    if cli.fail_on_empty && response.content.trim().is_empty() {
        return Err("Model response is empty and --fail-on-empty is enabled.".to_string());
    }

    let usage = response.usage.map(|usage| UsageData {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        total_tokens: usage.total_tokens,
    });

    if show_usage {
        print_usage(&usage, latency_ms);
    }

    let rendered = match output_format {
        OutputFormat::Text => response.content,
        OutputFormat::Json => {
            let output = JsonOutput {
                provider: provider.as_str().to_string(),
                model,
                answer: response.content,
                latency_ms,
                request: JsonRequest {
                    temperature,
                    max_tokens,
                    timeout_secs,
                    retries,
                    retry_delay_ms,
                },
                usage: usage.as_ref().and_then(json_usage),
            };
            format!(
                "{}\n",
                serde_json::to_string(&output)
                    .map_err(|err| format!("Failed to serialize JSON output: {err}"))?
            )
        }
    };

    print!("{rendered}");
    if let Some(path) = &cli.save {
        write_output(path, &rendered)?;
    }

    Ok(())
}

fn write_output(path: &Path, content: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "Failed to create output directory '{}': {err}",
                    parent.display()
                )
            })?;
        }
    }

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let tmp_name = format!(
        ".{}.tmp.{}.{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("mpask"),
        process::id(),
        now
    );
    let tmp_path = path.with_file_name(tmp_name);

    fs::write(&tmp_path, content)
        .map_err(|err| format!("Failed to write output file '{}': {err}", tmp_path.display()))?;

    if let Err(err) = fs::rename(&tmp_path, path) {
        let _ = fs::remove_file(&tmp_path);
        return Err(format!(
            "Failed to replace output file '{}': {err}",
            path.display()
        ));
    }

    Ok(())
}

fn resolve_profile(profile_name: Option<&str>) -> Result<ProfileConfig, String> {
    match profile_name {
        Some(name) => config::load_profile(name),
        None => Ok(ProfileConfig::default()),
    }
}

fn resolve_show_usage(cli_show_usage: bool, profile: &ProfileConfig) -> bool {
    if cli_show_usage {
        return true;
    }

    profile.show_usage.unwrap_or(false)
}

fn resolve_system(cli_system: Option<String>, profile: &ProfileConfig) -> Option<String> {
    if cli_system.is_some() {
        return cli_system;
    }

    profile.system.clone()
}

fn json_usage(usage: &UsageData) -> Option<JsonUsage> {
    if usage.prompt_tokens.is_none()
        && usage.completion_tokens.is_none()
        && usage.total_tokens.is_none()
    {
        return None;
    }

    Some(JsonUsage {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        total_tokens: usage.total_tokens,
    })
}

fn print_usage(usage: &Option<UsageData>, latency_ms: u128) {
    if let Some(usage) = usage {
        if let Some(usage) = json_usage(usage) {
            eprintln!(
                "usage: prompt_tokens={} completion_tokens={} total_tokens={} latency_ms={}",
                usage
                    .prompt_tokens
                    .map_or_else(|| "n/a".to_string(), |value| value.to_string()),
                usage
                    .completion_tokens
                    .map_or_else(|| "n/a".to_string(), |value| value.to_string()),
                usage
                    .total_tokens
                    .map_or_else(|| "n/a".to_string(), |value| value.to_string()),
                latency_ms
            );
            return;
        }
    }

    eprintln!("usage: unavailable latency_ms={latency_ms}");
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.and_then(|text| {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    })
}

fn build_messages(system: Option<&str>, prompt: &str) -> Vec<ChatMessage> {
    let mut messages = Vec::new();
    if let Some(system) = system {
        messages.push(ChatMessage::system(system));
    }
    messages.push(ChatMessage::user(prompt));
    messages
}

fn compose_prompt(preprompt: Option<&str>, main_prompt: &str, postprompt: Option<&str>) -> String {
    let mut parts = Vec::new();

    if let Some(pre) = preprompt {
        if !pre.trim().is_empty() {
            parts.push(pre.to_string());
        }
    }

    parts.push(main_prompt.to_string());

    if let Some(post) = postprompt {
        if !post.trim().is_empty() {
            parts.push(post.to_string());
        }
    }

    parts.join("\n\n")
}

fn resolve_provider(cli_provider: Option<ProviderArg>, profile: &ProfileConfig) -> Result<Provider, String> {
    if let Some(provider) = cli_provider {
        return Ok(match provider {
            ProviderArg::Openai => Provider::Openai,
            ProviderArg::Fireworks => Provider::Fireworks,
        });
    }

    if let Ok(raw) = env::var("MP_PROVIDER") {
        return parse_provider_value(&raw, "MP_PROVIDER");
    }

    if let Some(provider) = &profile.provider {
        return parse_provider_value(provider, "profile provider");
    }

    Ok(Provider::Openai)
}

fn parse_provider_value(raw: &str, source: &str) -> Result<Provider, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "openai" => Ok(Provider::Openai),
        "fireworks" => Ok(Provider::Fireworks),
        other => Err(format!(
            "Invalid {source} '{other}'. Supported values: openai, fireworks."
        )),
    }
}

fn resolve_model(cli_model: Option<String>, profile: &ProfileConfig) -> Result<String, String> {
    if let Some(model) = cli_model {
        let trimmed = model.trim();
        if !trimmed.is_empty() {
            return Ok(trimmed.to_string());
        }
    }

    if let Ok(model) = env::var("MP_MODEL") {
        let trimmed = model.trim();
        if !trimmed.is_empty() {
            return Ok(trimmed.to_string());
        }
    }

    if let Some(model) = &profile.model {
        let trimmed = model.trim();
        if !trimmed.is_empty() {
            return Ok(trimmed.to_string());
        }
    }

    Err("No model provided. Use --model or set MP_MODEL.".to_string())
}

fn resolve_temperature(
    cli_temperature: Option<f32>,
    profile: &ProfileConfig,
) -> Result<Option<f32>, String> {
    let temperature = if let Some(temperature) = cli_temperature {
        Some(temperature)
    } else if let Ok(raw) = env::var("MP_TEMPERATURE") {
        let parsed = raw.trim().parse::<f32>().map_err(|_| {
            format!("Invalid MP_TEMPERATURE '{raw}'. Must be a float in [0.0, 2.0].")
        })?;
        Some(parsed)
    } else {
        profile.temperature
    };

    if let Some(value) = temperature {
        if !(0.0..=2.0).contains(&value) {
            return Err(format!(
                "Invalid temperature {value}. Must be in [0.0, 2.0]."
            ));
        }
    }

    Ok(temperature)
}

fn resolve_max_tokens(
    cli_max_tokens: Option<u32>,
    profile: &ProfileConfig,
) -> Result<Option<u32>, String> {
    let max_tokens = if let Some(max_tokens) = cli_max_tokens {
        Some(max_tokens)
    } else if let Ok(raw) = env::var("MP_MAX_TOKENS") {
        let parsed = raw
            .trim()
            .parse::<u32>()
            .map_err(|_| format!("Invalid MP_MAX_TOKENS '{raw}'. Must be an integer > 0."))?;
        Some(parsed)
    } else {
        profile.max_tokens
    };

    if let Some(value) = max_tokens {
        if value == 0 {
            return Err("Invalid max tokens 0. Must be > 0.".to_string());
        }
    }

    Ok(max_tokens)
}

fn resolve_timeout(cli_timeout: Option<u64>, profile: &ProfileConfig) -> Result<Option<u64>, String> {
    let timeout = if let Some(timeout) = cli_timeout {
        Some(timeout)
    } else if let Ok(raw) = env::var("MP_TIMEOUT") {
        let parsed = raw
            .trim()
            .parse::<u64>()
            .map_err(|_| format!("Invalid MP_TIMEOUT '{raw}'. Must be an integer > 0."))?;
        Some(parsed)
    } else {
        profile.timeout
    };

    if let Some(value) = timeout {
        if value == 0 {
            return Err("Invalid timeout 0. Must be > 0 seconds.".to_string());
        }
    }

    Ok(timeout)
}

fn resolve_retries(cli_retries: Option<u32>, profile: &ProfileConfig) -> Result<u32, String> {
    if let Some(retries) = cli_retries {
        return Ok(retries);
    }

    if let Ok(raw) = env::var("MP_RETRIES") {
        return raw
            .trim()
            .parse::<u32>()
            .map_err(|_| format!("Invalid MP_RETRIES '{raw}'. Must be an integer >= 0."));
    }

    Ok(profile.retries.unwrap_or(0))
}

fn resolve_retry_delay(cli_retry_delay: Option<u64>, profile: &ProfileConfig) -> Result<u64, String> {
    let retry_delay = if let Some(retry_delay) = cli_retry_delay {
        retry_delay
    } else if let Ok(raw) = env::var("MP_RETRY_DELAY") {
        raw.trim()
            .parse::<u64>()
            .map_err(|_| format!("Invalid MP_RETRY_DELAY '{raw}'. Must be an integer > 0."))?
    } else {
        profile.retry_delay.unwrap_or(500)
    };

    if retry_delay == 0 {
        return Err("Invalid retry delay 0. Must be > 0 milliseconds.".to_string());
    }

    Ok(retry_delay)
}

fn resolve_output_format(
    output: Option<OutputFormat>,
    json: bool,
    profile: &ProfileConfig,
) -> Result<OutputFormat, String> {
    if json {
        return Ok(OutputFormat::Json);
    }

    if let Some(output) = output {
        return Ok(output);
    }

    if let Some(profile_output) = &profile.output {
        return parse_output_format(profile_output);
    }

    Ok(OutputFormat::Text)
}

fn parse_output_format(raw: &str) -> Result<OutputFormat, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "text" => Ok(OutputFormat::Text),
        "json" => Ok(OutputFormat::Json),
        other => Err(format!(
            "Invalid profile output '{other}'. Supported values: text, json."
        )),
    }
}

fn resolve_prompt(cli_prompt: Option<String>) -> Result<PromptInput, String> {
    if let Some(prompt) = cli_prompt {
        return Ok(PromptInput {
            text: prompt,
            source: PromptSource::Argument,
        });
    }

    if io::stdin().is_terminal() {
        return Err("No prompt provided. Pass an argument or pipe stdin.".to_string());
    }

    let mut buffer = String::new();
    io::stdin()
        .read_to_string(&mut buffer)
        .map_err(|err| format!("Failed to read stdin: {err}"))?;

    let text = buffer.trim().to_string();
    if text.is_empty() {
        return Err("Prompt is empty.".to_string());
    }

    Ok(PromptInput {
        text,
        source: PromptSource::Stdin,
    })
}

fn log_verbose(
    provider: Provider,
    model: &str,
    output_format: OutputFormat,
    dry_run: bool,
    show_usage: bool,
    prompt_source: PromptSource,
    messages: &[ChatMessage],
    options: &AskOptions,
) {
    let api_key_present = provider::is_api_key_present(provider);
    let total_chars: usize = messages.iter().map(|message| message.content.chars().count()).sum();

    eprintln!(
        "verbose: provider={} endpoint={} model={} output={} dry_run={} show_usage={} prompt_source={} messages={} chars={} api_key_present={}",
        provider.as_str(),
        provider::endpoint(provider),
        model,
        output_format.as_str(),
        dry_run,
        show_usage,
        prompt_source.as_str(),
        messages.len(),
        total_chars,
        api_key_present
    );
    eprintln!(
        "verbose: options temperature={} max_tokens={} timeout_secs={} retries={} retry_delay_ms={} backoff=exponential",
        options
            .temperature
            .map_or_else(|| "n/a".to_string(), |value| value.to_string()),
        options
            .max_tokens
            .map_or_else(|| "n/a".to_string(), |value| value.to_string()),
        options
            .timeout_secs
            .map_or_else(|| "n/a".to_string(), |value| value.to_string()),
        options.retries,
        options.retry_delay_ms
    );
}

fn render_version() -> String {
    let commit = option_env!("MP_GIT_SHA").unwrap_or("unknown");
    let built = option_env!("MP_BUILD_TS").unwrap_or("unknown");
    format!("{}\ncommit: {commit}\nbuilt: {built}", env!("CARGO_PKG_VERSION"))
}
