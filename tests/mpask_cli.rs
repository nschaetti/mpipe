use assert_cmd::Command;
use predicates::prelude::PredicateBooleanExt;
use predicates::str::{contains, is_empty};
use serde_json::{json, Value};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

const FIREWORKS_TEST_MODEL: &str = "accounts/fireworks/models/kimi-k2-instruct-0905";

fn mpask_cmd() -> Command {
    let mut cmd = Command::new(assert_cmd::cargo::cargo_bin!("mpask"));
    cmd.env_remove("MP_PROVIDER")
        .env_remove("MP_MODEL")
        .env_remove("MP_TEMPERATURE")
        .env_remove("MP_MAX_TOKENS")
        .env_remove("MP_TIMEOUT")
        .env_remove("MP_RETRIES")
        .env_remove("MP_RETRY_DELAY")
        .env_remove("MP_CONFIG")
        .env_remove("OPENAI_API_KEY")
        .env_remove("FIREWORKS_API_KEY");
    cmd
}

fn mpipe_cmd() -> Command {
    let mut cmd = Command::new(assert_cmd::cargo::cargo_bin!("mpipe"));
    cmd.env_remove("MP_PROVIDER")
        .env_remove("MP_MODEL")
        .env_remove("MP_TEMPERATURE")
        .env_remove("MP_MAX_TOKENS")
        .env_remove("MP_TIMEOUT")
        .env_remove("MP_RETRIES")
        .env_remove("MP_RETRY_DELAY")
        .env_remove("MP_CONFIG")
        .env_remove("OPENAI_API_KEY")
        .env_remove("FIREWORKS_API_KEY");
    cmd
}

fn unique_temp_path(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!("mpask-test-{label}-{nanos}"))
}

fn parse_stdout_json(output: &[u8]) -> Value {
    let text = String::from_utf8(output.to_vec()).expect("stdout should be utf-8");
    serde_json::from_str(text.trim()).expect("stdout should contain valid JSON")
}

#[test]
fn dry_run_succeeds_without_api_key() {
    let assert = mpask_cmd()
        .args([
            "--provider",
            "fireworks",
            "--model",
            FIREWORKS_TEST_MODEL,
            "--dry-run",
            "2+2?",
        ])
        .assert()
        .success();

    let body = parse_stdout_json(&assert.get_output().stdout);
    assert_eq!(body["dry_run"], Value::Bool(true));
    assert_eq!(body["provider"], Value::String("fireworks".to_string()));
    assert_eq!(
        body["model"],
        Value::String(FIREWORKS_TEST_MODEL.to_string())
    );
}

#[test]
fn dry_run_show_usage_prints_unavailable() {
    mpask_cmd()
        .args([
            "--provider",
            "fireworks",
            "--model",
            FIREWORKS_TEST_MODEL,
            "--dry-run",
            "--show-usage",
            "2+2?",
        ])
        .assert()
        .success()
        .stderr(contains("usage: unavailable latency_ms=0 (dry-run)"));
}

#[test]
fn missing_model_returns_explicit_error() {
    mpask_cmd()
        .arg("hello")
        .assert()
        .failure()
        .stderr(contains("No model provided. Use --model or set MP_MODEL."));
}

#[test]
fn invalid_provider_from_env_returns_error() {
    mpask_cmd()
        .env("MP_PROVIDER", "bad")
        .args(["--model", "x", "hello"])
        .assert()
        .failure()
        .stderr(contains(
            "Invalid MP_PROVIDER 'bad'. Supported values: openai, fireworks.",
        ));
}

#[test]
fn argument_prompt_has_priority_over_stdin() {
    let assert = mpask_cmd()
        .args([
            "--provider",
            "openai",
            "--model",
            "gpt-4o-mini",
            "--dry-run",
            "argument prompt",
        ])
        .write_stdin("stdin prompt")
        .assert()
        .success();

    let body = parse_stdout_json(&assert.get_output().stdout);
    let messages = body["messages"]
        .as_array()
        .expect("messages should be an array");
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], Value::String("user".to_string()));
    assert_eq!(
        messages[0]["content"],
        Value::String("argument prompt".to_string())
    );
}

#[test]
fn json_flag_sets_json_output_mode() {
    let assert = mpask_cmd()
        .args([
            "--provider",
            "fireworks",
            "--model",
            FIREWORKS_TEST_MODEL,
            "--dry-run",
            "--json",
            "hello",
        ])
        .assert()
        .success();

    let body = parse_stdout_json(&assert.get_output().stdout);
    assert_eq!(body["output"], Value::String("json".to_string()));
}

#[test]
fn output_json_sets_json_output_mode() {
    let assert = mpask_cmd()
        .args([
            "--provider",
            "fireworks",
            "--model",
            FIREWORKS_TEST_MODEL,
            "--dry-run",
            "--output",
            "json",
            "hello",
        ])
        .assert()
        .success();

    let body = parse_stdout_json(&assert.get_output().stdout);
    assert_eq!(body["output"], Value::String("json".to_string()));
}

#[test]
fn profile_loads_provider_and_model_for_dry_run() {
    let config_path = unique_temp_path("config");
    fs::write(
        &config_path,
        "[profiles.fw]\nprovider = \"fireworks\"\nmodel = \"accounts/fireworks/models/kimi-k2-instruct-0905\"\n",
    )
    .expect("config should be writable");

    let assert = mpask_cmd()
        .env("MP_CONFIG", &config_path)
        .args(["--profile", "fw", "--dry-run", "hello"])
        .assert()
        .success();

    let body = parse_stdout_json(&assert.get_output().stdout);
    assert_eq!(body["provider"], Value::String("fireworks".to_string()));
    assert_eq!(
        body["model"],
        Value::String("accounts/fireworks/models/kimi-k2-instruct-0905".to_string())
    );
}

#[test]
fn profile_is_not_implicit_when_not_passed() {
    let config_path = unique_temp_path("config-no-implicit");
    fs::write(
        &config_path,
        "[profiles.default]\nprovider = \"fireworks\"\nmodel = \"accounts/fireworks/models/kimi-k2-instruct-0905\"\n",
    )
    .expect("config should be writable");

    mpask_cmd()
        .env("MP_CONFIG", &config_path)
        .arg("hello")
        .assert()
        .failure()
        .stderr(contains("No model provided. Use --model or set MP_MODEL."));
}

#[test]
fn save_writes_and_overwrites_output_file() {
    let output_path = unique_temp_path("save-output");

    mpask_cmd()
        .args([
            "--provider",
            "fireworks",
            "--model",
            FIREWORKS_TEST_MODEL,
            "--dry-run",
            "--save",
            output_path.to_string_lossy().as_ref(),
            "first",
        ])
        .assert()
        .success();

    let first = fs::read_to_string(&output_path).expect("first output file should exist");
    assert!(first.contains("\"content\":\"first\""));

    mpask_cmd()
        .args([
            "--provider",
            "fireworks",
            "--model",
            FIREWORKS_TEST_MODEL,
            "--dry-run",
            "--save",
            output_path.to_string_lossy().as_ref(),
            "second",
        ])
        .assert()
        .success();

    let second = fs::read_to_string(&output_path).expect("second output file should exist");
    assert!(second.contains("\"content\":\"second\""));
    assert!(!second.contains("\"content\":\"first\""));
}

#[test]
fn save_with_invalid_parent_path_returns_explicit_error() {
    let parent_file = unique_temp_path("save-invalid-parent");
    fs::write(&parent_file, "not a directory").expect("parent marker file should be writable");
    let output_path = parent_file.join("out.json");

    mpask_cmd()
        .args([
            "--provider",
            "fireworks",
            "--model",
            FIREWORKS_TEST_MODEL,
            "--dry-run",
            "--save",
            output_path.to_string_lossy().as_ref(),
            "hello",
        ])
        .assert()
        .failure()
        .stderr(contains("Failed to create output directory"));
}

#[test]
fn long_prompt_with_max_tokens_is_reflected_in_dry_run_request() {
    let prompt = "x".repeat(24_000);

    let assert = mpask_cmd()
        .args([
            "--provider",
            "fireworks",
            "--model",
            FIREWORKS_TEST_MODEL,
            "--dry-run",
            "--max-tokens",
            "128",
        ])
        .write_stdin(prompt.clone())
        .assert()
        .success();

    let body = parse_stdout_json(&assert.get_output().stdout);
    let messages = body["messages"]
        .as_array()
        .expect("messages should be an array");

    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], Value::String("user".to_string()));
    assert_eq!(messages[0]["content"], Value::String(prompt));
    assert_eq!(body["request"]["max_tokens"], Value::from(128));
}

#[test]
fn precedence_for_temperature_timeout_and_output_is_respected() {
    let config_path = unique_temp_path("precedence-more-options");
    fs::write(
        &config_path,
        "[profiles.fw]\nprovider = \"fireworks\"\nmodel = \"accounts/fireworks/models/kimi-k2-instruct-0905\"\ntemperature = 0.1\ntimeout = 7\noutput = \"json\"\n",
    )
    .expect("config should be writable");

    let env_over_profile = mpask_cmd()
        .env("MP_CONFIG", &config_path)
        .env("MP_TEMPERATURE", "0.6")
        .env("MP_TIMEOUT", "21")
        .args(["--profile", "fw", "--dry-run", "hello"])
        .assert()
        .success();

    let env_body = parse_stdout_json(&env_over_profile.get_output().stdout);
    assert_eq!(env_body["request"]["temperature"], json!(0.6));
    assert_eq!(env_body["request"]["timeout_secs"], Value::from(21));
    assert_eq!(env_body["output"], Value::String("json".to_string()));

    let cli_over_env = mpask_cmd()
        .env("MP_CONFIG", &config_path)
        .env("MP_TEMPERATURE", "0.6")
        .env("MP_TIMEOUT", "21")
        .args([
            "--profile",
            "fw",
            "--dry-run",
            "--temperature",
            "1.2",
            "--timeout",
            "33",
            "--output",
            "text",
            "hello",
        ])
        .assert()
        .success();

    let cli_body = parse_stdout_json(&cli_over_env.get_output().stdout);
    assert_eq!(cli_body["request"]["temperature"], json!(1.2));
    assert_eq!(cli_body["request"]["timeout_secs"], Value::from(33));
    assert_eq!(cli_body["output"], Value::String("text".to_string()));
}

#[test]
fn verbose_does_not_leak_api_key() {
    let secret = "fireworks-secret-value";

    mpask_cmd()
        .env("FIREWORKS_API_KEY", secret)
        .args([
            "--provider",
            "fireworks",
            "--model",
            FIREWORKS_TEST_MODEL,
            "--dry-run",
            "--verbose",
            "hello",
        ])
        .assert()
        .success()
        .stderr(contains("api_key_present=true").and(contains(secret).not()));
}

#[test]
fn json_flag_overrides_output_text() {
    let assert = mpask_cmd()
        .args([
            "--provider",
            "fireworks",
            "--model",
            FIREWORKS_TEST_MODEL,
            "--dry-run",
            "--output",
            "text",
            "--json",
            "hello",
        ])
        .assert()
        .success();

    let body = parse_stdout_json(&assert.get_output().stdout);
    assert_eq!(body["output"], Value::String("json".to_string()));
}

#[test]
fn profile_file_missing_returns_explicit_error() {
    let config_path = unique_temp_path("missing-config");

    mpask_cmd()
        .env("MP_CONFIG", &config_path)
        .args(["--profile", "fw", "hello"])
        .assert()
        .failure()
        .stderr(contains("Failed to read config file"));
}

#[test]
fn invalid_profile_toml_returns_parse_error() {
    let config_path = unique_temp_path("invalid-toml");
    fs::write(&config_path, "[profiles.bad\nprovider = \"openai\"")
        .expect("config should be writable");

    mpask_cmd()
        .env("MP_CONFIG", &config_path)
        .args(["--profile", "bad", "hello"])
        .assert()
        .failure()
        .stderr(contains("Failed to parse config file"));
}

#[test]
fn profile_not_found_returns_error() {
    let config_path = unique_temp_path("profile-not-found");
    fs::write(&config_path, "[profiles.fw]\nprovider = \"fireworks\"\n")
        .expect("config should be writable");

    mpask_cmd()
        .env("MP_CONFIG", &config_path)
        .args(["--profile", "missing", "hello"])
        .assert()
        .failure()
        .stderr(contains("Profile 'missing' not found"));
}

#[test]
fn invalid_profile_provider_returns_error() {
    let config_path = unique_temp_path("invalid-provider");
    fs::write(
        &config_path,
        "[profiles.bad]\nprovider = \"unknown\"\nmodel = \"m\"\n",
    )
    .expect("config should be writable");

    mpask_cmd()
        .env("MP_CONFIG", &config_path)
        .args(["--profile", "bad", "hello"])
        .assert()
        .failure()
        .stderr(contains("Invalid profile provider 'unknown'"));
}

#[test]
fn invalid_profile_output_returns_error() {
    let config_path = unique_temp_path("invalid-output");
    fs::write(
        &config_path,
        "[profiles.bad]\nprovider = \"openai\"\nmodel = \"m\"\noutput = \"yaml\"\n",
    )
    .expect("config should be writable");

    mpask_cmd()
        .env("MP_CONFIG", &config_path)
        .args(["--profile", "bad", "hello"])
        .assert()
        .failure()
        .stderr(contains("Invalid profile output 'yaml'"));
}

#[test]
fn profile_env_and_cli_precedence_is_respected() {
    let config_path = unique_temp_path("precedence");
    fs::write(
        &config_path,
        "[profiles.fw]\nprovider = \"fireworks\"\nmodel = \"profile-model\"\n",
    )
    .expect("config should be writable");

    let assert = mpask_cmd()
        .env("MP_CONFIG", &config_path)
        .env("MP_PROVIDER", "openai")
        .env("MP_MODEL", "env-model")
        .args([
            "--profile",
            "fw",
            "--provider",
            "fireworks",
            "--model",
            "cli-model",
            "--dry-run",
            "hello",
        ])
        .assert()
        .success();

    let body = parse_stdout_json(&assert.get_output().stdout);
    assert_eq!(body["provider"], Value::String("fireworks".to_string()));
    assert_eq!(body["model"], Value::String("cli-model".to_string()));
}

#[test]
fn version_prints_build_metadata() {
    mpask_cmd()
        .arg("--version")
        .assert()
        .success()
        .stdout(contains("commit:").and(contains("built:")));
}

#[test]
fn mpipe_ask_dry_run_matches_mpask_output_shape() {
    let assert = mpipe_cmd()
        .args([
            "ask",
            "--provider",
            "fireworks",
            "--model",
            FIREWORKS_TEST_MODEL,
            "--dry-run",
            "hello",
        ])
        .assert()
        .success();

    let body = parse_stdout_json(&assert.get_output().stdout);
    assert_eq!(body["provider"], Value::String("fireworks".to_string()));
    assert_eq!(body["output"], Value::String("text".to_string()));
}

#[test]
fn mpipe_ask_version_prints_metadata() {
    mpipe_cmd()
        .args(["ask", "--version"])
        .assert()
        .success()
        .stdout(contains("commit:").and(contains("built:")));
}

#[test]
fn quiet_suppresses_show_usage_on_stderr() {
    mpask_cmd()
        .args([
            "--provider",
            "fireworks",
            "--model",
            FIREWORKS_TEST_MODEL,
            "--dry-run",
            "--show-usage",
            "--quiet",
            "hello",
        ])
        .assert()
        .success()
        .stderr(is_empty());
}

#[test]
fn quiet_suppresses_verbose_logs_on_stderr() {
    mpask_cmd()
        .args([
            "--provider",
            "fireworks",
            "--model",
            FIREWORKS_TEST_MODEL,
            "--dry-run",
            "--verbose",
            "--quiet",
            "hello",
        ])
        .assert()
        .success()
        .stderr(is_empty());
}

#[test]
fn quiet_keeps_fatal_errors_visible() {
    mpask_cmd()
        .args(["--quiet", "hello"])
        .assert()
        .failure()
        .stderr(contains("No model provided. Use --model or set MP_MODEL."));
}

#[test]
fn mpipe_completion_bash_outputs_script() {
    mpipe_cmd()
        .args(["completion", "bash"])
        .assert()
        .success()
        .stdout(contains("_mpipe").and(contains("complete")));
}

#[test]
fn mpipe_ask_quiet_suppresses_show_usage_on_stderr() {
    mpipe_cmd()
        .args([
            "ask",
            "--provider",
            "fireworks",
            "--model",
            FIREWORKS_TEST_MODEL,
            "--dry-run",
            "--show-usage",
            "--quiet",
            "hello",
        ])
        .assert()
        .success()
        .stderr(is_empty());
}

#[test]
fn mpipe_ask_quiet_suppresses_verbose_logs_on_stderr() {
    mpipe_cmd()
        .args([
            "ask",
            "--provider",
            "fireworks",
            "--model",
            FIREWORKS_TEST_MODEL,
            "--dry-run",
            "--verbose",
            "--quiet",
            "hello",
        ])
        .assert()
        .success()
        .stderr(is_empty());
}

#[test]
fn mpipe_ask_quiet_keeps_fatal_errors_visible() {
    mpipe_cmd()
        .args(["ask", "--quiet", "hello"])
        .assert()
        .failure()
        .stderr(contains("No model provided. Use --model or set MP_MODEL."));
}

#[test]
fn mpipe_ask_help_includes_examples() {
    mpipe_cmd()
        .args(["ask", "--help"])
        .assert()
        .success()
        .stdout(contains("Examples:").and(contains("--dry-run --json")));
}

#[test]
fn mpipe_help_mentions_completion_command() {
    mpipe_cmd()
        .arg("--help")
        .assert()
        .success()
        .stdout(contains("completion").and(contains("Generate shell completion script")));
}

#[test]
fn mpipe_completion_fish_outputs_script() {
    mpipe_cmd()
        .args(["completion", "fish"])
        .assert()
        .success()
        .stdout(contains("complete -c mpipe"));
}
