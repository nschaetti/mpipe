use assert_cmd::Command;
use predicates::str::contains;
use serde_json::Value;

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
        .env_remove("OPENAI_API_KEY")
        .env_remove("FIREWORKS_API_KEY");
    cmd
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
