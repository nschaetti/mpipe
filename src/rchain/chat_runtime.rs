use std::time::Duration;

use reqwest::StatusCode;
use serde::Serialize;
use tokio::time::sleep;

#[derive(Debug, Clone, Copy)]
pub(crate) struct RetryConfig {
    pub timeout_secs: Option<u64>,
    pub retries: u32,
    pub retry_delay_ms: u64,
}

#[derive(Debug)]
pub(crate) enum RequestFailure {
    Request(reqwest::Error),
    Api { status: StatusCode, body: String },
}

pub(crate) async fn send_chat_request_with_retry<T: Serialize + ?Sized>(
    client: &reqwest::Client,
    url: &str,
    api_key: &str,
    payload: &T,
    config: RetryConfig,
) -> Result<reqwest::Response, RequestFailure> {
    let max_attempts = config.retries.saturating_add(1);
    let mut attempt = 0;

    loop {
        let mut request = client.post(url).bearer_auth(api_key).json(payload);

        if let Some(timeout_secs) = config.timeout_secs {
            request = request.timeout(Duration::from_secs(timeout_secs));
        }

        match request.send().await {
            Ok(response) => {
                if response.status().is_success() {
                    return Ok(response);
                }

                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                let can_retry = is_retryable_status(status) && attempt + 1 < max_attempts;

                if can_retry {
                    sleep(retry_delay(attempt, config.retry_delay_ms)).await;
                    attempt += 1;
                    continue;
                }

                return Err(RequestFailure::Api { status, body });
            }
            Err(source) => {
                let can_retry = is_retryable_request_error(&source) && attempt + 1 < max_attempts;

                if can_retry {
                    sleep(retry_delay(attempt, config.retry_delay_ms)).await;
                    attempt += 1;
                    continue;
                }

                return Err(RequestFailure::Request(source));
            }
        }
    }
}

fn is_retryable_status(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}

fn is_retryable_request_error(err: &reqwest::Error) -> bool {
    err.is_timeout() || err.is_connect() || err.is_request()
}

fn retry_delay(attempt: u32, base_ms: u64) -> Duration {
    let factor = 1u64.checked_shl(attempt).unwrap_or(u64::MAX);
    let delay_ms = base_ms.saturating_mul(factor).min(30_000);
    Duration::from_millis(delay_ms)
}

#[cfg(test)]
mod tests {
    use super::{is_retryable_status, retry_delay};
    use reqwest::StatusCode;
    use std::time::Duration;

    #[test]
    fn retry_delay_uses_exponential_backoff() {
        assert_eq!(retry_delay(0, 200), Duration::from_millis(200));
        assert_eq!(retry_delay(1, 200), Duration::from_millis(400));
        assert_eq!(retry_delay(2, 200), Duration::from_millis(800));
    }

    #[test]
    fn retry_delay_caps_at_thirty_seconds() {
        assert_eq!(retry_delay(10, 500), Duration::from_millis(30_000));
        assert_eq!(retry_delay(30, 5_000), Duration::from_millis(30_000));
    }

    #[test]
    fn retryable_statuses_match_policy() {
        assert!(is_retryable_status(StatusCode::TOO_MANY_REQUESTS));
        assert!(is_retryable_status(StatusCode::INTERNAL_SERVER_ERROR));
        assert!(is_retryable_status(StatusCode::BAD_GATEWAY));

        assert!(!is_retryable_status(StatusCode::BAD_REQUEST));
        assert!(!is_retryable_status(StatusCode::UNAUTHORIZED));
        assert!(!is_retryable_status(StatusCode::NOT_FOUND));
    }
}
