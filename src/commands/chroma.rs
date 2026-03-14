use std::env;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use chromadb::client::{ChromaAuthMethod, ChromaClient, ChromaClientOptions};
use clap::Args;

const DEFAULT_CHROMA_LOCAL_HOST: &str = "127.0.0.1";
const DEFAULT_CHROMA_PORT: u16 = 8000;
const LOCAL_CHROMA_START_TIMEOUT: Duration = Duration::from_secs(10);
const LOCAL_CHROMA_POLL_INTERVAL: Duration = Duration::from_millis(250);

#[derive(Debug, Args, Clone, Default)]
pub struct ChromaConnectArgs {
    #[arg(long = "chroma-url")]
    pub chroma_url: Option<String>,

    #[arg(long = "chroma-host")]
    pub chroma_host: Option<String>,

    #[arg(long = "chroma-port")]
    pub chroma_port: Option<u16>,

    #[arg(long = "chroma-scheme")]
    pub chroma_scheme: Option<String>,

    #[arg(long = "chroma-path")]
    pub chroma_path: Option<PathBuf>,
}

#[derive(Debug, Clone)]
enum ChromaConnection {
    Remote {
        url: Option<String>,
    },
    LocalPersistent {
        url: String,
        host: String,
        port: u16,
        path: PathBuf,
    },
}

impl ChromaConnection {
    fn url(&self) -> Option<&str> {
        match self {
            Self::Remote { url } => url.as_deref(),
            Self::LocalPersistent { url, .. } => Some(url.as_str()),
        }
    }
}

pub struct LocalChromaGuard {
    child: Option<Child>,
}

impl Drop for LocalChromaGuard {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

pub async fn connect(
    args: &ChromaConnectArgs,
) -> Result<(ChromaClient, Option<LocalChromaGuard>), String> {
    let connection = resolve_chroma_connection(args)?;
    let guard = start_local_chroma_if_needed(&connection).await?;
    let client = ChromaClient::new(ChromaClientOptions {
        url: connection.url().map(str::to_string),
        auth: ChromaAuthMethod::None,
        database: "default_database".to_string(),
    })
    .await
    .map_err(|err| format!("Failed to connect to ChromaDB: {err}"))?;
    Ok((client, guard))
}

fn resolve_chroma_connection(args: &ChromaConnectArgs) -> Result<ChromaConnection, String> {
    let chroma_path = resolve_chroma_path(args);
    if let Some(path) = chroma_path {
        if args.chroma_url.is_some() {
            return Err("--chroma-url cannot be used with --chroma-path/CHROMA_PATH.".to_string());
        }

        let scheme = args
            .chroma_scheme
            .clone()
            .or_else(|| env::var("CHROMA_SCHEME").ok())
            .unwrap_or_else(|| "http".to_string());
        if !scheme.eq_ignore_ascii_case("http") {
            return Err(format!(
                "--chroma-path requires --chroma-scheme=http (got '{scheme}')."
            ));
        }

        let host = args
            .chroma_host
            .clone()
            .or_else(|| env::var("CHROMA_HOST").ok())
            .unwrap_or_else(|| DEFAULT_CHROMA_LOCAL_HOST.to_string());
        if host.contains("://") {
            return Err(
                "--chroma-host must be a hostname when using --chroma-path (no scheme)."
                    .to_string(),
            );
        }

        let port = resolve_chroma_port(args)?;
        let url = format!("http://{host}:{port}");

        return Ok(ChromaConnection::LocalPersistent {
            url,
            host,
            port,
            path,
        });
    }

    Ok(ChromaConnection::Remote {
        url: resolve_chroma_url(args)?,
    })
}

fn resolve_chroma_url(args: &ChromaConnectArgs) -> Result<Option<String>, String> {
    if let Some(url) = &args.chroma_url {
        let trimmed = url.trim();
        if trimmed.is_empty() {
            return Err("--chroma-url cannot be empty".to_string());
        }
        return Ok(Some(trimmed.to_string()));
    }

    let scheme = args
        .chroma_scheme
        .clone()
        .or_else(|| env::var("CHROMA_SCHEME").ok());
    let host = args
        .chroma_host
        .clone()
        .or_else(|| env::var("CHROMA_HOST").ok());
    let port = resolve_chroma_port_opt(args)?;

    if scheme.is_none() && host.is_none() && port.is_none() {
        return Ok(None);
    }

    if let Some(host) = host.clone()
        && host.contains("://")
        && scheme.is_none()
        && port.is_none()
    {
        return Ok(Some(host));
    }

    let scheme = scheme.unwrap_or_else(|| "http".to_string());
    let host = host.unwrap_or_else(|| "localhost".to_string());
    let port = port.unwrap_or(DEFAULT_CHROMA_PORT);
    Ok(Some(format!("{scheme}://{host}:{port}")))
}

fn resolve_chroma_port(args: &ChromaConnectArgs) -> Result<u16, String> {
    Ok(resolve_chroma_port_opt(args)?.unwrap_or(DEFAULT_CHROMA_PORT))
}

fn resolve_chroma_port_opt(args: &ChromaConnectArgs) -> Result<Option<u16>, String> {
    if let Some(port) = args.chroma_port {
        return Ok(Some(port));
    }

    match env::var("CHROMA_PORT") {
        Ok(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                return Ok(None);
            }
            let parsed = trimmed.parse::<u16>().map_err(|_| {
                format!("Invalid CHROMA_PORT '{trimmed}': expected integer 0-65535")
            })?;
            Ok(Some(parsed))
        }
        Err(_) => Ok(None),
    }
}

fn resolve_chroma_path(args: &ChromaConnectArgs) -> Option<PathBuf> {
    if let Some(path) = &args.chroma_path {
        return Some(path.clone());
    }

    env::var("CHROMA_PATH").ok().and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(PathBuf::from(trimmed))
        }
    })
}

async fn start_local_chroma_if_needed(
    connection: &ChromaConnection,
) -> Result<Option<LocalChromaGuard>, String> {
    let ChromaConnection::LocalPersistent {
        url,
        host,
        port,
        path,
    } = connection
    else {
        return Ok(None);
    };

    fs::create_dir_all(path).map_err(|err| {
        format!(
            "Failed to create Chroma persistence directory '{}': {err}",
            path.display()
        )
    })?;

    let mut child = Command::new("chroma")
        .arg("run")
        .arg("--path")
        .arg(path)
        .arg("--host")
        .arg(host)
        .arg("--port")
        .arg(port.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|err| {
            if err.kind() == io::ErrorKind::NotFound {
                "Failed to start local ChromaDB: 'chroma' command not found. Install Chroma or run a server manually and use --chroma-url."
                    .to_string()
            } else {
                format!("Failed to start local ChromaDB process: {err}")
            }
        })?;

    wait_for_chroma_ready(url).await.map_err(|err| {
        let _ = child.kill();
        let _ = child.wait();
        err
    })?;

    eprintln!(
        "mpipe: started local ChromaDB at {url} (path: {})",
        path.display()
    );

    Ok(Some(LocalChromaGuard { child: Some(child) }))
}

async fn wait_for_chroma_ready(url: &str) -> Result<(), String> {
    let deadline = std::time::Instant::now() + LOCAL_CHROMA_START_TIMEOUT;

    loop {
        let attempt_error = match ChromaClient::new(ChromaClientOptions {
            url: Some(url.to_string()),
            auth: ChromaAuthMethod::None,
            database: "default_database".to_string(),
        })
        .await
        {
            Ok(client) => match client.heartbeat().await {
                Ok(_) => return Ok(()),
                Err(err) => format!("heartbeat failed: {err}"),
            },
            Err(err) => format!("connection failed: {err}"),
        };

        if std::time::Instant::now() >= deadline {
            return Err(format!(
                "Local ChromaDB did not become ready at {url} within {}s ({}).",
                LOCAL_CHROMA_START_TIMEOUT.as_secs(),
                attempt_error
            ));
        }

        tokio::time::sleep(LOCAL_CHROMA_POLL_INTERVAL).await;
    }
}
