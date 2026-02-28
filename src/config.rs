use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ProfileConfig {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub system: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub timeout: Option<u64>,
    pub retries: Option<u32>,
    pub retry_delay: Option<u64>,
    pub output: Option<String>,
    pub show_usage: Option<bool>,
}

#[derive(Debug, Deserialize, Default)]
struct ConfigFile {
    profiles: Option<HashMap<String, ProfileConfig>>,
}

pub fn load_profile(name: &str) -> Result<ProfileConfig, String> {
    let path = config_path()?;
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("Failed to read config file '{}': {err}", path.display()))?;

    let config: ConfigFile = toml::from_str(&raw)
        .map_err(|err| format!("Failed to parse config file '{}': {err}", path.display()))?;

    let profiles = config.profiles.ok_or_else(|| {
        format!(
            "Config file '{}' does not contain a [profiles] section.",
            path.display()
        )
    })?;

    profiles.get(name).cloned().ok_or_else(|| {
        format!(
            "Profile '{}' not found in config file '{}'.",
            name,
            path.display()
        )
    })
}

fn config_path() -> Result<PathBuf, String> {
    if let Ok(path) = env::var("MP_CONFIG") {
        let trimmed = path.trim();
        if !trimmed.is_empty() {
            return Ok(PathBuf::from(trimmed));
        }
    }

    if let Ok(xdg) = env::var("XDG_CONFIG_HOME") {
        let trimmed = xdg.trim();
        if !trimmed.is_empty() {
            return Ok(PathBuf::from(trimmed).join("mpipe").join("config.toml"));
        }
    }

    let home = env::var("HOME").map_err(|_| {
        "Cannot resolve config path: set MP_CONFIG or HOME/XDG_CONFIG_HOME.".to_string()
    })?;
    Ok(PathBuf::from(home)
        .join(".config")
        .join("mpipe")
        .join("config.toml"))
}
