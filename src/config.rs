use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

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

#[derive(Debug, Clone, Deserialize, Default)]
struct ProviderDefaultsConfig {
    model: Option<String>,
    system: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    timeout: Option<u64>,
    retries: Option<u32>,
    retry_delay: Option<u64>,
    output: Option<String>,
    show_usage: Option<bool>,
}

#[derive(Debug, Deserialize, Default)]
struct ProviderSectionConfig {
    defaults: Option<ProviderDefaultsConfig>,
}

#[derive(Debug, Deserialize, Default)]
struct ConfigFile {
    profiles: Option<HashMap<String, ProfileConfig>>,
    providers: Option<HashMap<String, ProviderSectionConfig>>,
}

pub fn load_profile(name: &str) -> Result<ProfileConfig, String> {
    let (path, config) = load_and_validate_config_file()?;
    let profile = profile_from_config(&config, &path, name)?.clone();

    let provider_defaults = profile
        .provider
        .as_deref()
        .and_then(normalized_provider_value)
        .and_then(|provider| provider_defaults_for(&config, provider));

    Ok(merge_provider_defaults(
        provider_defaults.as_ref(),
        &profile,
    ))
}

pub fn validate_config(profile_name: Option<&str>) -> Result<PathBuf, String> {
    let (path, config) = load_and_validate_config_file()?;

    if let Some(name) = profile_name {
        let _ = profile_from_config(&config, &path, name)?;
    }

    Ok(path)
}

fn load_and_validate_config_file() -> Result<(PathBuf, ConfigFile), String> {
    let path = config_path()?;
    let path_display = path.display();
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("Failed to read config file '{}': {err}", path_display))?;

    let config: ConfigFile = toml::from_str(&raw)
        .map_err(|err| format!("Failed to parse config file '{}': {err}", path_display))?;

    validate_config_file(&config, &path)?;
    Ok((path, config))
}

fn profile_from_config<'a>(
    config: &'a ConfigFile,
    path: &Path,
    name: &str,
) -> Result<&'a ProfileConfig, String> {
    let profiles = config.profiles.as_ref().ok_or_else(|| {
        format!(
            "Config file '{}' does not contain a [profiles] section.",
            path.display()
        )
    })?;

    profiles.get(name).ok_or_else(|| {
        format!(
            "Profile '{}' not found in config file '{}'.",
            name,
            path.display()
        )
    })
}

fn provider_defaults_for(config: &ConfigFile, provider: &str) -> Option<ProviderDefaultsConfig> {
    let providers = config.providers.as_ref()?;

    providers.iter().find_map(|(name, section)| {
        let normalized = normalized_provider_value(name)?;
        if normalized == provider {
            section.defaults.clone()
        } else {
            None
        }
    })
}

fn merge_provider_defaults(
    defaults: Option<&ProviderDefaultsConfig>,
    profile: &ProfileConfig,
) -> ProfileConfig {
    let defaults = defaults.cloned().unwrap_or_default();

    ProfileConfig {
        provider: profile.provider.clone(),
        model: profile.model.clone().or(defaults.model),
        system: profile.system.clone().or(defaults.system),
        temperature: profile.temperature.or(defaults.temperature),
        max_tokens: profile.max_tokens.or(defaults.max_tokens),
        timeout: profile.timeout.or(defaults.timeout),
        retries: profile.retries.or(defaults.retries),
        retry_delay: profile.retry_delay.or(defaults.retry_delay),
        output: profile.output.clone().or(defaults.output),
        show_usage: profile.show_usage.or(defaults.show_usage),
    }
}

fn validate_config_file(config: &ConfigFile, path: &Path) -> Result<(), String> {
    if let Some(providers) = &config.providers {
        for (provider_name, provider_section) in providers {
            let Some(provider) = normalized_provider_value(provider_name) else {
                return Err(format!(
                    "Invalid provider section 'providers.{provider_name}' in config file '{}'. Supported values: openai, fireworks.",
                    path.display()
                ));
            };

            if let Some(defaults) = &provider_section.defaults {
                validate_profile_fields(path, &format!("providers.{provider}.defaults"), defaults)?;
            }
        }
    }

    if let Some(profiles) = &config.profiles {
        for (name, profile) in profiles {
            validate_profile(path, name, profile)?;
        }
    }

    Ok(())
}

fn validate_profile(path: &Path, name: &str, profile: &ProfileConfig) -> Result<(), String> {
    if let Some(provider_raw) = &profile.provider {
        let provider = provider_raw.trim().to_ascii_lowercase();
        if normalized_provider_value(provider_raw).is_none() {
            return Err(format!(
                "Invalid profile provider '{provider}'. Supported values: openai, fireworks. (at 'profiles.{name}.provider' in '{}')",
                path.display()
            ));
        }
    }

    validate_profile_fields(path, &format!("profiles.{name}"), profile)
}

trait ValidatableProfileFields {
    fn temperature(&self) -> Option<f32>;
    fn max_tokens(&self) -> Option<u32>;
    fn timeout(&self) -> Option<u64>;
    fn retry_delay(&self) -> Option<u64>;
    fn output(&self) -> Option<&str>;
}

impl ValidatableProfileFields for ProfileConfig {
    fn temperature(&self) -> Option<f32> {
        self.temperature
    }

    fn max_tokens(&self) -> Option<u32> {
        self.max_tokens
    }

    fn timeout(&self) -> Option<u64> {
        self.timeout
    }

    fn retry_delay(&self) -> Option<u64> {
        self.retry_delay
    }

    fn output(&self) -> Option<&str> {
        self.output.as_deref()
    }
}

impl ValidatableProfileFields for ProviderDefaultsConfig {
    fn temperature(&self) -> Option<f32> {
        self.temperature
    }

    fn max_tokens(&self) -> Option<u32> {
        self.max_tokens
    }

    fn timeout(&self) -> Option<u64> {
        self.timeout
    }

    fn retry_delay(&self) -> Option<u64> {
        self.retry_delay
    }

    fn output(&self) -> Option<&str> {
        self.output.as_deref()
    }
}

fn validate_profile_fields(
    path: &Path,
    section_path: &str,
    fields: &dyn ValidatableProfileFields,
) -> Result<(), String> {
    if let Some(value) = fields.temperature() {
        if !(0.0..=2.0).contains(&value) {
            return Err(format!(
                "Invalid value at '{section_path}.temperature' in config file '{}': {value} (must be in [0.0, 2.0]).",
                path.display()
            ));
        }
    }

    if let Some(value) = fields.max_tokens() {
        if value == 0 {
            return Err(format!(
                "Invalid value at '{section_path}.max_tokens' in config file '{}': 0 (must be > 0).",
                path.display()
            ));
        }
    }

    if let Some(value) = fields.timeout() {
        if value == 0 {
            return Err(format!(
                "Invalid value at '{section_path}.timeout' in config file '{}': 0 (must be > 0).",
                path.display()
            ));
        }
    }

    if let Some(value) = fields.retry_delay() {
        if value == 0 {
            return Err(format!(
                "Invalid value at '{section_path}.retry_delay' in config file '{}': 0 (must be > 0).",
                path.display()
            ));
        }
    }

    if let Some(raw_output) = fields.output() {
        let output = raw_output.trim().to_ascii_lowercase();
        if output != "text" && output != "json" {
            if let Some(name) = section_path.strip_prefix("profiles.") {
                return Err(format!(
                    "Invalid profile output '{output}'. Supported values: text, json. (at 'profiles.{name}.output' in '{}')",
                    path.display()
                ));
            }

            return Err(format!(
                "Invalid value at '{section_path}.output' in config file '{}': '{output}' (supported values: text, json).",
                path.display()
            ));
        }
    }

    Ok(())
}

fn normalized_provider_value(raw: &str) -> Option<&'static str> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "openai" => Some("openai"),
        "fireworks" => Some("fireworks"),
        _ => None,
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_path(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("mpipe-config-test-{label}-{nanos}.toml"))
    }

    #[test]
    fn load_profile_merges_provider_defaults() {
        let config_path = unique_temp_path("merge-provider-defaults");
        fs::write(
            &config_path,
            "[providers.fireworks.defaults]\noutput = \"json\"\ntimeout = 15\nretries = 2\n\n[profiles.fast]\nprovider = \"fireworks\"\nmodel = \"profile-model\"\ntimeout = 7\n",
        )
        .expect("config should be writable");

        let profile = {
            unsafe {
                env::set_var("MP_CONFIG", &config_path);
            }
            let loaded = load_profile("fast");
            unsafe {
                env::remove_var("MP_CONFIG");
            }
            loaded.expect("profile should load")
        };

        assert_eq!(profile.provider.as_deref(), Some("fireworks"));
        assert_eq!(profile.model.as_deref(), Some("profile-model"));
        assert_eq!(profile.output.as_deref(), Some("json"));
        assert_eq!(profile.retries, Some(2));
        assert_eq!(profile.timeout, Some(7));
    }

    #[test]
    fn load_profile_rejects_invalid_provider_defaults_section() {
        let config_path = unique_temp_path("invalid-provider-defaults-section");
        fs::write(
            &config_path,
            "[providers.unknown.defaults]\ntimeout = 10\n\n[profiles.fast]\nprovider = \"openai\"\nmodel = \"gpt-4o-mini\"\n",
        )
        .expect("config should be writable");

        let err = {
            unsafe {
                env::set_var("MP_CONFIG", &config_path);
            }
            let result = load_profile("fast").expect_err("should fail validation");
            unsafe {
                env::remove_var("MP_CONFIG");
            }
            result
        };

        assert!(err.contains("Invalid provider section 'providers.unknown'"));
    }

    #[test]
    fn load_profile_rejects_invalid_profile_temperature() {
        let config_path = unique_temp_path("invalid-profile-temperature");
        fs::write(
            &config_path,
            "[profiles.bad]\nprovider = \"openai\"\nmodel = \"gpt-4o-mini\"\ntemperature = 3.1\n",
        )
        .expect("config should be writable");

        let err = {
            unsafe {
                env::set_var("MP_CONFIG", &config_path);
            }
            let result = load_profile("bad").expect_err("should fail validation");
            unsafe {
                env::remove_var("MP_CONFIG");
            }
            result
        };

        assert!(err.contains("profiles.bad.temperature"));
        assert!(err.contains("must be in [0.0, 2.0]"));
    }

    #[test]
    fn load_profile_rejects_invalid_provider_defaults_output() {
        let config_path = unique_temp_path("invalid-provider-default-output");
        fs::write(
            &config_path,
            "[providers.openai.defaults]\noutput = \"yaml\"\n\n[profiles.good]\nprovider = \"openai\"\nmodel = \"gpt-4o-mini\"\n",
        )
        .expect("config should be writable");

        let err = {
            unsafe {
                env::set_var("MP_CONFIG", &config_path);
            }
            let result = load_profile("good").expect_err("should fail validation");
            unsafe {
                env::remove_var("MP_CONFIG");
            }
            result
        };

        assert!(err.contains("providers.openai.defaults.output"));
        assert!(err.contains("supported values: text, json"));
    }

    #[test]
    fn validate_config_accepts_valid_file_without_profiles() {
        let config_path = unique_temp_path("validate-without-profiles");
        fs::write(
            &config_path,
            "[providers.openai.defaults]\ntimeout = 30\noutput = \"text\"\n",
        )
        .expect("config should be writable");

        let resolved = {
            unsafe {
                env::set_var("MP_CONFIG", &config_path);
            }
            let result = validate_config(None).expect("config should validate");
            unsafe {
                env::remove_var("MP_CONFIG");
            }
            result
        };

        assert_eq!(resolved, config_path);
    }

    #[test]
    fn validate_config_with_profile_checks_profile_presence() {
        let config_path = unique_temp_path("validate-profile-presence");
        fs::write(
            &config_path,
            "[profiles.default]\nprovider = \"openai\"\nmodel = \"gpt-4o-mini\"\n",
        )
        .expect("config should be writable");

        let err = {
            unsafe {
                env::set_var("MP_CONFIG", &config_path);
            }
            let result = validate_config(Some("missing")).expect_err("missing profile should fail");
            unsafe {
                env::remove_var("MP_CONFIG");
            }
            result
        };

        assert!(err.contains("Profile 'missing' not found"));
    }
}
