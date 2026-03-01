use clap::{Args, ValueEnum};
use serde::Serialize;

#[derive(Debug, Args, Clone)]
pub struct ModelsArgs {
    #[arg(long, value_enum)]
    provider: Option<ProviderArg>,

    #[arg(long)]
    json: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ProviderArg {
    Openai,
    Fireworks,
}

impl ProviderArg {
    fn as_str(self) -> &'static str {
        match self {
            Self::Openai => "openai",
            Self::Fireworks => "fireworks",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ModelEntry {
    provider: &'static str,
    id: &'static str,
    recommended: bool,
}

#[derive(Debug, Serialize)]
struct JsonModelEntry {
    provider: &'static str,
    id: &'static str,
    source: &'static str,
    recommended: bool,
}

const MODEL_CATALOG: &[ModelEntry] = &[
    ModelEntry {
        provider: "fireworks",
        id: "accounts/fireworks/models/kimi-k2-instruct-0905",
        recommended: true,
    },
    ModelEntry {
        provider: "openai",
        id: "gpt-4o-mini",
        recommended: true,
    },
];

pub fn run(args: ModelsArgs) -> Result<(), String> {
    let mut models = MODEL_CATALOG.to_vec();
    models.sort_by_key(|entry| (entry.provider, entry.id));

    if let Some(provider) = args.provider {
        let provider = provider.as_str();
        models.retain(|entry| entry.provider == provider);
    }

    if args.json {
        let payload = models
            .into_iter()
            .map(|entry| JsonModelEntry {
                provider: entry.provider,
                id: entry.id,
                source: "local",
                recommended: entry.recommended,
            })
            .collect::<Vec<_>>();
        let rendered = serde_json::to_string(&payload)
            .map_err(|err| format!("Failed to serialize models output: {err}"))?;
        println!("{rendered}");
        return Ok(());
    }

    for entry in models {
        println!("{}\t{}", entry.provider, entry.id);
    }

    Ok(())
}
