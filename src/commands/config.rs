use clap::{Args, Subcommand};

use crate::config;

#[derive(Debug, Args, Clone)]
pub struct ConfigArgs {
    #[command(subcommand)]
    command: ConfigSubcommand,
}

#[derive(Debug, Subcommand, Clone)]
enum ConfigSubcommand {
    Check {
        #[arg(long)]
        profile: Option<String>,
    },
}

pub fn run(args: ConfigArgs) -> Result<(), String> {
    match args.command {
        ConfigSubcommand::Check { profile } => {
            let path = config::validate_config(profile.as_deref())?;
            println!("config OK: {}", path.display());
            Ok(())
        }
    }
}
