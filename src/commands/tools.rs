

use clap::{Args, ValueEnum};


#[derive(Debug, Args, Clone)]
pub struct ToolsArgs {
    #[arg(short = 'V', long = "version", action = clap::ArgAction::SetTrue)]
    pub version: bool,
}

#[derive(Debug, Args, Clone)]
pub struct ToolArgs {
    #[arg(short = 'V', long = "version", action = clap::ArgAction::SetTrue)]
    pub version: bool,
}

