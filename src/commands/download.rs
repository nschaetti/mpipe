use std::path::PathBuf;
use std::process::Command;

use clap::Args;

#[derive(Debug, Args, Clone)]
pub struct DownloadArgs {
    #[arg(help = "URL of the video to download")]
    pub url: String,

    #[arg(short = 'o', long = "output", help = "Output file path")]
    pub output: PathBuf,

    #[arg(long = "audio-only", help = "Download audio only (MP3)")]
    pub audio_only: bool,

    #[arg(
        long = "format",
        default_value = "mp4",
        help = "Output format (mp4, webm, mp3, wav, etc.)"
    )]
    pub format: String,

    #[arg(
        long = "quality",
        help = "Video quality (best, worst, 720p, 1080p, etc.)"
    )]
    pub quality: Option<String>,

    #[arg(long, help = "Show download progress")]
    pub verbose: bool,

    #[arg(long = "timeout", default_value = "600", help = "Timeout in seconds")]
    pub timeout: u64,
}

pub fn run(cli: DownloadArgs) -> Result<(), String> {
    if cli.verbose {
        eprintln!("mpipe: downloading from {}", cli.url);
    }

    let output_path = cli.output;
    let output_path_str = output_path
        .to_str()
        .ok_or_else(|| "Invalid output path".to_string())?;

    let mut args = vec![
        "--output".to_string(),
        output_path_str.to_string(),
        "--no-part".to_string(),
        "--no-clean-info-json".to_string(),
    ];

    if cli.audio_only {
        args.push("--extract-audio".to_string());
        args.push("--audio-format".to_string());
        args.push(cli.format.clone());
    } else {
        args.push("--format".to_string());
        if let Some(quality) = &cli.quality {
            args.push(format!("bestvideo[height<={}]+bestaudio/best", quality));
        } else {
            args.push("bestvideo+bestaudio/best".to_string());
        }
        args.push("--merge-output-format".to_string());
        args.push("mp4".to_string());
    }

    args.push(cli.url.clone());

    if cli.verbose {
        eprintln!("mpipe: running yt-dlp with args: {:?}", args);
    }

    let output = Command::new("yt-dlp")
        .args(&args)
        .output()
        .map_err(|e| format!("Failed to run yt-dlp: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("yt-dlp failed: {}", stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    if cli.verbose {
        eprintln!("mpipe: {}", stdout);
    }

    if output_path.exists() {
        println!("{}", output_path.display());
        Ok(())
    } else {
        Err(format!("Output file not found: {}", output_path.display()))
    }
}
