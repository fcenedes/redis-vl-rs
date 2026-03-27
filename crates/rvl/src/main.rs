//! Command-line interface for the RedisVL Rust workspace.

use std::{fs, process::ExitCode};

use clap::{Args, Parser, Subcommand};
use comfy_table::{Table, presets::UTF8_FULL};
use redis_vl::{IndexSchema, SearchIndex};

#[derive(Debug, Parser)]
#[command(author, version, about = "Redis Vector Library CLI")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Print the CLI version.
    Version,
    /// Index lifecycle commands.
    Index(IndexArgs),
    /// Print raw index stats from Redis.
    Stats(StatsArgs),
}

#[derive(Debug, Args)]
struct StatsArgs {
    /// Index schema file in YAML format.
    #[arg(long)]
    schema: String,
    /// Redis connection URL.
    #[arg(long, env = "REDIS_URL", default_value = "redis://127.0.0.1:6379")]
    redis_url: String,
}

#[derive(Debug, Args)]
struct IndexArgs {
    #[command(subcommand)]
    command: IndexCommand,
}

#[derive(Debug, Subcommand)]
enum IndexCommand {
    /// Create an index from a schema file.
    Create(IndexCommandArgs),
    /// Drop an index.
    Delete(IndexDeleteArgs),
    /// Alias for delete.
    Destroy(IndexDeleteArgs),
    /// Show index metadata.
    Info(IndexCommandArgs),
    /// List all search indices.
    Listall(IndexCommandArgs),
}

#[derive(Debug, Args)]
struct IndexCommandArgs {
    /// Index schema file in YAML format.
    #[arg(long)]
    schema: String,
    /// Redis connection URL.
    #[arg(long, env = "REDIS_URL", default_value = "redis://127.0.0.1:6379")]
    redis_url: String,
}

#[derive(Debug, Args)]
struct IndexDeleteArgs {
    /// Index schema file in YAML format.
    #[arg(long)]
    schema: String,
    /// Redis connection URL.
    #[arg(long, env = "REDIS_URL", default_value = "redis://127.0.0.1:6379")]
    redis_url: String,
    /// Delete indexed documents alongside the index.
    #[arg(long)]
    delete_documents: bool,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Command::Version => {
            println!("{}", env!("CARGO_PKG_VERSION"));
        }
        Command::Index(index) => match index.command {
            IndexCommand::Create(args) => {
                let index = load_index(&args.schema, &args.redis_url)?;
                index.create()?;
                println!("created index {}", index.schema().index.name);
            }
            IndexCommand::Delete(args) | IndexCommand::Destroy(args) => {
                let index = load_index(&args.schema, &args.redis_url)?;
                index.drop(args.delete_documents)?;
                println!("dropped index {}", index.schema().index.name);
            }
            IndexCommand::Info(args) => {
                let index = load_index(&args.schema, &args.redis_url)?;
                println!("{:#?}", index.info()?);
            }
            IndexCommand::Listall(args) => {
                let index = load_index(&args.schema, &args.redis_url)?;
                let indices = index.listall()?;
                let mut table = Table::new();
                table.load_preset(UTF8_FULL);
                table.set_header(vec!["Index Name"]);
                for name in indices {
                    table.add_row(vec![name]);
                }
                println!("{table}");
            }
        },
        Command::Stats(args) => {
            let index = load_index(&args.schema, &args.redis_url)?;
            println!("{:#?}", index.info()?);
        }
    }

    Ok(())
}

fn load_index(
    schema_path: &str,
    redis_url: &str,
) -> Result<SearchIndex, Box<dyn std::error::Error>> {
    let schema = IndexSchema::from_yaml_str(&fs::read_to_string(schema_path)?)?;
    Ok(SearchIndex::new(schema, redis_url))
}
