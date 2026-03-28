//! Command-line interface for the RedisVL Rust workspace.

use std::{fs, process::ExitCode};

use anyhow::{Context, bail};
use clap::{Args, Parser, Subcommand};
use comfy_table::{Table, presets::UTF8_FULL};
use redis_vl::{IndexSchema, SearchIndex};
use serde_json::{Map, Value};

#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "Redis Vector Library CLI",
    propagate_version = true,
    subcommand_required = true,
    arg_required_else_help = true
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Print the CLI version.
    Version(VersionArgs),
    /// Index lifecycle commands.
    #[command(subcommand_required = true, arg_required_else_help = true)]
    Index(IndexArgs),
    /// Print index stats from Redis.
    Stats(StatsArgs),
}

// ── Version ─────────────────────────────────────────────────────────────────

#[derive(Debug, Args)]
struct VersionArgs {
    /// Print only the version number.
    #[arg(short, long)]
    short: bool,
}

// ── Stats ───────────────────────────────────────────────────────────────────

#[derive(Debug, Args)]
struct StatsArgs {
    /// Index name (alternative to --schema).
    #[arg(short, long)]
    index: Option<String>,
    /// Index schema file in YAML format.
    #[arg(short, long)]
    schema: Option<String>,
    /// Redis connection URL.
    #[arg(long, env = "REDIS_URL", default_value = "redis://127.0.0.1:6379")]
    redis_url: String,
    /// Output results as JSON instead of a table.
    #[arg(long)]
    json: bool,
}

// ── Index ───────────────────────────────────────────────────────────────────

#[derive(Debug, Args)]
struct IndexArgs {
    #[command(subcommand)]
    command: IndexCommand,
}

#[derive(Debug, Subcommand)]
enum IndexCommand {
    /// Create an index from a schema file.
    Create(IndexCreateArgs),
    /// Delete an index (keeps documents).
    Delete(IndexRefArgs),
    /// Destroy an index and all of its documents.
    Destroy(IndexRefArgs),
    /// Show index metadata and field definitions.
    Info(IndexInfoArgs),
    /// List all search indices.
    Listall(ListallArgs),
}

/// Arguments for creating an index (requires --schema).
#[derive(Debug, Args)]
struct IndexCreateArgs {
    /// Index schema file in YAML format.
    #[arg(short, long)]
    schema: String,
    /// Redis connection URL.
    #[arg(long, env = "REDIS_URL", default_value = "redis://127.0.0.1:6379")]
    redis_url: String,
    /// Overwrite the index if it already exists.
    #[arg(short, long)]
    overwrite: bool,
}

/// Arguments that accept either --index or --schema to identify an index.
#[derive(Debug, Args)]
struct IndexRefArgs {
    /// Index name (alternative to --schema).
    #[arg(short, long)]
    index: Option<String>,
    /// Index schema file in YAML format.
    #[arg(short, long)]
    schema: Option<String>,
    /// Redis connection URL.
    #[arg(long, env = "REDIS_URL", default_value = "redis://127.0.0.1:6379")]
    redis_url: String,
}

/// Arguments for info (adds --json output support).
#[derive(Debug, Args)]
struct IndexInfoArgs {
    /// Index name (alternative to --schema).
    #[arg(short, long)]
    index: Option<String>,
    /// Index schema file in YAML format.
    #[arg(short, long)]
    schema: Option<String>,
    /// Redis connection URL.
    #[arg(long, env = "REDIS_URL", default_value = "redis://127.0.0.1:6379")]
    redis_url: String,
    /// Output results as JSON instead of a table.
    #[arg(long)]
    json: bool,
}

/// Arguments for listall (no index identification required).
#[derive(Debug, Args)]
struct ListallArgs {
    /// Redis connection URL.
    #[arg(long, env = "REDIS_URL", default_value = "redis://127.0.0.1:6379")]
    redis_url: String,
    /// Output results as JSON instead of a table.
    #[arg(long)]
    json: bool,
}

// ── Stats keys matching Python STATS_KEYS ───────────────────────────────────

const STATS_KEYS: &[&str] = &[
    "num_docs",
    "num_terms",
    "max_doc_id",
    "num_records",
    "percent_indexed",
    "hash_indexing_failures",
    "number_of_uses",
    "bytes_per_record_avg",
    "doc_table_size_mb",
    "inverted_sz_mb",
    "key_table_size_mb",
    "offset_bits_per_record_avg",
    "offset_vectors_sz_mb",
    "offsets_per_term_avg",
    "records_per_doc_avg",
    "sortable_values_size_mb",
    "total_indexing_time",
    "total_inverted_index_blocks",
    "vector_index_sz_mb",
];

// ── Entrypoint ──────────────────────────────────────────────────────────────

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Version(args) => {
            if args.short {
                println!("{}", env!("CARGO_PKG_VERSION"));
            } else {
                println!("RedisVL version {}", env!("CARGO_PKG_VERSION"));
            }
        }
        Command::Index(index) => match index.command {
            IndexCommand::Create(args) => {
                let index = load_index_from_schema(&args.schema, &args.redis_url)?;
                index
                    .create_with_options(args.overwrite, false)
                    .context("failed to create index")?;
                println!("Index created successfully");
            }
            IndexCommand::Delete(args) => {
                let index = resolve_index(&args.index, &args.schema, &args.redis_url)?;
                index.drop(false).context("failed to delete index")?;
                println!("Index deleted successfully");
            }
            IndexCommand::Destroy(args) => {
                let index = resolve_index(&args.index, &args.schema, &args.redis_url)?;
                index
                    .drop(true)
                    .context("failed to destroy index and documents")?;
                println!("Index and documents destroyed successfully");
            }
            IndexCommand::Info(args) => {
                let index = resolve_index(&args.index, &args.schema, &args.redis_url)?;
                let info = index.info().context("failed to fetch index info")?;
                if args.json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&info)
                            .context("failed to serialize info as JSON")?
                    );
                } else {
                    display_index_info(&info);
                }
            }
            IndexCommand::Listall(args) => {
                let index = listall_index(&args.redis_url)?;
                let indices = index.listall().context("failed to list indices")?;
                if args.json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&indices)
                            .context("failed to serialize index list as JSON")?
                    );
                } else {
                    println!("Indices:");
                    for (i, name) in indices.iter().enumerate() {
                        println!("{}. {name}", i + 1);
                    }
                }
            }
        },
        Command::Stats(args) => {
            let index = resolve_index(&args.index, &args.schema, &args.redis_url)?;
            let info = index.info().context("failed to fetch index stats")?;
            if args.json {
                let stats_map: Map<String, Value> = STATS_KEYS
                    .iter()
                    .filter_map(|key| info.get(*key).map(|v| ((*key).to_owned(), v.clone())))
                    .collect();
                println!(
                    "{}",
                    serde_json::to_string_pretty(&stats_map)
                        .context("failed to serialize stats as JSON")?
                );
            } else {
                display_stats(&info);
            }
        }
    }

    Ok(())
}

// ── Index resolution helpers ────────────────────────────────────────────────

/// Load a [`SearchIndex`] from a schema YAML file.
fn load_index_from_schema(schema_path: &str, redis_url: &str) -> anyhow::Result<SearchIndex> {
    let content = fs::read_to_string(schema_path)
        .with_context(|| format!("failed to read schema file '{schema_path}'"))?;
    let schema = IndexSchema::from_yaml_str(&content)
        .with_context(|| format!("failed to parse schema file '{schema_path}'"))?;
    Ok(SearchIndex::new(schema, redis_url))
}

/// Resolve a [`SearchIndex`] from either `--index` name or `--schema` file.
fn resolve_index(
    index_name: &Option<String>,
    schema_path: &Option<String>,
    redis_url: &str,
) -> anyhow::Result<SearchIndex> {
    if let Some(name) = index_name {
        SearchIndex::from_existing(name, redis_url)
            .with_context(|| format!("failed to connect to index '{name}'"))
    } else if let Some(path) = schema_path {
        load_index_from_schema(path, redis_url)
    } else {
        bail!("Index name or schema must be provided (use -i or -s)");
    }
}

/// Create a minimal [`SearchIndex`] for `listall` (only needs a connection).
fn listall_index(redis_url: &str) -> anyhow::Result<SearchIndex> {
    let schema = IndexSchema::from_json_value(serde_json::json!({
        "index": { "name": "_listall" }
    }))?;
    Ok(SearchIndex::new(schema, redis_url))
}

// ── Formatted output ────────────────────────────────────────────────────────

/// Display index info in a formatted table (mirrors Python `_display_in_table`).
fn display_index_info(info: &Map<String, Value>) {
    let index_name = info
        .get("index_name")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    let definition = info.get("index_definition");
    let storage_type = definition
        .and_then(|d| d.get("key_type"))
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let prefixes = definition
        .and_then(|d| d.get("prefixes"))
        .map(|v| format!("{v}"))
        .unwrap_or_default();
    let index_options = info
        .get("index_options")
        .map(|v| format!("{v}"))
        .unwrap_or_default();
    let indexing = info
        .get("indexing")
        .map(|v| format!("{v}"))
        .unwrap_or_default();

    println!("\nIndex Information:");
    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec![
        "Index Name",
        "Storage Type",
        "Prefixes",
        "Index Options",
        "Indexing",
    ]);
    table.add_row(vec![
        index_name,
        storage_type,
        &prefixes,
        &index_options,
        &indexing,
    ]);
    println!("{table}");

    // ── Fields table ──
    if let Some(Value::Array(attributes)) = info.get("attributes") {
        println!("Index Fields:");
        let mut field_table = Table::new();
        field_table.load_preset(UTF8_FULL);
        field_table.set_header(vec!["Name", "Attribute", "Type"]);

        for attr in attributes {
            if let Value::Array(parts) = attr {
                let get_val = |key: &str| -> String {
                    for chunk in parts.chunks(2) {
                        if chunk.len() == 2 {
                            if let Some(k) = chunk[0].as_str() {
                                if k == key {
                                    return chunk[1]
                                        .as_str()
                                        .map(|s| s.to_owned())
                                        .unwrap_or_else(|| format!("{}", chunk[1]));
                                }
                            }
                        }
                    }
                    String::new()
                };
                field_table.add_row(vec![
                    get_val("identifier"),
                    get_val("attribute"),
                    get_val("type"),
                ]);
            }
        }
        println!("{field_table}");
    }
}

/// Display stats in a formatted table (mirrors Python `_display_stats`).
fn display_stats(info: &Map<String, Value>) {
    println!("\nStatistics:");
    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec!["Stat Key", "Value"]);

    for key in STATS_KEYS {
        let val = info
            .get(*key)
            .map(|v| {
                v.as_str()
                    .map(|s| s.to_owned())
                    .unwrap_or_else(|| format!("{v}"))
            })
            .unwrap_or_default();
        table.add_row(vec![key.to_string(), val]);
    }
    println!("{table}");
}
