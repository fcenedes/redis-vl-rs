//! Demonstrates creating an index, loading data, and running vector queries.
//!
//! **Requires a running Redis instance** with the Search module (Redis 8+ or
//! Redis Stack). Set `REDIS_URL` to override the default `redis://127.0.0.1:6379`.
//!
//! Run with:
//! ```bash
//! cargo run -p redis-vl --example vector_search
//! ```

use redis_vl::filter::{Num, Tag};
use redis_vl::{CountQuery, FilterQuery, IndexSchema, SearchIndex, Vector, VectorQuery};
use serde_json::json;

fn redis_url() -> String {
    std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_owned())
}

fn main() -> Result<(), redis_vl::Error> {
    let yaml = r#"
index:
  name: vector-search-example
  prefix: vdoc
  storage_type: hash
fields:
  - name: title
    type: tag
  - name: category
    type: tag
  - name: price
    type: numeric
  - name: embedding
    type: vector
    attrs:
      algorithm: flat
      dims: 4
      distance_metric: cosine
      datatype: float32
"#;
    let schema = IndexSchema::from_yaml_str(yaml)?;
    let index = SearchIndex::new(schema, redis_url());

    // Create or overwrite the index
    index.create_with_options(true, true)?;
    println!("Index created: {}", index.name());

    // Load sample documents
    let docs = vec![
        json!({"id": "1", "title": "Widget A", "category": "electronics", "price": 29.99, "embedding": [0.1, 0.2, 0.3, 0.4]}),
        json!({"id": "2", "title": "Widget B", "category": "electronics", "price": 59.99, "embedding": [0.2, 0.3, 0.4, 0.5]}),
        json!({"id": "3", "title": "Book C",   "category": "books",       "price": 14.99, "embedding": [0.9, 0.1, 0.1, 0.1]}),
        json!({"id": "4", "title": "Book D",   "category": "books",       "price": 24.99, "embedding": [0.8, 0.2, 0.1, 0.2]}),
    ];
    let keys = index.load(&docs, "id", None)?;
    println!("Loaded {} documents: {:?}\n", keys.len(), keys);

    // --- Vector search ---
    let query_vec = Vector::new(&[0.15, 0.25, 0.35, 0.45_f32] as &[f32]);
    let query = VectorQuery::new(query_vec, "embedding", 3)
        .with_return_fields(["title", "category", "price"]);
    let result = index.search(&query)?;
    println!("Vector search (top 3): {} total", result.total);
    for doc in &result.docs {
        println!(
            "  {} – {} (${}) dist={}",
            doc.id(),
            doc["title"],
            doc["price"],
            doc["vector_distance"]
        );
    }

    // --- Vector search with filter ---
    let filtered_query = VectorQuery::new(
        Vector::new(&[0.15, 0.25, 0.35, 0.45_f32] as &[f32]),
        "embedding",
        3,
    )
    .with_filter(Tag::new("category").eq("electronics"))
    .with_return_fields(["title", "price"]);
    let result = index.search(&filtered_query)?;
    println!(
        "\nFiltered vector search (electronics only): {} total",
        result.total
    );
    for doc in &result.docs {
        println!("  {} – ${}", doc["title"], doc["price"]);
    }

    // --- Filter query ---
    let filter = FilterQuery::new(Num::new("price").lt(30.0));
    let result = index.search(&filter)?;
    println!("\nFilter query (price < 30): {} total", result.total);
    for doc in &result.docs {
        println!("  {} – ${}", doc["title"], doc["price"]);
    }

    // --- Count query ---
    let count = CountQuery::new().with_filter(Tag::new("category").eq("books"));
    let output = index.query(&count)?;
    println!("\nCount query (books): {:?}", output.as_count());

    // --- Pagination ---
    let all = FilterQuery::new(Tag::new("category").one_of(["electronics", "books"]));
    let pages = index.paginate(&all, 2)?;
    println!("\nPagination (page_size=2): {} pages", pages.len());
    for (i, page) in pages.iter().enumerate() {
        println!("  Page {}: {} docs", i + 1, page.len());
    }

    // Cleanup
    index.drop(true)?;
    println!("\nIndex dropped.");
    Ok(())
}
