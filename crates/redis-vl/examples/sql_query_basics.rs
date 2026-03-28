//! Demonstrates building SQL queries with `SQLQuery`.
//!
//! This example shows how to construct SQL queries and inspect their
//! translated Redis Search representations. Full execution requires
//! a running Redis instance.
//!
//! **Requires the `sql` feature flag:**
//! ```bash
//! cargo run -p redis-vl --features sql --example sql_query_basics
//! ```

#[cfg(feature = "sql")]
fn main() {
    use redis_vl::{SQLQuery, SqlParam};

    println!("=== Non-aggregate queries (→ FT.SEARCH) ===\n");

    // Basic SELECT with WHERE
    let q1 = SQLQuery::new("SELECT * FROM products WHERE category = 'electronics'");
    println!("Query 1: {:?}\n", q1);

    // WHERE with multiple conditions and parameters
    let q2 = SQLQuery::new(
        "SELECT title, price FROM products WHERE category = :cat AND price > :min ORDER BY price DESC LIMIT 10",
    )
    .with_param("cat", SqlParam::Str("electronics".into()))
    .with_param("min", SqlParam::Float(49.99));
    println!("Query 2: {:?}\n", q2);

    // BETWEEN and LIKE
    let q3 = SQLQuery::new(
        "SELECT * FROM products WHERE price BETWEEN 10 AND 100 AND title LIKE 'widget%'",
    );
    println!("Query 3: {:?}\n", q3);

    // IN clause
    let q4 =
        SQLQuery::new("SELECT * FROM products WHERE category IN ('electronics', 'books', 'toys')");
    println!("Query 4: {:?}\n", q4);

    // Date comparisons
    let q5 = SQLQuery::new(
        "SELECT * FROM events WHERE created_at > '2024-01-01' AND created_at < '2024-12-31'",
    );
    println!("Query 5: {:?}\n", q5);

    println!("=== Aggregate queries (→ FT.AGGREGATE) ===\n");

    // COUNT with GROUP BY
    let q6 = SQLQuery::new("SELECT category, COUNT(*) as cnt FROM products GROUP BY category");
    println!("Query 6: {:?}\n", q6);

    // Multiple aggregation functions
    let q7 = SQLQuery::new(
        "SELECT category, COUNT(*) as cnt, AVG(price) as avg_price, SUM(price) as total FROM products GROUP BY category",
    );
    println!("Query 7: {:?}\n", q7);

    // Global aggregation (no GROUP BY)
    let q8 = SQLQuery::new("SELECT COUNT(*) as total, AVG(price) as avg FROM products");
    println!("Query 8: {:?}\n", q8);

    // WHERE + GROUP BY
    let q9 = SQLQuery::new(
        "SELECT category, COUNT(*) as cnt FROM products WHERE price > 10 GROUP BY category",
    );
    println!("Query 9: {:?}\n", q9);

    println!("To execute these queries against Redis, use:");
    println!("  index.sql_query(&query).unwrap()");
    println!("  // or async: index.sql_query(&query).await?");
}

#[cfg(not(feature = "sql"))]
fn main() {
    eprintln!("This example requires the `sql` feature flag.");
    eprintln!("Run with: cargo run -p redis-vl --features sql --example sql_query_basics");
    std::process::exit(1);
}
