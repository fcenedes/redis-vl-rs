//! Live Redis integration tests for SQL queries against **hash** storage.
//!
//! Mirrors the upstream Python test file:
//! `tests/integration/test_sql_redis_hash.py`
//!
//! Gated by `REDISVL_RUN_INTEGRATION=1`.

use std::sync::atomic::{AtomicU32, Ordering};

use serde_json::{Map, Value, json};

use redis_vl::{QueryOutput, SQLQuery, SearchIndex};

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn integration_enabled() -> bool {
    std::env::var("REDISVL_RUN_INTEGRATION")
        .map(|v| v == "1")
        .unwrap_or(false)
}

fn redis_url() -> String {
    std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string())
}

/// Creates the products index (hash storage) matching Python's `sql_index` fixture.
fn create_sql_index() -> Option<SearchIndex> {
    if !integration_enabled() {
        return None;
    }

    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let schema = json!({
        "index": {
            "name": format!("sql_hash_parity_{id}"),
            "prefix": format!("sql_hash_prod_{id}"),
            "storage_type": "hash",
        },
        "fields": [
            { "name": "title", "type": "text", "attrs": { "sortable": true } },
            { "name": "name", "type": "text", "attrs": { "sortable": true } },
            { "name": "price", "type": "numeric", "attrs": { "sortable": true } },
            { "name": "stock", "type": "numeric", "attrs": { "sortable": true } },
            { "name": "rating", "type": "numeric", "attrs": { "sortable": true } },
            { "name": "category", "type": "tag", "attrs": { "sortable": true } },
            { "name": "tags", "type": "tag" },
        ],
    });

    let index = SearchIndex::from_json_value(schema, redis_url()).expect("schema should parse");
    if index.create_with_options(true, true).is_err() {
        return None;
    }
    Some(index)
}

fn product_data() -> Vec<Value> {
    vec![
        json!({"id":"1","title":"Gaming laptop Pro","name":"Gaming Laptop","price":899,"stock":10,"rating":4.5,"category":"electronics","tags":"sale,featured"}),
        json!({"id":"2","title":"Budget laptop Basic","name":"Budget Laptop","price":499,"stock":25,"rating":3.8,"category":"electronics","tags":"sale"}),
        json!({"id":"3","title":"Premium laptop Ultra","name":"Premium Laptop","price":1299,"stock":5,"rating":4.9,"category":"electronics","tags":"featured"}),
        json!({"id":"4","title":"Python Programming","name":"Python Book","price":45,"stock":100,"rating":4.7,"category":"books","tags":"bestseller"}),
        json!({"id":"5","title":"Redis in Action","name":"Redis Book","price":55,"stock":50,"rating":4.6,"category":"books","tags":"featured"}),
        json!({"id":"6","title":"Data Science Guide","name":"DS Book","price":65,"stock":30,"rating":4.4,"category":"books","tags":"sale"}),
        json!({"id":"7","title":"Wireless Mouse","name":"Mouse","price":29,"stock":200,"rating":4.2,"category":"electronics","tags":"sale"}),
        json!({"id":"8","title":"Mechanical Keyboard","name":"Keyboard","price":149,"stock":75,"rating":4.6,"category":"electronics","tags":"featured"}),
        json!({"id":"9","title":"USB Hub","name":"Hub","price":25,"stock":150,"rating":3.9,"category":"electronics","tags":"sale"}),
        json!({"id":"10","title":"Monitor Stand","name":"Stand","price":89,"stock":40,"rating":4.1,"category":"accessories","tags":"sale,featured"}),
        json!({"id":"11","title":"Desk Lamp","name":"Lamp","price":35,"stock":80,"rating":4.0,"category":"accessories","tags":"sale"}),
        json!({"id":"12","title":"Notebook Set","name":"Notebooks","price":15,"stock":300,"rating":4.3,"category":"stationery","tags":"bestseller"}),
        json!({"id":"13","title":"Laptop and Keyboard Bundle","name":"Bundle Pack","price":999,"stock":15,"rating":4.7,"category":"electronics","tags":"featured,sale"}),
    ]
}

fn load_and_get(index: &SearchIndex) -> &SearchIndex {
    index
        .load(&product_data(), "id", None)
        .expect("load should succeed");
    // Small delay to let Redis index the documents.
    std::thread::sleep(std::time::Duration::from_millis(200));
    index
}

fn docs(output: QueryOutput) -> Vec<Map<String, Value>> {
    output.as_documents().expect("expected documents").to_vec()
}

// ---- Basic SELECT tests ----

#[test]
fn sql_hash_select_all() {
    let Some(index) = create_sql_index() else {
        return;
    };
    load_and_get(&index);

    let q = SQLQuery::new(format!("SELECT * FROM {}", index.name()));
    let results = docs(index.sql_query(&q).expect("query should succeed"));
    assert!(results.len() > 0, "SELECT * should return rows");
    assert!(results[0].contains_key("title"));
    assert!(results[0].contains_key("price"));

    index.delete(true).expect("cleanup");
}

#[test]
fn sql_hash_select_specific_fields() {
    let Some(index) = create_sql_index() else {
        return;
    };
    load_and_get(&index);

    let q = SQLQuery::new(format!("SELECT title, price FROM {}", index.name()));
    let results = docs(index.sql_query(&q).expect("query should succeed"));
    assert!(results.len() > 0);
    assert!(results[0].contains_key("title"));
    assert!(results[0].contains_key("price"));

    index.delete(true).expect("cleanup");
}

// ---- WHERE clause tests ----

#[test]
fn sql_hash_where_tag_equals() {
    let Some(index) = create_sql_index() else {
        return;
    };
    load_and_get(&index);

    let q = SQLQuery::new(format!(
        "SELECT title, price, category FROM {} WHERE category = 'electronics'",
        index.name()
    ));
    let results = docs(index.sql_query(&q).expect("query should succeed"));
    assert!(results.len() > 0);
    for r in &results {
        assert_eq!(r["category"], "electronics");
    }

    index.delete(true).expect("cleanup");
}

#[test]
fn sql_hash_where_numeric_less_than() {
    let Some(index) = create_sql_index() else {
        return;
    };
    load_and_get(&index);

    let q = SQLQuery::new(format!(
        "SELECT title, price FROM {} WHERE price < 50",
        index.name()
    ));
    let results = docs(index.sql_query(&q).expect("query should succeed"));
    assert!(results.len() > 0);
    for r in &results {
        let price: f64 = r["price"].as_str().unwrap().parse().unwrap();
        assert!(price < 50.0, "price {} should be < 50", price);
    }

    index.delete(true).expect("cleanup");
}

// ---- WHERE combined conditions ----

#[test]
fn sql_hash_where_combined_and() {
    let Some(index) = create_sql_index() else {
        return;
    };
    load_and_get(&index);

    let q = SQLQuery::new(format!(
        "SELECT title, price, category FROM {} WHERE category = 'electronics' AND price < 100",
        index.name()
    ));
    let results = docs(index.sql_query(&q).expect("query should succeed"));
    for r in &results {
        assert_eq!(r["category"], "electronics");
        let price: f64 = r["price"].as_str().unwrap().parse().unwrap();
        assert!(price < 100.0);
    }

    index.delete(true).expect("cleanup");
}

#[test]
fn sql_hash_where_numeric_range() {
    let Some(index) = create_sql_index() else {
        return;
    };
    load_and_get(&index);

    let q = SQLQuery::new(format!(
        "SELECT title, price FROM {} WHERE price >= 25 AND price <= 50",
        index.name()
    ));
    let results = docs(index.sql_query(&q).expect("query should succeed"));
    for r in &results {
        let price: f64 = r["price"].as_str().unwrap().parse().unwrap();
        assert!(
            price >= 25.0 && price <= 50.0,
            "price {} out of range",
            price
        );
    }

    index.delete(true).expect("cleanup");
}

// ---- Tag operators ----

#[test]
fn sql_hash_tag_not_equals() {
    let Some(index) = create_sql_index() else {
        return;
    };
    load_and_get(&index);

    let q = SQLQuery::new(format!(
        "SELECT title, category FROM {} WHERE category != 'electronics'",
        index.name()
    ));
    let results = docs(index.sql_query(&q).expect("query should succeed"));
    assert!(results.len() > 0);
    for r in &results {
        assert_ne!(r["category"], "electronics");
    }

    index.delete(true).expect("cleanup");
}

#[test]
fn sql_hash_tag_in() {
    let Some(index) = create_sql_index() else {
        return;
    };
    load_and_get(&index);

    let q = SQLQuery::new(format!(
        "SELECT title, category FROM {} WHERE category IN ('books', 'accessories')",
        index.name()
    ));
    let results = docs(index.sql_query(&q).expect("query should succeed"));
    assert!(results.len() > 0);
    for r in &results {
        let cat = r["category"].as_str().unwrap();
        assert!(
            cat == "books" || cat == "accessories",
            "unexpected category: {cat}"
        );
    }

    index.delete(true).expect("cleanup");
}

// ---- Numeric operators ----

#[test]
fn sql_hash_numeric_greater_than() {
    let Some(index) = create_sql_index() else {
        return;
    };
    load_and_get(&index);

    let q = SQLQuery::new(format!(
        "SELECT title, price FROM {} WHERE price > 100",
        index.name()
    ));
    let results = docs(index.sql_query(&q).expect("query should succeed"));
    assert!(results.len() > 0);
    for r in &results {
        let price: f64 = r["price"].as_str().unwrap().parse().unwrap();
        assert!(price > 100.0);
    }

    index.delete(true).expect("cleanup");
}

#[test]
fn sql_hash_numeric_equals() {
    let Some(index) = create_sql_index() else {
        return;
    };
    load_and_get(&index);

    let q = SQLQuery::new(format!(
        "SELECT title, price FROM {} WHERE price = 45",
        index.name()
    ));
    let results = docs(index.sql_query(&q).expect("query should succeed"));
    assert!(results.len() >= 1);
    for r in &results {
        let price: f64 = r["price"].as_str().unwrap().parse().unwrap();
        assert_eq!(price, 45.0);
    }

    index.delete(true).expect("cleanup");
}

#[test]
fn sql_hash_numeric_between() {
    let Some(index) = create_sql_index() else {
        return;
    };
    load_and_get(&index);

    let q = SQLQuery::new(format!(
        "SELECT title, price FROM {} WHERE price BETWEEN 40 AND 60",
        index.name()
    ));
    let results = docs(index.sql_query(&q).expect("query should succeed"));
    assert!(results.len() > 0);
    for r in &results {
        let price: f64 = r["price"].as_str().unwrap().parse().unwrap();
        assert!(
            price >= 40.0 && price <= 60.0,
            "price {} not in BETWEEN range",
            price
        );
    }

    index.delete(true).expect("cleanup");
}

// ---- Text operators ----
//
// Note: the SQL parser without schema context treats single-word `=` as tag
// syntax.  The Python implementation avoids this by loading schema info from
// Redis.  Use LIKE for text-field searches until schema-aware parsing is added.

#[test]
fn sql_hash_text_like() {
    let Some(index) = create_sql_index() else {
        return;
    };
    load_and_get(&index);

    let q = SQLQuery::new(format!(
        "SELECT title, name FROM {} WHERE title LIKE '%laptop%'",
        index.name()
    ));
    let results = docs(index.sql_query(&q).expect("query should succeed"));
    assert!(results.len() >= 1);
    for r in &results {
        assert!(
            r["title"]
                .as_str()
                .unwrap()
                .to_lowercase()
                .contains("laptop"),
            "title should contain 'laptop'"
        );
    }

    index.delete(true).expect("cleanup");
}

#[test]
fn sql_hash_text_not_like() {
    let Some(index) = create_sql_index() else {
        return;
    };
    load_and_get(&index);

    let q = SQLQuery::new(format!(
        "SELECT title FROM {} WHERE title NOT LIKE '%laptop%'",
        index.name()
    ));
    let results = docs(index.sql_query(&q).expect("query should succeed"));
    assert!(results.len() >= 1);
    for r in &results {
        assert!(
            !r["title"]
                .as_str()
                .unwrap()
                .to_lowercase()
                .contains("laptop"),
            "title should NOT contain 'laptop'"
        );
    }

    index.delete(true).expect("cleanup");
}

// ---- ORDER BY tests ----

#[test]
fn sql_hash_order_by_asc() {
    let Some(index) = create_sql_index() else {
        return;
    };
    load_and_get(&index);

    let q = SQLQuery::new(format!(
        "SELECT title, price FROM {} ORDER BY price ASC",
        index.name()
    ));
    let results = docs(index.sql_query(&q).expect("query should succeed"));
    let prices: Vec<f64> = results
        .iter()
        .map(|r| r["price"].as_str().unwrap().parse().unwrap())
        .collect();
    let mut sorted = prices.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(prices, sorted, "prices should be ascending");

    index.delete(true).expect("cleanup");
}

#[test]
fn sql_hash_order_by_desc() {
    let Some(index) = create_sql_index() else {
        return;
    };
    load_and_get(&index);

    let q = SQLQuery::new(format!(
        "SELECT title, price FROM {} ORDER BY price DESC",
        index.name()
    ));
    let results = docs(index.sql_query(&q).expect("query should succeed"));
    let prices: Vec<f64> = results
        .iter()
        .map(|r| r["price"].as_str().unwrap().parse().unwrap())
        .collect();
    let mut sorted = prices.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert_eq!(prices, sorted, "prices should be descending");

    index.delete(true).expect("cleanup");
}

// ---- LIMIT / OFFSET tests ----

#[test]
fn sql_hash_limit() {
    let Some(index) = create_sql_index() else {
        return;
    };
    load_and_get(&index);

    let q = SQLQuery::new(format!("SELECT title FROM {} LIMIT 3", index.name()));
    let results = docs(index.sql_query(&q).expect("query should succeed"));
    assert_eq!(results.len(), 3);

    index.delete(true).expect("cleanup");
}

#[test]
fn sql_hash_limit_with_offset() {
    let Some(index) = create_sql_index() else {
        return;
    };
    load_and_get(&index);

    let q1 = SQLQuery::new(format!(
        "SELECT title FROM {} ORDER BY price ASC LIMIT 3 OFFSET 0",
        index.name()
    ));
    let q2 = SQLQuery::new(format!(
        "SELECT title FROM {} ORDER BY price ASC LIMIT 3 OFFSET 3",
        index.name()
    ));
    let r1 = docs(index.sql_query(&q1).expect("page 1"));
    let r2 = docs(index.sql_query(&q2).expect("page 2"));
    assert_eq!(r1.len(), 3);
    assert_eq!(r2.len(), 3);
    let t1: Vec<&str> = r1.iter().map(|r| r["title"].as_str().unwrap()).collect();
    let t2: Vec<&str> = r2.iter().map(|r| r["title"].as_str().unwrap()).collect();
    for t in &t1 {
        assert!(!t2.contains(t), "pages should not overlap");
    }

    index.delete(true).expect("cleanup");
}

// ---- Aggregation tests ----

#[test]
fn sql_hash_count_all() {
    let Some(index) = create_sql_index() else {
        return;
    };
    load_and_get(&index);

    let q = SQLQuery::new(format!("SELECT COUNT(*) as total FROM {}", index.name()));
    let results = docs(index.sql_query(&q).expect("query should succeed"));
    assert_eq!(results.len(), 1);
    let total: i64 = results[0]["total"].as_str().unwrap().parse().unwrap();
    assert_eq!(total, 13, "should count all 13 products");

    index.delete(true).expect("cleanup");
}

#[test]
fn sql_hash_group_by_with_count() {
    let Some(index) = create_sql_index() else {
        return;
    };
    load_and_get(&index);

    let q = SQLQuery::new(format!(
        "SELECT category, COUNT(*) as count FROM {} GROUP BY category",
        index.name()
    ));
    let results = docs(index.sql_query(&q).expect("query should succeed"));
    let categories: Vec<&str> = results
        .iter()
        .map(|r| r["category"].as_str().unwrap())
        .collect();
    assert!(categories.contains(&"electronics"));
    assert!(categories.contains(&"books"));

    index.delete(true).expect("cleanup");
}

#[test]
fn sql_hash_group_by_with_avg() {
    let Some(index) = create_sql_index() else {
        return;
    };
    load_and_get(&index);

    let q = SQLQuery::new(format!(
        "SELECT category, AVG(price) as avg_price FROM {} GROUP BY category",
        index.name()
    ));
    let results = docs(index.sql_query(&q).expect("query should succeed"));
    for r in &results {
        assert!(r.contains_key("category"));
        assert!(r.contains_key("avg_price"));
        let avg: f64 = r["avg_price"].as_str().unwrap().parse().unwrap();
        assert!(avg > 0.0);
    }

    index.delete(true).expect("cleanup");
}

#[test]
fn sql_hash_group_by_with_filter() {
    let Some(index) = create_sql_index() else {
        return;
    };
    load_and_get(&index);

    let q = SQLQuery::new(format!(
        "SELECT category, AVG(price) as avg_price FROM {} WHERE stock > 50 GROUP BY category",
        index.name()
    ));
    let results = docs(index.sql_query(&q).expect("query should succeed"));
    assert!(results.len() > 0);
    for r in &results {
        assert!(r.contains_key("category"));
        assert!(r.contains_key("avg_price"));
    }

    index.delete(true).expect("cleanup");
}

#[test]
fn sql_hash_group_by_with_sum() {
    let Some(index) = create_sql_index() else {
        return;
    };
    load_and_get(&index);

    let q = SQLQuery::new(format!(
        "SELECT category, SUM(price) as total_price FROM {} GROUP BY category",
        index.name()
    ));
    let results = docs(index.sql_query(&q).expect("query should succeed"));
    assert!(results.len() > 0);
    for r in &results {
        assert!(r.contains_key("total_price"));
        let total: f64 = r["total_price"].as_str().unwrap().parse().unwrap();
        assert!(total > 0.0);
    }

    index.delete(true).expect("cleanup");
}

#[test]
fn sql_hash_group_by_with_min_max() {
    let Some(index) = create_sql_index() else {
        return;
    };
    load_and_get(&index);

    let q = SQLQuery::new(format!(
        "SELECT category, MIN(price) as min_price, MAX(price) as max_price FROM {} GROUP BY category",
        index.name()
    ));
    let results = docs(index.sql_query(&q).expect("query should succeed"));
    assert!(results.len() > 0);
    for r in &results {
        assert!(r.contains_key("min_price"));
        assert!(r.contains_key("max_price"));
        let min_p: f64 = r["min_price"].as_str().unwrap().parse().unwrap();
        let max_p: f64 = r["max_price"].as_str().unwrap().parse().unwrap();
        assert!(min_p > 0.0);
        assert!(max_p >= min_p);
    }

    index.delete(true).expect("cleanup");
}

#[test]
fn sql_hash_global_sum() {
    let Some(index) = create_sql_index() else {
        return;
    };
    load_and_get(&index);

    let q = SQLQuery::new(format!("SELECT SUM(price) as total FROM {}", index.name()));
    let results = docs(index.sql_query(&q).expect("query should succeed"));
    assert_eq!(results.len(), 1);
    let total: f64 = results[0]["total"].as_str().unwrap().parse().unwrap();
    assert!(total > 0.0);

    index.delete(true).expect("cleanup");
}

#[test]
fn sql_hash_multiple_reducers() {
    let Some(index) = create_sql_index() else {
        return;
    };
    load_and_get(&index);

    let q = SQLQuery::new(format!(
        "SELECT category, COUNT(*) as count, SUM(price) as total, AVG(price) as avg_price, MIN(price) as min_price, MAX(price) as max_price FROM {} GROUP BY category",
        index.name()
    ));
    let results = docs(index.sql_query(&q).expect("query should succeed"));
    assert!(results.len() > 0);
    for r in &results {
        assert!(r.contains_key("category"));
        assert!(r.contains_key("count"));
        assert!(r.contains_key("total"));
        assert!(r.contains_key("avg_price"));
        assert!(r.contains_key("min_price"));
        assert!(r.contains_key("max_price"));
    }

    index.delete(true).expect("cleanup");
}

#[test]
fn sql_hash_count_distinct() {
    let Some(index) = create_sql_index() else {
        return;
    };
    load_and_get(&index);

    let q = SQLQuery::new(format!(
        "SELECT COUNT_DISTINCT(category) as unique_categories FROM {}",
        index.name()
    ));
    let results = docs(index.sql_query(&q).expect("query should succeed"));
    assert_eq!(results.len(), 1);
    let unique: i64 = results[0]["unique_categories"]
        .as_str()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(unique, 4, "should have 4 unique categories");

    index.delete(true).expect("cleanup");
}

// ---- Date literal tests ----

#[test]
fn sql_hash_date_literal_filter() {
    if !integration_enabled() {
        return;
    }

    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let schema = json!({
        "index": {
            "name": format!("sql_hash_date_{id}"),
            "prefix": format!("sql_hash_evt_{id}"),
            "storage_type": "hash",
        },
        "fields": [
            { "name": "name", "type": "text", "attrs": { "sortable": true } },
            { "name": "category", "type": "tag", "attrs": { "sortable": true } },
            { "name": "created_at", "type": "numeric", "attrs": { "sortable": true } },
        ],
    });

    let index = SearchIndex::from_json_value(schema, redis_url()).expect("schema");
    index.create_with_options(true, true).expect("create");

    // Timestamps for test events (UTC)
    let ts_2024_01_15 = 1705276800i64; // 2024-01-15T00:00:00Z
    let ts_2024_02_20 = 1708387200i64; // 2024-02-20T00:00:00Z
    let ts_2024_03_10 = 1710028800i64; // 2024-03-10T00:00:00Z
    let ts_2024_07_15 = 1721001600i64; // 2024-07-15T00:00:00Z
    let ts_2023_12_15 = 1702598400i64; // 2023-12-15T00:00:00Z

    let events = vec![
        json!({"id":"e1","name":"New Year Kickoff","category":"meeting","created_at":ts_2024_01_15}),
        json!({"id":"e2","name":"Product Launch","category":"release","created_at":ts_2024_02_20}),
        json!({"id":"e3","name":"Team Offsite","category":"meeting","created_at":ts_2024_03_10}),
        json!({"id":"e4","name":"Summer Summit","category":"conference","created_at":ts_2024_07_15}),
        json!({"id":"e5","name":"Holiday Party 2023","category":"social","created_at":ts_2023_12_15}),
    ];
    index.load(&events, "id", None).expect("load");
    std::thread::sleep(std::time::Duration::from_millis(200));

    let q = SQLQuery::new(format!(
        "SELECT name FROM {} WHERE created_at > '2024-01-01'",
        index.name()
    ));
    let results = docs(index.sql_query(&q).expect("query"));
    assert_eq!(results.len(), 4, "should find 4 events in 2024");
    let names: Vec<&str> = results
        .iter()
        .map(|r| r["name"].as_str().unwrap())
        .collect();
    assert!(!names.contains(&"Holiday Party 2023"));

    index.delete(true).expect("cleanup");
}

#[test]
fn sql_hash_date_between() {
    if !integration_enabled() {
        return;
    }

    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let schema = json!({
        "index": {
            "name": format!("sql_hash_dateb_{id}"),
            "prefix": format!("sql_hash_evtb_{id}"),
            "storage_type": "hash",
        },
        "fields": [
            { "name": "name", "type": "text", "attrs": { "sortable": true } },
            { "name": "created_at", "type": "numeric", "attrs": { "sortable": true } },
        ],
    });

    let index = SearchIndex::from_json_value(schema, redis_url()).expect("schema");
    index.create_with_options(true, true).expect("create");

    let events = vec![
        json!({"id":"e1","name":"New Year Kickoff","created_at":1705276800}),
        json!({"id":"e2","name":"Product Launch","created_at":1708387200}),
        json!({"id":"e3","name":"Team Offsite","created_at":1710028800}),
        json!({"id":"e4","name":"Summer Summit","created_at":1721001600}),
        json!({"id":"e5","name":"Holiday Party 2023","created_at":1702598400}),
    ];
    index.load(&events, "id", None).expect("load");
    std::thread::sleep(std::time::Duration::from_millis(200));

    let q = SQLQuery::new(format!(
        "SELECT name FROM {} WHERE created_at BETWEEN '2024-01-01' AND '2024-03-31'",
        index.name()
    ));
    let results = docs(index.sql_query(&q).expect("query"));
    assert_eq!(results.len(), 3, "should find Q1 2024 events");
    let names: Vec<&str> = results
        .iter()
        .map(|r| r["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"New Year Kickoff"));
    assert!(names.contains(&"Product Launch"));
    assert!(names.contains(&"Team Offsite"));

    index.delete(true).expect("cleanup");
}
