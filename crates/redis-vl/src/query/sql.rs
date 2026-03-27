//! SQL-like query type for Redis Search parity with Python `redisvl.query.SQLQuery`.
//!
//! This module mirrors the upstream Python `SQLQuery` class that accepts SQL-like
//! syntax and optional parameters. The actual SQL→Redis command translation
//! depends on an external `sql-redis`-equivalent crate and is **not** implemented
//! here. What **is** implemented:
//!
//! - `SQLQuery` type holding the raw SQL string and optional parameters
//! - `SqlParam` enum for typed parameter values (string, numeric, binary)
//! - Token-based parameter substitution that prevents partial-match bugs
//!   (`:id` won't clobber `:product_id`) and escapes single quotes in strings
//! - `QueryString` trait implementation so `SQLQuery` can be passed to
//!   `SearchIndex::query()` / `AsyncSearchIndex::query()`

use std::collections::HashMap;

use super::QueryString;

/// A typed SQL parameter value.
#[derive(Debug, Clone)]
pub enum SqlParam {
    /// An integer parameter.
    Int(i64),
    /// A floating-point parameter.
    Float(f64),
    /// A UTF-8 string parameter.
    Str(String),
    /// A binary blob (typically a serialized vector). Binary params are **not**
    /// substituted into the SQL text; they are kept as placeholders for the
    /// downstream executor.
    Bytes(Vec<u8>),
}

/// SQL-like query for Redis Search.
///
/// Holds a SQL `SELECT` statement and optional named parameters. Parameter
/// placeholders use the `:name` syntax (e.g. `:id`, `:product_id`).
///
/// # Example
///
/// ```
/// use redis_vl::{SQLQuery, SqlParam};
///
/// let query = SQLQuery::new("SELECT * FROM idx WHERE price > :min_price")
///     .with_param("min_price", SqlParam::Float(99.99));
///
/// assert!(!query.substituted_sql().contains(":min_price"));
/// assert!(query.substituted_sql().contains("99.99"));
/// ```
#[derive(Debug, Clone)]
pub struct SQLQuery {
    sql: String,
    params: HashMap<String, SqlParam>,
}

impl SQLQuery {
    /// Creates an SQL query wrapper with no parameters.
    pub fn new(sql: impl Into<String>) -> Self {
        Self {
            sql: sql.into(),
            params: HashMap::new(),
        }
    }

    /// Creates an SQL query with pre-populated parameters.
    pub fn with_params(sql: impl Into<String>, params: HashMap<String, SqlParam>) -> Self {
        Self {
            sql: sql.into(),
            params,
        }
    }

    /// Adds a single named parameter.
    pub fn with_param(mut self, name: impl Into<String>, value: SqlParam) -> Self {
        self.params.insert(name.into(), value);
        self
    }

    /// Returns the raw SQL string.
    pub fn sql(&self) -> &str {
        &self.sql
    }

    /// Returns the parameter map.
    pub fn params_map(&self) -> &HashMap<String, SqlParam> {
        &self.params
    }

    /// Returns the SQL string with non-binary parameters substituted.
    ///
    /// Uses a token-based approach: splits the SQL on `:param` boundaries to
    /// prevent partial matching (`:id` inside `:product_id` stays intact).
    /// Single quotes in string values are SQL-escaped (`'` → `''`).
    pub fn substituted_sql(&self) -> String {
        substitute_params(&self.sql, &self.params)
    }
}

impl QueryString for SQLQuery {
    fn to_redis_query(&self) -> String {
        self.substituted_sql()
    }
}

/// Substitutes `:name` parameter placeholders in `sql` using `params`.
///
/// Uses token-based splitting on `:identifier` boundaries so that `:id`
/// placeholders are never partially matched inside `:product_id`.
///
/// - `Int` and `Float` values are stringified directly.
/// - `Str` values are wrapped in single quotes with `'` escaped to `''`.
/// - `Bytes` values are left as their original placeholder (for downstream
///   executor handling).
fn substitute_params(sql: &str, params: &HashMap<String, SqlParam>) -> String {
    if params.is_empty() {
        return sql.to_owned();
    }

    // Split on `:identifier` tokens, keeping delimiters.
    let mut result = String::with_capacity(sql.len());
    let bytes = sql.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        if bytes[i] == b':' && i + 1 < len && is_ident_start(bytes[i + 1]) {
            // Found a potential parameter placeholder — consume the identifier.
            let start = i + 1;
            let mut end = start;
            while end < len && is_ident_continue(bytes[end]) {
                end += 1;
            }
            let key = &sql[start..end];
            if let Some(param) = params.get(key) {
                match param {
                    SqlParam::Int(v) => {
                        result.push_str(&v.to_string());
                    }
                    SqlParam::Float(v) => {
                        result.push_str(&v.to_string());
                    }
                    SqlParam::Str(v) => {
                        result.push('\'');
                        result.push_str(&v.replace('\'', "''"));
                        result.push('\'');
                    }
                    SqlParam::Bytes(_) => {
                        // Keep the original placeholder for binary params.
                        result.push(':');
                        result.push_str(key);
                    }
                }
            } else {
                // Unknown placeholder — keep as-is.
                result.push(':');
                result.push_str(key);
            }
            i = end;
        } else {
            result.push(sql[i..].chars().next().unwrap());
            i += sql[i..].chars().next().unwrap().len_utf8();
        }
    }

    result
}

fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

fn is_ident_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Parameter substitution: partial matching prevention ----

    #[test]
    fn similar_param_names_no_partial_match() {
        let query = SQLQuery::with_params(
            "SELECT * FROM idx WHERE id = :id AND product_id = :product_id",
            HashMap::from([
                ("id".to_owned(), SqlParam::Int(123)),
                ("product_id".to_owned(), SqlParam::Int(456)),
            ]),
        );
        let substituted = query.substituted_sql();
        assert!(substituted.contains("id = 123"));
        assert!(substituted.contains("product_id = 456"));
        assert!(!substituted.contains("product_123"));
    }

    #[test]
    fn prefix_param_names() {
        let query = SQLQuery::with_params(
            "SELECT * FROM idx WHERE user = :user AND user_id = :user_id AND user_name = :user_name",
            HashMap::from([
                ("user".to_owned(), SqlParam::Str("alice".to_owned())),
                ("user_id".to_owned(), SqlParam::Int(42)),
                (
                    "user_name".to_owned(),
                    SqlParam::Str("Alice Smith".to_owned()),
                ),
            ]),
        );
        let substituted = query.substituted_sql();
        assert!(substituted.contains("user = 'alice'"));
        assert!(substituted.contains("user_id = 42"));
        assert!(substituted.contains("user_name = 'Alice Smith'"));
        assert!(!substituted.contains("'alice'_id"));
        assert!(!substituted.contains("'alice'_name"));
    }

    #[test]
    fn suffix_param_names() {
        let query = SQLQuery::with_params(
            "SELECT * FROM idx WHERE vec = :vec AND query_vec = :query_vec",
            HashMap::from([
                ("vec".to_owned(), SqlParam::Float(1.0)),
                ("query_vec".to_owned(), SqlParam::Float(2.0)),
            ]),
        );
        let substituted = query.substituted_sql();
        assert!(substituted.contains("vec = 1") || substituted.contains("vec = 1.0"));
        assert!(substituted.contains("query_vec = 2") || substituted.contains("query_vec = 2.0"));
    }

    // ---- Parameter substitution: quote escaping ----

    #[test]
    fn single_quote_in_value() {
        let query = SQLQuery::new("SELECT * FROM idx WHERE name = :name")
            .with_param("name", SqlParam::Str("O'Brien".to_owned()));
        let substituted = query.substituted_sql();
        assert!(substituted.contains("name = 'O''Brien'"));
    }

    #[test]
    fn multiple_quotes_in_value() {
        let query = SQLQuery::new("SELECT * FROM idx WHERE phrase = :phrase")
            .with_param("phrase", SqlParam::Str("It's a 'test' string".to_owned()));
        let substituted = query.substituted_sql();
        assert!(substituted.contains("phrase = 'It''s a ''test'' string'"));
    }

    #[test]
    fn apostrophe_names() {
        let cases = [
            ("McDonald's", "'McDonald''s'"),
            ("O'Reilly", "'O''Reilly'"),
            ("D'Angelo", "'D''Angelo'"),
        ];
        for (name, expected) in cases {
            let query = SQLQuery::new("SELECT * FROM idx WHERE name = :name")
                .with_param("name", SqlParam::Str(name.to_owned()));
            let substituted = query.substituted_sql();
            assert!(
                substituted.contains(&format!("name = {expected}")),
                "Failed for {name}: got {substituted}"
            );
        }
    }

    // ---- Edge cases ----

    #[test]
    fn multiple_occurrences_same_param() {
        let query = SQLQuery::new("SELECT * FROM idx WHERE category = :cat OR subcategory = :cat")
            .with_param("cat", SqlParam::Str("electronics".to_owned()));
        let substituted = query.substituted_sql();
        assert_eq!(substituted.matches("'electronics'").count(), 2);
    }

    #[test]
    fn empty_string_value() {
        let query = SQLQuery::new("SELECT * FROM idx WHERE name = :name")
            .with_param("name", SqlParam::Str(String::new()));
        let substituted = query.substituted_sql();
        assert!(substituted.contains("name = ''"));
    }

    #[test]
    fn numeric_types() {
        let query = SQLQuery::with_params(
            "SELECT * FROM idx WHERE count = :count AND price = :price",
            HashMap::from([
                ("count".to_owned(), SqlParam::Int(42)),
                ("price".to_owned(), SqlParam::Float(99.99)),
            ]),
        );
        let substituted = query.substituted_sql();
        assert!(substituted.contains("count = 42"));
        assert!(substituted.contains("price = 99.99"));
    }

    #[test]
    fn bytes_param_not_substituted() {
        let query = SQLQuery::new("SELECT * FROM idx WHERE embedding = :vec")
            .with_param("vec", SqlParam::Bytes(vec![0x00, 0x01, 0x02, 0x03]));
        let substituted = query.substituted_sql();
        assert!(substituted.contains(":vec"));
    }

    #[test]
    fn special_characters_in_value() {
        let specials = [
            "hello@world.com",
            "path/to/file",
            "price: $100",
            "regex.*pattern",
            "back\\slash",
        ];
        for value in specials {
            let query = SQLQuery::new("SELECT * FROM idx WHERE field = :field")
                .with_param("field", SqlParam::Str(value.to_owned()));
            let substituted = query.substituted_sql();
            assert!(
                !substituted.contains(":field"),
                "Failed to substitute for value: {value}"
            );
        }
    }

    #[test]
    fn no_params_returns_original() {
        let query = SQLQuery::new("SELECT * FROM idx");
        assert_eq!(query.substituted_sql(), "SELECT * FROM idx");
    }

    #[test]
    fn unknown_placeholder_kept() {
        let query = SQLQuery::new("SELECT * FROM idx WHERE x = :unknown")
            .with_param("other", SqlParam::Int(1));
        assert!(query.substituted_sql().contains(":unknown"));
    }

    #[test]
    fn with_param_builder_pattern() {
        let query = SQLQuery::new("SELECT * FROM idx WHERE a = :a AND b = :b")
            .with_param("a", SqlParam::Int(1))
            .with_param("b", SqlParam::Str("hello".to_owned()));
        let sub = query.substituted_sql();
        assert!(sub.contains("a = 1"));
        assert!(sub.contains("b = 'hello'"));
    }

    #[test]
    fn sql_accessor() {
        let query = SQLQuery::new("SELECT 1");
        assert_eq!(query.sql(), "SELECT 1");
    }

    #[test]
    fn params_map_accessor() {
        let query = SQLQuery::new("SELECT 1").with_param("x", SqlParam::Int(42));
        assert_eq!(query.params_map().len(), 1);
    }
}
