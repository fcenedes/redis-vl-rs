//! SQL-like query type for Redis Search parity with Python `redisvl.query.SQLQuery`.
//!
//! This module mirrors the upstream Python `SQLQuery` class that accepts SQL-like
//! syntax and optional parameters. It implements a lightweight SQL-to-Redis-Search
//! translation layer that converts basic `SELECT` statements into `FT.SEARCH`
//! filter syntax.
//!
//! ## What is implemented
//!
//! - `SQLQuery` type holding the raw SQL string and optional parameters
//! - `SqlParam` enum for typed parameter values (string, numeric, binary)
//! - Token-based parameter substitution that prevents partial-match bugs
//!   (`:id` won't clobber `:product_id`) and escapes single quotes in strings
//! - `QueryString` trait implementation so `SQLQuery` can be passed to
//!   `SearchIndex::query()` / `AsyncSearchIndex::query()`
//! - SQL→Redis translation for non-aggregate `SELECT` queries:
//!   - `WHERE` clauses with tag `=`, `!=`, `IN`, `NOT IN`; numeric `=`, `!=`,
//!     `<`, `>`, `<=`, `>=`, `BETWEEN`; text `=`, `!=`; `LIKE` / `NOT LIKE`
//!   - `AND` and `OR` combinators with correct precedence (AND binds tighter)
//!   - ISO date literal parsing (`'2024-01-01'`, `'2024-01-15T10:30:00'`)
//!     in comparison and `BETWEEN` clauses
//!   - `ORDER BY field ASC|DESC`
//!   - `LIMIT n [OFFSET m]`
//!   - `SELECT field1, field2` → `RETURN` field projection
//!
//! ## What is out of scope (not yet implemented)
//!
//! - Aggregate queries (`COUNT`, `AVG`, `SUM`, `GROUP BY`) → these require
//!   `FT.AGGREGATE`, a completely different Redis command
//! - Vector search functions (`cosine_distance()`, `vector_distance()`)
//! - GEO functions (`geo_distance()`)
//! - Date functions (`YEAR()` in SELECT, `GROUP BY YEAR()`)
//! - `IS NULL` / `IS NOT NULL`
//! - Phrase-level stopword handling

use std::collections::HashMap;

use super::{QueryLimit, QueryString, SortBy, SortDirection};

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
/// When used with `SearchIndex::query()`, the SQL is parsed and translated
/// into a Redis Search `FT.SEARCH` filter string. Non-aggregate queries with
/// `WHERE`, `ORDER BY`, `LIMIT`, and `OFFSET` clauses are supported.
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

    /// Parses the SQL statement into a [`ParsedSelect`].
    ///
    /// Returns `None` if the SQL cannot be parsed (e.g. aggregate or
    /// unsupported syntax). In that case, the raw substituted SQL is used
    /// as the Redis query string (fallback behaviour).
    fn parsed(&self) -> Option<ParsedSelect> {
        parse_select(&self.substituted_sql())
    }
}

impl QueryString for SQLQuery {
    fn to_redis_query(&self) -> String {
        if let Some(parsed) = self.parsed() {
            parsed.filter_string()
        } else {
            // Fallback: return raw substituted SQL (backwards-compatible).
            self.substituted_sql()
        }
    }

    fn return_fields(&self) -> Vec<String> {
        self.parsed().map(|p| p.return_fields).unwrap_or_default()
    }

    fn sort_by(&self) -> Option<SortBy> {
        self.parsed().and_then(|p| p.sort_by)
    }

    fn limit(&self) -> Option<QueryLimit> {
        self.parsed().and_then(|p| p.limit)
    }

    fn should_unpack_json(&self) -> bool {
        // Unpack JSON when no explicit field projection (SELECT *).
        self.parsed()
            .map(|p| p.return_fields.is_empty())
            .unwrap_or(false)
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

// ---------------------------------------------------------------------------
// Lightweight SQL SELECT parser → Redis Search translation
// ---------------------------------------------------------------------------

/// A parsed SQL `SELECT` statement.
#[derive(Debug, Clone)]
struct ParsedSelect {
    /// Field names to project (empty = `SELECT *`).
    return_fields: Vec<String>,
    /// Redis Search filter string derived from the `WHERE` clause.
    where_filter: Option<String>,
    /// Sort specification from `ORDER BY`.
    sort_by: Option<SortBy>,
    /// Limit specification from `LIMIT … [OFFSET …]`.
    limit: Option<QueryLimit>,
}

impl ParsedSelect {
    /// Returns the Redis Search query string used by `FT.SEARCH`.
    fn filter_string(&self) -> String {
        self.where_filter.clone().unwrap_or_else(|| "*".to_owned())
    }
}

/// Tokenise and parse a SQL `SELECT` statement.
///
/// Returns `None` for unsupported syntax (aggregates, sub-queries, etc.).
fn parse_select(sql: &str) -> Option<ParsedSelect> {
    let tokens = tokenize(sql);
    if tokens.is_empty() {
        return None;
    }
    let mut pos = 0;

    // SELECT
    if !tok_eq(&tokens, pos, "SELECT") {
        return None;
    }
    pos += 1;

    // Bail on aggregate functions in SELECT list.
    for tok in &tokens {
        let upper = tok.to_ascii_uppercase();
        if matches!(
            upper.as_str(),
            "COUNT"
                | "AVG"
                | "SUM"
                | "MIN"
                | "MAX"
                | "STDDEV"
                | "QUANTILE"
                | "COUNT_DISTINCT"
                | "ARRAY_AGG"
                | "FIRST_VALUE"
        ) {
            return None;
        }
    }

    // Bail on vector/geo functions.
    for tok in &tokens {
        let lower = tok.to_ascii_lowercase();
        if lower == "cosine_distance" || lower == "vector_distance" || lower == "geo_distance" {
            return None;
        }
    }

    // Parse field list.
    let mut return_fields = Vec::new();
    if tok_eq(&tokens, pos, "*") {
        pos += 1;
    } else {
        loop {
            if pos >= tokens.len() {
                return None;
            }
            let field = &tokens[pos];
            if field.eq_ignore_ascii_case("FROM") {
                break;
            }
            // Skip aliases: field AS alias
            if !field.eq_ignore_ascii_case(",") && !field.eq_ignore_ascii_case("AS") {
                // Check if the previous token was AS (skip alias names)
                if pos > 0 && tokens[pos - 1].eq_ignore_ascii_case("AS") {
                    // This is an alias name, skip it
                } else {
                    return_fields.push(field.to_string());
                }
            }
            pos += 1;
        }
    }

    // FROM
    if !tok_eq(&tokens, pos, "FROM") {
        return None;
    }
    pos += 1;
    // Skip table name.
    if pos >= tokens.len() {
        return None;
    }
    pos += 1;

    // WHERE, ORDER BY, LIMIT, OFFSET — all optional.
    let mut where_filter: Option<String> = None;
    let mut sort_by: Option<SortBy> = None;
    let mut limit: Option<QueryLimit> = None;

    while pos < tokens.len() {
        if tok_eq(&tokens, pos, "WHERE") {
            pos += 1;
            let (filter_str, next) = parse_where_clause(&tokens, pos)?;
            where_filter = Some(filter_str);
            pos = next;
        } else if tok_eq(&tokens, pos, "ORDER") {
            if !tok_eq(&tokens, pos + 1, "BY") {
                return None;
            }
            pos += 2;
            if pos >= tokens.len() {
                return None;
            }
            let field = tokens[pos].clone();
            pos += 1;
            let direction = if tok_eq(&tokens, pos, "DESC") {
                pos += 1;
                SortDirection::Desc
            } else {
                if tok_eq(&tokens, pos, "ASC") {
                    pos += 1;
                }
                SortDirection::Asc
            };
            sort_by = Some(SortBy { field, direction });
        } else if tok_eq(&tokens, pos, "LIMIT") {
            pos += 1;
            let num = parse_usize(&tokens, pos)?;
            pos += 1;
            let offset = if tok_eq(&tokens, pos, "OFFSET") {
                pos += 1;
                let off = parse_usize(&tokens, pos)?;
                pos += 1;
                off
            } else {
                0
            };
            limit = Some(QueryLimit { offset, num });
        } else {
            // Unknown clause — skip.
            pos += 1;
        }
    }

    Some(ParsedSelect {
        return_fields,
        where_filter,
        sort_by,
        limit,
    })
}

/// Parse a WHERE clause starting at `pos`. Returns the Redis filter string and
/// the position after the last consumed token.
///
/// Supports `AND` and `OR` combinators with correct precedence: `AND` binds
/// tighter than `OR`, so `a AND b OR c AND d` is parsed as `(a b) | (c d)`.
fn parse_where_clause(tokens: &[String], mut pos: usize) -> Option<(String, usize)> {
    // We collect OR-separated groups of AND-joined conditions.
    let mut or_groups: Vec<Vec<String>> = Vec::new();
    let mut current_and_group: Vec<String> = Vec::new();

    loop {
        if pos >= tokens.len() {
            break;
        }
        // Stop at ORDER / LIMIT / GROUP (not part of WHERE).
        let upper = tokens[pos].to_ascii_uppercase();
        if matches!(upper.as_str(), "ORDER" | "LIMIT" | "GROUP" | "HAVING") {
            break;
        }
        // AND combinator — continue in current group.
        if upper == "AND" {
            pos += 1;
            continue;
        }
        // OR combinator — start a new group.
        if upper == "OR" {
            pos += 1;
            or_groups.push(std::mem::take(&mut current_and_group));
            continue;
        }

        let (filter, next) = parse_single_condition(tokens, pos)?;
        current_and_group.push(filter);
        pos = next;
    }

    // Push the last AND group.
    if !current_and_group.is_empty() {
        or_groups.push(current_and_group);
    }

    if or_groups.is_empty() {
        return Some(("*".to_owned(), pos));
    }

    // Build the filter string.
    let group_strs: Vec<String> = or_groups
        .into_iter()
        .map(|g| {
            if g.len() == 1 {
                g.into_iter().next().unwrap()
            } else {
                format!("({})", g.join(" "))
            }
        })
        .collect();

    let filter = if group_strs.len() == 1 {
        group_strs.into_iter().next().unwrap()
    } else {
        // OR-combine: (a | b) in Redis Search syntax.
        format!("({})", group_strs.join(" | "))
    };

    Some((filter, pos))
}

/// Parse a single WHERE condition starting at `pos`.
///
/// Returns the Redis filter string for this condition and the position after
/// the last consumed token.
fn parse_single_condition(tokens: &[String], mut pos: usize) -> Option<(String, usize)> {
    let field = &tokens[pos];
    pos += 1;
    if pos >= tokens.len() {
        return None;
    }

    let op = &tokens[pos];
    pos += 1;

    // BETWEEN handling: field BETWEEN lo AND hi
    if op.eq_ignore_ascii_case("BETWEEN") {
        let lo = parse_numeric_or_date_literal(tokens, pos)?;
        pos += 1;
        if !tok_eq(tokens, pos, "AND") {
            return None;
        }
        pos += 1;
        let hi = parse_numeric_or_date_literal(tokens, pos)?;
        pos += 1;
        return Some((
            format!("@{}:[{} {}]", field, format_num(lo), format_num(hi)),
            pos,
        ));
    }

    // NOT IN handling: field NOT IN ('a', 'b')
    if op.eq_ignore_ascii_case("NOT") && tok_eq(tokens, pos, "IN") {
        pos += 1; // skip "IN"
        if !tok_eq(tokens, pos, "(") {
            return None;
        }
        pos += 1;
        let mut vals = Vec::new();
        loop {
            if pos >= tokens.len() {
                return None;
            }
            if tokens[pos] == ")" {
                pos += 1;
                break;
            }
            if tokens[pos] == "," {
                pos += 1;
                continue;
            }
            vals.push(unquote(&tokens[pos]));
            pos += 1;
        }
        let escaped: Vec<String> = vals.iter().map(|v| escape_tag(v)).collect();
        return Some((format!("(-@{}:{{{}}})", field, escaped.join("|")), pos));
    }

    // IN handling: field IN ('a', 'b')
    if op.eq_ignore_ascii_case("IN") {
        if !tok_eq(tokens, pos, "(") {
            return None;
        }
        pos += 1;
        let mut vals = Vec::new();
        loop {
            if pos >= tokens.len() {
                return None;
            }
            if tokens[pos] == ")" {
                pos += 1;
                break;
            }
            if tokens[pos] == "," {
                pos += 1;
                continue;
            }
            vals.push(unquote(&tokens[pos]));
            pos += 1;
        }
        let escaped: Vec<String> = vals.iter().map(|v| escape_tag(v)).collect();
        return Some((format!("@{}:{{{}}}", field, escaped.join("|")), pos));
    }

    // LIKE handling: field LIKE 'pattern'
    if op.eq_ignore_ascii_case("LIKE") {
        if pos >= tokens.len() {
            return None;
        }
        let pattern = unquote(&tokens[pos]);
        pos += 1;
        let redis_pattern = sql_like_to_redis(&pattern);
        return Some((format!("@{}:({})", field, redis_pattern), pos));
    }

    // NOT LIKE handling: field NOT LIKE 'pattern'
    if op.eq_ignore_ascii_case("NOT") && tok_eq(tokens, pos, "LIKE") {
        pos += 1; // skip "LIKE"
        if pos >= tokens.len() {
            return None;
        }
        let pattern = unquote(&tokens[pos]);
        pos += 1;
        let redis_pattern = sql_like_to_redis(&pattern);
        return Some((format!("(-@{}:({}))", field, redis_pattern), pos));
    }

    // !=
    if op == "!=" {
        if pos >= tokens.len() {
            return None;
        }
        let value = unquote(&tokens[pos]);
        pos += 1;
        if is_numeric_str(&value) {
            let n: f64 = value.parse().ok()?;
            return Some((
                format!("(-@{}:[{} {}])", field, format_num(n), format_num(n)),
                pos,
            ));
        } else if let Some(ts) = try_parse_date(&value) {
            return Some((
                format!("(-@{}:[{} {}])", field, format_num(ts), format_num(ts)),
                pos,
            ));
        } else {
            // Tag or text negation.
            return Some((format!("(-@{}:{{{}}})", field, escape_tag(&value)), pos));
        }
    }

    // Comparison operators: =, <, >, <=, >=
    if pos >= tokens.len() {
        return None;
    }

    // Handle two-character ops: <=, >=
    let (real_op, value_str) = if (op == "<" || op == ">") && tokens[pos] == "=" {
        let combined = format!("{}=", op);
        pos += 1;
        if pos >= tokens.len() {
            return None;
        }
        let v = unquote(&tokens[pos]);
        pos += 1;
        (combined, v)
    } else {
        let v = unquote(&tokens[pos]);
        pos += 1;
        (op.clone(), v)
    };

    let filter = match real_op.as_str() {
        "=" => {
            if is_numeric_str(&value_str) {
                let n: f64 = value_str.parse().ok()?;
                format!("@{}:[{} {}]", field, format_num(n), format_num(n))
            } else if let Some(ts) = try_parse_date(&value_str) {
                format!("@{}:[{} {}]", field, format_num(ts), format_num(ts))
            } else {
                // Could be tag or text. Use tag syntax for simple values.
                // For text with wildcards or multi-word, use text syntax.
                let val = value_str.clone();
                if val.contains('*') || val.contains('%') {
                    // Wildcard/fuzzy → text field search.
                    format!("@{}:({})", field, val)
                } else if val.contains(' ') {
                    // Multi-word → phrase search.
                    format!("@{}:(\"{}\")", field, val)
                } else {
                    // Single term → tag match.
                    format!("@{}:{{{}}}", field, escape_tag(&val))
                }
            }
        }
        "<" => {
            let n = parse_num_or_date(&value_str)?;
            format!("@{}:[-inf ({}]", field, format_num(n))
        }
        ">" => {
            let n = parse_num_or_date(&value_str)?;
            format!("@{}:[({} +inf]", field, format_num(n))
        }
        "<=" => {
            let n = parse_num_or_date(&value_str)?;
            format!("@{}:[-inf {}]", field, format_num(n))
        }
        ">=" => {
            let n = parse_num_or_date(&value_str)?;
            format!("@{}:[{} +inf]", field, format_num(n))
        }
        _ => return None,
    };

    Some((filter, pos))
}

// ---------------------------------------------------------------------------
// SQL tokenizer
// ---------------------------------------------------------------------------

/// Tokenize SQL into a sequence of meaningful tokens.
///
/// Handles single-quoted strings, double-quoted identifiers, numbers, identifiers,
/// and single-character operators.
fn tokenize(sql: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = sql.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        // Skip whitespace.
        if chars[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }
        // Single-quoted string literal.
        if chars[i] == '\'' {
            let mut s = String::new();
            s.push('\'');
            i += 1;
            while i < len {
                if chars[i] == '\'' {
                    if i + 1 < len && chars[i + 1] == '\'' {
                        s.push('\'');
                        s.push('\'');
                        i += 2;
                    } else {
                        break;
                    }
                } else {
                    s.push(chars[i]);
                    i += 1;
                }
            }
            s.push('\'');
            if i < len {
                i += 1;
            }
            tokens.push(s);
            continue;
        }
        // Identifier or keyword.
        if chars[i].is_ascii_alphabetic() || chars[i] == '_' {
            let start = i;
            while i < len && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            tokens.push(chars[start..i].iter().collect());
            continue;
        }
        // Number (with optional negative sign or decimal point).
        if chars[i].is_ascii_digit()
            || (chars[i] == '-' && i + 1 < len && chars[i + 1].is_ascii_digit())
        {
            let start = i;
            if chars[i] == '-' {
                i += 1;
            }
            while i < len && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            tokens.push(chars[start..i].iter().collect());
            continue;
        }
        // Two-character operators: !=, <=, >=.
        if i + 1 < len {
            let two: String = chars[i..i + 2].iter().collect();
            if two == "!=" || two == "<=" || two == ">=" {
                tokens.push(two);
                i += 2;
                continue;
            }
        }
        // Single-character operators/punctuation.
        tokens.push(chars[i].to_string());
        i += 1;
    }
    tokens
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Case-insensitive token match at position `pos`.
fn tok_eq(tokens: &[String], pos: usize, expected: &str) -> bool {
    tokens
        .get(pos)
        .map_or(false, |t| t.eq_ignore_ascii_case(expected))
}

/// Parse a usize from a token at `pos`.
fn parse_usize(tokens: &[String], pos: usize) -> Option<usize> {
    tokens.get(pos)?.parse().ok()
}

/// Parse a numeric literal or ISO date string from a token at `pos`.
///
/// This extends `parse_numeric_literal` to handle date strings like
/// `'2024-01-01'` by converting them to Unix timestamps.
fn parse_numeric_or_date_literal(tokens: &[String], pos: usize) -> Option<f64> {
    let tok = tokens.get(pos)?;
    let s = unquote(tok);
    if let Ok(n) = s.parse::<f64>() {
        Some(n)
    } else {
        try_parse_date(&s)
    }
}

/// Try to parse a string as a number; if that fails, try as an ISO date.
fn parse_num_or_date(s: &str) -> Option<f64> {
    if let Ok(n) = s.parse::<f64>() {
        Some(n)
    } else {
        try_parse_date(s)
    }
}

/// Try to parse an ISO 8601 date string (`YYYY-MM-DD` or `YYYY-MM-DDTHH:MM:SS`)
/// and return the Unix timestamp as `f64`.
///
/// This mirrors the upstream Python `sql-redis` library's date literal handling.
fn try_parse_date(s: &str) -> Option<f64> {
    // Try YYYY-MM-DD
    if s.len() == 10 && s.as_bytes().get(4) == Some(&b'-') && s.as_bytes().get(7) == Some(&b'-') {
        let year: i32 = s[0..4].parse().ok()?;
        let month: u32 = s[5..7].parse().ok()?;
        let day: u32 = s[8..10].parse().ok()?;
        if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
            return None;
        }
        // Compute days from Unix epoch (1970-01-01) using a simplified calendar.
        let ts = date_to_unix_timestamp(year, month, day)?;
        return Some(ts as f64);
    }
    // Try YYYY-MM-DDTHH:MM:SS
    if s.len() >= 19 && (s.as_bytes().get(10) == Some(&b'T') || s.as_bytes().get(10) == Some(&b' '))
    {
        let year: i32 = s[0..4].parse().ok()?;
        let month: u32 = s[5..7].parse().ok()?;
        let day: u32 = s[8..10].parse().ok()?;
        let hour: u32 = s[11..13].parse().ok()?;
        let min: u32 = s[14..16].parse().ok()?;
        let sec: u32 = s[17..19].parse().ok()?;
        if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
            return None;
        }
        if hour > 23 || min > 59 || sec > 59 {
            return None;
        }
        let day_ts = date_to_unix_timestamp(year, month, day)?;
        let ts = day_ts + (hour as i64) * 3600 + (min as i64) * 60 + (sec as i64);
        return Some(ts as f64);
    }
    None
}

/// Convert a date (year, month, day) to a Unix timestamp (seconds since 1970-01-01 UTC).
fn date_to_unix_timestamp(year: i32, month: u32, day: u32) -> Option<i64> {
    // Days in months (non-leap).
    const DAYS_IN_MONTH: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    fn is_leap(y: i32) -> bool {
        (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
    }

    // Count days from 1970-01-01 to the given date.
    let mut days: i64 = 0;

    // Years
    if year >= 1970 {
        for y in 1970..year {
            days += if is_leap(y) { 366 } else { 365 };
        }
    } else {
        for y in year..1970 {
            days -= if is_leap(y) { 366 } else { 365 };
        }
    }

    // Months
    for m in 1..month {
        let mut d = DAYS_IN_MONTH[(m - 1) as usize];
        if m == 2 && is_leap(year) {
            d += 1;
        }
        days += d as i64;
    }

    // Days (1-based, so day 1 = 0 extra days)
    days += (day as i64) - 1;

    Some(days * 86400)
}

/// Convert a SQL `LIKE` pattern to Redis Search text syntax.
///
/// - `%` is mapped to `*` (match zero or more characters)
/// - `_` is left as-is (Redis does not have single-char wildcard; best effort)
///
/// Examples:
/// - `laptop%` → `laptop*`
/// - `%laptop` → `*laptop`
/// - `%laptop%` → `*laptop*`
fn sql_like_to_redis(pattern: &str) -> String {
    pattern.replace('%', "*")
}

/// Remove surrounding single quotes from a string literal.
fn unquote(s: &str) -> String {
    if s.len() >= 2 && s.starts_with('\'') && s.ends_with('\'') {
        let inner = &s[1..s.len() - 1];
        // Unescape double-quotes: '' → '
        inner.replace("''", "'")
    } else {
        s.to_string()
    }
}

/// Escape tag value characters for Redis Search `@field:{value}` syntax.
fn escape_tag(value: &str) -> String {
    value
        .chars()
        .flat_map(|ch| {
            if matches!(ch, ' ' | '$' | ':' | '&' | '/' | '-' | '.' | '*') {
                vec!['\\', ch]
            } else {
                vec![ch]
            }
        })
        .collect()
}

/// Check if a string looks like a numeric value.
fn is_numeric_str(s: &str) -> bool {
    s.parse::<f64>().is_ok()
}

/// Format a number: drop fractional part if it's .0.
fn format_num(n: f64) -> String {
    if n.fract() == 0.0 {
        format!("{:.0}", n)
    } else {
        n.to_string()
    }
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

    // ---- SQL→Redis translation tests ----

    #[test]
    fn select_star_no_where_produces_wildcard() {
        let query = SQLQuery::new("SELECT * FROM products");
        assert_eq!(query.to_redis_query(), "*");
    }

    #[test]
    fn select_specific_fields_sets_return_fields() {
        let query = SQLQuery::new("SELECT title, price FROM products");
        assert_eq!(query.to_redis_query(), "*");
        assert_eq!(query.return_fields(), vec!["title", "price"]);
    }

    #[test]
    fn where_tag_equals() {
        let query = SQLQuery::new("SELECT * FROM products WHERE category = 'electronics'");
        assert_eq!(query.to_redis_query(), "@category:{electronics}");
    }

    #[test]
    fn where_tag_not_equals() {
        let query = SQLQuery::new("SELECT * FROM products WHERE category != 'electronics'");
        assert_eq!(query.to_redis_query(), "(-@category:{electronics})");
    }

    #[test]
    fn where_tag_in() {
        let query =
            SQLQuery::new("SELECT * FROM products WHERE category IN ('books', 'accessories')");
        assert_eq!(query.to_redis_query(), "@category:{books|accessories}");
    }

    #[test]
    fn where_numeric_less_than() {
        let query = SQLQuery::new("SELECT * FROM products WHERE price < 50");
        assert_eq!(query.to_redis_query(), "@price:[-inf (50]");
    }

    #[test]
    fn where_numeric_greater_than() {
        let query = SQLQuery::new("SELECT * FROM products WHERE price > 100");
        assert_eq!(query.to_redis_query(), "@price:[(100 +inf]");
    }

    #[test]
    fn where_numeric_equals() {
        let query = SQLQuery::new("SELECT * FROM products WHERE price = 45");
        assert_eq!(query.to_redis_query(), "@price:[45 45]");
    }

    #[test]
    fn where_numeric_not_equals() {
        let query = SQLQuery::new("SELECT * FROM products WHERE price != 45");
        assert_eq!(query.to_redis_query(), "(-@price:[45 45])");
    }

    #[test]
    fn where_numeric_lte() {
        let query = SQLQuery::new("SELECT * FROM products WHERE price <= 50");
        assert_eq!(query.to_redis_query(), "@price:[-inf 50]");
    }

    #[test]
    fn where_numeric_gte() {
        let query = SQLQuery::new("SELECT * FROM products WHERE price >= 25");
        assert_eq!(query.to_redis_query(), "@price:[25 +inf]");
    }

    #[test]
    fn where_between() {
        let query = SQLQuery::new("SELECT * FROM products WHERE price BETWEEN 40 AND 60");
        assert_eq!(query.to_redis_query(), "@price:[40 60]");
    }

    #[test]
    fn where_combined_and() {
        let query =
            SQLQuery::new("SELECT * FROM products WHERE category = 'electronics' AND price < 100");
        assert_eq!(
            query.to_redis_query(),
            "(@category:{electronics} @price:[-inf (100])"
        );
    }

    #[test]
    fn order_by_asc() {
        let query = SQLQuery::new("SELECT title, price FROM products ORDER BY price ASC");
        let sb = query.sort_by().expect("sort_by should be set");
        assert_eq!(sb.field, "price");
        assert!(matches!(sb.direction, SortDirection::Asc));
    }

    #[test]
    fn order_by_desc() {
        let query = SQLQuery::new("SELECT title, price FROM products ORDER BY price DESC");
        let sb = query.sort_by().expect("sort_by should be set");
        assert_eq!(sb.field, "price");
        assert!(matches!(sb.direction, SortDirection::Desc));
    }

    #[test]
    fn limit_clause() {
        let query = SQLQuery::new("SELECT title FROM products LIMIT 3");
        let lim = query.limit().expect("limit should be set");
        assert_eq!(lim.num, 3);
        assert_eq!(lim.offset, 0);
    }

    #[test]
    fn limit_with_offset() {
        let query = SQLQuery::new("SELECT title FROM products ORDER BY price ASC LIMIT 3 OFFSET 3");
        let lim = query.limit().expect("limit should be set");
        assert_eq!(lim.num, 3);
        assert_eq!(lim.offset, 3);
    }

    #[test]
    fn where_with_order_and_limit() {
        let query = SQLQuery::new(
            "SELECT title, price FROM products WHERE category = 'electronics' ORDER BY price ASC LIMIT 5",
        );
        assert_eq!(query.to_redis_query(), "@category:{electronics}");
        assert_eq!(query.return_fields(), vec!["title", "price"]);
        let sb = query.sort_by().expect("sort_by");
        assert_eq!(sb.field, "price");
        let lim = query.limit().expect("limit");
        assert_eq!(lim.num, 5);
    }

    #[test]
    fn aggregate_query_returns_raw_sql_fallback() {
        // Aggregate queries are not translated—they fall back to the raw SQL.
        let query = SQLQuery::new("SELECT COUNT(*) as total FROM products");
        let result = query.to_redis_query();
        // Parsed as None → fallback to substituted_sql.
        assert!(result.contains("COUNT"));
    }

    #[test]
    fn text_equality_single_word() {
        let query = SQLQuery::new("SELECT * FROM products WHERE title = 'laptop'");
        assert_eq!(query.to_redis_query(), "@title:{laptop}");
    }

    #[test]
    fn text_equality_phrase() {
        let query = SQLQuery::new("SELECT * FROM products WHERE title = 'gaming laptop'");
        assert_eq!(query.to_redis_query(), "@title:(\"gaming laptop\")");
    }

    #[test]
    fn numeric_range_with_and() {
        let query = SQLQuery::new("SELECT * FROM products WHERE price >= 25 AND price <= 50");
        assert_eq!(
            query.to_redis_query(),
            "(@price:[25 +inf] @price:[-inf 50])"
        );
    }

    #[test]
    fn should_unpack_json_for_select_star() {
        let query = SQLQuery::new("SELECT * FROM products");
        assert!(query.should_unpack_json());
    }

    #[test]
    fn should_not_unpack_json_for_field_projection() {
        let query = SQLQuery::new("SELECT title, price FROM products");
        assert!(!query.should_unpack_json());
    }

    #[test]
    fn with_param_where_tag() {
        let query = SQLQuery::new("SELECT * FROM products WHERE category = :cat")
            .with_param("cat", SqlParam::Str("electronics".to_owned()));
        assert_eq!(query.to_redis_query(), "@category:{electronics}");
    }

    #[test]
    fn with_param_where_numeric() {
        let query = SQLQuery::new("SELECT * FROM products WHERE price > :min_price")
            .with_param("min_price", SqlParam::Float(99.99));
        assert_eq!(query.to_redis_query(), "@price:[(99.99 +inf]");
    }

    // ---- OR support ----

    #[test]
    fn where_simple_or() {
        let query = SQLQuery::new(
            "SELECT * FROM products WHERE category = 'electronics' OR category = 'books'",
        );
        assert_eq!(
            query.to_redis_query(),
            "(@category:{electronics} | @category:{books})"
        );
    }

    #[test]
    fn where_or_with_three_branches() {
        let query = SQLQuery::new(
            "SELECT * FROM products WHERE category = 'electronics' OR category = 'books' OR category = 'accessories'",
        );
        assert_eq!(
            query.to_redis_query(),
            "(@category:{electronics} | @category:{books} | @category:{accessories})"
        );
    }

    #[test]
    fn where_and_binds_tighter_than_or() {
        // a AND b OR c AND d → (a b) | (c d)
        let query = SQLQuery::new(
            "SELECT * FROM products WHERE category = 'electronics' AND price > 100 OR category = 'books' AND price < 50",
        );
        assert_eq!(
            query.to_redis_query(),
            "((@category:{electronics} @price:[(100 +inf]) | (@category:{books} @price:[-inf (50]))"
        );
    }

    #[test]
    fn where_or_with_single_conditions() {
        let query = SQLQuery::new("SELECT * FROM products WHERE price < 20 OR price > 1000");
        assert_eq!(
            query.to_redis_query(),
            "(@price:[-inf (20] | @price:[(1000 +inf])"
        );
    }

    #[test]
    fn where_or_preserves_order_limit() {
        let query = SQLQuery::new(
            "SELECT title FROM products WHERE category = 'a' OR category = 'b' ORDER BY price ASC LIMIT 5",
        );
        assert_eq!(query.to_redis_query(), "(@category:{a} | @category:{b})");
        assert!(query.sort_by().is_some());
        assert_eq!(query.limit().unwrap().num, 5);
    }

    // ---- NOT IN support ----

    #[test]
    fn where_not_in() {
        let query =
            SQLQuery::new("SELECT * FROM products WHERE category NOT IN ('electronics', 'books')");
        assert_eq!(query.to_redis_query(), "(-@category:{electronics|books})");
    }

    #[test]
    fn where_not_in_combined_with_and() {
        let query = SQLQuery::new(
            "SELECT * FROM products WHERE category NOT IN ('electronics') AND price > 50",
        );
        assert_eq!(
            query.to_redis_query(),
            "((-@category:{electronics}) @price:[(50 +inf])"
        );
    }

    // ---- LIKE support ----

    #[test]
    fn where_like_prefix() {
        let query = SQLQuery::new("SELECT * FROM products WHERE title LIKE 'laptop%'");
        assert_eq!(query.to_redis_query(), "@title:(laptop*)");
    }

    #[test]
    fn where_like_suffix() {
        let query = SQLQuery::new("SELECT * FROM products WHERE title LIKE '%laptop'");
        assert_eq!(query.to_redis_query(), "@title:(*laptop)");
    }

    #[test]
    fn where_like_contains() {
        let query = SQLQuery::new("SELECT * FROM products WHERE title LIKE '%laptop%'");
        assert_eq!(query.to_redis_query(), "@title:(*laptop*)");
    }

    #[test]
    fn where_not_like() {
        let query = SQLQuery::new("SELECT * FROM products WHERE title NOT LIKE 'laptop%'");
        assert_eq!(query.to_redis_query(), "(-@title:(laptop*))");
    }

    #[test]
    fn where_like_combined_with_and() {
        let query =
            SQLQuery::new("SELECT * FROM products WHERE title LIKE 'lap%' AND price < 1000");
        assert_eq!(
            query.to_redis_query(),
            "(@title:(lap*) @price:[-inf (1000])"
        );
    }

    // ---- Date literal parsing ----

    #[test]
    fn where_date_greater_than() {
        let query = SQLQuery::new("SELECT * FROM events WHERE created_at > '2024-01-01'");
        let result = query.to_redis_query();
        // 2024-01-01 00:00:00 UTC = 1704067200
        assert_eq!(result, "@created_at:[(1704067200 +inf]");
    }

    #[test]
    fn where_date_less_than() {
        let query = SQLQuery::new("SELECT * FROM events WHERE created_at < '2024-03-31'");
        let result = query.to_redis_query();
        // 2024-03-31 00:00:00 UTC = 1711843200
        assert_eq!(result, "@created_at:[-inf (1711843200]");
    }

    #[test]
    fn where_date_between() {
        let query = SQLQuery::new(
            "SELECT * FROM events WHERE created_at BETWEEN '2024-01-01' AND '2024-03-31'",
        );
        let result = query.to_redis_query();
        assert_eq!(result, "@created_at:[1704067200 1711843200]");
    }

    #[test]
    fn where_date_gte() {
        let query = SQLQuery::new("SELECT * FROM events WHERE created_at >= '2024-06-15'");
        let result = query.to_redis_query();
        // 2024-06-15 = 1718409600
        assert_eq!(result, "@created_at:[1718409600 +inf]");
    }

    #[test]
    fn where_date_combined_with_tag() {
        let query = SQLQuery::new(
            "SELECT * FROM events WHERE category = 'meeting' AND created_at > '2024-01-01'",
        );
        let result = query.to_redis_query();
        assert_eq!(
            result,
            "(@category:{meeting} @created_at:[(1704067200 +inf])"
        );
    }

    #[test]
    fn where_datetime_with_time() {
        let query = SQLQuery::new("SELECT * FROM events WHERE created_at > '2024-01-15T10:30:00'");
        let result = query.to_redis_query();
        // 2024-01-15 00:00:00 UTC = 1705276800, + 10*3600 + 30*60 = 37800 → 1705314600
        assert_eq!(result, "@created_at:[(1705314600 +inf]");
    }

    #[test]
    fn date_to_timestamp_known_values() {
        // 1970-01-01 → 0
        assert_eq!(try_parse_date("1970-01-01"), Some(0.0));
        // 2000-01-01 → 946684800
        assert_eq!(try_parse_date("2000-01-01"), Some(946684800.0));
        // 2024-01-01 → 1704067200
        assert_eq!(try_parse_date("2024-01-01"), Some(1704067200.0));
    }

    #[test]
    fn invalid_date_returns_none() {
        assert_eq!(try_parse_date("not-a-date"), None);
        assert_eq!(try_parse_date("2024-13-01"), None); // invalid month
        assert_eq!(try_parse_date("2024-00-01"), None); // month 0
        assert_eq!(try_parse_date("2024-01-32"), None); // day 32
    }

    // ---- OR combined with other new features ----

    #[test]
    fn where_or_with_like() {
        let query = SQLQuery::new(
            "SELECT * FROM products WHERE title LIKE 'laptop%' OR title LIKE 'phone%'",
        );
        assert_eq!(
            query.to_redis_query(),
            "(@title:(laptop*) | @title:(phone*))"
        );
    }

    #[test]
    fn where_or_with_date() {
        let query = SQLQuery::new(
            "SELECT * FROM events WHERE created_at < '2024-01-01' OR created_at > '2024-12-31'",
        );
        let result = query.to_redis_query();
        // 2024-12-31 = 1735603200
        assert_eq!(
            result,
            "(@created_at:[-inf (1704067200] | @created_at:[(1735603200 +inf])"
        );
    }
}
