//! Filter expression DSL for Redis Search.

use std::{
    fmt::{self, Display, Formatter},
    ops::{BitAnd, BitOr, Not},
};

use chrono::{DateTime, NaiveDate, NaiveDateTime, TimeZone, Utc};

/// Comparison operators used by numeric and timestamp predicates.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ComparisonOp {
    /// Strictly greater than.
    GreaterThan,
    /// Greater than or equal to.
    GreaterThanOrEqual,
    /// Strictly less than.
    LessThan,
    /// Less than or equal to.
    LessThanOrEqual,
}

/// Inclusivity used by `between` predicates.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum BetweenInclusivity {
    /// Include both endpoints.
    Both,
    /// Exclude both endpoints.
    Neither,
    /// Include only the left endpoint.
    Left,
    /// Include only the right endpoint.
    Right,
}

/// Filter expression tree.
#[derive(Debug, Clone)]
pub enum FilterExpression {
    /// Wildcard that matches everything.
    MatchAll,
    /// Raw Redis Search filter expression supplied verbatim.
    Raw(String),
    /// Tag equality or membership expression.
    Tag {
        /// Field name.
        field: String,
        /// Accepted values.
        values: Vec<String>,
    },
    /// Text equality expression.
    TextExact {
        /// Field name.
        field: String,
        /// Search term.
        value: String,
    },
    /// Text wildcard/pattern expression.
    TextLike {
        /// Field name.
        field: String,
        /// Search term.
        value: String,
    },
    /// Numeric equality/range expression.
    NumericRange {
        /// Field name.
        field: String,
        /// Minimum bound.
        min: String,
        /// Maximum bound.
        max: String,
    },
    /// Geo radius expression.
    GeoRadius {
        /// Field name.
        field: String,
        /// Longitude.
        longitude: f64,
        /// Latitude.
        latitude: f64,
        /// Radius.
        radius: f64,
        /// Unit.
        unit: String,
    },
    /// Timestamp equality/range expression.
    TimestampRange {
        /// Field name.
        field: String,
        /// Minimum bound.
        min: String,
        /// Maximum bound.
        max: String,
    },
    /// Logical AND.
    And(Box<FilterExpression>, Box<FilterExpression>),
    /// Logical OR.
    Or(Box<FilterExpression>, Box<FilterExpression>),
    /// Logical NOT.
    Not(Box<FilterExpression>),
    /// IsMissing predicate – matches documents where a field is absent.
    IsMissing {
        /// Field name.
        field: String,
    },
}

impl FilterExpression {
    /// Creates a raw Redis Search filter expression.
    pub fn raw(expression: impl Into<String>) -> Self {
        let expression = expression.into();
        if expression.trim().is_empty() || expression.trim() == "*" {
            Self::MatchAll
        } else {
            Self::Raw(expression)
        }
    }

    /// Serializes the filter into Redis Search query syntax.
    pub fn to_redis_syntax(&self) -> String {
        match self {
            Self::MatchAll => "*".to_owned(),
            Self::Raw(expression) => expression.clone(),
            Self::Tag { field, values } => {
                format!("@{}:{{{}}}", field, values.join("|"))
            }
            Self::TextExact { field, value } => format!("@{}:(\"{}\")", field, value),
            Self::TextLike { field, value } => format!("@{}:({})", field, value),
            Self::NumericRange { field, min, max } => format!("@{}:[{} {}]", field, min, max),
            Self::GeoRadius {
                field,
                longitude,
                latitude,
                radius,
                unit,
            } => format!(
                "@{}:[{} {} {} {}]",
                field, longitude, latitude, radius, unit
            ),
            Self::TimestampRange { field, min, max } => format!("@{}:[{} {}]", field, min, max),
            Self::And(left, right) => {
                format!("({} {})", left.to_redis_syntax(), right.to_redis_syntax())
            }
            Self::Or(left, right) => {
                format!("({} | {})", left.to_redis_syntax(), right.to_redis_syntax())
            }
            Self::Not(inner) => format!("(-{})", inner.to_redis_syntax()),
            Self::IsMissing { field } => format!("ismissing(@{})", field),
        }
    }

    fn is_match_all(&self) -> bool {
        matches!(self, Self::MatchAll)
    }
}

impl Display for FilterExpression {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_redis_syntax())
    }
}

impl From<&str> for FilterExpression {
    fn from(value: &str) -> Self {
        Self::raw(value)
    }
}

impl From<String> for FilterExpression {
    fn from(value: String) -> Self {
        Self::raw(value)
    }
}

impl BitAnd for FilterExpression {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        if self.is_match_all() {
            rhs
        } else if rhs.is_match_all() {
            self
        } else {
            Self::And(Box::new(self), Box::new(rhs))
        }
    }
}

impl BitOr for FilterExpression {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        if self.is_match_all() {
            rhs
        } else if rhs.is_match_all() {
            self
        } else {
            Self::Or(Box::new(self), Box::new(rhs))
        }
    }
}

impl Not for FilterExpression {
    type Output = Self;

    fn not(self) -> Self::Output {
        if self.is_match_all() {
            Self::MatchAll
        } else {
            Self::Not(Box::new(self))
        }
    }
}

/// Builder for tag predicates.
#[derive(Debug, Clone)]
pub struct Tag {
    field: String,
}

impl Tag {
    /// Creates a tag predicate builder.
    pub fn new(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
        }
    }

    /// Matches a single tag value.
    pub fn eq(self, value: impl Into<String>) -> FilterExpression {
        let value = value.into();
        if value.is_empty() {
            return FilterExpression::MatchAll;
        }
        FilterExpression::Tag {
            field: self.field,
            values: vec![escape_tag_value(&value, false)],
        }
    }

    /// Matches any of the provided tag values.
    pub fn one_of<I, S>(self, values: I) -> FilterExpression
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let values = values
            .into_iter()
            .map(Into::into)
            .filter(|value| !value.is_empty())
            .map(|value| escape_tag_value(&value, false))
            .collect::<Vec<_>>();

        if values.is_empty() {
            FilterExpression::MatchAll
        } else {
            FilterExpression::Tag {
                field: self.field,
                values,
            }
        }
    }

    /// Matches a wildcard tag expression without escaping `*`.
    pub fn like(self, value: impl Into<String>) -> FilterExpression {
        let value = value.into();
        if value.is_empty() {
            return FilterExpression::MatchAll;
        }
        FilterExpression::Tag {
            field: self.field,
            values: vec![escape_tag_value(&value, true)],
        }
    }

    /// Negates a tag equality predicate.
    pub fn ne(self, value: impl Into<String>) -> FilterExpression {
        !self.eq(value)
    }

    /// Matches documents where this tag field is absent.
    pub fn is_missing(self) -> FilterExpression {
        FilterExpression::IsMissing { field: self.field }
    }
}

/// Builder for text predicates.
#[derive(Debug, Clone)]
pub struct Text {
    field: String,
}

impl Text {
    /// Creates a text predicate builder.
    pub fn new(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
        }
    }

    /// Matches an exact text term.
    pub fn eq(self, value: impl Into<String>) -> FilterExpression {
        let value = value.into();
        if value.is_empty() {
            return FilterExpression::MatchAll;
        }
        FilterExpression::TextExact {
            field: self.field,
            value: escape_exact_text(&value),
        }
    }

    /// Negates an exact text term.
    pub fn ne(self, value: impl Into<String>) -> FilterExpression {
        let value = value.into();
        if value.is_empty() {
            return FilterExpression::MatchAll;
        }
        FilterExpression::Not(Box::new(FilterExpression::TextLike {
            field: self.field,
            value: format!("\"{}\"", escape_exact_text(&value)),
        }))
    }

    /// Matches a wildcard/pattern text term.
    pub fn like(self, value: impl Into<String>) -> FilterExpression {
        let value = value.into();
        if value.is_empty() {
            return FilterExpression::MatchAll;
        }
        FilterExpression::TextLike {
            field: self.field,
            value,
        }
    }

    /// Alias for `like`.
    pub fn matches(self, value: impl Into<String>) -> FilterExpression {
        self.like(value)
    }

    /// Matches documents where this text field is absent.
    pub fn is_missing(self) -> FilterExpression {
        FilterExpression::IsMissing { field: self.field }
    }
}

/// Builder for numeric predicates.
#[derive(Debug, Clone)]
pub struct Num {
    field: String,
}

impl Num {
    /// Creates a numeric predicate builder.
    pub fn new(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
        }
    }

    /// Matches values equal to the supplied value.
    pub fn eq(self, value: f64) -> FilterExpression {
        range_expr(self.field, format_number(value), format_number(value))
    }

    /// Negates values equal to the supplied value.
    pub fn ne(self, value: f64) -> FilterExpression {
        !self.eq(value)
    }

    /// Matches values greater than the supplied value.
    pub fn gt(self, value: f64) -> FilterExpression {
        range_expr(
            self.field,
            format!("({}", format_number(value)),
            "+inf".to_owned(),
        )
    }

    /// Matches values greater than or equal to the supplied value.
    pub fn gte(self, value: f64) -> FilterExpression {
        range_expr(self.field, format_number(value), "+inf".to_owned())
    }

    /// Matches values less than the supplied value.
    pub fn lt(self, value: f64) -> FilterExpression {
        range_expr(
            self.field,
            "-inf".to_owned(),
            format!("({}", format_number(value)),
        )
    }

    /// Matches values less than or equal to the supplied value.
    pub fn lte(self, value: f64) -> FilterExpression {
        range_expr(self.field, "-inf".to_owned(), format_number(value))
    }

    /// Matches values between the supplied bounds.
    pub fn between(self, left: f64, right: f64, inclusive: BetweenInclusivity) -> FilterExpression {
        let min = match inclusive {
            BetweenInclusivity::Both | BetweenInclusivity::Left => format_number(left),
            BetweenInclusivity::Neither | BetweenInclusivity::Right => {
                format!("({}", format_number(left))
            }
        };
        let max = match inclusive {
            BetweenInclusivity::Both | BetweenInclusivity::Right => format_number(right),
            BetweenInclusivity::Neither | BetweenInclusivity::Left => {
                format!("({}", format_number(right))
            }
        };
        range_expr(self.field, min, max)
    }

    /// Matches documents where this numeric field is absent.
    pub fn is_missing(self) -> FilterExpression {
        FilterExpression::IsMissing { field: self.field }
    }
}

/// Builder for geo predicates.
#[derive(Debug, Clone)]
pub struct Geo {
    field: String,
}

impl Geo {
    /// Creates a geo predicate builder.
    pub fn new(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
        }
    }

    /// Matches points within the supplied radius.
    pub fn eq(self, radius: GeoRadius) -> FilterExpression {
        FilterExpression::GeoRadius {
            field: self.field,
            longitude: radius.longitude,
            latitude: radius.latitude,
            radius: radius.radius,
            unit: radius.unit,
        }
    }

    /// Negates the supplied radius.
    pub fn ne(self, radius: GeoRadius) -> FilterExpression {
        !self.eq(radius)
    }

    /// Alias for `eq`.
    pub fn within_radius(self, radius: GeoRadius) -> FilterExpression {
        self.eq(radius)
    }

    /// Matches documents where this geo field is absent.
    pub fn is_missing(self) -> FilterExpression {
        FilterExpression::IsMissing { field: self.field }
    }
}

/// Geo radius selector.
#[derive(Debug, Clone)]
pub struct GeoRadius {
    longitude: f64,
    latitude: f64,
    radius: f64,
    unit: String,
}

impl GeoRadius {
    /// Creates a new geo radius selector.
    pub fn new(longitude: f64, latitude: f64, radius: f64, unit: impl Into<String>) -> Self {
        Self {
            longitude,
            latitude,
            radius,
            unit: unit.into(),
        }
    }
}

/// Builder for timestamp predicates.
#[derive(Debug, Clone)]
pub struct Timestamp {
    field: String,
}

impl Timestamp {
    /// Creates a timestamp predicate builder.
    pub fn new(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
        }
    }

    /// Matches timestamps equal to the supplied value or range-like date.
    pub fn eq<T>(self, value: T) -> FilterExpression
    where
        T: IntoTimestampRange,
    {
        let (min, max) = value.into_timestamp_range();
        FilterExpression::TimestampRange {
            field: self.field,
            min: format_timestamp(min),
            max: format_timestamp(max),
        }
    }

    /// Negates an equality predicate.
    pub fn ne<T>(self, value: T) -> FilterExpression
    where
        T: IntoTimestampRange,
    {
        !self.eq(value)
    }

    /// Matches timestamps before the supplied value.
    pub fn before<T>(self, value: T) -> FilterExpression
    where
        T: IntoTimestampPoint,
    {
        let value = value.into_timestamp_point();
        FilterExpression::TimestampRange {
            field: self.field,
            min: "-inf".to_owned(),
            max: format!("({}", format_timestamp(value)),
        }
    }

    /// Matches timestamps after the supplied value.
    pub fn after<T>(self, value: T) -> FilterExpression
    where
        T: IntoTimestampPoint,
    {
        let value = value.into_timestamp_point();
        FilterExpression::TimestampRange {
            field: self.field,
            min: format!("({}", format_timestamp(value)),
            max: "+inf".to_owned(),
        }
    }

    /// Matches timestamps on or after the supplied value.
    pub fn gte<T>(self, value: T) -> FilterExpression
    where
        T: IntoTimestampPoint,
    {
        let value = value.into_timestamp_point();
        FilterExpression::TimestampRange {
            field: self.field,
            min: format_timestamp(value),
            max: "+inf".to_owned(),
        }
    }

    /// Matches timestamps on or before the supplied value.
    pub fn lte<T>(self, value: T) -> FilterExpression
    where
        T: IntoTimestampPoint,
    {
        let value = value.into_timestamp_point();
        FilterExpression::TimestampRange {
            field: self.field,
            min: "-inf".to_owned(),
            max: format_timestamp(value),
        }
    }

    /// Matches timestamps between two values.
    pub fn between<L, R>(self, left: L, right: R, inclusive: BetweenInclusivity) -> FilterExpression
    where
        L: IntoTimestampPoint,
        R: IntoTimestampPoint,
    {
        let left = left.into_timestamp_point();
        let right = right.into_timestamp_point();
        let min = match inclusive {
            BetweenInclusivity::Both | BetweenInclusivity::Left => format_timestamp(left),
            BetweenInclusivity::Neither | BetweenInclusivity::Right => {
                format!("({}", format_timestamp(left))
            }
        };
        let max = match inclusive {
            BetweenInclusivity::Both | BetweenInclusivity::Right => format_timestamp(right),
            BetweenInclusivity::Neither | BetweenInclusivity::Left => {
                format!("({}", format_timestamp(right))
            }
        };
        FilterExpression::TimestampRange {
            field: self.field,
            min,
            max,
        }
    }

    /// Matches documents where this timestamp field is absent.
    pub fn is_missing(self) -> FilterExpression {
        FilterExpression::IsMissing { field: self.field }
    }
}

/// Converts an input into a timestamp range.
pub trait IntoTimestampRange {
    /// Returns inclusive min/max timestamps.
    fn into_timestamp_range(self) -> (f64, f64);
}

/// Converts an input into a single timestamp point.
pub trait IntoTimestampPoint {
    /// Returns a timestamp point in seconds since the epoch.
    fn into_timestamp_point(self) -> f64;
}

impl IntoTimestampPoint for i64 {
    fn into_timestamp_point(self) -> f64 {
        self as f64
    }
}

impl IntoTimestampPoint for f64 {
    fn into_timestamp_point(self) -> f64 {
        self
    }
}

impl IntoTimestampPoint for DateTime<Utc> {
    fn into_timestamp_point(self) -> f64 {
        self.timestamp() as f64
    }
}

impl IntoTimestampPoint for NaiveDateTime {
    fn into_timestamp_point(self) -> f64 {
        Utc.from_utc_datetime(&self).timestamp() as f64
    }
}

impl IntoTimestampPoint for &str {
    fn into_timestamp_point(self) -> f64 {
        parse_timestamp_string(self).0
    }
}

impl IntoTimestampRange for i64 {
    fn into_timestamp_range(self) -> (f64, f64) {
        let ts = self as f64;
        (ts, ts)
    }
}

impl IntoTimestampRange for f64 {
    fn into_timestamp_range(self) -> (f64, f64) {
        (self, self)
    }
}

impl IntoTimestampRange for DateTime<Utc> {
    fn into_timestamp_range(self) -> (f64, f64) {
        let ts = self.timestamp() as f64;
        (ts, ts)
    }
}

impl IntoTimestampRange for NaiveDateTime {
    fn into_timestamp_range(self) -> (f64, f64) {
        let ts = Utc.from_utc_datetime(&self).timestamp() as f64;
        (ts, ts)
    }
}

impl IntoTimestampRange for NaiveDate {
    fn into_timestamp_range(self) -> (f64, f64) {
        let start = self
            .and_hms_opt(0, 0, 0)
            .expect("valid start of day")
            .and_utc()
            .timestamp() as f64;
        let end = self
            .and_hms_micro_opt(23, 59, 59, 999_999)
            .expect("valid end of day")
            .and_utc()
            .timestamp() as f64
            + 0.999_999;
        (start, end)
    }
}

impl IntoTimestampRange for &str {
    fn into_timestamp_range(self) -> (f64, f64) {
        parse_timestamp_string(self)
    }
}

fn range_expr(field: String, min: String, max: String) -> FilterExpression {
    FilterExpression::NumericRange { field, min, max }
}

fn format_number(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.0}")
    } else {
        value.to_string()
    }
}

fn format_timestamp(value: f64) -> String {
    value.to_string()
}

fn escape_tag_value(value: &str, allow_wildcard: bool) -> String {
    value
        .chars()
        .flat_map(|ch| {
            let should_escape = matches!(ch, ' ' | '$' | ':' | '&' | '/' | '-' | '.')
                || (ch == '*' && !allow_wildcard);
            if should_escape {
                ['\\', ch].into_iter().collect::<Vec<_>>()
            } else {
                vec![ch]
            }
        })
        .collect()
}

fn escape_exact_text(value: &str) -> String {
    value.replace('"', "\\\"")
}

fn parse_timestamp_string(value: &str) -> (f64, f64) {
    if let Ok(date) = NaiveDate::parse_from_str(value, "%Y-%m-%d") {
        return date.into_timestamp_range();
    }

    let datetime = DateTime::parse_from_rfc3339(value)
        .map(|value| value.with_timezone(&Utc))
        .or_else(|_| {
            NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S")
                .map(|value| Utc.from_utc_datetime(&value))
        })
        .expect("valid ISO timestamp");
    let ts = datetime.timestamp() as f64;
    (ts, ts)
}

#[cfg(test)]
mod tests {
    use chrono::{NaiveDate, TimeZone, Utc};

    use super::{BetweenInclusivity, FilterExpression, Geo, GeoRadius, Num, Tag, Text, Timestamp};

    #[test]
    fn composed_filter_should_render() {
        let filter = Tag::new("user").eq("john")
            & Num::new("price").gte(10.0)
            & !Timestamp::new("ts").before(9);

        assert!(filter.to_string().contains("@user:{john}"));
    }

    #[test]
    fn geo_filter_should_render() {
        let filter = Geo::new("location").within_radius(GeoRadius::new(1.0, 2.0, 10.0, "km"));

        assert!(matches!(filter, FilterExpression::GeoRadius { .. }));
        assert_eq!(filter.to_string(), "@location:[1 2 10 km]");
    }

    #[test]
    fn tag_should_escape_specials_like_python_unit_test_filter() {
        assert_eq!(
            Tag::new("tag_field").eq("tag with space").to_string(),
            "@tag_field:{tag\\ with\\ space}"
        );
        assert_eq!(
            Tag::new("tag_field").eq("special$char").to_string(),
            "@tag_field:{special\\$char}"
        );
        assert_eq!(
            Tag::new("tag_field").like("tech*").to_string(),
            "@tag_field:{tech*}"
        );
        assert_eq!(
            Tag::new("tag_field").eq("tech*").to_string(),
            "@tag_field:{tech\\*}"
        );
    }

    #[test]
    fn match_all_should_be_neutral_in_combinations_like_python_unit_test_filter() {
        let all = FilterExpression::MatchAll;
        let tag = Tag::new("tag_field").eq("tag");
        assert_eq!((all.clone() & tag.clone()).to_string(), tag.to_string());
        assert_eq!((all | tag.clone()).to_string(), tag.to_string());
    }

    #[test]
    fn raw_filter_should_round_trip_like_python_manual_string_filters() {
        assert_eq!(
            FilterExpression::raw("@credit_score:{high}").to_string(),
            "@credit_score:{high}"
        );
        assert_eq!(FilterExpression::raw("*").to_string(), "*");
    }

    #[test]
    fn numeric_between_should_render_like_python_unit_test_filter() {
        assert_eq!(
            Num::new("numeric_field")
                .between(2.0, 5.0, BetweenInclusivity::Right)
                .to_string(),
            "@numeric_field:[(2 5]"
        );
    }

    #[test]
    fn text_filters_should_render_like_python_unit_test_filter() {
        assert_eq!(
            Text::new("text_field").eq("text").to_string(),
            "@text_field:(\"text\")"
        );
        assert_eq!(
            Text::new("text_field").ne("text").to_string(),
            "(-@text_field:(\"text\"))"
        );
        assert_eq!(
            Text::new("text_field").like("tex*").to_string(),
            "@text_field:(tex*)"
        );
    }

    #[test]
    fn timestamp_date_should_expand_to_day_like_python_unit_test_filter() {
        let date = NaiveDate::from_ymd_opt(2023, 3, 17).expect("valid date");
        let rendered = Timestamp::new("created_at").eq(date).to_string();
        let start = date
            .and_hms_opt(0, 0, 0)
            .expect("start")
            .and_utc()
            .timestamp() as f64;
        assert!(rendered.starts_with(&format!("@created_at:[{start} ")));
    }

    #[test]
    fn timestamp_between_should_render_like_python_unit_test_filter() {
        let start = Utc
            .with_ymd_and_hms(2023, 3, 17, 14, 30, 0)
            .single()
            .expect("start");
        let end = Utc
            .with_ymd_and_hms(2023, 3, 22, 14, 30, 0)
            .single()
            .expect("end");
        assert_eq!(
            Timestamp::new("created_at")
                .between(start, end, BetweenInclusivity::Left)
                .to_string(),
            format!(
                "@created_at:[{} ({}]",
                start.timestamp() as f64,
                end.timestamp() as f64
            )
        );
    }

    // ── IsMissing parity tests (upstream: test_filter.py) ──

    #[test]
    fn tag_is_missing_like_python_test_filter() {
        let expr = Tag::new("brand").is_missing();
        assert_eq!(expr.to_redis_syntax(), "ismissing(@brand)");
    }

    #[test]
    fn text_is_missing_like_python_test_filter() {
        let expr = Text::new("description").is_missing();
        assert_eq!(expr.to_redis_syntax(), "ismissing(@description)");
    }

    #[test]
    fn num_is_missing_like_python_test_filter() {
        let expr = Num::new("price").is_missing();
        assert_eq!(expr.to_redis_syntax(), "ismissing(@price)");
    }

    #[test]
    fn geo_is_missing_like_python_test_filter() {
        let expr = Geo::new("location").is_missing();
        assert_eq!(expr.to_redis_syntax(), "ismissing(@location)");
    }

    #[test]
    fn timestamp_is_missing_like_python_test_filter() {
        let expr = Timestamp::new("created_at").is_missing();
        assert_eq!(expr.to_redis_syntax(), "ismissing(@created_at)");
    }

    #[test]
    fn is_missing_combined_with_other_filters_like_python() {
        let missing = Tag::new("brand").is_missing();
        let price_filter = Num::new("price").gte(10.0);
        let combined = missing & price_filter;
        assert_eq!(
            combined.to_redis_syntax(),
            "(ismissing(@brand) @price:[10 +inf])"
        );
    }
}
