//! Demonstrates building and composing filter expressions.
//!
//! This example does not require a Redis connection – it only
//! shows how to construct filters and render their Redis query syntax.
//!
//! Run with:
//! ```bash
//! cargo run -p redis-vl --example filter_basics
//! ```

use redis_vl::BetweenInclusivity;
use redis_vl::filter::{Geo, GeoRadius, Num, Tag, Text, Timestamp};

fn main() {
    // --- Tag filter ---
    let color = Tag::new("color").eq("red");
    println!("Tag filter: {color}");

    // Tag with multiple values (OR within the tag)
    let colors = Tag::new("color").one_of(["red", "blue", "green"]);
    println!("Tag multi:  {colors}");

    // Tag not-equal
    let not_red = Tag::new("color").ne("red");
    println!("Tag ne:     {not_red}");

    // --- Numeric filter ---
    let cheap = Num::new("price").lt(50.0);
    println!("\nNum lt:      {cheap}");

    let range = Num::new("price").between(10.0, 100.0, BetweenInclusivity::Both);
    println!("Num between: {range}");

    let exact = Num::new("quantity").eq(42.0);
    println!("Num eq:      {exact}");

    // --- Text filter ---
    let text = Text::new("description").eq("premium");
    println!("\nText eq:   {text}");

    let like = Text::new("description").like("pre*");
    println!("Text like: {like}");

    // --- Geo filter ---
    let geo = Geo::new("location").eq(GeoRadius::new(-122.4194, 37.7749, 10.0, "km"));
    println!("\nGeo:       {geo}");

    // --- Timestamp filter ---
    let recent = Timestamp::new("created_at").gte(1_700_000_000.0);
    println!("\nTimestamp:  {recent}");

    // --- Boolean composition ---
    // AND: &
    let and_filter = Tag::new("color").eq("red") & Num::new("price").lt(100.0);
    println!("\nAND: {and_filter}");

    // OR: |
    let or_filter = Text::new("title").eq("sale") | Text::new("title").eq("discount");
    println!("OR:  {or_filter}");

    // NOT: !
    let not_filter = !Tag::new("status").eq("archived");
    println!("NOT: {not_filter}");

    // Complex composition
    let complex = (Tag::new("category").eq("electronics") & Num::new("price").lt(500.0))
        | (Tag::new("category").eq("books") & Num::new("price").lt(30.0));
    println!("\nComplex: {complex}");
}
