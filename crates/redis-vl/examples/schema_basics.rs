//! Demonstrates parsing and inspecting an `IndexSchema` from a YAML string.
//!
//! Run with:
//! ```bash
//! cargo run -p redis-vl --example schema_basics
//! ```

use redis_vl::IndexSchema;

fn main() {
    let yaml = r#"
index:
  name: example-index
  prefix: doc
  storage_type: hash
fields:
  - name: title
    type: tag
  - name: content
    type: text
  - name: score
    type: numeric
  - name: embedding
    type: vector
    attrs:
      algorithm: flat
      dims: 128
      distance_metric: cosine
      datatype: float32
"#;

    let schema = IndexSchema::from_yaml_str(yaml).expect("valid schema");
    println!("Index name  : {}", schema.index.name);
    println!("Prefix      : {}", schema.index.prefix.first());
    println!("Storage type: {:?}", schema.index.storage_type);
    println!("Fields      : {}", schema.fields.len());
    for field in &schema.fields {
        println!("  - {} ({:?})", field.name, field.kind);
    }

    let json_value = schema.to_json_value().expect("serializable");
    println!(
        "\nJSON representation:\n{}",
        serde_json::to_string_pretty(&json_value).unwrap()
    );
}
