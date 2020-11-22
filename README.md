
# pair_macro [![crates.io](https://img.shields.io/crates/v/pair_macro.svg)](https://crates.io/crates/pair_macro) [![Build Status](https://travis-ci.org/Amelia10007/pair_macro.svg?branch=master)](https://travis-ci.org/Amelia10007/pair_macro)
Create types consisting of the same type values such that Pair, Triplet, and so on.

This crate runs on `no-std` environment.

# Import
In your `Cargo.toml`:
```
pair_macro = "0.1.2"
```

# Examples
## Use a provided type `Pair`.
```rust
use pair_macro::Pair;

let p = Pair::new(1.0, 2.0); // Pair<f64>
let q = p.map(|v| v * 2.0);

assert_eq!(Pair::new(2.0, 4.0), q);
assert_eq!(2.0, q.x);
assert_eq!(4.0, q.y);
```

## Create a new pair type.
```rust
use pair_macro::create_pair_prelude::*;

create_pair!(MyOwnPair; a, b, c, d);

let p = MyOwnPair::new(1, 2, 3, 4); // MyOwnPair<i32>
let q = MyOwnPair::new(5, 6, 7, 8);
let r = p + q;
assert_eq!(6, r.a);
assert_eq!(8, r.b);
assert_eq!(10, r.c);
assert_eq!(12, r.d);
```

# Features
Pair types support serialize/deserialize by enabling [serde](https://crates.io/crates/serde) feature.  
In your `Cargo.toml`:
```
pair_macro = { version = "0.1.2", features = ["serde"] }
```
