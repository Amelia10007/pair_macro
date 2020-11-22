use pair_macro::create_pair_prelude::*;

create_pair!(MyOwnPair; a, b, c, d);

#[test]
fn test_use_my_own_pair() {
    let p = MyOwnPair::new(1, 2, 3, 4);
    let q = MyOwnPair::new(5, 6, 7, 8);
    let r = p + q;
    assert_eq!(6, r.a);
    assert_eq!(8, r.b);
    assert_eq!(10, r.c);
    assert_eq!(12, r.d);
}

#[cfg(test)]
#[cfg(feature = "serde")]
mod tests_feature_serde {
    use super::*;

    #[test]
    fn test_serialize() {
        let p = MyOwnPair::new(1, 2, 3, 4);
        let s = serde_json::to_string(&p).unwrap();
        let expected = r#"{"a":1,"b":2,"c":3,"d":4}"#;
        assert_eq!(expected, &s);
    }

    #[test]
    fn test_deserialize() {
        let s = r#"{"a":1,"b":2,"c":3,"d":4}"#;
        let p = serde_json::from_str(s).unwrap();
        assert_eq!(MyOwnPair::new(1, 2, 3, 4), p);
    }
}
