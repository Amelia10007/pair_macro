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
