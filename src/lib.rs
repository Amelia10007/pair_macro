//! # pair
//! Create types consisting of the same type values such that Pair, Triplet, and so on.
//!
//! This crate runs on `no-std` environment.
//! # Examples
//! ## Use a provided type `Pair`.
//! ```
//! use pair_macro::Pair;
//!
//! let p = Pair::new(1, 2);
//! let q = p.map(|v| v * 2);
//!
//! assert_eq!(Pair::new(2, 4), q);
//! assert_eq!(2, q.x);
//! assert_eq!(4, q.y);
//! ```
//!
//! ## Create a new pair type.
//! ```
//! use pair_macro::create_pair_prelude::*;
//!
//! create_pair!(MyOwnPair; a, b, c, d);
//!
//! let p = MyOwnPair::new(1, 2, 3, 4);
//! let q = MyOwnPair::new(5, 6, 7, 8);
//! let r = p + q;
//! assert_eq!(6, r.a);
//! assert_eq!(8, r.b);
//! assert_eq!(10, r.c);
//! assert_eq!(12, r.d);
//! ```

#![no_std]

/// This is required to create a pair type.
pub mod create_pair_prelude {
    pub use crate::{create_pair, impl_core_ops};

    pub use crate::num_traits_method;
    #[cfg(feature = "num-traits")]
    pub use num_traits::{Float, Zero};
}

#[allow(unused_imports)]
use create_pair_prelude::*;

/// Creates a pair type.
/// When this macro is used out of this crate, [`pair_macro::create_pair_prelude`](create_pair_prelude/index.html) must be imported.
///
/// # Params
/// 1. `name` the name of the new pair type.
/// 1. `field` the fields' name.
///
/// # Examples
/// ```
/// use pair_macro::create_pair_prelude::*;
///
/// create_pair!(MyOwnPair; a, b, c, d);
///
/// let p = MyOwnPair::new(1, 2, 3, 4);
/// let q = MyOwnPair::new(5, 6, 7, 8);
/// let r = p + q;
/// assert_eq!(6, r.a);
/// assert_eq!(8, r.b);
/// assert_eq!(10, r.c);
/// assert_eq!(12, r.d);
/// ```
#[macro_export]
macro_rules! create_pair {
    ( $name: tt; $($field: tt),* ) => {

        /// Represents a pair consisting of the same-type values.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
        pub struct $name<T> {
            $(pub $field: T),*
        }

        impl<T> $name<T>{
            /// Creates a new pair.
            /// # Examples
            /// ```
            /// use pair_macro::Pair;
            ///
            /// let p = Pair::new(1, 2);
            /// assert_eq!(1, p.x);
            /// assert_eq!(2, p.y);
            /// ```
            pub const fn new($($field: T),*) -> $name<T> {
                Self {$($field),*}
            }

            /// Creates a new pair by cloning the specified `value`.
            /// # Examples
            /// ```
            /// use pair_macro::Pair;
            ///
            /// let p = Pair::from_cloned(vec![1, 2]);
            /// assert_eq!(vec![1, 2], p.x);
            /// assert_eq!(vec![1, 2], p.y);
            /// ```
            pub fn from_cloned(value: T) -> $name<T>
                where T: Clone
            {
                Self {$($field: value.clone()),*}
            }

            /// Converts from `&name<T>` to `name<&T>`.
            /// # Examples
            /// ```
            /// use pair_macro::Pair;
            ///
            /// let p = Pair::new(1, 2);
            /// let p = p.as_ref();
            /// assert_eq!(&1, p.x);
            /// assert_eq!(&2, p.y);
            /// ```
            pub const fn as_ref(&self) -> $name<&T> {
                $name {$($field: &self.$field),*}
            }

            /// Converts from `&mut name<T>` to `name<&mut T>`.
            /// # Examples
            /// ```
            /// use pair_macro::Pair;
            ///
            /// let mut p = Pair::new(1, 2);
            /// let q = p.as_mut();
            /// *q.x = 3;
            /// *q.y = 4;
            /// assert_eq!(3, p.x);
            /// assert_eq!(4, p.y);
            /// ```
            pub fn as_mut(&mut self) -> $name<&mut T> {
                $name {$($field: &mut self.$field),*}
            }

            /// Applies an unary operation `f` to each value.
            /// # Examples
            /// ```
            /// use pair_macro::Pair;
            ///
            /// let p = Pair::new(1, 2).map(|value| value * 2);
            /// assert_eq!(2, p.x);
            /// assert_eq!(4, p.y);
            /// ```
            pub fn map<U, F>(self, mut f: F) -> $name<U>
                where F: FnMut(T) -> U
            {
                $name {$($field: f(self.$field)),*}
            }

            /// Applies an unary operation `f` in-place.
            ///
            /// # Returns
            /// A mutable reference to itself.
            ///
            /// # Examples
            /// ```
            /// use pair_macro::Pair;
            ///
            /// let mut p = Pair::new(2, 3);
            /// p.map_in_place(|value| *value += 1)
            ///     .map_in_place(|value| *value *= 2);
            /// assert_eq!(6, p.x);
            /// assert_eq!(8, p.y);
            /// ```
            pub fn map_in_place<F>(&mut self, mut f: F) -> &mut $name<T>
                where F: FnMut(&mut T)
            {
                $(f(&mut self.$field);)*
                self
            }

            /// Applies a binary operation `f` that takes two values, then produces another pair.
            /// # Examples
            /// ```
            /// use pair_macro::Pair;
            ///
            /// let p = Pair::new(6, 8);
            /// let q = Pair::new(3, 2);
            /// let r = p.map_entrywise(q, |lhs, rhs| lhs / rhs);
            /// assert_eq!(2, r.x);
            /// assert_eq!(4, r.y);
            /// ```
            pub fn map_entrywise<U, V, F>(self, rhs: $name<U>, mut f: F) -> $name<V>
                where F: FnMut(T, U) -> V
            {
                $name {$($field: f(self.$field, rhs.$field)),*}
            }

            /// Applies a binary operation `f` that takes two values, then stores the result in-place.
            ///
            /// # Returns
            /// A mutable reference to itself.
            ///
            /// # Examples
            /// ```
            /// use pair_macro::Pair;
            ///
            /// let mut p = Pair::new(2, 3);
            /// let q = Pair::new(4, 5);
            /// let r = Pair::new(6, 7);
            /// p.map_entrywise_in_place(q, |lhs, rhs| *lhs += rhs)
            ///     .map_entrywise_in_place(r, |lhs, rhs| *lhs *= rhs);
            /// assert_eq!(36, p.x);
            /// assert_eq!(56, p.y);
            /// ```
            pub fn map_entrywise_in_place<U, F>(&mut self, rhs: $name<U>, mut f: F) -> &mut $name<T>
                where F: FnMut(&mut T, U)
            {
                $(f(&mut self.$field, rhs.$field);)*
                self
            }

            /// Applies a function `f` to each values, then produces a single, final value.
            /// # Params
            /// 1. `init` the initial value that the accumulator will have on the first call.
            /// 1. `f` accumulator that takes two arguments: current accumation and an value.
            ///
            /// # Examples
            /// ```
            /// use pair_macro::Pair;
            ///
            /// let p = Pair::new(2, 3);
            /// let sum = p.fold(0, |accumulate, current| accumulate + current);
            /// let product = p.fold(1, |accumulate, current| accumulate * current);
            /// assert_eq!(5, sum);
            /// assert_eq!(6, product);
            /// ```
            pub fn fold<U, F>(self, init: U, mut f: F) -> U
                where F: FnMut(U, T) -> U
            {
                let mut value = init;
                $(value = f(value, self.$field);)*
                value
            }
        }

        impl<T> $name<Option<T>> {
            /// Converts from `name<Option<T>>` to `Option<name<T>>`.
            /// # Returns
            /// `Some(pair)` if all values of the pair are `Some(..)`, `None` otherwise.
            /// # Examples
            /// ```
            /// use pair_macro::Pair;
            ///
            /// let p = Pair::new(Some(1), Some(2)).into_option();
            /// assert_eq!(Some(Pair::new(1, 2)), p);
            ///
            /// let q = Pair::new(Some(1), None).into_option();
            /// assert!(q.is_none());
            ///
            /// let r = Pair::<Option<i32>>::new(None, None).into_option();
            /// assert!(r.is_none());
            /// ```
            pub fn into_option(self) -> Option<$name<T>> {
                Some($name {$($field: self.$field?),*})
            }
        }

        impl<T, E> $name<Result<T, E>> {
            /// Converts from `name<Result<T, E>>` to `Result<name<T>, E>`.
            /// # Returns
            /// `Ok(pair)` if all values of the pair are `Ok(..)`.
            /// `Err(e)` if any value of the pair is `Err(e)`, where `e` is the value of the first `Err(..)` value.
            /// # Examples
            /// ```
            /// use pair_macro::Pair;
            /// use core::str::FromStr;
            ///
            /// let p = Pair::new("1", "2").map(i32::from_str).into_result();
            /// assert_eq!(Ok(Pair::new(1, 2)), p);
            ///
            /// let q = Pair::new("1", "lamy").map(i32::from_str).into_result();
            /// assert!(q.is_err());
            ///
            /// let r = Pair::new("yukihana", "lamy").map(i32::from_str).into_result();
            /// assert!(r.is_err());
            /// ```
            pub fn into_result(self) -> Result<$name<T>, E> {
                Ok($name {$($field: self.$field?),*})
            }
        }

        impl<T> core::fmt::Display for $name<T>
            where T: core::fmt::Display
        {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, "(")?;
                $(write!(f, "{}, ", self.$field)?;)*
                write!(f, ")")?;
                Ok(())
            }
        }

        impl_core_ops!($name);

        #[cfg(feature = "num-traits")]
        num_traits_method!($name);
    }
}

/// Implements core unary and binary operations (such as `core::ops::Neg` and `core::ops::Add`) for the specified pair type.
#[macro_export]
#[doc(hidden)]
macro_rules! impl_core_ops {
    ($name: tt) => {
        impl<T> core::ops::Neg for $name<T>
        where
            T: core::ops::Neg,
        {
            type Output = $name<T::Output>;

            fn neg(self) -> Self::Output {
                self.map(|value| -value)
            }
        }

        impl<T, U> core::ops::Add<$name<U>> for $name<T>
        where
            T: core::ops::Add<U>,
        {
            type Output = $name<T::Output>;

            fn add(self, rhs: $name<U>) -> Self::Output {
                self.map_entrywise(rhs, |left, right| left + right)
            }
        }

        impl<T, U> core::ops::Sub<$name<U>> for $name<T>
        where
            T: core::ops::Sub<U>,
        {
            type Output = $name<T::Output>;

            fn sub(self, rhs: $name<U>) -> Self::Output {
                self.map_entrywise(rhs, |left, right| left - right)
            }
        }

        impl<T, U> core::ops::Mul<U> for $name<T>
        where
            T: core::ops::Mul<U>,
            U: Copy,
        {
            type Output = $name<T::Output>;

            fn mul(self, rhs: U) -> Self::Output {
                self.map(|value| value * rhs)
            }
        }

        impl<T, U> core::ops::Div<U> for $name<T>
        where
            T: core::ops::Div<U>,
            U: Copy,
        {
            type Output = $name<T::Output>;

            fn div(self, rhs: U) -> Self::Output {
                self.map(|value| value / rhs)
            }
        }

        impl<T, U> core::ops::Rem<U> for $name<T>
        where
            T: core::ops::Rem<U>,
            U: Copy,
        {
            type Output = $name<T::Output>;

            fn rem(self, rhs: U) -> Self::Output {
                self.map(|value| value % rhs)
            }
        }

        impl<T> core::ops::AddAssign for $name<T>
        where
            T: core::ops::AddAssign,
        {
            fn add_assign(&mut self, rhs: Self) {
                self.map_entrywise_in_place(rhs, |lhs, rhs| *lhs += rhs);
            }
        }

        impl<T> core::ops::SubAssign for $name<T>
        where
            T: core::ops::SubAssign,
        {
            fn sub_assign(&mut self, rhs: Self) {
                self.map_entrywise_in_place(rhs, |lhs, rhs| *lhs -= rhs);
            }
        }

        impl<T, U> core::ops::MulAssign<U> for $name<T>
        where
            T: core::ops::MulAssign<U>,
            U: Copy,
        {
            fn mul_assign(&mut self, rhs: U) {
                self.map_in_place(|value| *value *= rhs);
            }
        }

        impl<T, U> core::ops::DivAssign<U> for $name<T>
        where
            T: core::ops::DivAssign<U>,
            U: Copy,
        {
            fn div_assign(&mut self, rhs: U) {
                self.map_in_place(|value| *value /= rhs);
            }
        }

        impl<T, U> core::ops::RemAssign<U> for $name<T>
        where
            T: core::ops::RemAssign<U>,
            U: Copy,
        {
            fn rem_assign(&mut self, rhs: U) {
                self.map_in_place(|value| *value %= rhs);
            }
        }
    };
}

/// Implements mathematical methods to the specific pair type, if `num-traits` feature is enabled.
#[macro_export]
#[doc(hidden)]
macro_rules! num_traits_method {
    ($name: tt) => {
        impl<T> $name<T>
        where
            T: Copy + core::ops::Add<Output = T> + core::ops::Mul<Output = T> + Zero,
        {
            /// Returns the norm of this pair as a mathematical vector.
            /// # Examples
            /// ```
            /// use pair_macro::Pair;
            ///
            /// let p = Pair::new(3.0, 4.0);
            /// assert_eq!(25.0, p.norm());
            /// ```
            pub fn norm(self) -> T {
                self.map(|value| value * value)
                    .fold(T::zero(), |acc, cur| acc + cur)
            }
        }

        impl<T: Float> $name<T> {
            /// Returns the length of this pair as a mathematical vector.
            /// # Examples
            /// ```
            /// use pair_macro::Pair;
            ///
            /// let p = Pair::new(3.0, 4.0);
            /// assert_eq!(5.0, p.len());
            /// ```
            pub fn len(self) -> T {
                self.norm().sqrt()
            }

            /// Returns the normalization of this pair as a mathematical vector.
            /// # Examples
            /// ```
            /// use pair_macro::Pair;
            ///
            /// let p = Pair::new(3.0, 4.0);
            /// let n = p.normalize();
            /// assert_eq!(n, Pair::new(0.6, 0.8));
            /// assert_eq!(1.0, n.norm());
            /// ```
            pub fn normalize(self) -> $name<T> {
                self / self.len()
            }
        }
    };
}

create_pair!(Pair; x, y);
create_pair!(Triplet; x, y, z);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let p = Pair::new(2, 3);
        assert_eq!(2, p.x);
        assert_eq!(3, p.y);
    }

    #[test]
    fn test_from_cloned() {
        let v = [0, 1, 2];
        let p = Pair::from_cloned(v.clone());
        assert_eq!(v, p.x);
        assert_eq!(v, p.y);
    }

    #[test]
    fn test_as_ref() {
        let v = [0, 1, 2];
        let p = Pair::from_cloned(v.clone());
        let p = p.as_ref();
        assert_eq!(&v, p.x);
        assert_eq!(&v, p.y);
    }

    #[test]
    fn test_as_mut() {
        let mut p = Pair::new(1, 2);
        let q = p.as_mut();
        *q.x = 3;
        *q.y = 4;
        assert_eq!(3, p.x);
        assert_eq!(4, p.y);
    }

    #[test]
    fn test_map() {
        let p = Pair::new(2, 3).map(|value| value * value);
        assert_eq!(4, p.x);
        assert_eq!(9, p.y);
    }

    #[test]
    fn test_map_in_place() {
        let mut p = Pair::new(2, 3);
        p.map_in_place(|value| *value += 1)
            .map_in_place(|value| *value *= 2);
        assert_eq!(6, p.x);
        assert_eq!(8, p.y);
    }

    #[test]
    fn map_entrywise() {
        let p = Pair::new(2, 3);
        let q = Pair::new(4, 5);
        let r = p.map_entrywise(q, |lhs, rhs| lhs + rhs);
        assert_eq!(6, r.x);
        assert_eq!(8, r.y);
    }

    #[test]
    fn test_map_entrywise_in_place() {
        let mut p = Pair::new(2, 3);
        let q = Pair::new(4, 5);
        let r = Pair::new(6, 7);
        p.map_entrywise_in_place(q, |lhs, rhs| *lhs += rhs)
            .map_entrywise_in_place(r, |lhs, rhs| *lhs *= rhs);
        assert_eq!(36, p.x);
        assert_eq!(56, p.y);
    }

    #[test]
    fn test_fold() {
        let f = Pair::new(2, 3).fold(1, |acc, cur| acc * cur);
        assert_eq!(6, f);
    }

    #[cfg(feature = "num-traits")]
    #[test]
    fn test_norm() {
        let p = Pair::new(2.0, 3.0);
        assert_eq!(13.0, p.norm());
    }

    #[cfg(feature = "num-traits")]
    #[test]
    fn test_len() {
        let p = Pair::new(3.0, 4.0);
        assert_eq!(5.0, p.len());
    }

    #[cfg(feature = "num-traits")]
    #[test]
    fn test_normalize() {
        let p = Pair::new(3.0, 4.0);
        let n = p.normalize();

        assert_eq!(n, Pair::new(0.6, 0.8));
        assert_eq!(1.0, n.norm());
    }

    #[test]
    fn test_into_option() {
        let p = Pair::new(Some(2), Some(3)).into_option();
        assert_eq!(Some(Pair::new(2, 3)), p);

        assert!(Pair::new(Some(2), None).into_option().is_none());
        assert!(Pair::new(None, Some(3)).into_option().is_none());
        assert!(Pair::<Option<i32>>::new(None, None).into_option().is_none());
    }

    #[test]
    fn test_into_result() {
        use core::str::FromStr;

        let p = Pair::new("1", "2").map(i32::from_str).into_result();
        assert_eq!(Ok(Pair::new(1, 2)), p);

        let q = Pair::new("1", "lamy").map(i32::from_str).into_result();
        assert!(q.is_err());

        let r = Pair::new("polka", "lamy").map(i32::from_str).into_result();
        assert!(r.is_err());
    }

    #[test]
    fn test_neg() {
        assert_eq!(Pair::new(-2, -3), -Pair::new(2, 3));
    }

    #[test]
    fn test_add() {
        assert_eq!(Pair::new(6, 8), Pair::new(2, 3) + Pair::new(4, 5));
    }

    #[test]
    fn test_sub() {
        assert_eq!(Pair::new(-3, 0), Pair::new(2, 3) - Pair::new(5, 3));
    }

    #[test]
    fn test_mul() {
        assert_eq!(Pair::new(4, 6), Pair::new(2, 3) * 2);
    }

    #[test]
    fn test_div() {
        assert_eq!(Pair::new(2, 3), Pair::new(4, 6) / 2);
    }

    #[test]
    fn test_rem() {
        assert_eq!(Pair::new(1, 2), Pair::new(4, 5) % 3);
    }

    #[test]
    fn test_add_assign() {
        let mut p = Pair::new(2, 3);
        p += Pair::new(4, 5);
        assert_eq!(6, p.x);
        assert_eq!(8, p.y);
    }

    #[test]
    fn test_sub_assign() {
        let mut p = Pair::new(2, 3);
        p -= Pair::new(4, 6);
        assert_eq!(-2, p.x);
        assert_eq!(-3, p.y);
    }

    #[test]
    fn test_mul_assign() {
        let mut p = Pair::new(2, 3);
        p *= 2;
        assert_eq!(4, p.x);
        assert_eq!(6, p.y);
    }

    #[test]
    fn test_div_assign() {
        let mut p = Pair::new(4, 6);
        p /= 2;
        assert_eq!(2, p.x);
        assert_eq!(3, p.y);
    }

    #[test]
    fn test_rem_assign() {
        let mut p = Pair::new(4, 5);
        p %= 3;
        assert_eq!(1, p.x);
        assert_eq!(2, p.y);
    }
}
