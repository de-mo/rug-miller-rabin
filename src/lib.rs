use rug::rand::{ThreadRandGen, ThreadRandState};
use rug::Integer;
use std::iter::repeat_with;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

struct Seed(*const ());
impl ThreadRandGen for Seed {
    fn gen(&mut self) -> u32 {
        // not really random
        rand::random()
    }
}

fn decompose(n: &Integer) -> (Integer, Integer) {
    let one = Integer::ONE;
    let two = &Integer::from(2);
    let mut d = Integer::from(n - 1u8);
    let mut r = Integer::ZERO;

    while Integer::from(&d % two) == *one {
        d /= two;
        r += one;
    }

    (d, r)
}

fn miller_rabin(a: &Integer, n: &Integer, d: &Integer, r: &Integer) -> bool {
    let n_minus_one = Integer::from(n - 1u8);
    let mut x = Integer::from(a.pow_mod_ref(d, n).unwrap());
    let mut count = Integer::ONE.clone();
    let two = &Integer::from(2);

    if &x == Integer::ONE || x == n_minus_one {
        return false;
    }

    while count < *r {
        let _ = x.pow_mod_mut(two, n);

        if x == n_minus_one {
            return false;
        }

        count += 1u8;
    }

    true
}

/// Test whether an integer `n` is likely prime using the Miller-Rabin primality test.
///
/// # Examples
///
/// ```
/// use miller_rabin::is_prime;
///
/// // Mersenne Prime (2^31 - 1)
/// let n: u64 = 0x7FFF_FFFF;
/// // Try the miller-rabin test 100 times in parallel
/// // (or, if `n` is less than `u64::max()`, test only the numbers known to be sufficient).
/// // In general, the algorithm should fail at a rate of at most `4^{-k}`.
/// assert!(is_prime(&n, 16));
/// ```
pub fn is_prime(n: &Integer, k: usize) -> bool {
    let n_minus_one = Integer::from(n - 1u8);
    let (ref d, ref r) = decompose(n);

    if n <= Integer::ONE {
        return false;
    } else if n <= &Integer::from(3) {
        return true;
    } else if n <= &Integer::from(0xffff_ffff_ffff_ffffu64) {
        let samples: Vec<u8> = vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

        #[cfg(feature = "rayon")]
        return samples
            .par_iter()
            .filter(|&&m| m < n_minus_one)
            .find_any(|&&a| miller_rabin(&Integer::from(a), n, d, r))
            .is_none();

        #[cfg(not(feature = "rayon"))]
        return samples
            .iter()
            .filter(|&&m| m < n_minus_one)
            .find(|&&a| miller_rabin(&Integer::from(a), n, d, r))
            .is_none();
    }

    let samples: Vec<Integer> = repeat_with(|| {
        let mut seed = Seed(&());
        let mut rand = ThreadRandState::new_custom(&mut seed);
        n_minus_one.clone().random_below(&mut rand)
    })
    .filter(|m| m < &n_minus_one)
    .take(k)
    .collect();

    #[cfg(feature = "rayon")]
    return samples
        .par_iter()
        .find_any(|&a| miller_rabin(a, n, d, r))
        .is_none();

    #[cfg(not(feature = "rayon"))]
    samples.iter().find(|&a| miller_rabin(a, n, d, r)).is_none()
}

#[cfg(test)]
mod tests {
    const K: usize = 16;

    use super::*;
    use std::io;
    use num_bigint::BigUint;
    use num_traits::Num;
    use std::time::SystemTime;

    #[test]
    fn test_prime() -> io::Result<()> {
        let prime: u64 = 0x7fff_ffff;
        assert!(is_prime(&Integer::from(prime), K));
        Ok(())
    }

    #[test]
    fn test_prime_biguint() -> io::Result<()> {
        let prime = Integer::from(0x7fff_ffff);
        assert!(is_prime(&prime, K));
        Ok(())
    }

    #[test]
    fn test_composite() -> io::Result<()> {
        let composite = Integer::from(0xffff_ffff_ffff_ffffu64);
        assert!(!is_prime(&composite, K));
        Ok(())
    }

    #[test]
    fn test_small_primes() -> io::Result<()> {
        for prime in &[2u8, 3u8, 5u8, 7u8, 11u8, 13u8] {
            assert!(is_prime(&Integer::from(*prime), K));
        }

        Ok(())
    }

    #[test]
    fn test_big_mersenne_prime() -> io::Result<()> {
        let prime = Integer::from(
            Integer::parse_radix(b"170141183460469231731687303715884105727", 10).unwrap(),
        );

        assert!(is_prime(&prime, K));
        Ok(())
    }

    #[test]
    fn test_big_wagstaff_prime() -> io::Result<()> {
        let prime = Integer::from(
            Integer::parse_radix(b"56713727820156410577229101238628035243", 10).unwrap(),
        );

        assert!(is_prime(&prime, K));
        Ok(())
    }

    #[test]
    fn test_big_composite() -> io::Result<()> {
        let prime = Integer::from(
            Integer::parse_radix(b"170141183460469231731687303715884105725", 10).unwrap(),
        );

        assert!(!is_prime(&prime, K));
        Ok(())
    }

    #[test]
    fn test_p() {
        let p = Integer::from_str_radix("BEDCDE3405B8A18D6C7615FCFF97DB1C29CD2CA69F1BB1432E690E1E947836FC1DE9160D5C2ADEE52ED244F7997ECCE19FF979D00CC3CCE3784DA6C6495D0D87337B24ABB0FD848C79EBBCF298349396FAE4031A3B7EC2BF313CAEF36AB191CAD36D4AEFDFFA87F72DAACB2EA854FFFCCC66E99C2896911EBA93341C006DD3AA4DD06B432B2D3FCD79B5F7C61DED181B734B2DC1C869E498B2647E8C4301DBFD1787F1C7F5E687D118F2A5D410DB73689586377AA9273DEEC051B60DB813DD0C22FAD561BABE3C59CC67EB284387EE6D3F8C38F6A0B34DE82CEF929B853C3B1A52C6CD6B87AA0A882C30F8B716B3687CCB8EB9EC1BF67407C5142315D2BDFFA5D37E0ADB968593BC66A999695DF11B0164B21A62F7A0A7006D49EF8DEB31408E66AD53A4A6BE38F20EF09C84C729A9544EDF854274DC2120CAFA1BC08E20E7C7F1969DCD4C2C08DCB8AB419B6A8B22F1D6F183B1912E54B045C84E95E668D282073EF9216E3106C173FF9A1D29DC445059491209FA9540D06B666611EB5ECE77",16).unwrap();
        assert!(is_prime(&p, K));
    }

    #[test]
    #[ignore = "performance"]
    fn test_prformance() {
        let p1 = Integer::from_str_radix("BEDCDE3405B8A18D6C7615FCFF97DB1C29CD2CA69F1BB1432E690E1E947836FC1DE9160D5C2ADEE52ED244F7997ECCE19FF979D00CC3CCE3784DA6C6495D0D87337B24ABB0FD848C79EBBCF298349396FAE4031A3B7EC2BF313CAEF36AB191CAD36D4AEFDFFA87F72DAACB2EA854FFFCCC66E99C2896911EBA93341C006DD3AA4DD06B432B2D3FCD79B5F7C61DED181B734B2DC1C869E498B2647E8C4301DBFD1787F1C7F5E687D118F2A5D410DB73689586377AA9273DEEC051B60DB813DD0C22FAD561BABE3C59CC67EB284387EE6D3F8C38F6A0B34DE82CEF929B853C3B1A52C6CD6B87AA0A882C30F8B716B3687CCB8EB9EC1BF67407C5142315D2BDFFA5D37E0ADB968593BC66A999695DF11B0164B21A62F7A0A7006D49EF8DEB31408E66AD53A4A6BE38F20EF09C84C729A9544EDF854274DC2120CAFA1BC08E20E7C7F1969DCD4C2C08DCB8AB419B6A8B22F1D6F183B1912E54B045C84E95E668D282073EF9216E3106C173FF9A1D29DC445059491209FA9540D06B666611EB5ECE77",16).unwrap();
        let p2 = <BigUint>::from_str_radix("BEDCDE3405B8A18D6C7615FCFF97DB1C29CD2CA69F1BB1432E690E1E947836FC1DE9160D5C2ADEE52ED244F7997ECCE19FF979D00CC3CCE3784DA6C6495D0D87337B24ABB0FD848C79EBBCF298349396FAE4031A3B7EC2BF313CAEF36AB191CAD36D4AEFDFFA87F72DAACB2EA854FFFCCC66E99C2896911EBA93341C006DD3AA4DD06B432B2D3FCD79B5F7C61DED181B734B2DC1C869E498B2647E8C4301DBFD1787F1C7F5E687D118F2A5D410DB73689586377AA9273DEEC051B60DB813DD0C22FAD561BABE3C59CC67EB284387EE6D3F8C38F6A0B34DE82CEF929B853C3B1A52C6CD6B87AA0A882C30F8B716B3687CCB8EB9EC1BF67407C5142315D2BDFFA5D37E0ADB968593BC66A999695DF11B0164B21A62F7A0A7006D49EF8DEB31408E66AD53A4A6BE38F20EF09C84C729A9544EDF854274DC2120CAFA1BC08E20E7C7F1969DCD4C2C08DCB8AB419B6A8B22F1D6F183B1912E54B045C84E95E668D282073EF9216E3106C173FF9A1D29DC445059491209FA9540D06B666611EB5ECE77",16).unwrap();
        let start2 = SystemTime::now();
        assert!(miller_rabin::is_prime(&p2, 64));
        let duration2 = start2.elapsed().unwrap();
        let start1 = SystemTime::now();
        assert!(is_prime(&p1, 64));
        let duration1 = start1.elapsed().unwrap();
        assert!(duration1 < duration2);
        println!("Duration BigUInt: {}s", duration2.as_secs_f32());
        println!("Duration rug: {}s", duration1.as_secs_f32());
    }
}
