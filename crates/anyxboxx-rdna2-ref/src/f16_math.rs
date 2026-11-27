//! The half-precision transcendentals of the model (`half_rcp`, `half_sqrt`, `half_rsq`, `half_log2`,
//! `half_exp2`, `half_trig` of `emu/rdna2_exec_f16.py`): `v_rcp/sqrt/rsq/log/exp/sin/cos_f16`.
//!
//! Each follows the model's special cases first, then computes the double-precision value and rounds it once
//! to binary16. `rcp`, `sqrt` and `rsq` are IEEE operations (correctly rounded division and square root), so
//! they give the model's result on any host. `log`, `exp`, `sin` and `cos` call the platform's maths library,
//! as Python's `math.log2`, `2.0 ** x`, `math.sin` and `math.cos` do; the parity tests compare every one of the
//! 65,536 inputs with the model on the host they run on.

use std::f64::consts::TAU;
use std::hint::black_box;

use crate::half::{HALF_INFINITY, HALF_NAN, HALF_ONE, HALF_SIGN, half_from_f64, half_to_f64};

const MAGNITUDE: u32 = 0x7FFF;
/// `v_exp_f16`: from 2^16 the half result is infinity, at or below 2^-25 it is +0.
const EXP_OVERFLOW: f64 = 16.0;
const EXP_UNDERFLOW: f64 = -25.0;
const QUARTER: f64 = 0.25;
const EIGHTH: f64 = 0.125;

fn split(bits: u32) -> (u32, u32) {
    (bits & HALF_SIGN, bits & MAGNITUDE)
}

/// `half_rcp`: 1.0 / S0 as IEEE does it (±0 gives ±infinity, ±infinity gives ±0).
pub fn rcp(bits: u32) -> u32 {
    let (sign, magnitude) = split(bits);
    if magnitude > HALF_INFINITY {
        return HALF_NAN;
    }
    if magnitude == 0 {
        return sign | HALF_INFINITY;
    }
    if magnitude == HALF_INFINITY {
        return sign;
    }
    half_from_f64(1.0 / half_to_f64(bits))
}

/// `half_sqrt`: -0 gives -0, a negative non-zero value NaN.
pub fn sqrt(bits: u32) -> u32 {
    let (sign, magnitude) = split(bits);
    if magnitude > HALF_INFINITY {
        return HALF_NAN;
    }
    if magnitude == 0 {
        return bits;
    }
    if sign != 0 {
        return HALF_NAN;
    }
    if magnitude == HALF_INFINITY {
        return HALF_INFINITY;
    }
    half_from_f64(half_to_f64(bits).sqrt())
}

/// `half_rsq`: 1.0 / sqrt(S0) (±0 gives ±infinity, +infinity gives +0, negative NaN).
pub fn rsq(bits: u32) -> u32 {
    let (sign, magnitude) = split(bits);
    if magnitude > HALF_INFINITY {
        return HALF_NAN;
    }
    if magnitude == 0 {
        return sign | HALF_INFINITY;
    }
    if sign != 0 {
        return HALF_NAN;
    }
    if magnitude == HALF_INFINITY {
        return 0;
    }
    half_from_f64(1.0 / half_to_f64(bits).sqrt())
}

/// `half_log2`: ±0 gives -infinity, +1 gives +0, a negative value NaN.
pub fn log2(bits: u32) -> u32 {
    let (sign, magnitude) = split(bits);
    if magnitude > HALF_INFINITY {
        return HALF_NAN;
    }
    if magnitude == 0 {
        return HALF_SIGN | HALF_INFINITY;
    }
    if sign != 0 {
        return HALF_NAN;
    }
    if magnitude == HALF_INFINITY {
        return HALF_INFINITY;
    }
    if bits == HALF_ONE {
        return 0;
    }
    half_from_f64(half_to_f64(bits).log2())
}

/// `half_exp2`: 2 ** S0 (-infinity gives +0, ±0 gives +1, +infinity gives +infinity).
pub fn exp2(bits: u32) -> u32 {
    let magnitude = bits & MAGNITUDE;
    if magnitude > HALF_INFINITY {
        return HALF_NAN;
    }
    if magnitude == 0 {
        return HALF_ONE;
    }
    if bits == HALF_SIGN | HALF_INFINITY {
        return 0;
    }
    if bits == HALF_INFINITY {
        return HALF_INFINITY;
    }
    let value = half_to_f64(bits);
    if value >= EXP_OVERFLOW {
        return HALF_INFINITY;
    }
    if value <= EXP_UNDERFLOW {
        return 0;
    }
    // The model computes `2.0 ** value`, that is C `pow(2.0, value)`. `black_box` keeps the compiler from
    // rewriting a constant-base pow into `exp2`, which the maths library may round differently.
    half_from_f64(black_box(2.0f64).powf(value))
}

/// `half_trig`: sin or cos of S0 × 2π, with the reduction done exactly in turns before the multiplication by
/// 2π (sin(-0) = -0, sin(±65504) = +0, cos(±65504) = +1, cos(0.25) = +0). ±infinity and NaN give NaN.
pub fn trig(bits: u32, is_cos: bool) -> u32 {
    let magnitude = bits & MAGNITUDE;
    if magnitude >= HALF_INFINITY {
        return HALF_NAN;
    }
    if magnitude == 0 {
        return if is_cos { HALF_ONE } else { bits };
    }
    let value = half_to_f64(bits).abs();
    let negative = bits & HALF_SIGN != 0 && !is_cos;
    let fraction = value - value.trunc();
    let mut quadrant = (fraction * 4.0).floor() as u32;
    let mut offset = fraction - f64::from(quadrant) * QUARTER;
    if is_cos {
        quadrant = (quadrant + 1) & 3;
    }
    if offset == 0.0 {
        if quadrant & 1 == 0 {
            return 0;
        }
        return if (quadrant == 3) ^ negative {
            HALF_SIGN | HALF_ONE
        } else {
            HALF_ONE
        };
    }
    if quadrant & 1 == 1 {
        offset = QUARTER - offset;
    }
    let result = if offset > EIGHTH {
        ((QUARTER - offset) * TAU).cos()
    } else {
        (offset * TAU).sin()
    };
    let encoded = half_from_f64(result);
    if encoded != 0 && ((quadrant >= 2) ^ negative) {
        encoded ^ HALF_SIGN
    } else {
        encoded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn special_cases_follow_the_model() {
        assert_eq!(rcp(0x8000), 0xFC00);
        assert_eq!(rcp(0xFC00), 0x8000);
        assert_eq!(rcp(0x4000), 0x3800); // 1 / 2
        assert_eq!(sqrt(0x8000), 0x8000);
        assert_eq!(sqrt(0xBC00), HALF_NAN);
        assert_eq!(sqrt(0x4400), 0x4000); // sqrt(4) = 2
        assert_eq!(rsq(0x0000), HALF_INFINITY);
        assert_eq!(rsq(0x7C00), 0);
        assert_eq!(rsq(0x4400), 0x3800); // 1 / sqrt(4)
        assert_eq!(log2(0x8000), 0xFC00);
        assert_eq!(log2(HALF_ONE), 0);
        assert_eq!(log2(0x4400), 0x4000); // log2(4) = 2
        assert_eq!(exp2(0x8000), HALF_ONE);
        assert_eq!(exp2(0xFC00), 0);
        assert_eq!(exp2(0x4C00), HALF_INFINITY); // 2^16
        assert_eq!(exp2(0x4000), 0x4400); // 2^2 = 4
    }

    #[test]
    fn trig_reduces_in_turns_and_keeps_exact_quadrants() {
        assert_eq!(trig(0x8000, false), 0x8000); // sin(-0) = -0
        assert_eq!(trig(0x7BFF, false), 0); // 65504 is a whole number of turns
        assert_eq!(trig(0x7BFF, true), HALF_ONE);
        assert_eq!(trig(0x3400, false), HALF_ONE); // sin(0.25 turn) = 1
        assert_eq!(trig(0xB400, false), HALF_SIGN | HALF_ONE); // sin(-0.25 turn) = -1
        assert_eq!(trig(0x3400, true), 0); // cos(0.25 turn) = +0
        assert_eq!(trig(0x3800, true), HALF_SIGN | HALF_ONE); // cos(0.5 turn) = -1
        assert_eq!(trig(0x7C00, true), HALF_NAN);
    }
}
