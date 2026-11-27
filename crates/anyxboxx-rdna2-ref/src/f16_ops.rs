//! The scalar half-precision operations of the model: `OPERATIONS` of `emu/rdna2_exec_f16.py`, apart from
//! the transcendentals (`v_rcp/sqrt/rsq/log/exp/sin/cos_f16`), which are not ported yet and stay refused.
//!
//! A result is computed in double precision from the 16-bit inputs, then rounded once to binary16 with
//! ties to even (`half_from_f64`, the model's `half_bits`). Any NaN result is `HALF_NAN`, without payload.

use crate::float::{bits_float, frexp, ldexp, py_min};
use crate::half::{
    HALF_INFINITY, HALF_MASK, HALF_NAN, half_from_f64, half_max, half_min, half_to_f32_bits,
    half_to_f64, is_nan,
};

/// How an operation reads one source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A binary16 value: one half of a register, an inline constant or the low half of a literal (`h`).
    Half,
    /// A whole 32-bit word, read as single precision (`w`).
    Word,
    /// A 16-bit integer (`i`), the exponent of `v_ldexp_f16`.
    Integer,
}

use Kind::{Half, Integer, Word};

/// The inputs of one lane: the sources `a`, `b`, `c` as their kinds read them, `d` the low half of the
/// destination before the instruction (the accumulator of `v_fmac_f16`) and `k` the low 16 bits of the
/// literal (the constant of `v_fmamk_f16` and `v_fmaak_f16`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Inputs {
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub d: u32,
    pub k: u32,
}

/// The largest half below 1.0 (`ALMOST_ONE`), the upper bound of `v_fract_f16`.
const ALMOST_ONE: u32 = 0x3BFF;
/// The bound of the `v_ldexp_f16` exponent: past ±64 the half result is already 0 or infinity.
const LDEXP_BOUND: i64 = 64;
/// The magnitude bits of a half (everything but the sign).
const MAGNITUDE: u32 = 0x7FFF;

fn h(bits: u32) -> f64 {
    half_to_f64(bits)
}

/// The source kinds of a half-precision operation, or `None` when `name` is not one.
pub fn kinds(name: &str) -> Option<&'static [Kind]> {
    Some(match name {
        "v_add_f16" | "v_sub_f16" | "v_subrev_f16" | "v_mul_f16" | "v_fmac_f16" | "v_fmamk_f16"
        | "v_fmaak_f16" | "v_max_f16" | "v_min_f16" => &[Half, Half],
        "v_fma_f16" => &[Half, Half, Half],
        "v_ldexp_f16" => &[Half, Integer],
        "v_floor_f16"
        | "v_ceil_f16"
        | "v_trunc_f16"
        | "v_rndne_f16"
        | "v_cvt_f32_f16"
        | "v_frexp_mant_f16"
        | "v_frexp_exp_i16_f16"
        | "v_fract_f16" => &[Half],
        "v_cvt_pkrtz_f16_f32" => &[Word, Word],
        "v_cvt_f16_f32" => &[Word],
        _ => return None,
    })
}

/// `WORD_RESULT`: the operations that write the whole 32-bit register instead of one half.
pub fn is_word_result(name: &str) -> bool {
    matches!(name, "v_cvt_f32_f16" | "v_cvt_pkrtz_f16_f32")
}

/// `INTEGER_RESULT`: a 16-bit integer result, to which the f16 clamp does not apply.
pub fn is_integer_result(name: &str) -> bool {
    name == "v_frexp_exp_i16_f16"
}

/// The result of `name` for one lane, or `None` when `name` is not a half-precision operation.
pub fn compute(name: &str, inputs: Inputs) -> Option<u32> {
    let Inputs { a, b, c, d, k } = inputs;
    Some(match name {
        "v_add_f16" => half_from_f64(h(a) + h(b)),
        "v_sub_f16" => half_from_f64(h(a) - h(b)),
        "v_subrev_f16" => half_from_f64(h(b) - h(a)),
        "v_mul_f16" => half_from_f64(h(a) * h(b)),
        "v_fmac_f16" => half_from_f64(h(a) * h(b) + h(d)),
        "v_fmamk_f16" => half_from_f64(h(a) * h(k) + h(b)),
        "v_fmaak_f16" => half_from_f64(h(a) * h(b) + h(k)),
        "v_fma_f16" => half_from_f64(h(a) * h(b) + h(c)),
        "v_max_f16" => half_max(a, b),
        "v_min_f16" => half_min(a, b),
        "v_ldexp_f16" => {
            let exponent = i64::from(b as u16 as i16).clamp(-LDEXP_BOUND, LDEXP_BOUND);
            half_from_f64(ldexp(h(a), exponent))
        }
        "v_floor_f16" => round_integral(a, f64::floor),
        "v_ceil_f16" => round_integral(a, f64::ceil),
        "v_trunc_f16" => round_integral(a, f64::trunc),
        "v_rndne_f16" => round_integral(a, f64::round_ties_even),
        "v_cvt_pkrtz_f16_f32" => toward_zero(bits_float(b)) << 16 | toward_zero(bits_float(a)),
        "v_cvt_f16_f32" => half_from_f64(bits_float(a)),
        "v_cvt_f32_f16" => half_to_f32_bits(a),
        "v_frexp_mant_f16" => half_frexp(a).0,
        "v_frexp_exp_i16_f16" => half_frexp(a).1 as u32 & HALF_MASK,
        "v_fract_f16" => half_fract(a),
        _ => return None,
    })
}

/// `half_round`: an integer of the value, sign kept as in the f32 forms (IEEE 754 roundToIntegral:
/// trunc(-0.5) = -0); ±infinity unchanged, NaN gives `HALF_NAN`.
fn round_integral(bits: u32, function: fn(f64) -> f64) -> u32 {
    if is_nan(bits) {
        return HALF_NAN;
    }
    if bits & MAGNITUDE == HALF_INFINITY {
        return bits;
    }
    let value = h(bits);
    half_from_f64(function(value).copysign(value))
}

/// `half_toward_zero`: the nearest half, stepped back by one when it is farther from zero than `value`.
/// A finite value too large gives ±65504, never infinity; ±infinity stays, NaN gives `HALF_NAN`.
fn toward_zero(value: f64) -> u32 {
    let bits = half_from_f64(value);
    if !value.is_finite() {
        return bits;
    }
    if h(bits).abs() > value.abs() {
        bits - 1
    } else {
        bits
    }
}

/// `half_frexp`: (mantissa, exponent) with value = m × 2^e and 0.5 ≤ |m| < 1 (subnormals normalised).
/// ±0, ±infinity and NaN give the bits unchanged and exponent 0, as `v_frexp_*_f32`.
fn half_frexp(bits: u32) -> (u32, i32) {
    let magnitude = bits & MAGNITUDE;
    if magnitude == 0 || magnitude >= HALF_INFINITY {
        return (bits, 0);
    }
    let (mantissa, exponent) = frexp(h(bits));
    (half_from_f64(mantissa), exponent)
}

/// `half_fract`: S0 - floor(S0), at most `ALMOST_ONE` as in `v_fract_f32`; NaN for ±infinity and NaN.
/// The bound is applied before the rounding to binary16, as in the model.
fn half_fract(bits: u32) -> u32 {
    if is_nan(bits) || bits & MAGNITUDE == HALF_INFINITY {
        return HALF_NAN;
    }
    let value = h(bits);
    half_from_f64(py_min(value - value.floor().copysign(value), h(ALMOST_ONE)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unary(name: &str, a: u32) -> u32 {
        compute(
            name,
            Inputs {
                a,
                ..Inputs::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn rounding_keeps_the_sign_of_the_source() {
        assert_eq!(unary("v_trunc_f16", 0xB800), 0x8000); // trunc(-0.5) = -0
        assert_eq!(unary("v_ceil_f16", 0xB800), 0x8000); // ceil(-0.5) = -0
        assert_eq!(unary("v_floor_f16", 0xB800), 0xBC00); // floor(-0.5) = -1
        assert_eq!(unary("v_rndne_f16", 0x3C00 | 0x200), 0x4000); // 1.5 rounds to even: 2
        assert_eq!(unary("v_rndne_f16", 0x4100), 0x4000); // 2.5 rounds to even: 2
        assert_eq!(unary("v_floor_f16", 0xFC00), 0xFC00);
        assert_eq!(unary("v_floor_f16", 0x7C01), HALF_NAN);
    }

    #[test]
    fn fract_is_bounded_before_rounding() {
        // -2^-24 + 1 is above ALMOST_ONE: the bound gives 0x3BFF, not the rounded 1.0.
        assert_eq!(unary("v_fract_f16", 0x8001), ALMOST_ONE);
        assert_eq!(unary("v_fract_f16", 0xB400), 0x3A00); // fract(-0.25) = 0.75
        assert_eq!(unary("v_fract_f16", 0x8000), 0); // -0 gives +0
        assert_eq!(unary("v_fract_f16", 0x7C00), HALF_NAN);
    }

    #[test]
    fn frexp_normalises_subnormals_and_keeps_specials() {
        assert_eq!(unary("v_frexp_mant_f16", 0x0001), 0x3800); // 2^-24 = 0.5 × 2^-23
        assert_eq!(
            unary("v_frexp_exp_i16_f16", 0x0001),
            (-23i32) as u32 & HALF_MASK
        );
        assert_eq!(unary("v_frexp_mant_f16", 0xC200), 0xBA00); // -3 = -0.75 × 2^2
        assert_eq!(unary("v_frexp_exp_i16_f16", 0xC200), 2);
        assert_eq!(unary("v_frexp_mant_f16", 0xFC00), 0xFC00);
        assert_eq!(unary("v_frexp_exp_i16_f16", 0x8000), 0);
    }

    #[test]
    fn pkrtz_rounds_toward_zero_and_saturates_finite_values() {
        let pack = |a: f32, b: f32| {
            compute(
                "v_cvt_pkrtz_f16_f32",
                Inputs {
                    a: a.to_bits(),
                    b: b.to_bits(),
                    ..Inputs::default()
                },
            )
            .unwrap()
        };
        // 1 + 2^-11 + 2^-12 rounds to nearest up (0x3C01) but toward zero down (0x3C00).
        assert_eq!(pack(1.0 + 2f32.powi(-11) + 2f32.powi(-12), 0.0), 0x3C00);
        assert_eq!(pack(1e9, -1e9), 0xFBFF_7BFF);
        assert_eq!(
            pack(f32::INFINITY, f32::NAN),
            (HALF_NAN << 16) | HALF_INFINITY
        );
    }

    #[test]
    fn ldexp_reads_a_signed_16_bit_exponent() {
        let ldexp = |a: u32, b: u32| {
            compute(
                "v_ldexp_f16",
                Inputs {
                    a,
                    b,
                    ..Inputs::default()
                },
            )
            .unwrap()
        };
        assert_eq!(ldexp(0x3C00, 0xFFFF), 0x3800); // 1 × 2^-1
        assert_eq!(ldexp(0x3C00, 0x7FFF), HALF_INFINITY);
        assert_eq!(ldexp(0x3C00, 0x8000), 0);
    }

    #[test]
    fn the_transcendentals_are_not_ported_yet() {
        for name in ["v_rcp_f16", "v_sqrt_f16", "v_sin_f16"] {
            assert!(kinds(name).is_none());
            assert!(compute(name, Inputs::default()).is_none());
        }
    }
}
