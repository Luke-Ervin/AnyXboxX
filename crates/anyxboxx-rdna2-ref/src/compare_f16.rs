//! The half-precision compares (`COMPAREF16` of `emu/rdna2_exec_cmpf16.py`): `v_cmp[x]_<test>_f16` for the 16
//! float tests of the f32 compares, and `v_cmp[x]_class_f16`, in VOPC and VOP3.
//!
//! S0[15:0] and S1[15:0] are binary16 values, read like the other f16 sources (`f16::source`: a register gives
//! its low half, an inline float constant its binary16 encoding, an integer constant or the literal its low
//! 16 bits); the high halves are ignored. Ordered tests are false on NaN and their `n` negations true; -0 equals
//! +0. `v_cmp_class_f16` tests the IEEE class of S0 against the bit mask S1. The mask is written whole
//! (inactive lanes cleared): VCC (VOPC `v_cmp`), EXEC (`v_cmpx`), or the SGPR destination (VOP3 `v_cmp`).
//!
//! Refused, as in the model: the SDWA and DPP forms, and in VOP3 op_sel (not usable on VOPC per the ISA,
//! section 6.2.5), clamp, omod, and abs or neg on the class mask.

use crate::compare;
use crate::error::Error;
use crate::f16;
use crate::f16_ops::Kind;
use crate::half::{HALF_INFINITY, HALF_SIGN, half_to_f64};
use crate::vop3::Fields;
use crate::wave::{EXEC_LO, VCC_LO, VGPR_FIRST, Wave};

/// Source codes of the DPP8, DPP8_FI, SDWA and DPP16 forms.
const EXTENSION_CODES: [u32; 4] = [0xE9, 0xEA, 0xF9, 0xFA];
/// `op_sel`: bits 11 to 14 of a VOP3 word.
const OP_SEL_SHIFT: u32 = 11;
const OP_SEL_MASK: u64 = 0xF;
const MANTISSA: u32 = 0x3FF;
const QUIET: u32 = 0x200;

/// What a compare tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Predicate<'a> {
    /// A float test of `FLOAT_TESTS` by its name (`lt`, `nge`, ...).
    Float(&'a str),
    /// `v_cmp_class_f16`.
    Class,
}

/// (writes EXEC, predicate) of a half-precision compare, `None` for any other name.
fn split(name: &str) -> Option<(bool, Predicate<'_>)> {
    let (writes_exec, rest) = match name.strip_prefix("v_cmpx_") {
        Some(rest) => (true, rest),
        None => (false, name.strip_prefix("v_cmp_")?),
    };
    let test = rest.strip_suffix("_f16")?;
    if test == "class" {
        return Some((writes_exec, Predicate::Class));
    }
    compare::is_float_test(test).then_some((writes_exec, Predicate::Float(test)))
}

/// Whether `name` is one of the 34 half-precision compares.
pub fn is_compare(name: &str) -> bool {
    split(name).is_some()
}

/// The IEEE class bit (0 to 9) of a half, as `float_class(bits, 16)` of the model: 0 signalling NaN, 1 quiet
/// NaN, 2 -infinity, 3 -normal, 4 -subnormal, 5 -0, 6 +0, 7 +subnormal, 8 +normal, 9 +infinity.
fn half_class(bits: u32) -> u32 {
    let negative = bits & HALF_SIGN != 0;
    let exponent = bits & HALF_INFINITY;
    let mantissa = bits & MANTISSA;
    if exponent == HALF_INFINITY {
        if mantissa != 0 {
            return u32::from(mantissa & QUIET != 0);
        }
        return if negative { 2 } else { 9 };
    }
    if exponent == 0 {
        if mantissa == 0 {
            return if negative { 5 } else { 6 };
        }
        return if negative { 4 } else { 7 };
    }
    if negative { 3 } else { 8 }
}

fn holds(predicate: Predicate<'_>, a: u32, b: u32) -> bool {
    match predicate {
        Predicate::Float(test) => {
            compare::float_named(test, half_to_f64(a), half_to_f64(b)).unwrap_or(false)
        }
        Predicate::Class => (b >> half_class(a)) & 1 == 1,
    }
}

fn spec(name: &str) -> Result<(bool, Predicate<'_>), Error> {
    split(name).ok_or_else(|| Error::Untranslated(format!("instruction non traduite : {name}")))
}

/// VOPC: S0 is `src0`, S1 the VGPR `vsrc1`; the mask goes to VCC or EXEC.
pub fn execute(
    wave: &mut Wave,
    name: &str,
    src0: u32,
    vsrc1: u32,
    literal: Option<u32>,
) -> Result<(), Error> {
    if EXTENSION_CODES.contains(&src0) {
        return Err(Error::Untranslated(format!(
            "{name}: SDWA/DPP form (src0 = 0x{src0:X}) not supported for f16"
        )));
    }
    let (writes_exec, predicate) = spec(name)?;
    let mut mask = 0u64;
    for lane in wave.active()? {
        let a = f16::source(wave, name, Kind::Half, src0, lane, literal, false)?;
        let b = f16::source(
            wave,
            name,
            Kind::Half,
            VGPR_FIRST + vsrc1,
            lane,
            literal,
            false,
        )?;
        if holds(predicate, a, b) {
            mask |= 1 << lane;
        }
    }
    wave.write_mask(if writes_exec { EXEC_LO } else { VCC_LO }, mask)
}

/// VOP3 (`word` is the 64-bit encoding): S0 and S1 from the first two sources, abs and neg on bit 15 of a
/// float source; the mask goes to the SGPR `vdst` or to EXEC.
pub fn execute_vop3(
    wave: &mut Wave,
    name: &str,
    word: u64,
    literal: Option<u32>,
) -> Result<(), Error> {
    let (writes_exec, predicate) = spec(name)?;
    let fields = Fields::new(word);
    if (word >> OP_SEL_SHIFT) & OP_SEL_MASK != 0 {
        return Err(Error::Untranslated(format!(
            "{name} with op_sel (VOP3 compare, 16-bit sources): not usable per the RDNA 2 ISA"
        )));
    }
    if fields.clamp || fields.omod != 0 {
        return Err(Error::Untranslated(format!(
            "{name} with clamp or omod (VOP3 f16 compare): not measured on RDNA 2"
        )));
    }
    // The class mask takes no modifier (the model's kind "i").
    let mask_kind = if predicate == Predicate::Class {
        Kind::Integer
    } else {
        Kind::Half
    };
    if mask_kind == Kind::Integer && ((fields.abs | fields.neg) >> 1) & 1 == 1 {
        return Err(Error::Untranslated(format!(
            "{name} with abs/neg on the class mask S1: not measured on RDNA 2"
        )));
    }
    let mut mask = 0u64;
    for lane in wave.active()? {
        let read = |code| f16::source(wave, name, Kind::Half, code, lane, literal, false);
        let a = f16::modified(read(fields.sources[0])?, Kind::Half, &fields, 0);
        let b = f16::modified(read(fields.sources[1])?, mask_kind, &fields, 1);
        if holds(predicate, a, b) {
            mask |= 1 << lane;
        }
    }
    wave.write_mask(if writes_exec { EXEC_LO } else { fields.vdst }, mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_34_names_are_recognised() {
        let tests = [
            "f", "lt", "eq", "le", "gt", "lg", "ge", "o", "u", "nge", "nlg", "ngt", "nle", "neq",
            "nlt", "tru", "class",
        ];
        for prefix in ["v_cmp", "v_cmpx"] {
            for test in tests {
                assert!(
                    is_compare(&format!("{prefix}_{test}_f16")),
                    "{prefix}_{test}"
                );
            }
        }
        assert!(!is_compare("v_cmp_ne_f16")); // `ne` is an integer test
        assert!(!is_compare("v_cmp_lt_f32"));
        assert!(!is_compare("v_cmp_lt_u16"));
    }

    #[test]
    fn half_classes_follow_the_model() {
        let cases = [
            (0x7C01, 0),
            (0x7E00, 1),
            (0xFC00, 2),
            (0xBC00, 3),
            (0x8001, 4),
            (0x8000, 5),
            (0x0000, 6),
            (0x03FF, 7),
            (0x0400, 8),
            (0x7C00, 9),
        ];
        for (bits, class) in cases {
            assert_eq!(half_class(bits), class, "0x{bits:04X}");
        }
    }

    #[test]
    fn ordered_tests_are_false_on_nan_and_zeros_are_equal() {
        assert!(holds(Predicate::Float("eq"), 0x8000, 0x0000));
        assert!(!holds(Predicate::Float("lt"), 0x7E00, 0x3C00));
        assert!(holds(Predicate::Float("nge"), 0x7E00, 0x3C00));
        assert!(holds(Predicate::Float("u"), 0x3C00, 0x7C01));
        assert!(holds(Predicate::Float("lt"), 0x0001, 0x0002)); // subnormals by value
        assert!(holds(Predicate::Class, 0x7C00, 1 << 9));
        assert!(!holds(Predicate::Class, 0x7C00, 1 << 2));
    }
}
