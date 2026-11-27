//! Running the scalar half-precision instructions (VOP1, VOP2 and their VOP3 forms), ported from
//! `execute`, `execute_vop3`, `_source`, `_check_vop3` and `_modified` of `emu/rdna2_exec_f16.py`. The
//! operations themselves are in `f16_ops`.
//!
//! A half source is the low half of a register (the high half with VOP3 `op_sel`), a half inline constant,
//! or the low 16 bits of an integer constant or literal. The result goes to the low half of `vdst` (the high
//! half with VOP3 `op_sel` bit 3) and the other half is kept, except for the two operations that write the
//! whole register. Forms whose behaviour is not sourced (SDWA and DPP, `omod`, some `op_sel` and `clamp`
//! uses) are refused with `Untranslated` before any lane is written, as in the model.

use crate::error::Error;
use crate::f16_ops::{self, Inputs, Kind};
use crate::half::{HALF_MASK, HALF_SIGN, half_clamp, inline_half};
use crate::vector;
use crate::vop3::Fields;
use crate::wave::Wave;

/// The first VGPR source code.
const VGPR_FIRST: u32 = 256;
/// Source codes below this one are registers (SGPRs, VCC, M0, NULL, EXEC), read as words.
const SCALAR_REGISTER_END: u32 = 128;
/// Source codes of the DPP8, DPP8_FI, SDWA and DPP16 forms.
const EXTENSION_CODES: [u32; 4] = [0xE9, 0xEA, 0xF9, 0xFA];
/// The sign of a 32-bit word source.
const WORD_SIGN: u32 = 0x8000_0000;
/// `op_sel`: bits 11 to 14 of a VOP3 word (sources 0 to 2, then the destination).
const OP_SEL_SHIFT: u32 = 11;
const OP_SEL_MASK: u64 = 0xF;
const OP_SEL_DESTINATION: u32 = 1 << 3;

fn refuse(message: String) -> Error {
    Error::Untranslated(message)
}

/// Whether `name` is a half-precision operation this module runs.
pub fn is_operation(name: &str) -> bool {
    f16_ops::kinds(name).is_some()
}

fn kinds(name: &str) -> Result<&'static [Kind], Error> {
    f16_ops::kinds(name).ok_or_else(|| refuse(format!("instruction non traduite : {name}")))
}

fn compute(name: &str, inputs: Inputs) -> Result<u32, Error> {
    f16_ops::compute(name, inputs)
        .ok_or_else(|| refuse(format!("instruction non traduite : {name}")))
}

/// `_source`: a source as its kind reads it, for one lane.
fn source(
    wave: &Wave,
    name: &str,
    kind: Kind,
    code: u32,
    lane: usize,
    literal: Option<u32>,
    high: bool,
) -> Result<u32, Error> {
    if kind == Kind::Word {
        return wave.read_lane(code, lane, literal);
    }
    if !(SCALAR_REGISTER_END..VGPR_FIRST).contains(&code) {
        let word = wave.read_lane(code, lane, literal)?;
        return Ok(if high { word >> 16 } else { word & HALF_MASK });
    }
    if high {
        return Err(refuse(format!(
            "{name} : op_sel sur la moitié haute d'une constante non pris en charge"
        )));
    }
    if let Some(bits) = inline_half(code) {
        if kind == Kind::Integer {
            return Err(refuse(format!(
                "{name} : constante flottante sur une source entière non prise en charge"
            )));
        }
        return Ok(bits);
    }
    // An integer constant or the literal: its low 16 bits.
    Ok(wave.read_scalar(code, literal, 1)? as u32 & HALF_MASK)
}

/// VOP1 and VOP2: sources and result in the low half; the high half of `vdst` is kept.
pub fn execute(
    wave: &mut Wave,
    name: &str,
    fields: &vector::Fields,
    literal: Option<u32>,
) -> Result<(), Error> {
    if EXTENSION_CODES.contains(&fields.src0) {
        return Err(refuse(format!(
            "{name} : forme SDWA/DPP (src0 = 0x{:X}) non prise en charge en f16",
            fields.src0
        )));
    }
    let kinds = kinds(name)?;
    let codes: Vec<u32> = match fields.vsrc1 {
        Some(vsrc1) => vec![fields.src0, VGPR_FIRST + vsrc1],
        None => vec![fields.src0],
    };
    let k = literal.unwrap_or(0) & HALF_MASK;
    let vdst = fields.vdst as usize;
    for lane in wave.active()? {
        let mut values = [0u32; 2];
        for (index, (kind, code)) in kinds.iter().zip(&codes).enumerate() {
            values[index] = source(wave, name, *kind, *code, lane, literal, false)?;
        }
        let old = wave.v[vdst][lane];
        let inputs = Inputs {
            a: values[0],
            b: values[1],
            c: 0,
            d: old & HALF_MASK,
            k,
        };
        let result = compute(name, inputs)?;
        wave.v[vdst][lane] = if f16_ops::is_word_result(name) {
            result
        } else {
            (old & !HALF_MASK) | result
        };
    }
    Ok(())
}

fn bit(value: u32, index: usize) -> bool {
    (value >> index) & 1 == 1
}

/// `_check_vop3`: the VOP3 forms whose behaviour is not sourced, refused before any lane is touched.
fn check_vop3(
    name: &str,
    kinds: &[Kind],
    fields: &Fields,
    op_sel: u32,
    literal: Option<u32>,
) -> Result<(), Error> {
    let word_result = f16_ops::is_word_result(name);
    if word_result && (fields.clamp || fields.omod != 0) {
        return Err(refuse(format!(
            "{name} : clamp ou omod sur un résultat f32 non pris en charge"
        )));
    }
    if fields.omod != 0 {
        return Err(refuse(format!(
            "{name} : omod sur un résultat f16 non pris en charge"
        )));
    }
    if fields.clamp && f16_ops::is_integer_result(name) {
        return Err(refuse(format!(
            "{name} : clamp sur un résultat entier non mesuré"
        )));
    }
    if op_sel != 0 && name == "v_fmac_f16" {
        return Err(refuse(format!(
            "{name} : op_sel non pris en charge en VOP3"
        )));
    }
    if op_sel & OP_SEL_DESTINATION != 0 && word_result {
        return Err(refuse(format!(
            "{name} : op_sel sur une destination de 32 bits non pris en charge"
        )));
    }
    if matches!(name, "v_fmamk_f16" | "v_fmaak_f16") && literal.is_none() {
        return Err(refuse(format!(
            "{name} (VOP3) : sans littéral, la constante K n'existe pas"
        )));
    }
    for (index, kind) in kinds.iter().enumerate() {
        if *kind == Kind::Word && bit(op_sel, index) {
            return Err(refuse(format!(
                "{name} : op_sel sur une source de 32 bits non pris en charge"
            )));
        }
        if *kind == Kind::Integer && bit(fields.abs | fields.neg, index) {
            return Err(refuse(format!(
                "{name} : abs/neg sur une source entière non pris en charge"
            )));
        }
    }
    Ok(())
}

/// `_modified`: VOP3 abs and neg on the sign of the half (bit 15) or of the word (bit 31).
fn modified(value: u32, kind: Kind, fields: &Fields, index: usize) -> u32 {
    let sign = match kind {
        Kind::Integer => return value,
        Kind::Word => WORD_SIGN,
        Kind::Half => HALF_SIGN,
    };
    let mut value = value;
    if bit(fields.abs, index) {
        value &= !sign;
    }
    if bit(fields.neg, index) {
        value ^= sign;
    }
    value
}

/// VOP3 (`word` is the 64-bit encoding): `op_sel` bits 0 to 2 read the high half of a source, bit 3 writes
/// the result to the high half of `vdst` and keeps the low one; clamp bounds the half result to [0, 1].
pub fn execute_vop3(
    wave: &mut Wave,
    name: &str,
    word: u64,
    literal: Option<u32>,
) -> Result<(), Error> {
    let fields = Fields::new(word);
    let op_sel = ((word >> OP_SEL_SHIFT) & OP_SEL_MASK) as u32;
    let kinds = kinds(name)?;
    check_vop3(name, kinds, &fields, op_sel, literal)?;
    let high = op_sel & OP_SEL_DESTINATION != 0;
    let k = literal.unwrap_or(0) & HALF_MASK;
    let vdst = fields.vdst as usize;
    for lane in wave.active()? {
        let mut values = [0u32; 3];
        for (index, (kind, code)) in kinds.iter().zip(fields.sources).enumerate() {
            let raw = source(wave, name, *kind, code, lane, literal, bit(op_sel, index))?;
            values[index] = modified(raw, *kind, &fields, index);
        }
        let old = wave.v[vdst][lane];
        let inputs = Inputs {
            a: values[0],
            b: values[1],
            c: values[2],
            d: old & HALF_MASK,
            k,
        };
        let mut result = compute(name, inputs)?;
        if fields.clamp {
            result = half_clamp(result);
        }
        wave.v[vdst][lane] = if f16_ops::is_word_result(name) {
            result
        } else if high {
            (old & HALF_MASK) | result << 16
        } else {
            (old & !HALF_MASK) | result
        };
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const V1: u32 = VGPR_FIRST + 1;

    fn vop3_word(opcode: u32, vdst: u32, sources: [u32; 3], op_sel: u32, clamp: bool) -> u64 {
        let low = 0x35 << 26 | opcode << 16 | vdst | op_sel << 11 | u32::from(clamp) << 15;
        let high = sources[0] | sources[1] << 9 | sources[2] << 18;
        u64::from(low) | u64::from(high) << 32
    }

    #[test]
    fn a_vop2_add_writes_the_low_half_and_keeps_the_high_one() {
        let mut wave = Wave::new(32).unwrap();
        wave.v[1][0] = 0x1234_3C00; // low half 1.0
        wave.v[2][0] = 0xABCD_0000;
        let fields = vector::Fields {
            vdst: 2,
            src0: 242, // inline 1.0
            vsrc1: Some(1),
        };
        execute(&mut wave, "v_add_f16", &fields, None).unwrap();
        assert_eq!(wave.v[2][0], 0xABCD_4000); // 2.0 in the low half
    }

    #[test]
    fn op_sel_reads_and_writes_high_halves_in_vop3() {
        let mut wave = Wave::new(32).unwrap();
        wave.v[1][0] = 0x4000_3C00; // high 2.0, low 1.0
        wave.v[3][0] = 0x0000_5555;
        // v_mul_f16 (VOP3 0x135) v3, v1.h, v1.l, result to the high half: op_sel = 0b1001.
        let word = vop3_word(0x135, 3, [V1, V1, 0], 0b1001, false);
        execute_vop3(&mut wave, "v_mul_f16", word, None).unwrap();
        assert_eq!(wave.v[3][0], 0x4000_5555);
    }

    #[test]
    fn unsourced_vop3_forms_are_refused_before_any_lane() {
        let mut wave = Wave::new(32).unwrap();
        wave.v[1][0] = 0x3C00;
        let omod = vop3_word(0x132, 2, [V1, V1, 0], 0, false) | 1 << 59;
        let high_constant = vop3_word(0x132, 2, [242, V1, 0], 0b0001, false);
        let fmac_op_sel = vop3_word(0x136, 2, [V1, V1, 0], 0b0010, false);
        for (name, word) in [
            ("v_add_f16", omod),
            ("v_add_f16", high_constant),
            ("v_fmac_f16", fmac_op_sel),
        ] {
            let error = execute_vop3(&mut wave, name, word, None).unwrap_err();
            assert!(matches!(error, Error::Untranslated(_)), "{name}: {error:?}");
        }
        assert_eq!(wave.v[2][0], 0);
    }

    #[test]
    fn extension_forms_are_refused_in_vop1_and_vop2() {
        let mut wave = Wave::new(32).unwrap();
        let fields = vector::Fields {
            vdst: 2,
            src0: 0xF9,
            vsrc1: Some(1),
        };
        let error = execute(&mut wave, "v_add_f16", &fields, Some(0)).unwrap_err();
        assert!(matches!(error, Error::Untranslated(_)));
    }
}
