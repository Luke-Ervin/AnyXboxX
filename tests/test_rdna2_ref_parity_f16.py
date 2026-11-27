"""Parity of the scalar half-precision instructions between the Python model (emu/rdna2_exec_f16.py) and the
Rust port (crates/anyxboxx-rdna2-ref, f16.rs and f16_ops.rs): VOP1, VOP2 and their VOP3 forms, with the VOP3
modifiers (abs, neg, op_sel, clamp) and the forms the model refuses. The transcendentals are not ported yet
and are left out (F16_NAMES). The random generator of test_rdna2_ref_parity.py draws these names too.
"""
import random

from test_rdna2_ref_parity import (F16_NAMES, LITERAL, OPCODES, SEED, V0, assert_same, binary, program,
                                   run_python, vop1, vop2, vop3, words)

# Halves at the edges: signed zeros, ±1, infinities, quiet and signalling NaNs, the largest finite value,
# subnormals, the smallest normal, ties for the rounding instructions (0.5, 1.5, 2.5) and the value just
# below 1.0 (the bound of v_fract_f16).
HALF_SPECIALS = (0x0000, 0x8000, 0x3C00, 0xBC00, 0x7C00, 0xFC00, 0x7E00, 0x7C01, 0xFE01, 0x7BFF, 0xFBFF,
                 0x0001, 0x8001, 0x03FF, 0x0400, 0x3800, 0xB800, 0x3E00, 0x4100, 0xC100, 0x3BFF, 0x6400, 0x5BFF)
# Single-precision words for the f32 sources (v_cvt_f16_f32, v_cvt_pkrtz_f16_f32): 65504, the first value
# that rounds to infinity (65520) and the one just below it, 2^-24 and its tie 2^-25, 1 + 2^-11 + 2^-12
# (rounds up to nearest, down toward zero), infinities, NaNs and the largest finite f32.
WORD_SPECIALS = (0x3F800000, 0xBF800000, 0x477FE000, 0x477FF000, 0x477FEFFF, 0x33800000, 0x33000000,
                 0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001, 0x7F7FFFFF, 0x3F801800, 0x00000001,
                 0x80000000)
ENCODINGS = sorted((family, name) for family, name in OPCODES
                   if name in F16_NAMES and family in ("VOP1", "VOP2", "VOP3"))
LITERAL_K = ("v_fmamk_f16", "v_fmaak_f16")
VDST = 9
MBCNT_LO, MINUS_ONE, ZERO = "v_mbcnt_lo_u32_b32", 193, 128
CONSTANTS = (128, 129, 136, 193, 200, 208, *range(240, 249))
F16_PROGRAMS = 400


def operand(rng):
    """A 32-bit operand: two special halves, a special f32 word, or random bits."""
    roll = rng.random()
    if roll < 0.4:
        return rng.choice(HALF_SPECIALS) << 16 | rng.choice(HALF_SPECIALS)
    if roll < 0.6:
        return rng.choice(WORD_SPECIALS)
    return rng.getrandbits(32)


def setup(rng):
    """v1 to v3 and the destination loaded with operands; v4 different in every lane (lane index × K)."""
    loads = []
    for index in (1, 2, 3, VDST):
        loads += [vop1("v_mov_b32", index, LITERAL), operand(rng)]
    loads += [vop3(MBCNT_LO, 5, (MINUS_ONE, ZERO, 0)), vop2("v_mul_u32_u24", 4, LITERAL, 5), 0x00C0_FFEE]
    return loads


def instruction(family, name, sources, literal, modifiers=None):
    """The words of one f16 instruction; the literal follows when a source is the literal code, or always for
    the VOP2 forms of v_fmamk_f16 and v_fmaak_f16 (their constant K)."""
    if family == "VOP1":
        encoded = [vop1(name, VDST, sources[0])]
    elif family == "VOP2":
        encoded = [vop2(name, VDST, sources[0], sources[1] - V0)]
    else:
        encoded = list(vop3(name, VDST, sources, **(modifiers or {})))
    if LITERAL in sources[:1 if family != "VOP3" else 3] or (family == "VOP2" and name in LITERAL_K):
        encoded.append(literal)
    return encoded


def plain_sources(family, name):
    if family == "VOP3" and name in LITERAL_K:
        return (V0 + 1, LITERAL, V0 + 3)          # VOP3 has no K without a literal word
    return (V0 + 1, V0 + 2, V0 + 3)


def run_both(exe, program_words, scalars=()):
    for lanes in (32, 64):
        assert_same(exe, lanes, False, list(scalars), program_words)
    return run_python(32, False, list(scalars), program_words)


def test_every_half_precision_encoding_runs_in_both_models():
    """Each operation in each of its encodings, with register operands: the model runs it, and the Rust
    must give the same registers. Fails on a binary that refuses the f16 instructions."""
    exe = binary()
    rng = random.Random(SEED + 16)
    assert len(ENCODINGS) >= 40, ENCODINGS
    for family, name in ENCODINGS:
        program_words = program(*words(*setup(rng), *instruction(family, name, plain_sources(family, name),
                                                                  0x1234_3E00)))
        python = run_both(exe, program_words)
        assert python.startswith("steps"), (family, name, python)


def refused_cases():
    """(family, name, sources, modifiers): forms the model refuses with Untranslated before any lane."""
    registers = (V0 + 1, V0 + 2, V0 + 3)
    return [
        ("VOP3", "v_add_f16", registers, {"omod": 1}),
        ("VOP3", "v_cvt_f32_f16", registers, {"clamp": 1}),
        ("VOP3", "v_cvt_pkrtz_f16_f32", registers, {"op_sel": 0b1000}),
        ("VOP3", "v_cvt_pkrtz_f16_f32", registers, {"op_sel": 0b0010}),
        ("VOP3", "v_frexp_exp_i16_f16", registers, {"clamp": 1}),
        ("VOP3", "v_fmac_f16", registers, {"op_sel": 0b0001}),
        ("VOP3", "v_fmamk_f16", registers, {}),
        ("VOP3", "v_ldexp_f16", registers, {"abs_bits": 0b010}),
        ("VOP3", "v_ldexp_f16", (V0 + 1, 242, 0), {}),
        ("VOP3", "v_add_f16", (242, V0 + 1, 0), {"op_sel": 0b0001}),
        ("VOP2", "v_add_f16", (0xF9, V0 + 2), None),
        ("VOP1", "v_floor_f16", (0xFA,), None),
    ]


def test_half_precision_refusals_match_the_python_model():
    exe = binary()
    rng = random.Random(SEED + 17)
    for family, name, sources, modifiers in refused_cases():
        encoded = instruction(family, name, sources, 0x3C00, modifiers)
        if family != "VOP3" and sources[0] in (0xF9, 0xFA):
            encoded.append(rng.getrandbits(32))   # the extension word
        python = run_both(exe, program(*words(*setup(rng), *encoded)))
        assert python == "error untranslated\n", (family, name, modifiers, python)


def random_sources(rng, count):
    pool = [V0 + 1, V0 + 2, V0 + 3, V0 + 4, V0 + VDST, 0, 1, LITERAL, *CONSTANTS]
    return tuple(rng.choice(pool) if rng.random() < 0.5 else V0 + rng.choice((1, 2, 3, 4)) for _ in range(count))


def random_modifiers(rng):
    if rng.random() < 0.4:
        return {}
    modifiers = {"op_sel": rng.randrange(16), "abs_bits": rng.randrange(8), "neg": rng.randrange(8),
                 "clamp": rng.randrange(2)}
    if rng.random() < 0.1:
        modifiers["omod"] = rng.randrange(1, 4)
    return modifiers


def random_case(rng):
    """One program: the setup, then one f16 instruction with random operands and modifiers."""
    family, name = rng.choice(ENCODINGS)
    sources = random_sources(rng, 3)
    if family == "VOP2":
        sources = (sources[0], V0 + rng.choice((1, 2, 3, 4, VDST)))    # vsrc1 is a VGPR
    modifiers = random_modifiers(rng) if family == "VOP3" else None
    literal = operand(rng)
    scalars = [(0, operand(rng)), (1, operand(rng))]
    return program(*words(*setup(rng), *instruction(family, name, sources, literal, modifiers))), scalars


def test_random_half_precision_programs_match_the_python_model():
    exe = binary()
    rng = random.Random(SEED + 18)
    executed = 0
    for _ in range(F16_PROGRAMS):
        program_words, scalars = random_case(rng)
        executed += run_both(exe, program_words, scalars).startswith("steps")
    # Most programs must run, not be refused, for the comparison to cover the operations.
    assert executed > F16_PROGRAMS // 2, executed
