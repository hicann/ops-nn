# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
White-box precision test for rms_norm_hi_f4_quant torch_extension.

Covers the scale_alg modification (Eb/Ec threshold: mse_first=4/2, unbiased=3.75/1.875),
output_rstd (optional rstd), beta on/off, dtypes (fp16/bf16), gamma dtype (fp16/bf16/fp32)
and ranks (1D/2D/3D).  Golden is derived from the SE formula; scale is compared byte-exact,
rstd numerically and y at nibble (+/-1 LSB) tolerance.
"""

import numpy as np
import torch
import cann_ops_nn_custom as cann_ops_nn

from ml_dtypes import bfloat16
from en_dtypes import float4_e1m2

HIF4_BLOCK = 64
EXP_BIAS = 0x3E12  # ~= 1/7 in bf16
CONST_REC = np.uint16(EXP_BIAS).view(bfloat16)

SCALE_ALG_THD = {
    0: (4.0, 2.0),  # mse_first
    1: (3.75, 1.875),  # unbiased
}


def rms_golden(x, gamma, beta, epsilon):
    xf = x.cpu().float()
    gf = gamma.cpu().float()
    rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    y3 = xf * rstd * gf
    if beta is not None:
        y3 = y3 + beta.cpu().float()
    return y3, rstd


def bf16(x):
    return np.float32(x).astype(bfloat16)


def E6M2_REC(x):
    if np.isnan(x) or np.isinf(x) or x <= 0:
        return bfloat16(0.0)
    try:
        E6 = np.floor(np.log2(x + 2 ** (-1000)))
        if not np.isfinite(E6):
            return bfloat16(0.0)
        M2 = x * 2 ** (-E6 + 2) - 4.0
        M2 = int(np.clip(np.round(M2), 0, 3))
        M7 = {0: 0.0, 1: 77, 2: 43, 3: 18}[M2]
        E8 = -E6 if M2 == 0 else -E6 - 1
        if not np.isfinite(E8) or E8 > 127 or E8 < -128:
            return bfloat16(0.0)
        return bfloat16(2.0**E8 * (1 + M7 * 2.0 ** (-7)))
    except Exception:
        return bfloat16(0.0)


def BF16_to_E6M2(x):
    if np.isinf(x) or np.isnan(x):
        return bfloat16(0.0)
    if x == 0 or x < 2 ** (-48) or x >= 1.625 * (2**15):
        return bfloat16(0.0)
    x = max(x, 2 ** (-48))
    x = min(x, 2**15 * 1.50)
    E = np.floor(np.log2(x))
    return bfloat16(np.round(x * 2.0 ** (-E + 2)) * 2.0 ** (-2 + E))


def float_to_e6m2_int(x):
    if np.isnan(x) or np.isinf(x) or x <= 0:
        return np.uint8(0)
    E6 = np.floor(np.log2(x + 2 ** (-1000)))
    if not np.isfinite(E6):
        return np.uint8(0)
    M2 = int(np.clip(np.round(x * (2.0 ** (-E6 + 2)) - 4.0), 0, 3))
    E6_int = int(E6) + 48
    if E6_int < 0 or E6_int > 63:
        return np.uint8(0)
    return np.uint8(min(max(E6_int, 0), 63) * 4) | np.uint8(M2)


def _block_quant(ori, thd_eb, thd_ec):
    """Dequantize one 64-wide block: return (in_grp values, 4 scale bytes)."""
    tmpG = np.abs(ori).astype(bfloat16)
    V16 = np.array([np.max(tmpG[k * 4 : k * 4 + 4]) for k in range(16)])
    V8 = np.array([np.max(V16[k * 2 : k * 2 + 2]) for k in range(8)])
    Vmax = np.max(V8).astype(bfloat16)

    E6M2 = BF16_to_E6M2(Vmax * CONST_REC)
    E6M2_code = float_to_e6m2_int(E6M2)
    REC_E6M2 = E6M2_REC(E6M2)
    REC_bf16 = bfloat16(REC_E6M2) if np.isfinite(bfloat16(REC_E6M2)) else bfloat16(0.0)

    E1_8 = (V8.astype(bfloat16) * REC_bf16) >= thd_eb
    E1_8x2 = np.zeros(16)
    for k in range(8):
        E1_8x2[k * 2 : k * 2 + 2] = E1_8[k]
    E1_16 = (
        V16.astype(bfloat16) * REC_bf16 * 2.0 ** (-E1_8x2.astype(np.float64))
    ) >= thd_ec

    scale4 = np.zeros(4, dtype=np.uint8)
    scale4[0] = int(E6M2_code)
    E1_8_int = E1_8.astype(np.int32)
    scale4[1] = np.array([np.packbits(E1_8_int[:8][::-1])[0]], dtype=np.uint8)[0]
    E1_16_int = E1_16.astype(np.int32)
    E1_16_bit = np.array(
        [np.packbits(E1_16_int[:8][::-1])[0], np.packbits(E1_16_int[8:][::-1])[0]],
        dtype=np.uint8,
    )
    scale4[2] = E1_16_bit[0]
    scale4[3] = E1_16_bit[1]

    DE16 = E1_16 + E1_8x2
    DE64 = np.zeros(HIF4_BLOCK)
    for k in range(16):
        DE64[k * 4 : k * 4 + 4] = DE16[k]
    b_val = REC_bf16 if np.isfinite(REC_bf16) else bfloat16(0.0)
    power_term = 2.0 ** (-DE64.astype(np.float64))
    in_grp = (
        np.float64(np.asarray(bfloat16(ori), dtype=bfloat16))
        * np.float64(b_val)
        * power_term
    )
    return np.where(np.isfinite(in_grp), in_grp, 0.0), scale4


def quant_hif4(x_np, thd_eb, thd_ec):
    """Return (quant_y_np (S1P2 as float4_e1m2), scale_uint8 flat of len Mi*Ncnt*4)."""
    Mi, Ni = x_np.shape
    last = Ni
    Ncnt = int(np.ceil(last / HIF4_BLOCK))
    res = np.zeros((Mi, Ni))
    scale_uint8 = np.zeros(Mi * Ncnt * 4, dtype=np.uint8)
    ksi = 0

    for i in range(Mi):
        for j in range(Ncnt):
            ori = x_np[i, j * HIF4_BLOCK : j * HIF4_BLOCK + HIF4_BLOCK]
            if np.any(~np.isfinite(ori)):
                scale_uint8[ksi : ksi + 4] = 0
                ksi += 4
                continue
            in_grp, scale4 = _block_quant(ori, thd_eb, thd_ec)
            scale_uint8[ksi : ksi + 4] = scale4
            ksi += 4
            res[i, j * HIF4_BLOCK : j * HIF4_BLOCK + HIF4_BLOCK] = in_grp
    res = res.astype(float4_e1m2)
    return res, scale_uint8


def pack_y(quant_y_np):
    last = quant_y_np.shape[-1]
    out_last = last // 2
    flat_in = quant_y_np.view(np.uint8).reshape(-1, last)
    flat_out = np.zeros((flat_in.shape[0], out_last), dtype=np.uint8)
    for i in range(flat_in.shape[0]):
        for j in range(out_last):
            low = flat_in[i, 2 * j] & 0x0F
            high = (flat_in[i, 2 * j + 1] & 0x0F) << 4
            flat_out[i, j] = high | low
    return flat_out.reshape(list(quant_y_np.shape[:-1]) + [out_last])


def decode_nibbles(packed_u8):
    b = packed_u8.reshape(-1).astype(np.uint16)
    lo = (b & 0x0F).astype(np.int16)
    hi = ((b >> 4) & 0x0F).astype(np.int16)
    vals = np.empty(lo.size * 2, dtype=np.int16)
    vals[0::2] = lo
    vals[1::2] = hi
    return vals


RUNS = []


def add(
    name,
    shape,
    dtype,
    gamma_dtype,
    beta=False,
    output_rstd=False,
    scale_alg=0,
    seed=0,
    const=None,
):
    RUNS.append(
        dict(
            name=name,
            shape=shape,
            dtype=dtype,
            gamma_dtype=gamma_dtype,
            beta=beta,
            output_rstd=output_rstd,
            scale_alg=scale_alg,
            seed=seed,
            const=const,
        )
    )


add(
    "fp16_2d_beta_rstd_salg0",
    [8, 128],
    torch.float16,
    torch.float16,
    beta=True,
    output_rstd=True,
    scale_alg=0,
    seed=1,
)
add(
    "bf16_2d_nobeta_norstd_salg0",
    [8, 128],
    torch.bfloat16,
    torch.bfloat16,
    beta=False,
    output_rstd=False,
    scale_alg=0,
    seed=2,
)
add(
    "bf16_3d_gf32_beta_rstd_salg1",
    [2, 4, 256],
    torch.bfloat16,
    torch.float32,
    beta=True,
    output_rstd=True,
    scale_alg=1,
    seed=3,
)
add(
    "fp16_2d_beta_rstd_salg1",
    [4, 64],
    torch.float16,
    torch.float32,
    beta=True,
    output_rstd=True,
    scale_alg=1,
    seed=4,
)
add(
    "bf16_1d_rstd_salg0",
    [128],
    torch.bfloat16,
    torch.float32,
    beta=False,
    output_rstd=True,
    scale_alg=0,
    seed=5,
)
add(
    "const_ones_bf16_salg0",
    [2, 64],
    torch.bfloat16,
    torch.bfloat16,
    beta=False,
    output_rstd=True,
    scale_alg=0,
    const=1.0,
)
add(
    "fp16_3d_beta_norstd_salg1",
    [1, 3, 128],
    torch.float16,
    torch.float16,
    beta=True,
    output_rstd=False,
    scale_alg=1,
    seed=7,
)


def compare_golden(x, gamma, beta, eps, out_rstd, scale_alg, y, scale, rstd):
    y3_g, rstd_g = rms_golden(x, gamma, beta, eps)
    ok = True
    if out_rstd:
        assert rstd is not None, "rstd must be a tensor when output_rstd=True"
        rstd_cpu = rstd.cpu().float()
        rstd_ref = rstd_g.reshape(rstd_cpu.shape)
        rd = (rstd_cpu - rstd_ref).abs().max().item()
        tol = 1e-3 + 1e-2 * rstd_ref.abs().max().item()
        ok &= rd < tol
        print(f"    rstd      : max_diff={rd:.3e}  {'PASS' if rd < tol else 'FAIL'}")
    else:
        assert rstd is not None, (
            "rstd must be a defined tensor even when output_rstd=False"
        )
        rstd_cpu = rstd.cpu().float()
        exp_shape = list(y3_g.shape[:-1]) + [1]
        assert tuple(rstd_cpu.shape) == tuple(exp_shape), (
            f"rstd shape={tuple(rstd_cpu.shape)} expect={tuple(exp_shape)}"
        )
        assert rstd_cpu.abs().max().item() == 0.0, (
            "rstd must be all-zero when output_rstd=False"
        )
        print("    rstd      : all-zero placeholder tensor (output_rstd=False)  PASS")

    thd_eb, thd_ec = SCALE_ALG_THD[scale_alg]
    y3_np = y3_g.detach().float().numpy()
    last = y3_np.shape[-1]
    y3_2d = y3_np.reshape(-1, last)
    quant_y, scale_flat = quant_hif4(y3_2d, thd_eb, thd_ec)
    Ncnt = last // HIF4_BLOCK
    lead = y3_np.shape[:-1]
    scale_g = scale_flat.view(np.float32).reshape(list(lead) + [Ncnt, 1])
    s_npu_b = scale.cpu().reshape(-1).view(torch.uint8)
    s_gold_b = torch.from_numpy(scale_g.copy()).reshape(-1).view(torch.uint8)
    n = min(s_npu_b.numel(), s_gold_b.numel())
    sdiff = (s_npu_b[:n].int() ^ s_gold_b[:n].int()).abs().sum().item()
    s_match = sdiff == 0
    ok &= s_match
    rate = 1.0 - sdiff / n
    print(
        f"    scale     : byte_match_rate={rate:.6f}  "
        f"{'PASS(exact)' if s_match else 'FAIL(bytes differ)'}"
    )

    golden_packed = pack_y(quant_y.reshape(y3_np.shape))
    g_y = golden_packed.reshape(-1)
    n_y = y.cpu().reshape(-1).numpy().astype(np.uint8)
    gn = decode_nibbles(g_y)
    nn_ = decode_nibbles(n_y)
    diff = np.abs(nn_ - gn)
    exact = (diff == 0).mean()
    within1 = (diff <= 1).mean()
    print(f"    y         : nibble exact={exact:.4f}, within+/-1LSB={within1:.4f}")
    y_match = within1 >= 0.99
    ok &= y_match
    print(f"    [y]       : {'PASS' if y_match else 'FAIL(>1 LSB deviation)'}")
    return ok


def run_case(c):
    print(f"\n===== {c['name']} =====")
    torch.manual_seed(c["seed"])
    if c["const"] is not None:
        x = torch.full(c["shape"], c["const"], dtype=c["dtype"]).npu()
    else:
        x = torch.randn(c["shape"], dtype=c["dtype"]).npu()
    gamma = torch.ones(c["shape"][-1], dtype=c["gamma_dtype"]).npu()
    beta = (
        torch.zeros(c["shape"][-1], dtype=c["gamma_dtype"]).npu() if c["beta"] else None
    )
    eps = 1e-6

    y, scale, rstd = cann_ops_nn.rms_norm_hi_f4_quant(
        x,
        gamma,
        beta_optional=beta,
        epsilon=eps,
        output_rstd=c["output_rstd"],
        scale_alg=c["scale_alg"],
    )
    xd = str(x.dtype).replace("torch.", "")
    gd = str(c["gamma_dtype"]).replace("torch.", "")
    print(
        f"    x={tuple(x.shape)}{xd} gamma={gd} "
        f"beta={'Y' if c['beta'] else 'N'} out_rstd={'Y' if c['output_rstd'] else 'N'} scale_alg={c['scale_alg']}"
    )
    yd = str(y.dtype).replace("torch.", "")
    print(
        f"    y={tuple(y.shape)}{yd} scale={tuple(scale.shape)} rstd={tuple(rstd.shape)}"
    )
    return compare_golden(
        x, gamma, beta, eps, c["output_rstd"], c["scale_alg"], y, scale, rstd
    )


def whitebox_threshold_diff():
    print("\n===== whitebox: scale_alg=0 vs 1 must change Eb/Ec threshold =====")
    torch.manual_seed(99)
    x = torch.randn(2, 64, dtype=torch.bfloat16).npu()
    g = torch.ones(64, dtype=torch.bfloat16).npu()
    y0, s0, _ = cann_ops_nn.rms_norm_hi_f4_quant(x, g, output_rstd=False, scale_alg=0)
    y1, s1, _ = cann_ops_nn.rms_norm_hi_f4_quant(x, g, output_rstd=False, scale_alg=1)
    s0b = s0.cpu().reshape(-1).view(torch.uint8)
    s1b = s1.cpu().reshape(-1).view(torch.uint8)
    n = min(s0b.numel(), s1b.numel())
    ndiff = (s0b[:n].int() ^ s1b[:n].int()).abs().sum().item()
    print(f"    scale byte diff count between salg0 and salg1 = {ndiff}/{n}")
    assert ndiff > 0, "expected differing scale bytes across scale_alg"
    print("    PASS: scale_alg changes Eb/Ec encoding as expected")


def main():
    passed = 0
    failed = 0
    for c in RUNS:
        try:
            ok = run_case(c)
            print(f"    RESULT: {'PASS' if ok else 'FAIL'}")
            passed += ok
            failed += not ok
        except Exception as e:
            import traceback

            traceback.print_exc()
            failed += 1
            print(f"    RESULT: FAIL (exception {e})")
    whitebox_threshold_diff()
    print(f"\n===== TOTAL: {passed} passed, {failed} failed =====")
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
