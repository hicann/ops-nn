#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.
"""
Standalone ACL single-op runner for ApplyAdamD (aclop dynamic launch).

This runner is the device-under-test launch primitive used by the ATK `kernel`-backend executor
(executor_ApplyAdamD.py): it runs the DEPLOYED ApplyAdamD operator end-to-end (host tiling +
dispatch + AI Core kernel) via
    aclopCompileAndExecuteV2("ApplyAdamD", 10-in, 3-out, attrs, ACL_ENGINE_SYS, ACL_COMPILE_SYS)
through ctypes (libascendcl.so + libacl_op_compiler.so). Which implementation is dispatched
(experimental custom_nn vs builtin) is governed by ASCEND_CUSTOM_OPP_PATH at the OPP level.

It runs in its own ACL context (clean aclInit/aclFinalize), so the ATK executor invokes it as a
subprocess to fully isolate the ACL context from ATK / torch_npu.

Two modes:
  * --io <dir>      ATK mode: read <dir>/meta.json + <name>.bin (raw little-endian bytes), run the
                    op, write var_out.bin/m_out.bin/v_out.bin into <dir>. No interpretation of values.
  * --selftest ...  validate the launch glue independently: generate data, run on NPU, compare to an
                    fp32 fused CPU golden with the spec per-dtype tolerance.

Inputs order (matches op_host/apply_adam_d_def.cpp):
  var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad   (all same dtype)
  var/m/v/grad: shape [n] ; the 6 scalar params: shape [1].
Outputs (in-place / ref): var, m, v.
"""
import argparse
import ctypes
import json
import math
import os
import struct
import sys

# ---- ACL enums ----
ACL_FLOAT = 0
ACL_FLOAT16 = 1
ACL_BF16 = 27
ACL_FORMAT_ND = 2
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_ENGINE_SYS = 0
ACL_COMPILE_SYS = 0
ACL_SUCCESS = 0

DTYPE_MAP = {"fp32": (ACL_FLOAT, 4), "fp16": (ACL_FLOAT16, 2), "bf16": (ACL_BF16, 2)}


def _load_libs():
    acl = ctypes.CDLL("libascendcl.so", mode=ctypes.RTLD_GLOBAL)
    occ = ctypes.CDLL("libacl_op_compiler.so", mode=ctypes.RTLD_GLOBAL)
    acl.aclCreateTensorDesc.restype = ctypes.c_void_p
    acl.aclCreateTensorDesc.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int64), ctypes.c_int]
    acl.aclCreateDataBuffer.restype = ctypes.c_void_p
    acl.aclCreateDataBuffer.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    acl.aclopCreateAttr.restype = ctypes.c_void_p
    acl.aclopSetAttrBool.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint8]
    acl.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
    acl.aclrtMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    acl.aclrtCreateStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    acl.aclrtSynchronizeStream.argtypes = [ctypes.c_void_p]
    acl.aclrtSetDevice.argtypes = [ctypes.c_int]
    acl.aclrtResetDevice.argtypes = [ctypes.c_int]
    occ.aclopCompileAndExecuteV2.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p,
    ]
    occ.aclopCompileAndExecuteV2.restype = ctypes.c_int
    return acl, occ


def _ck(acl, ret, what):
    if ret != ACL_SUCCESS:
        msg = b""
        try:
            acl.aclGetRecentErrMsg.restype = ctypes.c_char_p
            msg = acl.aclGetRecentErrMsg() or b""
        except Exception:
            pass
        raise RuntimeError("ACL call failed (%s) ret=%d msg=%s" % (what, ret, msg.decode("utf-8", "ignore")))


def _numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def run_op(dtype, shapes, nesterov, in_bytes, acl=None, occ=None, device=0, own_ctx=True):
    """shapes: dict name->shape list for the 10 inputs (rank-0 -> [], empty -> [0]).
    in_bytes: dict name->raw bytes. Returns dict of raw bytes for var/m/v outputs."""
    acl_dt, esz = DTYPE_MAP[dtype]
    if acl is None:
        acl, occ = _load_libs()
    if own_ctx:
        acl.aclInit(None)  # may return repeat-init if already inited; ignore
        _ck(acl, acl.aclrtSetDevice(device), "aclrtSetDevice")
    stream = ctypes.c_void_p()
    _ck(acl, acl.aclrtCreateStream(ctypes.byref(stream)), "aclrtCreateStream")

    nelem = {nm: _numel(shapes[nm]) for nm in shapes}

    dev = {}
    for nm, raw in in_bytes.items():
        nb = nelem[nm] * esz
        assert len(raw) == nb, "%s expected %d bytes got %d" % (nm, nb, len(raw))
        alloc = max(nb, esz)  # malloc(0) is invalid; keep a valid pointer for empty tensors
        p = ctypes.c_void_p()
        _ck(acl, acl.aclrtMalloc(ctypes.byref(p), alloc, ACL_MEM_MALLOC_HUGE_FIRST), "malloc " + nm)
        if nb > 0:
            buf = (ctypes.c_char * nb).from_buffer_copy(raw)
            _ck(acl, acl.aclrtMemcpy(p, nb, buf, nb, ACL_MEMCPY_HOST_TO_DEVICE), "H2D " + nm)
        dev[nm] = (p, nb)

    def mk_desc(shape):
        nd = len(shape)
        dims = (ctypes.c_int64 * max(nd, 1))(*(shape if nd else [0]))
        d = acl.aclCreateTensorDesc(acl_dt, nd, dims, ACL_FORMAT_ND)
        if not d:
            raise RuntimeError("aclCreateTensorDesc returned null")
        return d

    # input order MUST match op def
    in_order = ["var", "m", "v", "beta1_power", "beta2_power", "lr", "beta1", "beta2", "epsilon", "grad"]
    in_desc = (ctypes.c_void_p * 10)()
    in_buf = (ctypes.c_void_p * 10)()
    for i, nm in enumerate(in_order):
        in_desc[i] = mk_desc(shapes[nm])
        in_buf[i] = acl.aclCreateDataBuffer(dev[nm][0], dev[nm][1])

    out_order = ["var", "m", "v"]
    out_desc = (ctypes.c_void_p * 3)()
    out_buf = (ctypes.c_void_p * 3)()
    for i, nm in enumerate(out_order):
        out_desc[i] = mk_desc(shapes[nm])
        out_buf[i] = acl.aclCreateDataBuffer(dev[nm][0], dev[nm][1])  # in-place ref: reuse addr

    attr = acl.aclopCreateAttr()
    _ck(acl, acl.aclopSetAttrBool(attr, b"use_locking", 0), "set use_locking")
    _ck(acl, acl.aclopSetAttrBool(attr, b"use_nesterov", 1 if nesterov else 0), "set use_nesterov")

    ret = occ.aclopCompileAndExecuteV2(b"ApplyAdamD", 10, in_desc, in_buf, 3, out_desc, out_buf,
                                       attr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, None, stream)
    _ck(acl, ret, "aclopCompileAndExecuteV2")
    _ck(acl, acl.aclrtSynchronizeStream(stream), "sync")

    out = {}
    for nm in out_order:
        p, nb = dev[nm]
        if nb == 0:
            out[nm] = b""
            continue
        host = (ctypes.c_char * nb)()
        _ck(acl, acl.aclrtMemcpy(host, nb, p, nb, ACL_MEMCPY_DEVICE_TO_HOST), "D2H " + nm)
        out[nm] = bytes(host)

    # cleanup
    for nm in in_order:
        pass
    for arr in (in_buf, out_buf):
        for i in range(len(arr)):
            if arr[i]:
                acl.aclDestroyDataBuffer(ctypes.c_void_p(arr[i]))
    for arr in (in_desc, out_desc):
        for i in range(len(arr)):
            if arr[i]:
                acl.aclDestroyTensorDesc(ctypes.c_void_p(arr[i]))
    acl.aclopDestroyAttr(ctypes.c_void_p(attr))
    for nm in dev:
        acl.aclrtFree(dev[nm][0])
    acl.aclrtDestroyStream(stream)
    if own_ctx:
        acl.aclrtResetDevice(device)
        acl.aclFinalize()
    return out


# ---------- float codec (selftest only) ----------
def f2half(f):
    import numpy as np
    return np.float16(f).tobytes()


def half2f(b):
    import numpy as np
    return float(np.frombuffer(b, dtype=np.float16)[0])


def f2bf16(f):
    x = struct.unpack("<I", struct.pack("<f", f))[0]
    # round to nearest even
    rounding = 0x7FFF + ((x >> 16) & 1)
    x = (x + rounding) >> 16
    return struct.pack("<H", x & 0xFFFF)


def bf162f(b):
    h = struct.unpack("<H", b)[0]
    return struct.unpack("<f", struct.pack("<I", h << 16))[0]


def encode(dtype, vals):
    if dtype == "fp32":
        return b"".join(struct.pack("<f", v) for v in vals)
    if dtype == "fp16":
        return b"".join(f2half(v) for v in vals)
    return b"".join(f2bf16(v) for v in vals)


def decode(dtype, raw):
    esz = DTYPE_MAP[dtype][1]
    out = []
    for i in range(len(raw) // esz):
        chunk = raw[i * esz:(i + 1) * esz]
        if dtype == "fp32":
            out.append(struct.unpack("<f", chunk)[0])
        elif dtype == "fp16":
            out.append(half2f(chunk))
        else:
            out.append(bf162f(chunk))
    return out


def selftest(dtype, n, nesterov):
    B1P, B2P, LR, B1, B2, EPS = 0.9, 0.999, 0.001, 0.9, 0.999, 1e-7
    var = [0.5 + 0.001 * (i % 97) for i in range(n)]
    m = [0.1 + 0.0005 * (i % 53) for i in range(n)]
    v = [0.2 + 0.0003 * (i % 31) for i in range(n)]
    grad = [-0.05 + 0.0007 * (i % 41) for i in range(n)]
    in_bytes = {
        "var": encode(dtype, var), "m": encode(dtype, m), "v": encode(dtype, v),
        "grad": encode(dtype, grad),
        "beta1_power": encode(dtype, [B1P]), "beta2_power": encode(dtype, [B2P]),
        "lr": encode(dtype, [LR]), "beta1": encode(dtype, [B1]),
        "beta2": encode(dtype, [B2]), "epsilon": encode(dtype, [EPS]),
    }
    shapes = {"var": [n], "m": [n], "v": [n], "grad": [n],
              "beta1_power": [1], "beta2_power": [1], "lr": [1],
              "beta1": [1], "beta2": [1], "epsilon": [1]}
    out = run_op(dtype, shapes, nesterov, in_bytes)
    got_var, got_m, got_v = decode(dtype, out["var"]), decode(dtype, out["m"]), decode(dtype, out["v"])
    lr_t = LR * math.sqrt(1.0 - B2P) / (1.0 - B1P)
    rtol = {"fp32": 1.220703125e-04, "fp16": 9.765625e-04, "bf16": 7.8125e-03}[dtype]
    atol = rtol
    bad = 0
    mx = 0.0
    for i in range(n):
        mnew = B1 * m[i] + (1 - B1) * grad[i]
        vnew = B2 * v[i] + (1 - B2) * (grad[i] * grad[i])
        num = (mnew * B1 + (1 - B1) * grad[i]) if nesterov else mnew
        gvar = var[i] - lr_t * num / (EPS + math.sqrt(vnew))
        for got, gold in ((got_var[i], gvar), (got_m[i], mnew), (got_v[i], vnew)):
            ad = abs(got - gold)
            rel = ad / (abs(gold) + 1e-7)
            mx = max(mx, rel)
            if ad > atol + rtol * abs(gold):
                bad += 1
    print("[SELFTEST] dtype=%s n=%d nesterov=%d maxRel=%.3e rtol=%.3e bad=%d -> %s"
          % (dtype, n, nesterov, mx, rtol, bad, "OK" if bad == 0 else "FAIL"))
    return 0 if bad == 0 else 3


def atk_mode(io_dir):
    meta = json.load(open(os.path.join(io_dir, "meta.json")))
    dtype = meta["dtype"]
    nesterov = int(meta.get("nesterov", 0))
    shapes = meta["shapes"]
    in_order = ["var", "m", "v", "beta1_power", "beta2_power", "lr", "beta1", "beta2", "epsilon", "grad"]
    in_bytes = {}
    for nm in in_order:
        with open(os.path.join(io_dir, nm + ".bin"), "rb") as fh:
            in_bytes[nm] = fh.read()
    out = run_op(dtype, shapes, nesterov, in_bytes)
    for nm in ("var", "m", "v"):
        with open(os.path.join(io_dir, nm + "_out.bin"), "wb") as fh:
            fh.write(out[nm])
    print("[ATK_MODE] ok dtype=%s nesterov=%d shape=%s" % (dtype, nesterov, shapes["var"]))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--io", help="ATK io dir (meta.json + <name>.bin)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--nesterov", type=int, default=0)
    args = ap.parse_args()
    if args.io:
        return atk_mode(args.io)
    if args.selftest:
        return selftest(args.dtype, args.n, args.nesterov)
    ap.error("specify --io <dir> or --selftest")


if __name__ == "__main__":
    sys.exit(main())
