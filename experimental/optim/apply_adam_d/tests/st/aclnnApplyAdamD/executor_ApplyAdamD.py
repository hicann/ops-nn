#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.
"""
ATK executors for ApplyAdamD (experimental/optim/apply_adam_d).

ApplyAdamD is a graph/TF-style fused Adam single-step op. This ST keeps a kernel-backend
path to validate the deployed op host tiling and AI Core kernel independently of the aclnn
example. It pairs the ATK `kernel` backend (device-under-test) with the `cpu` backend (golden):

  * golden  (api_type = "cpu_apply_adam_d")    : hand-written fp32 fused Adam golden (BaseApi).
  * device  (kernel_api_type = "kernel_apply_adam_d"): a KernelBaseApi whose __call__ launches the
    DEPLOYED ApplyAdamD on the NPU via aclop_runner.py (aclopCompileAndExecuteV2 single-op), run as
    an isolated subprocess so its ACL context never collides with ATK / torch_npu.

Both executors:
  * read the 10 tensor inputs (var,m,v,beta1_power,beta2_power,lr,beta1,beta2,epsilon,grad) from
    input_data.kwargs and the bool attr use_nesterov;
  * do NOT mutate the generated inputs (no init_by_input_data override) — both backends must consume
    byte-identical generated data. Validity of the fused math is guaranteed by the case json itself:
    the 6 Adam scalar params are fixed valid constants, and v (2nd moment) is generated from a
    strictly-positive range so v_new = beta2*v + (1-beta2)*grad^2 >= 0 and sqrt(v_new) is real.
  * return exactly three outputs (var, m, v) in op-output order so ATK compares each separately.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.kernel_base_api import KernelBaseApi

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNNER = os.path.join(_HERE, "aclop_runner.py")

_TORCH_TO_RUNNER = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}


def _nesterov(input_data):
    val = input_data.kwargs.get("use_nesterov", False)
    if isinstance(val, torch.Tensor):
        return bool(val.flatten()[0].item())
    return bool(val)


def _golden_fp32(k, nesterov):
    """Adam single-step in fp32 (matches arch kernel internal fp32 compute). Returns fp32 tensors."""
    var = k["var"].float()
    m = k["m"].float()
    v = k["v"].float()
    grad = k["grad"].float()
    beta1_power = k["beta1_power"].float().flatten()[0]
    beta2_power = k["beta2_power"].float().flatten()[0]
    lr = k["lr"].float().flatten()[0]
    beta1 = k["beta1"].float().flatten()[0]
    beta2 = k["beta2"].float().flatten()[0]
    epsilon = k["epsilon"].float().flatten()[0]

    lr_t = lr * torch.sqrt(1.0 - beta2_power) / (1.0 - beta1_power)
    m_out = m + (1.0 - beta1) * (grad - m)
    v_out = v + (1.0 - beta2) * (grad * grad - v)
    numerator = (m_out * beta1 + (1.0 - beta1) * grad) if nesterov else m_out
    var_out = var - lr_t * numerator / (epsilon + torch.sqrt(v_out))
    return var_out, m_out, v_out


@register("cpu_apply_adam_d")
class ApplyAdamDGolden(BaseApi):
    def __call__(self, input_data, with_output=False):
        nesterov = _nesterov(input_data)
        var_out, m_out, v_out = _golden_fp32(input_data.kwargs, nesterov)
        dt = input_data.kwargs["var"].dtype
        # round the fp32 reference back to the op dtype: the ideal low-precision result
        return var_out.to(dt).contiguous(), m_out.to(dt).contiguous(), v_out.to(dt).contiguous()


def _to_raw(t):
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().tobytes()
    return t.numpy().tobytes()


def _from_raw(raw, dtype, shape):
    if dtype == torch.bfloat16:
        arr = np.frombuffer(raw, dtype=np.int16).copy()
        return torch.from_numpy(arr).view(torch.bfloat16).reshape(shape)
    npdt = {torch.float16: np.float16, torch.float32: np.float32}[dtype]
    arr = np.frombuffer(raw, dtype=npdt).copy()
    return torch.from_numpy(arr).reshape(shape)


@register("kernel_apply_adam_d")
class ApplyAdamDKernel(KernelBaseApi):
    def __call__(self, input_data, with_output=False):
        import json
        k = input_data.kwargs
        dt = k["var"].dtype
        runner_dt = _TORCH_TO_RUNNER[dt]
        nesterov = _nesterov(input_data)
        var_shape = list(k["var"].shape)

        names = ["var", "m", "v", "beta1_power", "beta2_power", "lr", "beta1", "beta2", "epsilon", "grad"]
        with tempfile.TemporaryDirectory(prefix="aad_atk_") as d:
            shapes = {nm: list(k[nm].shape) for nm in names}
            meta = {"dtype": runner_dt, "nesterov": 1 if nesterov else 0, "shapes": shapes}
            with open(os.path.join(d, "meta.json"), "w") as fh:
                json.dump(meta, fh)
            for nm in names:
                with open(os.path.join(d, nm + ".bin"), "wb") as fh:
                    fh.write(_to_raw(k[nm]))
            # subprocess: isolated ACL context (clean aclInit/aclFinalize) for the deployed op
            env = dict(os.environ)
            res = subprocess.run([sys.executable, _RUNNER, "--io", d],
                                 capture_output=True, text=True, env=env)
            if res.returncode != 0:
                raise RuntimeError("aclop_runner failed (rc=%d)\nSTDOUT:%s\nSTDERR:%s"
                                   % (res.returncode, res.stdout, res.stderr))
            outs = []
            for nm in ("var", "m", "v"):
                with open(os.path.join(d, nm + "_out.bin"), "rb") as fh:
                    outs.append(_from_raw(fh.read(), dt, var_shape))
        return tuple(o.contiguous() for o in outs)
