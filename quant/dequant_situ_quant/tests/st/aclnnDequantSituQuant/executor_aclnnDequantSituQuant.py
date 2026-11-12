#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""ATK executor for aclnnDequantSituQuant (sample structure: BaseApi + AclnnBaseApi).

Math follows op_kernel/dequant_situ_quant.h:
    dequant : INT8  x*weightScale + bias
              INT32 x*weightScale*activationScale + bias (per expert, groupIndex)
              BF16/FP16 cast only (pre-dequantized)
    situ    : situ_a = beta*tanh(gate/beta)*sigmoid(gate); up = linearBeta*tanh(up/linearBeta) if linearBeta>0
    quant   : static  y = v / qs + qo
              dynamic scale = rowmax(|v[*smooth]|)/127 (zero row -> scale 0, compute with 1)
    cast    : fp32 -> int32(TRUNC) -> fp16(ROUND) -> int8(TRUNC)   (kernel CastFloatToInt8)
"""

import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

INT8_MIN = -128
INT8_MAX = 127


def _situ(dequant_out, beta, linear_beta, activate_left):
    last_dim = dequant_out.shape[-1]
    h = last_dim // 2
    if activate_left:
        gate = dequant_out[..., :h]
        up = dequant_out[..., h:]
    else:
        gate = dequant_out[..., h:]
        up = dequant_out[..., :h]
    beta_f = float(beta)
    situ_a = beta_f * torch.tanh(gate / beta_f) * torch.sigmoid(gate)
    lb = float(linear_beta)
    if lb > 0.0:
        up = lb * torch.tanh(up / lb)
    return situ_a * up


def _cast_to_int8(v):
    """Kernel CastFloatToInt8: fp32 -TRUNC-> int32 -ROUND-> fp16 -TRUNC-> int8."""
    v = torch.trunc(v)
    v = v.to(torch.int32)
    v = v.to(torch.float16)
    v = torch.trunc(v.float())
    v = torch.clamp(v, INT8_MIN, INT8_MAX)
    return v.to(torch.int8)


def _dequant_rows(x_rows, weight_scale, activation_scale, bias, group_idx=None):
    """Return fp32 dequantized rows for one group (None group -> whole tensor)."""
    is_int32 = x_rows.dtype == torch.int32
    xf = x_rows.to(torch.float32)
    if is_int32:
        if weight_scale is not None:
            xf = xf * weight_scale.to(torch.float32)
        if activation_scale is not None:
            xf = xf * activation_scale.to(torch.float32)
        if bias is not None:
            xf = xf + bias.to(torch.float32)
        return xf
    if x_rows.dtype == torch.int8:
        if weight_scale is not None:
            xf = xf * weight_scale.to(torch.float32)
        if bias is not None:
            xf = xf + bias.to(torch.float32)
        return xf
    return xf  # bf16/fp16: pre-dequantized


def _quant_rows(v, quant_scale, quant_offset, is_dynamic):
    if not is_dynamic:
        qs = (
            quant_scale.to(torch.float32)
            if quant_scale is not None
            else torch.tensor(1.0)
        )
        y = v / qs
        if quant_offset is not None:
            y = y + quant_offset.to(torch.float32)
        return y, torch.zeros(v.shape[:-1], dtype=torch.float32)
    if quant_scale is not None:
        v = v * quant_scale.to(torch.float32)
    absmax = v.abs().amax(dim=-1, keepdim=True)
    scale = (absmax / 127.0).to(torch.float32)
    eff = torch.where(scale == 0.0, torch.ones_like(scale), scale)
    return v / eff, scale.squeeze(-1)


def _golden(
    x,
    weight_scale=None,
    activation_scale=None,
    bias=None,
    quant_scale=None,
    quant_offset=None,
    group_index=None,
    beta=4.0,
    linear_beta=25.0,
    activate_left=True,
    quant_type="dynamic",
):
    is_dynamic = str(quant_type).strip().lower() != "static"
    x = x.detach().cpu()
    # bf16 tensors cannot be converted by numpy path; ATK passes torch tensors here
    if x.dtype in (torch.bfloat16, torch.float16):
        x = x.to(torch.float32)

    def _p(t):
        if t is None:
            return None
        t = t.detach().cpu()
        return t.to(torch.float32) if torch.is_floating_point(t) else t

    ws = _p(weight_scale)
    as_ = _p(activation_scale)
    # contract: activation_scale is [1] (shared scalar) or [M]/[M,1] (per row).
    # A scalar must stay unscaled/broadcast: slicing it per group would turn the
    # later groups into an empty [0,1] tensor and break broadcasting.
    as_is_scalar = as_ is not None and as_.numel() == 1
    if as_ is not None and not as_is_scalar:
        as_ = as_.reshape(-1, 1)
    b = _p(bias)
    qs = _p(quant_scale)
    qo = _p(quant_offset)
    gi = (
        None
        if group_index is None
        else group_index.detach().cpu().to(torch.int64).reshape(-1)
    )

    raw_shape = tuple(x.shape)
    if x.dim() > 2:
        x2 = x.reshape(-1, x.shape[-1])
    else:
        x2 = x
    rows = x2.shape[0]

    def _group_param(p, e):
        if p is None:
            return None
        if p.dim() == 2:
            return p[e] if p.shape[0] > 1 else p[0]
        return p

    ys = []
    ss = []
    if gi is None:
        y, s = _process_block(
            x2, ws, as_, b, qs, qo, beta, linear_beta, activate_left, is_dynamic
        )
        ys.append(y)
        ss.append(s)
    else:
        offset = 0
        has_tail = False
        for e in range(gi.numel()):
            cnt = int(gi[e].item())
            if cnt <= 0:
                continue
            end = min(offset + cnt, rows)
            part = x2[offset:end]
            as_part = (
                None if as_ is None else (as_ if as_is_scalar else as_[offset:end])
            )
            y, s = _process_block(
                part,
                _group_param(ws, e),
                as_part,
                _group_param(b, e),
                _group_param(qs, e),
                _group_param(qo, e),
                beta,
                linear_beta,
                activate_left,
                is_dynamic,
            )
            ys.append(y)
            ss.append(s)
            offset = end
        if offset < rows:
            has_tail = True
        if has_tail:
            ys.append(torch.zeros((rows - offset, x2.shape[-1] // 2), dtype=torch.int8))
            ss.append(torch.zeros(rows - offset, dtype=torch.float32))
    y = torch.cat(ys, dim=0)
    s = torch.cat(ss, dim=0)
    out_shape = raw_shape[:-1] + (x2.shape[-1] // 2,)
    y = y.reshape(out_shape)
    s = s.reshape(raw_shape[:-1])
    return y, s


def _process_block(
    x2, ws, as_, b, qs, qo, beta, linear_beta, activate_left, is_dynamic
):
    dq = _dequant_rows(x2, ws, as_, b)
    situ = _situ(dq, beta, linear_beta, activate_left)
    y, s = _quant_rows(situ, qs, qo, is_dynamic)
    return _cast_to_int8(y), s


# Optional tensor slots of aclnnDequantSituQuant, in aclnn/op_def positional order.
# The ATK json keeps these slots for every case (fixed input order); a slot marked
# absent by the dataset is not materialised (ATK passes None for range "default").
_OPTIONAL_SLOTS = (
    "weightScaleOptional",
    "activationScaleOptional",
    "biasOptional",
    "quantScaleOptional",
    "quantOffsetOptional",
    "groupIndexOptional",
)


def _forbidden_optional_slots(x_dtype, quant_mode):
    """Optional inputs that MUST be absent for the current x dtype / quant mode.

    Hard host constraints (op_host/dequant_situ_quant_tiling.cpp):
      INT8        : activation_scale / group_index absent; dynamic -> quant_offset absent
      INT32       : quant_scale / quant_offset absent
      BF16 / FP16 : all optional inputs except group_index absent
    This is defence-in-depth on top of the dataset encoding: even if an ATK
    version/JSON change materialises one of these tensors, they are forced null
    so the aclnn/tiling validation always sees the contract-compliant input set.
    """
    text = str(x_dtype).lower()
    is_static = str(quant_mode).strip().lower() == "static"
    if "int8" in text:
        absent = {"activationScaleOptional", "groupIndexOptional"}
        if not is_static:
            absent.add("quantOffsetOptional")
        return absent
    if "int32" in text:
        return {"quantScaleOptional", "quantOffsetOptional"}
    return {
        "weightScaleOptional",
        "activationScaleOptional",
        "biasOptional",
        "quantScaleOptional",
        "quantOffsetOptional",
    }


@register("ascend_method_dequant_situ_quant")
class MethodDequantSituQuant(BaseApi):
    """CPU golden + NPU (cann_ops_nn) executor, mirrors the dequant_swiglu_quant sample."""

    def __init__(self, task_result):
        super().__init__(task_result)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        kw = input_data.kwargs
        x = kw["x"]
        quant_mode = str(kw.get("quantModeOptional", "dynamic"))
        forbidden = _forbidden_optional_slots(x.dtype, quant_mode)

        def _pick(name):
            # Forbidden slots are always null; the rest keep the dataset value.
            return None if name in forbidden else kw.get(name)

        args = dict(
            weight_scale=_pick("weightScaleOptional"),
            activation_scale=_pick("activationScaleOptional"),
            bias=_pick("biasOptional"),
            quant_scale=_pick("quantScaleOptional"),
            quant_offset=_pick("quantOffsetOptional"),
            group_index=_pick("groupIndexOptional"),
            beta=kw.get("beta", 4.0),
            linear_beta=kw.get("linearBeta", 25.0),
            activate_left=kw.get("activateLeft", True),
            quant_type=quant_mode,
        )
        if self.device in ("cpu", "gpu"):
            return _golden(x, **args)
        import cann_ops_nn

        torch.npu.synchronize()
        y, s = cann_ops_nn.dequant_situ_quant(
            x.npu(),
            weight_scale=None
            if args["weight_scale"] is None
            else args["weight_scale"].npu(),
            activation_scale=None
            if args["activation_scale"] is None
            else args["activation_scale"].npu(),
            bias=None if args["bias"] is None else args["bias"].npu(),
            quant_scale=None
            if args["quant_scale"] is None
            else args["quant_scale"].npu(),
            quant_offset=None
            if args["quant_offset"] is None
            else args["quant_offset"].npu(),
            group_index=None
            if args["group_index"] is None
            else args["group_index"].npu(),
            beta=float(args["beta"]),
            linear_beta=float(args["linear_beta"]),
            activate_left=bool(args["activate_left"]),
            quant_type=str(args["quant_type"]),
        )
        torch.npu.synchronize()
        return y, s


@register("aclnn_dequant_situ_quant")
class AclnnDequantSituQuantApi(AclnnBaseApi):
    """Direct aclnn C-API executor (PyAclnn), mirrors the dequant_swiglu_quant sample."""

    def init_by_input_data(self, input_data: InputDataset):
        import ctypes

        from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor

        null_tensor_ptr = ctypes.cast(ctypes.c_void_p(None), ctypes.POINTER(AclTensor))
        kw = input_data.kwargs
        self.quant_mode = kw.get("quantModeOptional", "dynamic")
        forbidden = _forbidden_optional_slots(kw["x"].dtype, self.quant_mode)
        input_args, output_packages = super().init_by_input_data(input_data)
        # The json keeps a fixed 7-tensor argument layout, so the six optional
        # slots map to input_args[1..6] in _OPTIONAL_SLOTS order. Null a slot when
        # it is contract-forbidden for this dtype/mode or was not materialised.
        for idx, name in enumerate(_OPTIONAL_SLOTS, start=1):
            if name in forbidden or kw.get(name) is None:
                input_args[idx] = null_tensor_ptr
        return input_args, output_packages

    def after_call(self, output_packages):
        outputs = [self.acl_tensor_to_torch(p) for p in output_packages]
        if str(self.quant_mode).strip().lower() == "static":
            outputs[1] = torch.zeros(outputs[0].shape[:-1], dtype=torch.float32)
        return outputs[0], outputs[1]
