# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Public Torch interface adapter for the single CPU math reference.

Use --plugin torch_golden.py --pc=1 --golden-mode Enable with the Torch ST CSV.
It covers two input dtypes, four output formats, and both gradient flags.
The adapter maps public dtype constants, packs FP4, exposes E8M0 as uint8,
and returns an empty rstd when gradients are disabled.
"""

import importlib.util
from pathlib import Path

import numpy as np

_source = Path(__file__).with_name("golden.py")
if not _source.is_file():
    raise ModuleNotFoundError(
        "Public reference requires sibling golden.py", name="golden"
    )
_module_spec = importlib.util.spec_from_file_location("rms_cpu_reference", _source)
_reference = importlib.util.module_from_spec(_module_spec)
_module_spec.loader.exec_module(_reference)
_TORCH_DST = {291: 35, 292: 36, 296: 40, 297: 41}
__spec__ = {"torch_npu.npu_rms_norm_dynamic_mx_quant": "TorchPublicSpec"}


def _public_outputs(outputs, dst_type, requires_grad):
    y, scales, rstd = outputs
    if _TORCH_DST[int(dst_type)] in (40, 41):
        codes = y.view(np.uint8)
        y = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return (
        y,
        scales.view(np.uint8),
        rstd if requires_grad else np.empty((0,), dtype=np.float32),
    )


def torch_input(
    x,
    gamma,
    *,
    beta=None,
    epsilon=1e-6,
    scale_alg=0,
    round_mode="rint",
    dst_type=296,
    requires_grad=False,
    **kwargs,
):
    # TTK's requires_grad test metadata is not forwarded to the public API.
    x.requires_grad_(bool(requires_grad))


def torch_golden(
    x,
    gamma,
    *,
    beta=None,
    epsilon=1e-6,
    scale_alg=0,
    round_mode="rint",
    dst_type=296,
    requires_grad=False,
    **kwargs,
):
    assert bool(x.requires_grad) == bool(requires_grad), (
        "TTK lost the input gradient flag"
    )
    outputs = _reference._reference(
        x,
        gamma,
        beta,
        epsilon,
        scale_alg,
        round_mode,
        _TORCH_DST[int(dst_type)],
        bool(requires_grad),
    )
    return _public_outputs(outputs, dst_type, requires_grad)


class TorchPublicIndependentReference:
    """Independent FP64 RMS/affine and table MX encoder with public output storage."""

    def __call__(
        self,
        x,
        gamma,
        *,
        beta=None,
        epsilon=1e-6,
        scale_alg=0,
        round_mode="rint",
        dst_type=296,
        requires_grad=False,
        **kwargs,
    ):
        outputs = _reference.TorchIndependentReference()(
            x,
            gamma,
            beta=beta,
            epsilon=epsilon,
            scale_alg=scale_alg,
            round_mode=round_mode,
            dst_type=_TORCH_DST[int(dst_type)],
            output_rstd=bool(requires_grad),
        )
        return _public_outputs(outputs, dst_type, bool(requires_grad))


class TorchPublicSpec:
    golden = torch_golden
    third_party = {"torch": TorchPublicIndependentReference}
    customize_inputs = torch_input
    tolerance = dict(
        _reference.RmsNormDynamicMxQuantTestSpec.tolerance,
        uint8={"standard": "binary_equal"},
    )
