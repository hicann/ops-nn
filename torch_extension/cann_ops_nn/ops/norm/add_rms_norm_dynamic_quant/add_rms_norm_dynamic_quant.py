# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

from typing import Optional, Tuple

import torch
from torch.library import impl

from cann_ops_nn.op_builder import OpBuilder, get_as_library


class AddRmsNormDynamicQuantOpBuilder(OpBuilder):
    """
    AddRmsNormDynamicMxQuant 算子构建器

    基于 aclnnAddRmsNormDynamicMxQuantV2 API 实现，融合 Add + RMS Normalization + MX 动态量化。
    当 x3 提供时计算 x = (x3 + x1) + x2，否则 x = x1 + x2。
    始终走 V2 接口，x3 为 None 时内部传 nullptr，行为与 V1 一致。
    """

    def __init__(self):
        super().__init__("add_rms_norm_dynamic_quant")

    def sources(self) -> list:
        return [self.resolve_source("add_rms_norm_dynamic_quant.cpp")]

    def schema(self) -> str:
        return (
            "add_rms_norm_dynamic_quant("
            "Tensor x1, "
            "Tensor x2, "
            "Tensor gamma, "
            "Tensor? beta=None, "
            "Tensor? x3=None, "
            "float epsilon=1e-6, "
            "int scale_alg=0, "
            'str round_mode="rint", '
            "int dst_type=40, "
            "bool output_rstd=False"
            ") -> (Tensor y, Tensor x, Tensor mxscale, Tensor rstd)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def add_rms_norm_dynamic_quant_meta(
            x1: torch.Tensor,
            x2: torch.Tensor,
            gamma: torch.Tensor,
            beta: Optional[torch.Tensor] = None,
            x3: Optional[torch.Tensor] = None,
            epsilon: float = 1e-6,
            scale_alg: int = 0,
            round_mode: str = "rint",
            dst_type: int = 40,
            output_rstd: bool = False,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            # NOTE: these value-domain checks mirror the TORCH_CHECKs in
            # csrc/add_rms_norm_dynamic_quant.cpp and must stay in lockstep with
            # them (same order, same accept/reject semantics). The Meta kernel is
            # the device-independent, testable copy; the csrc copy only fires on
            # NPU hardware because it sits behind the device checks.
            torch._check(
                x1.dim() >= 1 and x1.dim() <= 7,
                lambda: f"x1 must be 1-7 dimensional, but got {x1.dim()}-d",
            )
            torch._check(
                x1.shape == x2.shape,
                lambda: f"x1 and x2 must have the same shape, got {x1.shape} vs {x2.shape}",
            )
            torch._check(
                gamma.dim() == 1 and gamma.shape[0] == x1.shape[-1],
                lambda: f"gamma must be 1D with size matching x1 last dim ({x1.shape[-1]})",
            )
            torch._check(
                x1.dtype in (torch.float16, torch.bfloat16),
                lambda: f"x1 dtype must be float16 or bfloat16, but got {x1.dtype}",
            )
            torch._check(
                x2.dtype == x1.dtype,
                lambda: f"x2 dtype must match x1 dtype ({x1.dtype}), but got {x2.dtype}",
            )
            torch._check(
                gamma.dtype == x1.dtype or gamma.dtype == torch.float32,
                lambda: (
                    f"gamma dtype must match x1 dtype ({x1.dtype}) or be float32, "
                    f"but got {gamma.dtype}"
                ),
            )
            if beta is not None:
                torch._check(
                    beta.shape == gamma.shape,
                    lambda: f"beta must have the same shape as gamma, got {beta.shape} vs {gamma.shape}",
                )
                torch._check(
                    beta.dtype == gamma.dtype,
                    lambda: f"beta dtype must match gamma dtype, got {beta.dtype} vs {gamma.dtype}",
                )
            if x3 is not None:
                torch._check(
                    x3.shape == x1.shape,
                    lambda: f"x3 must have the same shape as x1, got {x3.shape} vs {x1.shape}",
                )
                torch._check(
                    x3.dtype == x1.dtype,
                    lambda: f"x3 dtype must match x1 dtype, got {x3.dtype} vs {x1.dtype}",
                )

            torch._check(
                dst_type in (35, 36, 40, 41),
                lambda: (
                    f"invalid dst_type {dst_type}, expected 35/36/40/41; if a torch.dtype "
                    f"was passed, the dispatcher converted it to its ScalarType ordinal"
                ),
            )
            torch._check(
                scale_alg in (0, 1),
                lambda: f"scale_alg must be 0 (OCP) or 1 (cuBLAS, FP8 only), but got {scale_alg}",
            )
            is_fp4 = dst_type in (40, 41)
            if is_fp4:
                torch._check(
                    x1.shape[-1] % 2 == 0,
                    lambda: f"x1 last dim must be even for FP4 dst_type, got {x1.shape[-1]}",
                )
                torch._check(
                    scale_alg == 0,
                    lambda: f"scale_alg must be 0 (OCP) for FP4 dst_type, got {scale_alg}",
                )
                torch._check(
                    round_mode in ("rint", "floor", "round"),
                    lambda: (
                        f"round_mode must be one of rint/floor/round for FP4 dst_type, "
                        f"got {round_mode!r}"
                    ),
                )
            else:
                torch._check(
                    round_mode == "rint",
                    lambda: f"round_mode must be 'rint' for FP8 dst_type, got {round_mode!r}",
                )

            y_shape = list(x1.shape)
            if is_fp4:
                y_dtype = torch.uint8
                y_shape[-1] //= 2
            elif dst_type == 35:
                y_dtype = torch.float8_e5m2
            else:  # dst_type == 36; domain already validated above
                y_dtype = torch.float8_e4m3fn

            num_blocks = (x1.shape[-1] + 31) // 32
            mxscale_last = (num_blocks + 1) // 2
            mxscale_shape = list(x1.shape)
            mxscale_shape[-1] = mxscale_last
            mxscale_shape.append(2)

            rstd_shape = list(x1.shape)
            rstd_shape[-1] = 1

            y = torch.empty(y_shape, dtype=y_dtype, device="meta")
            x_out = torch.empty(x1.shape, dtype=x1.dtype, device="meta")
            mxscale = torch.empty(
                mxscale_shape, dtype=torch.float8_e8m0fnu, device="meta"
            )
            rstd = torch.empty(rstd_shape, dtype=torch.float32, device="meta")
            return y, x_out, mxscale, rstd


add_rms_norm_dynamic_quant_builder = AddRmsNormDynamicQuantOpBuilder()
add_rms_norm_dynamic_quant_builder._ensure_initialized()


def _type_name(value):
    """返回模块限定的类型名。

    numpy 2.x 中 ``type(np.bool_(True)).__name__`` 即为 ``"bool"``，直接渲染会产生
    "must be a Python bool, but got bool" 这类自相矛盾的信息，故带上模块名。
    """
    value_type = type(value)
    module = getattr(value_type, "__module__", "")
    if module and module != "builtins":
        return f"{module}.{value_type.__name__}"
    return value_type.__name__


def _validate_arg_types(
    x1, x2, gamma, beta, x3, epsilon, scale_alg, round_mode, dst_type, output_rstd
):
    """入口参数类型严格拦截。

    张量参数须为 torch.Tensor（beta/x3 允许 None）；标量参数仅接受 Python 原生
    类型，不做隐式转换或归一化：

    - epsilon：精确 int/float（np.float64 虽是 float 子类，仍拒绝）
    - round_mode：精确 str（np.str_ 虽是 str 子类，仍拒绝）
    - output_rstd：精确 bool（拒绝 np.bool_）
    - dst_type/scale_alg：int 且排除 bool —— 保留 enum.IntEnum 的可用性，
      同时拒绝 np.int64/np.bool_（二者均非 int 子类）

    严格拒绝第三方数值类型的动机：torch.dtype 经 dispatcher 会被静默转为其
    ScalarType 序号（float8_e5m2 -> 23、int4 -> 40，后者恰为合法 dst_type 取值），
    一旦开口支持框架类型，语义歧义与支持面都会持续扩散。

    注意：经 torch.ops dispatcher 调用时本函数仍会执行（@impl 即 PrivateUse1
    kernel），但参数此时已被 schema 归一化为原生类型，类型拦截在该路径不生效，
    仅由值域校验兜底。
    """
    for name, value in (("x1", x1), ("x2", x2), ("gamma", gamma)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"{name} must be a torch.Tensor, but got {_type_name(value)}"
            )
    for name, value in (("beta", beta), ("x3", x3)):
        if value is not None and not isinstance(value, torch.Tensor):
            raise TypeError(
                f"{name} must be a torch.Tensor or None, but got {_type_name(value)}"
            )
    if type(epsilon) is not int and type(epsilon) is not float:
        raise TypeError(
            f"epsilon must be a Python int or float, "
            f"but got {_type_name(epsilon)}: {epsilon!r}"
        )
    if isinstance(scale_alg, bool) or not isinstance(scale_alg, int):
        raise TypeError(
            f"scale_alg must be a Python int (0=OCP, 1=cuBLAS), "
            f"but got {_type_name(scale_alg)}: {scale_alg!r}"
        )
    if type(round_mode) is not str:
        raise TypeError(
            f"round_mode must be a Python str, "
            f"but got {_type_name(round_mode)}: {round_mode!r}"
        )
    if isinstance(dst_type, bool) or not isinstance(dst_type, int):
        raise TypeError(
            f"dst_type must be a Python int "
            f"(35=FP8_E5M2, 36=FP8_E4M3FN, 40=FP4_E2M1, 41=FP4_E1M2), "
            f"but got {_type_name(dst_type)}: {dst_type!r}"
        )
    if type(output_rstd) is not bool:
        raise TypeError(
            f"output_rstd must be a Python bool, "
            f"but got {_type_name(output_rstd)}: {output_rstd!r}"
        )


@impl(get_as_library(), add_rms_norm_dynamic_quant_builder.name, "PrivateUse1")
def add_rms_norm_dynamic_quant(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    beta: Optional[torch.Tensor] = None,
    x3: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6,
    scale_alg: int = 0,
    round_mode: str = "rint",
    dst_type: int = 40,
    output_rstd: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    NPU AddRmsNormDynamicMxQuant — Add + RmsNorm + MX 动态量化融合算子

    当 x3 提供时: x = (x3 + x1) + x2
    当 x3 为 None: x = x1 + x2

    参数类型严格要求：x1/x2/gamma 须为 torch.Tensor，beta/x3 须为
    torch.Tensor 或 None；标量仅接受 Python 原生类型（dst_type/scale_alg: int，
    round_mode: str，output_rstd: bool，epsilon: int/float），bool 与
    numpy/torch 等第三方类型（含 np.float64、np.str_ 这类内建类型子类）在入口
    直接抛 TypeError。
    """
    _validate_arg_types(
        x1, x2, gamma, beta, x3, epsilon, scale_alg, round_mode, dst_type, output_rstd
    )
    op_module = add_rms_norm_dynamic_quant_builder.load()
    return op_module.add_rms_norm_dynamic_quant(
        x1,
        x2,
        gamma,
        beta,
        x3,
        epsilon,
        scale_alg,
        round_mode,
        dst_type,
        output_rstd,
    )
