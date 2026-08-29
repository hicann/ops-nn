# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for Graph Mode (arch35 only)

try:
    from typing import Optional

    import torch
    from torchair.ge import attr
    from torchair.ge._ge_graph import DataType, Tensor, TensorSpec
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )
    from torchair._ge_concrete_graph.utils import (
        specific_op_input_layout,
        specific_op_output_layout,
    )

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False


def _device_name():
    try:
        import torch_npu

        return torch_npu.npu.get_device_name()
    except Exception:
        return "unknown"


def _is_arch35():
    try:
        import torch_npu

        device_name = torch_npu.npu.get_device_name().lower()
        return "ascend950" in device_name or "ascend910_95" in device_name
    except Exception:
        return False


def _norm3(values):
    """Normalize a 1D kernel/stride list to [d, h, w] (len 1 -> broadcast, len 2 -> pad)."""
    v = [int(x) for x in values]
    if len(v) == 1:
        return [v[0]] * 3
    return (v * 3)[:3]


def _norm_pads(values):
    """Normalize padding to the GE 6-value form [dL, dR, hT, hB, wL, wR]."""
    p = [int(x) for x in values]
    if len(p) == 6:
        return p
    if len(p) == 3:
        return [p[0], p[0], p[1], p[1], p[2], p[2]]
    return [p[0]] * 6


if _TORCHAIR_AVAILABLE:

    def AvgPool3DGrad(
        orig_input_shape: Tensor,
        grads: Tensor,
        *,
        ksize,
        strides,
        pads,
        ceil_mode,
        count_include_pad,
        divisor_override,
    ):
        """
        GE op wrapper for AvgPool3DGrad.

        REG_OP(AvgPool3DGrad)
            .INPUT(orig_input_shape, TensorType({DT_INT32}))
            .INPUT(grads, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
            .OUTPUT(output, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
            .REQUIRED_ATTR(ksize, ListInt)
            .REQUIRED_ATTR(strides, ListInt)
            .REQUIRED_ATTR(pads, ListInt)
            .ATTR(ceil_mode, Bool, false)
            .ATTR(count_include_pad, Bool, true)
            .ATTR(divisor_override, Int, 0)
            .ATTR(data_format, String, "NDHWC")
            .OP_END_FACTORY_REG(AvgPool3DGrad)
        """
        return ge_op(
            op_type="AvgPool3DGrad",
            inputs={"orig_input_shape": orig_input_shape, "grads": grads},
            attrs={
                "ksize": attr.ListInt(ksize),
                "strides": attr.ListInt(strides),
                "pads": attr.ListInt(pads),
                "ceil_mode": attr.Bool(ceil_mode),
                "count_include_pad": attr.Bool(count_include_pad),
                "divisor_override": attr.Int(divisor_override),
                "data_format": attr.Str("NCDHW"),
            },
            outputs=["output"],
            ir=IrDef("AvgPool3DGrad")
            .input("orig_input_shape", "DT_INT32")
            .input("grads", "DT_FLOAT16, DT_FLOAT, DT_BF16")
            .attr("ksize", attr.ListInt([]))
            .attr("strides", attr.ListInt([]))
            .attr("pads", attr.ListInt([]))
            .attr("ceil_mode", attr.Bool(False))
            .attr("count_include_pad", attr.Bool(True))
            .attr("divisor_override", attr.Int(0))
            .attr("data_format", attr.Str("NDHWC"))
            .output("output", "DT_FLOAT16, DT_FLOAT, DT_BF16"),
        )

    @register_fx_node_ge_converter(torch.ops.cann_ops_nn.avg_pool3d_backward.default)
    def convert_avg_pool3d_backward(
        grad_output: Tensor,
        self: Tensor,
        kernel_size: list,
        stride: list,
        padding: list,
        ceil_mode: bool,
        count_include_pad: bool,
        divisor_override: Optional[int] = None,
        meta_outputs: TensorSpec = None,
    ):
        if not _is_arch35():
            raise RuntimeError(
                "avg_pool3d_backward graph mode is only supported on arch35 (Ascend950), "
                f"but got device: {_device_name()}."
            )

        ksize = _norm3(kernel_size)
        strides = ksize if len(stride) == 0 else _norm3(stride)
        pads = _norm_pads(padding)

        divisor_override_value = 0 if not divisor_override else divisor_override

        if self.rank == 4:
            # 4D: (C,D,H,W) -> 5D: (1,C,D,H,W)
            one = ge.Const([1], dtype=DataType.DT_INT32)
            orig_shape_5d = ge.ConcatV2([one, ge.Shape(self)], concat_dim=0, N=2)
            grads_shape_5d = ge.ConcatV2(
                [one, ge.Shape(grad_output)], concat_dim=0, N=2
            )
            grads_5d = ge.Reshape(grad_output, grads_shape_5d)

            output_5d = AvgPool3DGrad(
                orig_shape_5d,
                grads_5d,
                ksize=ksize,
                strides=strides,
                pads=pads,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
                divisor_override=divisor_override_value,
            )
            output_5d.desc.dtype = grad_output.desc.dtype
            specific_op_input_layout(output_5d, indices=[1], layout="NCDHW")
            specific_op_output_layout(output_5d, indices=0, layout="NCDHW")

            # 5D -> 4D 还原
            output = ge.Reshape(output_5d, ge.Shape(self))
            output.desc.dtype = grad_output.desc.dtype
            specific_op_output_layout(output, indices=0, layout="NCDHW")
            return output
        else:
            # 5D 保持原逻辑
            output = AvgPool3DGrad(
                ge.Shape(self),
                grad_output,
                ksize=ksize,
                strides=strides,
                pads=pads,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
                divisor_override=divisor_override_value,
            )
            output.desc.dtype = grad_output.desc.dtype
            specific_op_input_layout(output, indices=[1], layout="NCDHW")
            specific_op_output_layout(output, indices=0, layout="NCDHW")
            return output

else:

    def convert_avg_pool3d_backward(*args, **kwargs):
        raise RuntimeError(
            "avg_pool3_d_grad graph converter: torchair is not available."
        )
