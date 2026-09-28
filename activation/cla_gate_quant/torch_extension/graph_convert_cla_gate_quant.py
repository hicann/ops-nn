# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for Graph Mode

try:
    import torch
    from torchair.ge import attr
    from torchair.ge._ge_graph import (
        DataType,
        Tensor,
        TensorSpec,
        _ge_dtype_to_ge_proto_dtype,
    )
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False


if _TORCHAIR_AVAILABLE:

    def _resolve_ge_dst_type(dst_type):
        """Normalize the PTA int dst_type to GE attr dst_type 35/36/40/41."""
        if dst_type is None:
            dst_type = 24  # torch.float8_e4m3fn
        mapping = {
            23: 35,  # torch.float8_e5m2
            24: 36,  # torch.float8_e4m3fn
            296: 40,  # torch_npu.float4_e2m1fn_x2
            297: 41,  # torch_npu.float4_e1m2fn_x2
        }
        if dst_type not in mapping:
            raise ValueError(
                "dst_type must be one of "
                "23 (torch.float8_e5m2), 24 (torch.float8_e4m3fn), "
                "296 (torch_npu.float4_e2m1fn_x2), 297 (torch_npu.float4_e1m2fn_x2), "
                f"or None, but got {dst_type}"
            )
        return mapping[dst_type]

    def _pack_fp4_to_uint8(tensor: Tensor, rank: int) -> Tensor:
        """Pack a logical FP4 GE tensor [..., R] into uint8 [..., R//2].

        ClaGateQuant declares FP4 outputs with the unpacked logical shape, while the torch
        API exposes them as packed uint8 holding two FP4 values per byte, so the logical
        shape is halved along the last dim before the bitcast.
        rank is passed in by the caller because the output tensor desc carries no shape at
        conversion time. Chains GE graph ops: Reshape -> Bitcast -> Reshape.
        """
        bit_shape = [1] * (rank - 1) + [2]
        div_x2 = ge.Cast(ge.Const(bit_shape), dst_type=DataType.DT_INT32)
        logical_shape = ge.Shape(tensor)
        packed_shape = ge.Div(logical_shape, div_x2)
        pair_shape = ge.ConcatV2(
            [packed_shape, ge.Cast(ge.Const([2]), dst_type=DataType.DT_INT32)],
            concat_dim=0,
            N=2,
        )
        packed = ge.Bitcast(ge.Reshape(tensor, pair_shape), type=DataType.DT_UINT8)
        packed = ge.Reshape(packed, packed_shape)
        packed.desc.dtype = _ge_dtype_to_ge_proto_dtype(DataType.DT_UINT8)
        return packed

    @register_fx_node_ge_converter(torch.ops.cann_ops_nn.cla_gate_quant.default)
    def convert_cla_gate_quant(
        global_attn: Tensor,
        local_attn: Tensor,
        global_gate_logits: Tensor,
        local_gate_logits: Tensor,
        *,
        dst_type=None,
        round_mode: str = "rint",
        scale_alg: int = 1,
        input_attn_layout: str = "TND",
        dual_axis_flag: bool = False,
        meta_outputs: TensorSpec = None,
    ):
        ge_dst_type = _resolve_ge_dst_type(dst_type)
        row_data, row_scale, col_data, col_scale = ge_op(
            op_type="ClaGateQuant",
            inputs={
                "global_attn": global_attn,
                "local_attn": local_attn,
                "global_gate_logits": global_gate_logits,
                "local_gate_logits": local_gate_logits,
            },
            attrs={
                "dst_type": attr.Int(ge_dst_type),
                "round_mode": attr.Str(round_mode),
                "scale_alg": attr.Int(scale_alg),
                "input_attn_layout": attr.Str(input_attn_layout),
                "dual_axis_flag": attr.Bool(dual_axis_flag),
            },
            outputs=[
                "row_data",
                "row_scale",
                "col_data",
                "col_scale",
            ],
            ir=IrDef("ClaGateQuant")
            .input("global_attn", "DT_FLOAT16, DT_BF16")
            .input("local_attn", "DT_FLOAT16, DT_BF16")
            .input("global_gate_logits", "DT_FLOAT16, DT_BF16")
            .input("local_gate_logits", "DT_FLOAT16, DT_BF16")
            .attr("dst_type", attr.Int(36))
            .attr("round_mode", attr.Str("rint"))
            .attr("scale_alg", attr.Int(1))
            .attr("input_attn_layout", attr.Str("TND"))
            .attr("dual_axis_flag", attr.Bool(False))
            .output(
                "row_data",
                "DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2",
            )
            .output("row_scale", "DT_FLOAT8_E8M0")
            .output(
                "col_data",
                "DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2",
            )
            .output("col_scale", "DT_FLOAT8_E8M0"),
        )

        if ge_dst_type in (40, 41):
            # row_data is [T, K]; col_data is [T, K] for dual axis and an empty [0] otherwise.
            row_data = _pack_fp4_to_uint8(row_data, rank=2)
            col_data = _pack_fp4_to_uint8(col_data, rank=2 if dual_axis_flag else 1)

        # GE declares the scales as DT_FLOAT8_E8M0 while the torch API exposes them as uint8;
        # reinterpret the bytes so the graph output dtype matches the FX net output.
        row_scale = ge.Bitcast(row_scale, type=DataType.DT_UINT8)
        col_scale = ge.Bitcast(col_scale, type=DataType.DT_UINT8)

        return row_data, row_scale, col_data, col_scale

else:

    def convert_cla_gate_quant(*args, **kwargs):
        raise RuntimeError("ClaGateQuant graph converter: torchair is not available.")
