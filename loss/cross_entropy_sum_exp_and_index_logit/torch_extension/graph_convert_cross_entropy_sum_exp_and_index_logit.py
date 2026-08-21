# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for Graph Mode (CrossEntropySumExpAndIndexLogit)

try:
    import torch
    from torchair.ge import attr
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_nn.cross_entropy_sum_exp_and_index_logit.default
    )
    def convert_cross_entropy_sum_exp_and_index_logit(
        vocab_parallel_logits: Tensor,
        target: Tensor,
        global_logits_max: Tensor,
        vocab_start_index: int,
        vocab_end_index: int,
        meta_outputs: TensorSpec = None,
    ):
        return ge_op(
            op_type="CrossEntropySumExpAndIndexLogit",
            inputs={
                "vocab_parallel_logits": vocab_parallel_logits,
                "target": target,
                "global_logits_max": global_logits_max,
            },
            attrs={
                "vocab_start_index": attr.Int(vocab_start_index),
                "vocab_end_index": attr.Int(vocab_end_index),
            },
            outputs=[
                "predicted_logits",
                "sum_exp_logits",
                "exp_logits",
                "target_offset",
                "target_mask",
            ],
            ir=IrDef("CrossEntropySumExpAndIndexLogit")
            .input("vocab_parallel_logits", "DT_FLOAT, DT_BF16")
            .input("target", "DT_INT32")
            .input("global_logits_max", "DT_FLOAT, DT_BF16")
            .required_attr("vocab_start_index", attr.Int)
            .required_attr("vocab_end_index", attr.Int)
            .output("predicted_logits", "DT_FLOAT")
            .output("sum_exp_logits", "DT_FLOAT")
            .output("exp_logits", "DT_FLOAT")
            .output("target_offset", "DT_INT32")
            .output("target_mask", "DT_INT32"),
        )

else:

    def convert_cross_entropy_sum_exp_and_index_logit(*args, **kwargs):
        raise RuntimeError(
            "CrossEntropySumExpAndIndexLogit graph converter: torchair is not available."
        )
