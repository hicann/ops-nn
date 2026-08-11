# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for TurboQuantCompressLatent

try:
    import torch
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
        torch.ops.cann_ops_nn.turbo_quant_compress_latent.default
    )
    def convert_turbo_quant_compress_latent(
        latent: Tensor,
        centroids: Tensor,
        meta_outputs: TensorSpec = None,
    ):
        return ge_op(
            op_type="TurboQuantCompressLatent",
            inputs={"latent": latent, "centroids": centroids},
            attrs={},
            outputs=["slot"],
            ir=IrDef("TurboQuantCompressLatent")
            .input("latent", "DT_FLOAT")
            .input("centroids", "DT_FLOAT")
            .output("slot", "DT_UINT8"),
        )

else:

    def convert_turbo_quant_compress_latent(*args, **kwargs):
        raise RuntimeError(
            "TurboQuantCompressLatent graph converter: torchair is not available."
        )
