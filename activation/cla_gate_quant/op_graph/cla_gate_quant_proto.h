/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cla_gate_quant_proto.h
 * \brief ClaGateQuant graph IR definition.
 */

#ifndef OPS_OP_PROTO_INC_CLA_GATE_QUANT_H_
#define OPS_OP_PROTO_INC_CLA_GATE_QUANT_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
#ifndef OPS_PROTO_DEF_CLAGATEQUANT
#define OPS_PROTO_DEF_CLAGATEQUANT
/**
 * @brief Fused operator of CLA (Cross-Layer Attention) two-branch head-wise gate
 * weighted merging and dual-axis dynamic block quantization.
 *
 * It first applies Sigmoid gating to the attention outputs of the Global/CLA branch
 * and the Local/SWA branch respectively and merges them with weighting, then reshapes
 * the merged result to [T, K] (K = N * D) and performs block-based dynamic quantization
 * along the K direction ([1,32] block, row-wise) and the T direction ([32,1] block,
 * col-wise), producing low-precision FP8/FP4 tensors and the corresponding E8M0
 * scaling factors.
 *
 * Formulas:
 * @code{.c}
 * s_g = sigmoid(global_gate_logits)
 * s_l = sigmoid(local_gate_logits)
 * merged[t, n, d] =
 *     s_g[t, n] * global_attn[t, n, d] + s_l[t, n] * local_attn[t, n, d]
 * shared_exp = floor(log2(max_i(abs(V_i)))) - emax
 * scale = 2^shared_exp
 * data_i = cast(V_i / scale)
 * @endcode
 *
 * where emax is the exponent of the largest normal number of the target data type,
 * and the exponent is shared within a block.
 *
 * @par Inputs:
 * @li global_attn: Attention output of the Global/CLA branch, shape [T, N, D], 3-D. The data type must be
 *     float16 or bfloat16.
 * @li local_attn: Attention output of the Local/SWA branch, shape [T, N, D], consistent with global_attn in
 *     shape and data type.
 * @li global_gate_logits: Pre-Sigmoid value of the global gate, shape [T, N], 2-D. The data type must be the
 *     same as global_attn.
 * @li local_gate_logits: Pre-Sigmoid value of the local gate, shape [T, N], consistent with global_gate_logits
 *     in shape and data type.
 *
 * @par Outputs:
 * @li row_data: Row-wise (K direction, block [1,32]) quantized data, shape [T, K] with K = N*D, 2-D.
 *     The data type is determined by dst_type and must be float8_e5m2, float8_e4m3fn, float4_e2m1 or
 *     float4_e1m2.
 * @li row_scale: E8M0 scaling factor of each row-wise [1,32] group, shape [T, ceil(K/64), 2], 3-D; the
 *     middle dim is the number of [1,32] groups along K paired two by two, ceil(K/64) = ceil(ceil(K/32)/2),
 *     with two adjacent factors packed into the last dimension and zero padding for even alignment.
 *     The data type must be float8_e8m0.
 * @li col_data: Col-wise (T direction, block [32,1]) quantized data, shape [T, K], 2-D; empty output of
 *     shape [0] when dual_axis_flag=false. The data type is the same as row_data.
 * @li col_scale: E8M0 scaling factor of each col-wise [32,1] group, shape [ceil(T/64), K, 2], 3-D; the
 *     first dim is the number of [32,1] groups along T paired two by two, ceil(T/64) = ceil(ceil(T/32)/2),
 *     with zero padding for even alignment; empty output of shape [0] when dual_axis_flag=false.
 *     The data type must be float8_e8m0.
 *
 * @par Attributes:
 * @li dst_type: Optional int, target quantization type, one of FLOAT8_E5M2 / FLOAT8_E4M3FN /
 *     FLOAT4_E2M1 / FLOAT4_E1M2. Defaults to 36 (FLOAT8_E4M3FN).
 * @li round_mode: Optional string, rounding mode. FP8 supports only "rint"; FP4 supports
 *     "rint"/"floor"/"round". Defaults to "rint".
 * @li scale_alg: Optional int, scale computation method. 1=cuBLAS, 0=OCP; FP4 supports only 0.
 *     Defaults to 1.
 * @li input_attn_layout: Optional string, layout format of the global_attn/local_attn inputs.
 *     Currently only "TND" is supported. Defaults to "TND".
 * @li dual_axis_flag: Optional bool. true=dual-axis quantization (output both row/col results);
 *     false=single-axis quantization (output row-wise only, col_data/col_scale are empty).
 *     Defaults to false.
 *
 * @par Restrictions:
 * @li global_attn / local_attn must be 3-D tensors with the same shape; N is in [1,128] and
 *     D is 128 or 256.
 * @li global_gate_logits / local_gate_logits must be 2-D tensors of shape [T, N], with the same
 *     data type as the attention outputs.
 * @li When dst_type is FP4, N*D must be divisible by 4 and scale_alg must be 0.
 *
 * @par Third-party framework compatibility
 * Custom operator, with no corresponding Caffe / ONNX / TensorFlow / PyTorch operator.
 */
REG_OP(ClaGateQuant)
    .INPUT(global_attn, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(local_attn, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(global_gate_logits, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(local_gate_logits, TensorType({DT_BF16, DT_FLOAT16}))
    .OUTPUT(row_data, TensorType({DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
    .OUTPUT(row_scale, TensorType({DT_FLOAT8_E8M0}))
    .OUTPUT(col_data, TensorType({DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
    .OUTPUT(col_scale, TensorType({DT_FLOAT8_E8M0}))
    .ATTR(dst_type, Int, 36)
    .ATTR(round_mode, String, "rint")
    .ATTR(scale_alg, Int, 1)
    .ATTR(input_attn_layout, String, "TND")
    .ATTR(dual_axis_flag, Bool, false)
    .OP_END_FACTORY_REG(ClaGateQuant)
#endif
} // namespace ge

#endif // OPS_OP_PROTO_INC_CLA_GATE_QUANT_H_
