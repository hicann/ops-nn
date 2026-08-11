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
 * \file turbo_quant_compress_latent_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_TURBO_QUANT_COMPRESS_LATENT_H_
#define OPS_BUILT_IN_OP_PROTO_INC_TURBO_QUANT_COMPRESS_LATENT_H_
#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Compresses an MLA KV latent into a TurboQuant 4-bit slot.
*
* Each token is L2-normalized, every element is mapped to the nearest of 16 sorted centroids, and the
* resulting nibbles are packed two per byte in dim order. The fp16 L2 norm is appended so the read side
* can reconstruct the original magnitude.

* @par Inputs:
- latent: A 2D tensor of type float32 with shape [numTokens, headDim]. The rotation into the quantization
*   basis (signed Hadamard) is expected to have been applied already; the tensor must NOT be normalized.
*   Only headDim = 512 is supported.
- centroids: A tensor of type float32 holding exactly 16 elements, sorted in ascending order.

* @par Outputs:
- slot: A 2D tensor of type uint8 with shape [numTokens, slotSize], where
*   slotSize = ceil((headDim / 2 + 2) / 64) * 64. Layout per token:
*   - [0, headDim / 2): the packed nibbles, low nibble first.
*   - [headDim / 2, headDim / 2 + 2): the L2 norm as float16.
*   - the remaining bytes are zero padding.

* @par Third-party framework compatibility:
* Custom operator with no direct mapping in Caffe/ONNX/TensorFlow/PyTorch.
*/
REG_OP(TurboQuantCompressLatent)
    .INPUT(latent, TensorType({DT_FLOAT}))
    .INPUT(centroids, TensorType({DT_FLOAT}))
    .OUTPUT(slot, TensorType({DT_UINT8}))
    .OP_END_FACTORY_REG(TurboQuantCompressLatent)
} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_TURBO_QUANT_COMPRESS_LATENT_H_
