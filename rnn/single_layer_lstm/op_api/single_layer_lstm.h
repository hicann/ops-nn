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
 * \file single_layer_lstm.h
 * \brief Level-0 wrapper for the SingleLayerLstm operator (single layer, single direction).
 */
#ifndef OP_API_INC_LEVEL0_OP_SINGLE_LAYER_LSTM_H_
#define OP_API_INC_LEVEL0_OP_SINGLE_LAYER_LSTM_H_

#include <cstdint>
#include "opdev/op_executor.h"

namespace l0op {

/* Whether a SingleLayerLstm node may be built for this call. aclnn picks the L0 node while building
 * the executor and tiling refuses at launch, where a refusal is fatal to the whole aclnnLSTM call
 * rather than a fallback -- so everything tiling can refuse for has to be decidable here. The
 * on-chip feasibility test comes from single_layer_lstm_budget.h, which both sides include; the
 * dtype, attribute and alignment rules are restated here and must be kept in step with tiling.
 *
 * `x` is [T, B, I]; `initH` is [1, B, H] or [B, H]. A null `initH` answers false: init_h is REQUIRED.
 * `reason`, written only on a false answer, names the rule that was broken, because on ascend950 the
 * caller reports the refusal instead of quietly choosing another node. */
bool SingleLayerLstmSupports(const aclTensor* x, const aclTensor* initH, const char* direction,
                             const char** reason = nullptr);

/* One SingleLayerLstm node. `w` is the fused weight [I+H, 4H], input rows first and hidden rows
 * already transposed; `b` is [4H]; `initH` and `initC` are [B, H]. The eight outputs in order:
 * y, outputH, outputC, i, j, f, o, tanhc -- each [T, B, H].
 *
 * `direction` is passed through for a future reverse kernel; tiling currently refuses anything but
 * "UNIDIRECTIONAL". The logical extents describe zero-padded storage, -1 meaning the physical
 * extent, and must satisfy 0 <= logicalInputSize <= I and 0 < logicalHiddenSize <= H. With
 * biasHhOptional, b and biasHhOptional are separate [4H] biases at x's dtype that the kernel sums
 * in FP32; without it b is already fused. */
const std::tuple<const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*,
                 const aclTensor*, const aclTensor*, const aclTensor*>
SingleLayerLstm(const aclTensor* x, const aclTensor* w, const aclTensor* b, const aclTensor* initH,
                const aclTensor* initC, const aclTensor* seqLengthOptional, const char* direction,
                const char* gateOrder, aclTensor* yOut, aclTensor* outputHOut, aclTensor* outputCOut, aclTensor* iOut,
                aclTensor* jOut, aclTensor* fOut, aclTensor* oOut, aclTensor* tanhcOut, aclOpExecutor* executor,
                int64_t logicalInputSize = -1, int64_t logicalHiddenSize = -1,
                const aclTensor* biasHhOptional = nullptr);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_OP_SINGLE_LAYER_LSTM_H_
