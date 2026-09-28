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
 * \file single_layer_lstm.cpp
 * \brief
 */
#include "single_layer_lstm.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

/* The same header the tiling includes. It is plain C++ on purpose, so the on-chip feasibility test
 * below is literally the one tiling will run -- not a second copy of it. */
#include "../op_host/arch35/single_layer_lstm_budget.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(SingleLayerLstm);

namespace {
/* The FP32 cube operands require both I and H to be multiples of 8. */
constexpr int64_t C0_BYTES = 32;
constexpr int64_t C0_FP32 = 8;
constexpr int64_t MIN_HIDDEN = 8;

/* Element width in bytes, or 0 for a dtype this operator does not implement. */
int64_t WidthOf(DataType dt)
{
    if (dt == DataType::DT_FLOAT) {
        return 4;
    }
    if (dt == DataType::DT_FLOAT16 || dt == DataType::DT_BF16) {
        return 2;
    }
    return 0;
}
} // namespace

/* Check tiling feasibility before adding the node to the executor. */
bool SingleLayerLstmSupports(const aclTensor* x, const aclTensor* initH, const char* direction, const char** reason)
{
    /* Return a diagnostic reason for unsupported inputs. */
    auto no = [reason](const char* why) {
        if (reason != nullptr) {
            *reason = why;
        }
        return false;
    };

    if (x == nullptr || initH == nullptr || direction == nullptr) {
        return no("x, initH or direction is null");
    }
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND950) {
        return no("this operator is implemented for ascend950 only");
    }
    /* All floating inputs and outputs share the declared dtype. */
    const int64_t inBytes = WidthOf(initH->GetDataType());
    if (inBytes == 0) {
        return no("initH must be float32, float16 or bfloat16");
    }
    if (x->GetDataType() != initH->GetDataType()) {
        return no("x and initH must have the same dtype");
    }
    if (strcmp(direction, "UNIDIRECTIONAL") != 0) {
        return no("only UNIDIRECTIONAL is implemented");
    }

    const op::Shape& xShape = x->GetViewShape();
    const op::Shape& hShape = initH->GetViewShape();
    if (xShape.GetDimNum() != 3 || hShape.GetDimNum() == 0) {
        return no("x must be [T, B, I] and initH must have at least one dimension");
    }
    const int64_t steps = xShape.GetDim(0);
    const int64_t batch = xShape.GetDim(1);
    const int64_t inputSize = xShape.GetDim(2);
    const int64_t hidden = hShape.GetDim(hShape.GetDimNum() - 1);
    if (steps <= 0 || batch <= 0 || inputSize <= 0 || hidden <= 0) {
        return no("every extent of T, B, input_size and hidden_size must be positive");
    }
    /* Both cube operands use FP32 alignment. */
    if (inputSize % C0_FP32 != 0) {
        return no("input_size must be a multiple of 8 at every dtype -- both phases feed the cube "
                  "float32, whatever the caller's width is");
    }
    if (hidden % C0_FP32 != 0) {
        return no("hidden_size must be a multiple of 8 at every dtype -- the recurrence and the "
                  "epilogue are float32 whatever the caller's width is");
    }
    /* There is no fixed upper hidden-size limit; the on-chip budget is checked below. */
    if (hidden < MIN_HIDDEN) {
        return no("hidden_size must be at least 8 -- one float32 fractal");
    }

    /* Use the same row-chunk feasibility calculation as tiling. */
    const uint32_t rowsPerBlock = SingleLayerLstmBudget::PickRowChunk(
        static_cast<uint32_t>(batch), static_cast<uint32_t>(hidden), static_cast<uint32_t>(steps),
        static_cast<uint32_t>(inputSize), static_cast<uint32_t>(inBytes));
    if (rowsPerBlock == 0) {
        return no("the persistent recurrence does not fit on chip at this hidden_size and T, even with "
                  "one batch row per cluster");
    }
    return true;
}

namespace {
using SingleLayerLstmResult = std::tuple<const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*,
                                         const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*>;

const SingleLayerLstmResult LSTM_NULLPTR_INNER = SingleLayerLstmResult(nullptr, nullptr, nullptr, nullptr, nullptr,
                                                                       nullptr, nullptr, nullptr);
} // namespace

const SingleLayerLstmResult SingleLayerLstm(const aclTensor* x, const aclTensor* w, const aclTensor* b,
                                            const aclTensor* initH, const aclTensor* initC,
                                            const aclTensor* seqLengthOptional, const char* direction,
                                            const char* gateOrder, aclTensor* yOut, aclTensor* outputHOut,
                                            aclTensor* outputCOut, aclTensor* iOut, aclTensor* jOut, aclTensor* fOut,
                                            aclTensor* oOut, aclTensor* tanhcOut, aclOpExecutor* executor,
                                            int64_t logicalInputSize, int64_t logicalHiddenSize,
                                            const aclTensor* biasHhOptional)
{
    L0_DFX(SingleLayerLstm, x, w, b, initH, initC, seqLengthOptional, direction, gateOrder, yOut, outputHOut,
           outputCOut, iOut, jOut, fOut, oOut, tanhcOut, logicalInputSize, logicalHiddenSize, biasHhOptional);

    /* The kernel is registered for ascend950 only. Returning the nullptr tuple here rather than
     * launching lets the caller report an unsupported-SoC error instead of the framework refusing at
     * binary lookup with "binary_info_config.json of socVersion [...] does not support opType", which
     * says nothing about which operator the caller asked for. */
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND950) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "SingleLayerLstm is implemented for ascend950 only.");
        return LSTM_NULLPTR_INNER;
    }

    auto ret = INFER_SHAPE(SingleLayerLstm, OP_INPUT(x, w, b, initH, initC, seqLengthOptional, biasHhOptional),
                           OP_OUTPUT(yOut, outputHOut, outputCOut, iOut, jOut, fOut, oOut, tanhcOut),
                           OP_ATTR(direction, gateOrder, logicalInputSize, logicalHiddenSize));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SingleLayerLstm InferShape failed."),
             return LSTM_NULLPTR_INNER);

    ret = ADD_TO_LAUNCHER_LIST_AICORE(SingleLayerLstm,
                                      OP_INPUT(x, w, b, initH, initC, seqLengthOptional, biasHhOptional),
                                      OP_OUTPUT(yOut, outputHOut, outputCOut, iOut, jOut, fOut, oOut, tanhcOut),
                                      OP_ATTR(direction, gateOrder, logicalInputSize, logicalHiddenSize));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SingleLayerLstm ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return LSTM_NULLPTR_INNER);

    return SingleLayerLstmResult(yOut, outputHOut, outputCOut, iOut, jOut, fOut, oOut, tanhcOut);
}

} // namespace l0op
