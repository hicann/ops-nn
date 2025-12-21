/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_trans_quant_param.h"

#include <vector>

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "quant_util.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace {
static const size_t ALIGN_NUM = 16;

static aclnnStatus CheckQuantParam(
    const float* offsetArray, uint64_t offsetSize, const float* scaleArray, uint64_t scaleSize)
{
    if (offsetArray != nullptr) {
        if (offsetSize < 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "offsetSize must be larger than 0, but got %lu", offsetSize);
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if (offsetSize != 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "offsetArray is nullptr, offsetSize must be 0, but got %lu", offsetSize);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    if (scaleArray != nullptr) {
        if (scaleSize < 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scaleSize must be larger than 0, but got %lu", scaleSize);
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "scaleArray can not be nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    if (scaleSize == 0 && offsetSize == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The offsetSize and scaleSize are equal to 0, please check");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}
}; // namespace

aclnnStatus aclnnTransQuantParam(
    const float* scaleArray, uint64_t scaleSize, const float* offsetArray, uint64_t offsetSize, uint64_t** quantParam,
    uint64_t* quantParamSize)
{
    CHECK_RET(
        CheckQuantParam(offsetArray, offsetSize, scaleArray, scaleSize) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(
        quantParam != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "The quantParam is nullptr, please check."),
        return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(
        quantParamSize != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "The quantParamSize is nullptr, please check."),
        return ACLNN_ERR_PARAM_NULLPTR);

    // align
    *quantParamSize = (scaleSize > offsetSize) ? scaleSize : offsetSize;
    if (*quantParamSize != 1) {
        auto remainder = *quantParamSize % ALIGN_NUM;
        if (remainder != 0) {
            *quantParamSize = *quantParamSize + ALIGN_NUM - remainder;
        }
    }
    *quantParam = (uint64_t*)calloc(*quantParamSize, sizeof(uint64_t));

    OP_CHECK(*quantParam != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The *quantParam is nullptr.");
             *quantParamSize = 0, return ACLNN_ERR_INNER_NULLPTR);

    if (scaleArray != nullptr) {
        TransQuantScaleToM1(scaleArray, scaleSize, quantParam);
    }

    if (offsetArray != nullptr) {
        TransQuantOffsetToToM1(offsetArray, offsetSize, quantParam);
    }

    OP_CHECK(*quantParam != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The *quantParam is nullptr.");
             *quantParamSize = 0, return ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif