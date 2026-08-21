/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>

#include "op_fallback.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CHECK_INPUT_IS_NULL(input, name)                          \
    do {                                                          \
        if ((input) == nullptr) {                                 \
            OP_LOGD("aclnnfallback", "input [%s] is null", name); \
        }                                                         \
    } while (0)

namespace fallback {
using namespace ge;
using namespace gert;
constexpr size_t INPUT_X_IDX = 0;
constexpr size_t INPUT_WEIGHT_IDX = 1;
constexpr size_t INPUT_ANTIQUANTSCALE_IDX = 2;
constexpr size_t INPUT_ANTIQUANTOFFSET_IDX = 3;
constexpr size_t INPUT_QUANTSCALE_IDX = 4;
constexpr size_t INPUT_QUANTOFFSET_IDX = 5;
constexpr size_t INPUT_BIAS_IDX = 6;
constexpr size_t OUTPUT_Y_IDX = 0;

struct AclTensorDeleter {
    void operator()(aclTensor* p) const
    {
        if (p == nullptr) {
            return;
        }
        static const auto aclDestroyTensor = GET_OP_API_FUNC(aclDestroyTensor);
        OP_CHECK_IF(aclDestroyTensor == nullptr, OP_LOGE("aclnnfallback", "aclDestroyTensor is null"), return);
        aclDestroyTensor(p);
    }
};
using AclTensorGuard = std::unique_ptr<aclTensor, AclTensorDeleter>;

aclTensor* ConvertMmType(const gert::Tensor* ge_tensor, bool transpose)
{
    if (ge_tensor == nullptr) {
        return nullptr;
    }
    auto gert_shape = ge_tensor->GetStorageShape();
    if (gert_shape.GetDimNum() <= 1) {
        return ConvertType(ge_tensor);
    }

    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    OP_CHECK_IF(aclCreateTensor == nullptr, OP_LOGE("aclnnfallback", "aclCreateTensor nullptr"), return nullptr);

    void* device_addr = (void*)ge_tensor->GetAddr();
    auto dataType = GetConvertType(ge_tensor);
    std::vector<int64_t> shape;
    for (size_t i = 0; i < gert_shape.GetDimNum(); ++i) {
        shape.push_back(gert_shape.GetDim(i));
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    auto viewShape = shape;
    if (transpose) {
        auto dimM = shape.size() - 2;
        auto dimN = shape.size() - 1;
        auto swap = strides[dimN];
        strides[dimN] = strides[dimM];
        strides[dimM] = swap;
        viewShape[dimN] = shape[dimM];
        viewShape[dimM] = shape[dimN];
    }
    auto acl_format = aclFormat::ACL_FORMAT_ND;
    aclTensor* out = aclCreateTensor(viewShape.data(), shape.size(), dataType, strides.data(), 0, acl_format,
                                     shape.data(), shape.size(), device_addr);
    OP_CHECK_IF(out == nullptr, OP_LOGE("aclnnfallback", "out nullptr"), return nullptr);

    return out;
}

static graphStatus WeightQuantBmmV2ExecuteFunc(OpExecuteContext* host_api_ctx)
{
    OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE("aclnnfallback", "host_api_ctx is null"), return GRAPH_FAILED);

    auto geX = host_api_ctx->GetInputTensor(INPUT_X_IDX);
    OP_CHECK_IF(geX == nullptr, OP_LOGE("aclnnfallback", "input x is null"), return GRAPH_FAILED);

    auto geWeight = host_api_ctx->GetInputTensor(INPUT_WEIGHT_IDX);
    OP_CHECK_IF(geWeight == nullptr, OP_LOGE("aclnnfallback", "input weight is null"), return GRAPH_FAILED);

    auto geAntiquantScale = host_api_ctx->GetInputTensor(INPUT_ANTIQUANTSCALE_IDX);
    OP_CHECK_IF(geAntiquantScale == nullptr, OP_LOGE("aclnnfallback", "input antiquantScale is null"),
                return GRAPH_FAILED);

    auto geAntiquantOffset = host_api_ctx->GetOptionalInputTensor(INPUT_ANTIQUANTOFFSET_IDX);
    CHECK_INPUT_IS_NULL(geAntiquantOffset, "antiquantOffset");

    auto geQuantScale = host_api_ctx->GetOptionalInputTensor(INPUT_QUANTSCALE_IDX);
    CHECK_INPUT_IS_NULL(geQuantScale, "quantScale");

    auto geQuantOffset = host_api_ctx->GetOptionalInputTensor(INPUT_QUANTOFFSET_IDX);
    CHECK_INPUT_IS_NULL(geQuantOffset, "quantOffset");

    auto geBias = host_api_ctx->GetOptionalInputTensor(INPUT_BIAS_IDX);
    CHECK_INPUT_IS_NULL(geBias, "bias");

    auto geY = host_api_ctx->GetOutputTensor(OUTPUT_Y_IDX);
    OP_CHECK_IF(geY == nullptr, OP_LOGE("aclnnfallback", "geY is null"), return GRAPH_FAILED);

    auto attrs = host_api_ctx->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE("aclnnfallback", "attrs is null"), return GRAPH_FAILED);

    const bool* ptrTransposeX = attrs->GetBool(0);
    const bool transposeX = (ptrTransposeX != nullptr ? *ptrTransposeX : false);
    const bool* ptrTransposeWeight = attrs->GetBool(1);
    const bool transposeWeight = (ptrTransposeWeight != nullptr ? *ptrTransposeWeight : false);
    OP_LOGD("aclnnfallback", "transposeX is %d, transposeWeight is %d", transposeX, transposeWeight);

    AclTensorGuard aclXGuard(ConvertMmType(geX, transposeX));
    OP_CHECK_IF(aclXGuard == nullptr, OP_LOGE("aclnnfallback", "aclX is null"), return ge::GRAPH_FAILED);
    AclTensorGuard aclWeightGuard(ConvertMmType(geWeight, transposeWeight));
    OP_CHECK_IF(aclWeightGuard == nullptr, OP_LOGE("aclnnfallback", "aclWeight is null"), return ge::GRAPH_FAILED);
    AclTensorGuard aclAntiquantScaleGuard(ConvertMmType(geAntiquantScale, transposeWeight));
    OP_CHECK_IF(aclAntiquantScaleGuard == nullptr, OP_LOGE("aclnnfallback", "aclAntiquantScale is null"),
                return ge::GRAPH_FAILED);
    AclTensorGuard aclAntiquantOffsetGuard(ConvertMmType(geAntiquantOffset, transposeWeight));

    const int64_t* ptrAntiquantGroupSize = attrs->GetInt(2);
    int antiquantGroupSize = (ptrAntiquantGroupSize != nullptr ? static_cast<int>(*ptrAntiquantGroupSize) : 0);

    const int64_t* ptrInnerPrecise = attrs->GetInt(4);
    int innerPrecise = ptrInnerPrecise != nullptr ? static_cast<int>(*ptrInnerPrecise) : 0;

    aclTensor* aclX = aclXGuard.release();
    aclTensor* aclWeight = aclWeightGuard.release();
    aclTensor* aclAntiquantScale = aclAntiquantScaleGuard.release();
    aclTensor* aclAntiquantOffset = aclAntiquantOffsetGuard.release();

    if (innerPrecise == 1) {
        auto apiRet = CANN_OPS_OPB_SYN_EXEC_ACLNN(host_api_ctx, aclnnWeightQuantBatchMatmulV3, aclX, aclWeight,
                                                  aclAntiquantScale, aclAntiquantOffset, geQuantScale, geQuantOffset,
                                                  geBias, antiquantGroupSize, innerPrecise, geY);
        OP_CHECK_IF(apiRet != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "apiRet failed:%d", apiRet), return GRAPH_FAILED);
        OP_LOGD("aclnnfallback", "aclnnWeightQuantBatchMatmulV3 run success");
    } else {
        auto apiRet = CANN_OPS_OPB_SYN_EXEC_ACLNN(host_api_ctx, aclnnWeightQuantBatchMatmulV2, aclX, aclWeight,
                                                  aclAntiquantScale, aclAntiquantOffset, geQuantScale, geQuantOffset,
                                                  geBias, antiquantGroupSize, geY);
        OP_CHECK_IF(apiRet != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "apiRet failed:%d", apiRet), return GRAPH_FAILED);
        OP_LOGD("aclnnfallback", "aclnnWeightQuantBatchMatmulV2 run success");
    }
    return GRAPH_SUCCESS;
}

IMPL_OP(WeightQuantBatchMatmulV2).OpExecuteFunc(WeightQuantBmmV2ExecuteFunc);
} // namespace fallback

#ifdef __cplusplus
}
#endif
