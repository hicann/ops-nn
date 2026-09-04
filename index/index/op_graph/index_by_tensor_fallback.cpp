/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include "op_fallback.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace fallback {
using namespace ge;
using namespace gert;
constexpr size_t INPUT_X_ID = 0;
constexpr size_t INPUT_INDICES_ID = 1;
constexpr size_t ATTR_MASK_IDX = 0;
constexpr size_t OUTPUT_Y_ID = 0;

static std::vector<const gert::Tensor*> getIndicesWithMask(std::vector<const gert::Tensor*> indices,
                                                           int64_t indices_num, const int64_t* mask, int64_t mask_num,
                                                           gert::Tensor* emptyTensor)
{
    if (mask == nullptr) {
        return indices;
    }

    std::vector<const gert::Tensor*> indicesList;
    int64_t numZeros = 0;
    for (int64_t i = 0; i < mask_num; i++) {
        if (mask[i] == 0) {
            indicesList.emplace_back(emptyTensor);
            numZeros++;
        } else if (mask[i] == 1) {
            indicesList.emplace_back(indices[i - numZeros]);
        } else {
            OP_LOGE("aclnnfallback", "Illegal value of mask");
        }
    }

    return indicesList;
}

static graphStatus IndexByTensorHostExecuteFunc(OpExecuteContext* hostApiCtx)
{
    OP_CHECK_IF(hostApiCtx == nullptr, OP_LOGE("IndexByTensor", "hostApiCtx is null"), return GRAPH_FAILED);
    OP_LOGD(hostApiCtx->GetNodeName(), "Enter IndexByTensorHostExecuteFunc");

    // self
    auto self_ge = hostApiCtx->GetInputTensor(INPUT_X_ID);
    OP_CHECK_IF(self_ge == nullptr, OP_LOGE(hostApiCtx->GetNodeName(), "self_ge is null"), return GRAPH_FAILED);

    auto input_num = hostApiCtx->GetComputeNodeInputNum();
    std::vector<const gert::Tensor*> ge_tenserListValue;
    for (size_t i = 1; i < input_num; i++) {
        auto ge_t = hostApiCtx->GetInputTensor(i);
        ge_tenserListValue.push_back(ge_t);
    }

    // output
    auto out_ge = hostApiCtx->GetOutputTensor(OUTPUT_Y_ID);
    OP_CHECK_IF(out_ge == nullptr, OP_LOGE(hostApiCtx->GetNodeName(), "out_ge is null"), return GRAPH_FAILED);

    // mask
    auto attrs = hostApiCtx->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(hostApiCtx->GetNodeName(), "attrs is null"), return GRAPH_FAILED);
    const gert::ContinuousVector* indicesMaskPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_MASK_IDX);
    const int64_t* ge_mask = reinterpret_cast<const int64_t*>(indicesMaskPtr->GetData());
    int64_t mask_num = indicesMaskPtr->GetSize();

    gert::Tensor emptyTensor({{0}, {0}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kFollowing, ge::DT_INT64, nullptr);

    std::vector<const gert::Tensor*> inputTensorList = getIndicesWithMask(ge_tenserListValue, ge_tenserListValue.size(),
                                                                          ge_mask, mask_num, &emptyTensor);

    auto api_ret = CANN_OPS_OPB_SYN_EXEC_ACLNN(hostApiCtx, aclnnIndex, self_ge, inputTensorList, out_ge);
    OP_CHECK_IF(api_ret != GRAPH_SUCCESS, OP_LOGE(hostApiCtx->GetNodeName(), "api_ret faild:%d", api_ret),
                return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP(IndexByTensor).OpExecuteFunc(IndexByTensorHostExecuteFunc);

} // namespace fallback

#ifdef __cplusplus
}
#endif
