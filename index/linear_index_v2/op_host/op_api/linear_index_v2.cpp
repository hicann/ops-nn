/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file linear_index_v2.cpp
 * \brief
 */
#include "linear_index_v2.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"

using namespace op;

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_INT64};

namespace l0op {
OP_TYPE_REGISTER(LinearIndexV2);

const aclTensor* LinearIndexV2(
    const aclTensorList* indicesList, const aclTensor* stride, const aclTensor* valueSize, aclOpExecutor* executor)
{
    L0_DFX(LinearIndexV2, indicesList, stride, valueSize);

    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion != SocVersion::ASCEND910B && socVersion != SocVersion::ASCEND910_93 &&
        socVersion != SocVersion::ASCEND910_95) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "[LinearIndexV2] only support ASCEND910B and ASCEND910_95");
        return nullptr;
    }

    if (indicesList->Size() == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "[LinearIndexV2] indicesList size must larger than 0");
        return nullptr;
    }

    if (!CheckType((*indicesList)[0]->GetDataType(), AICORE_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "[LinearIndexV2] indices must be in int32, int64");
        return nullptr;
    }
    // need find first unEmpty tensor's shape
    auto outShape = (*indicesList)[0]->GetViewShape();
    int64_t outputSize = 1;
    int64_t validIdx = 0;
    for (int i = 0; i < static_cast<int>(indicesList->Size()); i++) {
        int64_t size = 1;
        for (size_t j = 0; j < (*indicesList)[i]->GetViewShape().GetDimNum(); j++) {
            size *= (*indicesList)[i]->GetViewShape().GetDim(j);
        }
        if (size != 0) {
            outputSize = size;
            validIdx = i;
            break;
        }
    }
    outShape.SetDimNum(1);
    outShape.SetDim(0, outputSize);
    if (socVersion == SocVersion::ASCEND910_95) {
        auto index = executor->AllocTensor(
            outShape, (*indicesList)[validIdx]->GetDataType(), (*indicesList)[validIdx]->GetViewFormat());
        auto ret =
            ADD_TO_LAUNCHER_LIST_AICORE(LinearIndexV2, OP_INPUT(indicesList, stride, valueSize), OP_OUTPUT(index));
        OP_CHECK(
            ret == ACLNN_SUCCESS,
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LinearIndexV2AiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
            return nullptr);
        return index;
    } else {
        auto index = executor->AllocTensor(outShape, op::DataType::DT_INT32, (*indicesList)[0]->GetViewFormat());
        auto ret =
            ADD_TO_LAUNCHER_LIST_AICORE(LinearIndexV2, OP_INPUT(indicesList, stride, valueSize), OP_OUTPUT(index));
        OP_CHECK(
            ret == ACLNN_SUCCESS,
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LinearIndexV2AiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
            return nullptr);
        return index;
    }
}
} // namespace l0op
