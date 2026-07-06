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
 * \file index_check.cpp
 * \brief IndexCheck L0 interface implementation
 */

#include "index_check.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "op_api/aclnn_util.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(IndexCheck);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT64,
                                                                              op::DataType::DT_INT32};

static bool IsAiCoreSupport(const aclTensorList* indices)
{
    for (size_t i = 0; i < indices->Size(); i++) {
        if (!CheckType((*indices)[i]->GetDataType(), AICORE_DTYPE_SUPPORT_LIST)) {
            OP_LOGW("Tensor indices not implemented for %s, should be in dtype support list %s.",
                    op::ToString((*indices)[i]->GetDataType()).GetString(),
                    op::ToString(AICORE_DTYPE_SUPPORT_LIST).GetString());
            return false;
        }
    }
    return true;
}

void IndexCheck(const aclTensor* bounds, const aclTensorList* indices, aclOpExecutor* executor)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion != SocVersion::ASCEND910B && socVersion != SocVersion::ASCEND910_93) {
        OP_LOGD("IndexCheck only support ASCEND910B and ASCEND910_93, skip.");
        return;
    }

    if (!IsAiCoreSupport(indices)) {
        OP_LOGD("Unsupported dtype of indices in IndexCheck, skip.");
        return;
    }

    if (indices->Size() == 0) {
        OP_LOGD("available indices is empty, skip IndexCheck.");
        return;
    }

    if (static_cast<int64_t>(bounds->Size()) != static_cast<int64_t>(indices->Size())) {
        OP_LOGW("bounds size %d not equal indices size %d, skip IndexCheck.", bounds->Size(), indices->Size());
        return;
    }

    L0_DFX(IndexCheck, bounds, indices);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(IndexCheck, OP_INPUT(bounds, indices));

    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "IndexCheckAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return );
}

} // namespace l0op