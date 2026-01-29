/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "max_pool_v3.h"
#include "opdev/data_type_utils.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

namespace l0op {
OP_TYPE_REGISTER(MaxPoolV3);

// 1980仅支持float16
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_910_LIST = {op::DataType::DT_FLOAT16};
// 1971支持float16和float
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_910B_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_950_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16, op::DataType::DT_INT32, op::DataType::DT_INT64,
    op::DataType::DT_UINT8, op::DataType::DT_INT16, op::DataType::DT_INT8, op::DataType::DT_UINT16};

static const inline std::initializer_list<op::DataType> GetDtypeSupportListBySocVersion() {
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    switch (socVersion) {
        case SocVersion::ASCEND910_93:
        case SocVersion::ASCEND910B: {
            return DTYPE_SUPPORT_910B_LIST;
        }
        case SocVersion::ASCEND910: {
            return DTYPE_SUPPORT_910_LIST;
        }
        case SocVersion::ASCEND950: {
            return DTYPE_SUPPORT_950_LIST;
        }
        default: {
            return DTYPE_SUPPORT_910_LIST;
        }
    }
}

// 根据芯片类型、dtype判断算子是否支持走aicore
static inline bool IsAiCoreSupport(const aclTensor *self) {
    const std::initializer_list<op::DataType> dtypeSupportList = GetDtypeSupportListBySocVersion();
    if (!CheckType(self->GetDataType(), dtypeSupportList)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self dtype not supported. support list is %s",
                op::ToString(dtypeSupportList).GetString());
        return false;
    }

    return true;
}

// AICORE算子kernel
static const inline aclTensor *MaxPoolV3AiCore(const aclTensor *self, const aclIntArray *kernelShape4,
                                               const aclIntArray *strides4, const std::string &padsMode,
                                               const aclIntArray *pads4, const std::string& dataFormat,
                                               bool globalPooling, bool ceilModeBool, aclTensor *maxpoolv3Out,
                                               aclOpExecutor *executor) {
    L0_DFX(MaxPoolV3AiCore, self, kernelShape4, strides4, padsMode, pads4, dataFormat, globalPooling,
           ceilModeBool, maxpoolv3Out);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将MaxPoolV3算子加入任务队列
    // MaxPoolV3是算子的OpType，self是算子的输入，maxpoolv3Out是算子的输出
    ADD_TO_LAUNCHER_LIST_AICORE(MaxPoolV3,
                                OP_INPUT(self),
                                OP_OUTPUT(maxpoolv3Out),
                                OP_ATTR(kernelShape4, strides4, padsMode, pads4, dataFormat,
                                        globalPooling, ceilModeBool));

    return maxpoolv3Out;
}

const aclTensor *MaxPoolV3(const aclTensor *self, const aclIntArray *kernelShape4, const aclIntArray *strides4,
                           const aclIntArray *pads4, const std::string &dataFormat, const int64_t ceilMode,
                           aclOpExecutor *executor) {
    // 申请out tensor和indices tensor
    auto out = executor->AllocTensor(self->GetDataType(), self->GetStorageFormat(), self->GetOriginalFormat());

    const bool ceilModeBool = (ceilMode != 0);
    static const std::string padsMode = "CALCULATED";
    const bool globalPooling = false;

    auto ret = INFER_SHAPE(MaxPoolV3,
                           OP_INPUT(self),
                           OP_OUTPUT(out),
                           OP_ATTR(kernelShape4, strides4, padsMode, pads4, dataFormat,
                                   globalPooling, ceilModeBool));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return nullptr;
    }

    if (IsAiCoreSupport(self)) {
        // 调用Aicore算子计算
        return MaxPoolV3AiCore(self, kernelShape4, strides4, padsMode, pads4, dataFormat, globalPooling,
                               ceilModeBool, out, executor);
    }

    // 当前没有匹配的aicpu算子
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "No dtype not supported on AICPU");
    return nullptr;
}
} // namespace l0op

