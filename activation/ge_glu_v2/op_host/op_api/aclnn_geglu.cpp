/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_geglu.h"
#include "geglu_v2.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "op_api/op_api_def_nn.h"
#include "op_api/aclnn_util.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include <string>
#include <vector>
#include "base/err_msg.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<DataType> dtype_support_list = {op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};

static const std::initializer_list<DataType> dtype_support_list_with_bf16 = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201 || Ops::NN::AclnnUtil::IsRegbase()) {
        return dtype_support_list_with_bf16;
    }
    return dtype_support_list;
}

static bool CheckNotNull(const aclTensor* self, const aclTensor* out, const aclTensor* outGelu)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    OP_CHECK_NULL(outGelu, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out, const aclTensor* outGelu)
{
    auto supportList = GetDtypeSupportList();
    // 检查self的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

    // 检查out的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);

    // 检查outGelu的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(outGelu, supportList, return false);

    // 检查输入和输出的数据类型是否一致（空Tensor与rank-0不豁免dtype一致性校验，
    // 形状类约束仍由CheckShape与执行侧空Tensor早退处理）
    OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(outGelu, self->GetDataType(), return false);

    return true;
}

static bool CheckShape(const aclTensor* self, int64_t dim, const aclTensor* out, const aclTensor* outGelu)
{
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(out, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(outGelu, MAX_SUPPORT_DIMS_NUMS, return false);

    int64_t dimNum = self->GetViewShape().GetDimNum();
    if (dimNum == 0) {
        dimNum = 1;
    }

    if ((-dimNum > dim) || ((dimNum - 1) < dim)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Expected aclnnGeGlu dim value %ld to be in range [%ld, %ld] but check failed.", dim, -dimNum,
                dimNum - 1);
        return false;
    }

    auto sliceDim = dim < 0 ? dimNum + dim : dim;
    auto selfSliceDimSize = self->GetViewShape().GetDim(sliceDim);
    auto outSliceDimSize = out->GetViewShape().GetDim(sliceDim);
    auto outGeluSliceDimSize = outGelu->GetViewShape().GetDim(sliceDim);
    const int64_t SLICE_NUM = 2;
    if (!(selfSliceDimSize % SLICE_NUM == 0)) {
        const std::string reason = (("selfSliceDimSize[" + std::to_string(selfSliceDimSize) +
                                     "] must be even number."));
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s", reason.c_str());
        REPORT_PREDEFINED_ERR_MSG(
            "EZ0015", (std::vector<const char*>{"param_name", "op_name", "incorrect_size", "reason"}),
            (std::vector<const char*>{"self", "aclnnGeGlu", (std::to_string(selfSliceDimSize)).c_str(),
                                      reason.c_str()}));
        return false;
    };

    if (outSliceDimSize != selfSliceDimSize / SLICE_NUM) {
        const std::string reason = (("Expected aclnnGeGlu out slice dim size " + std::to_string(outSliceDimSize) +
                                     " to be equal to " + std::to_string(selfSliceDimSize / SLICE_NUM) +
                                     " but check failed."));
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s", reason.c_str());
        REPORT_PREDEFINED_ERR_MSG(
            "EZ0015", (std::vector<const char*>{"param_name", "op_name", "incorrect_size", "reason"}),
            (std::vector<const char*>{"out", "aclnnGeGlu", (std::to_string(outSliceDimSize)).c_str(), reason.c_str()}));
        return false;
    }

    if (outGeluSliceDimSize != selfSliceDimSize / SLICE_NUM) {
        const std::string reason = (("Expected aclnnGeGlu outGelu slice dim size " +
                                     std::to_string(outGeluSliceDimSize) + " to be equal to " +
                                     std::to_string(selfSliceDimSize / SLICE_NUM) + " but check failed."));
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s", reason.c_str());
        REPORT_PREDEFINED_ERR_MSG(
            "EZ0015", (std::vector<const char*>{"param_name", "op_name", "incorrect_size", "reason"}),
            (std::vector<const char*>{"outGelu", "aclnnGeGlu", (std::to_string(outGeluSliceDimSize)).c_str(),
                                      reason.c_str()}));
        return false;
    }

    int64_t outDimNum = out->GetViewShape().GetDimNum();
    int64_t outGeluDimNum = outGelu->GetViewShape().GetDimNum();

    OP_CHECK(outDimNum == outGeluDimNum && outDimNum == dimNum,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "outDimNum[%ld] and outGeluDimNum[%ld] must be the same with inputDimNum[%ld].", outDimNum,
                     outGeluDimNum, dimNum),
             return false);

    for (int64_t idx = 0; idx < dimNum; idx++) {
        if (idx == sliceDim) {
            continue;
        }
        if (!(self->GetViewShape().GetDim(idx) == out->GetViewShape().GetDim(idx) &&
              out->GetViewShape().GetDim(idx) == outGelu->GetViewShape().GetDim(idx))) {
            const std::string reason = ((
                "Except the slice dim[" + std::to_string(dim) +
                "], all other shape dim must be the same, but got input shape: " +
                std::string(op::ToString(self->GetViewShape()).GetString()) +
                ", out shape: " + std::string(op::ToString(out->GetViewShape()).GetString()) +
                ", out gelu shape: " + std::string(op::ToString(outGelu->GetViewShape()).GetString())));
            OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s", reason.c_str());
            REPORT_PREDEFINED_ERR_MSG(
                "EZ0010", (std::vector<const char*>{"param_names", "op_name", "incorrect_shapes", "reason"}),
                (std::vector<const char*>{"self,out,outGelu", "aclnnGeGlu",
                                          (std::string(op::ToString(self->GetViewShape()).GetString()) + "," +
                                           op::ToString(out->GetViewShape()).GetString() + "," +
                                           op::ToString(outGelu->GetViewShape()).GetString())
                                              .c_str(),
                                          reason.c_str()}));
            return false;
        };
    }
    return true;
}

static bool checkApproximate(int64_t approximate)
{
    const int64_t APPROXIMATE_NUM = 1;
    if (approximate < 0 || approximate > APPROXIMATE_NUM) {
        const std::string reason = (("Approximate should be 0 or 1, but current is " + std::to_string(approximate)));
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s", reason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ0026",
                                  (std::vector<const char*>{"param_name", "op_name", "incorrect_value", "reason"}),
                                  (std::vector<const char*>{"approximate", "aclnnGeGlu",
                                                            (std::to_string(approximate)).c_str(), reason.c_str()}));
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, int64_t dim, int64_t approximate, const aclTensor* out,
                               const aclTensor* outGelu)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, out, outGelu), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out, outGelu), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否满足约束
    CHECK_RET(CheckShape(self, dim, out, outGelu), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(checkApproximate(approximate), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus ExecGeGluGetWorkspaceSize(const aclTensor* self, int64_t dim, int64_t approximate, bool activateLeft,
                                      aclTensor* out, aclTensor* outGelu, uint64_t* workspaceSize,
                                      aclOpExecutor** executor)
{
    // 固定写法，参数检查
    auto ret = CheckParams(self, dim, approximate, out, outGelu);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空Tensor处理
    if (self->IsEmpty() || self->GetViewShape().GetDimNum() == 0) {
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // self如果非连续，需要转连续
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0算子GeGlu行计算
    auto geGluResult = l0op::GeGluV2(selfContiguous, dim, approximate, activateLeft, uniqueExecutor.get());
    const aclTensor* geGluOutResult = std::get<0>(geGluResult);
    const aclTensor* geluOutResult = std::get<1>(geGluResult);
    CHECK_RET(geGluOutResult != nullptr && geluOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyOutResult = l0op::ViewCopy(geGluOutResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyGeluOutResult = l0op::ViewCopy(geluOutResult, outGelu, uniqueExecutor.get());
    CHECK_RET(viewCopyGeluOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGeGluGetWorkspaceSize(const aclTensor* self, int64_t dim, int64_t approximate, aclTensor* out,
                                       aclTensor* outGelu, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnGeGlu, DFX_IN(self, dim, approximate), DFX_OUT(out, outGelu));
    return ExecGeGluGetWorkspaceSize(self, dim, approximate, false, out, outGelu, workspaceSize, executor);
}

aclnnStatus aclnnGeGluV3GetWorkspaceSize(const aclTensor* self, int64_t dim, int64_t approximate, bool activateLeft,
                                         aclTensor* out, aclTensor* outGelu, uint64_t* workspaceSize,
                                         aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnGeGluV3, DFX_IN(self, dim, approximate, activateLeft), DFX_OUT(out, outGelu));
    return ExecGeGluGetWorkspaceSize(self, dim, approximate, activateLeft, out, outGelu, workspaceSize, executor);
}

aclnnStatus aclnnGeGlu(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnGeGlu);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnGeGluV3(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnGeGluV3);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
