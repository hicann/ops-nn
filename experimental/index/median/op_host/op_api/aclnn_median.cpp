/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_median.h"
#include "median.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_util.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16,  DataType::DT_UINT8,
    DataType::DT_INT8,  DataType::DT_INT16,   DataType::DT_INT32, DataType::DT_INT64};

static aclnnStatus GetWorkspaceSizeCommon(const aclTensor* self, int64_t dim, bool keepDim, aclTensor* valuesOut,
                                          aclTensor* indicesOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    OP_CHECK_NULL(self, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(valuesOut, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH(valuesOut, self->GetDataType(), return ACLNN_ERR_PARAM_INVALID);

    // indicesOut == nullptr → 全局 median（aclnnMedian 语义：拍平所有元素求中位，只出 value，无 index）
    bool isGlobal = (indicesOut == nullptr);
    if (!isGlobal && indicesOut->GetDataType() != DataType::DT_INT64 &&
        indicesOut->GetDataType() != DataType::DT_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "indicesOut dtype should be INT32 or INT64.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (valuesOut->IsEmpty() || (!isGlobal && indicesOut->IsEmpty())) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 1. Contiguous
    auto selfContig = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContig != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // ===== 全局 median：拍平成 1D → L0 Median(dim=0) → 只 ViewCopy value（index 弃用）=====
    if (isGlobal) {
        int64_t numel = static_cast<int64_t>(selfContig->GetViewShape().GetShapeSize());
        if (numel == 0) { // 空张量：对齐 PyTorch（torch.median(empty) 抛 RuntimeError），返回参数非法
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Median does not support empty tensor.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        int64_t fs[1] = {numel};
        auto flat = l0op::Reshape(selfContig, uniqueExecutor.get()->AllocIntArray(fs, 1), uniqueExecutor.get());
        CHECK_RET(flat != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // 全局 median 仅需 values；L0 Median 内部也会分配 indices（该路径弃用），属已知的小额 device 内存开销。
        auto gMed = l0op::Median(flat, 0, false, uniqueExecutor.get());
        auto gValues = std::get<0>(gMed);
        CHECK_RET(gValues != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto gCopy = l0op::ViewCopy(gValues, valuesOut, uniqueExecutor.get());
        CHECK_RET(gCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    int64_t dimNum = static_cast<int64_t>(selfContig->GetViewShape().GetDimNum());
    if (dimNum == 0) { // 标量：reshape→[1]
        int64_t one[1] = {1};
        auto shapeOne = uniqueExecutor.get()->AllocIntArray(one, 1);
        selfContig = l0op::Reshape(selfContig, shapeOne, uniqueExecutor.get());
        CHECK_RET(selfContig != nullptr, ACLNN_ERR_INNER_NULLPTR);
        dimNum = 1;
    }

    int64_t realDim = (dim % dimNum + dimNum) % dimNum;
    int64_t lastAxis = dimNum - 1;

    // 2. 若 dim 非末轴，Transpose 把 dim 换到末轴
    const aclTensor* transposed = selfContig;
    std::vector<int64_t> permIn(dimNum);
    for (int64_t i = 0; i < dimNum; ++i)
        permIn[i] = i;
    if (realDim != lastAxis) {
        std::swap(permIn[realDim], permIn[lastAxis]);
        auto permInArray = uniqueExecutor.get()->AllocIntArray(permIn.data(), dimNum);
        transposed = l0op::Transpose(selfContig, permInArray, uniqueExecutor.get());
        CHECK_RET(transposed != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 3. 调用 L0 Median，固定 dim=末轴, keepDim=false
    auto medResult = l0op::Median(transposed, lastAxis, false, uniqueExecutor.get());
    auto values = std::get<0>(medResult);
    auto indices = std::get<1>(medResult);
    CHECK_RET(values != nullptr && indices != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 4. 把输出从 transposed-去末轴 还原到 原 shape-去 realDim
    if (realDim != lastAxis) {
        // 输出 shape = permIn[:-1] 对应原始 axis 顺序
        // 计算 permOut[j] = i 使得 permIn[i] == 我们要的第 j 个原始 axis
        std::vector<int64_t> permOut(dimNum - 1);
        int64_t j = 0;
        for (int64_t origAxis = 0; origAxis < dimNum; ++origAxis) {
            if (origAxis == realDim)
                continue;
            for (int64_t i = 0; i < dimNum - 1; ++i) {
                if (permIn[i] == origAxis) {
                    permOut[j++] = i;
                    break;
                }
            }
        }
        // 仅当 permOut 不是 identity 时才调 Transpose（identity 跳过省 2 次 ~24µs 开销）
        bool isIdentity = true;
        for (int64_t i = 0; i < dimNum - 1; ++i) {
            if (permOut[i] != i) {
                isIdentity = false;
                break;
            }
        }
        if (!isIdentity) {
            auto permOutArray = uniqueExecutor.get()->AllocIntArray(permOut.data(), dimNum - 1);
            values = l0op::Transpose(values, permOutArray, uniqueExecutor.get());
            indices = l0op::Transpose(indices, permOutArray, uniqueExecutor.get());
            CHECK_RET(values != nullptr && indices != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }

    // 5. keepDim：在 realDim 位置插入 size=1
    if (keepDim) {
        std::vector<int64_t> kdShape;
        auto srcShape = selfContig->GetViewShape();
        for (int64_t i = 0; i < dimNum; ++i) {
            kdShape.push_back(i == realDim ? 1 : srcShape.GetDim(i));
        }
        auto kdArray = uniqueExecutor.get()->AllocIntArray(kdShape.data(), dimNum);
        values = l0op::Reshape(values, kdArray, uniqueExecutor.get());
        indices = l0op::Reshape(indices, kdArray, uniqueExecutor.get());
        CHECK_RET(values != nullptr && indices != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 6. Cast indices → indicesOut.dtype (我们 L0 出 int32，用户可能要 int64)
    auto castedIndices = indices;
    if (indices->GetDataType() != indicesOut->GetDataType()) {
        castedIndices = l0op::Cast(indices, indicesOut->GetDataType(), uniqueExecutor.get());
        CHECK_RET(castedIndices != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 7. ViewCopy 到输出
    auto vCopy = l0op::ViewCopy(values, valuesOut, uniqueExecutor.get());
    auto iCopy = l0op::ViewCopy(castedIndices, indicesOut, uniqueExecutor.get());
    CHECK_RET(vCopy != nullptr && iCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMedianGetWorkspaceSize(const aclTensor* self, int64_t dim, bool keepDim, aclTensor* valuesOut,
                                        aclTensor* indicesOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnMedian, DFX_IN(self, dim, keepDim), DFX_OUT(valuesOut, indicesOut));
    return GetWorkspaceSizeCommon(self, dim, keepDim, valuesOut, indicesOut, workspaceSize, executor);
}

aclnnStatus aclnnMedian(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMedian);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
