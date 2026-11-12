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
 * \file index_fill.cpp
 * \brief
 */

#include "index_fill.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "op_api/aclnn_util.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(IndexFillD);

// 框架的 OP_TYPE_REGISTER 除生成 <OpType>OpTypeId() 外，还会定义一个全局的
// <OpType>_kernelName_Be_Defined_Multi_Times__ 变量用于发现重复注册。IndexFill / InplaceIndexFill 已由
// experimental/index/index_fill 注册，而本文件的两个 *ByTensor 接口仍按历史行为下发这两个 OpType，
// 若此处再次使用 OP_TYPE_REGISTER，experimental 全量编译时会与 index_fill 重复定义上述全局变量
// （multiple definition of 'l0op::IndexFill_kernelName_Be_Defined_Multi_Times__'）。
// 这里只复用其 OpType 注册逻辑（资源表为 TU 内链接，不产生重复符号），不再定义保护变量。
#ifdef ACLNN_WITH_BINARY
#define OPS_NN_REUSE_OP_TYPE_ID(kernelName)                                                          \
    OP_RESOURCE(kernelName);                                                                         \
    [[maybe_unused]] inline uint32_t kernelName##OpTypeId()                                          \
    {                                                                                                \
        op::GenInternalOpTypeId();                                                                   \
        static uint32_t kernelName##OpTypeId = op::GenOpTypeId(#kernelName, kernelName##_RESOURCES); \
        return kernelName##OpTypeId;                                                                 \
    }
#else
#define OPS_NN_REUSE_OP_TYPE_ID(kernelName)                                  \
    [[maybe_unused]] inline uint32_t kernelName##OpTypeId()                  \
    {                                                                        \
        op::GenInternalOpTypeId();                                           \
        static uint32_t kernelName##OpTypeId = op::GenOpTypeId(#kernelName); \
        return kernelName##OpTypeId;                                         \
    }
#endif

OPS_NN_REUSE_OP_TYPE_ID(IndexFill);
OPS_NN_REUSE_OP_TYPE_ID(InplaceIndexFill);
#undef OPS_NN_REUSE_OP_TYPE_ID

// AICORE
static const std::initializer_list<op::DataType> NULL_SUPPORT_LIST = {};

static const std::initializer_list<op::DataType> AICORE_DTYPE_910B_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BOOL,
    op::DataType::DT_BF16};

// 根据dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor* self)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_2201) {
        return CheckType(self->GetDataType(), AICORE_DTYPE_910B_SUPPORT_LIST);
    } else {
        return CheckType(self->GetDataType(), NULL_SUPPORT_LIST);
    }
}

// AICORE算子kernel
static const aclTensor* IndexFillDAiCore(const aclTensor* self, const aclTensor* assist1, const aclTensor* assist2,
                                         int64_t dim, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(IndexFillDAiCore, self, assist1, assist2);
    // 使用框架宏 ADD_TO_LAUNCHER_LIST_AICORE，将indexfilld算子加入任务队列
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(IndexFillD, OP_INPUT(self, assist1, assist2), OP_OUTPUT(out), OP_ATTR(dim));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "IndexFillDAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
    return out;
}

const aclTensor* IndexFillD(const aclTensor* self, const aclTensor* assist1, const aclTensor* assist2, int64_t dim,
                            aclOpExecutor* executor)
{
    auto out = executor->AllocTensor(self->GetViewShape(), self->GetDataType());
    if (IsAiCoreSupport(self)) {
        return IndexFillDAiCore(self, assist1, assist2, dim, out, executor);
    } else {
        return nullptr;
    }
}

const aclTensor* IndexFillDByTensor(const aclTensor* self, const aclTensor* indices, const aclTensor* value,
                                    int64_t dim, aclOpExecutor* executor)
{
    auto out = executor->AllocTensor(self->GetViewShape(), self->GetDataType());
    L0_DFX(IndexFillDByTensor, self, indices, value);
    // 使用框架宏 ADD_TO_LAUNCHER_LIST_AICORE，将indexfill算子加入任务队列
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(IndexFill, OP_INPUT(self, indices, value), OP_OUTPUT(out), OP_ATTR(dim));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "IndexFillDByTensor ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return nullptr);
    return out;
}

const aclTensor* InplaceIndexFillDByTensor(const aclTensor* self, const aclTensor* indices, const aclTensor* value,
                                           int64_t dim, aclOpExecutor* executor)
{
    L0_DFX(InplaceIndexFillDByTensor, self, indices, value);
    // 使用框架宏 ADD_TO_LAUNCHER_LIST_AICORE，将inplaceindexfill算子加入任务队列
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(InplaceIndexFill, OP_INPUT(self, indices, value), OP_OUTPUT(self),
                                           OP_ATTR(dim));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "InplaceIndexFillDByTensor ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return nullptr);
    return self;
}

} // namespace l0op
