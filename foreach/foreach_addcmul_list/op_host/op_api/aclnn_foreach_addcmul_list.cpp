/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_foreach_addcmul_list.h"
#include "foreach_addcmul_list.h"
#include "../../foreach_utils/op_host/foreach_contiguous_helper.h"
#include "aclnn_kernels/contiguous.h"
#include "op_api/op_api_def_nn.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "op_api/aclnn_util.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include "../../foreach_utils/op_host/foreach_null_check_helper.h"
#include <string>
#include <vector>
#include "base/err_msg.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static inline aclnnStatus CheckParams(const aclTensorList* x1, const aclTensorList* x2, const aclTensorList* x3,
                                      const aclTensor* scalars, const aclTensorList* out)
{
    // 与同族手写算子（aclnn_foreach_pow_scalar_list）口径保持一致。
    OP_CHECK_NULL(x1, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(x2, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(x3, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(scalars, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(out, return ACLNN_ERR_PARAM_NULLPTR);

    // 长度一致性必须先于任何按下标解引用（aclTensorList::operator[] 无边界检查），
    // 否则 x2/x3/out 比 x1 短时下方按 x1->Size() 遍历会越界读。
    CHECK_RET(x1->Size() == x2->Size() && x1->Size() == x3->Size() && x1->Size() == out->Size(),
              ACLNN_ERR_PARAM_INVALID);

    // 全空 TensorList（长度一致且均为 0）是合法 no-op：本函数下方及调用方后续逻辑都会访问
    // (*x1)[0]，必须在此短路返回成功，由调用方落零 workspace。与 add_list/sub_list 一致。
    if (x1->Size() == 0) {
        return ACLNN_SUCCESS;
    }
    // out 的 Tensor 个数须与 x1 严格一致：算子 infershape 要求输出实例数等于第一个动态输入实例数
    // （foreach_infershape.cpp），且 kernel 只按 x1 的成员数写入，多出的 out 无对应计算结果。
    if (out->Size() != x1->Size()) {
        const std::string reason = ("The out tensorlist must contain the same number of tensors as the x1 tensorlist, "
                                    "but got out=" +
                                    std::to_string(out->Size()) + " and x1=" + std::to_string(x1->Size()) + ".");
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s", reason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ0038",
                                  (std::vector<const char*>{"param_name", "op_name", "incorrect_size", "reason"}),
                                  (std::vector<const char*>{"out", "aclnnForeachAddcmulList",
                                                            (std::to_string(out->Size())).c_str(), reason.c_str()}));
        return ACLNN_ERR_PARAM_INVALID;
    }

    // scalars 的元素个数需等于 x1 的 Tensor 个数（逐位配对）
    if (scalars->GetViewShape().GetShapeSize() != static_cast<int64_t>(x1->Size())) {
        const std::string reason = (std::string("The element count of scalars must be equal to the number of tensors "
                                                "in x1; scalars=") +
                                    std::to_string(scalars->GetViewShape().GetShapeSize()) +
                                    ", x1=" + std::to_string(x1->Size()) + ".");
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s", reason.c_str());
        REPORT_PREDEFINED_ERR_MSG(
            "EZ0038", (std::vector<const char*>{"param_name", "op_name", "incorrect_size", "reason"}),
            (std::vector<const char*>{"scalars", "aclnnForeachAddcmulList",
                                      (std::to_string(scalars->GetViewShape().GetShapeSize())).c_str(),
                                      reason.c_str()}));
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 逐成员非空/可表示性检查
    CHECK_RET(CheckTensorListNotNull(x1, "x1"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorListNotNull(x2, "x2"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorListNotNull(x3, "x3"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorListNotNull(out, "out"), ACLNN_ERR_PARAM_INVALID);

    const std::initializer_list<DataType> tensorTypes = {DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16,
                                                         DataType::DT_INT32};
    OP_CHECK_DTYPE_NOT_SUPPORT((*x1)[0], tensorTypes, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH((*out)[0], (*x1)[0]->GetDataType(), return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH(scalars, (*x1)[0]->GetDataType(), return ACLNN_ERR_PARAM_INVALID);

    for (uint64_t i = 0; i < x1->Size(); ++i) {
        OP_CHECK_DTYPE_NOT_MATCH((*x1)[i], (*x1)[0]->GetDataType(), return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_DTYPE_NOT_MATCH((*x2)[i], (*x1)[0]->GetDataType(), return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_DTYPE_NOT_MATCH((*x3)[i], (*x1)[0]->GetDataType(), return ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(!IsPrivateFormat((*x1)[i]->GetStorageFormat()), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(!IsPrivateFormat((*x2)[i]->GetStorageFormat()), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(!IsPrivateFormat((*x3)[i]->GetStorageFormat()), ACLNN_ERR_PARAM_INVALID);
    }
    for (uint64_t i = 0; i < out->Size(); ++i) {
        OP_CHECK_DTYPE_NOT_MATCH((*out)[i], (*x1)[0]->GetDataType(), return ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(!IsPrivateFormat((*out)[i]->GetStorageFormat()), ACLNN_ERR_PARAM_INVALID);
    }

    // x2[i]/x3[i]/out[i] 的 shape size 必须与 x1[i] 完全相等：
    //  - kernel 按 x1[i] 的元素数读取 x2/x3 并写入 out；
    //  - tiling 要求各输入 list 成员 storage shape 完全一致；
    //  - 末尾 ForeachViewCopyToOutputTensorList 按 out 的完整形状回写连续化 buffer，
    //    若 out 比 x1 大且为非连续视图，未写入的尾部会被未初始化数据刷入用户 out。
    // 故此处统一按 shape size 严格相等校验，使错误在 L2 即返回 161002。
    for (uint64_t i = 0; i < x1->Size(); ++i) {
        const int64_t x1Size = (*x1)[i]->GetViewShape().GetShapeSize();
        const int64_t x2Size = (*x2)[i]->GetViewShape().GetShapeSize();
        const int64_t x3Size = (*x3)[i]->GetViewShape().GetShapeSize();
        const int64_t outSize = (*out)[i]->GetViewShape().GetShapeSize();
        if (x2Size != x1Size || x3Size != x1Size || outSize != x1Size) {
            const std::string reason = ("The shape size of the " + std::to_string(i) +
                                        "th tensor must be equal across x1, x2, x3 and out, but got x1=" +
                                        std::to_string(x1Size) + ", x2=" + std::to_string(x2Size) +
                                        ", x3=" + std::to_string(x3Size) + ", out=" + std::to_string(outSize) + ".");
            OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s", reason.c_str());
            const char* badParam = (x2Size != x1Size) ? "x2" : ((x3Size != x1Size) ? "x3" : "out");
            const int64_t badSize = (x2Size != x1Size) ? x2Size : ((x3Size != x1Size) ? x3Size : outSize);
            REPORT_PREDEFINED_ERR_MSG("EZ0015",
                                      (std::vector<const char*>{"param_name", "op_name", "incorrect_size", "reason"}),
                                      (std::vector<const char*>{badParam, "aclnnForeachAddcmulList",
                                                                std::to_string(badSize).c_str(), reason.c_str()}));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus ExecForeachAddcmulListGetWorkspaceSize(const aclTensorList* x1, const aclTensorList* x2,
                                                          const aclTensorList* x3, const aclTensor* scalars,
                                                          const aclTensorList* out, uint64_t* workspaceSize,
                                                          aclOpExecutor** executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x1, x2, x3, scalars, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空列表快路径：CheckParams 已保证四列表长度一致，此处 x1 为空即全空（合法 no-op）。
    if (x1->Size() == 0) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 输入如果非连续，需要转连续（空tensor直接用，保持索引一致）
    auto contiguousTensorsX1 = ForeachMakeContiguousTensorList(x1, uniqueExecutor.get());
    CHECK_RET(contiguousTensorsX1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto contiguousTensorsX2 = ForeachMakeContiguousTensorList(x2, uniqueExecutor.get());
    CHECK_RET(contiguousTensorsX2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto contiguousTensorsX3 = ForeachMakeContiguousTensorList(x3, uniqueExecutor.get());
    CHECK_RET(contiguousTensorsX3 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 输出如果非连续，需要转连续作为kernel输出buffer；连续/空则直接使用
    auto contiguousOut = ForeachMakeContiguousTensorList(out, uniqueExecutor.get());
    CHECK_RET(contiguousOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // scalars按张量下标平坦读取，stride≠1或共享storage的view会取错标量；
    // 与x1/x2/x3/out一致做Contiguous规整（对storage_shape != view_shape的
    // view生成独立规范tensor），连续且非视图时直接透传。
    const aclTensor* contiguousScalars = scalars;
    if (!(op::IsContiguous(scalars) && scalars->GetStorageShape() == scalars->GetViewShape())) {
        contiguousScalars = l0op::Contiguous(scalars, uniqueExecutor.get());
        CHECK_RET(contiguousScalars != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 调用l0算子ForeachAddcmulList进行计算，输出到连续buffer
    auto result = l0op::ForeachAddcmulList(contiguousTensorsX1, contiguousTensorsX2, contiguousTensorsX3,
                                           contiguousScalars, contiguousOut, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将连续计算结果拷贝到输出out上，out可能是非连续的tensor（空/连续跳过）
    CHECK_RET(ForeachViewCopyToOutputTensorList(contiguousOut, out, uniqueExecutor.get()), ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnForeachAddcmulListGetWorkspaceSize(const aclTensorList* x1, const aclTensorList* x2,
                                                    const aclTensorList* x3, const aclTensor* scalars,
                                                    const aclTensorList* out, uint64_t* workspaceSize,
                                                    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnForeachAddcmulList, DFX_IN(x1, x2, x3, scalars), DFX_OUT(out));
    return ExecForeachAddcmulListGetWorkspaceSize(x1, x2, x3, scalars, out, workspaceSize, executor);
}

aclnnStatus aclnnForeachAddcmulList(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnForeachAddcmulList);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
