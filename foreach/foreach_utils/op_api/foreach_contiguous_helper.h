/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FOREACH_CONTIGUOUS_HELPER_H_
#define FOREACH_CONTIGUOUS_HELPER_H_

#include <vector>
#include "aclnn_kernels/contiguous.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"

/*!
 * \brief 构造连续 tensorList：空 tensor 或已连续且 storage 与 view 一致则直接用原 tensor，否则
 *        l0op::Contiguous 转连续。保持与输入 list 索引/数量一一对应（空 tensor 不跳过，保留原 tensor），
 *        避免 kernel 端索引错位。失败返回 nullptr。
 *        注意：op::IsContiguous 仅判断 view strides 连续性，不感知 storage_shape / view_offset；
 *        参数融合场景下 view 连续但 storage 为共享大 buffer（storage_shape != view_shape），且
 *        Contiguous 后 storage_shape 会变为 view_shape 维度，多输入场景下若仅部分 input 被 Contiguous
 *        会导致 storage_shape 维度不一致触发 tiling 校验失败。故需额外比较 storage_shape == view_shape，
 *        确保 storage 与 view 的 shape 完全一致（含维度）才跳过。
 */
inline const aclTensorList* ForeachMakeContiguousTensorList(const aclTensorList* tensorList, aclOpExecutor* executor)
{
    std::vector<const aclTensor*> vec;
    for (uint64_t i = 0; i < tensorList->Size(); i++) {
        auto tensor = (*tensorList)[i];
        if (tensor->IsEmpty() || (op::IsContiguous(tensor) && tensor->GetStorageShape() == tensor->GetViewShape())) {
            vec.push_back(tensor);
        } else {
            auto cont = l0op::Contiguous(tensor, executor);
            if (cont == nullptr) {
                return nullptr;
            }
            vec.push_back(cont);
        }
    }
    return executor->AllocTensorList(vec.data(), vec.size());
}

/*!
 * \brief 将连续计算结果 ViewCopy 回写到（可能非连续的）输出 list。
 *        仅当 ForeachMakeContiguousTensorList 对该 out 做了 Contiguous（即 out 非空且不满足
 *        "连续且 storage_shape==view_shape"）时才需要回写；空 tensor 或已连续且 storage 与 view 一致的
 *        out 跳过（空无数据可拷；连续且 storage 一致时 kernel 已直接写入 out 内存）。
 *        要求 contOut 与 out 索引一一对应。失败返回 false。
 */
inline bool ForeachViewCopyToOutputTensorList(const aclTensorList* contOut, const aclTensorList* out,
                                              aclOpExecutor* executor)
{
    for (uint64_t i = 0; i < out->Size(); i++) {
        auto outTensor = (*out)[i];
        if (!outTensor->IsEmpty() &&
            !(op::IsContiguous(outTensor) && outTensor->GetStorageShape() == outTensor->GetViewShape())) {
            if (l0op::ViewCopy((*contOut)[i], outTensor, executor) == nullptr) {
                return false;
            }
        }
    }
    return true;
}

#endif // FOREACH_CONTIGUOUS_HELPER_H_
