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
 * \brief 构造连续 tensorList：空 tensor 或已连续则直接用原 tensor，非空非连续则 l0op::Contiguous 转连续。
 *        保持与输入 list 索引/数量一一对应（空 tensor 不跳过，保留原 tensor），避免 kernel 端索引错位。
 *        失败返回 nullptr。
 */
inline const aclTensorList* ForeachMakeContiguousTensorList(const aclTensorList* tensorList, aclOpExecutor* executor)
{
    std::vector<const aclTensor*> vec;
    for (uint64_t i = 0; i < tensorList->Size(); i++) {
        if ((*tensorList)[i]->IsEmpty() || op::IsContiguous((*tensorList)[i])) {
            vec.push_back((*tensorList)[i]);
        } else {
            auto cont = l0op::Contiguous((*tensorList)[i], executor);
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
 *        空 tensor 或已连续的 out 跳过（空无数据可拷；连续时 kernel 已直接写入 out 内存）。
 *        要求 contOut 与 out 索引一一对应。失败返回 false。
 */
inline bool ForeachViewCopyToOutputTensorList(const aclTensorList* contOut, const aclTensorList* out,
                                              aclOpExecutor* executor)
{
    for (uint64_t i = 0; i < out->Size(); i++) {
        if (!(*out)[i]->IsEmpty() && !op::IsContiguous((*out)[i])) {
            if (l0op::ViewCopy((*contOut)[i], (*out)[i], executor) == nullptr) {
                return false;
            }
        }
    }
    return true;
}

#endif // FOREACH_CONTIGUOUS_HELPER_H_
