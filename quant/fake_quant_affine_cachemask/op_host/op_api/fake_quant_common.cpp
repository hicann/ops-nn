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
 * \file fake_quant_common.cpp
 * \brief
 */

#include "fake_quant_common.h"

namespace FakeQuantCommon {

auto nullptrInner = std::tuple<aclTensor*, aclTensor*, aclTensor*>(nullptr, nullptr, nullptr);

std::tuple<const aclTensor*, const aclTensor*, const aclTensor*> GetContiguousInput(const aclTensor* self, const aclTensor* scale,
    const aclTensor* zeroPoint, aclOpExecutor* executor)
{
    auto selfContiguous = l0op::Contiguous(self, executor);
    OP_CHECK_NULL(selfContiguous, return nullptrInner);

    auto promoteType = op::PromoteType(self->GetDataType(), scale->GetDataType());
    // 将输入self的数据类型转换成隐式数据类型，根据具体算子语义按需调用
    auto selfCasted = l0op::Cast(selfContiguous, promoteType, executor);
    OP_CHECK_NULL(selfCasted, return nullptrInner);
    
    auto scaleContiguous = l0op::Contiguous(scale, executor);
    OP_CHECK_NULL(scaleContiguous, return nullptrInner);

    // 将输入other的数据类型转换成隐式数据类型，根据具体算子语义按需调用
    auto scaleCasted = l0op::Cast(scaleContiguous, promoteType, executor);
    OP_CHECK_NULL(scaleCasted, return nullptrInner);

    auto zeroPointContiguous = l0op::Contiguous(zeroPoint, executor);
    OP_CHECK_NULL(zeroPointContiguous, return nullptrInner);

    return std::tuple<const aclTensor*, const aclTensor*, const aclTensor*>(selfCasted, scaleCasted, zeroPointContiguous);
}

}
