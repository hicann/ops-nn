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

#ifndef FAKE_QUANT_COMMON_H_
#define FAKE_QUANT_COMMON_H_

#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/cast.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

namespace FakeQuantCommon {

std::tuple<const aclTensor*, const aclTensor*, const aclTensor*> GetContiguousInput(const aclTensor* self,
                                                                                    const aclTensor* scale,
                                                                                    const aclTensor* zeroPoint,
                                                                                    aclOpExecutor* executor);

inline aclnnStatus FinalizeOutput(aclTensor* fakeQuantOut, aclTensor* fakeQuantMask, const aclTensor* out,
                                  const aclTensor* mask, uint64_t* workspaceSize, aclOpExecutor** executor,
                                  UniqueExecutor& uniqueExecutor)
{
    CHECK_RET(fakeQuantOut != nullptr && fakeQuantMask != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto fakeCastOut = l0op::Cast(fakeQuantOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(fakeCastOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto fakeViewCopyResult = l0op::ViewCopy(fakeCastOut, out, uniqueExecutor.get());
    CHECK_RET(fakeViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto maskViewCopyResult = l0op::ViewCopy(fakeQuantMask, mask, uniqueExecutor.get());
    CHECK_RET(maskViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

} // namespace FakeQuantCommon

#endif