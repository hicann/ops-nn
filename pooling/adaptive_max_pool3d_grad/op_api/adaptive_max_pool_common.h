/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/platform.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/framework_op.h"

namespace AdaptiveMaxPoolCommon {
bool CheckNotNullPtr(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, aclTensor* gradInput);
const std::initializer_list<op::DataType> GetDtypeSupportListBySocVersion();
bool CheckShapeSame(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, const aclTensor* gradInput);
bool CheckDtypeValid(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, const aclTensor* gradInput);
aclnnStatus CheckParams(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, aclTensor* gradInput);
const aclTensor* selectLevelZeroOperation(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, aclTensor* /* gradInput */,
    aclOpExecutor* executor);
}
