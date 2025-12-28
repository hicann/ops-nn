/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL0_GROUPED_DYNAMIC_MX_QUANT_QUANT_H_
#define OP_API_INC_LEVEL0_GROUPED_DYNAMIC_MX_QUANT_QUANT_H_

#include <string>
#include "opdev/op_executor.h"

namespace l0op {
std::tuple<aclTensor*, aclTensor*> GroupedDynamicMxQuant(const aclTensor *x, const aclTensor *groupIndex, 
                                                         const char *roundMode, int64_t dstType, 
                                                         int64_t blocksize, aclOpExecutor *executor);
} // l0op

#endif // OP_API_INC_LEVEL0_GROUP_QUANT_H_
