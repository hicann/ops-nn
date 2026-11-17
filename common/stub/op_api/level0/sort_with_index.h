/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_SORT_WITH_INDEX_OP_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_SORT_WITH_INDEX_OP_H_

#include "opdev/op_executor.h"
#include "opdev/fast_vector.h"

namespace l0op {
// 旧版 math 可能只有 A5 同名接口，缺少 A2/A3 能力查询时走旧路径。
__attribute__((weak)) bool IsSortWithIndexA2A3Supported();

// 签名与 ops-math 保持一致，弱引用允许旧版 math 未导出该接口时回退。
__attribute__((weak)) const std::tuple<aclTensor*, aclTensor*> SortWithIndex(const aclTensor* self,
                                                                             const aclTensor* index, int64_t axis,
                                                                             bool descending, bool stable,
                                                                             aclOpExecutor* executor);
} // namespace l0op

#endif // PTA_NPU_OP_API_INC_LEVEL0_OP_SORT_WITH_INDEX_OP_H_
