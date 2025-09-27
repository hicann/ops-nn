/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apply_top_k_top_p_with_sorted.h
 * \brief
 */
#ifndef OP_API_INC_LEVEL0_OP_APPLY_TOP_K_TOP_P_WITH_SORTED_OP_H_
#define OP_API_INC_LEVEL0_OP_APPLY_TOP_K_TOP_P_WITH_SORTED_OP_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* ApplyTopKTopPWithSorted(const aclTensor* sortedValue, const aclTensor* sortedIndices,
                                         const aclTensor* p, const aclTensor* k, aclOpExecutor* executor);
}
#endif // OP_API_INC_LEVEL0_OP_APPLY_TOP_K_TOP_P_WITH_SORTED_OP_H_

