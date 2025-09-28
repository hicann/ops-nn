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
 * \file foreach_round_off_number_v2.h
 * \brief
 */

#ifndef OP_API_INC_LEVEL0_OP_FOREACH_ROUND_OF_NUMBER_V2_OP_H
#define OP_API_INC_LEVEL0_OP_FOREACH_ROUND_OF_NUMBER_V2_OP_H

#include "opdev/op_executor.h"

namespace l0op {
const aclTensorList* ForeachRoundOffNumberV2(
    const aclTensorList* x, const aclTensor* roundMode, const aclTensorList* out, aclOpExecutor* executor);
}

#endif // OP_API_INC_LEVEL0_OP_FOREACH_ROUND_OFF_NUMBER_V2_OP_H
