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
 * \file repeat_interleave_grad.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_REPEAT_INTERLEAVE_GRAD_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_REPEAT_INTERLEAVE_GRAD_H
#include <cstdint>
#include "op_common/atvoss/reduce/reduce_tiling.h"

namespace optiling {

struct RepeatInterleaveGradCompileInfo {
    int64_t coreNum;
    int64_t ubSize;
    int64_t blockSize;
    bool isAscendC;
    uint32_t clSize;
    uint32_t vRegSize;
};
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_REPEAT_INTERLEAVE_GRAD_H