/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_NN_SIGMOID_FOCAL_LOSS_GRAD_TESTS_UT_OP_KERNEL_TILING_DEF_H
#define OPS_NN_SIGMOID_FOCAL_LOSS_GRAD_TESTS_UT_OP_KERNEL_TILING_DEF_H

#include <cstring>

#ifndef DTYPE_PRED
#define DTYPE_PRED float
#endif

#ifndef DTYPE_DOUT
#define DTYPE_DOUT float
#endif

#ifndef GET_TILING_DATA_WITH_STRUCT
#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingPointer)                        \
    tilingStruct tilingData;                                                                        \
    std::memcpy(reinterpret_cast<void*>(&tilingData), reinterpret_cast<const void*>(tilingPointer), \
                sizeof(tilingStruct))
#endif

#endif
