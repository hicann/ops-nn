/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_LOSS_COSINE_EMBEDDING_LOSS_UT_TILING_DEF_H_
#define OPS_LOSS_COSINE_EMBEDDING_LOSS_UT_TILING_DEF_H_

#include <cstring>

#include "../../../op_kernel/arch35/cosine_embedding_loss_tiling_data.h"

#define DTYPE_X1 float
#define DTYPE_X2 float
#define DTYPE_TARGET float

template <typename T>
inline void InitTilingData(uint8_t* tiling, T* constData)
{
    std::memcpy(constData, tiling, sizeof(T));
}

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData;                                             \
    InitTilingData<tilingStruct>(tilingArg, &tilingData)

#endif // OPS_LOSS_COSINE_EMBEDDING_LOSS_UT_TILING_DEF_H_
