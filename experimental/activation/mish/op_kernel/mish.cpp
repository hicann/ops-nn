/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file mish.cpp
 * \brief
*/

#include "mish.h"

enum class MishTilingKey : uint32_t
{
    TILING_KEY_EXAMPLE_FLOAT = 0,
    TILING_KEY_EXAMPLE_OTHER = 1,
};
template <uint32_t schMode>
__global__ __aicore__ void mish(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
     REGISTER_TILING_DEFAULT(MishTilingData);
     GET_TILING_DATA_WITH_STRUCT(MishTilingData, tilingData, tiling);
     MyMish::KernelMish<DTYPE_X,DTYPE_Y> op; 
     op.Init(x, y, &tilingData);
     op.Process();                       
}
