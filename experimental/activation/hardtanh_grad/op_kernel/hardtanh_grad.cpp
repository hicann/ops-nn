/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hardtanh_grad.cpp
 * \brief
 */

#include "hardtanh_grad.h"

enum class HardtanhGradTilingKey : uint32_t
{
    TILING_KEY_EXAMPLE_SINGLE_BUFFER = 0,
    TILING_KEY_EXAMPLE_DOUBLE_BUFFER = 1,
};
template <uint32_t schMode>
__global__ __aicore__ void hardtanh_grad(GM_ADDR gradOutput, GM_ADDR self, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
     REGISTER_TILING_DEFAULT(HardtanhGradTilingData);
     GET_TILING_DATA_WITH_STRUCT(HardtanhGradTilingData, tilingData, tiling);
     AscendC::TPipe pipe;
     KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); 
    
    if constexpr (schMode == static_cast<uint32_t>(HardtanhGradTilingKey::TILING_KEY_EXAMPLE_DOUBLE_BUFFER)) 
    {
         MyHardtanhGrad::KernelHardtanhGrad<DTYPE_SELF, DTYPE_OUT> op; 
         op.Init(gradOutput, self, out, 
                 &tilingData, &pipe);   //   // 算子kernel实例初始化
         op.Process();  
    }
    if constexpr (schMode == static_cast<uint32_t>(HardtanhGradTilingKey::TILING_KEY_EXAMPLE_SINGLE_BUFFER)) {
         MyHardtanhGrad::KernelHardtanhGrad_NoDB<DTYPE_SELF, DTYPE_OUT> op; 
         op.Init(gradOutput, self, out, 
                 &tilingData, &pipe);   //   // 算子kernel实例初始化
    }    
}