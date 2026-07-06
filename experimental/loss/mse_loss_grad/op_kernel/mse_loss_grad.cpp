/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mse_loss_grad.cpp
 * \brief
 */

#include "mse_loss_grad.h"

enum class MseLossGradTilingKey : uint32_t {
    TILING_KEY_NONSCALE = 0,
    TILING_KEY_SCALE = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void mse_loss_grad(GM_ADDR predict, GM_ADDR label, GM_ADDR dout, GM_ADDR y, GM_ADDR workspace,
                                         GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MseLossGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(MseLossGradTilingData, tilingData, tiling);
    // AscendC::printf("mse_loss_grad enterance!\n");
    if constexpr (schMode == static_cast<uint32_t>(MseLossGradTilingKey::TILING_KEY_NONSCALE)) {
        NsMseLossGrad::MseLossGrad<DTYPE_PREDICT> op;  // 算子kernel实例获取
        op.Init(predict, label, dout, y, &tilingData); // 算子kernel实例初始化
        op.Process();                                  // 算子kernel实例执行
    }
    // 场景2
    if constexpr (schMode == static_cast<uint32_t>(MseLossGradTilingKey::TILING_KEY_SCALE)) {
        NsMseLossGrad::MseLossGradScale<DTYPE_PREDICT> op; // 算子kernel实例获取
        op.Init(predict, label, dout, y, &tilingData);     // 算子kernel实例初始化
        op.Process();                                      // 算子kernel实例执行
    }
}
