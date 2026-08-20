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
 * \file bn_training_update_v3.cpp
 * \brief BNTrainingUpdateV3 arch35 kernel entry（fp16/bf16/fp32 三二进制，DTYPE_X 编译期分发；
 *        仅 ND 单路径，tilingKey=0）
 */

#include "bn_training_update_v3.h"

using namespace BNTrainingUpdateV3Ops;

extern "C" __global__ __aicore__ void bn_training_update_v3(GM_ADDR x, GM_ADDR sum, GM_ADDR square_sum, GM_ADDR scale,
                                                            GM_ADDR offset, GM_ADDR y, GM_ADDR batch_mean,
                                                            GM_ADDR batch_variance, GM_ADDR reserve_1,
                                                            GM_ADDR reserve_2, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AscendC::AIC) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(
        BNTrainingUpdateV3TilingData); // arch35 kernel 侧注册 tiling 结构体（host 不再用 REGISTER_TILING_DATA_CLASS）
    GET_TILING_DATA_WITH_STRUCT(BNTrainingUpdateV3TilingData, tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(0)) {
        BNTrainingUpdateV3Kernel<DTYPE_X> op;
        op.Init(x, sum, square_sum, scale, offset, y, batch_mean, batch_variance, reserve_1, reserve_2, &tilingData,
                &pipe);
        op.Process();
    }
}
