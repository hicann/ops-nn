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
 * \file in_infer_v2_apt.cpp
 * \brief INInferV2 arch35 kernel entry（fp16/fp32 双二进制，DTYPE_X 编译期分发；
 *        仅 ND 单路径，tilingKey=0；hasGammaBeta 由 tilingData 运行时分发到模板 bool 参数）
 */

#include <type_traits>
#include "arch35/in_infer_v2.h"

using namespace INInferV2Ops;

extern "C" __global__ __aicore__ void in_infer_v2(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR mean,
                                                  GM_ADDR variance, GM_ADDR y, GM_ADDR batch_mean,
                                                  GM_ADDR batch_variance, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AscendC::AIC) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA_WITH_STRUCT(INInferV2TilingData, tilingData, tiling);
    TPipe pipe;
    if (tilingData.hasGammaBeta != 0) {
        INInferV2Kernel<DTYPE_X, true> op;
        op.Init(x, gamma, beta, mean, variance, y, batch_mean, batch_variance, &tilingData, &pipe);
        op.Process();
    } else {
        INInferV2Kernel<DTYPE_X, false> op;
        op.Init(x, gamma, beta, mean, variance, y, batch_mean, batch_variance, &tilingData, &pipe);
        op.Process();
    }
}
