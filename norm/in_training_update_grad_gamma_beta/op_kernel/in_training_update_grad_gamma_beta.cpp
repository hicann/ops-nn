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
 * \file in_training_update_grad_gamma_beta.cpp
 * \brief Ascend 950 kernel entry.
 */

#include "arch35/in_training_update_grad_gamma_beta_base.h"
#include "arch35/in_training_update_grad_gamma_beta_tiling_key.h"

template <int DTYPE_MODE>
__global__ __aicore__ void in_training_update_grad_gamma_beta(GM_ADDR resGamma, GM_ADDR resBeta, GM_ADDR pdGamma,
                                                              GM_ADDR pdBeta, GM_ADDR workspace, GM_ADDR tiling)
{
    static_assert(DTYPE_MODE == 0, "only float32 is supported");
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(INTrainingUpdateGradGammaBetaTilingData);
    GET_TILING_DATA_WITH_STRUCT(INTrainingUpdateGradGammaBetaTilingData, tilingData, tiling);
    (void)workspace;
    AscendC::TPipe pipe;
    NsINTrainingUpdateGradGammaBeta::INTrainingUpdateGradGammaBetaKernel op;
    op.Init(resGamma, resBeta, pdGamma, pdBeta, &tilingData, &pipe);
    op.Process();
}
