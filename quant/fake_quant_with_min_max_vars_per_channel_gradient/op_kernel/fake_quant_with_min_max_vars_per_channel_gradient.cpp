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
 * \file fake_quant_with_min_max_vars_per_channel_gradient.cpp
 * \brief Kernel entry for FakeQuantWithMinMaxVarsPerChannelGradient (FP32 / arch35).
 */

#include "kernel_operator.h"

#if (__NPU_ARCH__ == 3510)
#include "arch35/fake_quant_with_min_max_vars_per_channel_gradient_regbase.h"
#endif

extern "C" __global__ __aicore__ void fake_quant_with_min_max_vars_per_channel_gradient(
    GM_ADDR gradients, GM_ADDR x, GM_ADDR min, GM_ADDR max, GM_ADDR backprops_wrt_x,
    GM_ADDR backprops_wrt_min, GM_ADDR backprops_wrt_max, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);

#if (__NPU_ARCH__ == 3510)
    if (TILING_KEY_IS(1001)) {
        FakeQuantWMMVPCG::Kernel<float> op;
        op.Init(gradients, x, min, max, backprops_wrt_x, backprops_wrt_min, backprops_wrt_max,
                workspace, tilingData);
        op.Process();
    }
#endif
}
