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
 * \file threshold_grad_v2_d_apt.cpp
 * \brief threshold_grad_v2_d_apt
 */

#include "kernel_operator.h"
#include "arch35/threshold_grad_v2_d_dag.h"
#include "arch35/threshold_grad_v2_d_struct.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace AscendC;
using namespace ThresholdGradV2DOp;

template <uint64_t schMode, uint64_t dtype>
__global__ __aicore__ void threshold_grad_v2_d(GM_ADDR gradOutput, GM_ADDR self, GM_ADDR out, GM_ADDR workspace,
                                               GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if constexpr (dtype == static_cast<uint64_t>(THRESHOLD_GRAD_V2_D_TPL_FP16)) {
        BroadcastSch<schMode, ThresholdGradV2DDag<half>::OpDag> sch(tiling);
        sch.Process(gradOutput, self, out);
    } else if constexpr (dtype == static_cast<uint64_t>(THRESHOLD_GRAD_V2_D_TPL_BF16)) {
        BroadcastSch<schMode, ThresholdGradV2DDag<bfloat16_t>::OpDag> sch(tiling);
        sch.Process(gradOutput, self, out);
    } else if constexpr (dtype == static_cast<uint64_t>(THRESHOLD_GRAD_V2_D_TPL_FP32)) {
        BroadcastSch<schMode, ThresholdGradV2DDag<float>::OpDag> sch(tiling);
        sch.Process(gradOutput, self, out);
    } else if constexpr (dtype == static_cast<uint64_t>(THRESHOLD_GRAD_V2_D_TPL_INT32)) {
        BroadcastSch<schMode, ThresholdGradV2DDag<int32_t>::OpDag> sch(tiling);
        sch.Process(gradOutput, self, out);
    } else if constexpr (dtype == static_cast<uint64_t>(THRESHOLD_GRAD_V2_D_TPL_INT8)) {
        BroadcastSch<schMode, ThresholdGradV2DDag<int8_t>::OpDag> sch(tiling);
        sch.Process(gradOutput, self, out);
    } else if constexpr (dtype == static_cast<uint64_t>(THRESHOLD_GRAD_V2_D_TPL_UINT8)) {
        BroadcastSch<schMode, ThresholdGradV2DDag<uint8_t>::OpDag> sch(tiling);
        sch.Process(gradOutput, self, out);
    }
    return;
}
