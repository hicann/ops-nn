/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file smooth_l1_loss_v2.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "./arch35/smooth_l1_loss_grad_v2_dag.h"
#include "./arch35/smooth_l1_loss_grad_v2_tiling_key.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace AscendC;

template <uint64_t schMode, uint32_t doutIsScalar>
__global__ __aicore__ void smooth_l1_loss_grad_v2(GM_ADDR predict, GM_ADDR label, GM_ADDR dout, GM_ADDR y, GM_ADDR workspace,
                                                  GM_ADDR tiling)
{
    if constexpr (doutIsScalar == static_cast<uint32_t>(ATTR_IS_TRUE)) {
        using OpDag = SmoothL1LossGradV2Op::SmoothL1LossGradV2OpScalarDag<DTYPE_PREDICT, float>::OpDag;
        Ops::Base::BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(predict, label, dout, y);
    } else {
        using OpDag = SmoothL1LossGradV2Op::SmoothL1LossGradV2OpDag<DTYPE_PREDICT, float>::OpDag;
        Ops::Base::BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(predict, label, dout, y);
    }
}