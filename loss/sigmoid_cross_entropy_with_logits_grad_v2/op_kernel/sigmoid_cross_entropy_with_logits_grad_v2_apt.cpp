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
 * \file sigmoid_cross_entropy_with_logits_grad_v2_apt.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "arch35/sigmoid_cross_entropy_with_logits_grad_v2_dag.h"
#include "arch35/sigmoid_cross_entropy_with_logits_grad_v2_struct.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace AscendC;
using namespace SigmoidCrossEntropyWithLogitsGradV2Op;
using namespace SigmoidCrossEntropyWithLogitsGradV2Struct;

template <uint64_t Mode, uint32_t HAS_WEIGHT, uint32_t HAS_POS_WEIGHT, uint32_t IS_MEAN>
__global__ __aicore__ void sigmoid_cross_entropy_with_logits_grad_v2(
    GM_ADDR predict, GM_ADDR target, GM_ADDR dout, GM_ADDR weight, GM_ADDR posWeight, GM_ADDR gradient,
    GM_ADDR workspace, GM_ADDR tiling)
{
    if (HAS_WEIGHT) {
        if (HAS_POS_WEIGHT) {
            using OpDag = typename SigmoidCrossEntropyWithLogitsGradV2DagWeightPosWight<DTYPE_PREDICT, IS_MEAN>::OpDag;
            BroadcastSch<Mode, OpDag> sch(tiling);
            sch.Process(predict, target, dout, weight, posWeight, gradient);
        } else {
            using OpDag = typename SigmoidCrossEntropyWithLogitsGradV2DagWeight<DTYPE_PREDICT, IS_MEAN>::OpDag;
            BroadcastSch<Mode, OpDag> sch(tiling);
            sch.Process(predict, target, dout, weight, gradient);
        }
    } else {
        if (HAS_POS_WEIGHT) {
            using OpDag = typename SigmoidCrossEntropyWithLogitsGradV2DagPosWeight<DTYPE_PREDICT, IS_MEAN>::OpDag;
            BroadcastSch<Mode, OpDag> sch(tiling);
            sch.Process(predict, target, dout, posWeight, gradient);
        } else {
            using OpDag = typename SigmoidCrossEntropyWithLogitsGradV2Dag<DTYPE_PREDICT, IS_MEAN>::OpDag;
            BroadcastSch<Mode, OpDag> sch(tiling);
            sch.Process(predict, target, dout, gradient);
        }
    }
}