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
 * \file sigmoid_cross_entropy_with_logits_v2_apt.cpp
 * \brief sigmoid_cross_entropy_with_logits_v2_apt source file
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/sigmoid_cross_entropy_with_logits_v2_dag.h"
#include "arch35/sigmoid_cross_entropy_with_logits_v2_tiling_key.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace AscendC;
using namespace SigmoidCrossEntropyWithLogitsV2;

template <uint64_t schMode, uint32_t Reduction, uint32_t HasWeight, uint32_t HasPosWeight>
__global__ __aicore__ void sigmoid_cross_entropy_with_logits_v2(
    GM_ADDR predict, GM_ADDR target, GM_ADDR weight, GM_ADDR pos_weight, GM_ADDR loss,
    GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (HasWeight && HasPosWeight) {
        using OpDag = SigmoidCrossEntropyWithLogitsV2::SigmoidCEWithLogitsV2HasTwoWeight<DTYPE_PREDICT, DTYPE_LOSS>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(predict, target, weight, pos_weight, loss);
    } else if constexpr (HasWeight){
        using OpDag = SigmoidCrossEntropyWithLogitsV2::SigmoidCEWithLogitsV2WeightOnly<DTYPE_PREDICT, DTYPE_LOSS>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(predict, target, weight, loss);
    } else if constexpr (HasPosWeight){
        using OpDag = SigmoidCrossEntropyWithLogitsV2::SigmoidCEWithLogitsV2PosWeightOnly<DTYPE_PREDICT, DTYPE_LOSS>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(predict, target, pos_weight, loss);
    } else {
        using OpDag = SigmoidCrossEntropyWithLogitsV2::SigmoidCEWithLogitsV2<DTYPE_PREDICT, DTYPE_LOSS>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(predict, target, loss);
    }
}