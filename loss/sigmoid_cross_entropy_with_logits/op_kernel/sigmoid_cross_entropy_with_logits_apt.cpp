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
 * \file sigmoid_cross_entropy_with_logits_apt.cpp
 * \brief sigmoid_cross_entropy_with_logits_apt source file
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/sigmoid_cross_entropy_with_logits_dag.h"
#include "arch35/sigmoid_cross_entropy_with_logits_struct.h"
#include "atvoss/elewise/elewise_sch_with_scalar.h"
#include "atvoss/elewise/elewise_sch_16b.h"
#include "atvoss/util/dfx.h"

using namespace AscendC;
using namespace Ops::Base;
using namespace SigmoidCrossEntropyWithLogitsOp;

template <uint64_t dType>
__global__ __aicore__ void sigmoid_cross_entropy_with_logits(
    GM_ADDR predict, GM_ADDR target, GM_ADDR loss,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(EleBaseTilingData16B);
    GET_TILING_DATA_PTR_WITH_STRUCT(EleBaseTilingData16B, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if constexpr (dType == TPL_FP16) {
        ElementwiseSch16B<1, SigmoidCrossEntropyWithLogitsDagWithCast<half>::OpDag> sch(tilingData);
        sch.Init(predict, target, loss);
        sch.Process();
    } else if constexpr (dType == TPL_BF16) {
        ElementwiseSch16B<1, SigmoidCrossEntropyWithLogitsDagWithCast<bfloat16_t>::OpDag> sch(tilingData);
        sch.Init(predict, target, loss);
        sch.Process();
    } else if constexpr (dType == TPL_FP32) {
        ElementwiseSch16B<1, SigmoidCrossEntropyWithLogitsDagNoCast<float>::OpDag> sch(tilingData);
        sch.Init(predict, target, loss);
        sch.Process();
    }
}