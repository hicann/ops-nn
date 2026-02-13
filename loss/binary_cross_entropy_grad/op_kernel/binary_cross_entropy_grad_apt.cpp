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
 * \file binary_cross_entropy_grad_apt.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/binary_cross_entropy_grad_dag.h"
#include "arch35/binary_cross_entropy_grad_tiling_key.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace Ops::Base;
using namespace AscendC;
using namespace BinaryCrossEntropyGrad;
template <uint64_t schMode, uint32_t Reduction, uint32_t HasWeight>
__global__ __aicore__ void binary_cross_entropy_grad(GM_ADDR x, GM_ADDR y, GM_ADDR grad_output, GM_ADDR weight,
        GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    if constexpr (static_cast<u_int32_t>(Reduction) == static_cast<u_int32_t>(BCE_MEAN)) {
        if constexpr (static_cast<u_int32_t>(HasWeight) == static_cast<u_int32_t>(BCE_HAS_WEIGHT)) {
            using OpDag = BinaryCrossEntropyGrad::BCEGMeanHasWeight<DTYPE_X>::OpDag;
            BroadcastSch<schMode, OpDag> sch(tiling);
            sch.Process(x, y, grad_output, weight, output);
        } else {
            using OpDag = BinaryCrossEntropyGrad::BCEGMeanHasNoWeight<DTYPE_X>::OpDag;
            BroadcastSch<schMode, OpDag> sch(tiling);
            sch.Process(x, y, grad_output, output);
        }
    } else {
        if constexpr (static_cast<u_int32_t>(HasWeight) == static_cast<u_int32_t>(BCE_HAS_WEIGHT)) {
            using OpDag = BinaryCrossEntropyGrad::BCEGSumHasWeight<DTYPE_X>::OpDag;
            BroadcastSch<schMode, OpDag> sch(tiling);
            sch.Process(x, y, grad_output, weight, output);

        } else {
            using OpDag = BinaryCrossEntropyGrad::BCEGSumHasNoWeight<DTYPE_X>::OpDag;
            BroadcastSch<schMode, OpDag> sch(tiling);
            sch.Process(x, y, grad_output, output);
        }
    }
}