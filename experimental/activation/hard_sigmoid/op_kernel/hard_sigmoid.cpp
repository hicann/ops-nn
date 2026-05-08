/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file hard_sigmoid.cpp
 * \brief HardSigmoid Kernel entry (atvoss Elewise mode, ElementwiseSch)
 *
 * Formula: HardSigmoid(x) = max(0, min(1, x/6 + 0.5))
 *
 * DTYPE_X is injected by the build system based on _def.cpp input "x" dtype list.
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "hard_sigmoid_dag.h"
#include "hard_sigmoid_struct.h"
#include "atvoss/elewise/elewise_sch.h"

using namespace AscendC;
using namespace Ops::Base;

template <uint64_t schMode>
__global__ __aicore__ void hard_sigmoid(GM_ADDR x, GM_ADDR y,
                                         GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(HardSigmoidTilingData);
    GET_TILING_DATA_WITH_STRUCT(HardSigmoidTilingData, tilingData, tiling);
    TPipe pipe;

    if constexpr (std::is_same_v<DTYPE_X, int32_t>) {
        using OpDag = NsHardSigmoid::HardSigmoidInt32::OpDag;
        ElementwiseSch<schMode, OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.template SetVar<float, 0>(tilingData.alpha);
        sch.template SetVar<float, 1>(tilingData.beta);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (std::is_same_v<DTYPE_X, half> ||
                          std::is_same_v<DTYPE_X, bfloat16_t>) {
        using OpDag = NsHardSigmoid::HardSigmoidWithCast<DTYPE_X>::OpDag;
        ElementwiseSch<schMode, OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.template SetVar<float, 0>(tilingData.alpha);
        sch.template SetVar<float, 1>(tilingData.beta);
        sch.Init(x, y);
        sch.Process();
    } else {
        using OpDag = NsHardSigmoid::HardSigmoidCompute<DTYPE_X>::OpDag;
        ElementwiseSch<schMode, OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.template SetVar<float, 0>(tilingData.alpha);
        sch.template SetVar<float, 1>(tilingData.beta);
        sch.Init(x, y);
        sch.Process();
    }
}
