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
 * \file binary_cross_entropy_apt.cpp
 * \brief binary_cross_entropy source file
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "atvoss/elewise/elewise_sch.h"
#include "atvoss/reduce/reduce_sch.h"
#include "arch35/binary_cross_entropy_dag.h"
#include "arch35/binary_cross_entropy_tiling_key.h"
#include "arch35/binary_cross_entropy_struct.h"

using namespace ReduceOpTmpl;
using namespace AscendC;
using namespace BinaryCrossEntropy;

template <REDUCE_TPL_PARAM, uint32_t Reduction, uint32_t HasWeight, uint32_t DType>
__global__ __aicore__ void binary_cross_entropy(GM_ADDR x, GM_ADDR y, GM_ADDR weight, GM_ADDR output, GM_ADDR workspace,
                                                GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(optiling::BinaryCrossEntropyTilingData);
    GET_TILING_DATA_WITH_STRUCT(optiling::BinaryCrossEntropyTilingData, tilingData, tiling);
    TPipe pipe;
    using PromoteType = __reduceType::GetPromoteType<DTYPE_X>::T;
    if constexpr (Reduction == 0) {
        if constexpr (HasWeight == 1) {
            ElementwiseSch<0UL, BCEHasWeightElewise<DTYPE_X>::OpDag> sch(&tilingData.eleBaseTiling, &pipe);
            sch.Init(x, y, weight, output);
            sch.Process();
        } else {
            ElementwiseSch<0UL, BCEElewise<DTYPE_X>::OpDag> sch(&tilingData.eleBaseTiling, &pipe);
            sch.Init(x, y, output);
            sch.Process();
        }
    } else if constexpr (Reduction == 1 || Reduction == 2) {
        using Op = std::conditional_t<
            Reduction == 1,
            std::conditional_t<
                HasWeight,
                ReduceSch<REDUCE_TPL_VALUE, BCEHasWeightSumDag<DTYPE_X, PromoteType>::OpDag>,
                ReduceSch<REDUCE_TPL_VALUE, BCESumDag<DTYPE_X, PromoteType>::OpDag>>,
            std::conditional_t<
                HasWeight,
                ReduceSch<REDUCE_TPL_VALUE, BCEHasWeightMeanDag<DTYPE_X, PromoteType>::OpDag>,
                ReduceSch<REDUCE_TPL_VALUE, BCEMeanDag<DTYPE_X, PromoteType>::OpDag>>>;

        Op op((ReduceOpTilingData*)&tilingData.reduceTiling);
        if constexpr (Reduction == 2) {
            op.template SetVar<PromoteType, 0>(tilingData.reduceTiling.meanVar);
        }

        if constexpr (HasWeight == 1) {
            op.Init(&pipe, x, y, weight, output, workspace);
        } else {
            op.Init(&pipe, x, y, output, workspace);
        }
        op.Process(Reduction == 1 ? static_cast<DTYPE_X>(0) : static_cast<DTYPE_X>(NAN));
    }
}