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
 * \file swish_apt.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/swish_dag.h"
#include "arch35/swish_struct.h"
#include "atvoss/util/dfx.h"
#include "atvoss/elewise/elewise_sch_with_scalar.h"

using namespace AscendC;
using namespace SwishOp;
using namespace SwishDag;

template <uint64_t schMode, uint64_t attrWork, typename DtypeX>
__global__ __aicore__ void SwishKernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(EleBaseTilingData32B);
    GET_TILING_DATA_PTR_WITH_STRUCT(EleBaseTilingData32B, tilingData, tiling);          
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if constexpr (attrWork == static_cast<uint64_t>(TPL_SCALE_NEG_ONE)) {
        ElementwiseSchWithScalar<EleBaseTilingData32B,schMode, typename SwishDag::SwishNegOne<DtypeX>::OpDag> sch(tilingData);
        sch.Init(x, y);
        sch.Process();    
    } else if constexpr (attrWork == static_cast<uint64_t>(TPL_SCALE_ZERO)) {
        ElementwiseSchWithScalar<EleBaseTilingData32B,schMode, typename SwishDag::SwishZero<DtypeX>::OpDag> sch(tilingData);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (attrWork == static_cast<uint64_t>(TPL_SCALE_OTHER)) {
        ElementwiseSchWithScalar<EleBaseTilingData32B,schMode, typename SwishDag::SwishOther<DtypeX>::OpDag> sch(tilingData);
        sch.Init(x, y);
        sch.Process();
    }
    return;
}

template <uint64_t schMode, uint64_t attrWork>
__global__ __aicore__ void swish(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    return SwishKernel<schMode, attrWork, DTYPE_X>(x, y, workspace, tiling);
}

