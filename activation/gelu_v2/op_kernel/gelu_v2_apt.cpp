/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file gelu_v2_apt.cpp
 * \brief z = gelu(x)
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/gelu_v2_dag.h"
#include "arch35/gelu_v2_struct.h"
#include "atvoss/elewise/elewise_sch_16b.h"
#include "atvoss/util/dfx.h"

using namespace AscendC;

template <uint64_t schMode, uint64_t approximate, uint64_t dType>
__global__ __aicore__ void gelu_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) 
{
    REGISTER_TILING_DEFAULT(EleBaseTilingData16B);
    GET_TILING_DATA_PTR_WITH_STRUCT(EleBaseTilingData16B, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if constexpr(approximate == TPL_NONE) {
        if constexpr(dType == TPL_FP16) {
            ElementwiseSch16B<schMode, GeluV2Op::GeluV2Erf16BDag<half>::OpDag> sch(tilingData);
            sch.Init(x, y);
            sch.Process();
        } else if constexpr(dType == TPL_BF16) {
            ElementwiseSch16B<schMode, GeluV2Op::GeluV2Erf16BDag<bfloat16_t>::OpDag> sch(tilingData);
            sch.Init(x, y);
            sch.Process();
        } else if constexpr(dType == TPL_FP32) {
            ElementwiseSch16B<schMode, GeluV2Op::GeluV2Erf32BDag<float>::OpDag> sch(tilingData);
            sch.Init(x, y);
            sch.Process();
        } 
    } else if (approximate == TPL_TANH) {
        if constexpr(dType == TPL_FP16) {
            ElementwiseSch16B<schMode, GeluV2Op::GeluV2TanhDag<half>::OpDag> sch(tilingData);
            sch.Init(x, y);
            sch.Process();
        } else if constexpr(dType == TPL_BF16) {
            ElementwiseSch16B<schMode, GeluV2Op::GeluV2TanhDag<bfloat16_t>::OpDag> sch(tilingData);
            sch.Init(x, y);
            sch.Process();
        } else if constexpr(dType == TPL_FP32) {
            ElementwiseSch16B<schMode, GeluV2Op::GeluV2TanhDag<float>::OpDag> sch(tilingData);
            sch.Init(x, y);
            sch.Process();
        } 
    }
    return;
}