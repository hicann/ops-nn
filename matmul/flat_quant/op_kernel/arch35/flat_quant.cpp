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
 * \file flat_quant.cpp
 * \brief
 */
#include "flat_quant_high_v2.h"
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
#include "flat_quant_cmct.h" // ascend960(v121): CMCT kernel
#else
#include "blaze/flat_quant.h" // ascend950/350(3510): Blaze kernel
#endif
#include "../flat_quant_vec.h"
#include "../flat_quant_cube.h"
#include "../flat_quant_vec_one.h"
#include "../flat_quant_cube_one.h"
#include "../flat_quant_high.h"

using namespace FlatQuantNS;

extern "C" __global__ __aicore__ void flat_quant(GM_ADDR x, GM_ADDR kronecker_p1, GM_ADDR kronecker_p2,
                                                 GM_ADDR group_list, GM_ADDR out, GM_ADDR quant_scale,
                                                 GM_ADDR workspace, GM_ADDR tiling)
{
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
    // ascend960(v121): AIC:AIV = 1:1; 3510(950/350)保持隐式默认1:2, 不受影响
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
#endif
    AscendC::InitSocState();
    GET_TILING_DATA(tilingData, tiling);
    const FlatQuantTilingData* __restrict tiling_data = &tilingData;
    const TCubeTiling* __restrict mmTilingR = &(tiling_data->matmulTilingR);
    const TCubeTiling* __restrict mmTilingL = &(tiling_data->matmulTilingL);
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
    if constexpr (std::is_same<DTYPE_OUT, fp4x2_e2m1_t>::value) {
        // 非对齐 mxfp4: FlatQuantHighV2 回退路径 (key4)
        if (TILING_KEY_IS(4)) {
            FlatQuantHighV2<DTYPE_X> op;
            REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.matmulR, mmTilingR, op.matmulL, mmTilingL);
            op.Init(x, kronecker_p1, kronecker_p2, out, quant_scale, workspace, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(6)) {
            // mxfp4 对齐: CMCT 路径 (key6), capTag=TPL_CAP_MXFP
            FlatQuantCmctKernel<DTYPE_X, DTYPE_OUT, DTYPE_QUANT_SCALE, Cmct::Gemm::layout::RowMajor, TPL_CAP_MXFP>(
                x, kronecker_p1, kronecker_p2, out, quant_scale, workspace, tilingData);
        }
    } else if constexpr (std::is_same<DTYPE_OUT, hifloat4x2_t>::value) {
        // hif4 输出: CMCT 路径 (key6), capTag=TPL_CAP_HIF4, 仅 v121 支持
        if (TILING_KEY_IS(6)) {
            FlatQuantCmctKernel<DTYPE_X, DTYPE_OUT, DTYPE_QUANT_SCALE, Cmct::Gemm::layout::RowMajor, TPL_CAP_HIF4>(
                x, kronecker_p1, kronecker_p2, out, quant_scale, workspace, tilingData);
        }
    } else {
        if (TILING_KEY_IS(4)) {
            FlatQuantHigh<DTYPE_X> op;
            REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.matmulR, mmTilingR, op.matmulL, mmTilingL);
            op.Init(x, kronecker_p1, kronecker_p2, group_list, out, quant_scale, workspace, &tilingData);
            op.Process();
        }
    }
#else
    if constexpr (std::is_same<DTYPE_OUT, fp4x2_e2m1_t>::value) {
        // mxfp4 对齐: Blaze 路径 (key6)
        if (TILING_KEY_IS(6)) {
            FlatQuantBlazeKernel<DTYPE_X, DTYPE_OUT, DTYPE_QUANT_SCALE, asc::te::nd_ext_layout_ptn>(
                x, kronecker_p1, kronecker_p2, out, quant_scale, workspace, tilingData);
        }
    } else {
        if (TILING_KEY_IS(4)) {
            FlatQuantHigh<DTYPE_X> op;
            REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.matmulR, mmTilingR, op.matmulL, mmTilingL);
            op.Init(x, kronecker_p1, kronecker_p2, group_list, out, quant_scale, workspace, &tilingData);
            op.Process();
        }
    }
#endif
}
