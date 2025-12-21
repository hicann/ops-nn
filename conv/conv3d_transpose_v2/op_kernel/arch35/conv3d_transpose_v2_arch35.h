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
 * \file conv3d_transpose_v2_arch35.h
 * \brief
 */
#ifndef CONV3D_TRANSPOSE_V2_ARCH_35_H
#define CONV3D_TRANSPOSE_V2_ARCH_35_H
#include "../conv3d_backprop_input_v2/arch35/conv3d_backprop_input_v2/conv3d_backprop_input_v2.h"
#include "../conv3d_backprop_input_v2/arch35/conv3d_backprop_input_v2/conv3d_backprop_input_v2_init_output.h"
#include "../conv3d_backprop_input_v2/arch35/conv3d_backprop_input_v2/conv3d_dx_rowc_block.h"
#include "../conv3d_backprop_input_v2/arch35/conv3d_backprop_input_v2/conv3d_dx_kernel_split_block.h"
#include "../conv3d_backprop_input_v2/arch35/conv3d_backprop_input_v2/conv3d_backprop_input_v2_vec_transpose.h"

using namespace AscendC;

#define CONV3D_DX_OP_VECTRANSPOSE(...)                 \
    do {                                                        \
        __VA_ARGS__ opVecTranspose;                                 \
        opVecTranspose.Init(filter, workSpace, &tilingData);    \
        opVecTranspose.Process();                               \
        opVecTranspose.Destroy();                               \
    } while (0)

#define CONV3D_DX_TRANSPOSE_RUN_OP(...)        \
    do {                                                \
        __VA_ARGS__ op;                                     \
        op.Init(filter, x, y, usrWsp, &tilingData);     \
        op.Process();                                   \
    } while (0)

template <uint8_t loadB2Condition, uint8_t kernelSplitMode, uint8_t groupConvMode, bool isBasicBlockTiling, uint8_t loadB1Condition>
__global__ __aicore__ void conv3d_transpose_v2_arch35(GM_ADDR input_size, GM_ADDR x, GM_ADDR filter, GM_ADDR bias,
                                                          GM_ADDR offset_w, GM_ADDR y, GM_ADDR workSpace,
                                                          GM_ADDR tiling)
{
    if (workSpace == nullptr) {
        return;
    }

    GM_ADDR usrWsp = GetUserWorkspace(workSpace);
    if (usrWsp == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(conv_bp_v2_kernel::Conv3DBackpropInputV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(conv_bp_v2_kernel::Conv3DBackpropInputV2TilingData, tilingData, tiling);
#if defined(__DAV_310R6__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
#elif defined(__DAV_C310__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif

    if (tilingData.conv3DDxTiling.initOutputFlag == static_cast<int32_t>(InitOutputFlag::L0_INIT)) {
        Conv3dDxInitOutput<DTYPE_Y> opInitOutput;
        opInitOutput.Init(y, &tilingData);
        opInitOutput.Process();
        opInitOutput.Destroy();
    }

    if ASCEND_IS_AIV {
        if (tilingData.conv3DDxTiling.enableVecTrans) {
            // VecTranspose
            CONV3D_DX_OP_VECTRANSPOSE(DxVecTranspose::Conv3dDxVecTranspose<DTYPE_FILTER>);
        }
    }

    if constexpr (kernelSplitMode != TPL_NO_SPLIT_KERNEL) {
        CONV3D_DX_TRANSPOSE_RUN_OP(Conv3dDxKsBlock<DTYPE_FILTER, FORMAT_FILTER, DTYPE_X, FORMAT_X,
                                                   DTYPE_Y, FORMAT_Y, DTYPE_BIAS, FORMAT_BIAS,
                                                   loadB2Condition, kernelSplitMode, groupConvMode>);
    } else if constexpr ((isBasicBlockTiling == true) && (loadB1Condition == TPL_VEC_TO_L1_C04)) {
        CONV3D_DX_TRANSPOSE_RUN_OP(Conv3dDxOswBlock<DTYPE_FILTER, FORMAT_FILTER, DTYPE_X, FORMAT_X,
                                                    DTYPE_Y, FORMAT_Y, DTYPE_BIAS, FORMAT_BIAS,
                                                    loadB2Condition, kernelSplitMode, groupConvMode, TPL_GM_TO_L1, true>);
    } else if constexpr ((isBasicBlockTiling == false) && (loadB1Condition == TPL_GM_TO_L1)) {
        CONV3D_DX_TRANSPOSE_RUN_OP(Conv3dDx<DTYPE_FILTER, FORMAT_FILTER, DTYPE_X, FORMAT_X,
                                            DTYPE_Y, FORMAT_Y, DTYPE_BIAS, FORMAT_BIAS,
                                            loadB2Condition, kernelSplitMode, groupConvMode>);
    } else {
        CONV3D_DX_TRANSPOSE_RUN_OP(Conv3dDxOswBlock<DTYPE_FILTER, FORMAT_FILTER, DTYPE_X, FORMAT_X,
                                                    DTYPE_Y, FORMAT_Y, DTYPE_BIAS, FORMAT_BIAS,
                                                    loadB2Condition, kernelSplitMode, groupConvMode, loadB1Condition>);
    }
}
#endif  // CONV3D_TRANSPOSE_V2_ARCH_35_H