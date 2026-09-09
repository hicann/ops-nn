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
 * \file extend_conv_transpose_v2.cpp
 * \brief
 */

#define K_MAX_SHAPE_DIM 0
#include "../conv3d_backprop_input_v2/arch35/conv3d_backprop_input_v2/conv3d_dx_rowc_block.h"
#include "../conv3d_backprop_input_v2/arch35/conv3d_backprop_input_v2/conv3d_dx_kernel_split_block.h"
#include "../conv3d_backprop_input_v2/arch35/conv3d_backprop_input_v2/conv3d_backprop_input_v2_init_output_arch35.h"
#include "../conv3d_backprop_input_v2/arch35/conv3d_backprop_input_v2/conv3d_backprop_input_v2_vec_transpose.h"
#include "../conv3d_backprop_input_v2/arch35/conv3d_backprop_input_v2/conv3d_dx_small_kernel.h"

#ifndef DTYPE_SCALE0
#define DTYPE_SCALE0 uint64_t
#define FORMAT_SCALE0 FORMAT_MAX
#endif

#ifndef DTYPE_SCALE1
#define DTYPE_SCALE1 uint64_t
#define FORMAT_SCALE1 FORMAT_MAX
#endif

using namespace AscendC;

template <uint8_t loadB2Condition, uint8_t kernelSplitMode, uint8_t groupConvMode, bool isBasicBlockTiling,
          uint8_t loadB1Condition>
__global__ __aicore__ void extend_conv_transpose_v2(GM_ADDR input_size, GM_ADDR x, GM_ADDR filter, GM_ADDR bias,
                                                    GM_ADDR scale0, GM_ADDR scale1, GM_ADDR y0, GM_ADDR y1,
                                                    GM_ADDR workSpace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

    if (tilingData.initOutputFlag == static_cast<int32_t>(InitOutputFlag::L0_INIT)) {
        Conv3dDxInitOutput<DTYPE_Y> opInitOutput0;
        opInitOutput0.Init(y0, tilingData);
        opInitOutput0.Process(y0);
        opInitOutput0.Destroy();
        Conv3dDxInitOutput<DTYPE_Y1> opInitOutput1;
        opInitOutput1.Init(y1, tilingData);
        opInitOutput1.Process(y1);
        opInitOutput1.Destroy();
    }

    if (tilingData.enableVecTrans) {
        // VecTranspose
        DxVecTranspose::Conv3dDxVecTranspose<DTYPE_FILTER> opVecTranspose;
        opVecTranspose.Init(filter, workSpace, tilingData);
        opVecTranspose.Process();
        opVecTranspose.Destroy();
    }

    if constexpr (kernelSplitMode == TPL_NO_SPLIT_KERNEL && groupConvMode == TPL_GROUP_MODE_ORIGIN &&
                  isBasicBlockTiling == true && loadB1Condition == TPL_SMALL_KERNEL) {
        Conv3dDxSmallKernel<DTYPE_FILTER, FORMAT_FILTER, DTYPE_X, FORMAT_X, DTYPE_Y, FORMAT_Y, DTYPE_BIAS, FORMAT_BIAS,
                            loadB2Condition, kernelSplitMode, groupConvMode, loadB1Condition, false, DTYPE_SCALE0,
                            FORMAT_SCALE0, DTYPE_Y1, DTYPE_SCALE1, FORMAT_SCALE1>
            op;
        op.Init(filter, x, y0, workSpace, tilingData, bias, scale0, y1, scale1);
        op.Process();
        return;
    }

    if constexpr (kernelSplitMode != TPL_NO_SPLIT_KERNEL) {
        Conv3dDxKsBlock<DTYPE_FILTER, FORMAT_FILTER, DTYPE_X, FORMAT_X, DTYPE_Y, FORMAT_Y, DTYPE_BIAS, FORMAT_BIAS,
                        loadB2Condition, kernelSplitMode, groupConvMode, loadB1Condition, false, DTYPE_SCALE0,
                        FORMAT_SCALE0, DTYPE_Y1, DTYPE_SCALE1, FORMAT_SCALE1>
            op;
        op.Init(filter, x, y0, workSpace, tilingData, bias, scale0, y1, scale1);
        op.Process();
    } else if constexpr ((isBasicBlockTiling == true) && (loadB1Condition == TPL_VEC_TO_L1_C04)) {
        Conv3dDxOswBlock<DTYPE_FILTER, FORMAT_FILTER, DTYPE_X, FORMAT_X, DTYPE_Y, FORMAT_Y, DTYPE_BIAS, FORMAT_BIAS,
                         loadB2Condition, kernelSplitMode, groupConvMode, TPL_GM_TO_L1, true, DTYPE_SCALE0,
                         FORMAT_SCALE0, DTYPE_Y1, DTYPE_SCALE1, FORMAT_SCALE1>
            op;
        op.Init(filter, x, y0, workSpace, tilingData, bias, scale0, y1, scale1);
        op.Process();
    } else {
        Conv3dDxOswBlock<DTYPE_FILTER, FORMAT_FILTER, DTYPE_X, FORMAT_X, DTYPE_Y, FORMAT_Y, DTYPE_BIAS, FORMAT_BIAS,
                         loadB2Condition, kernelSplitMode, groupConvMode, loadB1Condition, false, DTYPE_SCALE0,
                         FORMAT_SCALE0, DTYPE_Y1, DTYPE_SCALE1, FORMAT_SCALE1>
            op;
        op.Init(filter, x, y0, workSpace, tilingData, bias, scale0, y1, scale1);
        op.Process();
    }
}
