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
 * \file conv3d_backprop_input_v2.cpp
 * \brief
 */
#include "./arch32/conv3d_backprop_input_v2_tiling_data.h"
#if __CCE_AICORE__ == 310
#include "conv3d_backprop_input_v2_arch35_tiling_key.h"
#if defined(__DAV_C310__)
#include "arch35/conv3d_backprop_input_v2_arch35.h"
#endif
#else
#include "./arch32/conv3d_dx_v2_basic_block.h"
#include "./arch32/conv3d_backprop_input_v2_init_output.h"
#include "./arch32/conv3d_backprop_input_v2.h"
#include "conv3d_backprop_input_v2_tiling_key.h"
#endif


using namespace AscendC;

#if __CCE_AICORE__ == 310
template <uint8_t loadB2Condition, uint8_t kernelSplitMode, uint8_t groupConvMode, bool isBaseBlockTiling, uint8_t loadB1Condition>
__global__ __aicore__ void conv3d_backprop_input_v2(
    GM_ADDR input_size, GM_ADDR filter, GM_ADDR out_backprop, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling)
{
#if defined(__DAV_C310__)
    conv3d_backprop_input_v2_arch35<loadB2Condition, kernelSplitMode, groupConvMode, isBaseBlockTiling,
                                    loadB1Condition>(input_size, filter, out_backprop, y, workSpace, tiling);
    return;
#endif
}
#else
template <uint8_t loadB2Condition, bool enableKernelSplit, bool useBasicBlock>
__global__ __aicore__ void conv3d_backprop_input_v2(
    GM_ADDR input_size, GM_ADDR filter, GM_ADDR out_backprop, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling)
{
    if (workSpace == nullptr) {
        return;
    }

    GM_ADDR usrWsp = GetUserWorkspace(workSpace);
    if (usrWsp == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(Conv3DBackpropInputV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(Conv3DBackpropInputV2TilingData, tilingData, tiling);
#if __CCE_AICORE__ == 220
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
#endif
    if (tilingData.conv3DDxTiling.initOutputFlag == static_cast<int32_t>(InitOutputFlag::L0_INIT)) {
        // init output with L0C now
        Conv3dDxInitOutput<DTYPE_Y, FORMAT_Y, InitOutputFlag::L0_INIT> opInitOutput;
        opInitOutput.Init(y, &tilingData);
        opInitOutput.Process();
        opInitOutput.Destroy();
    }
    if (enableKernelSplit == false) {
        if (useBasicBlock == false) {
            Conv3dDx<DTYPE_FILTER, FORMAT_FILTER, DTYPE_OUT_BACKPROP, FORMAT_OUT_BACKPROP, DTYPE_Y, FORMAT_Y, loadB2Condition> op;
            op.Init(filter, out_backprop, y, usrWsp, &tilingData);
            op.Process();
        } else {
            Conv3dDxBasicBlockSplitMN<DTYPE_FILTER, FORMAT_FILTER, DTYPE_OUT_BACKPROP, FORMAT_OUT_BACKPROP, DTYPE_Y, FORMAT_Y, loadB2Condition> op;
            op.Init(filter, out_backprop, y, usrWsp, &tilingData);
            op.Process();
        }
    }
}
#endif