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
 * \file conv3d_backprop_filter_v2_arch35.h
 * \brief
 */
#ifndef CONV3D_BACKPROP_FILTER_V2_ARCH_35_H
#define CONV3D_BACKPROP_FILTER_V2_ARCH_35_H
#include "conv3d_backprop_filter_v2/conv3d_backprop_filter_v2.h"
#include "conv3d_backprop_filter_v2/conv3d_backprop_filter_v2_init_output.h"
#include "conv3d_backprop_filter_v2/conv3d_dw_v2_basic_block.h"
#include "conv3d_backprop_filter_v2/conv3d_backprop_filter_v2_tiling_data.h"

using namespace AscendC;

template <uint32_t conv3DDWTemplateId>
__global__ __aicore__ void conv3d_backprop_filter_v2_arch35(GM_ADDR x, GM_ADDR filter_size, GM_ADDR out_backprop,
                                                                GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling) {
    if (workSpace == nullptr) {
        return;
    }
    SetSysWorkspace(workSpace);
    GM_ADDR user1 = GetUserWorkspace(workSpace);
    if (user1 == nullptr) {
        return;
    }

#if defined(__DAV_C220__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_0);
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
#endif

    REGISTER_TILING_DEFAULT(conv_bp_v2_kernel::Conv3DBackpropFilterV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(conv_bp_v2_kernel::Conv3DBackpropFilterV2TilingData, tilingData, tiling);
    // 1982的AscendC::SyncAll()在camodel中不生效，因此在单算子运行camodel时需注释清零动作；
    Conv3dDwInitOutput<DTYPE_Y> opInitOutput;
    opInitOutput.Init(y, &tilingData);
    opInitOutput.Process();
    opInitOutput.Destroy();

    if (conv3DDWTemplateId == TPL_STREAM_DFL) {
        Conv3dDw<DTYPE_X, FORMAT_X, DTYPE_OUT_BACKPROP, FORMAT_OUT_BACKPROP, DTYPE_Y, FORMAT_Y,
                 conv3DDWTemplateId> op;
        op.Init(x, out_backprop, y, user1, &tilingData);
        op.Process();
    } else if (conv3DDWTemplateId == TPL_STREAM_K) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);  // cube: 1 vector: 2
        Conv3dDwBasicBlockStreamK<DTYPE_X, FORMAT_X, DTYPE_OUT_BACKPROP, FORMAT_OUT_BACKPROP, DTYPE_Y, FORMAT_Y> op;
        op.Init(x, out_backprop, y, user1, &tilingData);
        op.Process();
    } else if (conv3DDWTemplateId == TPL_MN_STREAM_K) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);  // cube: 1 vector: 2
        Conv3dDwBasicBlockMNStreamK<DTYPE_X, FORMAT_X, DTYPE_OUT_BACKPROP, FORMAT_OUT_BACKPROP, DTYPE_Y, FORMAT_Y> op;
        op.Init(x, out_backprop, y, user1, &tilingData);
        op.Process();
    }
}
#endif  // CONV3D_BACKPROP_FILTER_V2_ARCH35_H