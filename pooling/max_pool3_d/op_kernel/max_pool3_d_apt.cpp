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
 * \file pool_3d.cpp
 * \brief pool_3d implied
 */

#include <cstdint>
#include "kernel_operator.h"
#include "../pool_3d/arch35/max_pool_3d_simt.h"
#include "../pool_3d/arch35/pool_3d_ncdhw_small_kernel.h"
#include "../pool_3d/arch35/pool_3d_ncdhw_small_kernel_pad.h"
#include "../pool_3d/arch35/pool_3d_ncdhw_big_kernel.h"
#include "../pool_3d/arch35/pool_3d_big_kernel_ndhwc.h"
#include "../pool_3d/arch35/pool_3d_ndhwc_small_kernel.h"
#include "../pool_3d/arch35/pool_3d_ndhwc_small_kernel_pad.h"
#include "../pool_3d/arch35/pool_3d_ndhwc_big_channel.h"
#include "../pool_3d/arch35/pool_3d_ndhwc_big_channel_pad.h"
#include "../pool_3d/arch35/pool_3d_ksize_one_kernel.h"

using namespace MaxPool3DSimt;

#define MAX_POOL_3D_TILING_KEY_SIMT_NCDHW_INT32 911100
#define MAX_POOL_3D_TILING_KEY_SIMT_NDHWC_INT32 911101
#define MAX_POOL_3D_TILING_KEY_SIMT_NCDHW_INT64 911110
#define MAX_POOL_3D_TILING_KEY_SIMT_NDHWC_INT64 911111
#define MAX_POOL_3D_TILING_KEY_SMALL_KERNEL_NO_PADDING_NCDHW 300001
#define MAX_POOL_3D_TILING_KEY_SMALL_KERNEL_PADDING_NCDHW 300002
#define MAX_POOL_3D_TILING_KEY_SMALL_KERNEL_NO_PADDING_SPARSE 300004
#define MAX_POOL_3D_TILING_KEY_ONE_KSIZE 100001
#define MAX_POOL_3D_TILING_KEY_BIG_KERNEL_NDHWC 411110
#define POOL_3D_TILING_KEY_SMALL_KERNEL_NDHWC 200001
#define POOL_3D_TILING_KEY_SMALL_KERNEL_NDHWC_PAD 211110
#define POOL_3D_TILING_KEY_BIG_CHANNELS_NDHWC 222220
#define POOL_3D_TILING_KEY_BIG_CHANNELS_NDHWC_PAD 222221
#define BIG_KERNEL_FORMAT_NCDHW 511110

constexpr int32_t NCDHW = 0;
constexpr int32_t NDHWC = 1;

extern "C" __global__ __aicore__ void max_pool3_d(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipeBase;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(MAX_POOL_3D_TILING_KEY_SIMT_NCDHW_INT32)) {
        GET_TILING_DATA_WITH_STRUCT(Pool3DSimtTilingData, tilingData, tiling);
        MaxPool3DSimtImpl<DTYPE_X, int32_t, NCDHW> op(&pipeBase, &tilingData);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(MAX_POOL_3D_TILING_KEY_SIMT_NDHWC_INT32)) {
        GET_TILING_DATA_WITH_STRUCT(Pool3DSimtTilingData, tilingData, tiling);
        MaxPool3DSimtImpl<DTYPE_X, int32_t, NDHWC> op(&pipeBase, &tilingData);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(MAX_POOL_3D_TILING_KEY_SIMT_NCDHW_INT64)) {
        GET_TILING_DATA_WITH_STRUCT(Pool3DSimtTilingData, tilingData, tiling);
        MaxPool3DSimtImpl<DTYPE_X, int64_t, NCDHW> op(&pipeBase, &tilingData);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(MAX_POOL_3D_TILING_KEY_SIMT_NDHWC_INT64)) {
        GET_TILING_DATA_WITH_STRUCT(Pool3DSimtTilingData, tilingData, tiling);
        MaxPool3DSimtImpl<DTYPE_X, int64_t, NDHWC> op(&pipeBase, &tilingData);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(MAX_POOL_3D_TILING_KEY_SMALL_KERNEL_NO_PADDING_NCDHW)) {
        GET_TILING_DATA_WITH_STRUCT(Pool3DNcdhwSmallKernelTilingData, tilingData, tiling);
        Pool3D::Pool3DNcdhwSmallKernel<DTYPE_X, Pool3D::OP_TYPE_MAX_POOL_3D> op(&pipeBase, &tilingData);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(MAX_POOL_3D_TILING_KEY_SMALL_KERNEL_PADDING_NCDHW)) {
        GET_TILING_DATA_WITH_STRUCT(Pool3DNcdhwSmallKernelTilingData, tilingData, tiling);
        Pool3D::Pool3DNcdhwSmallKernelPad<DTYPE_X, Pool3D::OP_TYPE_MAX_POOL_3D, false> op(&pipeBase, &tilingData);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(MAX_POOL_3D_TILING_KEY_BIG_KERNEL_NDHWC)) {
        GET_TILING_DATA_WITH_STRUCT(Pool3DBigKernelNDHWCTilingData, tilingData, tiling);
        Pool3D::Pool3DNDHWCBigKernel<DTYPE_X, Pool3D::OP_TYPE_MAX_POOL_3D> op(&pipeBase, &tilingData);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(POOL_3D_TILING_KEY_SMALL_KERNEL_NDHWC)) {
        GET_TILING_DATA_WITH_STRUCT(Pool3DSmallKernelNDHWCTilingData, tilingData, tiling);
        Pool3D::Pool3dNDHWCSmallKernel<DTYPE_X, Pool3D::OP_TYPE_MAX_POOL_3D> op(&pipeBase, &tilingData);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(POOL_3D_TILING_KEY_SMALL_KERNEL_NDHWC_PAD)) {
        GET_TILING_DATA_WITH_STRUCT(Pool3DSmallKernelNDHWCTilingData, tilingData, tiling);
        Pool3D::Pool3dNDHWCSmallKernelPad<DTYPE_X, Pool3D::OP_TYPE_MAX_POOL_3D> op(&pipeBase, &tilingData);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(POOL_3D_TILING_KEY_BIG_CHANNELS_NDHWC)) {
        GET_TILING_DATA_WITH_STRUCT(Pool3DSmallKernelNDHWCTilingData, tilingData, tiling);
        Pool3D::Pool3dNDHWCBigChannel<DTYPE_X, Pool3D::OP_TYPE_MAX_POOL_3D> op(&pipeBase, &tilingData);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(POOL_3D_TILING_KEY_BIG_CHANNELS_NDHWC_PAD)) {
        GET_TILING_DATA_WITH_STRUCT(Pool3DSmallKernelNDHWCTilingData, tilingData, tiling);
        Pool3D::Pool3dNDHWCBigChannelPad<DTYPE_X, Pool3D::OP_TYPE_MAX_POOL_3D> op(&pipeBase, &tilingData);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(BIG_KERNEL_FORMAT_NCDHW)) {
        GET_TILING_DATA_WITH_STRUCT(Pool3DNcdhwBigKernelTilingData, tilingData, tiling);
        Pool3D::Pool3DNcdhwBigKernel<DTYPE_X, Pool3D::OP_TYPE_MAX_POOL_3D> op(&pipeBase, &tilingData);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(MAX_POOL_3D_TILING_KEY_ONE_KSIZE)) {
        GET_TILING_DATA_WITH_STRUCT(Pool3DOneKsizeTilingData, tilingData, tiling);
        Pool3D::Pool3DKsizeOneKernel<DTYPE_X, Pool3D::OP_TYPE_MAX_POOL_3D> op(&pipeBase, &tilingData);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(MAX_POOL_3D_TILING_KEY_SMALL_KERNEL_NO_PADDING_SPARSE)) {
        GET_TILING_DATA_WITH_STRUCT(Pool3DNcdhwSmallKernelTilingData, tilingData, tiling);
        Pool3D::Pool3DNcdhwSmallKernel<DTYPE_X, Pool3D::OP_TYPE_MAX_POOL_3D, true> op(&pipeBase, &tilingData);
        op.Init(x, y);
        op.Process();
    }
}