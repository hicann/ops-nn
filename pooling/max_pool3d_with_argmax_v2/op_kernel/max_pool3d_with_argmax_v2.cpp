/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file max_pool3d_with_argmax_v2.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#if ((defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)) && !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3113))
#include "./arch35/max_pool3d_with_argmax_v2_big_kernel_regbase.h"
#include "./arch35/max_pool3d_with_argmax_v2_ksize_one.h"
#include "./arch35/max_pool3d_with_argmax_v2_gather.h"
#include "./arch35/max_pool3d_with_argmax_v2_simt.h"
#include "./arch35/max_pool3d_with_argmax_v2_nc_transpose.h"
#define SIMT_NCDHW_TILING_KEY_INT32_T256 600001
#define SIMT_NDHWC_TILING_KEY_INT32_T256 600002
#define SIMT_NCDHW_TILING_KEY_INT64_T256 600011
#define SIMT_NDHWC_TILING_KEY_INT64_T256 600012
#define SIMT_NCDHW_TILING_KEY_INT32_T512 600101
#define SIMT_NDHWC_TILING_KEY_INT32_T512 600102
#define SIMT_NCDHW_TILING_KEY_INT64_T512 600111
#define SIMT_NDHWC_TILING_KEY_INT64_T512 600112
#define BIG_KERNEL_FORMAT_NCHW 611110
#define GATHER_NO_PADDING_TILING_KEY 400001
#define GATHER_PADDING_TILING_KEY 400002
#define NC_TRANSPOSE_NO_PAD_KEY 500001
#define GATHER_NO_PADDING_CAL16_KEY 400011
#define GATHER_PADDING_CAL16_KEY 400012
#define GATHER_NO_PADDING_CAL64_KEY 400021
#define GATHER_PADDING_CAL64_KEY 400022
#define KSIZE_ONE_TILING_KEY 700001
#else
#include "max_pool3d_with_argmax_v2_nosplit.h"
#include "max_pool3d_with_argmax_v2_splitd.h"
#include "max_pool3d_with_argmax_v2_splith.h"
#include "max_pool3d_with_argmax_v2_splitw.h"
#include "max_pool3d_with_argmax_v2_huge_kernel.h"
#include "max_pool3d_with_argmax_v2_no_expand_indices.h"
#include "max_pool3d_with_argmax_big_kernel.h"
#endif

constexpr uint32_t PAD_DISABLE = 0;
constexpr uint32_t PAD_ENABLE = 1;
constexpr int64_t NCDHW = 0;
constexpr int64_t NDHWC = 1;
constexpr uint32_t THREAD_NUM_256 = 256;
constexpr uint32_t THREAD_NUM_512 = 512;

using namespace AscendC;

template <typename T>
using UbIndexType = typename std::conditional<
    std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value, int16_t, int32_t>::type;

extern "C" __global__ __aicore__ void max_pool3d_with_argmax_v2(
    GM_ADDR x, GM_ADDR y, GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling)
{
    // 100000 = 1, splitD=0 splitH=0 splitW=0 splitKernel=0 type=float(0)
    // 100001 = 1, splitD=0 splitH=0 splitW=0 splitKernel=0 type=half(1)
    // 100002 = 1, splitD=0 splitH=0 splitW=0 splitKernel=0 type=bfloat(2)
    // 110000 = 1, splitD=1 splitH=0 splitW=0 splitKernel=0 type=float(0)
    // 110001 = 1, splitD=1 splitH=0 splitW=0 splitKernel=0 type=half(1)
    // 110002 = 1, splitD=1 splitH=0 splitW=0 splitKernel=0 type=bfloat(2)
    // 111000 = 1, splitD=1 splitH=1 splitW=0 splitKernel=0 type=float(0)
    // 111001 = 1, splitD=1 splitH=1 splitW=0 splitKernel=0 type=half(1)
    // 111002 = 1, splitD=1 splitH=1 splitW=0 splitKernel=0 type=bfloat(2)
    // 111100 = 1, splitD=1 splitH=1 splitW=1 splitKernel=0 type=float(0)
    // 111101 = 1, splitD=1 splitH=1 splitW=1 splitKernel=0 type=half(1)
    // 111102 = 1, splitD=1 splitH=1 splitW=1 splitKernel=0 type=bfloat(2)
    // 111110 = 1, splitD=1 splitH=1 splitW=1 splitKernel=1 type=float(0)
    // 111111 = 1, splitD=1 splitH=1 splitW=1 splitKernel=1 type=half(1)
    // 111112 = 1, splitD=1 splitH=1 splitW=1 splitKernel=1 type=bfloat(2)
#if ((defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)) && !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3113))
    using mask_type = uint8_t;
#else
    using mask_type = uint16_t;
#endif
    TPipe pipe;

#if ((defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)) && !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3113))
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2GatherTilingData);
    using namespace MaxPool3DWithArgmaxV2WithSimt;
    using namespace MaxPool3DWithArgmaxV2WithBigKernelRegbase;
    using namespace MaxPool3DWithArgmaxV2GatherNameSpace;
    using namespace MaxPool3DWithArgmaxV2KsizeOneNameSpace;

    if (TILING_KEY_IS(KSIZE_ONE_TILING_KEY)) {
        REGISTER_TILING_FOR_TILINGKEY(
            "TILING_KEY_VAR == 700001", MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2KsizeOneTilingData);
        GET_TILING_DATA_WITH_STRUCT(
            MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2KsizeOneTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2KsizeOneTilingData* __restrict tilingData =
            &tilingDataIn;
        MaxPool3DWithArgmaxV2KsizeOneNameSpace::MaxPool3DWithArgmaxV2KsizeOneKernel<DTYPE_X, DTYPE_ARGMAX> op(
            pipe, tilingDataIn);
        op.Init(x, y, indices);
        op.Process();
    } else if (TILING_KEY_IS(GATHER_NO_PADDING_TILING_KEY)) {
        REGISTER_TILING_FOR_TILINGKEY(
            "TILING_KEY_VAR == 400001", MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2GatherTilingData);
        GET_TILING_DATA_WITH_STRUCT(
            MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2GatherTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2GatherTilingData* __restrict tilingData = &tilingDataIn;
        MaxPool3DWithArgmaxV2GatherNameSpace::MaxPool3DWithArgmaxV2GatherKernel<
            DTYPE_X, DTYPE_ARGMAX, PAD_DISABLE, UbIndexType<DTYPE_X>, int32_t>
            op(pipe, tilingDataIn);
        op.Init(x, y, indices);
        op.Process();
    } else if (TILING_KEY_IS(GATHER_PADDING_TILING_KEY)) {
        REGISTER_TILING_FOR_TILINGKEY(
            "TILING_KEY_VAR == 400002", MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2GatherTilingData);
        GET_TILING_DATA_WITH_STRUCT(
            MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2GatherTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2GatherTilingData* __restrict tilingData = &tilingDataIn;
        MaxPool3DWithArgmaxV2GatherNameSpace::MaxPool3DWithArgmaxV2GatherKernel<
            DTYPE_X, DTYPE_ARGMAX, PAD_ENABLE, UbIndexType<DTYPE_X>, int32_t>
            op(pipe, tilingDataIn);
        op.Init(x, y, indices);
        op.Process();
    } else if (TILING_KEY_IS(GATHER_NO_PADDING_CAL16_KEY)) {
        REGISTER_TILING_FOR_TILINGKEY(
            "TILING_KEY_VAR == 400011", MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2GatherTilingData);
        GET_TILING_DATA_WITH_STRUCT(
            MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2GatherTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2GatherTilingData* __restrict tilingData = &tilingDataIn;
        MaxPool3DWithArgmaxV2GatherNameSpace::MaxPool3DWithArgmaxV2GatherKernel<
            DTYPE_X, DTYPE_ARGMAX, PAD_DISABLE, UbIndexType<DTYPE_X>, int16_t>
            op(pipe, tilingDataIn);
        op.Init(x, y, indices);
        op.Process();
    } else if (TILING_KEY_IS(GATHER_PADDING_CAL16_KEY)) {
        REGISTER_TILING_FOR_TILINGKEY(
            "TILING_KEY_VAR == 400012", MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2GatherTilingData);
        GET_TILING_DATA_WITH_STRUCT(
            MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2GatherTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2GatherTilingData* __restrict tilingData = &tilingDataIn;
        MaxPool3DWithArgmaxV2GatherNameSpace::MaxPool3DWithArgmaxV2GatherKernel<
            DTYPE_X, DTYPE_ARGMAX, PAD_ENABLE, UbIndexType<DTYPE_X>, int16_t>
            op(pipe, tilingDataIn);
        op.Init(x, y, indices);
        op.Process();
    } else if (TILING_KEY_IS(GATHER_NO_PADDING_CAL64_KEY)) {
        REGISTER_TILING_FOR_TILINGKEY(
            "TILING_KEY_VAR == 400021", MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2GatherTilingData);
        GET_TILING_DATA_WITH_STRUCT(
            MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2GatherTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2GatherTilingData* __restrict tilingData = &tilingDataIn;
        MaxPool3DWithArgmaxV2GatherNameSpace::MaxPool3DWithArgmaxV2GatherKernel<
            DTYPE_X, DTYPE_ARGMAX, PAD_DISABLE, UbIndexType<DTYPE_X>, int64_t>
            op(pipe, tilingDataIn);
        op.Init(x, y, indices);
        op.Process();
    } else if (TILING_KEY_IS(GATHER_PADDING_CAL64_KEY)) {
        REGISTER_TILING_FOR_TILINGKEY(
            "TILING_KEY_VAR == 400022", MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2GatherTilingData);
        GET_TILING_DATA_WITH_STRUCT(
            MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2GatherTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2GatherTilingData* __restrict tilingData = &tilingDataIn;
        MaxPool3DWithArgmaxV2GatherNameSpace::MaxPool3DWithArgmaxV2GatherKernel<
            DTYPE_X, DTYPE_ARGMAX, PAD_ENABLE, UbIndexType<DTYPE_X>, int64_t>
            op(pipe, tilingDataIn);
        op.Init(x, y, indices);
        op.Process();
    } else if (TILING_KEY_IS(BIG_KERNEL_FORMAT_NCHW)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 611110", MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2BigKernelRegbaseTilingData);
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2BigKernelRegbaseTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2BigKernelRegbaseTilingData* __restrict tilingData = &tilingDataIn;
        MaxPool3DWithArgmaxV2WithBigKernelRegbase::MaxPool3DWithArgmaxV2BigKernelRegbase<DTYPE_X, DTYPE_ARGMAX> op(&pipe,                                                                                            tilingData);
        op.Init(x, y, indices);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_NCDHW_TILING_KEY_INT32_T256)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 600001", MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2SimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2SimtTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2SimtTilingData* __restrict tilingData = &tilingDataIn;
        MaxPool3DWithArgmaxV2WithSimt::MaxPool3DWithArgmaxV2Simt<DTYPE_X, DTYPE_ARGMAX, NCDHW, false, THREAD_NUM_256> op(tilingData);
        op.Init(x, y, indices);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_NDHWC_TILING_KEY_INT32_T256)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 600002", MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2SimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2SimtTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2SimtTilingData* __restrict tilingData = &tilingDataIn;
        MaxPool3DWithArgmaxV2WithSimt::MaxPool3DWithArgmaxV2Simt<DTYPE_X, DTYPE_ARGMAX, NDHWC, false, THREAD_NUM_256> op(tilingData);
        op.Init(x, y, indices);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_NCDHW_TILING_KEY_INT64_T256)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 600011", MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2SimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2SimtTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2SimtTilingData* __restrict tilingData = &tilingDataIn;
        MaxPool3DWithArgmaxV2WithSimt::MaxPool3DWithArgmaxV2Simt<DTYPE_X, DTYPE_ARGMAX, NCDHW, true, THREAD_NUM_256> op(tilingData);
        op.Init(x, y, indices);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_NDHWC_TILING_KEY_INT64_T256)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 600012", MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2SimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2SimtTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2SimtTilingData* __restrict tilingData = &tilingDataIn;
        MaxPool3DWithArgmaxV2WithSimt::MaxPool3DWithArgmaxV2Simt<DTYPE_X, DTYPE_ARGMAX, NDHWC, true, THREAD_NUM_256> op(tilingData);
        op.Init(x, y, indices);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_NCDHW_TILING_KEY_INT32_T512)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 600101", MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2SimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2SimtTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2SimtTilingData* __restrict tilingData = &tilingDataIn;
        MaxPool3DWithArgmaxV2WithSimt::MaxPool3DWithArgmaxV2Simt<DTYPE_X, DTYPE_ARGMAX, NCDHW, false, THREAD_NUM_512> op(tilingData);
        op.Init(x, y, indices);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_NDHWC_TILING_KEY_INT32_T512)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 600102", MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2SimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2SimtTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2SimtTilingData* __restrict tilingData = &tilingDataIn;
        MaxPool3DWithArgmaxV2WithSimt::MaxPool3DWithArgmaxV2Simt<DTYPE_X, DTYPE_ARGMAX, NDHWC, false, THREAD_NUM_512> op(tilingData);
        op.Init(x, y, indices);
        op.Process();
    } else if (TILING_KEY_IS(NC_TRANSPOSE_NO_PAD_KEY)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 500001", MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2NcTransposeTilingData);
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2NcTransposeTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2NcTransposeTilingData* __restrict tilingData = &tilingDataIn;
        using namespace MaxPool3DWithArgmaxV2NcTransposeNameSpace;
        MaxPool3DWithArgmaxV2NcTransposeKernel<DTYPE_X, DTYPE_ARGMAX, PAD_DISABLE> op(pipe, tilingDataIn);
        op.Init(x, y, indices);
        op.Process();
    }
#else
    if (TILING_KEY_IS(100000)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2NoSplitTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2NoSplitTilingData* __restrict tilingData = &tilingDataIn;
        KernelMaxPool3DWithArgmaxV2NoSplit<float, float> op(tilingData);
        op.Init(x, y, indices, nullptr, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(100001)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2NoSplitTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2NoSplitTilingData* __restrict tilingData = &tilingDataIn;
        KernelMaxPool3DWithArgmaxV2NoSplit<half, half> op(tilingData);
        op.Init(x, y, indices, nullptr, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(100002)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2NoSplitTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2NoSplitTilingData* __restrict tilingData = &tilingDataIn;
        KernelMaxPool3DWithArgmaxV2NoSplit<float, bfloat16_t> op(tilingData);
        op.Init(x, y, indices, nullptr, &pipe);
        op.Process();
#endif
    } else if (TILING_KEY_IS(110000)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2SplitDTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2SplitDTilingData* __restrict tilingData = &tilingDataIn;
        KernelMaxPool3DWithArgmaxV2SplitD<float, float> op(tilingData);
        op.Init(x, y, indices, nullptr, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(110001)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2SplitDTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2SplitDTilingData* __restrict tilingData = &tilingDataIn;
        KernelMaxPool3DWithArgmaxV2SplitD<half, half> op(tilingData);
        op.Init(x, y, indices, nullptr, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(110002)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2SplitDTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2SplitDTilingData* __restrict tilingData = &tilingDataIn;
        KernelMaxPool3DWithArgmaxV2SplitD<float, bfloat16_t> op(tilingData);
        op.Init(x, y, indices, nullptr, &pipe);
        op.Process();
#endif
    } else if (TILING_KEY_IS(111000)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2SplitHTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2SplitHTilingData* __restrict tilingData = &tilingDataIn;
        KernelMaxPool3DWithArgmaxV2SplitH<float, float> op(tilingData);
        op.Init(x, y, indices, nullptr, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(111001)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2SplitHTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2SplitHTilingData* __restrict tilingData = &tilingDataIn;
        KernelMaxPool3DWithArgmaxV2SplitH<half, half> op(tilingData);
        op.Init(x, y, indices, nullptr, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(111002)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2SplitHTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2SplitHTilingData* __restrict tilingData = &tilingDataIn;
        KernelMaxPool3DWithArgmaxV2SplitH<float, bfloat16_t> op(tilingData);
        op.Init(x, y, indices, nullptr, &pipe);
        op.Process();
#endif
    } else if (TILING_KEY_IS(111100)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2SplitWTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2SplitWTilingData* __restrict tilingData = &tilingDataIn;
        KernelMaxPool3DWithArgmaxV2SplitW<float, float> op(tilingData);
        op.Init(x, y, indices, nullptr, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(111101)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2SplitWTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2SplitWTilingData* __restrict tilingData = &tilingDataIn;
        KernelMaxPool3DWithArgmaxV2SplitW<half, half> op(tilingData);
        op.Init(x, y, indices, nullptr, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(111102)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2SplitWTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2SplitWTilingData* __restrict tilingData = &tilingDataIn;
        KernelMaxPool3DWithArgmaxV2SplitW<float, bfloat16_t> op(tilingData);
        op.Init(x, y, indices, nullptr, &pipe);
        op.Process();
#endif
    } else if (TILING_KEY_IS(111110)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2HugeKernelTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2HugeKernelTilingData* __restrict tilingData = &tilingDataIn;
        KernelMaxPool3DWithArgmaxV2HugeKernel<float, float> op(tilingData);
        op.Init(x, y, indices, GetUserWorkspace(workspace), &pipe);
        op.Process();
    } else if (TILING_KEY_IS(111111)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2HugeKernelTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2HugeKernelTilingData* __restrict tilingData = &tilingDataIn;
        KernelMaxPool3DWithArgmaxV2HugeKernel<float, half> op(tilingData);
        op.Init(x, y, indices, GetUserWorkspace(workspace), &pipe);
        op.Process();
    } else if (TILING_KEY_IS(111112)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2HugeKernelTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2HugeKernelTilingData* __restrict tilingData = &tilingDataIn;
        KernelMaxPool3DWithArgmaxV2HugeKernel<float, bfloat16_t> op(tilingData);
        op.Init(x, y, indices, GetUserWorkspace(workspace), &pipe);
        op.Process();
#endif
    } else if (TILING_KEY_IS(300001UL)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2NoExpandIndicesTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2NoExpandIndicesTilingData* __restrict tilingData = &tilingDataIn;
#if ORIG_DTYPE_X == DT_BF16
        MaxPool3DWithArgmaxV2NoExpandIndices<DTYPE_X, float, PAD_DISABLE> op;
#else
        MaxPool3DWithArgmaxV2NoExpandIndices<DTYPE_X, DTYPE_X, PAD_DISABLE> op;
#endif
        op.Init(x, y, indices, &pipe, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(300002UL)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2NoExpandIndicesTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2NoExpandIndicesTilingData* __restrict tilingData = &tilingDataIn;
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)) && ORIG_DTYPE_X == DT_BF16
        MaxPool3DWithArgmaxV2NoExpandIndices<DTYPE_X, float, PAD_ENABLE> op;
#else
        MaxPool3DWithArgmaxV2NoExpandIndices<DTYPE_X, DTYPE_X, PAD_ENABLE> op;
#endif
        op.Init(x, y, indices, &pipe, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(311110)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2BigKernelTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2BigKernelTilingData* __restrict tilingData = &tilingDataIn;
        MaxPool3DWithArgmaxBigKernel<float, float, false, mask_type> op;
        op.Init(x, y, indices, GetUserWorkspace(workspace), &pipe, tilingData, 0);
        op.Process();
    } else if (TILING_KEY_IS(311111)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2BigKernelTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2BigKernelTilingData* __restrict tilingData = &tilingDataIn;
        MaxPool3DWithArgmaxBigKernel<half, half, false, mask_type> op;
        op.Init(x, y, indices, GetUserWorkspace(workspace), &pipe, tilingData, 1);
        op.Process();
    } else if (TILING_KEY_IS(311112)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2BigKernelTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2BigKernelTilingData* __restrict tilingData = &tilingDataIn;
        MaxPool3DWithArgmaxBigKernel<bfloat16_t, float, false, mask_type> op;
        op.Init(x, y, indices, GetUserWorkspace(workspace), &pipe, tilingData, 2);
        op.Process();
#endif
    }

    return;
#endif
}