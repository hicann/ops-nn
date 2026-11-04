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
 * \file conv2dv2_apt.cpp
 * \brief
 */

#define K_MAX_SHAPE_DIM 0

#include "../inc/macro.h"
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202) && !defined(KERNELUT)
#include "conv2d_v2_common_tilingdata.h"
using Conv2DTilingData = Ops::NN::Conv2dV2::Conv2DTilingData;
#endif
#include "arch35/conv2d_v2.h"
#include "arch35/conv2d_v2_group.h"
#include "arch35/conv2d_v2_tilingkey.h"
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
#include "arch35/conv2d_v2_aswt.h"
#endif
#include "arch35/conv2d_small_kernel.h"
#include "arch35/conv2d_small_kernel_parallelism.h"
#include "arch35/conv2d_small_kernel_fm_partload.h"
#include "arch35/conv2d_v2_depthwise_simplify.h"
#include "conv2d_v2_tiling_data.h"
#include "arch35/full_mn_kernel/full_mn_kernel_pipeline.h"
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5162)
#include "arch51/conv2d_v2_instr_impl_51.h"
#include "../../common/arch51/conv_common_func_51.h"
#include "../../common/arch51/conv_iterate_impl_51.h"
#endif

using namespace AscendC;
using namespace Conv2DV2Key;

#if defined(FORMAT_X) && FORMAT_X == FORMAT_NCHW && defined(FORMAT_FILTER) && FORMAT_FILTER == FORMAT_NCHW && \
    defined(FORMAT_Y) && FORMAT_Y == FORMAT_NCHW
constexpr ConvFormat fmapFormat = ConvFormat::NCHW;
constexpr ConvFormat weightFormat = ConvFormat::NCHW;
constexpr ConvFormat outputFormat = ConvFormat::NCHW;
#elif defined(FORMAT_X) && FORMAT_X == FORMAT_NHWC && defined(FORMAT_FILTER) && FORMAT_FILTER == FORMAT_HWCN && \
    defined(FORMAT_Y) && FORMAT_Y == FORMAT_NHWC
constexpr ConvFormat fmapFormat = ConvFormat::NHWC;
constexpr ConvFormat weightFormat = ConvFormat::HWCN;
constexpr ConvFormat outputFormat = ConvFormat::NHWC;
#elif defined(FORMAT_X) && FORMAT_X == FORMAT_NCHW && defined(FORMAT_FILTER) && FORMAT_FILTER == FORMAT_FRACTAL_Z && \
    defined(FORMAT_Y) && FORMAT_Y == FORMAT_NCHW
constexpr ConvFormat fmapFormat = ConvFormat::NCHW;
constexpr ConvFormat weightFormat = ConvFormat::FRACTAL_Z;
constexpr ConvFormat outputFormat = ConvFormat::NCHW;
#elif defined(FORMAT_X) && FORMAT_X == FORMAT_NHWC && defined(FORMAT_FILTER) && FORMAT_FILTER == FORMAT_FRACTAL_Z && \
    defined(FORMAT_Y) && FORMAT_Y == FORMAT_NHWC
constexpr ConvFormat fmapFormat = ConvFormat::NHWC;
constexpr ConvFormat weightFormat = ConvFormat::FRACTAL_Z;
constexpr ConvFormat outputFormat = ConvFormat::NHWC;
#elif defined(FORMAT_X) && FORMAT_X == FORMAT_NCHW && defined(FORMAT_FILTER) && \
    FORMAT_FILTER == FORMAT_FRACTAL_Z_C04 && defined(FORMAT_Y) && FORMAT_Y == FORMAT_NCHW
constexpr ConvFormat fmapFormat = ConvFormat::NCHW;
constexpr ConvFormat weightFormat = ConvFormat::FRACTAL_Z_C04;
constexpr ConvFormat outputFormat = ConvFormat::NCHW;
#elif defined(FORMAT_X) && FORMAT_X == FORMAT_NHWC && defined(FORMAT_FILTER) && \
    FORMAT_FILTER == FORMAT_FRACTAL_Z_C04 && defined(FORMAT_Y) && FORMAT_Y == FORMAT_NHWC
constexpr ConvFormat fmapFormat = ConvFormat::NHWC;
constexpr ConvFormat weightFormat = ConvFormat::FRACTAL_Z_C04;
constexpr ConvFormat outputFormat = ConvFormat::NHWC;
#elif defined(FORMAT_X) && FORMAT_X == FORMAT_NCHW && defined(FORMAT_FILTER) && FORMAT_FILTER == FORMAT_FRACTAL_Z && \
    defined(FORMAT_Y) && FORMAT_Y == FORMAT_NHWC
constexpr ConvFormat fmapFormat = ConvFormat::NCHW;
constexpr ConvFormat weightFormat = ConvFormat::FRACTAL_Z;
constexpr ConvFormat outputFormat = ConvFormat::NHWC;
#elif defined(FORMAT_X) && FORMAT_X == FORMAT_NHWC && defined(FORMAT_FILTER) && FORMAT_FILTER == FORMAT_FRACTAL_Z && \
    defined(FORMAT_Y) && FORMAT_Y == FORMAT_NCHW
constexpr ConvFormat fmapFormat = ConvFormat::NHWC;
constexpr ConvFormat weightFormat = ConvFormat::FRACTAL_Z;
constexpr ConvFormat outputFormat = ConvFormat::NCHW;
#elif defined(FORMAT_X) && FORMAT_X == FORMAT_NCHW && defined(FORMAT_FILTER) && \
    FORMAT_FILTER == FORMAT_FRACTAL_Z_C04 && defined(FORMAT_Y) && FORMAT_Y == FORMAT_NHWC
constexpr ConvFormat fmapFormat = ConvFormat::NCHW;
constexpr ConvFormat weightFormat = ConvFormat::FRACTAL_Z_C04;
constexpr ConvFormat outputFormat = ConvFormat::NHWC;
#elif defined(FORMAT_X) && FORMAT_X == FORMAT_NHWC && defined(FORMAT_FILTER) && \
    FORMAT_FILTER == FORMAT_FRACTAL_Z_C04 && defined(FORMAT_Y) && FORMAT_Y == FORMAT_NCHW
constexpr ConvFormat fmapFormat = ConvFormat::NHWC;
constexpr ConvFormat weightFormat = ConvFormat::FRACTAL_Z_C04;
constexpr ConvFormat outputFormat = ConvFormat::NCHW;
#elif defined(FORMAT_X) && FORMAT_X == FORMAT_NCHW && defined(FORMAT_FILTER) && FORMAT_FILTER == FORMAT_NHWC && \
    defined(FORMAT_Y) && FORMAT_Y == FORMAT_NCHW
constexpr ConvFormat fmapFormat = ConvFormat::NCHW;
constexpr ConvFormat weightFormat = ConvFormat::NHWC;
constexpr ConvFormat outputFormat = ConvFormat::NCHW;
#endif
constexpr ConvFormat biasFormat = ConvFormat::ND;
constexpr ConvFormat scaleFormat = ConvFormat::ND;

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
#define CONV2D_V2_ASWT_FULL_MN_DISPATCH()                                                                              \
    do {                                                                                                               \
        if constexpr (IsASWT == CONV_ASWT_ENABLE) {                                                                    \
            using aswtConvCfg = Conv2DV1Param<FmapTiling, WeightTiling, L1PingPong, L0PingPong, OutputOrder,           \
                                              GroupType, EnableSmallChannel, WeightUbTrans, FmapCopyMode, InnerBatch,  \
                                              BatchOne, NoPad, SmallWeight>;                                           \
            ConvAWST::ASWTRunConv2dKernelByRegion<fmapType, weightType, outputType, biasType, scaleType, aswtConvCfg>( \
                x, filter, bias, y, tilingData);                                                                       \
            return;                                                                                                    \
        } else if constexpr (SmallKernel == CONV_NOT_SMALL_KERNEL &&                                                   \
                             FmapTiling == CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0 &&                                 \
                             WeightTiling == CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0 &&                             \
                             L1PingPong == CONV_L1_PINGPONG_ALL_OPEN && L0PingPong == CONV_L0_PINGPONG_ALL_OPEN &&     \
                             OutputOrder == CONV_OUTPUT_ORDER_M_MODE && GroupType == CONV_GROUP_TYPE_NORMAL_CONV &&    \
                             EnableSmallChannel == CONV_ENABLE_SMALL_CHANNEL_CLOSE &&                                  \
                             WeightUbTrans == CONV_WEIGHT_UB_TRANS_CLOSE && FmapCopyMode == CONV_FMAP_LOAD3D_MODE &&   \
                             InnerBatch == CONV_INNER_BATCH_SINGLE && fmapFormat == ConvFormat::NCHW &&                \
                             outputFormat == ConvFormat::NCHW && weightFormat == ConvFormat::NHWC) {                   \
            REGISTER_TILING_FOR_TILINGKEY("FmapTiling == CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0 && "                 \
                                          "WeightTiling == CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0 && "             \
                                          "L1PingPong == CONV_L1_PINGPONG_ALL_OPEN &&"                                 \
                                          "L0PingPong == CONV_L0_PINGPONG_ALL_OPEN &&"                                 \
                                          "OutputOrder == CONV_OUTPUT_ORDER_M_MODE &&"                                 \
                                          "GroupType == CONV_GROUP_TYPE_NORMAL_CONV &&"                                \
                                          "EnableSmallChannel == CONV_ENABLE_SMALL_CHANNEL_CLOSE &&"                   \
                                          "WeightUbTrans == CONV_WEIGHT_UB_TRANS_CLOSE &&"                             \
                                          "FmapCopyMode == CONV_FMAP_LOAD3D_MODE &&"                                   \
                                          "InnerBatch == CONV_INNER_BATCH_SINGLE",                                     \
                                          Ops::NN::Conv2dV2::Conv2DMNFullLoadKPartialTiling);                          \
            GET_TILING_DATA_WITH_STRUCT(Ops::NN::Conv2dV2::Conv2DMNFullLoadKPartialTiling, tilingData, tiling);        \
            if ASCEND_IS_AIC_CONV {                                                                                    \
                full_mn::Conv2dFullMNKernelPipeline<DTYPE_X, DTYPE_FILTER, DTYPE_Y, biasType::T,                       \
                                                    Ops::NN::Conv2dV2::Conv2DMNFullLoadKPartialTiling>                 \
                    pipeline;                                                                                          \
                pipeline(tilingData, x, filter, bias, y);                                                              \
            }                                                                                                          \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)
#else
#define CONV2D_V2_ASWT_FULL_MN_DISPATCH() ((void)0)
#endif

template <int8_t FmapTiling, int8_t WeightTiling, int8_t L1PingPong, int8_t L0PingPong, int8_t OutputOrder,
          int8_t GroupType, int8_t EnableSmallChannel, int8_t WeightUbTrans, int8_t FmapCopyMode, int8_t InnerBatch,
          int8_t BatchOne, int8_t NoPad, int8_t SmallWeight, int8_t SmallKernel, int8_t IsASWT, int8_t ConvToL1 = 0>
__global__ __aicore__ void conv2dv2(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR offset_w, GM_ADDR y,
                                    GM_ADDR workspace, GM_ADDR tiling)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202) && !defined(KERNELUT)
    REGISTER_TILING_DEFAULT(Ops::NN::Conv2dV2::Conv2DTilingData);
#endif
    GET_TILING_DATA(tilingData, tiling);

#if defined(DTYPE_X) && defined(DTYPE_FILTER) && defined(DTYPE_Y)
    using fmapType = ConvType<TPosition::GM, fmapFormat, DTYPE_X>;
    using weightType = ConvType<TPosition::GM, weightFormat, DTYPE_FILTER>;
    using outputType = ConvType<TPosition::GM, outputFormat, DTYPE_Y>;
#if defined(DTYPE_BIAS)
    using biasType = ConvType<TPosition::GM, biasFormat, DTYPE_BIAS>;
#else
    using biasType = ConvType<TPosition::GM, biasFormat, half>; // only for compile
#endif
    using scaleType = ConvType<TPosition::GM, scaleFormat, uint64_t>;

    CONV2D_V2_ASWT_FULL_MN_DISPATCH();

    if constexpr (SmallKernel == CONV_SMALL_KERNEL_FM_PARTLOAD) {
        constexpr bool isNHWCin = (fmapFormat == ConvFormat::NHWC);
        constexpr bool isNHWCout = (outputFormat == ConvFormat::NHWC);
        constexpr bool isHw = (OutputOrder == static_cast<int8_t>(ConvOutputOrder::HW_MODE));
        Conv2dSmallKernelFmPartload<DTYPE_X, DTYPE_FILTER, biasType::T, DTYPE_Y, half, isNHWCin, isNHWCout, isHw> op;
        op.Init(tilingData);
        op.Process(x, filter, bias, y, nullptr);
    } else if constexpr (SmallKernel == CONV_SMALL_KERNEL) {
        constexpr bool isNHWCin = (fmapFormat == ConvFormat::NHWC);
        constexpr bool isNHWCout = (outputFormat == ConvFormat::NHWC);
        constexpr bool isHw = (OutputOrder == static_cast<int8_t>(ConvOutputOrder::HW_MODE));

        if constexpr (weightFormat == ConvFormat::FRACTAL_Z && AscendC::IsSameType<DTYPE_X, half>::value) {
            const static uint32_t GK0 = C0_SIZE / sizeof(DTYPE_FILTER);
            uint32_t cinAligned = AlignB(tilingData.singleCoreCi, GK0);
            bool isParallelism = false;
            if constexpr (!isHw) { // The current L1 splitting is not suitable for the HW mode.
                if (tilingData.kernelHxkernelW == 1 && cinAligned >= 2 * 2 * GK0) {
                    isParallelism = true;
                } else if (tilingData.kernelHxkernelW != 1 && cinAligned >= 2 * GK0) {
                    isParallelism = true;
                }
            }

            if (isParallelism) {
                Conv2dSmallKernelParallelism<DTYPE_X, DTYPE_FILTER, biasType::T, DTYPE_Y, half, isNHWCin, isNHWCout,
                                             isHw>
                    op;
                op.Init(tilingData);
                op.Process(x, filter, bias, y, nullptr);
            } else {
                Conv2dSmallKernel<DTYPE_X, DTYPE_FILTER, biasType::T, DTYPE_Y, half, isNHWCin, isNHWCout,
                                  ConvFormat::FRACTAL_Z, isHw>
                    op;
                op.Init(tilingData);
                op.Process(x, filter, bias, y, nullptr);
            }
        } else {
            Conv2dSmallKernel<DTYPE_X, DTYPE_FILTER, biasType::T, DTYPE_Y, half, isNHWCin, isNHWCout, weightFormat,
                              isHw>
                op;
            op.Init(tilingData);
            op.Process(x, filter, bias, y, nullptr);
        }
    } else {
        if constexpr (GroupType == CONV_GROUP_TYPE_NORMAL_CONV) {
            Conv2dBase<fmapType, weightType, outputType, biasType, scaleType,
                       Conv2DV1Param<FmapTiling, WeightTiling, L1PingPong, L0PingPong, OutputOrder, GroupType,
                                     EnableSmallChannel, WeightUbTrans, FmapCopyMode, InnerBatch, BatchOne, NoPad,
                                     SmallWeight, IsExtendConv2D::FALSE, half, 0, ConvToL1>>
                baseConv2d;
            baseConv2d.RunConv2dKernel(x, filter, bias, y, tilingData);
        } else if constexpr (GroupType == CONV_GROUP_TYPE_OPT_SIMPLIFIED_GROUP_CONV) {
            DepthwiseConv2dSimplifiedKernel<
                Conv2DV1Param<FmapTiling, WeightTiling, L1PingPong, L0PingPong, OutputOrder, GroupType,
                              EnableSmallChannel, WeightUbTrans, FmapCopyMode, InnerBatch>,
                DTYPE_X, fmapFormat>
                depthwiseConv2d;
            depthwiseConv2d.Init(x, filter, bias, y, &tilingData);
            depthwiseConv2d.Process();
        } else {
            GroupConv2d<fmapType, weightType, outputType, biasType, scaleType,
                        Conv2DV1Param<FmapTiling, WeightTiling, L1PingPong, L0PingPong, OutputOrder, GroupType,
                                      EnableSmallChannel, WeightUbTrans, FmapCopyMode, InnerBatch, BatchOne, NoPad,
                                      SmallWeight, IsExtendConv2D::FALSE, half, 0, ConvToL1>>
                groupConv2d;
            groupConv2d.RunConv2dKernel(x, filter, bias, y, tilingData);
        }
    }

#endif
}
