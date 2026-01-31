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
 * \file conv2d_v2_tilingkey.h
 * \brief
 */

#ifndef CONV2D_V2_TILINGKEY_H
#define CONV2D_V2_TILINGKEY_H

#ifndef CONV_TILINGKEY_H
#include "../../common/arch35/conv_tilingkey.h"
#endif
#include "conv2d_v2_inner_batch_tilingkey.h"
#include "conv2d_v2_input_hwnc_tilingkey.h"

namespace Conv2DV2Key {
using namespace ConvKey;

ASCENDC_TPL_ARGS_DECL(Conv2DV2,
ASCENDC_TPL_UINT_DECL(FmapTiling, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
    CONV_FMAP_TILING_FULLLOAD_AL1, CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0, CONV_FMAP_TILING_OTHER),
ASCENDC_TPL_UINT_DECL(WeightTiling, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
    CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0, CONV_WEIGHT_TILING_OTHER),
ASCENDC_TPL_UINT_DECL(L1PingPong, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
    CONV_L1_PINGPONG_ALL_CLOSE, CONV_L1_PINGPONG_AL1_OPEN, CONV_L1_PINGPONG_BL1_OPEN, CONV_L1_PINGPONG_ALL_OPEN),
ASCENDC_TPL_UINT_DECL(L0PingPong, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
    CONV_L0_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_AL0_OPEN, CONV_L0_PINGPONG_BL0_OPEN, CONV_L0_PINGPONG_ALL_OPEN),
ASCENDC_TPL_UINT_DECL(OutputOrder, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST,
    CONV_OUTPUT_ORDER_HW_MODE, CONV_OUTPUT_ORDER_M_MODE),
ASCENDC_TPL_UINT_DECL(IterOrder, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST,
    CONV_ITER_ORDER_MITER_FIRST, CONV_ITER_ORDER_NITER_FIRST),
ASCENDC_TPL_UINT_DECL(GroupType, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
    CONV_GROUP_TYPE_NORMAL_CONV, CONV_GROUP_TYPE_ORI_GROUP_CONV, CONV_GROUP_TYPE_OPT_GROUP_CONV),
ASCENDC_TPL_UINT_DECL(EnableSmallChannel, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST,
    CONV_ENABLE_SMALL_CHANNEL_CLOSE, CONV_ENABLE_SMALL_CHANNEL_OPEN),
ASCENDC_TPL_UINT_DECL(WeightUbTrans, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST,
    CONV_WEIGHT_UB_TRANS_CLOSE, CONV_WEIGHT_UB_TRANS_OPEN),
ASCENDC_TPL_UINT_DECL(FmapCopyMode, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
    CONV_FMAP_LOAD3D_MODE, CONV_FMAP_DMA_MODE),
ASCENDC_TPL_UINT_DECL(InnerBatch, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
    CONV_INNER_BATCH_SINGLE, CONV_INNER_BATCH_KERNEL_1X1_MULTI, CONV_INNER_BATCH_MULTI),
ASCENDC_TPL_UINT_DECL(DisContinuous, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
    CONV_DIS_CONTINUOUS_CLOSE, CONV_DIS_CONTINUOUS_INPUT_HWNC)
);

#define CONV2D_COMMON_C04_TPL_UINT_SEL()                                                                             \
ASCENDC_TPL_UINT_SEL(EnableSmallChannel, ASCENDC_TPL_UI_LIST,                                                        \
    CONV_ENABLE_SMALL_CHANNEL_CLOSE),                                                                                \
ASCENDC_TPL_UINT_SEL(WeightUbTrans, ASCENDC_TPL_UI_LIST,                                                             \
    CONV_WEIGHT_UB_TRANS_CLOSE),                                                                                     \
ASCENDC_TPL_UINT_SEL(FmapCopyMode, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_FMAP_LOAD3D_MODE),                                                                                          \
ASCENDC_TPL_UINT_SEL(InnerBatch, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_INNER_BATCH_SINGLE),                                                                                        \
ASCENDC_TPL_UINT_SEL(DisContinuous, ASCENDC_TPL_UI_LIST,                                                             \
    CONV_DIS_CONTINUOUS_CLOSE)

#define CONV2D_COMMON_C04_MIXCORE_TPL_UINT_SEL()                                                                     \
ASCENDC_TPL_UINT_SEL(EnableSmallChannel, ASCENDC_TPL_UI_LIST,                                                        \
    CONV_ENABLE_SMALL_CHANNEL_OPEN),                                                                                 \
ASCENDC_TPL_UINT_SEL(WeightUbTrans, ASCENDC_TPL_UI_LIST,                                                             \
    CONV_WEIGHT_UB_TRANS_CLOSE),                                                                                     \
ASCENDC_TPL_UINT_SEL(FmapCopyMode, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_FMAP_LOAD3D_MODE),                                                                                          \
ASCENDC_TPL_UINT_SEL(InnerBatch, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_INNER_BATCH_SINGLE),                                                                                        \
ASCENDC_TPL_UINT_SEL(DisContinuous, ASCENDC_TPL_UI_LIST,                                                             \
    CONV_DIS_CONTINUOUS_CLOSE, CONV_DIS_CONTINUOUS_INPUT_HWNC)

#define CONV2D_ONLY_MN_FULLLOAD_SEL()                                                                                \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),                                                                   \
CONV_COMMON_ONLY_MN_FULLLOAD_SEL(),                                                                                  \
CONV2D_COMMON_C04_TPL_UINT_SEL()

#define CONV2D_NO_FULLLOAD_AL0_OPEN_SEL()                                                                            \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),                                                                   \
CONV_COMMON_NO_FULLLOAD_AL0_OPEN_SEL(),                                                                              \
CONV2D_COMMON_C04_TPL_UINT_SEL()

#define CONV2D_NO_FULLLOAD_BL0_OPEN_SEL()                                                                            \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),                                                                   \
CONV_COMMON_NO_FULLLOAD_BL0_OPEN_SEL(),                                                                              \
CONV2D_COMMON_C04_TPL_UINT_SEL()

#define CONV2D_NO_FULLLOAD_ALL_OPEN_SEL()                                                                            \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),                                                                   \
CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL(),                                                                              \
CONV2D_COMMON_C04_TPL_UINT_SEL()

#define CONV2D_ONLY_AL1_FULLLOAD_SEL()                                                                               \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),                                                                   \
CONV_COMMON_ONLY_AL1_FULLLOAD_SEL(),                                                                                 \
CONV2D_COMMON_C04_TPL_UINT_SEL()

#define CONV2D_ONLY_BL1_FULLLOAD_SEL()                                                                               \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),                                                                   \
CONV_COMMON_ONLY_BL1_FULLLOAD_SEL(),                                                                                 \
CONV2D_COMMON_C04_TPL_UINT_SEL()

#define CONV2D_ABL1_FULLLOAD_M_FIRST_SEL()                                                                           \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),                                                                   \
CONV_COMMON_ABL1_FULLLOAD_M_FIRST_SEL(),                                                                             \
CONV2D_COMMON_C04_TPL_UINT_SEL()

#define CONV2D_ABL1_FULLLOAD_N_FIRST_SEL()                                                                           \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),                                                                   \
CONV_COMMON_ABL1_FULLLOAD_N_FIRST_SEL(),                                                                             \
CONV2D_COMMON_C04_TPL_UINT_SEL()

#if (!defined(ASCENDC_TPL_PRE) && !defined(ASCENDC_TPL_KERNEL)) ||                                                   \
    (defined(ORIG_DTYPE_X) && ((ORIG_DTYPE_X == DT_FLOAT16) || (ORIG_DTYPE_X == DT_BF16) || (ORIG_DTYPE_X == DT_FLOAT)))
#define CONV2D_ONLY_MN_FULLLOAD_MIXCORE_SEL()                                                                        \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),                                                                \
CONV_COMMON_ONLY_MN_FULLLOAD_SEL(),                                                                                  \
CONV2D_COMMON_C04_MIXCORE_TPL_UINT_SEL()

#define CONV2D_NO_FULLLOAD_AL0_OPEN_MIXCORE_SEL()                                                                    \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),                                                                \
CONV_COMMON_NO_FULLLOAD_AL0_OPEN_SEL(),                                                                              \
CONV2D_COMMON_C04_MIXCORE_TPL_UINT_SEL()

#define CONV2D_NO_FULLLOAD_BL0_OPEN_MIXCORE_SEL()                                                                    \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),                                                                \
CONV_COMMON_NO_FULLLOAD_BL0_OPEN_SEL(),                                                                              \
CONV2D_COMMON_C04_MIXCORE_TPL_UINT_SEL()

#define CONV2D_NO_FULLLOAD_ALL_OPEN_MIXCORE_SEL()                                                                    \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),                                                                \
CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL(),                                                                              \
CONV2D_COMMON_C04_MIXCORE_TPL_UINT_SEL()

#define CONV2D_ONLY_AL1_FULLLOAD_MIXCORE_SEL()                                                                       \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),                                                                \
CONV_COMMON_ONLY_AL1_FULLLOAD_SEL(),                                                                                 \
CONV2D_COMMON_C04_MIXCORE_TPL_UINT_SEL()

#define CONV2D_ONLY_BL1_FULLLOAD_MIXCORE_SEL()                                                                       \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),                                                                \
CONV_COMMON_ONLY_BL1_FULLLOAD_SEL(),                                                                                 \
CONV2D_COMMON_C04_MIXCORE_TPL_UINT_SEL()

#define CONV2D_ABL1_FULLLOAD_M_FIRST_MIXCORE_SEL()                                                                   \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),                                                                \
CONV_COMMON_ABL1_FULLLOAD_M_FIRST_SEL(),                                                                             \
CONV2D_COMMON_C04_MIXCORE_TPL_UINT_SEL()

#define CONV2D_ABL1_FULLLOAD_N_FIRST_MIXCORE_SEL()                                                                   \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),                                                                \
CONV_COMMON_ABL1_FULLLOAD_N_FIRST_SEL(),                                                                             \
CONV2D_COMMON_C04_MIXCORE_TPL_UINT_SEL()
#else
#define CONV2D_ONLY_MN_FULLLOAD_MIXCORE_SEL()
#define CONV2D_NO_FULLLOAD_AL0_OPEN_MIXCORE_SEL()
#define CONV2D_NO_FULLLOAD_BL0_OPEN_MIXCORE_SEL()
#define CONV2D_NO_FULLLOAD_ALL_OPEN_MIXCORE_SEL()
#define CONV2D_ONLY_AL1_FULLLOAD_MIXCORE_SEL()
#define CONV2D_ONLY_BL1_FULLLOAD_MIXCORE_SEL()
#define CONV2D_ABL1_FULLLOAD_M_FIRST_MIXCORE_SEL()
#define CONV2D_ABL1_FULLLOAD_N_FIRST_MIXCORE_SEL()
#endif

#define CONV2D_OPT_GROUP_SEL()                                                                                       \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),                                                                \
CONV_COMMON_OPT_GROUP_SEL(),                                                                                         \
ASCENDC_TPL_UINT_SEL(EnableSmallChannel, ASCENDC_TPL_UI_LIST,                                                        \
    CONV_ENABLE_SMALL_CHANNEL_CLOSE),                                                                                \
ASCENDC_TPL_UINT_SEL(WeightUbTrans, ASCENDC_TPL_UI_LIST,                                                             \
    CONV_WEIGHT_UB_TRANS_CLOSE),                                                                                     \
ASCENDC_TPL_UINT_SEL(FmapCopyMode, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_FMAP_LOAD3D_MODE),                                                                                          \
ASCENDC_TPL_UINT_SEL(InnerBatch, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_INNER_BATCH_SINGLE),                                                                                        \
ASCENDC_TPL_UINT_SEL(DisContinuous, ASCENDC_TPL_UI_LIST,                                                             \
    CONV_DIS_CONTINUOUS_CLOSE)

#define CONV2D_ORI_GROUP_SEL()                                                                                       \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),                                                                   \
CONV_COMMON_ORI_GROUP_SEL(),                                                                                         \
ASCENDC_TPL_UINT_SEL(EnableSmallChannel, ASCENDC_TPL_UI_LIST,                                                        \
    CONV_ENABLE_SMALL_CHANNEL_CLOSE),                                                                                \
ASCENDC_TPL_UINT_SEL(WeightUbTrans, ASCENDC_TPL_UI_LIST,                                                             \
    CONV_WEIGHT_UB_TRANS_CLOSE),                                                                                     \
ASCENDC_TPL_UINT_SEL(FmapCopyMode, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_FMAP_LOAD3D_MODE),                                                                                          \
ASCENDC_TPL_UINT_SEL(InnerBatch, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_INNER_BATCH_SINGLE),                                                                                        \
ASCENDC_TPL_UINT_SEL(DisContinuous, ASCENDC_TPL_UI_LIST,                                                             \
    CONV_DIS_CONTINUOUS_CLOSE)

// Weight Ub Trans Mode TilingKey SEL
#if (!defined(ASCENDC_TPL_PRE) && !defined(ASCENDC_TPL_KERNEL)) ||                                                   \
     (defined(FORMAT_X) && FORMAT_X == FORMAT_NCHW && defined(FORMAT_FILTER) && FORMAT_FILTER == FORMAT_NCHW &&       \
     ((defined(FORMAT_Y) && FORMAT_Y == FORMAT_NCHW) || (defined(FORMAT_Y0) && FORMAT_Y0 == FORMAT_NCHW)))

#define CONV2D_COMMON_WEIGHT_UB_SEL()                                                                                \
ASCENDC_TPL_UINT_SEL(GroupType, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_GROUP_TYPE_NORMAL_CONV),                                                                                    \
ASCENDC_TPL_UINT_SEL(EnableSmallChannel, ASCENDC_TPL_UI_LIST,                                                        \
    CONV_ENABLE_SMALL_CHANNEL_CLOSE),                                                                                \
ASCENDC_TPL_UINT_SEL(WeightUbTrans, ASCENDC_TPL_UI_LIST,                                                             \
    CONV_WEIGHT_UB_TRANS_OPEN),                                                                                      \
ASCENDC_TPL_UINT_SEL(FmapCopyMode, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_FMAP_LOAD3D_MODE),                                                                                          \
ASCENDC_TPL_UINT_SEL(InnerBatch, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_INNER_BATCH_SINGLE),                                                                                        \
ASCENDC_TPL_UINT_SEL(DisContinuous, ASCENDC_TPL_UI_LIST,                                                             \
    CONV_DIS_CONTINUOUS_CLOSE, CONV_DIS_CONTINUOUS_INPUT_HWNC)

#define CONV2D_WEIGHT_UB_ONLY_MN_FULLLOAD_SEL()                                                                      \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),                                                                \
ASCENDC_TPL_UINT_SEL(FmapTiling, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0),                                                                       \
ASCENDC_TPL_UINT_SEL(WeightTiling, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0),                                                                     \
ASCENDC_TPL_UINT_SEL(L1PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L1_PINGPONG_BL1_OPEN, CONV_L1_PINGPONG_ALL_OPEN),                                                           \
ASCENDC_TPL_UINT_SEL(L0PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L0_PINGPONG_ALL_OPEN),                                                                                      \
ASCENDC_TPL_UINT_SEL(OutputOrder, ASCENDC_TPL_UI_LIST,                                                               \
    CONV_OUTPUT_ORDER_M_MODE),                                                                                       \
ASCENDC_TPL_UINT_SEL(IterOrder, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_ITER_ORDER_MITER_FIRST, CONV_ITER_ORDER_NITER_FIRST),                                                       \
CONV2D_COMMON_WEIGHT_UB_SEL()

#define CONV2D_WEIGHT_UB_ONLY_AL1_FULLLOAD_SEL()                                                                     \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),                                                                \
ASCENDC_TPL_UINT_SEL(FmapTiling, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_FMAP_TILING_FULLLOAD_AL1),                                                                                  \
ASCENDC_TPL_UINT_SEL(WeightTiling, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_WEIGHT_TILING_OTHER),                                                                                       \
ASCENDC_TPL_UINT_SEL(L1PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L1_PINGPONG_BL1_OPEN),                                                                                      \
ASCENDC_TPL_UINT_SEL(L0PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L0_PINGPONG_BL0_OPEN, CONV_L0_PINGPONG_ALL_OPEN),                                                           \
ASCENDC_TPL_UINT_SEL(OutputOrder, ASCENDC_TPL_UI_LIST,                                                               \
    CONV_OUTPUT_ORDER_HW_MODE, CONV_OUTPUT_ORDER_M_MODE),                                                            \
ASCENDC_TPL_UINT_SEL(IterOrder, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_ITER_ORDER_NITER_FIRST),                                                                                    \
CONV2D_COMMON_WEIGHT_UB_SEL()

#define CONV2D_WEIGHT_UB_NO_FULLLOAD_AL0_OPEN_SEL()                                                                  \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),                                                                \
ASCENDC_TPL_UINT_SEL(FmapTiling, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_FMAP_TILING_OTHER),                                                                                         \
ASCENDC_TPL_UINT_SEL(WeightTiling, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_WEIGHT_TILING_OTHER),                                                                                       \
ASCENDC_TPL_UINT_SEL(L1PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L1_PINGPONG_BL1_OPEN, CONV_L1_PINGPONG_ALL_OPEN),                                                           \
ASCENDC_TPL_UINT_SEL(L0PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L0_PINGPONG_AL0_OPEN),                                                                                      \
ASCENDC_TPL_UINT_SEL(OutputOrder, ASCENDC_TPL_UI_LIST,                                                               \
    CONV_OUTPUT_ORDER_HW_MODE, CONV_OUTPUT_ORDER_M_MODE),                                                            \
ASCENDC_TPL_UINT_SEL(IterOrder, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_ITER_ORDER_MITER_FIRST),                                                                                    \
CONV2D_COMMON_WEIGHT_UB_SEL()

#define CONV2D_WEIGHT_UB_NO_FULLLOAD_BL0_OPEN_SEL()                                                                  \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),                                                                \
ASCENDC_TPL_UINT_SEL(FmapTiling, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_FMAP_TILING_OTHER),                                                                                         \
ASCENDC_TPL_UINT_SEL(WeightTiling, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_WEIGHT_TILING_OTHER),                                                                                       \
ASCENDC_TPL_UINT_SEL(L1PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L1_PINGPONG_BL1_OPEN, CONV_L1_PINGPONG_ALL_OPEN),                                                           \
ASCENDC_TPL_UINT_SEL(L0PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L0_PINGPONG_BL0_OPEN),                                                                                      \
ASCENDC_TPL_UINT_SEL(OutputOrder, ASCENDC_TPL_UI_LIST,                                                               \
    CONV_OUTPUT_ORDER_HW_MODE, CONV_OUTPUT_ORDER_M_MODE),                                                            \
ASCENDC_TPL_UINT_SEL(IterOrder, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_ITER_ORDER_NITER_FIRST),                                                                                    \
CONV2D_COMMON_WEIGHT_UB_SEL()

#define CONV2D_WEIGHT_UB_NO_FULLLOAD_ALL_OPEN_SEL()                                                                  \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),                                                                \
ASCENDC_TPL_UINT_SEL(FmapTiling, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_FMAP_TILING_OTHER),                                                                                         \
ASCENDC_TPL_UINT_SEL(WeightTiling, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_WEIGHT_TILING_OTHER),                                                                                       \
ASCENDC_TPL_UINT_SEL(L1PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L1_PINGPONG_BL1_OPEN, CONV_L1_PINGPONG_ALL_OPEN),                                                           \
ASCENDC_TPL_UINT_SEL(L0PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L0_PINGPONG_ALL_OPEN),                                                                                      \
ASCENDC_TPL_UINT_SEL(OutputOrder, ASCENDC_TPL_UI_LIST,                                                               \
    CONV_OUTPUT_ORDER_HW_MODE, CONV_OUTPUT_ORDER_M_MODE),                                                            \
ASCENDC_TPL_UINT_SEL(IterOrder, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_ITER_ORDER_MITER_FIRST, CONV_ITER_ORDER_NITER_FIRST),                                                       \
CONV2D_COMMON_WEIGHT_UB_SEL()

#else
#define CONV2D_WEIGHT_UB_ONLY_MN_FULLLOAD_SEL()
#define CONV2D_WEIGHT_UB_ONLY_AL1_FULLLOAD_SEL()
#define CONV2D_WEIGHT_UB_NO_FULLLOAD_AL0_OPEN_SEL()
#define CONV2D_WEIGHT_UB_NO_FULLLOAD_BL0_OPEN_SEL()
#define CONV2D_WEIGHT_UB_NO_FULLLOAD_ALL_OPEN_SEL()
#endif

// Fmap DMA Mode TilingKey SEL
#define CONV2D_COMMON_DMA_SEL()                                                                                      \
ASCENDC_TPL_UINT_SEL(GroupType, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_GROUP_TYPE_NORMAL_CONV, CONV_GROUP_TYPE_ORI_GROUP_CONV),                                                    \
ASCENDC_TPL_UINT_SEL(EnableSmallChannel, ASCENDC_TPL_UI_LIST,                                                        \
    CONV_ENABLE_SMALL_CHANNEL_CLOSE),                                                                                \
ASCENDC_TPL_UINT_SEL(WeightUbTrans, ASCENDC_TPL_UI_LIST,                                                             \
    CONV_WEIGHT_UB_TRANS_CLOSE),                                                                                     \
ASCENDC_TPL_UINT_SEL(FmapCopyMode, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_FMAP_DMA_MODE),                                                                                             \
ASCENDC_TPL_UINT_SEL(InnerBatch, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_INNER_BATCH_SINGLE),                                                                                        \
ASCENDC_TPL_UINT_SEL(DisContinuous, ASCENDC_TPL_UI_LIST,                                                             \
    CONV_DIS_CONTINUOUS_CLOSE)

#define CONV2D_DMA_L1_FULLLOAD_L0B_CLOSE_SEL()                                                                       \
ASCENDC_TPL_UINT_SEL(FmapTiling, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_FMAP_TILING_FULLLOAD_AL1),                                                                                  \
ASCENDC_TPL_UINT_SEL(WeightTiling, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_WEIGHT_TILING_FULLLOAD_BL1),                                                                                \
ASCENDC_TPL_UINT_SEL(L1PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L1_PINGPONG_ALL_CLOSE),                                                                                     \
ASCENDC_TPL_UINT_SEL(L0PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L0_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_AL0_OPEN),                                                          \
ASCENDC_TPL_UINT_SEL(OutputOrder, ASCENDC_TPL_UI_LIST,                                                               \
    CONV_OUTPUT_ORDER_HW_MODE),                                                                                      \
ASCENDC_TPL_UINT_SEL(IterOrder, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_ITER_ORDER_MITER_FIRST),                                                                                    \
CONV2D_COMMON_DMA_SEL()
 
#define CONV2D_DMA_L1_FULLLOAD_ONLY_L0B_OPEN_SEL()                                                                   \
ASCENDC_TPL_UINT_SEL(FmapTiling, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_FMAP_TILING_FULLLOAD_AL1),                                                                                  \
ASCENDC_TPL_UINT_SEL(WeightTiling, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_WEIGHT_TILING_FULLLOAD_BL1),                                                                                \
ASCENDC_TPL_UINT_SEL(L1PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L1_PINGPONG_ALL_CLOSE),                                                                                     \
ASCENDC_TPL_UINT_SEL(L0PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L0_PINGPONG_BL0_OPEN),                                                                                      \
ASCENDC_TPL_UINT_SEL(OutputOrder, ASCENDC_TPL_UI_LIST,                                                               \
    CONV_OUTPUT_ORDER_HW_MODE),                                                                                      \
ASCENDC_TPL_UINT_SEL(IterOrder, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_ITER_ORDER_NITER_FIRST),                                                                                    \
CONV2D_COMMON_DMA_SEL()
 
#define CONV2D_DMA_L1_FULLLOAD_ALL_OPEN_SEL()                                                                        \
ASCENDC_TPL_UINT_SEL(FmapTiling, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_FMAP_TILING_FULLLOAD_AL1),                                                                                  \
ASCENDC_TPL_UINT_SEL(WeightTiling, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_WEIGHT_TILING_FULLLOAD_BL1),                                                                                \
ASCENDC_TPL_UINT_SEL(L1PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L1_PINGPONG_ALL_CLOSE),                                                                                     \
ASCENDC_TPL_UINT_SEL(L0PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L0_PINGPONG_ALL_OPEN),                                                                                      \
ASCENDC_TPL_UINT_SEL(OutputOrder, ASCENDC_TPL_UI_LIST,                                                               \
    CONV_OUTPUT_ORDER_HW_MODE),                                                                                      \
ASCENDC_TPL_UINT_SEL(IterOrder, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_ITER_ORDER_MITER_FIRST, CONV_ITER_ORDER_NITER_FIRST),                                                       \
CONV2D_COMMON_DMA_SEL()

#define CONV2D_DMA_AL1_NO_FULLLOAD_M_FIRST_SEL()                                                                     \
ASCENDC_TPL_UINT_SEL(FmapTiling, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_FMAP_TILING_OTHER),                                                                                         \
ASCENDC_TPL_UINT_SEL(WeightTiling, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_WEIGHT_TILING_OTHER),                                                      \
ASCENDC_TPL_UINT_SEL(L1PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L1_PINGPONG_ALL_CLOSE),                                                                                     \
ASCENDC_TPL_UINT_SEL(L0PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L0_PINGPONG_AL0_OPEN, CONV_L0_PINGPONG_ALL_OPEN),                                                           \
ASCENDC_TPL_UINT_SEL(OutputOrder, ASCENDC_TPL_UI_LIST,                                                               \
    CONV_OUTPUT_ORDER_HW_MODE),                                                                                      \
ASCENDC_TPL_UINT_SEL(IterOrder, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_ITER_ORDER_MITER_FIRST),                                                                                    \
CONV2D_COMMON_DMA_SEL()
 
#define CONV2D_DMA_BL1_NO_FULLLOAD_N_FIRST_SEL()                                                                     \
ASCENDC_TPL_UINT_SEL(FmapTiling, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_FMAP_TILING_FULLLOAD_AL1, CONV_FMAP_TILING_OTHER),                                                          \
ASCENDC_TPL_UINT_SEL(WeightTiling, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_WEIGHT_TILING_OTHER),                                                                                       \
ASCENDC_TPL_UINT_SEL(L1PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L1_PINGPONG_ALL_CLOSE),                                                                                     \
ASCENDC_TPL_UINT_SEL(L0PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L0_PINGPONG_BL0_OPEN, CONV_L0_PINGPONG_ALL_OPEN),                                                           \
ASCENDC_TPL_UINT_SEL(OutputOrder, ASCENDC_TPL_UI_LIST,                                                               \
    CONV_OUTPUT_ORDER_HW_MODE),                                                                                      \
ASCENDC_TPL_UINT_SEL(IterOrder, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_ITER_ORDER_NITER_FIRST),                                                                                    \
CONV2D_COMMON_DMA_SEL()

ASCENDC_TPL_SEL(
ASCENDC_TPL_ARGS_SEL(CONV2D_ONLY_MN_FULLLOAD_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_NO_FULLLOAD_AL0_OPEN_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_NO_FULLLOAD_BL0_OPEN_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_NO_FULLLOAD_ALL_OPEN_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_ONLY_AL1_FULLLOAD_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_ONLY_BL1_FULLLOAD_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_ABL1_FULLLOAD_M_FIRST_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_ABL1_FULLLOAD_N_FIRST_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_ONLY_MN_FULLLOAD_MIXCORE_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_NO_FULLLOAD_AL0_OPEN_MIXCORE_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_NO_FULLLOAD_BL0_OPEN_MIXCORE_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_NO_FULLLOAD_ALL_OPEN_MIXCORE_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_ONLY_AL1_FULLLOAD_MIXCORE_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_ONLY_BL1_FULLLOAD_MIXCORE_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_ABL1_FULLLOAD_M_FIRST_MIXCORE_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_ABL1_FULLLOAD_N_FIRST_MIXCORE_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_OPT_GROUP_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_ORI_GROUP_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_WEIGHT_UB_ONLY_MN_FULLLOAD_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_WEIGHT_UB_ONLY_AL1_FULLLOAD_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_WEIGHT_UB_NO_FULLLOAD_AL0_OPEN_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_WEIGHT_UB_NO_FULLLOAD_BL0_OPEN_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_WEIGHT_UB_NO_FULLLOAD_ALL_OPEN_SEL()),
ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2), CONV2D_DMA_L1_FULLLOAD_L0B_CLOSE_SEL()),
ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2), CONV2D_DMA_L1_FULLLOAD_ONLY_L0B_OPEN_SEL()),
ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2), CONV2D_DMA_L1_FULLLOAD_ALL_OPEN_SEL()),
ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2), CONV2D_DMA_AL1_NO_FULLLOAD_M_FIRST_SEL()),
ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2), CONV2D_DMA_BL1_NO_FULLLOAD_N_FIRST_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_INNER_BATCH_WEIGHT_UB_ONLY_MN_FULLLOAD_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_INNER_BATCH_WEIGHT_UB_ONLY_AL1_FULLLOAD_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_INNER_BATCH_WEIGHT_UB_NO_FULLLOAD_AL0_OPEN_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_INNER_BATCH_WEIGHT_UB_NO_FULLLOAD_BL0_OPEN_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_INNER_BATCH_WEIGHT_UB_NO_FULLLOAD_ALL_OPEN_SEL()),
ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY), CONV_INNER_BATCH_ONLY_MN_FULLLOAD_SEL()),
ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY), CONV_INNER_BATCH_NO_FULLLOAD_AL0_OPEN_SEL()),
ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY), CONV_INNER_BATCH_NO_FULLLOAD_BL0_OPEN_SEL()),
ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY), CONV_INNER_BATCH_NO_FULLLOAD_ALL_OPEN_SEL()),
ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY), CONV_INNER_BATCH_ONLY_AL1_FULLLOAD_SEL()),
ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY), CONV_INNER_BATCH_ONLY_BL1_FULLLOAD_SEL()),
ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY), CONV_INNER_BATCH_ABL1_FULLLOAD_M_FIRST_SEL()),
ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY), CONV_INNER_BATCH_ABL1_FULLLOAD_N_FIRST_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_INPUT_HWNC_ONLY_MN_FULLLOAD_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_INPUT_HWNC_NO_FULLLOAD_AL0_OPEN_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_INPUT_HWNC_NO_FULLLOAD_BL0_OPEN_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_INPUT_HWNC_NO_FULLLOAD_ALL_OPEN_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_INPUT_HWNC_ONLY_AL1_FULLLOAD_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_INPUT_HWNC_ONLY_BL1_FULLLOAD_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_INPUT_HWNC_ABL1_FULLLOAD_M_FIRST_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_INPUT_HWNC_ABL1_FULLLOAD_N_FIRST_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_INNER_BATCH_INPUT_HWNC_ONLY_MN_FULLLOAD_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_INNER_BATCH_INPUT_HWNC_NO_FULLLOAD_AL0_OPEN_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_INNER_BATCH_INPUT_HWNC_NO_FULLLOAD_BL0_OPEN_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_INNER_BATCH_INPUT_HWNC_NO_FULLLOAD_ALL_OPEN_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_INNER_BATCH_INPUT_HWNC_ONLY_AL1_FULLLOAD_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_INNER_BATCH_INPUT_HWNC_ONLY_BL1_FULLLOAD_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_INNER_BATCH_INPUT_HWNC_ABL1_FULLLOAD_M_FIRST_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV2D_INNER_BATCH_INPUT_HWNC_ABL1_FULLLOAD_N_FIRST_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV_INNER_BATCH_ONLY_WEIGHT_FULLLOAD_L0_ALLCLOSE_ENABLE_C04_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV_INNER_BATCH_ONLY_MN_FULLLOAD_ENABLE_C04_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV_INNER_BATCH_NO_FULLLOAD_AL0_OPEN_ENABLE_C04_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV_INNER_BATCH_NO_FULLLOAD_BL0_OPEN_ENABLE_C04_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV_INNER_BATCH_NO_FULLLOAD_ALL_OPEN_ENABLE_C04_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV_INNER_BATCH_ONLY_AL1_FULLLOAD_ENABLE_C04_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV_INNER_BATCH_ONLY_BL1_FULLLOAD_ENABLE_C04_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV_INNER_BATCH_ABL1_FULLLOAD_M_FIRST_ENABLE_C04_SEL()),
ASCENDC_TPL_ARGS_SEL(CONV_INNER_BATCH_ABL1_FULLLOAD_N_FIRST_ENABLE_C04_SEL()),
);

}

#endif  // CONV2D_V2_TILINGKEY_H