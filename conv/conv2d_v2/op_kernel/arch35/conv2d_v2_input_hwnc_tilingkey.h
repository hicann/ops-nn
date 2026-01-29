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
 * \file conv2d_v2_input_hwnc_tilingkey.h
 * \brief
 */

#ifndef CONV2D_V2_INPUT_HWNC_TILINGKEY_H
#define CONV2D_V2_INPUT_HWNC_TILINGKEY_H

#ifndef CONV_TILINGKEY_H
#include "../../common/arch35/conv_tilingkey.h"
#endif

namespace Conv2DV2Key {
using namespace ConvKey;

#if (!defined(ASCENDC_TPL_PRE) && !defined(ASCENDC_TPL_KERNEL)) ||                                                   \
    (defined(ORIG_DTYPE_X) && ((ORIG_DTYPE_X == DT_FLOAT16) || (ORIG_DTYPE_X == DT_BF16) || (ORIG_DTYPE_X == DT_FLOAT)))

#define CONV2D_COMMON_INPUT_HWNC_TPL_UINT_SEL()                                                                      \
ASCENDC_TPL_UINT_SEL(EnableSmallChannel, ASCENDC_TPL_UI_LIST,                                                        \
    CONV_ENABLE_SMALL_CHANNEL_CLOSE),                                                                                \
ASCENDC_TPL_UINT_SEL(WeightUbTrans, ASCENDC_TPL_UI_LIST,                                                             \
    CONV_WEIGHT_UB_TRANS_CLOSE),                                                                                     \
ASCENDC_TPL_UINT_SEL(FmapCopyMode, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_FMAP_LOAD3D_MODE),                                                                                          \
ASCENDC_TPL_UINT_SEL(InnerBatch, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_INNER_BATCH_SINGLE),                                                                                        \
ASCENDC_TPL_UINT_SEL(DisContinuous, ASCENDC_TPL_UI_LIST,                                                             \
    CONV_DIS_CONTINUOUS_INPUT_HWNC)

#define CONV2D_INPUT_HWNC_ONLY_MN_FULLLOAD_SEL()                                                                     \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),                                                                   \
CONV_COMMON_ONLY_MN_FULLLOAD_SEL(),                                                                                  \
CONV2D_COMMON_INPUT_HWNC_TPL_UINT_SEL()

#define CONV2D_INPUT_HWNC_NO_FULLLOAD_AL0_OPEN_SEL()                                                                 \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),                                                                   \
CONV_COMMON_NO_FULLLOAD_AL0_OPEN_SEL(),                                                                              \
CONV2D_COMMON_INPUT_HWNC_TPL_UINT_SEL()

#define CONV2D_INPUT_HWNC_NO_FULLLOAD_BL0_OPEN_SEL()                                                                 \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),                                                                   \
CONV_COMMON_NO_FULLLOAD_BL0_OPEN_SEL(),                                                                              \
CONV2D_COMMON_INPUT_HWNC_TPL_UINT_SEL()

#define CONV2D_INPUT_HWNC_NO_FULLLOAD_ALL_OPEN_SEL()                                                                 \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),                                                                   \
CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL(),                                                                              \
CONV2D_COMMON_INPUT_HWNC_TPL_UINT_SEL()

#define CONV2D_INPUT_HWNC_ONLY_AL1_FULLLOAD_SEL()                                                                    \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),                                                                   \
CONV_COMMON_ONLY_AL1_FULLLOAD_SEL(),                                                                                 \
CONV2D_COMMON_INPUT_HWNC_TPL_UINT_SEL()

#define CONV2D_INPUT_HWNC_ONLY_BL1_FULLLOAD_SEL()                                                                    \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),                                                                   \
CONV_COMMON_ONLY_BL1_FULLLOAD_SEL(),                                                                                 \
CONV2D_COMMON_INPUT_HWNC_TPL_UINT_SEL()

#define CONV2D_INPUT_HWNC_ABL1_FULLLOAD_M_FIRST_SEL()                                                                \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),                                                                   \
CONV_COMMON_ABL1_FULLLOAD_M_FIRST_SEL(),                                                                             \
CONV2D_COMMON_INPUT_HWNC_TPL_UINT_SEL()

#define CONV2D_INPUT_HWNC_ABL1_FULLLOAD_N_FIRST_SEL()                                                                \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),                                                                   \
CONV_COMMON_ABL1_FULLLOAD_N_FIRST_SEL(),                                                                             \
CONV2D_COMMON_INPUT_HWNC_TPL_UINT_SEL()

#define CONV2D_COMMON_INNER_BATCH_INPUT_HWNC_SEL()                                                                   \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),                                                                   \
ASCENDC_TPL_UINT_SEL(GroupType, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_GROUP_TYPE_NORMAL_CONV),                                                                                    \
ASCENDC_TPL_UINT_SEL(EnableSmallChannel, ASCENDC_TPL_UI_LIST,                                                        \
    CONV_ENABLE_SMALL_CHANNEL_CLOSE),                                                                                \
ASCENDC_TPL_UINT_SEL(WeightUbTrans, ASCENDC_TPL_UI_LIST,                                                             \
    CONV_WEIGHT_UB_TRANS_CLOSE),                                                                                     \
ASCENDC_TPL_UINT_SEL(FmapCopyMode, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_FMAP_LOAD3D_MODE),                                                                                          \
ASCENDC_TPL_UINT_SEL(InnerBatch, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_INNER_BATCH_KERNEL_1X1_MULTI, CONV_INNER_BATCH_MULTI),                                                      \
ASCENDC_TPL_UINT_SEL(DisContinuous, ASCENDC_TPL_UI_LIST,                                                             \
    CONV_DIS_CONTINUOUS_INPUT_HWNC)

#define CONV2D_INNER_BATCH_INPUT_HWNC_ONLY_MN_FULLLOAD_SEL()                                                         \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),                                                                   \
ASCENDC_TPL_UINT_SEL(FmapTiling, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0),                                                                       \
ASCENDC_TPL_UINT_SEL(WeightTiling, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0),                                                                     \
ASCENDC_TPL_UINT_SEL(L1PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L1_PINGPONG_ALL_CLOSE, CONV_L1_PINGPONG_AL1_OPEN, CONV_L1_PINGPONG_BL1_OPEN, CONV_L1_PINGPONG_ALL_OPEN),    \
ASCENDC_TPL_UINT_SEL(L0PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L0_PINGPONG_ALL_OPEN),                                                                                      \
ASCENDC_TPL_UINT_SEL(OutputOrder, ASCENDC_TPL_UI_LIST,                                                               \
    CONV_OUTPUT_ORDER_M_MODE),                                                                                       \
ASCENDC_TPL_UINT_SEL(IterOrder, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_ITER_ORDER_MITER_FIRST, CONV_ITER_ORDER_NITER_FIRST),                                                       \
CONV2D_COMMON_INNER_BATCH_INPUT_HWNC_SEL()

#define CONV2D_INNER_BATCH_INPUT_HWNC_NO_FULLLOAD_AL0_OPEN_SEL()                                                     \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),                                                                   \
ASCENDC_TPL_UINT_SEL(FmapTiling, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_FMAP_TILING_OTHER),                                                                                         \
ASCENDC_TPL_UINT_SEL(WeightTiling, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_WEIGHT_TILING_OTHER),                                                                                       \
ASCENDC_TPL_UINT_SEL(L1PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L1_PINGPONG_ALL_CLOSE, CONV_L1_PINGPONG_AL1_OPEN, CONV_L1_PINGPONG_BL1_OPEN, CONV_L1_PINGPONG_ALL_OPEN),    \
ASCENDC_TPL_UINT_SEL(L0PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L0_PINGPONG_AL0_OPEN),                                                                                      \
ASCENDC_TPL_UINT_SEL(OutputOrder, ASCENDC_TPL_UI_LIST,                                                               \
    CONV_OUTPUT_ORDER_M_MODE),                                                                                       \
ASCENDC_TPL_UINT_SEL(IterOrder, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_ITER_ORDER_MITER_FIRST),                                                                                    \
CONV2D_COMMON_INNER_BATCH_INPUT_HWNC_SEL()

#define CONV2D_INNER_BATCH_INPUT_HWNC_NO_FULLLOAD_BL0_OPEN_SEL()                                                     \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),                                                                   \
ASCENDC_TPL_UINT_SEL(FmapTiling, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_FMAP_TILING_OTHER),                                                                                         \
ASCENDC_TPL_UINT_SEL(WeightTiling, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_WEIGHT_TILING_OTHER),                                                                                       \
ASCENDC_TPL_UINT_SEL(L1PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L1_PINGPONG_ALL_CLOSE, CONV_L1_PINGPONG_AL1_OPEN, CONV_L1_PINGPONG_BL1_OPEN, CONV_L1_PINGPONG_ALL_OPEN),    \
ASCENDC_TPL_UINT_SEL(L0PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L0_PINGPONG_BL0_OPEN),                                                                                      \
ASCENDC_TPL_UINT_SEL(OutputOrder, ASCENDC_TPL_UI_LIST,                                                               \
    CONV_OUTPUT_ORDER_M_MODE),                                                                                       \
ASCENDC_TPL_UINT_SEL(IterOrder, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_ITER_ORDER_NITER_FIRST),                                                                                    \
CONV2D_COMMON_INNER_BATCH_INPUT_HWNC_SEL()

#define CONV2D_INNER_BATCH_INPUT_HWNC_NO_FULLLOAD_ALL_OPEN_SEL()                                                     \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),                                                                   \
ASCENDC_TPL_UINT_SEL(FmapTiling, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_FMAP_TILING_OTHER),                                                                                         \
ASCENDC_TPL_UINT_SEL(WeightTiling, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_WEIGHT_TILING_OTHER),                                                                                       \
ASCENDC_TPL_UINT_SEL(L1PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L1_PINGPONG_ALL_CLOSE, CONV_L1_PINGPONG_AL1_OPEN, CONV_L1_PINGPONG_BL1_OPEN, CONV_L1_PINGPONG_ALL_OPEN),    \
ASCENDC_TPL_UINT_SEL(L0PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L0_PINGPONG_ALL_OPEN),                                                                                      \
ASCENDC_TPL_UINT_SEL(OutputOrder, ASCENDC_TPL_UI_LIST,                                                               \
    CONV_OUTPUT_ORDER_M_MODE),                                                                                       \
ASCENDC_TPL_UINT_SEL(IterOrder, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_ITER_ORDER_MITER_FIRST, CONV_ITER_ORDER_NITER_FIRST),                                                       \
CONV2D_COMMON_INNER_BATCH_INPUT_HWNC_SEL()

#define CONV2D_INNER_BATCH_INPUT_HWNC_ONLY_AL1_FULLLOAD_SEL()                                                        \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),                                                                   \
ASCENDC_TPL_UINT_SEL(FmapTiling, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_FMAP_TILING_FULLLOAD_AL1),                                                                                  \
ASCENDC_TPL_UINT_SEL(WeightTiling, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_WEIGHT_TILING_OTHER),                                                                                       \
ASCENDC_TPL_UINT_SEL(L1PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L1_PINGPONG_ALL_CLOSE, CONV_L1_PINGPONG_BL1_OPEN),                                                          \
ASCENDC_TPL_UINT_SEL(L0PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L0_PINGPONG_BL0_OPEN, CONV_L0_PINGPONG_ALL_OPEN),                                                           \
ASCENDC_TPL_UINT_SEL(OutputOrder, ASCENDC_TPL_UI_LIST,                                                               \
    CONV_OUTPUT_ORDER_M_MODE),                                                                                       \
ASCENDC_TPL_UINT_SEL(IterOrder, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_ITER_ORDER_NITER_FIRST),                                                                                    \
CONV2D_COMMON_INNER_BATCH_INPUT_HWNC_SEL()

#define CONV2D_INNER_BATCH_INPUT_HWNC_ONLY_BL1_FULLLOAD_SEL()                                                        \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),                                                                   \
ASCENDC_TPL_UINT_SEL(FmapTiling, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_FMAP_TILING_OTHER),                                                                                         \
ASCENDC_TPL_UINT_SEL(WeightTiling, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_WEIGHT_TILING_FULLLOAD_BL1),                                                                                \
ASCENDC_TPL_UINT_SEL(L1PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L1_PINGPONG_ALL_CLOSE, CONV_L1_PINGPONG_AL1_OPEN),                                                          \
ASCENDC_TPL_UINT_SEL(L0PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L0_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_AL0_OPEN, CONV_L0_PINGPONG_ALL_OPEN),                               \
ASCENDC_TPL_UINT_SEL(OutputOrder, ASCENDC_TPL_UI_LIST,                                                               \
    CONV_OUTPUT_ORDER_M_MODE),                                                                                       \
ASCENDC_TPL_UINT_SEL(IterOrder, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_ITER_ORDER_MITER_FIRST),                                                                                    \
CONV2D_COMMON_INNER_BATCH_INPUT_HWNC_SEL()

#define CONV2D_INNER_BATCH_INPUT_HWNC_ABL1_FULLLOAD_M_FIRST_SEL()                                                    \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),                                                                   \
ASCENDC_TPL_UINT_SEL(FmapTiling, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_FMAP_TILING_FULLLOAD_AL1),                                                                                  \
ASCENDC_TPL_UINT_SEL(WeightTiling, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_WEIGHT_TILING_FULLLOAD_BL1),                                                                                \
ASCENDC_TPL_UINT_SEL(L1PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L1_PINGPONG_ALL_CLOSE),                                                                                     \
ASCENDC_TPL_UINT_SEL(L0PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L0_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_AL0_OPEN, CONV_L0_PINGPONG_ALL_OPEN),                               \
ASCENDC_TPL_UINT_SEL(OutputOrder, ASCENDC_TPL_UI_LIST,                                                               \
    CONV_OUTPUT_ORDER_M_MODE),                                                                                       \
ASCENDC_TPL_UINT_SEL(IterOrder, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_ITER_ORDER_MITER_FIRST),                                                                                    \
CONV2D_COMMON_INNER_BATCH_INPUT_HWNC_SEL()

#define CONV2D_INNER_BATCH_INPUT_HWNC_ABL1_FULLLOAD_N_FIRST_SEL()                                                    \
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),                                                                   \
ASCENDC_TPL_UINT_SEL(FmapTiling, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_FMAP_TILING_FULLLOAD_AL1),                                                                                  \
ASCENDC_TPL_UINT_SEL(WeightTiling, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_WEIGHT_TILING_FULLLOAD_BL1),                                                                                \
ASCENDC_TPL_UINT_SEL(L1PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L1_PINGPONG_ALL_CLOSE),                                                                                     \
ASCENDC_TPL_UINT_SEL(L0PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L0_PINGPONG_BL0_OPEN, CONV_L0_PINGPONG_ALL_OPEN),                                                           \
ASCENDC_TPL_UINT_SEL(OutputOrder, ASCENDC_TPL_UI_LIST,                                                               \
    CONV_OUTPUT_ORDER_M_MODE),                                                                                       \
ASCENDC_TPL_UINT_SEL(IterOrder, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_ITER_ORDER_NITER_FIRST),                                                                                    \
CONV2D_COMMON_INNER_BATCH_INPUT_HWNC_SEL()

#else
#define CONV2D_INPUT_HWNC_ONLY_MN_FULLLOAD_SEL()
#define CONV2D_INPUT_HWNC_NO_FULLLOAD_AL0_OPEN_SEL()
#define CONV2D_INPUT_HWNC_NO_FULLLOAD_BL0_OPEN_SEL()
#define CONV2D_INPUT_HWNC_NO_FULLLOAD_ALL_OPEN_SEL()
#define CONV2D_INPUT_HWNC_ONLY_AL1_FULLLOAD_SEL()
#define CONV2D_INPUT_HWNC_ONLY_BL1_FULLLOAD_SEL()
#define CONV2D_INPUT_HWNC_ABL1_FULLLOAD_M_FIRST_SEL()
#define CONV2D_INPUT_HWNC_ABL1_FULLLOAD_N_FIRST_SEL()
#define CONV2D_INNER_BATCH_INPUT_HWNC_ONLY_MN_FULLLOAD_SEL()
#define CONV2D_INNER_BATCH_INPUT_HWNC_NO_FULLLOAD_AL0_OPEN_SEL()
#define CONV2D_INNER_BATCH_INPUT_HWNC_NO_FULLLOAD_BL0_OPEN_SEL()
#define CONV2D_INNER_BATCH_INPUT_HWNC_NO_FULLLOAD_ALL_OPEN_SEL()
#define CONV2D_INNER_BATCH_INPUT_HWNC_ONLY_AL1_FULLLOAD_SEL()
#define CONV2D_INNER_BATCH_INPUT_HWNC_ONLY_BL1_FULLLOAD_SEL()
#define CONV2D_INNER_BATCH_INPUT_HWNC_ABL1_FULLLOAD_M_FIRST_SEL()
#define CONV2D_INNER_BATCH_INPUT_HWNC_ABL1_FULLLOAD_N_FIRST_SEL()
#endif

}

#endif  // CONV2D_V2_INPUT_HWNC_TILINGKEY_H