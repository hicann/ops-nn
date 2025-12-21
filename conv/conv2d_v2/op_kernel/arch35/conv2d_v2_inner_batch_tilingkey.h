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
 * \file conv2dv2_innter_batch_tilingkey.h
 * \brief
 */

#ifndef CONV2DV2_INNER_BATCH_TILINGKEY_H
#define CONV2DV2_INNER_BATCH_TILINGKEY_H

#include "conv_tilingkey.h"

namespace Conv2DV2Key {
using namespace ConvKey;

// Weight Ub Trans Mode TilingKey SEL
#if (!defined(ASCENDC_TPL_PRE) && !defined(ASCENDC_TPL_KERNEL)) ||                                                   \
    (defined(ORIG_DTYPE_X) && ((ORIG_DTYPE_X == DT_FLOAT16) || (ORIG_DTYPE_X == DT_BF16) || (ORIG_DTYPE_X == DT_FLOAT)) &&                         \
     defined(FORMAT_X) && FORMAT_X == FORMAT_NCHW && defined(FORMAT_FILTER) && FORMAT_FILTER == FORMAT_NCHW &&       \
     defined(FORMAT_Y) && FORMAT_Y == FORMAT_NCHW)

#define CONV2D_INNER_BATCH_WEIGHT_UB_ONLY_MN_FULLLOAD_SEL()                                                          \
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
ASCENDC_TPL_UINT_SEL(GroupType, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_GROUP_TYPE_NORMAL_CONV),                                                                                    \
ASCENDC_TPL_UINT_SEL(EnableSmallChannel, ASCENDC_TPL_UI_LIST,                                                        \
    CONV_ENABLE_SMALL_CHANNEL_CLOSE),                                                                                \
ASCENDC_TPL_UINT_SEL(WeightUbTrans, ASCENDC_TPL_UI_LIST,                                                             \
    CONV_WEIGHT_UB_TRANS_OPEN),                                                                                      \
ASCENDC_TPL_UINT_SEL(FmapCopyMode, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_FMAP_LOAD3D_MODE),                                                                                          \
ASCENDC_TPL_UINT_SEL(InnerBatch, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_INNER_BATCH_MULTI)

#define CONV2D_INNER_BATCH_WEIGHT_UB_ONLY_AL1_FULLLOAD_SEL()                                                         \
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
ASCENDC_TPL_UINT_SEL(GroupType, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_GROUP_TYPE_NORMAL_CONV),                                                                                    \
ASCENDC_TPL_UINT_SEL(EnableSmallChannel, ASCENDC_TPL_UI_LIST,                                                        \
    CONV_ENABLE_SMALL_CHANNEL_CLOSE),                                                                                \
ASCENDC_TPL_UINT_SEL(WeightUbTrans, ASCENDC_TPL_UI_LIST,                                                             \
    CONV_WEIGHT_UB_TRANS_OPEN),                                                                                      \
ASCENDC_TPL_UINT_SEL(FmapCopyMode, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_FMAP_LOAD3D_MODE),                                                                                          \
ASCENDC_TPL_UINT_SEL(InnerBatch, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_INNER_BATCH_MULTI)

#define CONV2D_INNER_BATCH_WEIGHT_UB_NO_FULLLOAD_AL0_OPEN_SEL()                                                      \
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
ASCENDC_TPL_UINT_SEL(GroupType, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_GROUP_TYPE_NORMAL_CONV),                                                                                    \
ASCENDC_TPL_UINT_SEL(EnableSmallChannel, ASCENDC_TPL_UI_LIST,                                                        \
    CONV_ENABLE_SMALL_CHANNEL_CLOSE),                                                                                \
ASCENDC_TPL_UINT_SEL(WeightUbTrans, ASCENDC_TPL_UI_LIST,                                                             \
    CONV_WEIGHT_UB_TRANS_OPEN),                                                                                      \
ASCENDC_TPL_UINT_SEL(FmapCopyMode, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_FMAP_LOAD3D_MODE),                                                                                          \
ASCENDC_TPL_UINT_SEL(InnerBatch, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_INNER_BATCH_MULTI)

#define CONV2D_INNER_BATCH_WEIGHT_UB_NO_FULLLOAD_BL0_OPEN_SEL()                                                      \
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
ASCENDC_TPL_UINT_SEL(GroupType, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_GROUP_TYPE_NORMAL_CONV),                                                                                    \
ASCENDC_TPL_UINT_SEL(EnableSmallChannel, ASCENDC_TPL_UI_LIST,                                                        \
    CONV_ENABLE_SMALL_CHANNEL_CLOSE),                                                                                \
ASCENDC_TPL_UINT_SEL(WeightUbTrans, ASCENDC_TPL_UI_LIST,                                                             \
    CONV_WEIGHT_UB_TRANS_OPEN),                                                                                      \
ASCENDC_TPL_UINT_SEL(FmapCopyMode, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_FMAP_LOAD3D_MODE),                                                                                          \
ASCENDC_TPL_UINT_SEL(InnerBatch, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_INNER_BATCH_MULTI)

#define CONV2D_INNER_BATCH_WEIGHT_UB_NO_FULLLOAD_ALL_OPEN_SEL()                                                      \
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
ASCENDC_TPL_UINT_SEL(GroupType, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_GROUP_TYPE_NORMAL_CONV),                                                                                    \
ASCENDC_TPL_UINT_SEL(EnableSmallChannel, ASCENDC_TPL_UI_LIST,                                                        \
    CONV_ENABLE_SMALL_CHANNEL_CLOSE),                                                                                \
ASCENDC_TPL_UINT_SEL(WeightUbTrans, ASCENDC_TPL_UI_LIST,                                                             \
    CONV_WEIGHT_UB_TRANS_OPEN),                                                                                      \
ASCENDC_TPL_UINT_SEL(FmapCopyMode, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_FMAP_LOAD3D_MODE),                                                                                          \
ASCENDC_TPL_UINT_SEL(InnerBatch, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_INNER_BATCH_MULTI)

#else
#define CONV2D_INNER_BATCH_WEIGHT_UB_ONLY_MN_FULLLOAD_SEL()
#define CONV2D_INNER_BATCH_WEIGHT_UB_ONLY_AL1_FULLLOAD_SEL()
#define CONV2D_INNER_BATCH_WEIGHT_UB_NO_FULLLOAD_AL0_OPEN_SEL()
#define CONV2D_INNER_BATCH_WEIGHT_UB_NO_FULLLOAD_BL0_OPEN_SEL()
#define CONV2D_INNER_BATCH_WEIGHT_UB_NO_FULLLOAD_ALL_OPEN_SEL()
#endif

#define CONV_INNER_BATCH_ONLY_MN_FULLLOAD_SEL()                                                                      \
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
ASCENDC_TPL_UINT_SEL(GroupType, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_GROUP_TYPE_NORMAL_CONV),                                                                                    \
ASCENDC_TPL_UINT_SEL(EnableSmallChannel, ASCENDC_TPL_UI_LIST,                                                        \
    CONV_ENABLE_SMALL_CHANNEL_CLOSE),                                                                                \
ASCENDC_TPL_UINT_SEL(WeightUbTrans, ASCENDC_TPL_UI_LIST,                                                             \
    CONV_WEIGHT_UB_TRANS_CLOSE),                                                                                      \
ASCENDC_TPL_UINT_SEL(FmapCopyMode, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_FMAP_LOAD3D_MODE),                                                                                          \
ASCENDC_TPL_UINT_SEL(InnerBatch, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_INNER_BATCH_KERNEL_1X1_MULTI, CONV_INNER_BATCH_MULTI)

#define CONV_INNER_BATCH_NO_FULLLOAD_AL0_OPEN_SEL()                                                                  \
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
ASCENDC_TPL_UINT_SEL(GroupType, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_GROUP_TYPE_NORMAL_CONV),                                                                                    \
ASCENDC_TPL_UINT_SEL(EnableSmallChannel, ASCENDC_TPL_UI_LIST,                                                        \
    CONV_ENABLE_SMALL_CHANNEL_CLOSE),                                                                                \
ASCENDC_TPL_UINT_SEL(WeightUbTrans, ASCENDC_TPL_UI_LIST,                                                             \
    CONV_WEIGHT_UB_TRANS_CLOSE),                                                                                      \
ASCENDC_TPL_UINT_SEL(FmapCopyMode, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_FMAP_LOAD3D_MODE),                                                                                          \
ASCENDC_TPL_UINT_SEL(InnerBatch, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_INNER_BATCH_KERNEL_1X1_MULTI, CONV_INNER_BATCH_MULTI)

#define CONV_INNER_BATCH_NO_FULLLOAD_BL0_OPEN_SEL()                                                                  \
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
ASCENDC_TPL_UINT_SEL(GroupType, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_GROUP_TYPE_NORMAL_CONV),                                                                                    \
ASCENDC_TPL_UINT_SEL(EnableSmallChannel, ASCENDC_TPL_UI_LIST,                                                        \
    CONV_ENABLE_SMALL_CHANNEL_CLOSE),                                                                                \
ASCENDC_TPL_UINT_SEL(WeightUbTrans, ASCENDC_TPL_UI_LIST,                                                             \
    CONV_WEIGHT_UB_TRANS_CLOSE),                                                                                      \
ASCENDC_TPL_UINT_SEL(FmapCopyMode, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_FMAP_LOAD3D_MODE),                                                                                          \
ASCENDC_TPL_UINT_SEL(InnerBatch, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_INNER_BATCH_KERNEL_1X1_MULTI, CONV_INNER_BATCH_MULTI)

#define CONV_INNER_BATCH_NO_FULLLOAD_ALL_OPEN_SEL()                                                                  \
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
ASCENDC_TPL_UINT_SEL(GroupType, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_GROUP_TYPE_NORMAL_CONV),                                                                                    \
ASCENDC_TPL_UINT_SEL(EnableSmallChannel, ASCENDC_TPL_UI_LIST,                                                        \
    CONV_ENABLE_SMALL_CHANNEL_CLOSE),                                                                                \
ASCENDC_TPL_UINT_SEL(WeightUbTrans, ASCENDC_TPL_UI_LIST,                                                             \
    CONV_WEIGHT_UB_TRANS_CLOSE),                                                                                      \
ASCENDC_TPL_UINT_SEL(FmapCopyMode, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_FMAP_LOAD3D_MODE),                                                                                          \
ASCENDC_TPL_UINT_SEL(InnerBatch, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_INNER_BATCH_KERNEL_1X1_MULTI, CONV_INNER_BATCH_MULTI)

#define CONV_INNER_BATCH_ONLY_AL1_FULLLOAD_SEL()                                                                     \
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
ASCENDC_TPL_UINT_SEL(GroupType, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_GROUP_TYPE_NORMAL_CONV),                                                                                    \
ASCENDC_TPL_UINT_SEL(EnableSmallChannel, ASCENDC_TPL_UI_LIST,                                                        \
    CONV_ENABLE_SMALL_CHANNEL_CLOSE),                                                                                \
ASCENDC_TPL_UINT_SEL(WeightUbTrans, ASCENDC_TPL_UI_LIST,                                                             \
    CONV_WEIGHT_UB_TRANS_CLOSE),                                                                                      \
ASCENDC_TPL_UINT_SEL(FmapCopyMode, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_FMAP_LOAD3D_MODE),                                                                                          \
ASCENDC_TPL_UINT_SEL(InnerBatch, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_INNER_BATCH_KERNEL_1X1_MULTI, CONV_INNER_BATCH_MULTI)

#define CONV_INNER_BATCH_ONLY_BL1_FULLLOAD_SEL()                                                                     \
ASCENDC_TPL_UINT_SEL(FmapTiling, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_FMAP_TILING_OTHER),                                                                                         \
ASCENDC_TPL_UINT_SEL(WeightTiling, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_WEIGHT_TILING_FULLLOAD_BL1),                                                                                \
ASCENDC_TPL_UINT_SEL(L1PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L1_PINGPONG_ALL_CLOSE, CONV_L1_PINGPONG_AL1_OPEN),                                                          \
ASCENDC_TPL_UINT_SEL(L0PingPong, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_L0_PINGPONG_AL0_OPEN, CONV_L0_PINGPONG_ALL_OPEN),                                                           \
ASCENDC_TPL_UINT_SEL(OutputOrder, ASCENDC_TPL_UI_LIST,                                                               \
    CONV_OUTPUT_ORDER_M_MODE),                                                                                       \
ASCENDC_TPL_UINT_SEL(IterOrder, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_ITER_ORDER_MITER_FIRST),                                                                                    \
ASCENDC_TPL_UINT_SEL(GroupType, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_GROUP_TYPE_NORMAL_CONV),                                                                                    \
ASCENDC_TPL_UINT_SEL(EnableSmallChannel, ASCENDC_TPL_UI_LIST,                                                        \
    CONV_ENABLE_SMALL_CHANNEL_CLOSE),                                                                                \
ASCENDC_TPL_UINT_SEL(WeightUbTrans, ASCENDC_TPL_UI_LIST,                                                             \
    CONV_WEIGHT_UB_TRANS_CLOSE),                                                                                      \
ASCENDC_TPL_UINT_SEL(FmapCopyMode, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_FMAP_LOAD3D_MODE),                                                                                          \
ASCENDC_TPL_UINT_SEL(InnerBatch, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_INNER_BATCH_KERNEL_1X1_MULTI, CONV_INNER_BATCH_MULTI)

#define CONV_INNER_BATCH_ABL1_FULLLOAD_M_FIRST_SEL()                                                                 \
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
ASCENDC_TPL_UINT_SEL(GroupType, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_GROUP_TYPE_NORMAL_CONV),                                                                                    \
ASCENDC_TPL_UINT_SEL(EnableSmallChannel, ASCENDC_TPL_UI_LIST,                                                        \
    CONV_ENABLE_SMALL_CHANNEL_CLOSE),                                                                                \
ASCENDC_TPL_UINT_SEL(WeightUbTrans, ASCENDC_TPL_UI_LIST,                                                             \
    CONV_WEIGHT_UB_TRANS_CLOSE),                                                                                      \
ASCENDC_TPL_UINT_SEL(FmapCopyMode, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_FMAP_LOAD3D_MODE),                                                                                          \
ASCENDC_TPL_UINT_SEL(InnerBatch, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_INNER_BATCH_KERNEL_1X1_MULTI, CONV_INNER_BATCH_MULTI)

#define CONV_INNER_BATCH_ABL1_FULLLOAD_N_FIRST_SEL()                                                                 \
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
ASCENDC_TPL_UINT_SEL(GroupType, ASCENDC_TPL_UI_LIST,                                                                 \
    CONV_GROUP_TYPE_NORMAL_CONV),                                                                                    \
ASCENDC_TPL_UINT_SEL(EnableSmallChannel, ASCENDC_TPL_UI_LIST,                                                        \
    CONV_ENABLE_SMALL_CHANNEL_CLOSE),                                                                                \
ASCENDC_TPL_UINT_SEL(WeightUbTrans, ASCENDC_TPL_UI_LIST,                                                             \
    CONV_WEIGHT_UB_TRANS_CLOSE),                                                                                      \
ASCENDC_TPL_UINT_SEL(FmapCopyMode, ASCENDC_TPL_UI_LIST,                                                              \
    CONV_FMAP_LOAD3D_MODE),                                                                                          \
ASCENDC_TPL_UINT_SEL(InnerBatch, ASCENDC_TPL_UI_LIST,                                                                \
    CONV_INNER_BATCH_KERNEL_1X1_MULTI, CONV_INNER_BATCH_MULTI)

}

#endif  // CONV2DV2_INNER_BATCH_TILINGKEY_H