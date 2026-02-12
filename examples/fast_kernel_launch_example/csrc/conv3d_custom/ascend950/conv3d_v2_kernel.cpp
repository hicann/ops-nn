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
 * \file conv3dv2_kernel.cpp
 * \brief
 */

#include "conv3d_v2_kernel.h"
#include "acl/acl.h"
#include "conv3d_v2_template.h"

void Conv3dv2Template(
    GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR scale, GM_ADDR offset,
    GM_ADDR offset_w, GM_ADDR y, GM_ADDR workspace, Ops::NN::Conv3dV2::Conv3DV2TilingData& tiling,
    int8_t FmapTiling, int8_t WeightTiling, int8_t L1PingPong, int8_t L0PingPong,
    int8_t OutputOrder, int8_t IterOrder,
    const std::string& dtype, int32_t numBlocks, aclrtStream stream)
{   
    if (FmapTiling == CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0 && WeightTiling == CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ONLY_MN_FULLLOAD_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0, CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0 && WeightTiling == CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ONLY_MN_FULLLOAD_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0, CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0 && WeightTiling == CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0 && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ONLY_MN_FULLLOAD_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0, CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0 && WeightTiling == CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0 && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ONLY_MN_FULLLOAD_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0, CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0 && WeightTiling == CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0 && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ONLY_MN_FULLLOAD_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0, CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0 && WeightTiling == CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0 && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ONLY_MN_FULLLOAD_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0, CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0 && WeightTiling == CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0 && L1PingPong == CONV_L1_PINGPONG_ALL_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ONLY_MN_FULLLOAD_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0, CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0, CONV_L1_PINGPONG_ALL_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0 && WeightTiling == CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0 && L1PingPong == CONV_L1_PINGPONG_ALL_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ONLY_MN_FULLLOAD_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0, CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0, CONV_L1_PINGPONG_ALL_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_NO_FULLLOAD_AL0_OPEN_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_NO_FULLLOAD_AL0_OPEN_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_NO_FULLLOAD_AL0_OPEN_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_NO_FULLLOAD_AL0_OPEN_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_NO_FULLLOAD_AL0_OPEN_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_NO_FULLLOAD_AL0_OPEN_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_NO_FULLLOAD_AL0_OPEN_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_OPEN, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_NO_FULLLOAD_AL0_OPEN_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_OPEN, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_NO_FULLLOAD_BL0_OPEN_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_NO_FULLLOAD_BL0_OPEN_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_NO_FULLLOAD_BL0_OPEN_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_NO_FULLLOAD_BL0_OPEN_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_NO_FULLLOAD_BL0_OPEN_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_NO_FULLLOAD_BL0_OPEN_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_NO_FULLLOAD_BL0_OPEN_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_OPEN, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_NO_FULLLOAD_BL0_OPEN_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_OPEN, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ONLY_AL1_FULLLOAD_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ONLY_AL1_FULLLOAD_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ONLY_AL1_FULLLOAD_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ONLY_AL1_FULLLOAD_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ONLY_AL1_FULLLOAD_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ONLY_AL1_FULLLOAD_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ONLY_AL1_FULLLOAD_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ONLY_AL1_FULLLOAD_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ONLY_BL1_FULLLOAD_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ONLY_BL1_FULLLOAD_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ONLY_BL1_FULLLOAD_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ONLY_BL1_FULLLOAD_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ONLY_BL1_FULLLOAD_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ONLY_BL1_FULLLOAD_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ONLY_BL1_FULLLOAD_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ONLY_BL1_FULLLOAD_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_CLOSE && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ABL1_FULLLOAD_M_FIRST_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_CLOSE, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_CLOSE && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ABL1_FULLLOAD_M_FIRST_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_CLOSE, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ABL1_FULLLOAD_M_FIRST_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ABL1_FULLLOAD_M_FIRST_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ABL1_FULLLOAD_M_FIRST_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ABL1_FULLLOAD_M_FIRST_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ABL1_FULLLOAD_N_FIRST_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ABL1_FULLLOAD_N_FIRST_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ABL1_FULLLOAD_N_FIRST_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float") {  // CONV_COMMON_ABL1_FULLLOAD_N_FIRST_SEL
        conv3dv2_template<float, float, float, float, float,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0 && WeightTiling == CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ONLY_MN_FULLLOAD_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0, CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0 && WeightTiling == CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ONLY_MN_FULLLOAD_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0, CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0 && WeightTiling == CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0 && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ONLY_MN_FULLLOAD_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0, CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0 && WeightTiling == CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0 && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ONLY_MN_FULLLOAD_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0, CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0 && WeightTiling == CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0 && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ONLY_MN_FULLLOAD_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0, CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0 && WeightTiling == CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0 && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ONLY_MN_FULLLOAD_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0, CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0 && WeightTiling == CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0 && L1PingPong == CONV_L1_PINGPONG_ALL_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ONLY_MN_FULLLOAD_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0, CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0, CONV_L1_PINGPONG_ALL_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0 && WeightTiling == CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0 && L1PingPong == CONV_L1_PINGPONG_ALL_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ONLY_MN_FULLLOAD_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0, CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0, CONV_L1_PINGPONG_ALL_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_NO_FULLLOAD_AL0_OPEN_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_NO_FULLLOAD_AL0_OPEN_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_NO_FULLLOAD_AL0_OPEN_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_NO_FULLLOAD_AL0_OPEN_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_NO_FULLLOAD_AL0_OPEN_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_NO_FULLLOAD_AL0_OPEN_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_NO_FULLLOAD_AL0_OPEN_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_OPEN, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_NO_FULLLOAD_AL0_OPEN_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_OPEN, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_NO_FULLLOAD_BL0_OPEN_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_NO_FULLLOAD_BL0_OPEN_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_NO_FULLLOAD_BL0_OPEN_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_NO_FULLLOAD_BL0_OPEN_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_NO_FULLLOAD_BL0_OPEN_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_NO_FULLLOAD_BL0_OPEN_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_NO_FULLLOAD_BL0_OPEN_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_OPEN, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_NO_FULLLOAD_BL0_OPEN_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_OPEN, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ONLY_AL1_FULLLOAD_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ONLY_AL1_FULLLOAD_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ONLY_AL1_FULLLOAD_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ONLY_AL1_FULLLOAD_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ONLY_AL1_FULLLOAD_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ONLY_AL1_FULLLOAD_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ONLY_AL1_FULLLOAD_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ONLY_AL1_FULLLOAD_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ONLY_BL1_FULLLOAD_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ONLY_BL1_FULLLOAD_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ONLY_BL1_FULLLOAD_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ONLY_BL1_FULLLOAD_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ONLY_BL1_FULLLOAD_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ONLY_BL1_FULLLOAD_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ONLY_BL1_FULLLOAD_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ONLY_BL1_FULLLOAD_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_CLOSE && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ABL1_FULLLOAD_M_FIRST_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_CLOSE, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_CLOSE && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ABL1_FULLLOAD_M_FIRST_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_CLOSE, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ABL1_FULLLOAD_M_FIRST_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ABL1_FULLLOAD_M_FIRST_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ABL1_FULLLOAD_M_FIRST_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ABL1_FULLLOAD_M_FIRST_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ABL1_FULLLOAD_N_FIRST_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ABL1_FULLLOAD_N_FIRST_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ABL1_FULLLOAD_N_FIRST_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "float16") {  // CONV_COMMON_ABL1_FULLLOAD_N_FIRST_SEL
        conv3dv2_template<half, half, half, float, half,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0 && WeightTiling == CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ONLY_MN_FULLLOAD_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0, CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0 && WeightTiling == CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ONLY_MN_FULLLOAD_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0, CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0 && WeightTiling == CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0 && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ONLY_MN_FULLLOAD_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0, CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0 && WeightTiling == CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0 && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ONLY_MN_FULLLOAD_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0, CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0 && WeightTiling == CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0 && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ONLY_MN_FULLLOAD_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0, CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0 && WeightTiling == CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0 && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ONLY_MN_FULLLOAD_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0, CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0 && WeightTiling == CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0 && L1PingPong == CONV_L1_PINGPONG_ALL_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ONLY_MN_FULLLOAD_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0, CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0, CONV_L1_PINGPONG_ALL_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0 && WeightTiling == CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0 && L1PingPong == CONV_L1_PINGPONG_ALL_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ONLY_MN_FULLLOAD_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0, CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0, CONV_L1_PINGPONG_ALL_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_NO_FULLLOAD_AL0_OPEN_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_NO_FULLLOAD_AL0_OPEN_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_NO_FULLLOAD_AL0_OPEN_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_NO_FULLLOAD_AL0_OPEN_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_NO_FULLLOAD_AL0_OPEN_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_NO_FULLLOAD_AL0_OPEN_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_NO_FULLLOAD_AL0_OPEN_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_OPEN, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_NO_FULLLOAD_AL0_OPEN_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_OPEN, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_NO_FULLLOAD_BL0_OPEN_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_NO_FULLLOAD_BL0_OPEN_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_NO_FULLLOAD_BL0_OPEN_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_NO_FULLLOAD_BL0_OPEN_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_NO_FULLLOAD_BL0_OPEN_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_NO_FULLLOAD_BL0_OPEN_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_NO_FULLLOAD_BL0_OPEN_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_OPEN, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_NO_FULLLOAD_BL0_OPEN_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_OPEN, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_NO_FULLLOAD_ALL_OPEN_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ONLY_AL1_FULLLOAD_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ONLY_AL1_FULLLOAD_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ONLY_AL1_FULLLOAD_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ONLY_AL1_FULLLOAD_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ONLY_AL1_FULLLOAD_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ONLY_AL1_FULLLOAD_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ONLY_AL1_FULLLOAD_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_OTHER && L1PingPong == CONV_L1_PINGPONG_BL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ONLY_AL1_FULLLOAD_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_OTHER, CONV_L1_PINGPONG_BL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ONLY_BL1_FULLLOAD_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ONLY_BL1_FULLLOAD_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ONLY_BL1_FULLLOAD_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ONLY_BL1_FULLLOAD_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ONLY_BL1_FULLLOAD_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ONLY_BL1_FULLLOAD_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ONLY_BL1_FULLLOAD_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_OTHER && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_AL1_OPEN &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ONLY_BL1_FULLLOAD_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_OTHER, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_AL1_OPEN, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_CLOSE && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ABL1_FULLLOAD_M_FIRST_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_CLOSE, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_CLOSE && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ABL1_FULLLOAD_M_FIRST_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_CLOSE, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ABL1_FULLLOAD_M_FIRST_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_AL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ABL1_FULLLOAD_M_FIRST_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_AL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ABL1_FULLLOAD_M_FIRST_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_MITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ABL1_FULLLOAD_M_FIRST_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_MITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ABL1_FULLLOAD_N_FIRST_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_BL0_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ABL1_FULLLOAD_N_FIRST_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_BL0_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_HW_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ABL1_FULLLOAD_N_FIRST_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_HW_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    } else if (FmapTiling == CONV_FMAP_TILING_FULLLOAD_AL1 && WeightTiling == CONV_WEIGHT_TILING_FULLLOAD_BL1 && L1PingPong == CONV_L1_PINGPONG_ALL_CLOSE &&
        L0PingPong == CONV_L0_PINGPONG_ALL_OPEN && OutputOrder == CONV_OUTPUT_ORDER_M_MODE && IterOrder == CONV_ITER_ORDER_NITER_FIRST &&
        dtype == "bfloat16") {  // CONV_COMMON_ABL1_FULLLOAD_N_FIRST_SEL
        conv3dv2_template<bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
            CONV_FMAP_TILING_FULLLOAD_AL1, CONV_WEIGHT_TILING_FULLLOAD_BL1, CONV_L1_PINGPONG_ALL_CLOSE, CONV_L0_PINGPONG_ALL_OPEN, CONV_OUTPUT_ORDER_M_MODE, CONV_ITER_ORDER_NITER_FIRST><<<numBlocks, nullptr, stream>>>
            (x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    }
    else {
        printf(
            "[conv3dv2] Unsupported config: Fmap=%d, Weight=%d, L1PP=%d, L0PP=%d, OutOrder=%d, IterOrder=%d, dtype=%s",
            FmapTiling, WeightTiling, L1PingPong, L0PingPong, OutputOrder, IterOrder, dtype.c_str());
        return;
    }
}