/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !!
 * \file gather_elements.cpp
 * \brief
 */
#include "arch35/gather_elements.h"
#include "arch35/gather_elements_no_contiguos.h"
#include "arch35/gather_elements_full_load.h"

using namespace GatherElements;

#define SIMT_DIM_1_INT32_AXIS_0 100
#define SIMT_DIM_2_INT32_AXIS_0 200
#define SIMT_DIM_2_INT32_AXIS_1 201
#define SIMT_DIM_3_INT32_AXIS_0 300
#define SIMT_DIM_3_INT32_AXIS_1 301
#define SIMT_DIM_3_INT32_AXIS_2 302
#define SIMT_DIM_4_INT32_AXIS_0 400
#define SIMT_DIM_4_INT32_AXIS_1 401
#define SIMT_DIM_4_INT32_AXIS_2 402
#define SIMT_DIM_4_INT32_AXIS_3 403
#define SIMT_DIM_5_INT32_AXIS_0 500
#define SIMT_DIM_5_INT32_AXIS_1 501
#define SIMT_DIM_5_INT32_AXIS_2 502
#define SIMT_DIM_5_INT32_AXIS_3 503
#define SIMT_DIM_5_INT32_AXIS_4 504
#define SIMT_DIM_6_INT32_AXIS_0 600
#define SIMT_DIM_6_INT32_AXIS_1 601
#define SIMT_DIM_6_INT32_AXIS_2 602
#define SIMT_DIM_6_INT32_AXIS_3 603
#define SIMT_DIM_6_INT32_AXIS_4 604
#define SIMT_DIM_6_INT32_AXIS_5 605
#define SIMT_DIM_7_INT32_AXIS_0 700
#define SIMT_DIM_7_INT32_AXIS_1 701
#define SIMT_DIM_7_INT32_AXIS_2 702
#define SIMT_DIM_7_INT32_AXIS_3 703
#define SIMT_DIM_7_INT32_AXIS_4 704
#define SIMT_DIM_7_INT32_AXIS_5 705
#define SIMT_DIM_7_INT32_AXIS_6 706
#define SIMT_DIM_8_INT32_AXIS_0 800
#define SIMT_DIM_8_INT32_AXIS_1 801
#define SIMT_DIM_8_INT32_AXIS_2 802
#define SIMT_DIM_8_INT32_AXIS_3 803
#define SIMT_DIM_8_INT32_AXIS_4 804
#define SIMT_DIM_8_INT32_AXIS_5 805
#define SIMT_DIM_8_INT32_AXIS_6 806
#define SIMT_DIM_8_INT32_AXIS_7 807
#define SIMT_DIM_1_INT64_AXIS_0 110
#define SIMT_DIM_2_INT64_AXIS_0 210
#define SIMT_DIM_2_INT64_AXIS_1 211
#define SIMT_DIM_3_INT64_AXIS_0 310
#define SIMT_DIM_3_INT64_AXIS_1 311
#define SIMT_DIM_3_INT64_AXIS_2 312
#define SIMT_DIM_4_INT64_AXIS_0 410
#define SIMT_DIM_4_INT64_AXIS_1 411
#define SIMT_DIM_4_INT64_AXIS_2 412
#define SIMT_DIM_4_INT64_AXIS_3 413
#define SIMT_DIM_5_INT64_AXIS_0 510
#define SIMT_DIM_5_INT64_AXIS_1 511
#define SIMT_DIM_5_INT64_AXIS_2 512
#define SIMT_DIM_5_INT64_AXIS_3 513
#define SIMT_DIM_5_INT64_AXIS_4 514
#define SIMT_DIM_6_INT64_AXIS_0 610
#define SIMT_DIM_6_INT64_AXIS_1 611
#define SIMT_DIM_6_INT64_AXIS_2 612
#define SIMT_DIM_6_INT64_AXIS_3 613
#define SIMT_DIM_6_INT64_AXIS_4 614
#define SIMT_DIM_6_INT64_AXIS_5 615
#define SIMT_DIM_7_INT64_AXIS_0 710
#define SIMT_DIM_7_INT64_AXIS_1 711
#define SIMT_DIM_7_INT64_AXIS_2 712
#define SIMT_DIM_7_INT64_AXIS_3 713
#define SIMT_DIM_7_INT64_AXIS_4 714
#define SIMT_DIM_7_INT64_AXIS_5 715
#define SIMT_DIM_7_INT64_AXIS_6 716
#define SIMT_DIM_8_INT64_AXIS_0 810
#define SIMT_DIM_8_INT64_AXIS_1 811
#define SIMT_DIM_8_INT64_AXIS_2 812
#define SIMT_DIM_8_INT64_AXIS_3 813
#define SIMT_DIM_8_INT64_AXIS_4 814
#define SIMT_DIM_8_INT64_AXIS_5 815
#define SIMT_DIM_8_INT64_AXIS_6 816
#define SIMT_DIM_8_INT64_AXIS_7 817
#define DIM_1_AXIS_0 1100
#define DIM_2_AXIS_0 1200
#define DIM_2_AXIS_1 1210
#define DIM_3_AXIS_0 1300
#define DIM_3_AXIS_1 1310
#define DIM_3_AXIS_2 1320
#define DIM_4_AXIS_0 1400
#define DIM_4_AXIS_1 1410
#define DIM_4_AXIS_2 1420
#define DIM_4_AXIS_3 1430
#define DIM_5_AXIS_0 1500
#define DIM_5_AXIS_1 1510
#define DIM_5_AXIS_2 1520
#define DIM_5_AXIS_3 1530
#define DIM_5_AXIS_4 1540
#define DIM_6_AXIS_0 1600
#define DIM_6_AXIS_1 1610
#define DIM_6_AXIS_2 1620
#define DIM_6_AXIS_3 1630
#define DIM_6_AXIS_4 1640
#define DIM_6_AXIS_5 1650
#define DIM_7_AXIS_0 1700
#define DIM_7_AXIS_1 1710
#define DIM_7_AXIS_2 1720
#define DIM_7_AXIS_3 1730
#define DIM_7_AXIS_4 1740
#define DIM_7_AXIS_5 1750
#define DIM_7_AXIS_6 1760
#define DIM_8_AXIS_0 1800
#define DIM_8_AXIS_1 1810
#define DIM_8_AXIS_2 1820
#define DIM_8_AXIS_3 1830
#define DIM_8_AXIS_4 1840
#define DIM_8_AXIS_5 1850
#define DIM_8_AXIS_6 1860
#define DIM_8_AXIS_7 1870

#define SIMT_DIM_3_INT32_AXIS_2_NO_CONTIGUOUS_SPECILA 21302
#define SIMT_DIM_3_INT64_AXIS_2_NO_CONTIGUOUS_SPECILA 21312

#define SIMT_DIM_1_INT32_AXIS_0_NO_CONTIGUOUS 20100
#define SIMT_DIM_2_INT32_AXIS_0_NO_CONTIGUOUS 20200
#define SIMT_DIM_2_INT32_AXIS_1_NO_CONTIGUOUS 20201
#define SIMT_DIM_3_INT32_AXIS_0_NO_CONTIGUOUS 20300
#define SIMT_DIM_3_INT32_AXIS_1_NO_CONTIGUOUS 20301
#define SIMT_DIM_3_INT32_AXIS_2_NO_CONTIGUOUS 20302
#define SIMT_DIM_4_INT32_AXIS_0_NO_CONTIGUOUS 20400
#define SIMT_DIM_4_INT32_AXIS_1_NO_CONTIGUOUS 20401
#define SIMT_DIM_4_INT32_AXIS_2_NO_CONTIGUOUS 20402
#define SIMT_DIM_4_INT32_AXIS_3_NO_CONTIGUOUS 20403
#define SIMT_DIM_5_INT32_AXIS_0_NO_CONTIGUOUS 20500
#define SIMT_DIM_5_INT32_AXIS_1_NO_CONTIGUOUS 20501
#define SIMT_DIM_5_INT32_AXIS_2_NO_CONTIGUOUS 20502
#define SIMT_DIM_5_INT32_AXIS_3_NO_CONTIGUOUS 20503
#define SIMT_DIM_5_INT32_AXIS_4_NO_CONTIGUOUS 20504
#define SIMT_DIM_6_INT32_AXIS_0_NO_CONTIGUOUS 20600
#define SIMT_DIM_6_INT32_AXIS_1_NO_CONTIGUOUS 20601
#define SIMT_DIM_6_INT32_AXIS_2_NO_CONTIGUOUS 20602
#define SIMT_DIM_6_INT32_AXIS_3_NO_CONTIGUOUS 20603
#define SIMT_DIM_6_INT32_AXIS_4_NO_CONTIGUOUS 20604
#define SIMT_DIM_6_INT32_AXIS_5_NO_CONTIGUOUS 20605
#define SIMT_DIM_7_INT32_AXIS_0_NO_CONTIGUOUS 20700
#define SIMT_DIM_7_INT32_AXIS_1_NO_CONTIGUOUS 20701
#define SIMT_DIM_7_INT32_AXIS_2_NO_CONTIGUOUS 20702
#define SIMT_DIM_7_INT32_AXIS_3_NO_CONTIGUOUS 20703
#define SIMT_DIM_7_INT32_AXIS_4_NO_CONTIGUOUS 20704
#define SIMT_DIM_7_INT32_AXIS_5_NO_CONTIGUOUS 20705
#define SIMT_DIM_7_INT32_AXIS_6_NO_CONTIGUOUS 20706
#define SIMT_DIM_8_INT32_AXIS_0_NO_CONTIGUOUS 20800
#define SIMT_DIM_8_INT32_AXIS_1_NO_CONTIGUOUS 20801
#define SIMT_DIM_8_INT32_AXIS_2_NO_CONTIGUOUS 20802
#define SIMT_DIM_8_INT32_AXIS_3_NO_CONTIGUOUS 20803
#define SIMT_DIM_8_INT32_AXIS_4_NO_CONTIGUOUS 20804
#define SIMT_DIM_8_INT32_AXIS_5_NO_CONTIGUOUS 20805
#define SIMT_DIM_8_INT32_AXIS_6_NO_CONTIGUOUS 20806
#define SIMT_DIM_8_INT32_AXIS_7_NO_CONTIGUOUS 20807
#define SIMT_DIM_1_INT64_AXIS_0_NO_CONTIGUOUS 20110
#define SIMT_DIM_2_INT64_AXIS_0_NO_CONTIGUOUS 20210
#define SIMT_DIM_2_INT64_AXIS_1_NO_CONTIGUOUS 20211
#define SIMT_DIM_3_INT64_AXIS_0_NO_CONTIGUOUS 20310
#define SIMT_DIM_3_INT64_AXIS_1_NO_CONTIGUOUS 20311
#define SIMT_DIM_3_INT64_AXIS_2_NO_CONTIGUOUS 20312
#define SIMT_DIM_4_INT64_AXIS_0_NO_CONTIGUOUS 20410
#define SIMT_DIM_4_INT64_AXIS_1_NO_CONTIGUOUS 20411
#define SIMT_DIM_4_INT64_AXIS_2_NO_CONTIGUOUS 20412
#define SIMT_DIM_4_INT64_AXIS_3_NO_CONTIGUOUS 20413
#define SIMT_DIM_5_INT64_AXIS_0_NO_CONTIGUOUS 20510
#define SIMT_DIM_5_INT64_AXIS_1_NO_CONTIGUOUS 20511
#define SIMT_DIM_5_INT64_AXIS_2_NO_CONTIGUOUS 20512
#define SIMT_DIM_5_INT64_AXIS_3_NO_CONTIGUOUS 20513
#define SIMT_DIM_5_INT64_AXIS_4_NO_CONTIGUOUS 20514
#define SIMT_DIM_6_INT64_AXIS_0_NO_CONTIGUOUS 20610
#define SIMT_DIM_6_INT64_AXIS_1_NO_CONTIGUOUS 20611
#define SIMT_DIM_6_INT64_AXIS_2_NO_CONTIGUOUS 20612
#define SIMT_DIM_6_INT64_AXIS_3_NO_CONTIGUOUS 20613
#define SIMT_DIM_6_INT64_AXIS_4_NO_CONTIGUOUS 20614
#define SIMT_DIM_6_INT64_AXIS_5_NO_CONTIGUOUS 20615
#define SIMT_DIM_7_INT64_AXIS_0_NO_CONTIGUOUS 20710
#define SIMT_DIM_7_INT64_AXIS_1_NO_CONTIGUOUS 20711
#define SIMT_DIM_7_INT64_AXIS_2_NO_CONTIGUOUS 20712
#define SIMT_DIM_7_INT64_AXIS_3_NO_CONTIGUOUS 20713
#define SIMT_DIM_7_INT64_AXIS_4_NO_CONTIGUOUS 20714
#define SIMT_DIM_7_INT64_AXIS_5_NO_CONTIGUOUS 20715
#define SIMT_DIM_7_INT64_AXIS_6_NO_CONTIGUOUS 20716
#define SIMT_DIM_8_INT64_AXIS_0_NO_CONTIGUOUS 20810
#define SIMT_DIM_8_INT64_AXIS_1_NO_CONTIGUOUS 20811
#define SIMT_DIM_8_INT64_AXIS_2_NO_CONTIGUOUS 20812
#define SIMT_DIM_8_INT64_AXIS_3_NO_CONTIGUOUS 20813
#define SIMT_DIM_8_INT64_AXIS_4_NO_CONTIGUOUS 20814
#define SIMT_DIM_8_INT64_AXIS_5_NO_CONTIGUOUS 20815
#define SIMT_DIM_8_INT64_AXIS_6_NO_CONTIGUOUS 20816
#define SIMT_DIM_8_INT64_AXIS_7_NO_CONTIGUOUS 20817

static constexpr int64_t B8 = 1;

template <typename T>
struct ComputeTypeGet {
    using type = typename std::conditional<sizeof(T) == B8, int8_t, T>::type;
};

extern "C" __global__ __aicore__ void gather_elements(GM_ADDR x, GM_ADDR index, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    using xType = typename ComputeTypeGet<DTYPE_X>::type;
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(SIMT_DIM_3_INT32_AXIS_2_NO_CONTIGUOUS_SPECILA)) {   
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM3, DIM2, true> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_3_INT64_AXIS_2_NO_CONTIGUOUS_SPECILA)) {   
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM3, DIM2, true> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_1_INT32_AXIS_0_NO_CONTIGUOUS)) {   
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM1, 0, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_2_INT32_AXIS_0_NO_CONTIGUOUS)) {   
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM2, 0, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_2_INT32_AXIS_1_NO_CONTIGUOUS)) {   
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM2, DIM1, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_3_INT32_AXIS_0_NO_CONTIGUOUS)) {   
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM3, 0, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_3_INT32_AXIS_1_NO_CONTIGUOUS)) {   
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM3, DIM1, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_3_INT32_AXIS_2_NO_CONTIGUOUS)) {   
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM3, DIM2, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    }  else if (TILING_KEY_IS(SIMT_DIM_4_INT32_AXIS_0_NO_CONTIGUOUS)) {   
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM4, 0, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_4_INT32_AXIS_1_NO_CONTIGUOUS)) {   
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM4, DIM1, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_4_INT32_AXIS_2_NO_CONTIGUOUS)) {   
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM4, DIM2, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_4_INT32_AXIS_3_NO_CONTIGUOUS)) {   
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM4, DIM3, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_5_INT32_AXIS_0_NO_CONTIGUOUS)) {  
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM5, 0, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_5_INT32_AXIS_1_NO_CONTIGUOUS)) {  
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM5, DIM1, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_5_INT32_AXIS_2_NO_CONTIGUOUS)) {  
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM5, DIM2, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_5_INT32_AXIS_3_NO_CONTIGUOUS)) {  
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM5, DIM3, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_5_INT32_AXIS_4_NO_CONTIGUOUS)) {  
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM5, DIM4, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_6_INT32_AXIS_0_NO_CONTIGUOUS)) {  
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM6, 0, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_6_INT32_AXIS_1_NO_CONTIGUOUS)) {  
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM6, DIM1, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_6_INT32_AXIS_2_NO_CONTIGUOUS)) {  
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM6, DIM2, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_6_INT32_AXIS_3_NO_CONTIGUOUS)) {  
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM6, DIM3, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_6_INT32_AXIS_4_NO_CONTIGUOUS)) {  
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM6, DIM4, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_6_INT32_AXIS_5_NO_CONTIGUOUS)) {  
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM6, DIM5, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_7_INT32_AXIS_0_NO_CONTIGUOUS)) {  
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM7, 0, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_7_INT32_AXIS_1_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM7, DIM1, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_7_INT32_AXIS_2_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM7, DIM2, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_7_INT32_AXIS_3_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM7, DIM3, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_7_INT32_AXIS_4_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM7, DIM4, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_7_INT32_AXIS_5_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM7, DIM5, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_7_INT32_AXIS_6_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM7, DIM6, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_INT32_AXIS_0_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM8, 0, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_INT32_AXIS_1_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM8, DIM1, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_INT32_AXIS_2_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM8, DIM2, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_INT32_AXIS_3_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM8, DIM3, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_INT32_AXIS_4_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM8, DIM4, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_INT32_AXIS_5_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM8, DIM5, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_INT32_AXIS_6_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM8, DIM6, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_INT32_AXIS_7_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint32_t, DIM8, DIM7, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_1_INT64_AXIS_0_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM1, 0, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_2_INT64_AXIS_0_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM2, 0, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_2_INT64_AXIS_1_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM2, DIM1, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_3_INT64_AXIS_0_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM3, 0, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_3_INT64_AXIS_1_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM3, DIM1, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_3_INT64_AXIS_2_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM3, DIM2, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    }  else if (TILING_KEY_IS(SIMT_DIM_4_INT64_AXIS_0_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM4, 0, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_4_INT64_AXIS_1_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM4, DIM1, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_4_INT64_AXIS_2_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM4, DIM2, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_4_INT64_AXIS_3_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM4, DIM3, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_5_INT64_AXIS_0_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM5, 0, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_5_INT64_AXIS_1_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM5, DIM1, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_5_INT64_AXIS_2_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM5, DIM2, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_5_INT64_AXIS_3_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM5, DIM3, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_5_INT64_AXIS_4_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM5, DIM4, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_6_INT64_AXIS_0_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM6, 0, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_6_INT64_AXIS_1_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM6, DIM1, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_6_INT64_AXIS_2_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM6, DIM2, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_6_INT64_AXIS_3_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM6, DIM3, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_6_INT64_AXIS_4_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM6, DIM4, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_6_INT64_AXIS_5_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM6, DIM5, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_7_INT64_AXIS_0_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM7, 0, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_7_INT64_AXIS_1_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM7, DIM1, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_7_INT64_AXIS_2_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM7, DIM2, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_7_INT64_AXIS_3_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM7, DIM3, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_7_INT64_AXIS_4_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM7, DIM4, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_7_INT64_AXIS_5_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM7, DIM5, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_7_INT64_AXIS_6_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM7, DIM6, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_INT64_AXIS_0_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM8, 0, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_INT64_AXIS_1_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM8, DIM1, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_INT64_AXIS_2_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM8, DIM2, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_INT64_AXIS_3_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM8, DIM3, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_INT64_AXIS_4_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM8, DIM4, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_INT64_AXIS_5_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM8, DIM5, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_INT64_AXIS_6_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM8, DIM6, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_INT64_AXIS_7_NO_CONTIGUOUS)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsNoContiguousTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernelNoContiguous<xType, DTYPE_INDEX, uint64_t, DIM8, DIM7, false> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_1_INT32_AXIS_0)) {
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM1, 0> op(pipe);
        op.ProcessOptim((__gm__ xType*)x, (__gm__ DTYPE_INDEX*)index, (__gm__ volatile xType*)y,
        (__gm__ const GatherElementsTilingData*)tiling);
    } else if (TILING_KEY_IS(SIMT_DIM_2_INT32_AXIS_0)) {
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM2, 0> op(pipe);
        op.ProcessOptim((__gm__ xType*)x, (__gm__ DTYPE_INDEX*)index, (__gm__ volatile xType*)y,
        (__gm__ const GatherElementsTilingData*)tiling);
    } else if (TILING_KEY_IS(SIMT_DIM_2_INT32_AXIS_1)) {
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM2, 1> op(pipe);
        op.ProcessOptim((__gm__ xType*)x, (__gm__ DTYPE_INDEX*)index, (__gm__ volatile xType*)y,
        (__gm__ const GatherElementsTilingData*)tiling);
    } else if (TILING_KEY_IS(SIMT_DIM_3_INT32_AXIS_0)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM3, 0> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_3_INT32_AXIS_1)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM3, DIM1> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_3_INT32_AXIS_2)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM3, DIM2> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_4_INT32_AXIS_0)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM4, 0> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_4_INT32_AXIS_1)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM4, DIM1> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_4_INT32_AXIS_2)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM4, DIM2> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_4_INT32_AXIS_3)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM4, DIM3> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_5_INT32_AXIS_0)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM5, 0> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_5_INT32_AXIS_1)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM5, DIM1> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_5_INT32_AXIS_2)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM5, DIM2> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_5_INT32_AXIS_3)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM5, DIM3> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_5_INT32_AXIS_4)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM5, DIM4> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_6_INT32_AXIS_0)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM6, 0> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_6_INT32_AXIS_1)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM6, DIM1> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_6_INT32_AXIS_2)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM6, DIM2> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_6_INT32_AXIS_3)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM6, DIM3> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_6_INT32_AXIS_4)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM6, DIM4> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_6_INT32_AXIS_5)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM6, DIM5> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_7_INT32_AXIS_0)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM7, 0> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_7_INT32_AXIS_1)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM7, DIM1> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_7_INT32_AXIS_2)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM7, DIM2> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_7_INT32_AXIS_3)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM7, DIM3> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_7_INT32_AXIS_4)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM7, DIM4> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_7_INT32_AXIS_5)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM7, DIM5> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_7_INT32_AXIS_6)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM7, DIM6> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_INT32_AXIS_0)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM8, 0> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_INT32_AXIS_1)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM8, DIM1> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_INT32_AXIS_2)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM8, DIM2> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_INT32_AXIS_3)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM8, DIM3> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_INT32_AXIS_4)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM8, DIM4> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_INT32_AXIS_5)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM8, DIM5> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_INT32_AXIS_6)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM8, DIM6> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_INT32_AXIS_7)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint32_t, DIM8, DIM7> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_1_INT64_AXIS_0)) {
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM1, 0> op(pipe);
        op.ProcessOptim((__gm__ xType*)x, (__gm__ DTYPE_INDEX*)index, (__gm__ volatile xType*)y,
        (__gm__ const GatherElementsTilingData*)tiling);
    } else if (TILING_KEY_IS(SIMT_DIM_2_INT64_AXIS_0)) {
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM2, 0> op(pipe);
        op.ProcessOptim((__gm__ xType*)x, (__gm__ DTYPE_INDEX*)index, (__gm__ volatile xType*)y,
        (__gm__ const GatherElementsTilingData*)tiling);
    } else if (TILING_KEY_IS(SIMT_DIM_2_INT64_AXIS_1)) {
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM2, 1> op(pipe);
        op.ProcessOptim((__gm__ xType*)x, (__gm__ DTYPE_INDEX*)index, (__gm__ volatile xType*)y,
        (__gm__ const GatherElementsTilingData*)tiling);
    } else if (TILING_KEY_IS(SIMT_DIM_3_INT64_AXIS_0)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM3, 0> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_3_INT64_AXIS_1)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM3, DIM1> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_3_INT64_AXIS_2)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM3, DIM2> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_4_INT64_AXIS_0)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM4, 0> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_4_INT64_AXIS_1)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM4, DIM1> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_4_INT64_AXIS_2)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM4, DIM2> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_4_INT64_AXIS_3)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM4, DIM3> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_5_INT64_AXIS_0)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM5, 0> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_5_INT64_AXIS_1)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM5, DIM1> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_5_INT64_AXIS_2)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM5, DIM2> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_5_INT64_AXIS_3)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM5, DIM3> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_5_INT64_AXIS_4)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM5, DIM4> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_6_INT64_AXIS_0)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM6, 0> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_6_INT64_AXIS_1)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM6, DIM1> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_6_INT64_AXIS_2)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM6, DIM2> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_6_INT64_AXIS_3)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM6, DIM3> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_6_INT64_AXIS_4)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM6, DIM4> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_6_INT64_AXIS_5)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM6, DIM5> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_7_INT64_AXIS_0)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM7, 0> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_7_INT64_AXIS_1)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM7, DIM1> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_7_INT64_AXIS_2)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM7, DIM2> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_7_INT64_AXIS_3)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM7, DIM3> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_7_INT64_AXIS_4)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM7, DIM4> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_7_INT64_AXIS_5)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM7, DIM5> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_7_INT64_AXIS_6)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM7, DIM6> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_INT64_AXIS_0)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM8, 0> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_INT64_AXIS_1)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM8, DIM1> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_INT64_AXIS_2)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM8, DIM2> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_INT64_AXIS_3)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM8, DIM3> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_INT64_AXIS_4)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM8, DIM4> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_INT64_AXIS_5)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM8, DIM5> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_INT64_AXIS_6)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM8, DIM6> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_INT64_AXIS_7)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsKernel<xType, DTYPE_INDEX, uint64_t, DIM8, DIM7> op(pipe);
        op.Init(x, index, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(DIM_1_AXIS_0)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM1, 0> op(&tilingData, pipe);;
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_2_AXIS_0)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM2, 0> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_2_AXIS_1)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM2, 1> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_3_AXIS_0)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM3, 0> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_3_AXIS_1)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM3, DIM1> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_3_AXIS_2)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM3, DIM2> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_4_AXIS_0)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM4, 0> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_4_AXIS_1)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM4, DIM1> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_4_AXIS_2)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM4, DIM2> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_4_AXIS_3)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM4, DIM3> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_5_AXIS_0)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM5, 0> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_5_AXIS_1)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM5, DIM1> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_5_AXIS_2)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM5, DIM2> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_5_AXIS_3)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM5, DIM3> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_5_AXIS_4)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM5, DIM4> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_6_AXIS_0)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM6, 0> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_6_AXIS_1)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM6, DIM1> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_6_AXIS_2)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM6, DIM2> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_6_AXIS_3)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM6, DIM3> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_6_AXIS_4)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM6, DIM4> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_6_AXIS_5)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM6, DIM5> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_7_AXIS_0)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM7, 0> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_7_AXIS_1)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM7, DIM1> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_7_AXIS_2)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM7, DIM2> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_7_AXIS_3)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM7, DIM3> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_7_AXIS_4)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM7, DIM4> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_7_AXIS_5)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM7, DIM5> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_7_AXIS_6)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM7, DIM6> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_8_AXIS_0)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM8, 0> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_8_AXIS_1)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM8, DIM1> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_8_AXIS_2)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM8, DIM2> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_8_AXIS_3)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM8, DIM3> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_8_AXIS_4)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM8, DIM4> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_8_AXIS_5)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM8, DIM5> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_8_AXIS_6)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM8, DIM6> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    } else if (TILING_KEY_IS(DIM_8_AXIS_7)) {
        GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
        GatherElements::GatherElementsFullLoadKernel<xType, DTYPE_INDEX, DIM8, DIM7> op(&tilingData, pipe);
        op.Init(x, index, y);
        op.Process();
    }
}