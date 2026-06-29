/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_rms_norm_quant_v2_apt.cpp
 * \brief
 */
#include "../add_rms_norm_quant/arch35/add_rms_norm_quant_regbase.h"
#include "../add_rms_norm_quant/arch35/add_rms_norm_quant_regbase_perf.h"
#include "../add_rms_norm_quant/arch35/add_rms_norm_quant_regbase_split_reduce.h"

using namespace AddRmsNormQuant;
using namespace AscendC;

#ifndef DTYPE_ZERO_POINTS2
#define DTYPE_ZERO_POINTS2 float
#endif

#ifndef DTYPE_ZERO_POINTS1
#define DTYPE_ZERO_POINTS1 DTYPE_ZERO_POINTS2
#endif

#define INIT_AND_PROCESS_REGBASE                                                                         \
    do {                                                                                                 \
        op.Init(x1, x2, gamma, scales1, scales2, zero_points1, zero_points2, bias, y1, y2, x, res_out, tilingData); \
        op.Process();                                                                                    \
    } while (0)

extern "C" __global__ __aicore__ void add_rms_norm_quant_v2(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR scales1, GM_ADDR scales2, GM_ADDR zero_points1, GM_ADDR zero_points2,
    GM_ADDR bias, GM_ADDR y1, GM_ADDR y2, GM_ADDR x, GM_ADDR res_out, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(AddRmsNormQuantRegbaseTilingData, tilingDataIn, tiling);
    const AddRmsNormQuantRegbaseTilingData* __restrict tilingData = &tilingDataIn;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    #if !defined(ORIG_DTYPE_BIAS) && !defined(ORIG_DTYPE_SCALES2) && !defined(ORIG_DTYPE_ZERO_POINTS1) && !defined(ORIG_DTYPE_ZERO_POINTS2)
        if (TILING_KEY_IS(1000)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1000> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(1001)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1001> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(1002)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1002> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2000)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2000> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2001)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2001> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2002)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2002> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        }
    #endif
    #if defined(ORIG_DTYPE_BIAS) && !defined(ORIG_DTYPE_SCALES2) && !defined(ORIG_DTYPE_ZERO_POINTS1) && !defined(ORIG_DTYPE_ZERO_POINTS2)
        if (TILING_KEY_IS(1100)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1100> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(1101)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1101> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(1102)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1102> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2100)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2100> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2101)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2101> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2102)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2102> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        }
    #endif
    #if !defined(ORIG_DTYPE_BIAS) && !defined(ORIG_DTYPE_SCALES2) && !defined(ORIG_DTYPE_ZERO_POINTS1) && defined(ORIG_DTYPE_ZERO_POINTS2)
        if (TILING_KEY_IS(1010)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1010> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(1011)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1011> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(1012)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1012> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2010)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2010> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2011)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2011> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2012)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2012> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        }
    #endif
    #if defined(ORIG_DTYPE_BIAS) && !defined(ORIG_DTYPE_SCALES2) && !defined(ORIG_DTYPE_ZERO_POINTS1) && defined(ORIG_DTYPE_ZERO_POINTS2)
        if (TILING_KEY_IS(1110)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1110> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(1111)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1111> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(1112)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1112> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2110)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2110> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2111)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2111> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2112)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2112> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        }
    #endif
    #if !defined(ORIG_DTYPE_BIAS) && !defined(ORIG_DTYPE_SCALES2) && defined(ORIG_DTYPE_ZERO_POINTS1) && !defined(ORIG_DTYPE_ZERO_POINTS2)
        if (TILING_KEY_IS(1040)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1040> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(1041)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1041> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(1042)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1042> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2040)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2040> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2041)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2041> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2042)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2042> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        }
    #endif
    #if defined(ORIG_DTYPE_BIAS) && !defined(ORIG_DTYPE_SCALES2) && defined(ORIG_DTYPE_ZERO_POINTS1) && !defined(ORIG_DTYPE_ZERO_POINTS2)
        if (TILING_KEY_IS(1140)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1140> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(1141)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1141> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(1142)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1142> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2140)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2140> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2141)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2141> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2142)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2142> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        }
    #endif
    #if !defined(ORIG_DTYPE_BIAS) && !defined(ORIG_DTYPE_SCALES2) && defined(ORIG_DTYPE_ZERO_POINTS1) && defined(ORIG_DTYPE_ZERO_POINTS2)
        if (TILING_KEY_IS(1050)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1050> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(1051)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1051> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(1052)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1052> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2050)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2050> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2051)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2051> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2052)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2052> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        }
    #endif
    #if defined(ORIG_DTYPE_BIAS) && !defined(ORIG_DTYPE_SCALES2) && defined(ORIG_DTYPE_ZERO_POINTS1) && defined(ORIG_DTYPE_ZERO_POINTS2)
        if (TILING_KEY_IS(1150)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1150> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(1151)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1151> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(1152)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1152> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2150)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2150> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2151)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2151> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2152)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2152> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        }
    #endif
    #if !defined(ORIG_DTYPE_BIAS) && defined(ORIG_DTYPE_SCALES2) && !defined(ORIG_DTYPE_ZERO_POINTS1) && !defined(ORIG_DTYPE_ZERO_POINTS2)
        if (TILING_KEY_IS(1020)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1020> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(1021)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1021> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(1022)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1022> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2020)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2020> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2021)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2021> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2022)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2022> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        }
    #endif
    #if defined(ORIG_DTYPE_BIAS) && defined(ORIG_DTYPE_SCALES2) && !defined(ORIG_DTYPE_ZERO_POINTS1) && !defined(ORIG_DTYPE_ZERO_POINTS2)
        if (TILING_KEY_IS(1120)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1120> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(1121)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1121> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(1122)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1122> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2120)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2120> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2121)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2121> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2122)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2122> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        }
    #endif
    #if !defined(ORIG_DTYPE_BIAS) && defined(ORIG_DTYPE_SCALES2) && !defined(ORIG_DTYPE_ZERO_POINTS1) && defined(ORIG_DTYPE_ZERO_POINTS2)
        if (TILING_KEY_IS(1030)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1030> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(1031)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1031> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(1032)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1032> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2030)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2030> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2031)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2031> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2032)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2032> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        }
    #endif
    #if defined(ORIG_DTYPE_BIAS) && defined(ORIG_DTYPE_SCALES2) && !defined(ORIG_DTYPE_ZERO_POINTS1) && defined(ORIG_DTYPE_ZERO_POINTS2)
        if (TILING_KEY_IS(1130)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1130> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(1131)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1131> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(1132)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1132> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2130)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2130> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2131)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2131> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2132)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2132> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        }
    #endif
    #if !defined(ORIG_DTYPE_BIAS) && defined(ORIG_DTYPE_SCALES2) && defined(ORIG_DTYPE_ZERO_POINTS1) && !defined(ORIG_DTYPE_ZERO_POINTS2)
        if (TILING_KEY_IS(1060)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1060> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(1061)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1061> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(1062)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1062> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2060)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2060> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2061)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2061> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2062)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2062> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        }
    #endif
    #if defined(ORIG_DTYPE_BIAS) && defined(ORIG_DTYPE_SCALES2) && defined(ORIG_DTYPE_ZERO_POINTS1) && !defined(ORIG_DTYPE_ZERO_POINTS2)
        if (TILING_KEY_IS(1160)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1160> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(1161)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1161> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(1162)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1162> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2160)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2160> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2161)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2161> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2162)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2162> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        }
    #endif
    #if !defined(ORIG_DTYPE_BIAS) && defined(ORIG_DTYPE_SCALES2) && defined(ORIG_DTYPE_ZERO_POINTS1) && defined(ORIG_DTYPE_ZERO_POINTS2)
        if (TILING_KEY_IS(1070)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1070> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(1071)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1071> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(1072)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1072> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2070)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2070> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2071)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2071> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2072)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2072> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        }
    #endif
    #if defined(ORIG_DTYPE_BIAS) && defined(ORIG_DTYPE_SCALES2) && defined(ORIG_DTYPE_ZERO_POINTS1) && defined(ORIG_DTYPE_ZERO_POINTS2)
        if (TILING_KEY_IS(1170)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1170> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(1171)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1171> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(1172)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1172> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2170)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2170> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2171)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2171> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        } else if (TILING_KEY_IS(2172)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2172> op(&pipe);
            INIT_AND_PROCESS_REGBASE;
        }
    #endif
}
