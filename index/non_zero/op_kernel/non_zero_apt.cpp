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
 * \file non_zero_apt.cpp
 * \brief
 */
#include "arch35/non_zero_small_mask_trans_1.h"
#include "arch35/non_zero_small_mask_trans_2.h"
#include "arch35/non_zero_small_mask_trans_3.h"
#include "arch35/non_zero_small_mask_trans_4.h"
#include "arch35/non_zero_small_mask_trans_5.h"
#include "arch35/non_zero_small_mask_trans_6.h"
#include "arch35/non_zero_small_mask_trans_7.h"
#include "arch35/non_zero_small_mask_trans_8.h"
#include "arch35/non_zero_small_mask_no_trans_2.h"
#include "arch35/non_zero_small_mask_no_trans_3.h"
#include "arch35/non_zero_small_mask_no_trans_4.h"
#include "arch35/non_zero_small_mask_no_trans_5.h"
#include "arch35/non_zero_small_mask_no_trans_6.h"
#include "arch35/non_zero_small_mask_no_trans_7.h"
#include "arch35/non_zero_small_mask_no_trans_8.h"
#include "arch35/non_zero_big_mask.h"
#include "arch35/non_zero_base.h"
#include "arch35/non_zero_big_mask_dim1.h"
#include "arch35/non_zero_big_mask_dim2.h"
#include "arch35/non_zero_big_mask_dim3.h"
#include "arch35/non_zero_big_mask_dim4.h"
#include "arch35/non_zero_big_mask_dim5.h"
#include "arch35/non_zero_big_mask_dim6.h"
#include "arch35/non_zero_big_mask_dim7.h"
#include "arch35/non_zero_big_mask_dim8.h"
#include "arch35/non_zero_null.h"
#include "arch35/non_zero_full_load_base.h"
#include "arch35/non_zero_full_load_dim1.h"
#include "arch35/non_zero_full_load_dim2.h"
#include "arch35/non_zero_full_load_dim3.h"
#include "arch35/non_zero_full_load_dim4.h"

#define TILING_KEY_SMALL_MASK_1_DIM 1
#define TILING_KEY_SMALL_MASK_2_DIM 2
#define TILING_KEY_SMALL_MASK_3_DIM 3
#define TILING_KEY_SMALL_MASK_4_DIM 4
#define TILING_KEY_SMALL_MASK_5_DIM 5
#define TILING_KEY_SMALL_MASK_6_DIM 6
#define TILING_KEY_SMALL_MASK_7_DIM 7
#define TILING_KEY_SMALL_MASK_8_DIM 8
#define TILING_KEY_SMALL_MASK_NO_TRANS_2_DIM 9
#define TILING_KEY_SMALL_MASK_NO_TRANS_3_DIM 10
#define TILING_KEY_SMALL_MASK_NO_TRANS_4_DIM 11
#define TILING_KEY_SMALL_MASK_NO_TRANS_5_DIM 12
#define TILING_KEY_SMALL_MASK_NO_TRANS_6_DIM 13
#define TILING_KEY_SMALL_MASK_NO_TRANS_7_DIM 14
#define TILING_KEY_SMALL_MASK_NO_TRANS_8_DIM 15
#define TILING_KEY_BIG_MASK_1_DIM 20001
#define TILING_KEY_BIG_MASK_2_DIM 20002
#define TILING_KEY_BIG_MASK_3_DIM 20003
#define TILING_KEY_BIG_MASK_4_DIM 20004
#define TILING_KEY_BIG_MASK_5_DIM 20005
#define TILING_KEY_BIG_MASK_6_DIM 20006
#define TILING_KEY_BIG_MASK_7_DIM 20007
#define TILING_KEY_BIG_MASK_8_DIM 20008
#define TILING_KEY_NULL_INPUT_TENSOR 30001
#define TILING_KEY_FULL_LOAD_1_DIM 40001
#define TILING_KEY_FULL_LOAD_2_DIM 40002
#define TILING_KEY_FULL_LOAD_3_DIM 40003
#define TILING_KEY_FULL_LOAD_4_DIM 40004

using namespace AscendC;

extern "C" __global__ __aicore__ void non_zero(
    GM_ADDR x, GM_ADDR y, GM_ADDR outShape, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (TILING_KEY_IS(TILING_KEY_NULL_INPUT_TENSOR)) {
        NonZero::NonZeroNullInputTensor<DTYPE_X, DTYPE_Y> op;
        op.Init(x, y, outShape, userWS, &tilingData);
        op.Process();
        return;
    }
    if constexpr (IsSameType<DTYPE_X, bool>::value) {
        if (TILING_KEY_IS(TILING_KEY_FULL_LOAD_1_DIM)) {
            NonZero::NonZeroFullLoadDim1<int8_t, DTYPE_Y, 1> op;
            op.Init(x, y, outShape, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_FULL_LOAD_2_DIM)) {
            NonZero::NonZeroFullLoadDim2<int8_t, DTYPE_Y, 2> op;
            op.Init(x, y, outShape, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_FULL_LOAD_3_DIM)) {
            NonZero::NonZeroFullLoadDim3<int8_t, DTYPE_Y, 3> op;
            op.Init(x, y, outShape, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_FULL_LOAD_4_DIM)) {
            NonZero::NonZeroFullLoadDim4<int8_t, DTYPE_Y, 4> op;
            op.Init(x, y, outShape, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_SMALL_MASK_1_DIM)) {
            NonZero::NonZeroSmallMask1<int8_t, DTYPE_Y, 1> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_SMALL_MASK_2_DIM)) {
            NonZero::NonZeroSmallMask2<int8_t, DTYPE_Y, 2> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_SMALL_MASK_3_DIM)) {
            NonZero::NonZeroSmallMask3<int8_t, DTYPE_Y, 3> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_SMALL_MASK_4_DIM)) {
            NonZero::NonZeroSmallMask4<int8_t, DTYPE_Y, 4> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_SMALL_MASK_5_DIM)) {
            NonZero::NonZeroSmallMask5<int8_t, DTYPE_Y, 5> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_SMALL_MASK_6_DIM)) {
            NonZero::NonZeroSmallMask6<int8_t, DTYPE_Y, 6> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_SMALL_MASK_7_DIM)) {
            NonZero::NonZeroSmallMask7<int8_t, DTYPE_Y, 7> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_SMALL_MASK_8_DIM)) {
            NonZero::NonZeroSmallMask8<int8_t, DTYPE_Y, 8> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_SMALL_MASK_NO_TRANS_2_DIM)) {
            NonZero::NonZeroSmallMaskNo2<int8_t, DTYPE_Y, 9> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_SMALL_MASK_NO_TRANS_3_DIM)) {
            NonZero::NonZeroSmallMaskNo3<int8_t, DTYPE_Y, 10> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_SMALL_MASK_NO_TRANS_4_DIM)) {
            NonZero::NonZeroSmallMaskNo4<int8_t, DTYPE_Y, 11> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_SMALL_MASK_NO_TRANS_5_DIM)) {
            NonZero::NonZeroSmallMaskNo5<int8_t, DTYPE_Y, 12> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_SMALL_MASK_NO_TRANS_6_DIM)) {
            NonZero::NonZeroSmallMaskNo6<int8_t, DTYPE_Y, 13> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_SMALL_MASK_NO_TRANS_7_DIM)) {
            NonZero::NonZeroSmallMaskNo7<int8_t, DTYPE_Y, 14> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_SMALL_MASK_NO_TRANS_8_DIM)) {
            NonZero::NonZeroSmallMaskNo8<int8_t, DTYPE_Y, 15> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_BIG_MASK_1_DIM)) {
            NonZero::NonZeroBigMaskDim1<int8_t, DTYPE_Y> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process<
                NonZero::NonZeroBigMaskDim1<int8_t, DTYPE_Y>,
                &NonZero::NonZeroBigMaskDim1<int8_t, DTYPE_Y>::ComputeOutput>(&op);
        } else if (TILING_KEY_IS(TILING_KEY_BIG_MASK_2_DIM)) {
            NonZero::NonZeroBigMaskDim2<int8_t, DTYPE_Y> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process<
                NonZero::NonZeroBigMaskDim2<int8_t, DTYPE_Y>,
                &NonZero::NonZeroBigMaskDim2<int8_t, DTYPE_Y>::ComputeOutput>(&op);
        } else if (TILING_KEY_IS(TILING_KEY_BIG_MASK_3_DIM)) {
            NonZero::NonZeroBigMaskDim3<int8_t, DTYPE_Y> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process<
                NonZero::NonZeroBigMaskDim3<int8_t, DTYPE_Y>,
                &NonZero::NonZeroBigMaskDim3<int8_t, DTYPE_Y>::ComputeOutput>(&op);
        } else if (TILING_KEY_IS(TILING_KEY_BIG_MASK_4_DIM)) {
            NonZero::NonZeroBigMaskDim4<int8_t, DTYPE_Y> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process<
                NonZero::NonZeroBigMaskDim4<int8_t, DTYPE_Y>,
                &NonZero::NonZeroBigMaskDim4<int8_t, DTYPE_Y>::ComputeOutput>(&op);
        } else if (TILING_KEY_IS(TILING_KEY_BIG_MASK_5_DIM)) {
            NonZero::NonZeroBigMaskDim5<int8_t, DTYPE_Y> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process<
                NonZero::NonZeroBigMaskDim5<int8_t, DTYPE_Y>,
                &NonZero::NonZeroBigMaskDim5<int8_t, DTYPE_Y>::ComputeOutput>(&op);
        } else if (TILING_KEY_IS(TILING_KEY_BIG_MASK_6_DIM)) {
            NonZero::NonZeroBigMaskDim6<int8_t, DTYPE_Y> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process<
                NonZero::NonZeroBigMaskDim6<int8_t, DTYPE_Y>,
                &NonZero::NonZeroBigMaskDim6<int8_t, DTYPE_Y>::ComputeOutput>(&op);
        } else if (TILING_KEY_IS(TILING_KEY_BIG_MASK_7_DIM)) {
            NonZero::NonZeroBigMaskDim7<int8_t, DTYPE_Y> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process<
                NonZero::NonZeroBigMaskDim7<int8_t, DTYPE_Y>,
                &NonZero::NonZeroBigMaskDim7<int8_t, DTYPE_Y>::ComputeOutput>(&op);
        } else if (TILING_KEY_IS(TILING_KEY_BIG_MASK_8_DIM)) {
            NonZero::NonZeroBigMaskDim8<int8_t, DTYPE_Y> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process<
                NonZero::NonZeroBigMaskDim8<int8_t, DTYPE_Y>,
                &NonZero::NonZeroBigMaskDim8<int8_t, DTYPE_Y>::ComputeOutput>(&op);
        }
    } else {
        if (TILING_KEY_IS(TILING_KEY_FULL_LOAD_1_DIM)) {
            NonZero::NonZeroFullLoadDim1<DTYPE_X, DTYPE_Y, 1> op;
            op.Init(x, y, outShape, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_FULL_LOAD_2_DIM)) {
            NonZero::NonZeroFullLoadDim2<DTYPE_X, DTYPE_Y, 2> op;
            op.Init(x, y, outShape, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_FULL_LOAD_3_DIM)) {
            NonZero::NonZeroFullLoadDim3<DTYPE_X, DTYPE_Y, 3> op;
            op.Init(x, y, outShape, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_FULL_LOAD_4_DIM)) {
            NonZero::NonZeroFullLoadDim4<DTYPE_X, DTYPE_Y, 4> op;
            op.Init(x, y, outShape, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_SMALL_MASK_1_DIM)) {
            NonZero::NonZeroSmallMask1<DTYPE_X, DTYPE_Y, 1> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_SMALL_MASK_2_DIM)) {
            NonZero::NonZeroSmallMask2<DTYPE_X, DTYPE_Y, 2> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_SMALL_MASK_3_DIM)) {
            NonZero::NonZeroSmallMask3<DTYPE_X, DTYPE_Y, 3> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_SMALL_MASK_4_DIM)) {
            NonZero::NonZeroSmallMask4<DTYPE_X, DTYPE_Y, 4> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_SMALL_MASK_5_DIM)) {
            NonZero::NonZeroSmallMask5<DTYPE_X, DTYPE_Y, 5> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_SMALL_MASK_6_DIM)) {
            NonZero::NonZeroSmallMask6<DTYPE_X, DTYPE_Y, 6> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_SMALL_MASK_7_DIM)) {
            NonZero::NonZeroSmallMask7<DTYPE_X, DTYPE_Y, 7> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_SMALL_MASK_8_DIM)) {
            NonZero::NonZeroSmallMask8<DTYPE_X, DTYPE_Y, 8> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_SMALL_MASK_NO_TRANS_2_DIM)) {
            NonZero::NonZeroSmallMaskNo2<DTYPE_X, DTYPE_Y, 9> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_SMALL_MASK_NO_TRANS_3_DIM)) {
            NonZero::NonZeroSmallMaskNo3<DTYPE_X, DTYPE_Y, 10> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_SMALL_MASK_NO_TRANS_4_DIM)) {
            NonZero::NonZeroSmallMaskNo4<DTYPE_X, DTYPE_Y, 11> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_SMALL_MASK_NO_TRANS_5_DIM)) {
            NonZero::NonZeroSmallMaskNo5<DTYPE_X, DTYPE_Y, 12> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_SMALL_MASK_NO_TRANS_6_DIM)) {
            NonZero::NonZeroSmallMaskNo6<DTYPE_X, DTYPE_Y, 13> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_SMALL_MASK_NO_TRANS_7_DIM)) {
            NonZero::NonZeroSmallMaskNo7<DTYPE_X, DTYPE_Y, 14> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_SMALL_MASK_NO_TRANS_8_DIM)) {
            NonZero::NonZeroSmallMaskNo8<DTYPE_X, DTYPE_Y, 15> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_BIG_MASK_1_DIM)) {
            NonZero::NonZeroBigMaskDim1<DTYPE_X, DTYPE_Y> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process<
                NonZero::NonZeroBigMaskDim1<DTYPE_X, DTYPE_Y>,
                &NonZero::NonZeroBigMaskDim1<DTYPE_X, DTYPE_Y>::ComputeOutput>(&op);
        } else if (TILING_KEY_IS(TILING_KEY_BIG_MASK_2_DIM)) {
            NonZero::NonZeroBigMaskDim2<DTYPE_X, DTYPE_Y> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process<
                NonZero::NonZeroBigMaskDim2<DTYPE_X, DTYPE_Y>,
                &NonZero::NonZeroBigMaskDim2<DTYPE_X, DTYPE_Y>::ComputeOutput>(&op);
        } else if (TILING_KEY_IS(TILING_KEY_BIG_MASK_3_DIM)) {
            NonZero::NonZeroBigMaskDim3<DTYPE_X, DTYPE_Y> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process<
                NonZero::NonZeroBigMaskDim3<DTYPE_X, DTYPE_Y>,
                &NonZero::NonZeroBigMaskDim3<DTYPE_X, DTYPE_Y>::ComputeOutput>(&op);
        } else if (TILING_KEY_IS(TILING_KEY_BIG_MASK_4_DIM)) {
            NonZero::NonZeroBigMaskDim4<DTYPE_X, DTYPE_Y> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process<
                NonZero::NonZeroBigMaskDim4<DTYPE_X, DTYPE_Y>,
                &NonZero::NonZeroBigMaskDim4<DTYPE_X, DTYPE_Y>::ComputeOutput>(&op);
        } else if (TILING_KEY_IS(TILING_KEY_BIG_MASK_5_DIM)) {
            NonZero::NonZeroBigMaskDim5<DTYPE_X, DTYPE_Y> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process<
                NonZero::NonZeroBigMaskDim5<DTYPE_X, DTYPE_Y>,
                &NonZero::NonZeroBigMaskDim5<DTYPE_X, DTYPE_Y>::ComputeOutput>(&op);
        } else if (TILING_KEY_IS(TILING_KEY_BIG_MASK_6_DIM)) {
            NonZero::NonZeroBigMaskDim6<DTYPE_X, DTYPE_Y> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process<
                NonZero::NonZeroBigMaskDim6<DTYPE_X, DTYPE_Y>,
                &NonZero::NonZeroBigMaskDim6<DTYPE_X, DTYPE_Y>::ComputeOutput>(&op);
        } else if (TILING_KEY_IS(TILING_KEY_BIG_MASK_7_DIM)) {
            NonZero::NonZeroBigMaskDim7<DTYPE_X, DTYPE_Y> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process<
                NonZero::NonZeroBigMaskDim7<DTYPE_X, DTYPE_Y>,
                &NonZero::NonZeroBigMaskDim7<DTYPE_X, DTYPE_Y>::ComputeOutput>(&op);
        } else if (TILING_KEY_IS(TILING_KEY_BIG_MASK_8_DIM)) {
            NonZero::NonZeroBigMaskDim8<DTYPE_X, DTYPE_Y> op;
            op.Init(x, y, outShape, userWS, &tilingData);
            op.Process<
                NonZero::NonZeroBigMaskDim8<DTYPE_X, DTYPE_Y>,
                &NonZero::NonZeroBigMaskDim8<DTYPE_X, DTYPE_Y>::ComputeOutput>(&op);
        }
    }
}
