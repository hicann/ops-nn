/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_elements.cpp
 * \brief scatter_elements
 */
#include "arch35/scatter_elements.h"
#include "arch35/scatter_elements_deterministic.h"

using namespace ScatterElements;

#define SCAC_ELE_UINT32_IDX32_REDU_NONE_B8 1000001
#define SCAC_ELE_UINT32_IDX32_REDU_NONE_B16 1000002
#define SCAC_ELE_UINT32_IDX32_REDU_NONE_B32 1000004
#define SCAC_ELE_UINT32_IDX32_REDU_NONE_B64 1000008

#define SCAC_ELE_UINT32_IDX32_REDU_ADD_FP32 1000100
#define SCAC_ELE_UINT32_IDX32_REDU_ADD_FP16 1000101
#define SCAC_ELE_UINT32_IDX32_REDU_ADD_INT8 1000102
#define SCAC_ELE_UINT32_IDX32_REDU_ADD_INT32 1000103
#define SCAC_ELE_UINT32_IDX32_REDU_ADD_UINT8 1000104
#define SCAC_ELE_UINT32_IDX32_REDU_ADD_INT16 1000106
#define SCAC_ELE_UINT32_IDX32_REDU_ADD_BF16 1000127
#define SCAC_ELE_UINT32_IDX32_REDU_ADD_INT64 1000109
#define SCAC_ELE_UINT32_IDX32_REDU_ADD_BOOL 1000112

#define SCAC_ELE_UINT32_IDX32_REDU_MUL_FP32 1000200
#define SCAC_ELE_UINT32_IDX32_REDU_MUL_FP16 1000201
#define SCAC_ELE_UINT32_IDX32_REDU_MUL_INT8 1000202
#define SCAC_ELE_UINT32_IDX32_REDU_MUL_INT32 1000203
#define SCAC_ELE_UINT32_IDX32_REDU_MUL_UINT8 1000204
#define SCAC_ELE_UINT32_IDX32_REDU_MUL_INT16 1000206
#define SCAC_ELE_UINT32_IDX32_REDU_MUL_BF16 1000227
#define SCAC_ELE_UINT32_IDX32_REDU_MUL_INT64 1000209

#define SCAC_ELE_UINT32_IDX64_REDU_NONE_B8 1001001
#define SCAC_ELE_UINT32_IDX64_REDU_NONE_B16 1001002
#define SCAC_ELE_UINT32_IDX64_REDU_NONE_B32 1001004
#define SCAC_ELE_UINT32_IDX64_REDU_NONE_B64 1001008

#define SCAC_ELE_UINT32_IDX64_REDU_ADD_FP32 1001100
#define SCAC_ELE_UINT32_IDX64_REDU_ADD_FP16 1001101
#define SCAC_ELE_UINT32_IDX64_REDU_ADD_INT8 1001102
#define SCAC_ELE_UINT32_IDX64_REDU_ADD_INT32 1001103
#define SCAC_ELE_UINT32_IDX64_REDU_ADD_UINT8 1001104
#define SCAC_ELE_UINT32_IDX64_REDU_ADD_INT16 1001106
#define SCAC_ELE_UINT32_IDX64_REDU_ADD_BF16 1001127
#define SCAC_ELE_UINT32_IDX64_REDU_ADD_INT64 1001109
#define SCAC_ELE_UINT32_IDX64_REDU_ADD_BOOL 1001112

#define SCAC_ELE_UINT32_IDX64_REDU_MUL_FP32 1001200
#define SCAC_ELE_UINT32_IDX64_REDU_MUL_FP16 1001201
#define SCAC_ELE_UINT32_IDX64_REDU_MUL_INT8 1001202
#define SCAC_ELE_UINT32_IDX64_REDU_MUL_INT32 1001203
#define SCAC_ELE_UINT32_IDX64_REDU_MUL_UINT8 1001204
#define SCAC_ELE_UINT32_IDX64_REDU_MUL_INT16 1001206
#define SCAC_ELE_UINT32_IDX64_REDU_MUL_BF16 1001227
#define SCAC_ELE_UINT32_IDX64_REDU_MUL_INT64 1001209

#define SCAC_ELE_UINT64_IDX32_REDU_NONE_B8 1010001
#define SCAC_ELE_UINT64_IDX32_REDU_NONE_B16 1010002
#define SCAC_ELE_UINT64_IDX32_REDU_NONE_B32 1010004
#define SCAC_ELE_UINT64_IDX32_REDU_NONE_B64 1010008

#define SCAC_ELE_UINT64_IDX32_REDU_ADD_FP32 1010100
#define SCAC_ELE_UINT64_IDX32_REDU_ADD_FP16 1010101
#define SCAC_ELE_UINT64_IDX32_REDU_ADD_INT8 1010102
#define SCAC_ELE_UINT64_IDX32_REDU_ADD_INT32 1010103
#define SCAC_ELE_UINT64_IDX32_REDU_ADD_UINT8 1010104
#define SCAC_ELE_UINT64_IDX32_REDU_ADD_INT16 1010106
#define SCAC_ELE_UINT64_IDX32_REDU_ADD_BF16 1010127
#define SCAC_ELE_UINT64_IDX32_REDU_ADD_INT64 1010109
#define SCAC_ELE_UINT64_IDX32_REDU_ADD_BOOL 1010112

#define SCAC_ELE_UINT64_IDX32_REDU_MUL_FP32 1010200
#define SCAC_ELE_UINT64_IDX32_REDU_MUL_FP16 1010201
#define SCAC_ELE_UINT64_IDX32_REDU_MUL_INT8 1010202
#define SCAC_ELE_UINT64_IDX32_REDU_MUL_INT32 1010203
#define SCAC_ELE_UINT64_IDX32_REDU_MUL_UINT8 1010204
#define SCAC_ELE_UINT64_IDX32_REDU_MUL_INT16 1010206
#define SCAC_ELE_UINT64_IDX32_REDU_MUL_BF16 1010227
#define SCAC_ELE_UINT64_IDX32_REDU_MUL_INT64 1010209

#define SCAC_ELE_UINT64_IDX64_REDU_NONE_B8 1011001
#define SCAC_ELE_UINT64_IDX64_REDU_NONE_B16 1011002
#define SCAC_ELE_UINT64_IDX64_REDU_NONE_B32 1011004
#define SCAC_ELE_UINT64_IDX64_REDU_NONE_B64 1011008

#define SCAC_ELE_UINT64_IDX64_REDU_ADD_FP32 1011100
#define SCAC_ELE_UINT64_IDX64_REDU_ADD_FP16 1011101
#define SCAC_ELE_UINT64_IDX64_REDU_ADD_INT8 1011102
#define SCAC_ELE_UINT64_IDX64_REDU_ADD_INT32 1011103
#define SCAC_ELE_UINT64_IDX64_REDU_ADD_UINT8 1011104
#define SCAC_ELE_UINT64_IDX64_REDU_ADD_INT16 1011106
#define SCAC_ELE_UINT64_IDX64_REDU_ADD_BF16 1011127
#define SCAC_ELE_UINT64_IDX64_REDU_ADD_INT64 1011109
#define SCAC_ELE_UINT64_IDX64_REDU_ADD_BOOL 1011112

#define SCAC_ELE_UINT64_IDX64_REDU_MUL_FP32 1011200
#define SCAC_ELE_UINT64_IDX64_REDU_MUL_FP16 1011201
#define SCAC_ELE_UINT64_IDX64_REDU_MUL_INT8 1011202
#define SCAC_ELE_UINT64_IDX64_REDU_MUL_INT32 1011203
#define SCAC_ELE_UINT64_IDX64_REDU_MUL_UINT8 1011204
#define SCAC_ELE_UINT64_IDX64_REDU_MUL_INT16 1011206
#define SCAC_ELE_UINT64_IDX64_REDU_MUL_BF16 1011227
#define SCAC_ELE_UINT64_IDX64_REDU_MUL_INT64 1011209

extern "C" __global__ __aicore__ void scatter_elements(GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y,
                                                       GM_ADDR workspace, GM_ADDR tiling)
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
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

    #if (ORIG_DTYPE_DATA == DT_INT8 || ORIG_DTYPE_DATA == DT_UINT8 || ORIG_DTYPE_DATA == DT_BOOL) && ORIG_DTYPE_INDICES == DT_INT32
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX32_REDU_NONE_B8);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX32_REDU_NONE_B8);

        #if TILING_KEY_VAR == SCAC_ELE_UINT32_IDX32_REDU_NONE_B8
            if (tilingData.isDeterministic) {
                KernelScatterElementsDeterm<int8_t, int32_t, uint32_t, REDU_NONE> op(&tilingData, &pipe);
                op.Init(x, indices, updates, y);
                op.Process();
            } else {
                KernelScatterElements<int8_t, int32_t, uint32_t, uint32_t, REDU_NONE> op(pipe);
                op.Init(x, indices, updates, y, userWS, &tilingData);
                op.Process();
            }
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX32_REDU_NONE_B8
            if (tilingData.isDeterministic) {
                KernelScatterElementsDeterm<int8_t, int32_t, uint64_t, REDU_NONE> op(&tilingData, &pipe);
                op.Init(x, indices, updates, y);
                op.Process();
            } else {
                KernelScatterElements<int8_t, int32_t, uint64_t, uint32_t, REDU_NONE> op(pipe);
                op.Init(x, indices, updates, y, userWS, &tilingData);
                op.Process();
            }
        #endif
    #endif

    #if (ORIG_DTYPE_DATA == DT_INT8 || ORIG_DTYPE_DATA == DT_UINT8 || ORIG_DTYPE_DATA == DT_BOOL) && ORIG_DTYPE_INDICES == DT_INT64
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX64_REDU_NONE_B8);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX64_REDU_NONE_B8);

        #if TILING_KEY_VAR == SCAC_ELE_UINT32_IDX64_REDU_NONE_B8
            if (tilingData.isDeterministic) {
                KernelScatterElementsDeterm<int8_t, int64_t, uint32_t, REDU_NONE> op(&tilingData, &pipe);
                op.Init(x, indices, updates, y);
                op.Process();
            } else {
                KernelScatterElements<int8_t, int64_t, uint32_t, uint32_t, REDU_NONE> op(pipe);
                op.Init(x, indices, updates, y, userWS, &tilingData);
                op.Process();
            }
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX64_REDU_NONE_B8
            if (tilingData.isDeterministic) {
                KernelScatterElementsDeterm<int8_t, int64_t, uint64_t, REDU_NONE> op(&tilingData, &pipe);
                op.Init(x, indices, updates, y);
                op.Process();
            } else {
                KernelScatterElements<int8_t, int64_t, uint64_t, uint32_t, REDU_NONE> op(pipe);
                op.Init(x, indices, updates, y, userWS, &tilingData);
                op.Process();
            }
        #endif
    #endif

    #if (ORIG_DTYPE_DATA == DT_INT16 || ORIG_DTYPE_DATA == DT_FLOAT16 || ORIG_DTYPE_DATA == DT_BF16)  && ORIG_DTYPE_INDICES == DT_INT32
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX32_REDU_NONE_B16);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX32_REDU_NONE_B16);

        #if TILING_KEY_VAR == SCAC_ELE_UINT32_IDX32_REDU_NONE_B16
            if (tilingData.isDeterministic) {
                KernelScatterElementsDeterm<int16_t, int32_t, uint32_t, REDU_NONE> op(&tilingData, &pipe);
                op.Init(x, indices, updates, y);
                op.Process();
            } else {
                KernelScatterElements<int16_t, int32_t, uint32_t, uint32_t, REDU_NONE> op(pipe);
                op.Init(x, indices, updates, y, userWS, &tilingData);
                op.Process();
            }
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX32_REDU_NONE_B16
            if (tilingData.isDeterministic) {
                KernelScatterElementsDeterm<int16_t, int32_t, uint64_t, REDU_NONE> op(&tilingData, &pipe);
                op.Init(x, indices, updates, y);
                op.Process();
            } else {
                KernelScatterElements<int16_t, int32_t, uint64_t, uint32_t, REDU_NONE> op(pipe);
                op.Init(x, indices, updates, y, userWS, &tilingData);
                op.Process();
            }
        #endif
    #endif

    #if (ORIG_DTYPE_DATA == DT_INT16 || ORIG_DTYPE_DATA == DT_FLOAT16 || ORIG_DTYPE_DATA == DT_BF16)  && ORIG_DTYPE_INDICES == DT_INT64
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX64_REDU_NONE_B16);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX64_REDU_NONE_B16);

        #if TILING_KEY_VAR == SCAC_ELE_UINT32_IDX64_REDU_NONE_B16
            if (tilingData.isDeterministic) {
                KernelScatterElementsDeterm<int16_t, int64_t, uint32_t, REDU_NONE> op(&tilingData, &pipe);
                op.Init(x, indices, updates, y);
                op.Process();
            } else {
                KernelScatterElements<int16_t, int64_t, uint32_t, uint32_t, REDU_NONE> op(pipe);
                op.Init(x, indices, updates, y, userWS, &tilingData);
                op.Process();
            }
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX64_REDU_NONE_B16
            if (tilingData.isDeterministic) {
                KernelScatterElementsDeterm<int16_t, int64_t, uint64_t, REDU_NONE> op(&tilingData, &pipe);
                op.Init(x, indices, updates, y);
                op.Process();
            } else {
                KernelScatterElements<int16_t, int64_t, uint64_t, uint32_t, REDU_NONE> op(pipe);
                op.Init(x, indices, updates, y, userWS, &tilingData);
                op.Process();
            }
        #endif
    #endif

    #if (ORIG_DTYPE_DATA == DT_INT32 || ORIG_DTYPE_DATA == DT_FLOAT)  && ORIG_DTYPE_INDICES == DT_INT32
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX32_REDU_NONE_B32);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX32_REDU_NONE_B32);

        #if TILING_KEY_VAR == SCAC_ELE_UINT32_IDX32_REDU_NONE_B32
            if (tilingData.isDeterministic) {
                KernelScatterElementsDeterm<int32_t, int32_t, uint32_t, REDU_NONE> op(&tilingData, &pipe);
                op.Init(x, indices, updates, y);
                op.Process();
            } else {
                KernelScatterElements<int32_t, int32_t, uint32_t, uint32_t, REDU_NONE> op(pipe);
                op.Init(x, indices, updates, y, userWS, &tilingData);
                op.Process();
            }
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX32_REDU_NONE_B32
            if (tilingData.isDeterministic) {
                KernelScatterElementsDeterm<int32_t, int32_t, uint64_t, REDU_NONE> op(&tilingData, &pipe);
                op.Init(x, indices, updates, y);
                op.Process();
            } else {
                KernelScatterElements<int32_t, int32_t, uint64_t, uint32_t, REDU_NONE> op(pipe);
                op.Init(x, indices, updates, y, userWS, &tilingData);
                op.Process();
            }
        #endif
    #endif

    #if (ORIG_DTYPE_DATA == DT_INT32 || ORIG_DTYPE_DATA == DT_FLOAT)  && ORIG_DTYPE_INDICES == DT_INT64
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX64_REDU_NONE_B32);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX64_REDU_NONE_B32);

        #if TILING_KEY_VAR == SCAC_ELE_UINT32_IDX64_REDU_NONE_B32
            if (tilingData.isDeterministic) {
                KernelScatterElementsDeterm<int32_t, int64_t, uint32_t, REDU_NONE> op(&tilingData, &pipe);
                op.Init(x, indices, updates, y);
                op.Process();
            } else {
                KernelScatterElements<int32_t, int64_t, uint32_t, uint32_t, REDU_NONE> op(pipe);
                op.Init(x, indices, updates, y, userWS, &tilingData);
                op.Process();
            }
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX64_REDU_NONE_B32
            if (tilingData.isDeterministic) {
                KernelScatterElementsDeterm<int32_t, int64_t, uint64_t, REDU_NONE> op(&tilingData, &pipe);
                op.Init(x, indices, updates, y);
                op.Process();
            } else {
                KernelScatterElements<int32_t, int64_t, uint64_t, uint32_t, REDU_NONE> op(pipe);
                op.Init(x, indices, updates, y, userWS, &tilingData);
                op.Process();
            }
        #endif
    #endif

    #if (ORIG_DTYPE_DATA == DT_INT64 || ORIG_DTYPE_DATA == DT_DOUBLE) && ORIG_DTYPE_INDICES == DT_INT32
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX32_REDU_NONE_B64);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX32_REDU_NONE_B64);

        #if TILING_KEY_VAR == SCAC_ELE_UINT32_IDX32_REDU_NONE_B64
            if (tilingData.isDeterministic) {
                KernelScatterElementsDeterm<int64_t, int32_t, uint32_t, REDU_NONE> op(&tilingData, &pipe);
                op.Init(x, indices, updates, y);
                op.Process();
            } else {
                KernelScatterElements<int64_t, int32_t, uint32_t, uint32_t, REDU_NONE> op(pipe);
                op.Init(x, indices, updates, y, userWS, &tilingData);
                op.Process();
            }
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX32_REDU_NONE_B64
            if (tilingData.isDeterministic) {
                KernelScatterElementsDeterm<int64_t, int32_t, uint64_t, REDU_NONE> op(&tilingData, &pipe);
                op.Init(x, indices, updates, y);
                op.Process();
            } else {
                KernelScatterElements<int64_t, int32_t, uint64_t, uint32_t, REDU_NONE> op(pipe);
                op.Init(x, indices, updates, y, userWS, &tilingData);
                op.Process();
            }
        #endif
    #endif

    #if (ORIG_DTYPE_DATA == DT_INT64 || ORIG_DTYPE_DATA == DT_DOUBLE) && ORIG_DTYPE_INDICES == DT_INT64
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX64_REDU_NONE_B64);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX64_REDU_NONE_B64);

        #if TILING_KEY_VAR == SCAC_ELE_UINT32_IDX64_REDU_NONE_B64
            if (tilingData.isDeterministic) {
                KernelScatterElementsDeterm<int64_t, int64_t, uint32_t, REDU_NONE> op(&tilingData, &pipe);
                op.Init(x, indices, updates, y);
                op.Process();
            } else {
                KernelScatterElements<int64_t, int64_t, uint32_t, uint32_t, REDU_NONE> op(pipe);
                op.Init(x, indices, updates, y, userWS, &tilingData);
                op.Process();
            }
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX64_REDU_NONE_B64
            if (tilingData.isDeterministic) {
                KernelScatterElementsDeterm<int64_t, int64_t, uint64_t, REDU_NONE> op(&tilingData, &pipe);
                op.Init(x, indices, updates, y);
                op.Process();
            } else {
                KernelScatterElements<int64_t, int64_t, uint64_t, uint32_t, REDU_NONE> op(pipe);
                op.Init(x, indices, updates, y, userWS, &tilingData);
                op.Process();
            }
        #endif
    #endif

    #if ORIG_DTYPE_DATA == DT_FLOAT && ORIG_DTYPE_INDICES == DT_INT32
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX32_REDU_ADD_FP32);
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX32_REDU_MUL_FP32);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX32_REDU_ADD_FP32);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX32_REDU_MUL_FP32);

        #if TILING_KEY_VAR == SCAC_ELE_UINT32_IDX32_REDU_ADD_FP32
            if (tilingData.isDeterministic) {
                KernelScatterElementsDeterm<float, int32_t, uint32_t, REDU_ADD> op(&tilingData, &pipe);
                op.Init(x, indices, updates, y);
                op.Process();
            } else {
                KernelScatterElements<float, int32_t, uint32_t, uint32_t, REDU_ADD> op(pipe);
                op.Init(x, indices, updates, y, userWS, &tilingData);
                op.Process();
            }
        #elif TILING_KEY_VAR == SCAC_ELE_UINT32_IDX32_REDU_MUL_FP32
            KernelScatterElements<float, int32_t, uint32_t, uint32_t, REDU_MUL> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX32_REDU_ADD_FP32
            if (tilingData.isDeterministic) {
                KernelScatterElementsDeterm<float, int32_t, uint64_t, REDU_ADD> op(&tilingData, &pipe);
                op.Init(x, indices, updates, y);
                op.Process();
            } else {
                KernelScatterElements<float, int32_t, uint64_t, uint32_t, REDU_ADD> op(pipe);
                op.Init(x, indices, updates, y, userWS, &tilingData);
                op.Process();
            }
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX32_REDU_MUL_FP32
            KernelScatterElements<float, int32_t, uint64_t, uint32_t, REDU_MUL> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #endif
    #endif

    #if ORIG_DTYPE_DATA == DT_FLOAT && ORIG_DTYPE_INDICES == DT_INT64
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX64_REDU_ADD_FP32);
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX64_REDU_MUL_FP32);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX64_REDU_ADD_FP32);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX64_REDU_MUL_FP32);

        #if TILING_KEY_VAR == SCAC_ELE_UINT32_IDX64_REDU_ADD_FP32
            if (tilingData.isDeterministic) {
                KernelScatterElementsDeterm<float, int64_t, uint32_t, REDU_ADD> op(&tilingData, &pipe);
                op.Init(x, indices, updates, y);
                op.Process();
            } else {
                KernelScatterElements<float, int64_t, uint32_t, uint32_t, REDU_ADD> op(pipe);
                op.Init(x, indices, updates, y, userWS, &tilingData);
                op.Process();
            }
        #elif TILING_KEY_VAR == SCAC_ELE_UINT32_IDX64_REDU_MUL_FP32
            KernelScatterElements<float, int64_t, uint32_t, uint32_t, REDU_MUL> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX64_REDU_ADD_FP32
            if (tilingData.isDeterministic) {
                KernelScatterElementsDeterm<float, int64_t, uint64_t, REDU_ADD> op(&tilingData, &pipe);
                op.Init(x, indices, updates, y);
                op.Process();
            } else {
                KernelScatterElements<float, int64_t, uint64_t, uint32_t, REDU_ADD> op(pipe);
                op.Init(x, indices, updates, y, userWS, &tilingData);
                op.Process();
            }
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX64_REDU_MUL_FP32
            KernelScatterElements<float, int64_t, uint64_t, uint32_t, REDU_MUL> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #endif
    #endif

    #if ORIG_DTYPE_DATA == DT_FLOAT16 && ORIG_DTYPE_INDICES == DT_INT32
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX32_REDU_ADD_FP16);
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX32_REDU_MUL_FP16);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX32_REDU_ADD_FP16);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX32_REDU_MUL_FP16);

        #if TILING_KEY_VAR == SCAC_ELE_UINT32_IDX32_REDU_ADD_FP16
            if (tilingData.isDeterministic) {
                KernelScatterElementsDeterm<half, int32_t, uint32_t, REDU_ADD> op(&tilingData, &pipe);
                op.Init(x, indices, updates, y);
                op.Process();
            } else {
                KernelScatterElements<half, int32_t, uint32_t, uint32_t, REDU_ADD> op(pipe);
                op.Init(x, indices, updates, y, userWS, &tilingData);
                op.Process();
            }
        #elif TILING_KEY_VAR == SCAC_ELE_UINT32_IDX32_REDU_MUL_FP16
            KernelScatterElements<half, int32_t, uint32_t, uint32_t, REDU_MUL> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX32_REDU_ADD_FP16
            if (tilingData.isDeterministic) {
                KernelScatterElementsDeterm<half, int32_t, uint64_t, REDU_ADD> op(&tilingData, &pipe);
                op.Init(x, indices, updates, y);
                op.Process();
            } else {
                KernelScatterElements<half, int32_t, uint64_t, uint32_t, REDU_ADD> op(pipe);
                op.Init(x, indices, updates, y, userWS, &tilingData);
                op.Process();
            }
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX32_REDU_MUL_FP16
            KernelScatterElements<half, int32_t, uint64_t, uint32_t, REDU_MUL> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #endif
    #endif

    #if ORIG_DTYPE_DATA == DT_FLOAT16 && ORIG_DTYPE_INDICES == DT_INT64
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX64_REDU_ADD_FP16);
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX64_REDU_MUL_FP16);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX64_REDU_ADD_FP16);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX64_REDU_MUL_FP16);

        #if TILING_KEY_VAR == SCAC_ELE_UINT32_IDX64_REDU_ADD_FP16
            if (tilingData.isDeterministic) {
                KernelScatterElementsDeterm<half, int64_t, uint32_t, REDU_ADD> op(&tilingData, &pipe);
                op.Init(x, indices, updates, y);
                op.Process();
            } else {
                KernelScatterElements<half, int64_t, uint32_t, uint32_t, REDU_ADD> op(pipe);
                op.Init(x, indices, updates, y, userWS, &tilingData);
                op.Process();
            }
        #elif TILING_KEY_VAR == SCAC_ELE_UINT32_IDX64_REDU_MUL_FP16
            KernelScatterElements<half, int64_t, uint32_t, uint32_t, REDU_MUL> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX64_REDU_ADD_FP16
            if (tilingData.isDeterministic) {
                KernelScatterElementsDeterm<half, int64_t, uint64_t, REDU_ADD> op(&tilingData, &pipe);
                op.Init(x, indices, updates, y);
                op.Process();
            } else {
                KernelScatterElements<half, int64_t, uint64_t, uint32_t, REDU_ADD> op(pipe);
                op.Init(x, indices, updates, y, userWS, &tilingData);
                op.Process();
            }
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX64_REDU_MUL_FP16
            KernelScatterElements<half, int64_t, uint64_t, uint32_t, REDU_MUL> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #endif
    #endif

    #if ORIG_DTYPE_DATA == DT_BF16 && ORIG_DTYPE_INDICES == DT_INT32
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX32_REDU_ADD_BF16);
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX32_REDU_MUL_BF16);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX32_REDU_ADD_BF16);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX32_REDU_MUL_BF16);

        #if TILING_KEY_VAR == SCAC_ELE_UINT32_IDX32_REDU_ADD_BF16
            if (tilingData.isDeterministic) {
                KernelScatterElementsDeterm<bfloat16_t, int32_t, uint32_t, REDU_ADD> op(&tilingData, &pipe);
                op.Init(x, indices, updates, y);
                op.Process();
            } else {
                KernelScatterElements<bfloat16_t, int32_t, uint32_t, uint32_t, REDU_ADD> op(pipe);
                op.Init(x, indices, updates, y, userWS, &tilingData);
                op.Process();
            }
        #elif TILING_KEY_VAR == SCAC_ELE_UINT32_IDX32_REDU_MUL_BF16
            KernelScatterElements<bfloat16_t, int32_t, uint32_t, uint32_t, REDU_MUL> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX32_REDU_ADD_BF16
            if (tilingData.isDeterministic) {
                KernelScatterElementsDeterm<bfloat16_t, int32_t, uint64_t, REDU_ADD> op(&tilingData, &pipe);
                op.Init(x, indices, updates, y);
                op.Process();
            } else {
                KernelScatterElements<bfloat16_t, int32_t, uint64_t, uint32_t, REDU_ADD> op(pipe);
                op.Init(x, indices, updates, y, userWS, &tilingData);
                op.Process();
            }
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX32_REDU_MUL_BF16
            KernelScatterElements<bfloat16_t, int32_t, uint64_t, uint32_t, REDU_MUL> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #endif
    #endif

    #if ORIG_DTYPE_DATA == DT_BF16 && ORIG_DTYPE_INDICES == DT_INT64
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX64_REDU_ADD_BF16);
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX64_REDU_MUL_BF16);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX64_REDU_ADD_BF16);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX64_REDU_MUL_BF16);

        #if TILING_KEY_VAR == SCAC_ELE_UINT32_IDX64_REDU_ADD_BF16
            if (tilingData.isDeterministic) {
                KernelScatterElementsDeterm<bfloat16_t, int64_t, uint32_t, REDU_ADD> op(&tilingData, &pipe);
                op.Init(x, indices, updates, y);
                op.Process();
            } else {
                KernelScatterElements<bfloat16_t, int64_t, uint32_t, uint32_t, REDU_ADD> op(pipe);
                op.Init(x, indices, updates, y, userWS, &tilingData);
                op.Process();
            }
        #elif TILING_KEY_VAR == SCAC_ELE_UINT32_IDX64_REDU_MUL_BF16
            KernelScatterElements<bfloat16_t, int64_t, uint32_t, uint32_t, REDU_MUL> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX64_REDU_ADD_BF16
            if (tilingData.isDeterministic) {
                KernelScatterElementsDeterm<bfloat16_t, int64_t, uint64_t, REDU_ADD> op(&tilingData, &pipe);
                op.Init(x, indices, updates, y);
                op.Process();
            } else {
                KernelScatterElements<bfloat16_t, int64_t, uint64_t, uint32_t, REDU_ADD> op(pipe);
                op.Init(x, indices, updates, y, userWS, &tilingData);
                op.Process();
            }
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX64_REDU_MUL_BF16
            KernelScatterElements<bfloat16_t, int64_t, uint64_t, uint32_t, REDU_MUL> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #endif
    #endif

    #if ORIG_DTYPE_DATA == DT_INT8 && ORIG_DTYPE_INDICES == DT_INT32
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX32_REDU_ADD_INT8);
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX32_REDU_MUL_INT8);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX32_REDU_ADD_INT8);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX32_REDU_MUL_INT8);

        #if TILING_KEY_VAR == SCAC_ELE_UINT32_IDX32_REDU_ADD_INT8
            KernelScatterElements<int8_t, int32_t, uint32_t, int32_t, REDU_ADD> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT32_IDX32_REDU_MUL_INT8
            KernelScatterElements<int8_t, int32_t, uint32_t, uint32_t, REDU_MUL> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX32_REDU_ADD_INT8
            KernelScatterElements<int8_t, int32_t, uint64_t, int32_t, REDU_ADD> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX32_REDU_MUL_INT8
            KernelScatterElements<int8_t, int32_t, uint64_t, uint32_t, REDU_MUL> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #endif
    #endif

    #if ORIG_DTYPE_DATA == DT_INT8 && ORIG_DTYPE_INDICES == DT_INT64
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX64_REDU_ADD_INT8);
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX64_REDU_MUL_INT8);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX64_REDU_ADD_INT8);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX64_REDU_MUL_INT8);

        #if TILING_KEY_VAR == SCAC_ELE_UINT32_IDX64_REDU_ADD_INT8
            KernelScatterElements<int8_t, int64_t, uint32_t, int32_t, REDU_ADD> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT32_IDX64_REDU_MUL_INT8
            KernelScatterElements<int8_t, int64_t, uint32_t, uint32_t, REDU_MUL> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX64_REDU_ADD_INT8
            KernelScatterElements<int8_t, int64_t, uint64_t, int32_t, REDU_ADD> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX64_REDU_MUL_INT8
            KernelScatterElements<int8_t, int64_t, uint64_t, uint32_t, REDU_MUL> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #endif
    #endif

    #if ORIG_DTYPE_DATA == DT_UINT8 && ORIG_DTYPE_INDICES == DT_INT32
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX32_REDU_ADD_UINT8);
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX32_REDU_MUL_UINT8);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX32_REDU_ADD_UINT8);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX32_REDU_MUL_UINT8);

        #if TILING_KEY_VAR == SCAC_ELE_UINT32_IDX32_REDU_ADD_UINT8
            KernelScatterElements<uint8_t, int32_t, uint32_t, uint32_t, REDU_ADD> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT32_IDX32_REDU_MUL_UINT8
            KernelScatterElements<uint8_t, int32_t, uint32_t, uint32_t, REDU_MUL> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX32_REDU_ADD_UINT8
            KernelScatterElements<uint8_t, int32_t, uint64_t, uint32_t, REDU_ADD> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX32_REDU_MUL_UINT8
            KernelScatterElements<uint8_t, int32_t, uint64_t, uint32_t, REDU_MUL> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #endif
    #endif

    #if ORIG_DTYPE_DATA == DT_UINT8 && ORIG_DTYPE_INDICES == DT_INT64
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX64_REDU_ADD_UINT8);
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX64_REDU_MUL_UINT8);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX64_REDU_ADD_UINT8);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX64_REDU_MUL_UINT8);

        #if TILING_KEY_VAR == SCAC_ELE_UINT32_IDX64_REDU_ADD_UINT8
            KernelScatterElements<uint8_t, int64_t, uint32_t, uint32_t, REDU_ADD> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT32_IDX64_REDU_MUL_UINT8
            KernelScatterElements<uint8_t, int64_t, uint32_t, uint32_t, REDU_MUL> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX64_REDU_ADD_UINT8
            KernelScatterElements<uint8_t, int64_t, uint64_t, uint32_t, REDU_ADD> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX64_REDU_MUL_UINT8
            KernelScatterElements<uint8_t, int64_t, uint64_t, uint32_t, REDU_MUL> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #endif
    #endif

    #if ORIG_DTYPE_DATA == DT_INT32 && ORIG_DTYPE_INDICES == DT_INT32
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX32_REDU_ADD_INT32);
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX32_REDU_MUL_INT32);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX32_REDU_ADD_INT32);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX32_REDU_MUL_INT32);

        #if TILING_KEY_VAR == SCAC_ELE_UINT32_IDX32_REDU_ADD_INT32
            KernelScatterElements<int32_t, int32_t, uint32_t, uint32_t, REDU_ADD> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT32_IDX32_REDU_MUL_INT32
            KernelScatterElements<int32_t, int32_t, uint32_t, uint32_t, REDU_MUL> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX32_REDU_ADD_INT32
            KernelScatterElements<int32_t, int32_t, uint64_t, uint32_t, REDU_ADD> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX32_REDU_MUL_INT32
            KernelScatterElements<int32_t, int32_t, uint64_t, uint32_t, REDU_MUL> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #endif
    #endif

    #if ORIG_DTYPE_DATA == DT_INT32 && ORIG_DTYPE_INDICES == DT_INT64
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX64_REDU_ADD_INT32);
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX64_REDU_MUL_INT32);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX64_REDU_ADD_INT32);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX64_REDU_MUL_INT32);

        #if TILING_KEY_VAR == SCAC_ELE_UINT32_IDX64_REDU_ADD_INT32
            KernelScatterElements<int32_t, int64_t, uint32_t, uint32_t, REDU_ADD> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT32_IDX64_REDU_MUL_INT32
            KernelScatterElements<int32_t, int64_t, uint32_t, uint32_t, REDU_MUL> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX64_REDU_ADD_INT32
            KernelScatterElements<int32_t, int64_t, uint64_t, uint32_t, REDU_ADD> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX64_REDU_MUL_INT32
            KernelScatterElements<int32_t, int64_t, uint64_t, uint32_t, REDU_MUL> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #endif
    #endif

    #if ORIG_DTYPE_DATA == DT_INT16 && ORIG_DTYPE_INDICES == DT_INT32
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX32_REDU_ADD_INT16);
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX32_REDU_MUL_INT16);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX32_REDU_ADD_INT16);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX32_REDU_MUL_INT16);

        #if TILING_KEY_VAR == SCAC_ELE_UINT32_IDX32_REDU_ADD_INT16
            KernelScatterElements<int16_t, int32_t, uint32_t, int32_t, REDU_ADD> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT32_IDX32_REDU_MUL_INT16
            KernelScatterElements<int16_t, int32_t, uint32_t, uint32_t, REDU_MUL> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX32_REDU_ADD_INT16
            KernelScatterElements<int16_t, int32_t, uint64_t, int32_t, REDU_ADD> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX32_REDU_MUL_INT16
            KernelScatterElements<int16_t, int32_t, uint64_t, uint32_t, REDU_MUL> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #endif
    #endif

    #if ORIG_DTYPE_DATA == DT_INT16 && ORIG_DTYPE_INDICES == DT_INT64
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX64_REDU_ADD_INT16);
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX64_REDU_MUL_INT16);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX64_REDU_ADD_INT16);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX64_REDU_MUL_INT16);

        #if TILING_KEY_VAR == SCAC_ELE_UINT32_IDX64_REDU_ADD_INT16
            KernelScatterElements<int16_t, int64_t, uint32_t, int32_t, REDU_ADD> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT32_IDX64_REDU_MUL_INT16
            KernelScatterElements<int16_t, int64_t, uint32_t, uint32_t, REDU_MUL> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX64_REDU_ADD_INT16
            KernelScatterElements<int16_t, int64_t, uint64_t, int32_t, REDU_ADD> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX64_REDU_MUL_INT16
            KernelScatterElements<int16_t, int64_t, uint64_t, uint32_t, REDU_MUL> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #endif
    #endif

    #if ORIG_DTYPE_DATA == DT_INT64 && ORIG_DTYPE_INDICES == DT_INT32
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX32_REDU_ADD_INT64);
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX32_REDU_MUL_INT64);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX32_REDU_ADD_INT64);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX32_REDU_MUL_INT64);

        #if TILING_KEY_VAR == SCAC_ELE_UINT32_IDX32_REDU_ADD_INT64
            KernelScatterElements<int64_t, int32_t, uint32_t, uint32_t, REDU_ADD> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT32_IDX32_REDU_MUL_INT64
            KernelScatterElements<int64_t, int32_t, uint32_t, uint32_t, REDU_MUL> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX32_REDU_ADD_INT64
            KernelScatterElements<int64_t, int32_t, uint64_t, uint32_t, REDU_ADD> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX32_REDU_MUL_INT64
            KernelScatterElements<int64_t, int32_t, uint64_t, uint32_t, REDU_MUL> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #endif
    #endif

    #if ORIG_DTYPE_DATA == DT_INT64 && ORIG_DTYPE_INDICES == DT_INT64
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX64_REDU_ADD_INT64);
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX64_REDU_MUL_INT64);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX64_REDU_ADD_INT64);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX64_REDU_MUL_INT64);

        #if TILING_KEY_VAR == SCAC_ELE_UINT32_IDX64_REDU_ADD_INT64
            KernelScatterElements<int64_t, int64_t, uint32_t, uint32_t, REDU_ADD> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT32_IDX64_REDU_MUL_INT64
            KernelScatterElements<int64_t, int64_t, uint32_t, uint32_t, REDU_MUL> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX64_REDU_ADD_INT64
            KernelScatterElements<int64_t, int64_t, uint64_t, uint32_t, REDU_ADD> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX64_REDU_MUL_INT64
            KernelScatterElements<int64_t, int64_t, uint64_t, uint32_t, REDU_MUL> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #endif
    #endif

    #if ORIG_DTYPE_DATA == DT_BOOL && ORIG_DTYPE_INDICES == DT_INT32
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX32_REDU_ADD_BOOL);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX32_REDU_ADD_BOOL);

        #if TILING_KEY_VAR == SCAC_ELE_UINT32_IDX32_REDU_ADD_BOOL
            KernelScatterElements<uint8_t, int32_t, uint32_t, half, REDU_ADD> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX32_REDU_ADD_BOOL
            KernelScatterElements<uint8_t, int32_t, uint64_t, half, REDU_ADD> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #endif
    #endif
    
    #if ORIG_DTYPE_DATA == DT_BOOL && ORIG_DTYPE_INDICES == DT_INT64
        TILING_KEY_IS(SCAC_ELE_UINT32_IDX64_REDU_ADD_BOOL);
        TILING_KEY_IS(SCAC_ELE_UINT64_IDX64_REDU_ADD_BOOL);

        #if TILING_KEY_VAR == SCAC_ELE_UINT32_IDX64_REDU_ADD_BOOL
            KernelScatterElements<uint8_t, int64_t, uint32_t, half, REDU_ADD> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #elif TILING_KEY_VAR == SCAC_ELE_UINT64_IDX64_REDU_ADD_BOOL
            KernelScatterElements<uint8_t, int64_t, uint64_t, half, REDU_ADD> op(pipe);
            op.Init(x, indices, updates, y, userWS, &tilingData);
            op.Process();
        #endif
    #endif
}