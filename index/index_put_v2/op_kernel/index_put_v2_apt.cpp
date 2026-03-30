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
 * \file index_put_v2.cpp
 * \brief Ascendc IndexPutV2 kernel
 */

#include "../index/arch35/index.h"
#include "../index/arch35/index_no_continuous.h"
#include "arch35/index_put_v2_simd.h"

using namespace Index;
using namespace IndexPutV2;

extern "C" __global__ __aicore__ void index_put_v2(GM_ADDR inputX, GM_ADDR value, GM_ADDR indexedSizes,
                                                   GM_ADDR indexedStrides, GM_ADDR indices, GM_ADDR output,
                                                   GM_ADDR workspace, GM_ADDR tiling) {
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    if (TILING_KEY_IS(4001)) {
        GET_TILING_DATA_WITH_STRUCT(IndexPutV2SimdTilingData, tilingData, tiling);
        IndexPutV2Simd<int8_t, int32_t, true> op;
        op.Init(inputX, value, indexedSizes, indexedStrides, indices, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(4002)) {
        GET_TILING_DATA_WITH_STRUCT(IndexPutV2SimdTilingData, tilingData, tiling);
        IndexPutV2Simd<half, int32_t, true> op;
        op.Init(inputX, value, indexedSizes, indexedStrides, indices, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(4003)) {
        GET_TILING_DATA_WITH_STRUCT(IndexPutV2SimdTilingData, tilingData, tiling);
        IndexPutV2Simd<bfloat16_t, int32_t, true> op;
        op.Init(inputX, value, indexedSizes, indexedStrides, indices, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(4004)) {
        GET_TILING_DATA_WITH_STRUCT(IndexPutV2SimdTilingData, tilingData, tiling);
        IndexPutV2Simd<int32_t, int32_t, true> op;
        op.Init(inputX, value, indexedSizes, indexedStrides, indices, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(4005)) {
        GET_TILING_DATA_WITH_STRUCT(IndexPutV2SimdTilingData, tilingData, tiling);
        IndexPutV2Simd<float, int32_t, true> op;
        op.Init(inputX, value, indexedSizes, indexedStrides, indices, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(4101)) {
        GET_TILING_DATA_WITH_STRUCT(IndexPutV2SimdTilingData, tilingData, tiling);
        IndexPutV2Simd<int8_t, int64_t, true> op;
        op.Init(inputX, value, indexedSizes, indexedStrides, indices, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(4102)) {
        GET_TILING_DATA_WITH_STRUCT(IndexPutV2SimdTilingData, tilingData, tiling);
        IndexPutV2Simd<half, int64_t, true> op;
        op.Init(inputX, value, indexedSizes, indexedStrides, indices, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(4103)) {
        GET_TILING_DATA_WITH_STRUCT(IndexPutV2SimdTilingData, tilingData, tiling);
        IndexPutV2Simd<bfloat16_t, int64_t, true> op;
        op.Init(inputX, value, indexedSizes, indexedStrides, indices, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(4104)) {
        GET_TILING_DATA_WITH_STRUCT(IndexPutV2SimdTilingData, tilingData, tiling);
        IndexPutV2Simd<int32_t, int64_t, true> op;
        op.Init(inputX, value, indexedSizes, indexedStrides, indices, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(4105)) {
        GET_TILING_DATA_WITH_STRUCT(IndexPutV2SimdTilingData, tilingData, tiling);
        IndexPutV2Simd<float, int64_t, true> op;
        op.Init(inputX, value, indexedSizes, indexedStrides, indices, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3000)) {
        GET_TILING_DATA_WITH_STRUCT(IndexPutV2SimdTilingData, tilingData, tiling);
        IndexPutV2Simd<uint8_t, int32_t, false>op;
        op.Init(inputX, value, indexedSizes, indexedStrides, indices, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3001)) {
        GET_TILING_DATA_WITH_STRUCT(IndexPutV2SimdTilingData, tilingData, tiling);
        IndexPutV2Simd<int8_t, int32_t, false>op;
        op.Init(inputX, value, indexedSizes, indexedStrides, indices, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3002)) {
        GET_TILING_DATA_WITH_STRUCT(IndexPutV2SimdTilingData, tilingData, tiling);
        IndexPutV2Simd<half, int32_t, false>op;
        op.Init(inputX, value, indexedSizes, indexedStrides, indices, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3003)) {
        GET_TILING_DATA_WITH_STRUCT(IndexPutV2SimdTilingData, tilingData, tiling);
        IndexPutV2Simd<bfloat16_t, int32_t, false>op;
        op.Init(inputX, value, indexedSizes, indexedStrides, indices, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3004)) {
        GET_TILING_DATA_WITH_STRUCT(IndexPutV2SimdTilingData, tilingData, tiling);
        IndexPutV2Simd<int32_t, int32_t, false>op;
        op.Init(inputX, value, indexedSizes, indexedStrides, indices, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3005)) {
        GET_TILING_DATA_WITH_STRUCT(IndexPutV2SimdTilingData, tilingData, tiling);
        IndexPutV2Simd<float, int32_t, false>op;
        op.Init(inputX, value, indexedSizes, indexedStrides, indices, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3006)) {
        GET_TILING_DATA_WITH_STRUCT(IndexPutV2SimdTilingData, tilingData, tiling);
        IndexPutV2Simd<int64_t, int32_t, false>op;
        op.Init(inputX, value, indexedSizes, indexedStrides, indices, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3007)) {
        GET_TILING_DATA_WITH_STRUCT(IndexPutV2SimdTilingData, tilingData, tiling);
        IndexPutV2Simd<bool, int32_t, false>op;
        op.Init(inputX, value, indexedSizes, indexedStrides, indices, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3100)) {
        GET_TILING_DATA_WITH_STRUCT(IndexPutV2SimdTilingData, tilingData, tiling);
        IndexPutV2Simd<int8_t, int64_t, false>op;
        op.Init(inputX, value, indexedSizes, indexedStrides, indices, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3101)) {
        GET_TILING_DATA_WITH_STRUCT(IndexPutV2SimdTilingData, tilingData, tiling);
        IndexPutV2Simd<int8_t, int64_t, false>op;
        op.Init(inputX, value, indexedSizes, indexedStrides, indices, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3102)) {
        GET_TILING_DATA_WITH_STRUCT(IndexPutV2SimdTilingData, tilingData, tiling);
        IndexPutV2Simd<half, int64_t, false>op;
        op.Init(inputX, value, indexedSizes, indexedStrides, indices, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3103)) {
        GET_TILING_DATA_WITH_STRUCT(IndexPutV2SimdTilingData, tilingData, tiling);
        IndexPutV2Simd<bfloat16_t, int64_t, false>op;
        op.Init(inputX, value, indexedSizes, indexedStrides, indices, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3104)) {
        GET_TILING_DATA_WITH_STRUCT(IndexPutV2SimdTilingData, tilingData, tiling);
        IndexPutV2Simd<int32_t, int64_t, false>op;
        op.Init(inputX, value, indexedSizes, indexedStrides, indices, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3105)) {
        GET_TILING_DATA_WITH_STRUCT(IndexPutV2SimdTilingData, tilingData, tiling);
        IndexPutV2Simd<float, int64_t, false>op;
        op.Init(inputX, value, indexedSizes, indexedStrides, indices, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3106)) {
        GET_TILING_DATA_WITH_STRUCT(IndexPutV2SimdTilingData, tilingData, tiling);
        IndexPutV2Simd<int64_t, int32_t, false>op;
        op.Init(inputX, value, indexedSizes, indexedStrides, indices, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3107)) {
        GET_TILING_DATA_WITH_STRUCT(IndexPutV2SimdTilingData, tilingData, tiling);
        IndexPutV2Simd<bool, int32_t, false>op;
        op.Init(inputX, value, indexedSizes, indexedStrides, indices, tilingData);
        op.Process();
    }

    GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, simttilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (simttilingData.accumulateMode) {
        if (TILING_KEY_IS(20002)) {
            GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
            KernelIndexNoContiguous<half, IndexPutAdd<half>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process(); 
        } else if (TILING_KEY_IS(20003)) {
            GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
            KernelIndexNoContiguous<bfloat16_t, IndexPutAdd<bfloat16_t>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process(); 
        } else if (TILING_KEY_IS(20004)) {
            GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
            KernelIndexNoContiguous<int32_t, IndexPutAdd<int32_t>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process(); 
        } else if (TILING_KEY_IS(20005)) {
            GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
            KernelIndexNoContiguous<float, IndexPutAdd<float>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process(); 
        } else if (TILING_KEY_IS(20008)) {
            GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
            KernelIndexNoContiguous<int64_t, IndexPutAdd<int64_t>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process(); 
        } else if (TILING_KEY_IS(20011)) {
            GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
            KernelIndexNoContiguous<bool, IndexPutAdd<bool>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process(); 
        } else if (TILING_KEY_IS(20102)) {
            GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
            KernelIndexNoContiguous<half, IndexPutAdd<half>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process(); 
        } else if (TILING_KEY_IS(20103)) {
            GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
            KernelIndexNoContiguous<bfloat16_t, IndexPutAdd<bfloat16_t>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process(); 
        } else if (TILING_KEY_IS(20104)) {
            GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
            KernelIndexNoContiguous<int32_t, IndexPutAdd<int32_t>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process(); 
        } else if (TILING_KEY_IS(20105)) {
            GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
            KernelIndexNoContiguous<float, IndexPutAdd<float>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process(); 
        } else if (TILING_KEY_IS(20108)) {
            GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
            KernelIndexNoContiguous<int64_t, IndexPutAdd<int64_t>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process(); 
        } else if (TILING_KEY_IS(20111)) {
            GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
            KernelIndexNoContiguous<bool, IndexPutAdd<bool>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process(); 
        } else if (TILING_KEY_IS(2)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<half, IndexPutAdd<half>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(3)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<bfloat16_t, IndexPutAdd<bfloat16_t>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(4)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<int32_t, IndexPutAdd<int32_t>, int32_t,  uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(6)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<int32_t, IndexPutAdd<int32_t>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(5)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<float, IndexPutAdd<float>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(8)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<int64_t, IndexPutAdd<int64_t>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(10)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<int64_t, IndexPutAdd<int64_t>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(11)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<bool, IndexPutAdd<bool>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(12)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<int64_t, IndexPutAdd<int64_t>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(102)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<half, IndexPutAdd<half>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(103)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<bfloat16_t, IndexPutAdd<bfloat16_t>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(104)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<int32_t, IndexPutAdd<int32_t>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(106)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<int32_t, IndexPutAdd<int32_t>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(105)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<float, IndexPutAdd<float>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(108)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<int64_t, IndexPutAdd<int64_t>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(110)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<int64_t, IndexPutAdd<int64_t>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(111)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<bool, IndexPutAdd<bool>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(112)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<int64_t, IndexPutAdd<int64_t>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        }
    } else {
        if (TILING_KEY_IS(20000)) {
            GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
            KernelIndexNoContiguous<uint8_t, IndexPutAssign<uint8_t>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(20001)) {
            GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
            KernelIndexNoContiguous<int8_t, IndexPutAssign<int8_t>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();        
        } else if (TILING_KEY_IS(20002)) {
            GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
            KernelIndexNoContiguous<half, IndexPutAssign<half>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process(); 
        } else if (TILING_KEY_IS(20003)) {
            GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
            KernelIndexNoContiguous<bfloat16_t, IndexPutAssign<bfloat16_t>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process(); 
        } else if (TILING_KEY_IS(20004)) {
            GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
            KernelIndexNoContiguous<int32_t, IndexPutAssign<int32_t>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process(); 
        } else if (TILING_KEY_IS(20005)) {
            GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
            KernelIndexNoContiguous<float, IndexPutAssign<float>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process(); 
        } else if (TILING_KEY_IS(20008)) {
            GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
            KernelIndexNoContiguous<int64_t, IndexPutAssign<int64_t>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process(); 
        } else if (TILING_KEY_IS(20011)) {
            GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
            KernelIndexNoContiguous<bool, IndexPutAssign<bool>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process(); 
        } else if (TILING_KEY_IS(20100)) {
            GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
            KernelIndexNoContiguous<uint8_t, IndexPutAssign<uint8_t>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(20101)) {
            GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
            KernelIndexNoContiguous<int8_t, IndexPutAssign<int8_t>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();        
        } else if (TILING_KEY_IS(20102)) {
            GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
            KernelIndexNoContiguous<half, IndexPutAssign<half>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process(); 
        } else if (TILING_KEY_IS(20103)) {
            GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
            KernelIndexNoContiguous<bfloat16_t, IndexPutAssign<bfloat16_t>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process(); 
        } else if (TILING_KEY_IS(20104)) {
            GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
            KernelIndexNoContiguous<int32_t, IndexPutAssign<int32_t>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process(); 
        } else if (TILING_KEY_IS(20105)) {
            GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
            KernelIndexNoContiguous<float, IndexPutAssign<float>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process(); 
        } else if (TILING_KEY_IS(20108)) {
            GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
            KernelIndexNoContiguous<int64_t, IndexPutAssign<int64_t>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process(); 
        } else if (TILING_KEY_IS(20111)) {
            GET_TILING_DATA_WITH_STRUCT(IndexNonContinuousTilingData, tilingData, tiling);
            KernelIndexNoContiguous<bool, IndexPutAssign<bool>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process(); 
        } else if (TILING_KEY_IS(0)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<uint8_t, IndexPutAssign<uint8_t>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(1)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<int8_t, IndexPutAssign<int8_t>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(2)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<half, IndexPutAssign<half>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(3)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<bfloat16_t, IndexPutAssign<bfloat16_t>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(4)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<int32_t, IndexPutAssign<int32_t>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(6)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<int32_t, IndexPutAssign<int32_t>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(5)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<float, IndexPutAssign<float>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(8)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<int64_t, IndexPutAssign<int64_t>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(10)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<int64_t, IndexPutAssign<int64_t>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(11)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<bool, IndexPutAssign<bool>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(12)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<int64_t, IndexPutAssign<int64_t>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(16)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<int4, IndexPutAssign<int4>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(20)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<int4, IndexPutAssign<int4>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(24)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<int4, IndexPutAssign<int4>, int32_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        }else if (TILING_KEY_IS(100)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<uint8_t, IndexPutAssign<uint8_t>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(101)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<int8_t, IndexPutAssign<int8_t>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(102)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<half, IndexPutAssign<half>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(103)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<bfloat16_t, IndexPutAssign<bfloat16_t>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(104)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<int32_t, IndexPutAssign<int32_t>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(106)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<int32_t, IndexPutAssign<int32_t>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(105)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<float, IndexPutAssign<float>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(108)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<int64_t, IndexPutAssign<int64_t>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(110)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<int64_t, IndexPutAssign<int64_t>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(111)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<bool, IndexPutAssign<bool>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(112)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<int64_t, IndexPutAssign<int64_t>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(116)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<int4, IndexPutAssign<int4>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(120)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<int4, IndexPutAssign<int4>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        } else if (TILING_KEY_IS(124)) {
            GET_TILING_DATA_WITH_STRUCT(IndexSimtTilingData, tilingData, tiling);
            KernelIndex<int4, IndexPutAssign<int4>, int64_t, uint32_t> op;
            op.Init(output, inputX, indexedSizes, indexedStrides, indices, tilingData, value);
            op.Process();
        }
    }
}