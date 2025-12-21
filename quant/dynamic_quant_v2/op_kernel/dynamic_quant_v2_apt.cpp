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
 * \file dynamic_quant_v2_regbase_apt.cpp
 * \brief dynamic_quant_v2 kernel enter
 */

#include "kernel_operator.h"
#include "../dynamic_quant/v35/dynamic_quant_struct.h"
#include "../dynamic_quant/v35/dynamic_quant_regbase_full_load.h"
#include "../dynamic_quant/v35/dynamic_quant_regbase_moe_full_load.h"
#include "../dynamic_quant/v35/dynamic_quant_regbase_large_shape_db.h"
#include "../dynamic_quant/v35/dynamic_quant_regbase_moe_large_shape.h"
#include "../dynamic_quant/v35/dynamic_quant_regbase_full_load_pertensor.h"
#include "../dynamic_quant/v35/dynamic_quant_regbase_large_shape_db_pertensor.h"

namespace
{
using namespace AscendC;
using namespace DynamicQuantNDOpt;
using namespace DynamicQuantNDOpt2;
using namespace DynamicQuantNDPerTensorOpt;
using namespace DynamicQuantNDPerTensorOpt2;

template<uint64_t V>
using UIntAsBool = std::integral_constant<bool, V != 0>;

template <uint64_t useDb, uint64_t quantMode, uint64_t hasSmooth, uint64_t isSymmetrical>
__global__ __aicore__ void dynamic_quant_v2(GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR group_index, GM_ADDR y,
                                            GM_ADDR scale, GM_ADDR offset, GM_ADDR workSpace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR usrWorkspace = GetUserWorkspace(workSpace);

    if constexpr (quantMode == TPL_COMMON_FULL_LOAD) {
        DynamicQuantRegbaseFullLoad<
            DTYPE_X, DTYPE_Y,
            UIntAsBool<hasSmooth>::value,
            static_cast<uint32_t>(useDb + 1),
            UIntAsBool<isSymmetrical>::value
        > op(&pipe);
        op.Init(x, smooth_scales, y, scale, offset, workSpace, &tilingData);
        op.Process();
    } else if constexpr (quantMode == TPL_COMMON_LARGE_SHAPE) {
        DynamicQuantLargeShapeDb<DTYPE_X, DTYPE_Y,
            static_cast<int64_t>(hasSmooth),
            UIntAsBool<isSymmetrical>::value
        > op(&pipe);
        op.Init(x, smooth_scales, y, scale, offset, workSpace, tilingData);
        op.Process();
    } else if constexpr (quantMode == TPL_MOE_FULL_LOAD) {
        DynamicQuantV2Op::DynamicQuantRegbaseFullLoadMOE<
            DTYPE_X, DTYPE_Y,
            UIntAsBool<hasSmooth>::value,
            static_cast<uint32_t>(useDb + 1),
            UIntAsBool<isSymmetrical>::value
        > op(&pipe);
        op.Init(x, smooth_scales, group_index, y, scale, offset, workSpace, &tilingData);
        op.Process();
    } else if constexpr (quantMode == TPL_MOE_LARGE_SHAPE) {
        DynamicQuantRegBase::DynamicQuantLargeShapeMOE<
            DTYPE_X, DTYPE_Y,
            static_cast<int64_t>(hasSmooth),
            UIntAsBool<isSymmetrical>::value
        > op(&pipe);
        op.Init(x, smooth_scales, group_index, y, scale, offset, workSpace, tilingData);
        op.Process();
    } else if constexpr (quantMode == TPL_PER_TENSOR_FULL_LOAD) {
        DynamicQuantRegbaseFullLoadPertensor<DTYPE_X, DTYPE_Y, static_cast<int64_t>(hasSmooth),static_cast<uint32_t>(useDb + 1),UIntAsBool<isSymmetrical>::value> op(&pipe);
        op.Init(x, smooth_scales, y, scale, offset, usrWorkspace, &tilingData);
        op.Process();
    } else if constexpr (quantMode == TPL_PER_TENSOR_LARGE_SHAPE) {
        DynamicQuantLargeShapeDbPertensor<DTYPE_X, DTYPE_Y, static_cast<int64_t>(hasSmooth),UIntAsBool<isSymmetrical>::value> op(&pipe);
        op.Init(x, smooth_scales, y, scale, offset, usrWorkspace, tilingData);
        op.Process();
    } else if constexpr (quantMode == TPL_EMPTY_TENSOR) {
        return ;
    }
}
}