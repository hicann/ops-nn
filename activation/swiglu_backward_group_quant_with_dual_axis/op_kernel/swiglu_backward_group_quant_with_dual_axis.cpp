/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "arch35/swiglu_backward_group_quant_with_dual_axis_regbase.h"
#include "arch35/swiglu_backward_group_quant_with_dual_axis_tilingdata.h"
#include "arch35/swiglu_backward_group_quant_with_dual_axis_tiling_key.h"

using namespace SwigluBackwardGroupQuantWithDualAxisOp;

#if (__NPU_ARCH__ == 3510)
#define SAVE_MX_OVERFLOW_MODE() \
    int64_t overflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>()
#define RESTORE_MX_OVERFLOW_MODE() AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(overflowMode)
#else
#define SAVE_MX_OVERFLOW_MODE()
#define RESTORE_MX_OVERFLOW_MODE()
#endif

#define INVOKE_SWIGLU_BACKWARD_GROUP_QUANT_WITH_DUAL_AXIS(mode, hasGroupIndex, hasWeight, weightType, hasClampLimit) \
    do {                                                                                                             \
        SAVE_MX_OVERFLOW_MODE();                                                                                     \
        GET_TILING_DATA_WITH_STRUCT(SwigluBackwardGroupQuantWithDualAxisMxTilingData, tilingData, tiling);           \
        AscendC::TPipe pipe;                                                                                         \
        SwigluBackwardGroupQuantWithDualAxisMx::SwigluBackwardGroupQuantWithDualAxisMxBase<                          \
            DTYPE_X, DTYPE_Y1, weightType, mode, hasGroupIndex == TPL_GROUP_INDEX, hasWeight == TPL_WEIGHT,          \
            hasClampLimit == TPL_CLAMP>                                                                              \
            op(&tilingData, &pipe);                                                                                  \
        op.Init(x, gradY, weight, yOrigin, groupIndex, nullptr, gradWeight, y1, mxScale1, y2, mxScale2);             \
        op.Process();                                                                                                \
        RESTORE_MX_OVERFLOW_MODE();                                                                                  \
    } while (0)

template <uint64_t mode = TPL_MODE_ROTATE, uint64_t quantMode = TPL_MX_MODE,
          uint64_t hasGroupIndex = TPL_NO_GROUP_INDEX, uint64_t hasWeight = TPL_NO_WEIGHT,
          uint64_t weightDtype = TPL_WEIGHT_FP16, uint64_t hasClampLimit = TPL_NO_CLAMP>
__global__ __aicore__ void swiglu_backward_group_quant_with_dual_axis(GM_ADDR gradY, GM_ADDR x, GM_ADDR weight,
                                                                      GM_ADDR yOrigin, GM_ADDR groupIndex, GM_ADDR y1,
                                                                      GM_ADDR mxScale1, GM_ADDR y2, GM_ADDR mxScale2,
                                                                      GM_ADDR gradWeight, GM_ADDR workspace,
                                                                      GM_ADDR tiling)
{
    static_assert(quantMode == TPL_MX_MODE, "only MX mode is supported");
    REGISTER_TILING_DEFAULT(SwigluBackwardGroupQuantWithDualAxisMxTilingData);
    if constexpr (weightDtype == TPL_WEIGHT_FP16) {
        INVOKE_SWIGLU_BACKWARD_GROUP_QUANT_WITH_DUAL_AXIS(mode, hasGroupIndex, hasWeight, half, hasClampLimit);
    } else if constexpr (weightDtype == TPL_WEIGHT_BF16) {
        INVOKE_SWIGLU_BACKWARD_GROUP_QUANT_WITH_DUAL_AXIS(mode, hasGroupIndex, hasWeight, bfloat16_t, hasClampLimit);
    } else {
        INVOKE_SWIGLU_BACKWARD_GROUP_QUANT_WITH_DUAL_AXIS(mode, hasGroupIndex, hasWeight, float, hasClampLimit);
    }
}
