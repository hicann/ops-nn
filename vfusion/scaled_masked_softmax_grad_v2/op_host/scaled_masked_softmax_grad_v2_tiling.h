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
 * \file scaled_masked_softmax_grad_v2_tiling.h
 * \brief
 */
#ifndef __ASCENDC_SCALED_MASKED_SOFTMAX_GRAD_V2_TILING_H__
#define __ASCENDC_SCALED_MASKED_SOFTMAX_GRAD_V2_TILING_H__

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ScaledMaskedSoftmaxGradV2TilingData)
TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum);

TILING_DATA_FIELD_DEF(uint64_t, batch);
TILING_DATA_FIELD_DEF(uint64_t, channel);
TILING_DATA_FIELD_DEF(uint64_t, seqLength);
TILING_DATA_FIELD_DEF(uint64_t, headDim);

TILING_DATA_FIELD_DEF(uint64_t, totalLine);
TILING_DATA_FIELD_DEF(uint64_t, paddedHeadDim);        // 64个数对齐后的D
TILING_DATA_FIELD_DEF(uint64_t, totalLinePerHeadCore); // 前核处理的总行数
TILING_DATA_FIELD_DEF(uint64_t, totalLinePerTailCore); // 尾核处理的总行数
TILING_DATA_FIELD_DEF(uint64_t, maxLinePerLoop);       // 每次循环能处理的最大行数
TILING_DATA_FIELD_DEF(uint64_t, tailLinePerHeadCore);  // 前核的尾循环的行数
TILING_DATA_FIELD_DEF(uint64_t, tailLinePerTailCore);  // 尾核的尾循环的行数
TILING_DATA_FIELD_DEF(uint64_t, headCoreNum);          // 前核的数量
TILING_DATA_FIELD_DEF(uint64_t, maskMoveMode);         // mask的搬运模式
TILING_DATA_FIELD_DEF(uint64_t, selectSize);           // 用于设置select接口的tmpBuf大小

TILING_DATA_FIELD_DEF(float, scale);
TILING_DATA_FIELD_DEF(uint32_t, fixedTriuMask);

TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxGradTilingData); // softmax高阶api所需tiling

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScaledMaskedSoftmaxGradV2, ScaledMaskedSoftmaxGradV2TilingData)

struct ScaledMaskedSoftmaxGradV2CompileInfo {
};

} // namespace optiling

#endif // __ASCENDC_SCALED_MASKED_SOFTMAX_GRAD_V2_TILING_H__