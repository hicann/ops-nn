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
 *
 *
 *
 *
 *
 *
 * \file masked_softmax_with_rel_pos_bias_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MASKED_SOFTMAX_WITH_RELPOSBIAS_TILING_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MASKED_SOFTMAX_WITH_RELPOSBIAS_TILING_H_
#include <cstdint>
#include <vector>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

// tiling 要考虑字节对齐
BEGIN_TILING_DATA_DEF(MaskedSoftmaxWithRelPosBiasTilingData)
TILING_DATA_FIELD_DEF(uint32_t, tailStartCoreIdx);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreSize);
TILING_DATA_FIELD_DEF(uint32_t, w);
TILING_DATA_FIELD_DEF(uint32_t, n);
TILING_DATA_FIELD_DEF(uint32_t, s1);
TILING_DATA_FIELD_DEF(uint32_t, s2);
TILING_DATA_FIELD_DEF(uint32_t, wns1s2);
TILING_DATA_FIELD_DEF(uint32_t, wns1);
TILING_DATA_FIELD_DEF(uint32_t, ws1s2);
TILING_DATA_FIELD_DEF(uint32_t, ns1s2);
TILING_DATA_FIELD_DEF(uint32_t, ws1);
TILING_DATA_FIELD_DEF(uint32_t, ns1);
TILING_DATA_FIELD_DEF(uint32_t, s1s2);
TILING_DATA_FIELD_DEF(uint32_t, wns1s2Aligned);
TILING_DATA_FIELD_DEF(uint32_t, s1s2Aligned);
TILING_DATA_FIELD_DEF(uint32_t, ns1s2Aligned);
TILING_DATA_FIELD_DEF(uint32_t, s2Aligned);
TILING_DATA_FIELD_DEF(uint32_t, s2DtypeSize);
TILING_DATA_FIELD_DEF(uint32_t, inQueueLen);
TILING_DATA_FIELD_DEF(uint32_t, maskQueueLen);
TILING_DATA_FIELD_DEF(uint32_t, attenBiasNum);
TILING_DATA_FIELD_DEF(uint32_t, stackNum);
TILING_DATA_FIELD_DEF(uint32_t, loopNum);
TILING_DATA_FIELD_DEF(uint32_t, loopTailNum);
TILING_DATA_FIELD_DEF(uint32_t, xCopyEleNum);
TILING_DATA_FIELD_DEF(uint32_t, tailXCopyEleNum);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(uint32_t, castTempSize);
TILING_DATA_FIELD_DEF(uint32_t, tmpXubSize);
TILING_DATA_FIELD_DEF(uint32_t, tmpMaskUbSize);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, tailSoftmaxTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaskedSoftmaxWithRelPosBias, MaskedSoftmaxWithRelPosBiasTilingData)
} // optiling
#endif