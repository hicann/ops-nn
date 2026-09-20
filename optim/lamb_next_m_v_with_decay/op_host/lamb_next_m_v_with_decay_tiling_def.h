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
 * \file lamb_next_m_v_with_decay_tiling_def.h
 * \brief LambNextMVWithDecay tiling data definition (arch35 手写内核)
 *
 * 只用于给 framework 定 raw tiling buffer 的容量; 实际下发由 host 侧
 * memcpy 整个 LambBrcTilingData<13, 4> POD 完成, 故字段与 POD 严格一一对应。
 */

#ifndef LAMB_NEXT_M_V_WITH_DECAY_TILING_DEF_H
#define LAMB_NEXT_M_V_WITH_DECAY_TILING_DEF_H

#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"

namespace optiling {
constexpr int32_t LAMB_NEXT_M_V_WITH_DECAY_IN_NUM = 13;
constexpr int32_t LAMB_NEXT_M_V_WITH_DECAY_MAX_DIM = 8;
constexpr int32_t LAMB_NEXT_M_V_WITH_DECAY_STRIDE_NUM = LAMB_NEXT_M_V_WITH_DECAY_IN_NUM *
                                                        LAMB_NEXT_M_V_WITH_DECAY_MAX_DIM;

BEGIN_TILING_DATA_DEF(LambNextMVWithDecayTilingData)
TILING_DATA_FIELD_DEF(uint64_t, totalNum);
TILING_DATA_FIELD_DEF(uint64_t, totalRows);
TILING_DATA_FIELD_DEF_ARR(uint64_t, LAMB_NEXT_M_V_WITH_DECAY_STRIDE_NUM, effStride);
TILING_DATA_FIELD_DEF_ARR(uint64_t, LAMB_NEXT_M_V_WITH_DECAY_IN_NUM, srcBlockLen);
TILING_DATA_FIELD_DEF(uint64_t, perCoreElems);
TILING_DATA_FIELD_DEF(uint64_t, rowsPerCore);
TILING_DATA_FIELD_DEF(uint32_t, tileLen);
TILING_DATA_FIELD_DEF(uint32_t, blockLen);
TILING_DATA_FIELD_DEF(uint32_t, rowsPerTile);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, splitAxis);
TILING_DATA_FIELD_DEF(uint32_t, collapsedRank);
TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
TILING_DATA_FIELD_DEF_ARR(uint32_t, LAMB_NEXT_M_V_WITH_DECAY_IN_NUM, inKind);
TILING_DATA_FIELD_DEF_ARR(uint32_t, LAMB_NEXT_M_V_WITH_DECAY_MAX_DIM, outShape);
TILING_DATA_FIELD_DEF_ARR(uint32_t, LAMB_NEXT_M_V_WITH_DECAY_STRIDE_NUM, inShape);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LambNextMVWithDecay, LambNextMVWithDecayTilingData)
} // namespace optiling
#endif // LAMB_NEXT_M_V_WITH_DECAY_TILING_DEF_H
