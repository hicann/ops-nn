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
 * \file quantize_tiling.h
 * \brief Quantize host tiling data + compile info, ascend910b (DAV_2201) standard model.
 */

#ifndef QUANTIZE_TILING_H
#define QUANTIZE_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

struct QuantizeCompileInfo {
    int64_t coreNum = 0;
    uint64_t ubSize = 0;
};

BEGIN_TILING_DATA_DEF(QuantizeTilingData)
TILING_DATA_FIELD_DEF(uint32_t, numCore);        // AI cores actually launched (== block dim)
TILING_DATA_FIELD_DEF(uint32_t, hasZeroPoint);   // 1 if zero_points present, else 0
TILING_DATA_FIELD_DEF(int64_t, channelNum);      // per-channel: x.shape[axis]; per-tensor: 1
TILING_DATA_FIELD_DEF(int64_t, rowLen);          // per-channel: prod(dims after axis); per-tensor: 1
TILING_DATA_FIELD_DEF(int64_t, totalRows);       // per-channel: prod(dims up to and incl. axis); per-tensor: 1
TILING_DATA_FIELD_DEF(int64_t, totalElems);      // total element count (both modes)
TILING_DATA_FIELD_DEF(int64_t, blockFactor);     // per-channel: rows/core; per-tensor: elems/core
TILING_DATA_FIELD_DEF(int64_t, blockTailFactor); // factor of the last active core
TILING_DATA_FIELD_DEF(int64_t, baseLen);         // UB tile length in elements
TILING_DATA_FIELD_DEF(uint32_t, zpDtype);        // runtime zero_points ge::DataType code (0 when absent)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Quantize, QuantizeTilingData)

} // namespace optiling

#endif // QUANTIZE_TILING_H
