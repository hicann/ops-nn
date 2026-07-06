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
 * \file ascend_anti_quant_tilingdata.h
 * \brief TilingData shared by host and kernel (pure regbase paradigm).
 *
 *   Layout:
 *     baseTiling : 1D elewise base tiling block (self-defined, field layout
 *                  matches the historical Ops::Base::EleBaseTilingData so the
 *                  kernel reads stay binary-compatible)
 *     scale      : numeric attr (raw value; kernel applies sqrt-mode squaring)
 *     offset     : numeric attr
 *     sqrtMode   : runtime branch flag, 0 or 1 (NOT part of tiling key)
 *     reserved   : padding for 16B alignment
 */
#ifndef _ASCEND_ANTI_QUANT_TILINGDATA_
#define _ASCEND_ANTI_QUANT_TILINGDATA_

#include <cstdint>

struct AscendAntiQuantBaseTilingData {
    int64_t dim0;
    int32_t coreNum;
    int32_t ubFormer;
    int64_t blockFormer;
    int64_t blockNum;
    int64_t ubLoopOfFormerBlock;
    int64_t ubLoopOfTailBlock;
    int64_t ubTailOfFormerBlock;
    int64_t ubTailOfTailBlock;
    int64_t elemNum;
    uint64_t scheMode;
};

struct AscendAntiQuantTilingData {
    AscendAntiQuantBaseTilingData baseTiling;
    float scale;
    float offset;
    int32_t sqrtMode;
    int32_t reserved;
};
#endif // _ASCEND_ANTI_QUANT_TILINGDATA_
