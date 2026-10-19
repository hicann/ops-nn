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
 * \file index_put_v2_simd_dma_tiling.cpp
 * \brief DMA tiling implementation for IndexPutV2SimdDma.
 */

#include <algorithm>
#include <cmath>
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "log/log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "index_put_v2_simd_dma_tiling.h"
#include "../../op_kernel/arch35/index_put_v2_tiling_key.h"

using namespace AscendC;

namespace optiling {
using namespace IndexPutV2;

constexpr size_t MAX_DIM = 8;
constexpr uint32_t TWO = 2;
static constexpr int64_t COLS_ALIGN_BYTES = 128;
constexpr uint64_t ubBlockAlign = 32;
constexpr int64_t SCATTER_OFFSET_NUM_MAX = 4095;

bool IndexPutV2SimdDmaTiling::IsCapable()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr ||
        platform_ascendc::PlatformAscendC(platformInfo).GetCurNpuArch() != NpuArch::DAV_9201) {
        return false;
    }
    return IndexPutV2SimdTiling::IsCapable();
}

void IndexPutV2SimdDmaTiling::DoUBTiling()
{
    uint64_t availableUbsize = ubSize_ - MAX_DIM * sizeof(int64_t) * TWO;

    // Reserve UB for DMA offset buffers:
    //   gatherOffsetBuf_: CeilAlign(indexedDimNum_ * sizeof(int64_t), 32) bytes (fixed)
    //   scatterOffsetBuf_: accounted per-row in oneRowBuffer below
    //   maskBuf_: CeilAlign(rowsFactor * sizeof(uint8_t), 32) bytes, accounted per-row
    uint64_t gatherOffsetBufSize = Ops::Base::CeilAlign(indexedDimNum_ * sizeof(int64_t), ubBlockAlign);
    availableUbsize -= gatherOffsetBufSize;

    int64_t minRows = 8;
    int64_t doubleB = 2;
    int32_t rankDim = 0;
    uint64_t oneRowBuffer = 0;
    uint64_t ubBlockSize = static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_));

    for (int64_t i = 0; i < indexedSizesNum_; i++) {
        if (indexedSizes_[i] == 1) {
            rankDim++;
        }
    }

    minRows = std::min(minRows, normalCoreRowsNum_);
    // Additional per-row cost: sizeof(int64_t) for scatter offset + sizeof(uint8_t) for compare mask
    uint64_t tmpBuf = minRows * Ops::Base::CeilAlign(normalCoreColsNum_ * valueTypeSize, ubBlockSize) * doubleB +
                      Ops::Base::CeilAlign(minRows * indicesTypeSize, ubBlockSize) * (rankDim * doubleB + 1) +
                      Ops::Base::CeilAlign(minRows * sizeof(int64_t), ubBlockSize) +
                      Ops::Base::CeilAlign(minRows * sizeof(uint8_t), ubBlockSize);
    if (tmpBuf < availableUbsize) {
        colsFactor_ = normalCoreColsNum_;
        oneRowBuffer = Ops::Base::CeilAlign(normalCoreColsNum_ * valueTypeSize, ubBlockSize) * doubleB +
                       Ops::Base::CeilAlign(indicesTypeSize, ubBlockSize) * (rankDim * doubleB + 1) +
                       Ops::Base::CeilAlign(sizeof(int64_t), ubBlockSize) +
                       Ops::Base::CeilAlign(sizeof(uint8_t), ubBlockSize);
        rowsFactor_ = availableUbsize / oneRowBuffer;
        rowsFactor_ = std::min(rowsFactor_, normalCoreRowsNum_);
    } else {
        rowsFactor_ = minRows;
        colsFactor_ = (availableUbsize -
                       Ops::Base::CeilAlign(rowsFactor_ * indicesTypeSize, ubBlockSize) * (rankDim * doubleB + 1) -
                       Ops::Base::CeilAlign(rowsFactor_ * sizeof(int64_t), ubBlockSize) -
                       Ops::Base::CeilAlign(rowsFactor_ * sizeof(uint8_t), ubBlockSize)) /
                      rowsFactor_ / doubleB / valueTypeSize;
        colsFactor_ = Ops::Base::FloorAlign(colsFactor_, COLS_ALIGN_BYTES / static_cast<int64_t>(valueTypeSize));
        colsFactor_ = std::min(colsFactor_, normalCoreColsNum_);
    }

    rowsFactor_ = std::min(rowsFactor_, SCATTER_OFFSET_NUM_MAX);
}

uint64_t IndexPutV2SimdDmaTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(0, 0, 1, 1, 0, static_cast<uint64_t>(accumulateMode_), 0);
}

REGISTER_OPS_TILING_TEMPLATE(IndexPutV2, IndexPutV2SimdDmaTiling, 8);
} // namespace optiling
