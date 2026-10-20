/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file index_fill_tiling_simd_dma.h
 * \brief
 */

#ifndef INDEX_FILL_TILING_SIMD_DMA_H_
#define INDEX_FILL_TILING_SIMD_DMA_H_

#include "index_fill_tiling_common.h"

namespace optiling {
class IndexFillSimdDmaTiling : public IndexFillCommonTiling {
public:
    explicit IndexFillSimdDmaTiling(gert::TilingContext* context) : IndexFillCommonTiling(context) {}
    ~IndexFillSimdDmaTiling() override {}

private:
    uint64_t blockFactorPN_;
    uint64_t tailBlockNumPN_;
    uint64_t usedCoreNumPN_;
    uint64_t blockFactorUbBufferMask_;
    uint64_t indicesUbFactorSizeAlign_;
    uint64_t indicesUbFactor_;
    uint64_t normBlockQNumLoop_;
    uint64_t normBlockQNumTail_;
    uint64_t tailBlockQNumLoop_;
    uint64_t tailBlockQNumTail_;
    uint64_t normBlockQNum_;
    uint64_t tailBlockQNum_;
    uint64_t blockFactorAlignQ_;
    bool isUseGatherDma_ = false;

    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    void SetTilingData() override;
    uint64_t GetTilingKey() const override;
    void DumpTilingInfo() override;

private:
    void CalcUsedCoreNum();
    void CalcUBBlock();
};

} // namespace optiling
#endif // INDEX_FILL_TILING_SIMD_DMA_H_
