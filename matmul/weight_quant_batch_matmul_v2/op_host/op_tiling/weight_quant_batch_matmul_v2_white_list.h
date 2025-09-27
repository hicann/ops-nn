/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file weight_quant_batch_matmul_v2_white_list.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_WHITE_LIST_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_WHITE_LIST_H

#include "matmul/common/op_host/math_util.h"
#include "op_cache_def_tiling.h"

namespace optiling {

class WhiteListShape
{
public:
    bool operator<(const WhiteListShape& right) const
    {
        return memcmp(this, &right, sizeof(WhiteListShape)) < 0;
    }

    uint64_t mSize_;
    uint64_t kSize_;
    uint64_t nSize_;
    bool hasBias_;
    bool transA_;
    bool transB_;
    uint64_t aicNum_ : 40;
};

inline void SetMatmulTilingFromCacheData(
    WeightQuantBatchMatmulCacheTilingData& cacheTilingData, optiling::TCubeTiling& matmulTiling, uint64_t m, uint64_t n,
    int32_t isBias)
{
    matmulTiling.set_M(m);
    matmulTiling.set_N(n);
    matmulTiling.set_Ka(cacheTilingData.ka_);
    matmulTiling.set_Kb(cacheTilingData.kb_);
    matmulTiling.set_singleCoreM(ops::CeilDiv(m, static_cast<uint64_t>(cacheTilingData.mDim_)));
    matmulTiling.set_singleCoreN(cacheTilingData.singleCoreN_);
    matmulTiling.set_singleCoreK(cacheTilingData.singleCoreK_);
    matmulTiling.set_baseM(cacheTilingData.baseM_);
    matmulTiling.set_baseN(cacheTilingData.baseN_);
    matmulTiling.set_baseK(cacheTilingData.baseK_);
    matmulTiling.set_depthA1(cacheTilingData.depthA1_);
    matmulTiling.set_depthB1(cacheTilingData.depthB1_);
    matmulTiling.set_stepM(cacheTilingData.stepM_);
    matmulTiling.set_stepN(cacheTilingData.stepN_);
    matmulTiling.set_stepKa(cacheTilingData.stepKa_);
    matmulTiling.set_stepKb(cacheTilingData.stepKb_);
    matmulTiling.set_isBias(isBias);
    matmulTiling.set_transLength(cacheTilingData.transLength_);
    matmulTiling.set_iterateOrder(cacheTilingData.iterateOrder_);
    matmulTiling.set_shareL1Size(cacheTilingData.shareL1Size_);
    matmulTiling.set_shareL0CSize(cacheTilingData.shareL0CSize_);
    matmulTiling.set_dbL0A(cacheTilingData.dbL0A_);
    matmulTiling.set_dbL0B(cacheTilingData.dbL0B_);
    matmulTiling.set_dbL0C(cacheTilingData.dbL0C_);
    matmulTiling.set_usedCoreNum(1);
    matmulTiling.set_batchM(1);
    matmulTiling.set_batchN(1);
    matmulTiling.set_singleBatchM(1);
    matmulTiling.set_singleBatchN(1);
}

} // namespace optiling
#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_WHITE_LIST_H
