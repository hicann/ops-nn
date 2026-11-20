/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file transpose_batch_mat_mul_asw_tiling.h
 * \brief
 */

#pragma once

#include "matmul/batch_mat_mul_v3/op_host/op_tiling/arch35/batch_matmul_v3_asw_basic_tiling.h"
#include "transpose_batch_mat_mul_tiling_key.h"
#include "matmul/transpose_batch_mat_mul/op_kernel/arch35/transpose_batch_mat_mul_tiling_data.h"

namespace optiling {
namespace transpose_batch_mat_mul_advanced {
using namespace matmul_v3_advanced;
using batch_matmul_v3_advanced::BatchMatMulV3AswBasicTiling;
class TransposeBatchMatMulAswTiling : public BatchMatMulV3AswBasicTiling {
public:
    TransposeBatchMatMulAswTiling(gert::TilingContext* context, MatMulTilingCfg& cfg)
        : BatchMatMulV3AswBasicTiling(context, cfg) {};

    ~TransposeBatchMatMulAswTiling() override = default;

protected:
    bool IsCapable() override { return true; };

    ge::graphStatus DoOpTiling() override;

    uint64_t GetTilingKey() const override;

    uint64_t GetNumBlocks() const override;

    ge::graphStatus GetTilingData(TilingResult& tiling) const override;

private:
    using BatchMatMulV3AswBasicTiling::GetTilingDataProcess;
    ge::graphStatus GetTilingDataProcess(TransposeBatchMatMulBasicTilingData& tilingData) const;

    void GetTransposeBatchMatMulInfo();
    void CheckModel();

    TBMMPermX1 permX1_ = TBMMPermX1::PERM_X1_0_1_2;
    TBMMPermX2 permX2_ = TBMMPermX2::PERM_X2_0_1_2;
    TBMMBatchSplit batchSplitMode_ = TBMMBatchSplit::BATCH_SPLIT_FALSE;
    TBMMModel model_{TBMMModel::TBMM_BASIC};
    uint32_t batchSplitFactor_ = 1;
    TBMMNonContiguousType nonContiguousType_ = TBMMNonContiguousType::CONTIGUOUS;
};
} // namespace transpose_batch_mat_mul_advanced
} // namespace optiling
