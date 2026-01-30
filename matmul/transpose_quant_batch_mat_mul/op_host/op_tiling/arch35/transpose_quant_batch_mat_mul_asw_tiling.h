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
 * \file transpose_quant_batch_mat_mul_tiling.h
 * \brief
 */

#ifndef __TRANSPOSE_QUANT_BATCH_MAT_MUL_ASW_TILING_H__
#define __TRANSPOSE_QUANT_BATCH_MAT_MUL_ASW_TILING_H__
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_base_tiling_advanced.h"
#include "../../../op_kernel/arch35/transpose_quant_batch_mat_mul_tiling_key_public.h"

namespace optiling {
namespace transpose_quant_batch_mat_mul_advanced {
class TransposeQuantBatchMatMulAswTiling : public MatMulV3BaseTiling {
public:
    TransposeQuantBatchMatMulAswTiling(gert::TilingContext* context, MatMulTilingCfg& cfg) :
        MatMulV3BaseTiling(context, cfg) {};
    ~TransposeQuantBatchMatMulAswTiling() override = default;

protected:
    bool IsCapable() override
    {
        return true;
    };

    ge::graphStatus DoOpTiling() override;

    uint64_t GetTilingKey() const override;

    uint64_t GetNumBlocks() const override;

    ge::graphStatus GetTilingData(TilingResult& tiling) const override;

    ge::graphStatus GetTilingDataProcess(BatchMatMulV3TilingData& tilingData) const override;

    std::vector<size_t> GetWorkspaceSize() const override;
    void AdjustBasicBlock(); 

private:
    TQBMMPermX1 permX1_ = TQBMMPermX1::PERM_X1_1_0_2;
    TQBMMPermX2 permX2_ = TQBMMPermX2::PERM_X2_0_1_2;
    TQBMMBatchSplit batchSplitMode_ = TQBMMBatchSplit::BATCH_SPLIT_FALSE;
    uint32_t batchSplitFactor_ = 1;
};
} // namespace transpose_quant_batch_mat_mul_advanced
} // namespace optiling
#endif // TRANSPOSE_QUANT_BATCH_MAT_MUL_ASW_TILING_H