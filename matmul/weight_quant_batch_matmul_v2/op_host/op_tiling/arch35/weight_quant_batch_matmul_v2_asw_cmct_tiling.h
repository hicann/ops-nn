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
 * \file weight_quant_batch_matmul_v2_asw_cmct_tiling.h
 * \brief
 */

#pragma once

#include "../weight_quant_batch_matmul_v2_tiling.h"
#include "matmul/weight_quant_batch_matmul_v2/op_kernel/arch35/weight_quant_batch_matmul_v2_arch35_tiling_data.h"

namespace optiling {
namespace weight_quant_batch_matmul_v2 {

struct AswBasicRunInfo {
    uint64_t usedCoreNum = 1;
    uint64_t baseM = 1;
    uint64_t baseN = 1;
    uint64_t baseK = 1;
    uint64_t stepM = 1;
    uint64_t stepN = 1;
    uint64_t stepKa = 1;
    uint64_t stepKb = 1;
    uint64_t dbL0c = 1;
    uint64_t ubDb = 1;
    uint64_t l1BufferNum = 2;
    uint64_t mBlockCnt = 1; // m方向基本块数量
    uint64_t nBlockCnt = 1; // n方向基本块数量
    uint64_t mTailCnt = 1;  // 尾轮基本块m方向重切粒度
    uint64_t nTailCnt = 1;  // 尾轮基本块n方向重切粒度
    double cubeBoundParam = 0.0;
    double cubeBoundEdge = 0.0;
};

class WeightQuantBatchMatmulV2TilingAswCmct : public WeightQuantBatchMatmulV2Tiling {
public:
    explicit WeightQuantBatchMatmulV2TilingAswCmct(gert::TilingContext* context)
        : WeightQuantBatchMatmulV2Tiling(context)
    {
        if (context->GetCompileInfo() == nullptr) {
            InitCompileInfo();
        }
    }
    ~WeightQuantBatchMatmulV2TilingAswCmct() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    std::unique_ptr<wqbmmv2_tiling::WqbmmV2AswTilingData> tilingData_;
    size_t tilingDataSize_ = sizeof(wqbmmv2_tiling::WqbmmV2AswTilingData);
    AswBasicRunInfo runInfo_;

    ge::graphStatus InstantiateTilingData();
    void ResetBaseTiling();
    ge::graphStatus CalRebalanceBlock();
    uint64_t GetMaxBaseWithLimit(uint64_t baseMNBufferLimit, uint64_t baseAlignUnit, bool isRightMatrix,
                                 bool isMemoryBound) const;
    double GetBalanceRateWithTail(uint64_t baseM, uint64_t baseN) const;
    void CalBaseK();
    void CalTailBasicBlock();
    bool IsValidWeightNzTailSplit(uint64_t splitCnt) const;
    void CalL1Tiling();
    void CalL1BufferNum();
    bool CheckAntiQuantScale(uint64_t baseN, uint64_t dbL0c = 1) const;
    ge::graphStatus SetTilingData();
    wqbmmv2_tiling::L2CacheMode SetDisableL2cache(uint32_t mL1, uint32_t kaL1, uint32_t kbL1, uint32_t nL1) const;
    uint64_t GetShapeWithDataType(uint64_t shapeSize, ge::DataType dtype) const;
    uint64_t GetSizeWithDataType(uint64_t shapeSize, ge::DataType dtype) const;

private:
    OptimizationAlgorithmSubCategory algorithmSubCategory_ = OptimizationAlgorithmSubCategory::ASW_CMCT;
    Mte2Configuration mte2Config_ = Mte2Configuration::MTE2_INNER_SIZE_512_BUF_NUM_2;
};
} // namespace weight_quant_batch_matmul_v2
} // namespace optiling
