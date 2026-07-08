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
 * \file adaptive_sliding_window_mix_basic_api_tiling.h
 * \brief Mix BasicAPI tiling base class (shared by batch and non-batch).
 */
#pragma once

#include "adaptive_sliding_window_tiling.h"
#include "matmul/quant_batch_matmul_v3/op_kernel/arch35/quant_batch_matmul_v3_tiling_data.h"

namespace optiling {

class AdaptiveSlidingWindowMixBasicAPITiling : public AdaptiveSlidingWindowTiling {
public:
    explicit AdaptiveSlidingWindowMixBasicAPITiling(gert::TilingContext* context);
    AdaptiveSlidingWindowMixBasicAPITiling(
        gert::TilingContext* context, DequantBmm::QuantBatchMatmulV3BasicAPITilingData* out);
    ~AdaptiveSlidingWindowMixBasicAPITiling() override = default;

    ge::graphStatus DoLibApiTiling() override;

private:
    void Reset();
    void CalculateNBufferNum4Mix();
    void UpdateAFullLoadStatus();
    bool IsWithoutBatchTilingData() const;
    void SetWithoutBatchTilingData();
    struct BaseBlockOptimizeInfo {
        uint64_t baseMAlign = 0;
        uint64_t baseNAlign = 0;
        uint64_t baseKAlign = 0;
        uint64_t mCore = 0;
        uint64_t nCore = 0;
        uint64_t blockCnt = 0;
        uint64_t coreNumMN = 1;
        uint64_t targetBlockCnt = 1;
        bool hasSmallMTail = false;
        bool hasSmallNTail = false;
    };

    bool InitBaseBlockOptimizeInfo(BaseBlockOptimizeInfo &info);
    uint64_t CalcBaseBlockScore(uint64_t baseM, uint64_t baseN, const BaseBlockOptimizeInfo &info);
    void TryUpdateBaseBlockCandidate(uint64_t candMCore, uint64_t candNCore,
                                     const BaseBlockOptimizeInfo &info,
                                     uint64_t &bestBaseM, uint64_t &bestBaseN,
                                     uint64_t &bestBaseK, uint64_t &bestScore);

protected:
    bool IsCapable() override;
    bool CheckCoreNum() const override;
    ge::graphStatus GetWorkspaceSize() override;
    uint64_t GetBatchCoreCnt() const override;
    uint64_t GetApiLevel(NpuArch npuArch) const override;
    const void* GetTilingData() const override;
    uint64_t GetBaseMAlignSize() const;
    uint64_t GetBaseNAlignSize() const;
    void OptimizeBaseBlock();
    bool CalcBasicBlock() override;
    void AnalyseFullLoadInfo() override;
    void CalcTailRoundBasicBlockSplit() override;
    bool CalL1Tiling() override;
    void SetTilingData() override;

    DequantBmm::QuantBatchMatmulV3BasicAPITilingData tilingDataSelf_;
    DequantBmm::QuantBatchMatmulV3BasicAPITilingData& tilingData_;
    DequantBmm::QuantBatchMatmulV3TensorAPIWithoutBatchTilingData withoutBatchTilingData_;
    bool useWithoutBatchTilingData_ = false;
};
} // namespace optiling
