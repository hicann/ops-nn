/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adaptive_sliding_window_mx_basic_api_tiling.h
 * \brief
 */
#pragma once

#include <cstdint>

#include "exe_graph/runtime/tiling_context.h"
#include "graph/types.h"
#include "platform/soc_spec.h"

#include "adaptive_sliding_window_tiling.h"
#include "matmul/quant_batch_matmul_v3/op_kernel/arch35/quant_batch_matmul_v3_tiling_data.h"

namespace optiling {

class AdaptiveSlidingWindowMXBasicAPITiling : public AdaptiveSlidingWindowTiling {
public:
    explicit AdaptiveSlidingWindowMXBasicAPITiling(gert::TilingContext* context);
    AdaptiveSlidingWindowMXBasicAPITiling(gert::TilingContext* context,
                                          DequantBmm::QuantBatchMatmulV3BasicAPITilingData* out);
    ~AdaptiveSlidingWindowMXBasicAPITiling() override = default;

    ge::graphStatus DoLibApiTiling() override;

private:
    struct MxL1EstimateParams {
        uint64_t kL1;
        uint64_t scaleKL1;
        uint64_t baseM;
        uint64_t baseN;
        bool isAFullLoad;
    };

    struct MxL1Plan {
        uint64_t kL1;
        uint64_t scaleKL1;
        uint32_t l1BufferNum;
    };

    void CalculateNBufferNum();
    uint64_t DeriveScaleKL1(uint32_t& scaleFactorA, uint64_t stepKa, uint32_t& scaleFactorB, uint64_t stepKb,
                            uint64_t baseK) const;
    MxL1EstimateParams BuildL1EstimateParams(uint64_t kL1, uint64_t baseScaleKL1, uint64_t baseM, uint64_t baseN,
                                             bool isAFullLoad) const;
    MxL1Plan SelectMxL1Plan(const MxL1EstimateParams& currentParams, uint64_t baseK, uint64_t baseScaleKL1) const;
    MxL1Plan MakeMxL1Plan(const MxL1EstimateParams& params, uint32_t l1BufferNum) const;
    uint64_t GetHalfKFallbackScaleKL1(uint64_t scaleKL1, uint64_t kL1) const;
    uint64_t GetFullCoverScaleKL1IfPossible(const MxL1EstimateParams& params, uint32_t l1BufferNum) const;
    bool CanReduceStepKToTwo(uint64_t stepKTwoKL1) const;
    bool CanFitL1BufferNum(const MxL1EstimateParams& params, uint32_t l1BufferNum) const;
    uint64_t CalcUsedL1Size(const MxL1EstimateParams& params, uint32_t l1BufferNum) const;
    uint64_t CalcMxFullKLoadSize(uint64_t outerSize, ge::DataType dataDtype, ge::DataType scaleDtype) const;
    bool ShouldKeepAFullLoadByRepeatLoadRatio() const;
    bool IsMte2Bound(double gmBandwidthTbps, double l2BandwidthTbps) const;
    double EstimateMte2TimeUs(double gmBandwidthTbps, double l2BandwidthTbps) const;
    double EstimateMacTimeUs() const;
    void UpdateAFullLoadStatus();
    bool CanSelectMultiBufferByL1Plan(bool isAFullLoad, uint64_t baseM, uint64_t baseN) const;
    bool IsWithoutBatchTilingData() const;
    void SetWithoutBatchTilingData();
    void NormalizeSingleRoundTailSplitBasicBlock();
    void AdjustScaleFactorForL0CPingpong(uint32_t& scaleFactor, uint32_t step, uint32_t baseK) const;

protected:
    void ResetTilingData() override;
    bool IsCapable() override;
    uint64_t GetBatchCoreCnt() const override;
    const void* GetTilingData() const override;
    uint64_t GetApiLevel(NpuArch npuArch) const override;
    uint64_t GetBatchMode() const override;
    uint64_t GetKernelType() const override;
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
