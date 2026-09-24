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
 * \file adaptive_sliding_window_cube_basic_api_tiling.h
 * \brief
 */
#pragma once

#include <cstdint>

#include "adaptive_sliding_window_tiling.h"
#include "matmul/quant_batch_matmul_v3/op_kernel/arch35/quant_batch_matmul_v3_tiling_data.h"

namespace optiling {

class AdaptiveSlidingWindowCubeBasicAPITiling : public AdaptiveSlidingWindowTiling {
public:
    explicit AdaptiveSlidingWindowCubeBasicAPITiling(gert::TilingContext* context);
    AdaptiveSlidingWindowCubeBasicAPITiling(gert::TilingContext* context,
                                            DequantBmm::QuantBatchMatmulV3BasicAPITilingData* out);
    ~AdaptiveSlidingWindowCubeBasicAPITiling() override = default;

    ge::graphStatus DoLibApiTiling() override;

private:
    // Fixed geometry for one L1 plan search; the candidate step varies as a scalar kL1 alongside it.
    struct CubeL1Context {
        uint64_t baseM;
        uint64_t baseN;
        uint64_t baseK;
        bool isAFullLoad;
    };

    struct CubeL1Plan {
        uint64_t kAL1;
        uint64_t kBL1;
        uint32_t l1BufferNum;
    };

    void CalculateNBufferNum();
    void CalculateLegacyNBufferNum();
    CubeL1Plan SelectCubeL1Plan(const CubeL1Context& ctx, uint64_t currentKAL1, uint64_t currentKBL1) const;
    bool TryApplyReducedStepK(CubeL1Plan& plan, const CubeL1Context& ctx, uint64_t currentKL1,
                              uint32_t l1BufferNum) const;
    uint64_t SelectReducedStepK(const CubeL1Context& ctx, uint64_t currentKL1, uint32_t l1BufferNum) const;
    bool CanFitL1BufferNum(const CubeL1Context& ctx, uint64_t kL1, uint32_t l1BufferNum) const;
    uint64_t CalcUsedL1Size(const CubeL1Context& ctx, uint64_t kL1, uint32_t l1BufferNum) const;
    uint64_t CalcAFullLoadL1Size(uint64_t baseM) const;
    uint64_t CalcAFullKLoadSize(uint64_t mSize) const;
    uint64_t CalcBFullKLoadSize(uint64_t nSize) const;
    bool ShouldKeepAFullLoadByRepeatLoadRatio() const;
    bool IsMte2Bound(double gmBandwidthTbps, double l2BandwidthTbps) const;
    double EstimateMte2TimeUs(double gmBandwidthTbps, double l2BandwidthTbps) const;
    double EstimateMacTimeUs() const;
    bool CanSelectMultiBufferByL1Plan(bool isAFullLoad, uint64_t baseM, uint64_t baseN) const;
    bool IsWithoutBatchTilingData() const;
    void SetWithoutBatchTilingData();
    void UpdateAFullLoadStatus();
    void UpdateBFullLoadStatus();

protected:
    void ResetTilingData() override;
    bool IsCapable() override;
    ge::graphStatus GetWorkspaceSize() override;
    uint64_t GetBatchCoreCnt() const override;
    const void* GetTilingData() const override;
    uint64_t GetKernelType() const override;
    uint64_t GetApiLevel(NpuArch npuArch) const override;
    uint64_t GetBatchMode() const override;
    bool CalcBasicBlock() override;
    void AnalyseFullLoadInfo() override;
    bool CalL1Tiling() override;
    void SetTilingData() override;

    DequantBmm::QuantBatchMatmulV3BasicAPITilingData tilingDataSelf_;
    DequantBmm::QuantBatchMatmulV3BasicAPITilingData& tilingData_;
    DequantBmm::QuantBatchMatmulV3TensorAPIWithoutBatchTilingData withoutBatchTilingData_;
    bool useWithoutBatchTilingData_ = false;
    bool isSupportS4S4_ = false;
};
} // namespace optiling
