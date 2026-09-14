/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adaptive_avg_pool3d_grad_ncdhw_small_kernel_tiling.h
 * \brief
 * ATTENTION: MAKE SURE 'BEGIN_TILING_DATA_DEF' STAY IN THE SAME LINE (27) USING BLANK LINES.
 */

#ifndef ADAPTIVE_AVG_POOL3D_GRAD_Small_kernel_TILING_H_
#define ADAPTIVE_AVG_POOL3D_GRAD_Small_kernel_TILING_H_
#include "adaptive_avg_pool3d_grad_ncdhw_tiling_common.h"

namespace optiling {

struct AdaptiveAvgPool3dGradNCDHWSmallKernelSplitInfo : public AdaptiveAvgPool3dGradNCDHWSplitCommon {
    // DoBufferCalculate (small kernel 专属 trans 队列 buffer)
    int64_t inputQueBufferSize{0};
    int64_t transOutQueBufferSize{0};
    int64_t transQueBufferSize{0};
};

class AdaptiveAvgPool3dGradTilingSmallKernel
    : public AdaptiveAvgPool3dGradNCDHWTilingCommon<AdaptiveAvgPool3dGradNCDHWSmallKernelSplitInfo> {
public:
    explicit AdaptiveAvgPool3dGradTilingSmallKernel(gert::TilingContext* context)
        : AdaptiveAvgPool3dGradNCDHWTilingCommon<AdaptiveAvgPool3dGradNCDHWSmallKernelSplitInfo>(context)
    {}

    ~AdaptiveAvgPool3dGradTilingSmallKernel() override {}

    AdaptiveAvgPool3dGradOp::AdaptiveAvgPool3dNCDHWGradSmallKernelTilingDataV35* tilingData = context_->GetTilingData<
        AdaptiveAvgPool3dGradOp::AdaptiveAvgPool3dNCDHWGradSmallKernelTilingDataV35>();

protected:
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;

    bool IsCapable() override;
    ge::graphStatus PostTiling() override;

    void InitializationVars();
    void DoBufferCalculate() override;
    bool TrySplitNC();
    void SearchBestTiling() override;
    bool ExhaustiveSearchBestTiling(int64_t computeVl, int64_t ncSearchMax, int64_t& bestHighAxisInner,
                                    int64_t& bestDOutputInner, int64_t& bestHOutputInner, int64_t& bestWOutputInner,
                                    int64_t& bestBlockNum, int64_t& bestUsedCoreNum, int64_t& bestHighAxisPadding,
                                    int64_t& bestHighAxisTail, int64_t& bestBufferSize, long double& bestCost,
                                    bool& found);
    long double EvalTilingCandidate(int64_t highAxisInner, int64_t highAxisOuter, int64_t highAxisTail,
                                    int64_t highAxisPadding, int64_t dOutputInner, int64_t dOutputOuter,
                                    int64_t hOutputInner, int64_t hOutputOuter, int64_t wOutputInner,
                                    int64_t wOutputOuter, int64_t blockNum, int64_t computeVl,
                                    int64_t normalCoreProcessNum);
    long double AddCostPenalties(long double cost, int64_t highAxisInner, int64_t highAxisOuter, int64_t highAxisTail,
                                 int64_t dOutputInner, int64_t hOutputInner, int64_t wOutputInner, int64_t blockNum,
                                 int64_t computeVl, int64_t normalCoreProcessNum, int64_t oneBlockWork);
    bool TryRecordBetterTiling(long double cost, int64_t dOutputInner, int64_t hOutputInner, int64_t wOutputInner,
                               int64_t blockNum, int64_t usedCoreNum, int64_t highAxisInner, int64_t highAxisPadding,
                               int64_t highAxisTail, int64_t& bestHighAxisInner, int64_t& bestDOutputInner,
                               int64_t& bestHOutputInner, int64_t& bestWOutputInner, int64_t& bestBlockNum,
                               int64_t& bestUsedCoreNum, int64_t& bestHighAxisPadding, int64_t& bestHighAxisTail,
                               int64_t& bestBufferSize, long double& bestCost, bool& found);
    void ApplyCoarseFallback();
    void SetTilingData();
    void PrintSplitData() const;
    void SplitUnalignDHW();
    void DynamicAdjustmentDWH();
};

} // namespace optiling

#endif // ADAPTIVE_AVG_POOL3D_GRAD_Small_kernel_TILING_H_
