/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file avg_pool3_d_grad_ncdhw_tiling.h
 * \brief NCDHW scheme tiling for 3D average pooling backward (arch35).
 *        Logic modeled on avg_pool_v2_grad_nchw_tiling.cpp, extended to D/H/W.
 */

#ifndef OP_IMPL_AVG_POOL3_D_GRAD_NCDHW_TILING_H_
#define OP_IMPL_AVG_POOL3_D_GRAD_NCDHW_TILING_H_

#include "avg_pool3_d_grad_tiling_base.h"
#include "avg_pool3_d_grad_tiling_common.h"
#include "../../op_kernel/arch35/avg_pool3_d_grad_tiling_data.h"
#include "../../op_kernel/arch35/avg_pool3_d_grad_tiling_key.h"

namespace optiling {

struct AvgPool3DGradNCDHWBaseInfo {
    int64_t vRegSize{0};
    int64_t ubBlockSize{0};
    int64_t inputBytes{0};
    int64_t availableUb{0};
    int64_t totalCoreNum{0};
    int64_t coreUsedForBestPerformance{0};
    int64_t inputNCSize{0};
    int64_t dataNumInOneBlock{0};
    int64_t proDataNumInOneBeat{0};
    int64_t dProBatchSize{1};
    int64_t hProBatchSize{1};
    int64_t wProBatchSize{1};
    int64_t isPad{0};
    int64_t isOverlap{0};
};

struct AvgPool3DGradNCDHWSplitInfo {
    int64_t isCheckRange{0};
    int64_t isStrideAligned{0};
    int64_t highAxisInner{1};
    int64_t highAxisTail{1};
    int64_t highAxisOuter{1};
    int64_t dOutputInner{1};
    int64_t dOutputTail{1};
    int64_t dOutputOuter{1};
    int64_t hOutputInner{1};
    int64_t hOutputTail{1};
    int64_t hOutputOuter{1};
    int64_t wOutputInner{1};
    int64_t wOutputTail{1};
    int64_t wOutputOuter{1};
    int64_t normalCoreProcessNum{1};
    int64_t tailCoreProcessNum{1};
    int64_t usedCoreNum{1};
    int64_t totalBaseBlockNum{1};
    int64_t outputBufferSize{0};
    int64_t gradBufferSize{0};
    int64_t totalBufferSize{0};
    int64_t dInputInner{0};
    int64_t hInputInner{0};
    int64_t wInputInner{0};
};

class AvgPool3DGradNCDHWTiling : public AvgPool3DGradTilingBase {
public:
    explicit AvgPool3DGradNCDHWTiling(gert::TilingContext* context) : AvgPool3DGradTilingBase(context) {}
    ~AvgPool3DGradNCDHWTiling() override {}

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetShapeAttrsInfo() override;

private:
    void InitializationVars();
    bool IsMeetUBSize();
    bool IsMeetTargetCoreNum() const;
    void DoBufferCalculate();
    bool TrySplitNC();
    bool TrySplitAlignD();
    bool TrySplitAlignH();
    bool TrySplitAlignW();
    void SplitUnalignDHW();
    void DynamicAdjustmentDHW();
    void SearchBestTiling();
    void DoUBTiling();
    void DoBlockTiling();
    void SearchOuterSingle(int64_t& inner, int64_t step);
    void AdjustInnerSplitForMultiCore();
    ge::graphStatus SetTilingData();
    void PrintBaseData() const;
    void PrintSplitData() const;

    AvgPool3DGradNCDHWBaseInfo baseData;
    AvgPool3DGradNCDHWSplitInfo splitData;
};

} // namespace optiling

#endif // OP_IMPL_AVG_POOL3_D_GRAD_NCDHW_TILING_H_
