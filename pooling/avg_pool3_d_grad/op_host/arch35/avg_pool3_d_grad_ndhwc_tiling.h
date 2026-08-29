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
 * \file avg_pool3_d_grad_ndhwc_tiling.h
 * \brief NDHWC scheme tiling for 3D average pooling backward (arch35).
 *        Logic modeled on avg_pool_v2_grad_nhwc_tiling.cpp, extended to D/H/W.
 */

#ifndef OP_IMPL_AVG_POOL3_D_GRAD_NDHWC_TILING_H_
#define OP_IMPL_AVG_POOL3_D_GRAD_NDHWC_TILING_H_

#include "avg_pool3_d_grad_tiling_base.h"
#include "avg_pool3_d_grad_tiling_common.h"
#include "../../op_kernel/arch35/avg_pool3_d_grad_tiling_data.h"
#include "../../op_kernel/arch35/avg_pool3_d_grad_tiling_key.h"

namespace optiling {

struct AvgPool3DGradNDHWCBaseInfo {
    int64_t vRegSize{0};
    int64_t ubBlockSize{0};
    int64_t inputBytes{0};
    int64_t availableUb{0};
    int64_t totalCoreNum{0};
    int64_t coreUsedForBestPerformance{0};
    int64_t dataNumInOneBlock{0};
    int64_t proDataNumInOneBeat{0};
    int64_t moveDataNumCacheLine{0};
    int64_t dProBatchSize{1};
    int64_t hProBatchSize{1};
    int64_t wProBatchSize{1};
    int64_t isPad{0};
    int64_t isOverlap{0};
};

struct AvgPool3DGradNDHWCSplitInfo {
    int64_t isCheckRange{0};
    int64_t nOutputInner{1};
    int64_t nOutputTail{1};
    int64_t nOutputOuter{1};
    int64_t dOutputInner{1};
    int64_t dOutputTail{1};
    int64_t dOutputOuter{1};
    int64_t hOutputInner{1};
    int64_t hOutputTail{1};
    int64_t hOutputOuter{1};
    int64_t wOutputInner{1};
    int64_t wOutputTail{1};
    int64_t wOutputOuter{1};
    int64_t cOutputInner{1};
    int64_t cOutputTail{1};
    int64_t cOutputOuter{1};
    int64_t normalCoreProcessNum{1};
    int64_t tailCoreProcessNum{1};
    int64_t usedCoreNum{1};
    int64_t totalBaseBlockNum{1};
    int64_t outputBufferSize{0};
    int64_t inputGradBufferSize{0};
    int64_t totalBufferSize{0};
};

class AvgPool3DGradNDHWCTiling : public AvgPool3DGradTilingBase {
public:
    explicit AvgPool3DGradNDHWCTiling(gert::TilingContext* context) : AvgPool3DGradTilingBase(context) {}
    ~AvgPool3DGradNDHWCTiling() override {}

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetShapeAttrsInfo() override;

private:
    void InitializationVars();
    void DoBufferCalculate();
    bool IsMeetUBSize();
    bool IsMeetTargetCoreNum() const;
    bool TrySplitN();
    bool TrySplitAlignD();
    bool TrySplitAlignH();
    bool TrySplitAlignW();
    bool TrySplitAlignC();
    void SplitUnalignDHWC();
    void DynamicAdjustmentDHW();
    void SearchBestTiling();
    void DoUBTiling();
    void DoBlockTiling();
    void SetTilingData();

    AvgPool3DGradNDHWCBaseInfo baseData;
    AvgPool3DGradNDHWCSplitInfo splitData;
};

} // namespace optiling

#endif // OP_IMPL_AVG_POOL3_D_GRAD_NDHWC_TILING_H_
