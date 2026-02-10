/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file max_pool_grad_with_argmax_v3_nchw_tiling.h
 * \brief
 */

#ifndef MAX_POOL_GRAD_WITH_AGRMAX_V3_NCHW_TILING_H_
#define MAX_POOL_GRAD_WITH_AGRMAX_V3_NCHW_TILING_H_

#include "max_pool_grad_with_argmax_v3_tiling_base.h"

namespace optiling {

struct MaxPoolGradWithArgmaxV3NCHWBaseInfo {
    int64_t vRegSize{0};
    int64_t ubBlockSize{0};
    int64_t inputBytes{0};
    int64_t indexBytes{0};
    int64_t availableUb{0};
    int64_t totalCoreNum{0};
    int64_t coreUsedForBestPerformance{0};
    int64_t hProBatchSize{0};
    int64_t wProBatchSize{0};
    int64_t inputNCSize{0};
    int64_t maxDataNumInOneBlock{0};
    int64_t proDataNumInOneBeatT2{0};
    int64_t isPad{0};
    int64_t isOverlap{0};
};

struct MaxPoolGradWithArgmaxV3NCHWSplitInfo {
    // DoUBTiling
    int64_t isCheckRange{0};

    int64_t highAxisInner{0};
    int64_t highAxisTail{0};
    int64_t highAxisOuter{0};

    int64_t hOutputInner{0};
    int64_t hOutputTail{0};
    int64_t hOutputOuter{0};

    int64_t wOutputInner{0};
    int64_t wOutputTail{0};
    int64_t wOutputOuter{0};

    // DoBlockTiling
    int64_t normalCoreProcessNum{0};
    int64_t tailCoreProcessNum{0};
    int64_t usedCoreNum{0};
    int64_t totalBaseBlockNum{0};

    // DoBufferCalculate
    int64_t outputBufferSize{0};
    int64_t gradBufferSize{0};
    int64_t argmaxBufferSize{0};
    int64_t totalBufferSize{0};
};

class MaxPoolGradWithArgmaxV3NCHWTiling : public MaxPoolGradWithArgmaxV3BaseTiling {
public:
    explicit MaxPoolGradWithArgmaxV3NCHWTiling(gert::TilingContext* context)
        : MaxPoolGradWithArgmaxV3BaseTiling(context)
    {}

    ~MaxPoolGradWithArgmaxV3NCHWTiling() override
    {}

private:
    void DoUBTiling();
    void InitializationVars();
    bool TrySplitNC();
    bool TrySplitAlignH();
    bool TrySplitAlignW();
    void SplitUnalignHW();
    bool IsMeetTargetCoreNum() const;
    bool IsMeetUBSize();
    void SearchBestTiling();
    void DynamicAdjustmentWH();
    void SetTilingData();
    uint64_t GetTilingKey() const override;
    void PrintBaseData() const;
    void PrintSplitData() const;
    void DoBlockTiling();
    void DoBufferCalculate();
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

    MaxPoolGradWithArgmaxV3NCHWBaseInfo baseData;
    MaxPoolGradWithArgmaxV3NCHWSplitInfo splitData;
};

} // namespace optiling

#endif