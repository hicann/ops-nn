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
 * \file max_pool_grad_with_argmax_v3_nhwc_tiling.h
 * \brief
 */

#ifndef MAX_POOL_GRAD_WITH_AGRMAX_V3_NHWC_TILING_H_
#define MAX_POOL_GRAD_WITH_AGRMAX_V3_NHWC_TILING_H_

#include "max_pool_grad_with_argmax_v3_tiling_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MaxPoolGradWithArgmaxV3NHWCTilingData)
TILING_DATA_FIELD_DEF(int64_t, hArgmax);
TILING_DATA_FIELD_DEF(int64_t, wArgmax);
TILING_DATA_FIELD_DEF(int64_t, cOutput);
TILING_DATA_FIELD_DEF(int64_t, hOutput);
TILING_DATA_FIELD_DEF(int64_t, wOutput);
TILING_DATA_FIELD_DEF(int64_t, hKernel);
TILING_DATA_FIELD_DEF(int64_t, wKernel);
TILING_DATA_FIELD_DEF(int64_t, hStride);
TILING_DATA_FIELD_DEF(int64_t, wStride);
TILING_DATA_FIELD_DEF(int64_t, padH);
TILING_DATA_FIELD_DEF(int64_t, padW);
TILING_DATA_FIELD_DEF(int64_t, dilationH);
TILING_DATA_FIELD_DEF(int64_t, dilationW);
TILING_DATA_FIELD_DEF(int64_t, nOutputInner);
TILING_DATA_FIELD_DEF(int64_t, nOutputTail);
TILING_DATA_FIELD_DEF(int64_t, nOutputOuter);
TILING_DATA_FIELD_DEF(int64_t, hOutputInner);
TILING_DATA_FIELD_DEF(int64_t, hOutputTail);
TILING_DATA_FIELD_DEF(int64_t, hOutputOuter);
TILING_DATA_FIELD_DEF(int64_t, wOutputInner);
TILING_DATA_FIELD_DEF(int64_t, wOutputTail);
TILING_DATA_FIELD_DEF(int64_t, wOutputOuter);
TILING_DATA_FIELD_DEF(int64_t, cOutputInner);
TILING_DATA_FIELD_DEF(int64_t, cOutputTail);
TILING_DATA_FIELD_DEF(int64_t, cOutputOuter);
TILING_DATA_FIELD_DEF(int64_t, normalCoreProcessNum);
TILING_DATA_FIELD_DEF(int64_t, tailCoreProcessNum);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, outputBufferSize);
TILING_DATA_FIELD_DEF(int64_t, gradBufferSize);
TILING_DATA_FIELD_DEF(int64_t, argmaxBufferSize);
TILING_DATA_FIELD_DEF(int64_t, hProBatchSize);
TILING_DATA_FIELD_DEF(int64_t, wProBatchSize);
TILING_DATA_FIELD_DEF(int64_t, tilingKey);
END_TILING_DATA_DEF;

// no check range 200          check range 201
// no check range int64  210          check range int64  211
REGISTER_TILING_DATA_CLASS(MaxPoolGradWithArgmaxV3_200, MaxPoolGradWithArgmaxV3NHWCTilingData);
REGISTER_TILING_DATA_CLASS(MaxPoolGradWithArgmaxV3_201, MaxPoolGradWithArgmaxV3NHWCTilingData);
REGISTER_TILING_DATA_CLASS(MaxPoolGradWithArgmaxV3_210, MaxPoolGradWithArgmaxV3NHWCTilingData);
REGISTER_TILING_DATA_CLASS(MaxPoolGradWithArgmaxV3_211, MaxPoolGradWithArgmaxV3NHWCTilingData);

struct MaxPoolGradWithArgmaxV3NHWCBaseInfo {
    int64_t vRegSize{0};
    int64_t ubBlockSize{0};
    int64_t inputBytes{0};
    int64_t indexBytes{0};
    int64_t availableUb{0};
    int64_t maxDataNumInOneBlock{0};
    int64_t proDataNumInOneBeatT2{0};
    int64_t totalCoreNum{0};
    int64_t coreUsedForBestPerformance{0};
    int64_t isPad{0};
    int64_t isOverlap{0};
    int64_t hProBatchSize{0};
    int64_t wProBatchSize{0};
    int64_t moveDataNumCacheLineT2{0};
};

struct MaxPoolGradWithArgmaxV3NHWCSplitInfo {
    // DoUBTiling
    int64_t isCheckRange{0};

    int64_t nOutputInner{0};
    int64_t nOutputTail{0};
    int64_t nOutputOuter{0};

    int64_t hOutputInner{0};
    int64_t hOutputTail{0};
    int64_t hOutputOuter{0};

    int64_t wOutputInner{0};
    int64_t wOutputTail{0};
    int64_t wOutputOuter{0};

    int64_t cOutputInner{0};
    int64_t cOutputTail{0};
    int64_t cOutputOuter{0};

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

class MaxPoolGradWithArgmaxV3NHWCTiling : public MaxPoolGradWithArgmaxV3BaseTiling {
public:
    explicit MaxPoolGradWithArgmaxV3NHWCTiling(gert::TilingContext* context)
        : MaxPoolGradWithArgmaxV3BaseTiling(context)
    {}

    ~MaxPoolGradWithArgmaxV3NHWCTiling() override
    {}

private:
    void DoUBTiling();
    void InitializationVars();
    bool TrySplitN();
    bool TrySplitAlignH();
    bool TrySplitAlignW();
    bool TrySplitAlignC();
    void SplitUnalignHWC();
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

    MaxPoolGradWithArgmaxV3NHWCTilingData tilingData;
    MaxPoolGradWithArgmaxV3NHWCBaseInfo baseData;
    MaxPoolGradWithArgmaxV3NHWCSplitInfo splitData;
};

} // namespace optiling

#endif