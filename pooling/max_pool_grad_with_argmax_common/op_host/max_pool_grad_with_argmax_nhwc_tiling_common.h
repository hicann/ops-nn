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
 * \file max_pool_grad_with_argmax_nhwc_tiling_common.h
 * \brief
 */

#ifndef MAX_POOL_GRAD_WITH_AGRMAX_NHWC_TILING_COMMON_H_
#define MAX_POOL_GRAD_WITH_AGRMAX_NHWC_TILING_COMMON_H_

#include "max_pool_grad_with_argmax_tiling.h"
#include "../op_kernel/max_pool_grad_with_argmax_struct_common.h"

namespace optiling
{
struct MaxPoolGradWithArgmaxNHWCBaseInfo {
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

struct MaxPoolGradWithArgmaxNHWCSplitInfo {
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

class MaxPoolGradWithArgmaxNHWCTilingCommon
{
public:
    MaxPoolGradWithArgmaxNHWCTilingCommon(MaxPoolGradWithArgmaxInputInfoCommon* input)
    {
        inputData = input;
    }
    void InitializationVars(gert::TilingContext* context_, MaxPoolGradWithArgmaxHardwareInfo* hardwareData);
    ge::graphStatus DoOpTiling(gert::TilingContext* context, uint64_t key);
    ge::graphStatus PostTiling(gert::TilingContext* context_);
    MaxPoolGradWithArgmaxNHWCSplitInfo GetSplitData();
    MaxPoolGradWithArgmaxNHWCBaseInfo GetBaseData();
    bool CheckUBSize();

private:
    void DoUBTiling();
    bool TrySplitN();
    bool TrySplitAlignH();
    bool TrySplitAlignW();
    bool TrySplitAlignC();
    void SplitUnalignHWC();
    bool IsMeetTargetCoreNum() const;
    bool IsMeetUBSize();
    void SearchBestTiling();
    void DynamicAdjustmentWH();
    void SetTilingData(gert::TilingContext* context, uint64_t key);
    void PrintBaseData() const;
    void PrintSplitData() const;
    void DoBlockTiling();
    void DoBufferCalculate();

    MaxPoolGradWithArgmaxNHWCBaseInfo baseData;
    MaxPoolGradWithArgmaxNHWCSplitInfo splitData;
    MaxPoolGradWithArgmaxInputInfoCommon* inputData;
};

}  // namespace optiling

#endif