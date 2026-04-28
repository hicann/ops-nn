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
 * \file max_pool_grad_nchw_tiling_common.h
 * \brief NCHW格式MaxPoolGrad通用Tiling实现
 */

#ifndef MAX_POOL_GRAD_NCHW_TILING_COMMON_H_
#define MAX_POOL_GRAD_NCHW_TILING_COMMON_H_

#include "../../op_kernel/arch35/max_pool_grad_with_argmax_struct_common.h"
#include "max_pool_grad_with_argmax_tiling_common.h"

namespace optiling {
static constexpr int64_t T3_INT64_NCHW = 10;

struct MaxPoolGradNCHWBaseInfo {
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

struct MaxPoolGradNCHWSplitInfo {
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
    int64_t inputBufferSize{0};
    int64_t outputBufferSize{0};
    int64_t gradBufferSize{0};
    int64_t argmaxBufferSize{0};
    int64_t totalBufferSize{0};
};

class MaxPoolGradNCHWTilingCommon {
public:
    MaxPoolGradNCHWTilingCommon(MaxPoolGradWithArgmaxInputInfoCommon* input) : inputData(input)
    {}
    void InitializationVars(gert::TilingContext* context_, MaxPoolGradWithArgmaxHardwareInfo* hardwareData);
    ge::graphStatus DoOpTiling(gert::TilingContext* context, uint64_t key);
    ge::graphStatus PostTiling(gert::TilingContext* context_);
    MaxPoolGradNCHWSplitInfo GetSplitData() const;
    MaxPoolGradNCHWBaseInfo GetBaseData();
    bool CheckUBSize();

protected:
    virtual void DoBufferCalculate();
    virtual void SetTilingData(gert::TilingContext* context, uint64_t key);
    void DoUBTiling();
    bool TrySplitNC();
    bool TrySplitAlignH();
    bool TrySplitAlignW();
    void SplitUnalignHW();
    bool IsMeetTargetCoreNum() const;
    bool IsMeetUBSize();
    void SearchBestTiling();
    void DynamicAdjustmentWH();
    void PrintBaseData() const;
    void PrintSplitData() const;
    void DoBlockTiling();

    MaxPoolGradNCHWBaseInfo baseData;
    MaxPoolGradNCHWSplitInfo splitData;
    MaxPoolGradWithArgmaxInputInfoCommon* inputData;
};

} // namespace optiling

#endif // MAX_POOL_GRAD_NCHW_TILING_COMMON_H_