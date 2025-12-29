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
 * \file max_pool_with_argmax_v3_nhwc_tiling.h
 * \brief
 */

#ifndef MAX_POOL_WITH_AGRMAX_V3_NHWC_TILING_H_
#define MAX_POOL_WITH_AGRMAX_V3_NHWC_TILING_H_

#include "max_pool_with_argmax_v3_tiling.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MaxPoolWithArgmaxV3NhwcTilingData)
TILING_DATA_FIELD_DEF(int64_t, cInput);
TILING_DATA_FIELD_DEF(int64_t, hInput);
TILING_DATA_FIELD_DEF(int64_t, wInput);
TILING_DATA_FIELD_DEF(int64_t, hOutput);
TILING_DATA_FIELD_DEF(int64_t, wOutput);
TILING_DATA_FIELD_DEF(int64_t, hKernel);
TILING_DATA_FIELD_DEF(int64_t, wKernel);
TILING_DATA_FIELD_DEF(int64_t, hStride);
TILING_DATA_FIELD_DEF(int64_t, wStride);
TILING_DATA_FIELD_DEF(int64_t, padLeft);
TILING_DATA_FIELD_DEF(int64_t, padTop);
TILING_DATA_FIELD_DEF(int64_t, hDilation);
TILING_DATA_FIELD_DEF(int64_t, wDilation);
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
TILING_DATA_FIELD_DEF(int64_t, inputBufferSize);
TILING_DATA_FIELD_DEF(int64_t, maxValueBufferSize);
TILING_DATA_FIELD_DEF(int64_t, argmaxBufferSize);
TILING_DATA_FIELD_DEF(int64_t, isPad);
TILING_DATA_FIELD_DEF(int64_t, isSplitKernel);
TILING_DATA_FIELD_DEF(int64_t, hKernelInner);
TILING_DATA_FIELD_DEF(int64_t, hKernelTail);
TILING_DATA_FIELD_DEF(int64_t, hKernelOuter);
TILING_DATA_FIELD_DEF(int64_t, wKernelInner);
TILING_DATA_FIELD_DEF(int64_t, wKernelTail);
TILING_DATA_FIELD_DEF(int64_t, wKernelOuter);
TILING_DATA_FIELD_DEF(int64_t, tilingKey);
END_TILING_DATA_DEF;

// small c  700001 - no padding, 700002 - padding
// large c  800001 - no padding, 800002 - padding
REGISTER_TILING_DATA_CLASS(MaxPoolWithArgmaxV3_700001, MaxPoolWithArgmaxV3NhwcTilingData);
REGISTER_TILING_DATA_CLASS(MaxPoolWithArgmaxV3_700002, MaxPoolWithArgmaxV3NhwcTilingData);
REGISTER_TILING_DATA_CLASS(MaxPoolWithArgmaxV3_800001, MaxPoolWithArgmaxV3NhwcTilingData);
REGISTER_TILING_DATA_CLASS(MaxPoolWithArgmaxV3_800002, MaxPoolWithArgmaxV3NhwcTilingData);

struct MaxPoolWithArgmaxV3NhwcBaseInfo {
    int64_t inputBytes{0};
    int64_t indexBytes{0};
    int64_t availableUb{0};
    int64_t totalCoreNum{0};
    int64_t coreUsedForBestPerformance{0};
    int64_t concurrentCount{0};
    int64_t templateMode{0};

    int64_t padTop{0};
    int64_t padLeft{0};
    int64_t hDilation{0};
    int64_t wDilation{0};
    int64_t hStride{0};
    int64_t wStride{0};
    int64_t hKernel{0};
    int64_t wKernel{0};
    int64_t nInput{0};
    int64_t hInput{0};
    int64_t wInput{0};
    int64_t cInput{0};
    int64_t hOutput{0};
    int64_t wOutput{0};

    int64_t isPad{0};

    std::string ToString() const
    {
        std::stringstream info;
        info << "MaxPoolWithArgmaxV3NhwcBaseInfo {";
        info << "inputBytes:" << inputBytes << ",indexBytes:" << indexBytes << ",availableUb:" << availableUb
             << ",totalCoreNum:" << totalCoreNum << ",coreUsedForBestPerformance:" << coreUsedForBestPerformance
             << ",concurrentCount:" << concurrentCount << ",templateMode:" << templateMode << ",padTop:" << padTop
             << ",padLeft:" << padLeft << ",hDilation:" << hDilation << ",wDilation:" << wDilation
             << ",hStride:" << hStride << ",wStride:" << wStride << ",hKernel:" << hKernel << ",wKernel:" << wKernel
             << ",nInput:" << nInput << ",hInput:" << hInput << ",wInput:" << wInput << ",cInput:" << cInput
             << ",hOutput:" << hOutput << ",wOutput:" << wOutput << ",isPad:" << isPad;
        info << " }";
        return info.str();
    }
};

struct MaxPoolWithArgmaxV3NhwcSplitInfo {
    // DoUBTiling
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

    int64_t isSplitKernel{0};
    int64_t hKernelInner{0};
    int64_t hKernelTail{0};
    int64_t hKernelOuter{0};
    int64_t wKernelInner{0};
    int64_t wKernelTail{0};
    int64_t wKernelOuter{0};

    // DoBlockTiling
    int64_t normalCoreProcessNum{0};
    int64_t tailCoreProcessNum{0};
    int64_t usedCoreNum{0};
    int64_t totalBaseBlockNum{0};

    // DoBufferCalculate
    int64_t hInputInner{0};
    int64_t wInputInner{0};
    int64_t inputBufferSize{0};
    int64_t maxValueBufferSize{0};
    int64_t argmaxBufferSize{0};
    int64_t totalBufferSize{0};

    std::string ToString() const
    {
        std::stringstream info;
        info << "MaxPoolWithArgmaxV3NhwcSplitInfo {";
        info << "nOutputInner:" << nOutputInner << ",nOutputTail:" << nOutputTail << ",nOutputOuter:" << nOutputOuter
             << ",hOutputInner:" << hOutputInner << ",hOutputTail:" << hOutputTail << ",hOutputOuter:" << hOutputOuter
             << ",wOutputInner:" << wOutputInner << ",wOutputTail:" << wOutputTail << ",wOutputOuter:" << wOutputOuter
             << ",cOutputInner:" << cOutputInner << ",cOutputTail:" << cOutputTail << ",cOutputOuter:" << cOutputOuter
             << ",isSplitKernel:" << isSplitKernel << ",hKernelInner:" << hKernelInner << ",hKernelTail:" << hKernelTail
             << ",hKernelOuter:" << hKernelOuter << ",wKernelInner:" << wKernelInner << ",wKernelTail:" << wKernelTail
             << ",wKernelOuter:" << wKernelOuter << ",normalCoreProcessNum:" << normalCoreProcessNum
             << ",tailCoreProcessNum:" << tailCoreProcessNum << ",usedCoreNum:" << usedCoreNum
             << ",totalBaseBlockNum:" << totalBaseBlockNum << ",hInputInner:" << hInputInner
             << ",wInputInner:" << wInputInner << ",inputBufferSize:" << inputBufferSize
             << ",maxValueBufferSize:" << maxValueBufferSize << ",argmaxBufferSize:" << argmaxBufferSize
             << ",totalBufferSize:" << totalBufferSize;
        info << " }";
        return info.str();
    }
};

class MaxPoolWithArgmaxV3NhwcTiling : public MaxPoolWithArgmaxV3BaseTiling {
public:
    explicit MaxPoolWithArgmaxV3NhwcTiling(gert::TilingContext* context) : MaxPoolWithArgmaxV3BaseTiling(context)
    {}

    ~MaxPoolWithArgmaxV3NhwcTiling() override
    {}

private:
    void DoUBTiling();
    void InitializationVars();
    bool IsMeetTargetCoreNum() const;
    bool IsMeetUBSize();
    void SearchBestTiling();
    void BinarySearch(int64_t start, int64_t end, int64_t* value, int64_t rate = 1);
    bool TrySplitN();
    bool TrySplitH();
    bool TrySplitW();
    void SplitC();
    void SplitKernel();
    void DynamicAdjustmentKernelWH();
    void SetTilingData();
    uint64_t GetTilingKey() const override;
    void DoBlockTiling();
    void RerouteTemplateBySplit();
    void DoBufferCalculate();
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

    MaxPoolWithArgmaxV3NhwcTilingData tilingData_;
    MaxPoolWithArgmaxV3NhwcBaseInfo baseData_;
    MaxPoolWithArgmaxV3NhwcSplitInfo splitData_;
};

} // namespace optiling

#endif