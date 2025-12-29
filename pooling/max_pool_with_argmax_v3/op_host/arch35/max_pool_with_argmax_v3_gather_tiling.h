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
 * \file max_pool_with_argmax_v3_gather_tiling.h
 * \brief
 */

#ifndef MAX_POOL_WITH_AGRMAX_V3_GATHER_TILING_H_
#define MAX_POOL_WITH_AGRMAX_V3_GATHER_TILING_H_

#include "max_pool_with_argmax_v3_tiling.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MaxPoolWithArgmaxV3GatherTilingData)
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
TILING_DATA_FIELD_DEF(int64_t, highAxisInner);
TILING_DATA_FIELD_DEF(int64_t, highAxisTail);
TILING_DATA_FIELD_DEF(int64_t, highAxisOuter);
TILING_DATA_FIELD_DEF(int64_t, hOutputInner);
TILING_DATA_FIELD_DEF(int64_t, hOutputTail);
TILING_DATA_FIELD_DEF(int64_t, hOutputOuter);
TILING_DATA_FIELD_DEF(int64_t, wOutputInner);
TILING_DATA_FIELD_DEF(int64_t, wOutputTail);
TILING_DATA_FIELD_DEF(int64_t, wOutputOuter);
TILING_DATA_FIELD_DEF(int64_t, normalCoreProcessNum);
TILING_DATA_FIELD_DEF(int64_t, tailCoreProcessNum);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, inputBufferSize);
TILING_DATA_FIELD_DEF(int64_t, maxValueBufferSize);
TILING_DATA_FIELD_DEF(int64_t, argmaxBufferSize);
TILING_DATA_FIELD_DEF(int64_t, isPad);
TILING_DATA_FIELD_DEF(int64_t, hDilation);
TILING_DATA_FIELD_DEF(int64_t, wDilation);
END_TILING_DATA_DEF;

// 300001 - no padding, 300002 - padding
REGISTER_TILING_DATA_CLASS(MaxPoolWithArgmaxV3_300001, MaxPoolWithArgmaxV3GatherTilingData);
REGISTER_TILING_DATA_CLASS(MaxPoolWithArgmaxV3_300002, MaxPoolWithArgmaxV3GatherTilingData);

struct MaxPoolWithArgmaxV3GatherBaseInfo {
    int64_t inputBytes = 0;
    int64_t indexBytes = 0;
    int64_t availableUb = 0;
    int64_t totalCoreNum = 0;
    int64_t oneBlockNumT1 = 0;
    int64_t oneBlockNumT2 = 0;
    int64_t coreUsedForBestPerformance = 0;

    int64_t padTop = 0;
    int64_t padLeft = 0;
    int64_t hStride = 0;
    int64_t wStride = 0;
    int64_t hKernel = 0;
    int64_t wKernel = 0;
    int64_t hInput = 0;
    int64_t wInput = 0;
    int64_t hOutput = 0;
    int64_t wOutput = 0;
    int64_t highAxisTotal = 0;
    int64_t isPad = 0;
    int64_t hDilation = 0;
    int64_t wDilation = 0;
    std::string ToString() const
    {
        std::stringstream info;
        info << "MaxPoolWithArgmaxV3GatherBaseInfo {";
        info << "inputBytes:" << inputBytes << ",indexBytes:" << indexBytes << ",availableUb:" << availableUb
             << ",totalCoreNum:" << totalCoreNum << ",coreUsedForBestPerformance:" << coreUsedForBestPerformance
             << ",padTop:" << padTop << ",padLeft:" << padLeft << ",hStride:" << hStride << ",wStride:" << wStride
             << ",hKernel:" << hKernel << ",wKernel:" << wKernel << ",nInput:" << highAxisTotal << ",hInput:" << hInput
             << ",wInput:" << wInput << ",hOutput:" << hOutput << ",wOutput:" << wOutput << ",isPad:" << isPad
             << ",hDilation:" << hDilation << ",wDilation:" << wDilation;
        info << " }";
        return info.str();
    }
};

struct MaxPoolWithArgmaxV3GatherSplitInfo {
    // InitializationVars
    int64_t highAxisInner = 0;
    int64_t highAxisTail = 0;
    int64_t highAxisOuter = 0;
    int64_t highAxisAligned = 0;

    // DoUBTiling
    int64_t hOutputInner = 0;
    int64_t hOutputTail = 0;
    int64_t hOutputOuter = 0;
    int64_t wOutputInner = 0;
    int64_t wOutputTail = 0;
    int64_t wOutputOuter = 0;

    // DoBlockTiling
    int64_t normalCoreProcessNum = 0;
    int64_t tailCoreProcessNum = 0;
    int64_t usedCoreNum = 0;
    int64_t totalBaseBlockNum = 0;

    // DoBufferCalculate
    int64_t hInputInner = 0;
    int64_t wInputInner = 0;
    int64_t baseBlockPlaneSizeAligned = 0;
    int64_t inputBufferSize = 0;
    int64_t maxValueBufferSize = 0;
    int64_t argmaxBufferSize = 0;
    int64_t totalBufferSize = 0;
    std::string ToString() const
    {
        std::stringstream info;
        info << "MaxPoolWithArgmaxV3NhwcSplitInfo {";
        info << "highAxisInner:" << highAxisInner << ",highAxisTail:" << highAxisTail
             << ",highAxisOuter:" << highAxisOuter << ",hOutputInner:" << hOutputInner << ",hOutputTail:" << hOutputTail
             << ",hOutputOuter:" << hOutputOuter << ",wOutputInner:" << wOutputInner << ",wOutputTail:" << wOutputTail
             << ",wOutputOuter:" << wOutputOuter << ",normalCoreProcessNum:" << normalCoreProcessNum
             << ",tailCoreProcessNum:" << tailCoreProcessNum << ",usedCoreNum:" << usedCoreNum
             << ",totalBaseBlockNum:" << totalBaseBlockNum << ",hInputInner:" << hInputInner
             << ",wInputInner:" << wInputInner << ",inputBufferSize:" << inputBufferSize
             << ",maxValueBufferSize:" << maxValueBufferSize << ",argmaxBufferSize:" << argmaxBufferSize
             << ",totalBufferSize:" << totalBufferSize;
        info << " }";
        return info.str();
    }
};

class MaxPoolWithArgmaxV3GatherTiling : public MaxPoolWithArgmaxV3BaseTiling {
public:
    explicit MaxPoolWithArgmaxV3GatherTiling(gert::TilingContext* context) : MaxPoolWithArgmaxV3BaseTiling(context)
    {}

    ~MaxPoolWithArgmaxV3GatherTiling() override
    {}

private:
    void DoUBTiling();
    void InitializationVars();
    bool IsMeetTargetCoreNum() const;
    void SearchBestTiling();
    bool IsMeetUBSize();
    void SetTilingData();
    void BinarySearch(int64_t start, int64_t end, int64_t* value);
    bool TrySplitNC();
    bool TrySplitH();
    bool TrySplitW();
    uint64_t GetTilingKey() const override;
    void PrintBaseData() const;
    void PrintSplitData() const;
    void DoBlockTiling();
    void DoBufferCalculate();
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

    MaxPoolWithArgmaxV3GatherTilingData tilingData_;
    MaxPoolWithArgmaxV3GatherBaseInfo baseData_;
    MaxPoolWithArgmaxV3GatherSplitInfo splitData_;
};

} // namespace optiling

#endif