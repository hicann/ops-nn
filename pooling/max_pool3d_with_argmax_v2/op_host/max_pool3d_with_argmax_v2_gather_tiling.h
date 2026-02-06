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
 * \file max_pool3d_with_argmax_v2_gather_tiling.h
 * \brief
 */

#ifndef MAX_POOL3D_WITH_AGRMAX_V2_GATHER_TILING_H_
#define MAX_POOL3D_WITH_AGRMAX_V2_GATHER_TILING_H_

#include "max_pool3d_with_argmax_v2_tiling_base.h"
#include "../op_kernel/arch35/max_pool3d_with_argmax_v2_tiling_struct.h"

namespace optiling
{
struct MaxPool3DWithArgmaxV2GatherBaseInfo {
    int64_t inputBytes = 0;
    int64_t indexBytes = 0;
    int64_t availableUb = 0;
    int64_t totalCoreNum = 0;
    int64_t oneBlockNumT1 = 0;
    int64_t oneBlockNumT2 = 0;
    int64_t coreUsedForBestPerformance = 0;

    int64_t padFront = 0;
    int64_t padTop = 0;
    int64_t padLeft = 0;
    int64_t dStride = 0;
    int64_t hStride = 0;
    int64_t wStride = 0;
    int64_t dKernel = 0;
    int64_t hKernel = 0;
    int64_t wKernel = 0;
    int64_t dInput = 0;
    int64_t hInput = 0;
    int64_t wInput = 0;
    int64_t dOutput = 0;
    int64_t hOutput = 0;
    int64_t wOutput = 0;
    int64_t highAxisTotal = 0;
    int64_t isPad = 0;
    int64_t dDilation = 0;
    int64_t hDilation = 0;
    int64_t wDilation = 0;
    std::string ToString() const
    {
        std::stringstream info;
        info << "MaxPool3DWithArgmaxV2GatherBaseInfo {";

        info << "inputBytes:" << inputBytes 
             << ", indexBytes:" << indexBytes 
             << ", availableUb:" << availableUb
             << ", totalCoreNum:" << totalCoreNum 
             << ", oneBlockNumT1:" << oneBlockNumT1 
             << ", oneBlockNumT2:" << oneBlockNumT2
             << ", coreUsedForBestPerformance:" << coreUsedForBestPerformance;

        info << ", padFront:" << padFront 
             << ", padTop:" << padTop 
             << ", padLeft:" << padLeft;

        info << ", dStride:" << dStride 
             << ", hStride:" << hStride 
             << ", wStride:" << wStride;

        info << ", dKernel:" << dKernel 
             << ", hKernel:" << hKernel 
             << ", wKernel:" << wKernel;

        info << ", dInput:" << dInput 
             << ", hInput:" << hInput 
             << ", wInput:" << wInput;

        info << ", dOutput:" << dOutput 
             << ", hOutput:" << hOutput 
             << ", wOutput:" << wOutput;

        info << ", highAxisTotal:" << highAxisTotal 
             << ", isPad:" << isPad;

        info << ", dDilation:" << dDilation 
             << ", hDilation:" << hDilation 
             << ", wDilation:" << wDilation;
        info << " }";
        return info.str();
    }
};

struct MaxPool3DWithArgmaxV2GatherSplitInfo {
    // InitializationVars
    int64_t highAxisInner = 0;
    int64_t highAxisTail = 0;
    int64_t highAxisOuter = 0;
    int64_t highAxisAligned = 0;

    // DoUBTiling
    int64_t dOutputInner = 0;
    int64_t dOutputTail = 0;
    int64_t dOutputOuter = 0;
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
    int64_t dInputInner = 0;
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
        info << "MaxPool3DWithArgmaxV2GatherSplitInfo {";
        
        info << " highAxisInner:" << highAxisInner
             << ", highAxisTail:" << highAxisTail
             << ", highAxisOuter:" << highAxisOuter
             << ", highAxisAligned:" << highAxisAligned;
         
        info << ", dOutputInner:" << dOutputInner
             << ", dOutputTail:" << dOutputTail
             << ", dOutputOuter:" << dOutputOuter
             << ", hOutputInner:" << hOutputInner
             << ", hOutputTail:" << hOutputTail
             << ", hOutputOuter:" << hOutputOuter
             << ", wOutputInner:" << wOutputInner
             << ", wOutputTail:" << wOutputTail
             << ", wOutputOuter:" << wOutputOuter;
        
        info << ", normalCoreProcessNum:" << normalCoreProcessNum
             << ", tailCoreProcessNum:" << tailCoreProcessNum
             << ", usedCoreNum:" << usedCoreNum
             << ", totalBaseBlockNum:" << totalBaseBlockNum;
        
        info << ", dInputInner:" << dInputInner
             << ", hInputInner:" << hInputInner
             << ", wInputInner:" << wInputInner
             << ", baseBlockPlaneSizeAligned:" << baseBlockPlaneSizeAligned
             << ", inputBufferSize:" << inputBufferSize
             << ", maxValueBufferSize:" << maxValueBufferSize
             << ", argmaxBufferSize:" << argmaxBufferSize
             << ", totalBufferSize:" << totalBufferSize;
        
        info << " }";
        return info.str();
    }
};

class MaxPool3DWithArgmaxV2GatherTiling : public MaxPool3DWithArgmaxV2BaseTiling
{
public:
     explicit MaxPool3DWithArgmaxV2GatherTiling(gert::TilingContext* context) : MaxPool3DWithArgmaxV2BaseTiling(context)
    {
    }

    ~MaxPool3DWithArgmaxV2GatherTiling() override
    {
    }

private:
    void DoUBTiling();
    void InitializationVars();
    bool IsMeetTargetCoreNum() const;
    void SearchBestTiling();
    bool IsMeetUBSize();
    void SetTilingData();
    void BinarySearch(int64_t start, int64_t end, int64_t* value);
    bool TrySplitNC();
    bool TrySplitD();
    bool TrySplitH();
    bool TrySplitW();
    uint64_t GetTilingKey() const override;
    void PrintBaseData() const;
    void PrintSplitData() const;
    void DoBlockTiling();
    void DoBufferCalculate();
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetShapeAttrsInfo() override;

    MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2GatherTilingData* tilingData_ = 
        context_->GetTilingData<MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2GatherTilingData>();
    MaxPool3DWithArgmaxV2GatherBaseInfo baseData_;
    MaxPool3DWithArgmaxV2GatherSplitInfo splitData_;
};

}  // namespace optiling

#endif