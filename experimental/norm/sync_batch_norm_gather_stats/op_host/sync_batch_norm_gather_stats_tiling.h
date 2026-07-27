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
 * \file sync_batch_norm_gather_stats_tiling.h
 * \brief
 */
#ifndef SYNC_BATCH_NORM_GATHER_STATS_TILING_H
#define SYNC_BATCH_NORM_GATHER_STATS_TILING_H

#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "../op_kernel/sync_batch_norm_gather_stats_tiling_data.h"

namespace optiling {
struct SyncBatchNormGatherStatsCompileInfo {
    int64_t blockSize = 0;
    int64_t ubSize = 0;
    int64_t coreNum = 0;
    bool socIs310P = false;
};

class SyncBatchNormGatherStatsTiling {
public:
    explicit SyncBatchNormGatherStatsTiling(gert::TilingContext* context) : context_(context) {}
    ~SyncBatchNormGatherStatsTiling() = default;

    ge::graphStatus DoTiling();

protected:
    ge::graphStatus GetShapeAttrsInfo();
    ge::graphStatus GetPlatformInfo();
    bool IsCapable();
    ge::graphStatus DoOpTiling();
    ge::graphStatus DoLibApiTiling();
    ge::graphStatus GetWorkspaceSize();
    ge::graphStatus PostTiling();
    uint64_t GetTilingKey() const;

    bool TotalSumShapeCheck();
    bool TotalSquareSumShapeCheck();
    bool SampleCountShapeCheck();
    bool MeanShapeCheck();
    bool VarShapeCheck();
    bool BatchMeanCheck();
    bool BatchInvstdCheck();
    bool RunningMeanCheck();
    bool RunningVarCheck();
    void PrintTilingData();
    void SetTilingDataInfo();

    // Refactored helper functions for GetShapeAttrsInfo
    bool ValidateInputShapes();
    bool ValidateOutputShapes();
    ge::graphStatus GetAttrs();
    bool ValidateInputDtypes();
    bool ValidateOutputDtypes();
    bool ValidateInputDimensions();
    bool ValidateOutputDimensions();

    ge::graphStatus DoNNotFullLoadTiling();
    void ComputeSmallCNNotFullLoad(uint32_t dTypeSize, int64_t cBlockFactor, int64_t cTileBlockFactor);
    ge::graphStatus SearchNNotFullLoadTiling(uint32_t dTypeSize, int64_t cBlockFactor, int64_t cTileBlockFactor);
    bool TrySearchNTileNum(int64_t nTileNum, bool isFirst, uint32_t dTypeSize, int64_t cTileNumBase,
                           int64_t cBlockFactor, int64_t cTileBlockFactor);
    void FillNNotFullLoadTilingData();
    void RebalanceBlocksForHalfCore(uint32_t dTypeSize);
    ge::graphStatus CheckArch20CacheLineInvariant(uint32_t dTypeSize, int64_t perCoreCNum) const;
    void AlignPerCoreToCacheLine(uint32_t dTypeSize);
    int64_t FindNearestPower2(const int64_t value);
    int64_t GetCacheID(const int64_t idx);

private:
    gert::TilingContext* context_ = nullptr;
    const char* opName = "SyncBatchNormGatherStats";

    gert::Shape totalSumShape_;
    gert::Shape totalSquareSumShape_;
    gert::Shape sampleCountShape_;
    gert::Shape meanShape_;
    gert::Shape varShape_;
    gert::Shape batchMeanShape_;
    gert::Shape batchInvStdShape_;
    gert::Shape runningMeanShape_;
    gert::Shape runningVarShape_;

    uint64_t totalSumDim0_ = 0;
    uint64_t totalSumDim1_ = 0;
    uint64_t totalSquareSumDim0_ = 0;
    uint64_t totalSquareSumDim1_ = 0;
    uint64_t sampleCountDim0_ = 0;
    uint64_t meanDim0_ = 0;
    uint64_t varDim0_ = 0;
    uint64_t batchMeanDim0_ = 0;
    uint64_t batchInvStdDim0_ = 0;
    uint64_t runningMeanDim0_ = 0;
    uint64_t runningVarDim0_ = 0;
    float momentum_ = 0;
    float eps_ = 0;
    int64_t nLen = 0;
    int64_t cLen = 0;
    int64_t coreNum_ = 0;
    bool socIs310P_ = false;
    int64_t ubSize_ = 0;
    int64_t blockSize_ = 0;

    ge::DataType totalSumDType_{ge::DT_UNDEFINED};
    ge::DataType totalSquareSumDType_{ge::DT_UNDEFINED};
    ge::DataType sampleCountDType_{ge::DT_UNDEFINED};
    ge::DataType meanDType_{ge::DT_UNDEFINED};
    ge::DataType varDType_{ge::DT_UNDEFINED};

    ge::DataType batchMeanDType_{ge::DT_UNDEFINED};
    ge::DataType batchInvStdDType_{ge::DT_UNDEFINED};
    ge::DataType runningMeanDType_{ge::DT_UNDEFINED};
    ge::DataType runningVarDType_{ge::DT_UNDEFINED};

    int64_t cTileNum_ = 0;
    int64_t blockFormer = 0;
    int64_t ubOuter = 0;
    int64_t ubTail = 0;
    int64_t blockNum = 0;
    int64_t blockTail = 0;
    int64_t workspaceSize_ = 0;

    int64_t nTileNum_ = 0;
    int64_t nLoop_ = 0;
    int64_t nTail_ = 0;
    int64_t basicBlockLoop_ = 0;
    int64_t mainFoldCount_ = 0;
    int64_t cacheBufferCount_ = 0;
    int64_t resultCacheID_ = 0;
    int64_t cLoopMain_ = 0;
    int64_t cTailMain_ = 0;
    int64_t cLoopTail_ = 0;
    int64_t cTailTail_ = 0;
};
}; // namespace optiling
#endif // SYNC_BATCH_NORM_GATHER_STATS_TILING_H
