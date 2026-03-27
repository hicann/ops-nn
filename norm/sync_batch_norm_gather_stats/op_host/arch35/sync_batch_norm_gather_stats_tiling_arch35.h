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
 * \file sync_batch_norm_gather_stats_tiling_arch35.h
 * \brief
 */
#ifndef SYNC_BATCH_NORM_GATHER_STATS_TILING_ARCH35_H
#define SYNC_BATCH_NORM_GATHER_STATS_TILING_ARCH35_H

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"

namespace optiling {
struct SyncBatchNormGatherStatsCompileInfo {
    int64_t blockSize = 0;
    int64_t ubSize = 0;
    int64_t coreNum = 0;
};

BEGIN_TILING_DATA_DEF(SyncBatchNormGatherStatsTilingData)
TILING_DATA_FIELD_DEF(uint64_t, blockDim);
TILING_DATA_FIELD_DEF(uint64_t, blockFormer);
TILING_DATA_FIELD_DEF(uint64_t, blockTail); 
TILING_DATA_FIELD_DEF(uint64_t, nLen); 
TILING_DATA_FIELD_DEF(uint64_t, cLen);
TILING_DATA_FIELD_DEF(uint64_t, ubFormer); 
TILING_DATA_FIELD_DEF(uint64_t, ubTail); 
TILING_DATA_FIELD_DEF(float, momentum);
TILING_DATA_FIELD_DEF(float, eps);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SyncBatchNormGatherStats, SyncBatchNormGatherStatsTilingData)

BEGIN_TILING_DATA_DEF(SyncBatchNormGatherStatsNNotFullLoadTilingData)
TILING_DATA_FIELD_DEF(int64_t, blockDim);
TILING_DATA_FIELD_DEF(int64_t, cLen);
TILING_DATA_FIELD_DEF(int64_t, cFactor);
TILING_DATA_FIELD_DEF(int64_t, cLoopMainBlock);
TILING_DATA_FIELD_DEF(int64_t, cTileMainBlock);
TILING_DATA_FIELD_DEF(int64_t, cLoopTailBlock);
TILING_DATA_FIELD_DEF(int64_t, cTailTailBlock);
TILING_DATA_FIELD_DEF(int64_t, nFactor);
TILING_DATA_FIELD_DEF(int64_t, nLoop);
TILING_DATA_FIELD_DEF(int64_t, nMainFoldCount);
TILING_DATA_FIELD_DEF(int64_t, nTail);
TILING_DATA_FIELD_DEF(int64_t, cacheBufferCount);
TILING_DATA_FIELD_DEF(int32_t, resultCacheId);
TILING_DATA_FIELD_DEF(float, momentum);
TILING_DATA_FIELD_DEF(float, eps);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SyncBatchNormGatherStats_20001, SyncBatchNormGatherStatsNNotFullLoadTilingData)

class SyncBatchNormGatherStatsTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit SyncBatchNormGatherStatsTiling(gert::TilingContext *context) : TilingBaseClass(context) {
    }

    void Reset(gert::TilingContext* context) override {
        TilingBaseClass::Reset(context);
    }
    ~SyncBatchNormGatherStatsTiling() override = default;

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

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

    ge::graphStatus DoNNotFullLoadTiling();
    int64_t FindNearestPower2(const int64_t value);
    int64_t GetCacheID(const int64_t idx);

private:
    const char *opName = "SyncBatchNormGatherStats";
    SyncBatchNormGatherStatsTilingData tilingData;
    SyncBatchNormGatherStatsNNotFullLoadTilingData nNotFullLoadTilingData;
    
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