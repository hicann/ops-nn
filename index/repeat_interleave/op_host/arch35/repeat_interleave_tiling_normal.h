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
 * \file repeat_interleave_tiling_normal.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_REPEAT_INTERLEAVE_TILING_NORMAL_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_REPEAT_INTERLEAVE_TILING_NORMAL_H_

#include "repeat_interleave_tiling_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(RepeatInterleaveTilingKernelDataNorm)
TILING_DATA_FIELD_DEF(int64_t, totalCoreNum);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, eachCoreBatchCount);
TILING_DATA_FIELD_DEF(int64_t, tailCoreBatchCount);
TILING_DATA_FIELD_DEF(int64_t, isSplitCP);
TILING_DATA_FIELD_DEF(int64_t, normalCP);
TILING_DATA_FIELD_DEF(int64_t, tailCP);
TILING_DATA_FIELD_DEF(int64_t, cpSlice);
TILING_DATA_FIELD_DEF(int64_t, ubFactor);
TILING_DATA_FIELD_DEF(int64_t, repeatsCount);
TILING_DATA_FIELD_DEF(int64_t, totalRepeatSum);
TILING_DATA_FIELD_DEF_ARR(int64_t, REPEAT_INTERLEAVE_MERGED_DIM_LENGTH, mergedDims);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RepeatInterleave, RepeatInterleaveTilingKernelDataNorm)

BEGIN_TILING_DATA_DEF(RepeatInterleaveTilingKernelDataSmall)
TILING_DATA_FIELD_DEF(int64_t, totalCoreNum);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, eachCoreBatchCount);
TILING_DATA_FIELD_DEF(int64_t, tailCoreBatchCount);
TILING_DATA_FIELD_DEF(int64_t, normalRepeatsCount);
TILING_DATA_FIELD_DEF(int64_t, tailRepeatsCount);
TILING_DATA_FIELD_DEF(int64_t, repeatsSlice);
TILING_DATA_FIELD_DEF(int64_t, ubFactor);
TILING_DATA_FIELD_DEF(int64_t, totalRepeatSum);
TILING_DATA_FIELD_DEF(int64_t, averageRepeatTime);
TILING_DATA_FIELD_DEF_ARR(int64_t, REPEAT_INTERLEAVE_MERGED_DIM_LENGTH, mergedDims);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RepeatInterleave_201, RepeatInterleaveTilingKernelDataSmall)
REGISTER_TILING_DATA_CLASS(RepeatInterleave_202, RepeatInterleaveTilingKernelDataSmall)

BEGIN_TILING_DATA_DEF(RepeatInterleaveCumSumTilingData)
TILING_DATA_FIELD_DEF(int64_t, cumSumCoreNum);                 // cumSum使用核数
TILING_DATA_FIELD_DEF(int64_t, cumSumNormalCoreRepeatsCount);  // cumSum阶段正常核处理的repeats个数
TILING_DATA_FIELD_DEF(int64_t, cumSumTailCoreRepeatsCount);    // cumSum阶段尾核处理的repeats个数
TILING_DATA_FIELD_DEF(int64_t, cumSumNormalCoreLoops);         // cumSum阶段正常核循环次数
TILING_DATA_FIELD_DEF(int64_t, cumSumTailCoreLoops);           // cumSum阶段尾核循环次数
TILING_DATA_FIELD_DEF(int64_t, cumSumNormalUbFactors);         // cumSum阶段正常循环处理repeats个数
TILING_DATA_FIELD_DEF(int64_t, cumSumNormalCoreTailUbFactors); // cumSum阶段正常核尾循环处理repeats个数
TILING_DATA_FIELD_DEF(int64_t, cumSumTailCoreTailUbFactors);   // cumSum阶段尾核尾循环处理repeats个数

TILING_DATA_FIELD_DEF(int64_t, eachCoreBatchCount);
TILING_DATA_FIELD_DEF(int64_t, tailCoreBatchCount);
TILING_DATA_FIELD_DEF(int64_t, threadNumX);
TILING_DATA_FIELD_DEF(int64_t, threadNumY);
TILING_DATA_FIELD_DEF(int64_t, threadNumZ);
TILING_DATA_FIELD_DEF(int64_t, totalRepeatSum);

TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);             // 第二阶段repeat时使用的核数
TILING_DATA_FIELD_DEF(int64_t, batchCoreNum);            // batch轴开核数
TILING_DATA_FIELD_DEF(int64_t, repeatsCoreNum);          // repeats轴开核数
TILING_DATA_FIELD_DEF(int64_t, normalCoreOutputRepeats); // 正常repeat核处理的sumRepeat个数
TILING_DATA_FIELD_DEF(int64_t, tailCoreOutputRepeats);   // 尾repeat核处理的sumRepeat个数
TILING_DATA_FIELD_DEF_ARR(int64_t, REPEAT_INTERLEAVE_MERGED_DIM_LENGTH, mergedDims);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RepeatInterleave_4000, RepeatInterleaveCumSumTilingData)
REGISTER_TILING_DATA_CLASS(RepeatInterleave_4100, RepeatInterleaveCumSumTilingData)
REGISTER_TILING_DATA_CLASS(RepeatInterleave_4101, RepeatInterleaveCumSumTilingData)
REGISTER_TILING_DATA_CLASS(RepeatInterleave_5000, RepeatInterleaveCumSumTilingData)
REGISTER_TILING_DATA_CLASS(RepeatInterleave_5100, RepeatInterleaveCumSumTilingData)
REGISTER_TILING_DATA_CLASS(RepeatInterleave_5101, RepeatInterleaveCumSumTilingData)
REGISTER_TILING_DATA_CLASS(RepeatInterleave_15000, RepeatInterleaveCumSumTilingData)
REGISTER_TILING_DATA_CLASS(RepeatInterleave_15100, RepeatInterleaveCumSumTilingData)
REGISTER_TILING_DATA_CLASS(RepeatInterleave_15101, RepeatInterleaveCumSumTilingData)

class RepeatInterleaveTilingKernelNorm : public RepeatInterleaveBaseTiling {
public:
    explicit RepeatInterleaveTilingKernelNorm(gert::TilingContext* context) : RepeatInterleaveBaseTiling(context) {}
    ~RepeatInterleaveTilingKernelNorm() override {}

protected:
    int64_t eachCoreBatchCount_{0};
    int64_t tailCoreBatchCount_{0};
    int64_t normalCP_{0};
    int64_t tailCP_{0};
    int64_t cpSlice_{0}; // cpCoreNum
    int64_t repeatsCount_{-1};
    int64_t isSplitCP_{0};
    bool isSplitShape_{false};
    bool isSimtSplitRepeatSum_{false};
    bool isSimtSplitBatch_{false};
    bool isCumSumInUb_{false};
    int64_t threadNumX_{0};
    int64_t threadNumY_{0};
    int64_t threadNumZ_{0};
    int64_t normalRepeatsCount_{0};
    int64_t tailRepeatsCount_{0};
    int64_t repeatsSlice_{0};
    int64_t usedCoreNumBefore_{0};
    int64_t averageRepeatTime_{0};
    int64_t isUseInt64_{0};
    RepeatInterleaveTilingKernelDataNorm kernelTilingData_;
    RepeatInterleaveTilingKernelDataSmall kernelTilingDataSmall_;
    RepeatInterleaveCumSumTilingData cumSumTilingData_;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
    ge::graphStatus GetWorkspaceSize() override;
    void DumpCumSumTilingInfo();
    void SplitCP();
    void GetUbFactor();
    void UseInt64();
    void SplitRepeatsShape();
    void CalcSimtAddr();
    void setCumSumTilingData();
    bool isSplitRepeatSumSimt();
    ge::graphStatus SplitRepeatSimtWithCumSumTiling();
    bool isSplitBatchSimt();
    ge::graphStatus splitBatchSimtWithCumSumTiling();
};
} // namespace optiling
#endif
