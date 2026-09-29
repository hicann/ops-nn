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
 * \file repeat_interleave_tiling_repeat.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_REPEAT_INTERLEAVE_TILING_REPEAT_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_REPEAT_INTERLEAVE_TILING_REPEAT_H_

#include "repeat_interleave_tiling_base.h"
#include "repeat_interleave_tiling_normal.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(RepeatInterleaveTilingKernelRepeat)
TILING_DATA_FIELD_DEF(int64_t, cumSumCoreNum);                 // cumSum使用核数
TILING_DATA_FIELD_DEF(int64_t, cumSumNormalCoreRepeatsCount);  // cumSum阶段正常核处理的repeats个数
TILING_DATA_FIELD_DEF(int64_t, cumSumTailCoreRepeatsCount);    // cumSum阶段尾核处理的repeats个数
TILING_DATA_FIELD_DEF(int64_t, cumSumNormalCoreLoops);         // cumSum阶段正常核循环次数
TILING_DATA_FIELD_DEF(int64_t, cumSumTailCoreLoops);           // cumSum阶段尾核循环次数
TILING_DATA_FIELD_DEF(int64_t, cumSumNormalUbFactors);         // cumSum阶段正常循环处理repeats个数
TILING_DATA_FIELD_DEF(int64_t, cumSumNormalCoreTailUbFactors); // cumSum阶段正常核尾循环处理repeats个数
TILING_DATA_FIELD_DEF(int64_t, cumSumTailCoreTailUbFactors);   // cumSum阶段尾核尾循环处理repeats个数
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, eachCoreCpCount);
TILING_DATA_FIELD_DEF(int64_t, tailCoreCpCount);
TILING_DATA_FIELD_DEF(int64_t, cpCountInUb);
TILING_DATA_FIELD_DEF(int64_t, isSplitCP);
TILING_DATA_FIELD_DEF(int64_t, cpSliceNum);
TILING_DATA_FIELD_DEF(int64_t, cpSliceFactorAlign);
TILING_DATA_FIELD_DEF(int64_t, cpSliceFactorTail);
TILING_DATA_FIELD_DEF(int64_t, isSplitCPForKernel);
TILING_DATA_FIELD_DEF(int64_t, cpSplitBlocks);
TILING_DATA_FIELD_DEF(int64_t, totalRepeatSum);
TILING_DATA_FIELD_DEF_ARR(int64_t, REPEAT_INTERLEAVE_MERGED_DIM_LENGTH, mergedDims);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RepeatInterleave_3000, RepeatInterleaveTilingKernelRepeat)
REGISTER_TILING_DATA_CLASS(RepeatInterleave_3001, RepeatInterleaveTilingKernelRepeat)
REGISTER_TILING_DATA_CLASS(RepeatInterleave_3011, RepeatInterleaveTilingKernelRepeat)
REGISTER_TILING_DATA_CLASS(RepeatInterleave_3101, RepeatInterleaveTilingKernelRepeat)

class RepeatInterleaveTilingKernelRepeatTiling : public RepeatInterleaveBaseTiling {
public:
    explicit RepeatInterleaveTilingKernelRepeatTiling(gert::TilingContext* context)
        : RepeatInterleaveBaseTiling(context)
    {}
    ~RepeatInterleaveTilingKernelRepeatTiling() override {}

protected:
    int64_t eachCoreCpCount_{0};
    int64_t tailCoreCpCount_{0};
    int64_t cpCountInUb_{0};
    int64_t isSplitCP_{0};
    int64_t cpSliceNum_{1};
    int64_t cpSliceFactorAlign_{0};
    int64_t cpSliceFactorTail_{0};
    int64_t isSplitCPForKernel_{0};
    int64_t cpSplitBlocks_{1};
    int64_t singleBufSize_{0};
    RepeatInterleaveTilingKernelRepeat kernelTilingRepeatData_;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
    void SetRepeatKernelTilingData();
    void GetUbFactor();
    void SplitCPForKernelUbFactor();
    void SplitCPUbFactor();
};
} // namespace optiling
#endif
