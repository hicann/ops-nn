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

class RepeatInterleaveTilingKernelNorm : public RepeatInterleaveBaseTiling {
public:
    explicit RepeatInterleaveTilingKernelNorm(gert::TilingContext* context) : RepeatInterleaveBaseTiling(context)
    {}
    ~RepeatInterleaveTilingKernelNorm() override
    {}

protected:
    int64_t eachCoreBatchCount_{0};
    int64_t tailCoreBatchCount_{0};
    int64_t normalCP_{0};
    int64_t tailCP_{0};
    int64_t cpSlice_{0};
    int64_t ubFactor_{0};
    int64_t repeatsCount_{-1};
    int64_t isSplitCP_{0};
    int64_t totalRepeatSum_{0};
    bool isSplitShape_{false};
    int64_t normalRepeatsCount_{0};
    int64_t tailRepeatsCount_{0};
    int64_t repeatsSlice_{0};
    int64_t usedCoreNumBefore_{0};
    int64_t averageRepeatTime_{0};
    int64_t isUseInt64_{0};
    RepeatInterleaveTilingKernelDataNorm kernelTilingData_;
    RepeatInterleaveTilingKernelDataSmall kernelTilingDataSmall_;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
    void SplitCP();
    void GetUbFactor();
    void UseInt64();
    void SplitRepeatsShape();
};
} // namespace optiling
#endif
