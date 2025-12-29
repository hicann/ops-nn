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
 * \file max_pool_v3_small_kernel_nhwc_tiling.h
 * \brief
 */

#ifndef MAX_POOL_V3_SMALL_KERNEL_NHWC_TILING_H_
#define MAX_POOL_V3_SMALL_KERNEL_NHWC_TILING_H_

#include "max_pool_v3_tiling.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MaxPoolV3NHWCSmallKernelTilingData)
TILING_DATA_FIELD_DEF(int64_t, hInDim);
TILING_DATA_FIELD_DEF(int64_t, wInDim);
TILING_DATA_FIELD_DEF(int64_t, nOutDim);
TILING_DATA_FIELD_DEF(int64_t, hOutDim);
TILING_DATA_FIELD_DEF(int64_t, wOutDim);
TILING_DATA_FIELD_DEF(int64_t, kW);
TILING_DATA_FIELD_DEF(int64_t, kH);
TILING_DATA_FIELD_DEF(int64_t, sW);
TILING_DATA_FIELD_DEF(int64_t, sH);
TILING_DATA_FIELD_DEF(int64_t, tPad);
TILING_DATA_FIELD_DEF(int64_t, lPad);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, blockTail);
TILING_DATA_FIELD_DEF(int64_t, ubFactorN);
TILING_DATA_FIELD_DEF(int64_t, outUbFactorH);
TILING_DATA_FIELD_DEF(int64_t, outUbFactorW);
TILING_DATA_FIELD_DEF(int64_t, nLoop);
TILING_DATA_FIELD_DEF(int64_t, hLoop);
TILING_DATA_FIELD_DEF(int64_t, wLoop);
TILING_DATA_FIELD_DEF(int64_t, channels);
TILING_DATA_FIELD_DEF(int64_t, inUbSize);
TILING_DATA_FIELD_DEF(int64_t, outUbSize);
TILING_DATA_FIELD_DEF(int64_t, gatherMode);
TILING_DATA_FIELD_DEF(int64_t, copyMode);
TILING_DATA_FIELD_DEF(int64_t, onceCopyRow);
TILING_DATA_FIELD_DEF(int64_t, splitMode);
END_TILING_DATA_DEF;

// 400001 - no padding, 400002 - padding
REGISTER_TILING_DATA_CLASS(MaxPoolV3_400001, MaxPoolV3NHWCSmallKernelTilingData);
REGISTER_TILING_DATA_CLASS(MaxPoolV3_400002, MaxPoolV3NHWCSmallKernelTilingData);

class MaxPoolV3NHWCSmallKernelTiling : public MaxPoolV3BaseTiling {
public:
    explicit MaxPoolV3NHWCSmallKernelTiling(gert::TilingContext* context) : MaxPoolV3BaseTiling(context)
    {}
    ~MaxPoolV3NHWCSmallKernelTiling() override
    {}

private:
    void DoUBTiling();
    void DoUBTilingSingle();
    void DoBlockTiling();
    void SetTilingData();
    void InitializationVars();
    uint64_t GetTilingKey() const;
    bool IsCapable() override;
    bool IsBufferCapable();
    int64_t CalcBufferSize(int64_t inRows, int64_t inCols, int64_t outRows, int64_t outCols, bool isPadding);
    void CalcSplitMaxCols(int64_t minInRows);
    void CalcSplitMaxRows(int64_t maxInCols);
    void CalcSplitMaxBatch(int64_t oneBacthBuffer, int64_t oneBatchInputSize);
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
    void CalcGatherMode();
    void CalcCopyMode();
    MaxPoolV3NHWCSmallKernelTilingData tilingData;
    int64_t blockFactor_{0};
    int64_t blockTail_{0};
    int64_t ubFactorN_{0};
    int64_t outUbFactorH_{0};
    int64_t outUbFactorW_{0};
    int64_t nLoop_{0};
    int64_t hLoop_{0};
    int64_t wLoop_{0};
    bool isPadding_{false};
    int64_t oneBlockNum_{32};
    int64_t paraNum_{64};
    int64_t availableUb_{0};
    int64_t inUbSize_{0};
    int64_t outUbSize_{0};
    int64_t usedCoreNum_{0};
    int64_t gatherMode_{0};
    int64_t copyMode_{0};
    int64_t maxGatherScatterElm_{0};
    int64_t onceCopyRow_{1};
    int64_t splitMode_{0};
    bool isZero_{false};
};

} // namespace optiling

#endif