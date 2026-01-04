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
 * \file pool_3d_ncdhw_small_kernel_tiling.h
 * \brief
 */

#ifndef POOL_3D_NCDHW_SMALL_KERNEL_TILING_H_
#define POOL_3D_NCDHW_SMALL_KERNEL_TILING_H_

#include <array>

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"
#include "avg_pool_3d_tiling_common.h"
#include "max_pool_3d_tiling_common.h"
#include "pool_3d_tiling_common.h"

namespace optiling
{
BEGIN_TILING_DATA_DEF(Pool3DNcdhwSmallKernelTilingData)
TILING_DATA_FIELD_DEF(int64_t, dInDim);
TILING_DATA_FIELD_DEF(int64_t, hInDim);
TILING_DATA_FIELD_DEF(int64_t, wInDim);
TILING_DATA_FIELD_DEF(int64_t, nOutDim);
TILING_DATA_FIELD_DEF(int64_t, dOutDim);
TILING_DATA_FIELD_DEF(int64_t, hOutDim);
TILING_DATA_FIELD_DEF(int64_t, wOutDim);
TILING_DATA_FIELD_DEF(int64_t, channel);
TILING_DATA_FIELD_DEF(int64_t, kD);
TILING_DATA_FIELD_DEF(int64_t, kH);
TILING_DATA_FIELD_DEF(int64_t, kW);
TILING_DATA_FIELD_DEF(int64_t, sD);
TILING_DATA_FIELD_DEF(int64_t, sH);
TILING_DATA_FIELD_DEF(int64_t, sW);
TILING_DATA_FIELD_DEF(int64_t, dD);
TILING_DATA_FIELD_DEF(int64_t, dH);
TILING_DATA_FIELD_DEF(int64_t, dW);
TILING_DATA_FIELD_DEF(int64_t, frontPad);
TILING_DATA_FIELD_DEF(int64_t, backendPad);
TILING_DATA_FIELD_DEF(int64_t, topPad);
TILING_DATA_FIELD_DEF(int64_t, bottomPad);
TILING_DATA_FIELD_DEF(int64_t, leftPad);
TILING_DATA_FIELD_DEF(int64_t, rightPad);
TILING_DATA_FIELD_DEF(int64_t, divisor);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, blockTail);
TILING_DATA_FIELD_DEF(int64_t, ubFactorN);
TILING_DATA_FIELD_DEF(int64_t, outUbFactorD);
TILING_DATA_FIELD_DEF(int64_t, outUbFactorH);
TILING_DATA_FIELD_DEF(int64_t, outUbFactorW);
TILING_DATA_FIELD_DEF(int64_t, nLoop);
TILING_DATA_FIELD_DEF(int64_t, dLoop);
TILING_DATA_FIELD_DEF(int64_t, hLoop);
TILING_DATA_FIELD_DEF(int64_t, wLoop);
TILING_DATA_FIELD_DEF(int64_t, inUbSize);
TILING_DATA_FIELD_DEF(int64_t, outUbSize);
TILING_DATA_FIELD_DEF(int64_t, divisorUbSize);
TILING_DATA_FIELD_DEF(int32_t, gatherMode);
TILING_DATA_FIELD_DEF(int32_t, splitMode);
TILING_DATA_FIELD_DEF(int32_t, divisorMode);
TILING_DATA_FIELD_DEF(int32_t, realCalcDivisor);
TILING_DATA_FIELD_DEF(int32_t, useTraiTwo);
TILING_DATA_FIELD_DEF(int32_t, sparseMode);
END_TILING_DATA_DEF;

// 300001 - no padding, 300002 - padding
REGISTER_TILING_DATA_CLASS(MaxPool3D_300001, Pool3DNcdhwSmallKernelTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3D_300002, Pool3DNcdhwSmallKernelTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3D_300004, Pool3DNcdhwSmallKernelTilingData);

// 300001 - no padding, 300002 - padding
REGISTER_TILING_DATA_CLASS(AvgPool3D_300001, Pool3DNcdhwSmallKernelTilingData);
REGISTER_TILING_DATA_CLASS(AvgPool3D_300002, Pool3DNcdhwSmallKernelTilingData);
REGISTER_TILING_DATA_CLASS(AvgPool3D_300003, Pool3DNcdhwSmallKernelTilingData);
REGISTER_TILING_DATA_CLASS(AvgPool3D_300004, Pool3DNcdhwSmallKernelTilingData);

class Pool3DNcdhwSmallKernelTiling : public TilingBaseClass
{
public:
    explicit Pool3DNcdhwSmallKernelTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {
    }

    ~Pool3DNcdhwSmallKernelTiling() override
    {
    }
    
protected:
    void DoUBTiling();
    void DoUBTilingSingle();
    void DoBlockTiling();
    void SetTilingData();
    void InitializationVars();
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;
    bool IsBufferCapable();
    int64_t CalcBufferSize(int64_t inDepths, int64_t inRows, int64_t inCols, int64_t outDepths, int64_t outRows, int64_t outCols,
                                                   bool isPadding, bool needCalcDivisorBuffer=false);
    void CalcSplitMaxCols(int64_t minInDepths, int64_t minInRows);
    void CalcSplitMaxRows(int64_t minInDepth, int64_t maxInCols);
    void CalcSplitMaxBatch(int64_t oneBacthBuffer, int64_t oneBatchInputSize);
    void CalcSplitMaxDepth(int64_t maxInRows, int64_t maxInCols);
    void CalcDivsiorUbSize(bool isPad);
    void CalcKernelAndPadInfo() ;
    void CalcEnableSplit();
    void CalcSparseMode();
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    void DumpTilingInfo() override;
    void CalcGatherMode();
    void CalcDivisorMode();
    void CalcDivisor();
    int64_t CalcInputSizeByOutput(int64_t outputSize, uint32_t dim, bool isSparse);
    int64_t CalcOutputSizeByInput(int64_t inputSize, uint32_t dim, bool isSparse);
    int64_t CalxMaxInputSize(int64_t inputSize, int64_t orgInputAndPad, bool isSparse);
    Pool3DNcdhwSmallKernelTilingData tilingData_;
    Pool3DInputInfo inputData_;
    uint64_t coreNum_ = 1;
    uint64_t ubSize_ = 0;
    int64_t blockFactor_{0};
    int64_t blockTail_{0};
    int64_t ubFactorN_{0};
    int64_t outUbFactorD_{0};
    int64_t outUbFactorH_{0};
    int64_t outUbFactorW_{0};
    int64_t nLoop_{0};
    int64_t dLoop_{0};
    int64_t hLoop_{0};
    int64_t wLoop_{0};
    bool isPadding_{false};
    int64_t oneBlockNum_{32};
    int64_t paraNum_{64};
    int64_t availableUb_{0};
    int64_t inUbSize_{0};
    int64_t outUbSize_{0};
    int64_t divisorUbSize_{0};
    int64_t usedCoreNum_{0};
    int64_t channel_{1};
    int32_t gatherMode_{0};
    int32_t divisorMode_{0};
    int64_t maxGatherScatterElm_{0};
    int32_t splitMode_{0};
    int64_t divisor_{1};
    int32_t realCalcDivisor_{0};
    int32_t useTraiTwo_{0};
    int32_t sparseMode_{0};
    bool needDivsiorUb_ = false;
    bool allNeedInPad_ = false;
    bool needCalcDivisorBuffer_ = false;
    std::array<int64_t, DHW_DIMS> effectiveKsize_;
    bool isZero_{false};
    bool isSparseW_{false};
    bool isSparseH_{false};
    bool isSparseD_{false};
    bool enableSplitBatch_{true};
    bool enableSplitD_{true};
    bool enableSplitH_{true};
    bool enableSplitW_{true};
};

class AvgPool3DNcdhwSmallKernelTiling : public Pool3DNcdhwSmallKernelTiling
{
public:
    explicit AvgPool3DNcdhwSmallKernelTiling(gert::TilingContext* context) : Pool3DNcdhwSmallKernelTiling(context)
    {
    }
    ~AvgPool3DNcdhwSmallKernelTiling() override
    {
    }

private:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
};

class MaxPool3DNcdhwSmallKernelTiling : public Pool3DNcdhwSmallKernelTiling
{
public:
    explicit MaxPool3DNcdhwSmallKernelTiling(gert::TilingContext* context) : Pool3DNcdhwSmallKernelTiling(context)
    {
    }
    ~MaxPool3DNcdhwSmallKernelTiling() override
    {
    }

private:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
};
}  // namespace optiling
#endif