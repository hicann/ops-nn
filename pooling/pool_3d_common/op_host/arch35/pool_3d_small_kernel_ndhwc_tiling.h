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
 * \file pool_3d_small_kernel_ndhwc_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_POOL_3D_SMALL_KERNEL_NDHWC_TILING_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_POOL_3D_SMALL_KERNEL_NDHWC_TILING_H_

#include <array>

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "avg_pool_3d_tiling_common.h"
#include "max_pool_3d_tiling_common.h"

namespace optiling
{
    
BEGIN_TILING_DATA_DEF(Pool3DSmallKernelNDHWCTilingData)
TILING_DATA_FIELD_DEF(int64_t, dInDim);
TILING_DATA_FIELD_DEF(int64_t, hInDim);
TILING_DATA_FIELD_DEF(int64_t, wInDim);
TILING_DATA_FIELD_DEF(int64_t, nOutDim);
TILING_DATA_FIELD_DEF(int64_t, dOutDim);
TILING_DATA_FIELD_DEF(int64_t, hOutDim);
TILING_DATA_FIELD_DEF(int64_t, wOutDim);
TILING_DATA_FIELD_DEF(int64_t, kD);
TILING_DATA_FIELD_DEF(int64_t, kH);
TILING_DATA_FIELD_DEF(int64_t, kW);
TILING_DATA_FIELD_DEF(int64_t, sD);
TILING_DATA_FIELD_DEF(int64_t, sH);
TILING_DATA_FIELD_DEF(int64_t, sW);
TILING_DATA_FIELD_DEF(int64_t, dD);
TILING_DATA_FIELD_DEF(int64_t, dH);
TILING_DATA_FIELD_DEF(int64_t, dW);
TILING_DATA_FIELD_DEF(int64_t, fPad);
TILING_DATA_FIELD_DEF(int64_t, backPad);
TILING_DATA_FIELD_DEF(int64_t, tPad);
TILING_DATA_FIELD_DEF(int64_t, bottomPad);
TILING_DATA_FIELD_DEF(int64_t, lPad);
TILING_DATA_FIELD_DEF(int64_t, rPad);
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
TILING_DATA_FIELD_DEF(int64_t, channels);
TILING_DATA_FIELD_DEF(int64_t, inUbSize);
TILING_DATA_FIELD_DEF(int64_t, outUbSize);
TILING_DATA_FIELD_DEF(int64_t, divisorUbSize);
TILING_DATA_FIELD_DEF(int64_t, gatherMode);
TILING_DATA_FIELD_DEF(int64_t, copyMode);
TILING_DATA_FIELD_DEF(int64_t, onceCopyRow);
TILING_DATA_FIELD_DEF(int64_t, splitMode);
TILING_DATA_FIELD_DEF(int64_t, divisor);
TILING_DATA_FIELD_DEF(int64_t, divisorMode);
TILING_DATA_FIELD_DEF(int64_t, realCalcDivisor);
TILING_DATA_FIELD_DEF(int64_t, useTraiTwo);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AvgPool3D_200001, Pool3DSmallKernelNDHWCTilingData);
REGISTER_TILING_DATA_CLASS(AvgPool3D_211110, Pool3DSmallKernelNDHWCTilingData);
REGISTER_TILING_DATA_CLASS(AvgPool3D_222220, Pool3DSmallKernelNDHWCTilingData);
REGISTER_TILING_DATA_CLASS(AvgPool3D_222221, Pool3DSmallKernelNDHWCTilingData);
REGISTER_TILING_DATA_CLASS(AvgPool3D_211111, Pool3DSmallKernelNDHWCTilingData);
REGISTER_TILING_DATA_CLASS(AvgPool3D_222222, Pool3DSmallKernelNDHWCTilingData);

REGISTER_TILING_DATA_CLASS(MaxPool3D_200001, Pool3DSmallKernelNDHWCTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3D_211110, Pool3DSmallKernelNDHWCTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3D_222220, Pool3DSmallKernelNDHWCTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3D_222221, Pool3DSmallKernelNDHWCTilingData);

class Pool3DSmallKernelNDHWCTiling : public TilingBaseClass
{
public:
    explicit Pool3DSmallKernelNDHWCTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {
    }

    ~Pool3DSmallKernelNDHWCTiling() override
    {
    }

protected:
    void DoUBTiling();
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
    void SetTilingData();

public:
    Pool3DSmallKernelNDHWCTilingData tilingData_;
    Pool3DInputInfo inputData_;
    ge::DataType dtype = ge::DataType::DT_FLOAT;
    uint64_t coreNum_ = 1;
    uint64_t ubSize_ = 0;

private:
    
    void InitializationVars();
    bool IsBufferCapable();
    void DoUBTilingSingle();
    void DoBlockTiling();
    int64_t CalcBufferSize(int64_t inDeps, int64_t inRows, int64_t inCols, int64_t outDeps, int64_t outRows,
                           int64_t outCols, bool isPadding, bool needCalCDivisorBuffer = false);
    void CalcSplitMaxCols(int64_t minInRows, int64_t minInDeps);
    void CalcSplitMaxRows(int64_t maxInCols, int64_t minInDeps);
    void CalcSplitMaxDeps(int64_t maxInCols, int64_t maxInRows);
    void CalcSplitMaxBatch(int64_t oneBacthBuffer, int64_t oneBatchInputSize);
    void CalcGatherMode();
    void CalcDivsiorUbSize(bool isPad);
    void CalcDivisorMode();
    void CalcDivisor();

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
    int64_t gatherMode_{0};
    int64_t copyMode_{0};
    int64_t maxGatherScatterElm_{0};
    int64_t onceCopyRow_{1};
    int64_t splitMode_{0};
    bool isZero_{false};
    int64_t effectiveKsize_[3] = {1};
    bool needCalcDivisorBuffer_ = false;
    int64_t realCalcDivisor_{0};
    int64_t divisorMode_{0};
    int64_t useTraiTwo_{0};
    bool needDivsiorUb_ = false;
    bool allNeedInPad_ = false;
    int64_t divisor_{1};
};

class AvgPool3DSmallKernelNDHWCTiling : public Pool3DSmallKernelNDHWCTiling
{
public:
    explicit AvgPool3DSmallKernelNDHWCTiling(gert::TilingContext* context) : Pool3DSmallKernelNDHWCTiling(context)
    {
    }
    ~AvgPool3DSmallKernelNDHWCTiling() override
    {
    }

private:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
};

class MaxPool3DSmallKernelNDHWCTiling : public Pool3DSmallKernelNDHWCTiling
{
public:
    explicit MaxPool3DSmallKernelNDHWCTiling(gert::TilingContext* context) : Pool3DSmallKernelNDHWCTiling(context)
    {
    }
    ~MaxPool3DSmallKernelNDHWCTiling() override
    {
    }

private:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
};

}  // namespace optiling

#endif