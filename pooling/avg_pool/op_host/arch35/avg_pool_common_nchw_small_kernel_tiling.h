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
 * \file avg_pool_common_nchw_small_kernel_tiling.h
 * \brief
 */

#ifndef AVG_POOL_COMMON_NCHW_SMALL_KERNEL_TILING_H_
#define AVG_POOL_COMMON_NCHW_SMALL_KERNEL_TILING_H_

#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "avg_pool_common.h"
#include "pooling/avg_pool/op_host/avg_pool_tiling_common.h"
#include "pooling/avg_pool_v2/op_host/arch35/avg_pool_v2_common_tiling.h"
#include "pooling/avg_pool/op_kernel/arch35/avg_pool_struct.h"

namespace optiling
{

class AvgPoolCommonNCHWSmallKernelTiling : public TilingBaseClass
{
public:
    explicit AvgPoolCommonNCHWSmallKernelTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {
    }
    ~AvgPoolCommonNCHWSmallKernelTiling() override
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

private:
    void InitializationVars();
    bool IsBufferCapable();
    void DoUBTilingSingle();
    void DoBlockTiling();
    int64_t CalcBufferSize(int64_t inRows, int64_t inCols, int64_t outRows,
                           int64_t outCols, bool isPadding);
    void CalcSplitMaxCols(int64_t minInRows);
    void CalcSplitMaxRows(int64_t maxInCols);
    void CalcSplitMaxBatch(int64_t oneBacthBuffer, int64_t oneBatchInputSize);
    
    void CalcGatherMode();
    void CalcDivsiorUbSize(bool isPad);
    void CalcDivisorMode();
    void CalcDivisor();
    void CalcCopyMode();

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
    int64_t indiceUbSize_{0};
    int64_t divisorUbSize_{0};
    int64_t usedCoreNum_{0};
    int64_t gatherMode_{0};
    int64_t copyMode_{0};
    int64_t maxGatherScatterElm_{0};
    int64_t onceCopyRow_{1};
    int64_t splitMode_{0};
    bool isZero_{false};
    bool needCalcDivisorBuffer_ = false;
    int64_t realCalcDivisor_{0};
    int64_t divisorMode_{0};
    bool needDivsiorUb_ = false;
    bool allNeedInPad_ = false;
    int64_t divisor_{1};

public:
    AvgPoolInputInfo inputData;
    ge::DataType dtype = ge::DataType::DT_FLOAT;
    uint64_t dtypeSize = 0;
    uint64_t coreNum = 1;
    uint64_t ubSize = 0;
    int32_t nDim_ = 0;  // NCHW -> N
    int32_t cDim_ = 1;  // NCHW -> C
    int32_t hDim_ = 2;  // NCHW -> H
    int32_t wDim_ = 3;  // NCHW -> W
};

class AvgPoolNCHWSmallKernelTiling : public AvgPoolCommonNCHWSmallKernelTiling
{
public:
    explicit AvgPoolNCHWSmallKernelTiling(gert::TilingContext* context) : AvgPoolCommonNCHWSmallKernelTiling(context)
    {
    }
    ~AvgPoolNCHWSmallKernelTiling() override
    {
    }

private:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
};

class AvgPoolV2NCHWSmallKernelTiling : public AvgPoolCommonNCHWSmallKernelTiling
{
public:
    explicit AvgPoolV2NCHWSmallKernelTiling(gert::TilingContext* context) : AvgPoolCommonNCHWSmallKernelTiling(context)
    {
    }
    ~AvgPoolV2NCHWSmallKernelTiling() override
    {
    }

private:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
};

}  // namespace optiling

#endif