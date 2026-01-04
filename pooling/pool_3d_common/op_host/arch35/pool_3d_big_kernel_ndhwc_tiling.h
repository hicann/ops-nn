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
 * \file pool_3d_big_kernel_ndhwc_tiling.h
 * \brief big kernel imply for pool_3d ndhwc format
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_POOL_3D_BIG_KERNEL_NDHWC_TILING_BASE_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_POOL_3D_BIG_KERNEL_NDHWC_TILING_BASE_H_

#include <array>

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"
#include "avg_pool_3d_tiling_common.h"
#include "max_pool_3d_tiling_common.h"

namespace optiling
{
    
BEGIN_TILING_DATA_DEF(Pool3DBigKernelNDHWCTilingData)
TILING_DATA_FIELD_DEF(int64_t, dInDim);
TILING_DATA_FIELD_DEF(int64_t, hInDim);
TILING_DATA_FIELD_DEF(int64_t, wInDim);
TILING_DATA_FIELD_DEF(int64_t, dOutDim);
TILING_DATA_FIELD_DEF(int64_t, hOutDim);
TILING_DATA_FIELD_DEF(int64_t, wOutDim);
TILING_DATA_FIELD_DEF(int64_t, kD);
TILING_DATA_FIELD_DEF(int64_t, kH);
TILING_DATA_FIELD_DEF(int64_t, kW);
TILING_DATA_FIELD_DEF(int64_t, sD);
TILING_DATA_FIELD_DEF(int64_t, sH);
TILING_DATA_FIELD_DEF(int64_t, sW);
TILING_DATA_FIELD_DEF(int64_t, fPad);
TILING_DATA_FIELD_DEF(int64_t, backPad);
TILING_DATA_FIELD_DEF(int64_t, tPad);
TILING_DATA_FIELD_DEF(int64_t, bottomPad);
TILING_DATA_FIELD_DEF(int64_t, lPad);
TILING_DATA_FIELD_DEF(int64_t, rPad);
TILING_DATA_FIELD_DEF(int64_t, dD);
TILING_DATA_FIELD_DEF(int64_t, dH);
TILING_DATA_FIELD_DEF(int64_t, dW);
TILING_DATA_FIELD_DEF(int64_t, divisorOverride);
TILING_DATA_FIELD_DEF(int64_t, countIncludePad);
TILING_DATA_FIELD_DEF(int64_t, channel);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, blockTail);
TILING_DATA_FIELD_DEF(int64_t, totalIdx);
TILING_DATA_FIELD_DEF(int64_t, coreNums);
TILING_DATA_FIELD_DEF(int64_t, inUbSize);
TILING_DATA_FIELD_DEF(int64_t, outUbSize);
TILING_DATA_FIELD_DEF(int64_t, isSigOut);
TILING_DATA_FIELD_DEF(int64_t, tilingMode);
TILING_DATA_FIELD_DEF(int64_t, onceOutNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaxPool3D_411110, Pool3DBigKernelNDHWCTilingData);
REGISTER_TILING_DATA_CLASS(AvgPool3D_411110, Pool3DBigKernelNDHWCTilingData);

class Pool3DBigKernelNDHWCTiling : public TilingBaseClass
{
public:
    explicit Pool3DBigKernelNDHWCTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {
    }

    ~Pool3DBigKernelNDHWCTiling() override
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
    Pool3DBigKernelNDHWCTilingData tiling_;
    uint64_t totalCoreNum_ = 1;
    uint64_t ubSize_ = 0;
    Pool3DInputInfo inputData_;
    ge::DataType dtype_ = ge::DataType::DT_FLOAT;
    int64_t totalIdx_{0};
    int64_t blockFactor_{0};
    int64_t blockTail_{0};
    int64_t coreNums_{0};
    int64_t inUbSize_{0};
    int64_t outUbSize_{0};
    int64_t isSigOut_{0};
    int64_t tilingMode_{0};
    int64_t onceOutNum_{1};
    bool isMaxPool3D_{false};
};

class AvgPool3DBigKernelNDHWCTiling : public Pool3DBigKernelNDHWCTiling
{
public:
    explicit AvgPool3DBigKernelNDHWCTiling(gert::TilingContext* context) : Pool3DBigKernelNDHWCTiling(context)
    {
    }
    ~AvgPool3DBigKernelNDHWCTiling() override
    {
    }

private:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
};

class MaxPool3DBigKernelNDHWCTiling : public Pool3DBigKernelNDHWCTiling
{
public:
    explicit MaxPool3DBigKernelNDHWCTiling(gert::TilingContext* context) : Pool3DBigKernelNDHWCTiling(context)
    {
    }
    ~MaxPool3DBigKernelNDHWCTiling() override
    {
    }

private:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
};

}  // namespace optiling

#endif