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
 * \file pool_3d_ncdhw_big_kernel_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_POOL_3D_BIG_KERNEL_TILING_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_POOL_3D_BIG_KERNEL_TILING_H_

#include <array>

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"
#include "avg_pool_3d_tiling_common.h"
#include "max_pool_3d_tiling_common.h"

namespace optiling
{

BEGIN_TILING_DATA_DEF(Pool3DNcdhwBigKernelTilingData)
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
TILING_DATA_FIELD_DEF(int64_t, dD);
TILING_DATA_FIELD_DEF(int64_t, dH);
TILING_DATA_FIELD_DEF(int64_t, dW);
TILING_DATA_FIELD_DEF(int64_t, fPad);
TILING_DATA_FIELD_DEF(int64_t, bkPad);
TILING_DATA_FIELD_DEF(int64_t, tPad);
TILING_DATA_FIELD_DEF(int64_t, bPad);
TILING_DATA_FIELD_DEF(int64_t, lPad);
TILING_DATA_FIELD_DEF(int64_t, rPad);
TILING_DATA_FIELD_DEF(int64_t, divisorOverride);
TILING_DATA_FIELD_DEF(int64_t, countIncludePad);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, blockTail);
TILING_DATA_FIELD_DEF(int64_t, totalIdx);
TILING_DATA_FIELD_DEF(int64_t, coreNums);
TILING_DATA_FIELD_DEF(int64_t, maxCount);
TILING_DATA_FIELD_DEF(int64_t, isSigOut);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaxPool3D_511110, Pool3DNcdhwBigKernelTilingData);
REGISTER_TILING_DATA_CLASS(AvgPool3D_511110, Pool3DNcdhwBigKernelTilingData);

class Pool3DNcdhwBigKernelTiling : public TilingBaseClass
{
public:
    explicit Pool3DNcdhwBigKernelTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {
    }

    ~Pool3DNcdhwBigKernelTiling() override
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
    Pool3DNcdhwBigKernelTilingData tiling;
    Pool3DInputInfo inputData_;
    ge::DataType dtype = ge::DataType::DT_FLOAT;
    uint64_t coreNum_ = 1;
    uint64_t ubSize_ = 0;

    int64_t totalIdx_{0};
    int64_t blockFactor_{0};
    int64_t blockTail_{0};
    int64_t maxCount_{0};
    int64_t isSigOut_{0};
    int64_t coreNums_{0};
};

class AvgPool3DNcdhwBigKernelTiling : public Pool3DNcdhwBigKernelTiling
{
public:
    explicit AvgPool3DNcdhwBigKernelTiling(gert::TilingContext* context) : Pool3DNcdhwBigKernelTiling(context)
    {
    }
    ~AvgPool3DNcdhwBigKernelTiling() override
    {
    }

private:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
};

class MaxPool3DNcdhwBigKernelTiling : public Pool3DNcdhwBigKernelTiling
{
public:
    explicit MaxPool3DNcdhwBigKernelTiling(gert::TilingContext* context) : Pool3DNcdhwBigKernelTiling(context)
    {
    }
    ~MaxPool3DNcdhwBigKernelTiling() override
    {
    }

private:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
};
}  // namespace optiling

#endif