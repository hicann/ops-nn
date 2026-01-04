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
 * \file pool_3d_simt_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_POOL_3D_SIMT_TILING_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_POOL_3D_SIMT_TILING_H_

#include <array>

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"
#include "avg_pool_3d_tiling_common.h"
#include "max_pool_3d_tiling_common.h"

namespace optiling
{
BEGIN_TILING_DATA_DEF(Pool3DSimtTilingData)
TILING_DATA_FIELD_DEF(int64_t, nDim);
TILING_DATA_FIELD_DEF(int64_t, cDim);
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
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AvgPool3D_911100, Pool3DSimtTilingData);
REGISTER_TILING_DATA_CLASS(AvgPool3D_911101, Pool3DSimtTilingData);
REGISTER_TILING_DATA_CLASS(AvgPool3D_911110, Pool3DSimtTilingData);
REGISTER_TILING_DATA_CLASS(AvgPool3D_911111, Pool3DSimtTilingData);

REGISTER_TILING_DATA_CLASS(MaxPool3D_911100, Pool3DSimtTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3D_911101, Pool3DSimtTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3D_911110, Pool3DSimtTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3D_911111, Pool3DSimtTilingData);

class Pool3DSimtTiling : public TilingBaseClass
{
public:
    explicit Pool3DSimtTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {
    }

    ~Pool3DSimtTiling() override
    {
    }

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
    void SetTilingData();

public:
    Pool3DSimtTilingData tiling_;
    Pool3DInputInfo inputData;
    ge::DataType dtype = ge::DataType::DT_FLOAT;
    uint64_t coreNum = 1;
    uint64_t ubSize = 0;
};

class AvgPool3DSimtTiling : public Pool3DSimtTiling
{
public:
    explicit AvgPool3DSimtTiling(gert::TilingContext* context) : Pool3DSimtTiling(context)
    {
    }
    ~AvgPool3DSimtTiling() override
    {
    }

private:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
};

class MaxPool3DSimtTiling : public Pool3DSimtTiling
{
public:
    explicit MaxPool3DSimtTiling(gert::TilingContext* context) : Pool3DSimtTiling(context)
    {
    }
    ~MaxPool3DSimtTiling() override
    {
    }

private:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
};

}
#endif