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
 * \file pool_3d_one_kize_tiling.h
 * \brief
 */

#ifndef POOL_3D_ONE_KSIZE_TILING_H_
#define POOL_3D_ONE_KSIZE_TILING_H_

#include <array>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"
#include "avg_pool_3d_tiling_common.h"
#include "max_pool_3d_tiling_common.h"
#include "pool_3d_tiling_common.h"

namespace optiling
{
BEGIN_TILING_DATA_DEF(Pool3DOneKsizeTilingData)
TILING_DATA_FIELD_DEF(int64_t, inUbSize);
TILING_DATA_FIELD_DEF(int64_t, dataFormat);
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
TILING_DATA_FIELD_DEF(int64_t, channel);
TILING_DATA_FIELD_DEF(int64_t, dInDim);
TILING_DATA_FIELD_DEF(int64_t, hInDim);
TILING_DATA_FIELD_DEF(int64_t, wInDim);
TILING_DATA_FIELD_DEF(int64_t, nOutDim);
TILING_DATA_FIELD_DEF(int64_t, dOutDim);
TILING_DATA_FIELD_DEF(int64_t, hOutDim);
TILING_DATA_FIELD_DEF(int64_t, wOutDim);
TILING_DATA_FIELD_DEF(int64_t, sD);
TILING_DATA_FIELD_DEF(int64_t, sH);
TILING_DATA_FIELD_DEF(int64_t, sW);
TILING_DATA_FIELD_DEF(int64_t, divisor);
END_TILING_DATA_DEF;

// 100001 - ksize one
REGISTER_TILING_DATA_CLASS(AvgPool3D_100001, Pool3DOneKsizeTilingData);
REGISTER_TILING_DATA_CLASS(MaxPool3D_100001, Pool3DOneKsizeTilingData);

class Pool3DOneKsizeTiling : public TilingBaseClass
{
public:
    explicit Pool3DOneKsizeTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {
    }

    ~Pool3DOneKsizeTiling() override
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
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    void DumpTilingInfo() override;
    void CalcDivisor();
    Pool3DOneKsizeTilingData tilingData_;
    Pool3DInputInfo inputData_;
    uint64_t coreNum_ = 1;
    uint64_t ubSize_ = 0;
    int64_t blockFactor_{0};
    int64_t blockTail_{0};
    int64_t ubFactorN_{0};
    int64_t outUbFactorD_{0};
    int64_t outUbFactorH_{0};
    int64_t outUbFactorW_{0};
    int64_t nLoop_{1};
    int64_t dLoop_{1};
    int64_t hLoop_{1};
    int64_t wLoop_{1};
    int64_t oneBlockNum_{32};
    int64_t availableUb_{0};
    int64_t inUbSize_{0};
    int64_t outUbSize_{0};
    int64_t usedCoreNum_{0};
    int64_t divisor_{0};
    int64_t dataFormat_{0};
    bool isZero_{false};
};

class AvgPool3DOneKsizeTiling : public Pool3DOneKsizeTiling
{
public:
    explicit AvgPool3DOneKsizeTiling(gert::TilingContext* context) : Pool3DOneKsizeTiling(context)
    {
    }
    ~AvgPool3DOneKsizeTiling() override
    {
    }

private:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
};

class MaxPool3DOneKsizeTiling : public Pool3DOneKsizeTiling
{
public:
    explicit MaxPool3DOneKsizeTiling(gert::TilingContext* context) : Pool3DOneKsizeTiling(context)
    {
    }
    ~MaxPool3DOneKsizeTiling() override
    {
    }

private:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
};

}  // namespace optiling

#endif