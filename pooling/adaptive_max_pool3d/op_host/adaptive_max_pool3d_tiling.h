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
 * \file adaptive_max_pool3d_tiling.h
 * \brief
 * ATTENTION: MAKE SURE 'BEGIN_TILING_DATA_DEF' STAY IN THE SAME LINE (37) (41) USING BLANK LINES.
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 */

#ifndef ADAPTIVE_MAX_POOL3D_TILING_H
#define ADAPTIVE_MAX_POOL3D_TILING_H

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include "util/math_util.h"

namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;
BEGIN_TILING_DATA_DEF(AdaptiveMaxPool3dTilingData)
TILING_DATA_FIELD_DEF(uint64_t, N);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(AdaptiveMaxPool3dSmallPoolTilingData)
TILING_DATA_FIELD_DEF(uint64_t, N);
TILING_DATA_FIELD_DEF(uint64_t, C);
TILING_DATA_FIELD_DEF(uint64_t, Di);
TILING_DATA_FIELD_DEF(uint64_t, Hi);
TILING_DATA_FIELD_DEF(uint64_t, Wi);
TILING_DATA_FIELD_DEF(uint64_t, Do);
TILING_DATA_FIELD_DEF(uint64_t, Ho);
TILING_DATA_FIELD_DEF(uint64_t, Wo);
TILING_DATA_FIELD_DEF(uint64_t, coreNums);
TILING_DATA_FIELD_DEF(uint64_t, useCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, totalIdx);
TILING_DATA_FIELD_DEF(uint64_t, blockFactor);
TILING_DATA_FIELD_DEF(uint64_t, blockTail);
TILING_DATA_FIELD_DEF(uint64_t, ncFactor);
TILING_DATA_FIELD_DEF(uint64_t, doFactor);
TILING_DATA_FIELD_DEF(uint64_t, hoFactor);
TILING_DATA_FIELD_DEF(uint64_t, woFactor);
TILING_DATA_FIELD_DEF(uint64_t, ncOuter);
TILING_DATA_FIELD_DEF(uint64_t, doOuter);
TILING_DATA_FIELD_DEF(uint64_t, hoOuter);
TILING_DATA_FIELD_DEF(uint64_t, woOuter);
TILING_DATA_FIELD_DEF(uint64_t, ncTail);
TILING_DATA_FIELD_DEF(uint64_t, doTail);
TILING_DATA_FIELD_DEF(uint64_t, hoTail);
TILING_DATA_FIELD_DEF(uint64_t, woTail);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(AdaptiveMaxPool3dBigPoolTilingData)
TILING_DATA_FIELD_DEF(uint64_t, N);
TILING_DATA_FIELD_DEF(uint64_t, C);
TILING_DATA_FIELD_DEF(uint64_t, Di);
TILING_DATA_FIELD_DEF(uint64_t, Hi);
TILING_DATA_FIELD_DEF(uint64_t, Wi);
TILING_DATA_FIELD_DEF(uint64_t, Do);
TILING_DATA_FIELD_DEF(uint64_t, Ho);
TILING_DATA_FIELD_DEF(uint64_t, Wo);
TILING_DATA_FIELD_DEF(uint64_t, coreNums);
TILING_DATA_FIELD_DEF(uint64_t, useCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, totalIdx);
TILING_DATA_FIELD_DEF(uint64_t, blockFactor);
TILING_DATA_FIELD_DEF(uint64_t, blockTail);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AdaptiveMaxPool3d, AdaptiveMaxPool3dTilingData);
REGISTER_TILING_DATA_CLASS(AdaptiveMaxPool3d_320000, AdaptiveMaxPool3dSmallPoolTilingData);
REGISTER_TILING_DATA_CLASS(AdaptiveMaxPool3d_321000, AdaptiveMaxPool3dSmallPoolTilingData);
REGISTER_TILING_DATA_CLASS(AdaptiveMaxPool3d_322000, AdaptiveMaxPool3dSmallPoolTilingData);
REGISTER_TILING_DATA_CLASS(AdaptiveMaxPool3d_310000, AdaptiveMaxPool3dBigPoolTilingData);
REGISTER_TILING_DATA_CLASS(AdaptiveMaxPool3d_311000, AdaptiveMaxPool3dBigPoolTilingData);
REGISTER_TILING_DATA_CLASS(AdaptiveMaxPool3d_312000, AdaptiveMaxPool3dBigPoolTilingData);

struct InputInfo {
    uint64_t coreNum{0};
    uint64_t ubSizePlatForm{0};
    ge::DataType xDtype{ge::DT_FLOAT};
    uint64_t N{0};
    uint64_t C{0};
    uint64_t Di{0};
    uint64_t Hi{0};
    uint64_t Wi{0};
    uint64_t Do{0};
    uint64_t Ho{0};
    uint64_t Wo{0};
};

struct CalculateInfo {
    uint64_t useCoreNum{0};
    uint64_t totalIdx{0};
    uint64_t blockFactor{0};
    uint64_t blockTail{0};
    uint64_t ncFactor{0};
    uint64_t doFactor{0};
    uint64_t hoFactor{0};
    uint64_t woFactor{0};
    uint64_t ncOuter{0};
    uint64_t doOuter{0};
    uint64_t hoOuter{0};
    uint64_t woOuter{0};
    uint64_t ncTail{0};
    uint64_t doTail{0};
    uint64_t hoTail{0};
    uint64_t woTail{0};
    uint64_t kernelDMax{0};
    uint64_t kernelHMax{0};
    uint64_t kernelWMax{0};
};

struct AdaptiveMaxPool3dCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

class AdaptiveMaxPool3dTilingBase : public TilingBaseClass
{
public:
    explicit AdaptiveMaxPool3dTilingBase(gert::TilingContext* context) : TilingBaseClass(context)
    {}
    ~AdaptiveMaxPool3dTilingBase() override
    {}
    InputInfo input_;
    CalculateInfo calInfo_;

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
};

class AdaptiveMaxPool3dSmallPoolTiling : public AdaptiveMaxPool3dTilingBase
{
public:
    explicit AdaptiveMaxPool3dSmallPoolTiling(gert::TilingContext* context) : AdaptiveMaxPool3dTilingBase(context)
    {}
    ~AdaptiveMaxPool3dSmallPoolTiling() override
    {}
    AdaptiveMaxPool3dSmallPoolTilingData tilingdata_;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

private:
    uint64_t CalKernelMax(uint64_t inputSize, uint64_t outputSize);
    void SetTilingData();
};

class AdaptiveMaxPool3dBigPoolTiling : public AdaptiveMaxPool3dTilingBase
{
public:
    explicit AdaptiveMaxPool3dBigPoolTiling(gert::TilingContext* context) : AdaptiveMaxPool3dTilingBase(context)
    {}
    ~AdaptiveMaxPool3dBigPoolTiling() override
    {}
    AdaptiveMaxPool3dBigPoolTilingData tilingdata_;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

private:
    void SetTilingData();
};

} // namespace optiling
#endif // ADAPTIVE_MAX_POOL3D_TILING_H