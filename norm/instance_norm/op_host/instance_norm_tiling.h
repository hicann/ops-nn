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
 * \file instance_norm_tiling.h
 * \brief
 */

#ifndef INSTANCE_NORM_TILING_H
#define INSTANCE_NORM_TILING_H

#include <cmath>
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "../op_kernel/arch35/instance_norm_tiling_data.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_templates_registry.h"

using namespace Ops::NN::Optiling;

namespace optiling {
static constexpr uint64_t IN_REDUCE_EMPTY_PRIORITY = 5000;
static constexpr uint64_t IN_AR_FULL_REDUCE_PRIORITY = 9000;
static constexpr uint64_t IN_AR_WELFORD_PRIORITY = 9100;
static constexpr uint64_t IN_ARA_FULL_REDUCE_PRIORITY = 10000;
static constexpr uint64_t IN_ARA_WELFORD_PRIORITY = 15000;

constexpr float DEFAULT_EPSILON = 1e-6;
constexpr int64_t INPUT_NUM = 3;
constexpr uint32_t INPUT_X_INDEX = 0;
constexpr uint32_t INPUT_GAMMA_INDEX = 1;
constexpr uint32_t INPUT_BETA_INDEX = 2;
constexpr uint32_t OUTPUT_Y_INDEX = 0;
constexpr uint32_t OUTPUT_MEAN_INDEX = 1;
constexpr uint32_t OUTPUT_VARIANCE_INDEX = 2;
constexpr uint32_t ATTR_DATA_FORMAT_IDX = 0;
constexpr uint32_t ATTR_EPSILON_IDX = 1;

struct InstanceNormCompileInfo{
    uint64_t coreNum;       // 系统核数
    uint64_t ubSize;        // UB空间
    uint32_t vectorLength;  // 256
    uint64_t ubBlockSize;     // 32B，UB的字节对齐单位
};

class InstanceNormRegbaseTilingBase : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit InstanceNormRegbaseTilingBase(gert::TilingContext* context) : Ops::NN::Optiling::TilingBaseClass(context)
    {}

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
        a1 = 0;
        a0 = 0;
        r = 0;
        vlfp32 = 0;
        ubBlockSize = 0;
        epsilon = 0;
    }
protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus CheckDtypeValid();
    ge::graphStatus CheckShapeValid();
    ge::graphStatus CheckXYShapeValid();
    ge::graphStatus CheckGammaBettaShapeValid();
    ge::graphStatus CheckMeanVarianceShapeValid();
    ge::graphStatus CheckShapeAllNotNegative(gert::Shape& shape);
protected:
    int64_t a1;
    int64_t a0;
    int64_t r;
    int64_t vlfp32;
    int64_t vectorLength;
    int64_t ubBlockSize;  // 用于在UB上进行32B的字节对齐
    float epsilon;

    int64_t blockNum_{1};
    int64_t a0InnerLength_{0};
    int64_t totalTiles_{0};
    int64_t tilesPerCore_{0};
    int64_t a0OuterNum_{0};
    int64_t a0TailLength_{0};

    ge::DataType dataType;
    ge::DataType gammaDataType;
    ge::DataType meanDataType;
    ge::Format format;
    gert::Shape xStorageShape;
};

class InstanceNormReduceEmptyTiling : public InstanceNormRegbaseTilingBase {
public:
    explicit InstanceNormReduceEmptyTiling(gert::TilingContext* context) : InstanceNormRegbaseTilingBase(context)
    {}
    ~InstanceNormReduceEmptyTiling() override = default;
    void Reset(gert::TilingContext* context) override;
protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
private:
    InstanceNormReduceEmptyTilingData td_;
};

class InstanceNormARFullReduceTiling : public InstanceNormRegbaseTilingBase {
public:
    explicit InstanceNormARFullReduceTiling(gert::TilingContext* context) : InstanceNormRegbaseTilingBase(context)
    {}
    ~InstanceNormARFullReduceTiling() override = default;

    void Reset(gert::TilingContext* context) override
    {
        InstanceNormRegbaseTilingBase::Reset(context);
        blockNum_ = 0;
        binaryAddQuotient = 0;
    }

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus BinaryAddTiling();
private:
    int64_t binaryAddQuotient;
    InstanceNormARFullReduceTilingData td_;
};


class InstanceNormARWelfordTiling : public InstanceNormRegbaseTilingBase {
public:
    explicit InstanceNormARWelfordTiling(gert::TilingContext* context) : InstanceNormRegbaseTilingBase(context)
    {}
    ~InstanceNormARWelfordTiling() override = default;
    void Reset(gert::TilingContext* context) override;
protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus PostTiling() override;
    bool IsValidwelfordTileLength(int64_t welfordTileLength);
    int64_t GammaBetaTypeSize = 1;
private:
    InstanceNormARWelfordTilingData td_;
};

class InstanceNormARAFullReduceTiling : public InstanceNormRegbaseTilingBase {
public:
    explicit InstanceNormARAFullReduceTiling(gert::TilingContext* context) : InstanceNormRegbaseTilingBase(context)
    {}
    ~InstanceNormARAFullReduceTiling() override = default;
protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus BinaryAddTiling();
private:
    int64_t binaryAddQuotient;
    InstanceNormARAFullReduceTilingData td_;
};

class InstanceNormARAWelfordTiling : public InstanceNormRegbaseTilingBase {
public:
    explicit InstanceNormARAWelfordTiling(gert::TilingContext* context) : InstanceNormRegbaseTilingBase(context)
    {}
    ~InstanceNormARAWelfordTiling() override = default;
protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;
    void SetInputInfo();
    ge::graphStatus BinaryAddTiling(int64_t elemSize, int64_t gammaElemSize, int64_t tileA0Len);
private:
    int64_t blockNum;   // 这个变量要合并到基类里面去
    InstanceNormARAWelfordTilingData td_;
};

} // namespace optiling
#endif // INSTANCE_NORM_TILING_H
