/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 *
 *
 *
 */

/*!
 * \file quantized_batch_norm_tiling.h
 * \brief
 */

#ifndef QUANTIZED_BATCH_NORM_TILING_H
#define QUANTIZED_BATCH_NORM_TILING_H

#include "register/tilingdata_base.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "tiling_base/tiling_base.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {
using namespace Ops::NN::Optiling;

BEGIN_TILING_DATA_DEF(QuantizedBatchNormBaseTilingData)
TILING_DATA_FIELD_DEF(int64_t, patternR1);
TILING_DATA_FIELD_DEF(int64_t, patternR0);
TILING_DATA_FIELD_DEF(int64_t, patternA);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, tailCoreBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, aUbFactor);
TILING_DATA_FIELD_DEF(int64_t, aUbLoop);
TILING_DATA_FIELD_DEF(int64_t, aUbTail);
TILING_DATA_FIELD_DEF(int64_t, tailCoreAUbLoop);
TILING_DATA_FIELD_DEF(int64_t, tailCoreAUbTail);
TILING_DATA_FIELD_DEF(int64_t, r0UbFactor);
TILING_DATA_FIELD_DEF(int64_t, r0UbLoop);
TILING_DATA_FIELD_DEF(int64_t, r0UbTail);
TILING_DATA_FIELD_DEF(int64_t, procNR0);
TILING_DATA_FIELD_DEF(int64_t, nR0Loop);
TILING_DATA_FIELD_DEF(int64_t, lastLoopNR0);
TILING_DATA_FIELD_DEF(int64_t, patternR0Align);
TILING_DATA_FIELD_DEF(float, epsilon);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(QuantizedBatchNorm, QuantizedBatchNormBaseTilingData)

BEGIN_TILING_DATA_DEF(QuantizedBatchNormWelfordTilingData)
TILING_DATA_FIELD_DEF(int64_t, patternR1);
TILING_DATA_FIELD_DEF(int64_t, patternR0);
TILING_DATA_FIELD_DEF(int64_t, patternA);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, tailCoreBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, aUbFactor);
TILING_DATA_FIELD_DEF(int64_t, aUbTail);
TILING_DATA_FIELD_DEF(int64_t, aUbLoop);
TILING_DATA_FIELD_DEF(int64_t, tailCoreAUbLoop);
TILING_DATA_FIELD_DEF(int64_t, tailCoreAUbTail);
TILING_DATA_FIELD_DEF(int64_t, r0UbLoop);
TILING_DATA_FIELD_DEF(int64_t, r0UbFactor);
TILING_DATA_FIELD_DEF(int64_t, r0UbTail);
TILING_DATA_FIELD_DEF(int64_t, procNR0);
TILING_DATA_FIELD_DEF(int64_t, nR0Loop);
TILING_DATA_FIELD_DEF(int64_t, patternR0Align);
TILING_DATA_FIELD_DEF(int64_t, lastLoopNR0);
TILING_DATA_FIELD_DEF(float, epsilon);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(QuantizedBatchNorm_1000, QuantizedBatchNormWelfordTilingData)
REGISTER_TILING_DATA_CLASS(QuantizedBatchNorm_1001, QuantizedBatchNormWelfordTilingData)
REGISTER_TILING_DATA_CLASS(QuantizedBatchNorm_1002, QuantizedBatchNormWelfordTilingData)
REGISTER_TILING_DATA_CLASS(QuantizedBatchNorm_1003, QuantizedBatchNormWelfordTilingData)
REGISTER_TILING_DATA_CLASS(QuantizedBatchNorm_1012, QuantizedBatchNormWelfordTilingData)
REGISTER_TILING_DATA_CLASS(QuantizedBatchNorm_1013, QuantizedBatchNormWelfordTilingData)

struct ParamsQuantizedBatchNorm {
    uint64_t coreNum;
    uint64_t ubSizePlatForm;
    int64_t patternR1;
    int64_t patternR0;
    int64_t patternR0Align;
    int64_t patternA;
    float epsilon;
    string nodeName;
    ge::DataType xDtype;
};

struct QuantizedBatchNormCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

class QuantizedBatchNormTilingBase : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit QuantizedBatchNormTilingBase(gert::TilingContext* context) : Ops::NN::Optiling::TilingBaseClass(context)
    {}
    ~QuantizedBatchNormTilingBase() override = default;
    ParamsQuantizedBatchNorm commonParams;

protected:
    bool IsCapable() override
    {
        return true;
    };
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override
    {
        return ge::GRAPH_SUCCESS;
    };
    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    };
    uint64_t GetTilingKey() const override
    {
        return 0;
    };
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override
    {
        return ge::GRAPH_SUCCESS;
    };
    bool CheckInputDtype();
    bool CheckInputShape();
    bool CheckOthersInputShape();
};

class QuantizedBatchNormWelfordTiling : public QuantizedBatchNormTilingBase {
public:
    explicit QuantizedBatchNormWelfordTiling(gert::TilingContext* context) : QuantizedBatchNormTilingBase(context)
    {}
    ~QuantizedBatchNormWelfordTiling() override = default;
    QuantizedBatchNormWelfordTilingData td_;

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

private:
    uint64_t usedCoreNum;
    uint64_t welfordTilingkey;
    void DoUbTiling(int64_t& aUbFactor, int64_t& r0UbFactor);
    uint32_t FindDichotomizeAddDiffSize(uint32_t parallelN);
};

} // namespace optiling
#endif // QUANTIZED_BATCH_NORM_TILING_H