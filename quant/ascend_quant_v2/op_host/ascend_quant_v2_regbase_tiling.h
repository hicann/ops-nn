/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ascend_quant_v2_regbase_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_ASCEND_QUANT_V2_REGBASE_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_ASCEND_QUANT_V2_REGBASE_TILING_H

#include <cstdint>
#include <vector>
#include "register/tilingdata_base.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "tiling_base/tiling_base.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "ascend_quant_v2_tiling.h"
using namespace Ops::NN::Optiling;

namespace optiling {
namespace ascendquantv2regbase {
enum class RoundMode
{
    MODE_ROUND = 0,
    MODE_FLOOR = 1,
    MODE_CEIL = 2,
    MODE_TRUNC = 3,
    MODE_RINT = 4,
    MODE_HYBRID = 5,
    MODE_UNDEFINED = -1,
};

class AscendQuantV2Regbase {
public:
    explicit AscendQuantV2Regbase(gert::TilingContext* context) : context_(context){};
    ge::graphStatus DoAscendQuantV2Tiling();

protected:
    ge::graphStatus GetCompileInfo();
    ge::graphStatus GetOpParam();
    ge::graphStatus CheckInputValid(
        const gert::Shape& input1, const gert::Shape& input2, int32_t axis, const std::string& param) const;
    ge::graphStatus CheckAttrs(const gert::Shape& xShape);
    RoundMode GetRoundMode(std::string& roundMode);
    void MergeInputShape(const gert::Shape& input);
    uint32_t GetCoreNum(int64_t factor, int64_t coreNum) const;
    uint32_t GetCoreNumDoubleCut(int64_t shape0, int64_t shape1, int64_t coreNum) const;
    int64_t GetNewShape0(int64_t shape0, int64_t shape1);
    void CalcTiling();
    void CalcTilingKey();
    int64_t CalcMaxBaseLen(int64_t ubSize) const;
    int64_t CalcMaxN(int64_t ubSize, int64_t base) const;
    void WriteTilingData();
    void CalcPerHeadTiling();
    void CalcPerHeadBlockFactor();
    void CalcPerHeadUBFactor(int64_t cacheLineNum);
    void CalcPerTensorBlockFactor(int64_t size);
    void CalcPerChannelBlockFactor(int64_t size);
    void CalcPerTensorUBFactor(int64_t numPerCache);
    void CalcPerChannelUBFactor(int64_t numPerCache);

    ge::graphStatus CheckShapeEqual(const gert::Shape& shape1, const gert::Shape& shape2) const;
    ge::graphStatus CheckInputDtype();
    const gert::Shape& EnsureNotScalar(const gert::Shape& inShape);

private:
    gert::TilingContext* context_ = nullptr;

    int64_t coreNum_{0};
    uint64_t ubSize_{0};
    bool isAscend910B_{false};
    int64_t reserveUb_{2048};
    int64_t cacheLine_{256};

    gert::Shape xInputShape_;
    ge::DataType xDtype_{ge::DT_UNDEFINED};
    ge::DataType yDtype_{ge::DT_UNDEFINED};
    bool hasOffset_{true};
    bool useDoubleCut{false};
    int16_t sqrtMode_ = 0;
    RoundMode roundMode_ = RoundMode::MODE_UNDEFINED;
    int32_t dstType_ = 0;
    int32_t axis_{-1};

    int64_t actCoreNum_{0};
    int64_t blockUnion_{1};
    int64_t blockBind_{1};
    int32_t blockAxis_{-1};
    int64_t blockFactor_{-1};
    int64_t blockTailFactor_{-1};
    int32_t ubAxis_{-1};
    int64_t ubFactor_{-1};
    int64_t baseN_{1};
    int64_t baseLen_{1};
    uint64_t tilingKey_{0};
    bool isPerTensor_{false};
    bool isPerHead_{false};
    bool isFp8{false};
    uint32_t mode_ = 0;
    std::string errorMsg_ = "";
};
} // namespace ascendquantv2regbase
} // namespace optiling
#endif