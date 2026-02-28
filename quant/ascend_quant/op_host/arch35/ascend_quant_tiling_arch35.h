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
 * \file ascend_quant_tiling_arch35.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_ASCEND_QUANT_REGBASE_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_ASCEND_QUANT_REGBASE_TILING_H

#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "ascend_quant_tiling.h"

namespace optiling {
using namespace Ops::NN::Optiling;
namespace ascendquantregbase {
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

class AscendQuantRegbase {
public:
    explicit AscendQuantRegbase(gert::TilingContext* context) : context_(context) {};
    ge::graphStatus DoAscendQuantTiling();

protected:
    ge::graphStatus GetPlatform();
    ge::graphStatus GetOpParam();
    ge::graphStatus CheckDtype();
    ge::graphStatus CheckAttrs();
    ge::graphStatus CheckShape(const gert::Shape& xShape, const gert::Shape& yShape) const;
    RoundMode GetRoundMode(std::string& roundMode);
    void MergeInputShape(const gert::Shape& input);
    int64_t GetCoreNum(int64_t factor, int64_t coreNum) const;
    int64_t CalcMaxBaseLen(int64_t ubSize) const;
    void CalcTiling();
    void CalcTilingKey();
    void CalcPerTensorBlockFactor(int64_t size);
    void CalcPerTensorUBFactor(int64_t numPerCache);
    void WriteTilingData();

private:
    gert::TilingContext* context_ = nullptr;

    int64_t coreNum_{0};
    uint64_t ubSize_{0};
    int64_t reserveUb_{2048};
    int64_t cacheLine_{256};

    gert::Shape xInputShape_;
    ge::DataType xDtype_{ge::DT_UNDEFINED};
    ge::DataType yDtype_{ge::DT_UNDEFINED};
    RoundMode roundMode_ = RoundMode::MODE_UNDEFINED;
    float scale_;
    float offset_;
    int32_t dstType_ = 0;

    int64_t actCoreNum_{0};
    int64_t blockFactor_{-1};
    int64_t blockTailFactor_{-1};
    int64_t baseLen_{1};
    uint64_t tilingKey_{0};
    std::string errorMsg_ = "";
};
} // namespace ascendquantregbase
} // namespace optiling
#endif