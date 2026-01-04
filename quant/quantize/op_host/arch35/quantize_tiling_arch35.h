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
 * \file quantize_tiling_arch35.h
 * \brief quantize op tiling
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_QUANTIZE_TILING_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_QUANTIZE_TILING_H_

#include <cstdint>
#include "register/tilingdata_base.h"
#include "quant/quantize/op_kernel/arch35/quantize_tilingdata.h"
#include "exe_graph/runtime/tiling_context.h"

namespace optiling {
namespace quantize {
class Quantize {
public:
    explicit Quantize(gert::TilingContext* context) : context_(context){};
    ge::graphStatus DoQuantizeTiling();

protected:
    ge::graphStatus GetCompileInfo();
    ge::graphStatus GetOpParam();
    ge::graphStatus CheckDtype();
    ge::graphStatus CheckAttrs();
    ge::DataType GetDataType(const std::string &dtype);
    ge::graphStatus CheckShape();
    void SelectMode();
    void MergeInputShape();
    int64_t GetCoreNum(int64_t factor, int64_t coreNum);
    void CalcPerTensorBlockFactor(int64_t size);
    void CalcPerChannelBlockFactor(int64_t size);
    void CalcPerTensorUBFactor(int64_t cacheLineNum);
    void CalcPerChannelUBFactor(int64_t cacheLineNum);
    void CalcPerChannelNddmaUBFactor();
    void CalcPerHeadBlockFactor();
    void CalcPerHeadUBFactor(int64_t cacheLineNum);
    void CalcTiling();
    int64_t CalcMaxBaseLen(int64_t ubSize);
    int64_t CalcPerChannelMaxN(int64_t ubSize, int64_t base);
    uint32_t GetCoreNumDoubleCut(int64_t shapeDim0, int64_t shapeDim1, int64_t coreNum);
    void CalTilingKey();
    void WriteTilingData();

private:
    gert::TilingContext* context_ = nullptr;

    int64_t coreNum_{0};
    int64_t ubSize_{0};
    int64_t reserveUb_{2048};
    int64_t cacheLine_{128};

    gert::Shape xInputShape_;
    gert::Shape scalesInputShape_;
    gert::Shape zeroPointsInputShape_;
    gert::Shape yOutputShape_;
    int64_t xDimNum_{0};
    int64_t scalesDimNum_{0};
    int64_t zeroPointsDimNum_{0};
    ge::DataType xDtype_{ge::DT_UNDEFINED};
    ge::DataType scalesDtype_{ge::DT_UNDEFINED};
    ge::DataType zeroPointsDtype_{ge::DT_UNDEFINED};
    ge::DataType yDtype_{ge::DT_UNDEFINED};

    uint32_t mode_{0};
    bool hasZeroPoint_{true};
    ge::DataType dtype_{ge::DT_UNDEFINED};
    std::string dtypeStr_;
    int64_t axis_{-1};
    std::string errorMsg_;

    int64_t actCoreNum_{0};
    int64_t blockAxis_{-1};
    int64_t blockUnion_{1};
    int64_t blockFactor_{-1};
    int64_t blockTailFactor_{-1};
    int64_t ubAxis_{-1};
    int64_t ubFactor_{-1};
    int64_t baseN_{1};
    int64_t baseLen_{1};
    int64_t tilingKey_{0};
};
} // namespace quantize
} // namespace optiling
#endif