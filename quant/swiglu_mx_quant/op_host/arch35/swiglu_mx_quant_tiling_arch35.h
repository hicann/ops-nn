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
 * \file swiglu_mx_quant_tiling_arch35.h
 * \brief Tiling data structure and parameters for SwiGLU + MX quantization
 */

#ifndef QUANT_SWIGLU_MX_QUANT_TILING_ARCH35_H
#define QUANT_SWIGLU_MX_QUANT_TILING_ARCH35_H

#include <cstdint>
#include <vector>
#include <string>
#include <set>
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "../../op_kernel/arch35/swiglu_mx_quant_tiling_data.h"

namespace optiling {

// ==================== CompileInfo（编译时存储芯片信息）====================
struct SwigluMxQuantCompileInfo {
    int64_t totalCoreNum{0};    // 总核数
    int64_t ubSize{0};          // UB 大小
};

// ==================== 输入信息 ====================
struct SwigluMxQuantInputInfo {
    ge::DataType xDtype{ge::DT_UNDEFINED};
    int64_t dimNum{0};              // 输入维度数量
    int64_t inputDim0{1};           // 合轴后的第0维
    int64_t inputDim1{0};           // 合轴后的第1维
    int64_t inputDim2{0};           // 合轴后的第2维
    int64_t groupIndexType{0};      // 0=不存在, 1=int32, 2=int64
    int64_t groupIndexNum{0};       // group_index 的第0维 shape，不存在时为 0
};

// ==================== 输出信息 ====================
struct SwigluMxQuantOutputInfo {
    ge::DataType yDtype{ge::DT_UNDEFINED};
    ge::DataType mxscaleDtype{ge::DT_UNDEFINED};
    int64_t outputDim2{0};
};

// ==================== 属性参数 ====================
struct SwigluMxQuantAttrParam {
    int64_t activateDim{-1};
    bool activateLeft{false};
    int64_t swigluMode{0};
    float clampLimit{7.0f};
    float gluAlpha{1.702f};
    float gluBias{1.0f};
    int64_t axis{-1};
    int64_t dstType{40};
    int64_t roundMode{4};       // MODE_RINT
    int64_t blockSize{32};
    int64_t scaleAlg{0};
    float maxDtypeValue{0.0f};
    int64_t groupMode{0};
    int64_t groupIndexType{0};  // 0=不存在, 1=int32, 2=int64
};

// ==================== Tiling 计算结果 ====================
struct SwigluMxQuantTilingResult {
    int64_t basicDim2{256};
    int64_t basicDim1{1};
    int64_t maxBasicNumUbDim2{0};
    int64_t maxBasicNumUbDim1{0};
    int64_t ubLoopPerRow{0};
    int64_t ubTailPerRow{0};
    int64_t isFullLoad{0};
    int64_t usedCoreNum{0};
    // Inter-core split
    int64_t frontCoreNum{0};
    int64_t frontCoreBasicNumDim1{0};
    int64_t frontCoreLoopTimes{0};
    int64_t frontCoreLastLoopBasicNum{0};
    int64_t tailCoreBasicNumDim1{0};
    int64_t tailCoreLoopTimes{0};
    int64_t tailCoreLastLoopBasicNum{0};
};

// ==================== Round Mode 枚举 ====================
enum class RoundModeList {
    MODE_RINT = 1,
    MODE_FLOOR = 2,
    MODE_CEIL = 3,
    MODE_ROUND = 4,
    MODE_TRUNC = 5,
    MODE_ODD = 6,
    MODE_HYPER = 7,
    MODE_UNDEFINED = -1,
};

// ==================== SwigluMxQuantRegbaseTiling 类定义 ====================

class SwigluMxQuantRegbaseTiling
{
public:
    explicit SwigluMxQuantRegbaseTiling(gert::TilingContext* context) : context_(context) {};

    ge::graphStatus GetNpuInfo();
    ge::graphStatus ParseAttrs();
    ge::graphStatus ValidateInput();
    ge::graphStatus ValidateOutput();
    ge::graphStatus PreProcess();
    ge::graphStatus CalculateTiling();
    int64_t CalculateTilingKey();
    ge::graphStatus FillTilingData();
    ge::graphStatus SetParams();
    void PrintTilingData() const;

private:
    // Compile info
    SwigluMxQuantCompileInfo compileInfo_;

    // Input/Output info
    SwigluMxQuantInputInfo inputInfo_;
    SwigluMxQuantOutputInfo outputInfo_;

    // Attr params
    SwigluMxQuantAttrParam attrParam_;

    // Tiling result
    SwigluMxQuantTilingResult tilingResult_;

    // Context and tiling data
    gert::TilingContext* context_ = nullptr;
    SwigluMxQuantTilingData* tilingData_ = nullptr;
};

// ==================== 主函数声明 ====================
ge::graphStatus Tiling4SwigluMxQuant(gert::TilingContext* context);
ge::graphStatus TilingPrepare4SwigluMxQuant(gert::TilingParseContext* context);

} // namespace optiling
#endif // QUANT_SWIGLU_MX_QUANT_TILING_ARCH35_H
