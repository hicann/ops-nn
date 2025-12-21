/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file gelu_quant_regbase_tiling.h
 * \brief
 */

#ifndef GELU_QUANT_TILING_ARCH35_H
#define GELU_QUANT_TILING_ARCH35_H

#include "gelu_quant_tiling.h"

namespace optiling {
namespace geluquantregbase {

enum class RoundMode : int
{
    MODE_NONE = 0,
    MODE_RINT = 1,
    MODE_FLOOR = 2,
    MODE_CEIL = 3,
    MODE_ROUND = 4,
    MODE_TRUNC = 5,
    MODE_ODD = 6,
    MODE_HYBRID = 7,
    MODE_UNDEFINED = -1,
};

struct GeluQuantRegbaseBaseInfoParams {
    // platformInfo
    int64_t vectorCoreNum{0};
    uint64_t ubSize{0};

    // shapeInfo
    int64_t xDimNum{0};           // x的维度
    int64_t endAxisLen{0};        // 最后一根轴的长度
    int64_t endAxisLenAligned{0}; // fp32情况下，最后一根轴向上8对齐的长度
    int64_t fusedFrontAxis{1};    // 除了最后一根轴所有轴的size
    int64_t fusedAllAxis{1};      // 所有轴的size
    int64_t elementNumAlign{0};   // fp32情况下，size向上8对齐

    // dtype
    ge::DataType xInputDtype{ge::DT_FLOAT};
    ge::DataType scaleInputDtype{ge::DT_FLOAT};
    ge::DataType offsetInputDtype{ge::DT_FLOAT};

    // optional
    int64_t inputScaleType{0}; // scale还是tensor，EMPTY_TENSOR = 0;SCALAR_TENSOR = 1;NORMAL_TENSOR = 2;
    int64_t inputOffsetType{0};

    // attr
    int64_t quantMode{0};
    int64_t approximate{0};
    uint32_t dstType{0};
    geluquantregbase::RoundMode roundMode = geluquantregbase::RoundMode::MODE_UNDEFINED;
};

struct GeluQuantRegbaseSplitCoreParams {
    int64_t normalCoreProcessNum{0}; // 普通核元素个数
    int64_t usedCoreNum{0};          // 使用到的核数
    int64_t tailCoreProcessNum{0};   // 尾核处理元素次数

    int64_t coexistentNodeNum{0};        // 每次计算参与所有单个元素的数量（节点数），比如输入、中间结果、输出
    int64_t coexistentNodeElementNum{0}; // 单个UB内每次能处理多少个结果

    int64_t templateMode{0};

    int64_t rowInner{1}; // 每个UB中可以处理的row数量
    int64_t rowOuter{1}; // 每个普通核需要处理多少次UB才能全部处理完
    int64_t rowTail{1};  // 尾核需要处理多少次UB才能全部处理完
    int64_t colInner{1}; // 尾轴长度
    int64_t colOuter{1}; // 1
    int64_t colTail{1};  // 尾轴长度
    uint64_t tilingKey{0};
};

class GeluQuantRegbaseTiling {
public:
    explicit GeluQuantRegbaseTiling(gert::TilingContext* context) : context_(context), nodeName_(context->GetNodeName())
    {}
    ~GeluQuantRegbaseTiling() = default;

    GeluQuantTilingData tilingData;
    ge::graphStatus RunGeluQuantRegbaseTiling();
    ge::graphStatus GetInputInfo();
    RoundMode GetRoundMode(std::string& roundMode);
    const gert::Shape& EnsureNotScalar(const gert::Shape& inShape);
    ge::graphStatus ProcessAttrsInfo();
    ge::graphStatus ProcessRequiredInfo();
    ge::graphStatus ProcessOptionalScaleInfo();
    ge::graphStatus ProcessOptionalOffsetInfo();
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus DoTiling();
    ge::graphStatus DoStaticQuantTiling();
    ge::graphStatus DoStaticQuantPerTensorTiling();
    ge::graphStatus DoStaticQuantFullKernelSmallEndAxis();
    ge::graphStatus DoStaticQuantNotFullKernelSplitEndAxis();
    ge::graphStatus DoDynamicQuantTiling();
    ge::graphStatus PostTiling();
    void DumpTilingInfo() const;
    uint64_t GetTilingKey() const;
    void SaveToTilingData();

protected:
    gert::TilingContext* context_ = nullptr;
    const ge::char_t* nodeName_;
    std::string errorMsg_ = "";

private:
    GeluQuantRegbaseBaseInfoParams baseInfoOp;
    GeluQuantRegbaseSplitCoreParams splitCoreOp;
};
} // namespace geluquantregbase
} // namespace optiling
#endif // GELU_QUANT_REGBASE_TILING_H