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
 * \file swiglu_backward_mx_quant_with_dual_axis_tiling_arch35.h
 * \brief Host-side tiling declarations for SwigluBackwardMxQuantWithDualAxis
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_SWIGLU_BACKWARD_MX_QUANT_WITH_DUAL_AXIS_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_SWIGLU_BACKWARD_MX_QUANT_WITH_DUAL_AXIS_H

#include <cstdint>
#include <vector>
#include <string>
#include <set>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "../../op_kernel/arch35/swiglu_backward_mx_quant_with_dual_axis_tilingdata.h"

namespace optiling {

struct SwigluBackwardMxQuantWithDualAxisCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

struct SwigluBackwardMxQuantWithDualAxisTilingParam {
    int64_t totalCoreNum{0};
    int64_t usedCoreNum{0};
    int64_t ubSize{0};
    int64_t roundMode{0};
    int64_t dstType{0};
    int64_t scaleAlg{0};
    int64_t isGroupIdx{0};
    int64_t activateLeft{1};
    int64_t blockW{0};
    int64_t splitBlockH{0};
    int64_t dimM{1};
    int64_t dimN{1};
    int64_t numGroups{0};
    int64_t dimNBlockNum{0};
    int64_t dimNTail{0};
    int64_t workspaceSize{0};
    int64_t dtypeSize{2};
    int64_t dimBatch{1};
    ge::DataType yDtype{ge::DT_UNDEFINED};
};

enum class RoundModeList {
    MODE_ROUND = 0,
    MODE_FLOOR = 1,
    MODE_CEIL = 2,
    MODE_TRUNC = 3,
    MODE_RINT = 4,
    MODE_HYBRID = 5,
    MODE_UNDEFINED = -1,
};

ge::graphStatus TilingForSwigluBackwardMxQuantWithDualAxis(gert::TilingContext* context);
ge::graphStatus TilingPrepareForSwigluBackwardMxQuantWithDualAxis(gert::TilingParseContext* context);

class SwigluBackwardMxQuantWithDualAxisTiling {
public:
    explicit SwigluBackwardMxQuantWithDualAxisTiling(gert::TilingContext* context) : context_(context) {}
    ~SwigluBackwardMxQuantWithDualAxisTiling() {}
    ge::graphStatus DoTiling();

private:
    ge::graphStatus GetAttr();
    ge::graphStatus CheckDtype();
    ge::graphStatus CheckOutputScaleDtype();
    ge::graphStatus CheckShape();
    ge::graphStatus CheckScaleShape(const gert::Shape& mxScale1Shape, const gert::Shape& mxScale2Shape,
                                    const gert::Shape& xShape);
    ge::graphStatus CheckMxScale1Shape(const gert::Shape& mxScale1Shape, const gert::Shape& xShape, int64_t dimGradX,
                                       int64_t batchDimCount);
    ge::graphStatus CheckMxScale2Shape(const gert::Shape& mxScale2Shape, const gert::Shape& xShape, int64_t dimGradX,
                                       int64_t batchDimCount, int64_t expectScaleDimNum);
    ge::graphStatus CheckYShape(const gert::Shape& xShape, const gert::Shape& y1Shape, const gert::Shape& y2Shape);
    ge::graphStatus CheckGroupIndexShape();
    ge::graphStatus CheckYGradShape(const gert::Shape& xShape, int64_t xDimNum);
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus ComputeTilingParams();
    ge::graphStatus SetTilingData();

    void SetTilingKeyAndCore();
    void PrintTilingData();

    RoundModeList GetRoundMode(const std::string& roundMode);

private:
    uint64_t roundMode_ = 0;
    uint64_t scaleAlg_ = 0;
    uint64_t mode_ = 0;
    uint64_t isGroupIdx_ = 0;
    gert::TilingContext* context_ = nullptr;
    SwigluBackwardMxQuantWithDualAxisTilingData* tilingData_ = nullptr;
    SwigluBackwardMxQuantWithDualAxisTilingParam tilingParams_;
};

} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_SWIGLU_BACKWARD_MX_QUANT_WITH_DUAL_AXIS_H
