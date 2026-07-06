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
 * \file rotate_quant_tiling.h
 * \brief RotateQuant算子arch35 tiling类定义
 */
#ifndef __OP_HOST_ROTATE_QUANT_TILING_H__
#define __OP_HOST_ROTATE_QUANT_TILING_H__

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "../../op_kernel/arch35/rotate_quant_tiling_data.h"

namespace Ops {
namespace NN {
namespace RotateQuant {

struct RotateQuantAptCompileInfo {
    uint64_t l1Size{0};
    uint64_t l2Size{0};
    uint64_t l0ASize{0};
    uint64_t l0BSize{0};
    uint64_t l0CSize{0};
    uint64_t aicNum{0};
};

struct RotateQuantAptInfo {
    int64_t M = 0;
    int64_t N = 0;
    int64_t K = 0;
    int64_t nKRatio = 0;
    uint64_t rotType = 0;
    float alpha = 0.0;
    float dstTypeMax = 0.0;
    int32_t axis = -1;
    int32_t roundMode = 0;
    int32_t scaleAlg = 0;
    bool trans = false;
    bool initFlag = false;
    ge::DataType dataType = ge::DT_FLOAT16;
    ge::DataType yDtype = ge::DT_FLOAT4_E2M1;
    const char* opName = "RotateQuantApt";
    bool hasAlpha = false;
    uint32_t realCoreNum = 0;
};

class RotateQuantAptTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit RotateQuantAptTiling(gert::TilingContext* context) : Ops::NN::Optiling::TilingBaseClass(context)
    {
        InitCompileInfo();
    };
    ~RotateQuantAptTiling() override = default;

protected:
    bool IsCapable() override { return true; }
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

protected:
    void InitCompileInfo();
    ge::graphStatus CheckContext();
    ge::graphStatus AnalyzeDtype();
    ge::graphStatus AnalyzeAttrs();
    ge::graphStatus AnalyzeShapes();
    ge::graphStatus ParseAttrValues();
    ge::graphStatus CheckAlphaInput();
    ge::graphStatus ValidateRotShape(const gert::Shape& rotShape);
    ge::graphStatus ValidateScaleShape(const gert::Shape& mxscaleShape, const gert::Shape& scaleShape);
    void PrintTilingData();
    ge::graphStatus CalcTilingData();
    bool SetMatmulTiling();

    RotateQuantAptOpt::RotateQuantAptTilingData tilingData_;
    RotateQuantAptInfo inputParams_;
    RotateQuantAptCompileInfo compileInfo_;
    size_t tilingDataSize_ = 0;
};
} // namespace RotateQuant
} // namespace NN
} // namespace Ops
#endif // __OP_HOST_ROTATE_QUANT_TILING_H__