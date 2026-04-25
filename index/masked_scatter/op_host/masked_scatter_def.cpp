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
 * \file masked_scatter_def.cpp
 * \brief MaskedScatter ophost
 */
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> SUPPORT_DTYPE = {
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_DOUBLE, ge::DT_UINT8, ge::DT_INT8,
    ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_BOOL, ge::DT_BF16
};

static const std::vector<ge::DataType> MASK_SUPPORT_DTYPE = {
    ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL,
    ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL
};

static const std::vector<ge::Format> SUPPORT_FORMAT = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
};

static ge::graphStatus CheckIfAICoreSupported(const ge::Operator& op, ge::AscendString& result) {
    auto xShape = op.GetInputDesc("x").GetShape();
    auto maskShape = op.GetInputDesc("mask").GetShape();
    auto updatesShape = op.GetInputDesc("updates").GetShape();
    size_t xDimNum = xShape.GetDimNum();
    size_t maskDimNum = maskShape.GetDimNum();
    size_t updatesDimNum = updatesShape.GetDimNum();
    if (maskDimNum != xDimNum) {
        result = ge::AscendString(
            R"({"isSupported": "False", "dynamicCompileStatic": "True", "reason": "Unsupported Shape."})");
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < maskDimNum - 1; i++) {
        if (maskShape.GetDim(i) != xShape.GetDim(i)) {
            result = ge::AscendString(
                R"({"isSupported": "False", "dynamicCompileStatic": "True", "reason": "Unsupported Shape."})");
            return ge::GRAPH_FAILED;
        }
    }
    if (maskShape.GetDim(maskDimNum - 1) != xShape.GetDim(xDimNum - 1) &&
        !(maskShape.GetDim(maskDimNum - 1) == 1 &&
        xShape.GetDim(xDimNum - 1) == updatesShape.GetDim(updatesDimNum - 1))) {
        result = ge::AscendString(
            R"({"isSupported": "False", "dynamicCompileStatic": "True", "reason": "Unsupported Shape."})");
        return ge::GRAPH_FAILED;
    }
    result = ge::AscendString(
            R"({"isSupported": "True", "dynamicCompileStatic": "True", "reason": "AICore CheckSupport Passed."})");
    return ge::GRAPH_SUCCESS;
}

class MaskedScatter : public OpDef {
 public:
  explicit MaskedScatter(const char* name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .AutoContiguous()
        .DataType(SUPPORT_DTYPE)
        .Format(SUPPORT_FORMAT)
        .UnknownShapeFormat(SUPPORT_FORMAT);
    this->Input("mask")
        .ParamType(REQUIRED)
        .AutoContiguous()
        .DataType(MASK_SUPPORT_DTYPE)
        .Format(SUPPORT_FORMAT)
        .UnknownShapeFormat(SUPPORT_FORMAT);
    this->Input("updates")
        .ParamType(REQUIRED)
        .AutoContiguous()
        .DataType(SUPPORT_DTYPE)
        .Format(SUPPORT_FORMAT)
        .UnknownShapeFormat(SUPPORT_FORMAT);
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType(SUPPORT_DTYPE)
        .Format(SUPPORT_FORMAT)
        .UnknownShapeFormat(SUPPORT_FORMAT);

    OpAICoreConfig aicore_config;
    aicore_config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(false)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(true)
        .ExtendCfgInfo("opFile.value", "masked_scatter_apt");
    this->AICore().AddConfig("ascend950", aicore_config);

    OpAICoreConfig aicore_config_910B;
    aicore_config_910B.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(false)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(true)
        .PrecisionReduceFlag(true);
    this->AICore().AddConfig("ascend910b", aicore_config_910B);
    this->AICore().AddConfig("ascend910_93", aicore_config_910B);
    this->AICore().SetCheckSupport(CheckIfAICoreSupported);
  }
};

OP_ADD(MaskedScatter);

// 手动注册opDef.AICore()里设置的CheckSupport函数
// 需要当前目录下的CMakeLists.txt将本_def.cpp加入${OPHOST_NAME}_tiling_obj编译目标内
static int MASKED_SCATTER_REGISTERED = [](const char* name) {
    MaskedScatter opDef(name);
    optiling::OpCheckFuncHelper(FUNC_CHECK_SUPPORTED, name, opDef.AICore().GetCheckSupport());
    return 0;
}("MaskedScatter");
}  // namespace ops