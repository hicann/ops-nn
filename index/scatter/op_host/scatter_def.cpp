/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_def.cpp
 * \brief
 */
#include <vector>
#include "register/op_def_registry.h"

namespace ops {

static const int64_t AXIS_DEFAULT = 0;

static const std::vector<ge::DataType> varDataType = {
    ge::DT_INT8, ge::DT_UINT8, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32,
    ge::DT_INT8, ge::DT_UINT8, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32,
    ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E8M0,
    ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E8M0
};

static const std::vector<ge::DataType> indicesDataType = {
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64
};

static const std::vector<ge::DataType> updatesDataType = {
    ge::DT_INT8, ge::DT_UINT8, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32,
    ge::DT_INT8, ge::DT_UINT8, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32,
    ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E8M0,
    ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E8M0
};

static const std::vector<ge::Format> format = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
};

class Scatter : public OpDef {
public:
    explicit Scatter(const char* name) : OpDef(name) {
        this->Input("var")
            .ParamType(REQUIRED)
            .DataType(varDataType)
            .Format(format)
            .UnknownShapeFormat(format);
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType(indicesDataType)
            .Format(format)
            .UnknownShapeFormat(format);
        this->Input("updates")
            .ParamType(REQUIRED)
            .DataType(updatesDataType)
            .Format(format)
            .UnknownShapeFormat(format);
        this->Output("var")
            .ParamType(REQUIRED)
            .DataType(varDataType)
            .Format(format)
            .UnknownShapeFormat(format);
        this->Attr("reduce")
            .AttrType(REQUIRED)
            .String("update");
        this->Attr("axis")
            .AttrType(OPTIONAL)
            .Int(AXIS_DEFAULT);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "scatter_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(Scatter);
}  // namespace ops