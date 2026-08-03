/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"

#include <vector>

namespace ops {
namespace {
// A2-compatible combinations: x1/x2 share a dtype and target varies independently.
const std::vector<ge::DataType> kXDtypes = {
    ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_INT32, ge::DT_FLOAT16,
    ge::DT_FLOAT, ge::DT_INT32,   ge::DT_FLOAT16, ge::DT_FLOAT,
};
const std::vector<ge::DataType> kTargetDtypes = {
    ge::DT_INT32,   ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
};
const std::vector<ge::DataType> kOutputDtypes = {
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
};
const std::vector<ge::Format> kNdFormats = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
};
} // namespace

class CosineEmbeddingLoss : public OpDef {
public:
    explicit CosineEmbeddingLoss(const char* name) : OpDef(name)
    {
        this->Input("x1").ParamType(REQUIRED).DataType(kXDtypes).Format(kNdFormats).UnknownShapeFormat(kNdFormats);
        this->Input("x2").ParamType(REQUIRED).DataType(kXDtypes).Format(kNdFormats).UnknownShapeFormat(kNdFormats);
        this->Input("target")
            .ParamType(REQUIRED)
            .DataType(kTargetDtypes)
            .Format(kNdFormats)
            .UnknownShapeFormat(kNdFormats);
        this->Output("y").ParamType(REQUIRED).DataType(kOutputDtypes).Format(kNdFormats).UnknownShapeFormat(kNdFormats);
        this->Attr("margin").AttrType(OPTIONAL).Float(0.0f);
        this->Attr("reduction").AttrType(OPTIONAL).String("mean");

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .PrecisionReduceFlag(false);
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(CosineEmbeddingLoss);
} // namespace ops
