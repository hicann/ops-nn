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

namespace ops {
static const std::vector<ge::DataType> kDataType = {ge::DT_FLOAT, ge::DT_FLOAT16};
static const std::vector<ge::Format> kDataFormat = {ge::FORMAT_ND, ge::FORMAT_ND};

class Centralization : public OpDef {
public:
    explicit Centralization(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType(kDataType).Format(kDataFormat);
        this->Output("y").ParamType(REQUIRED).DataType(kDataType).Format(kDataFormat);
        this->Attr("axes").AttrType(OPTIONAL).ListInt({-1});

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .PrecisionReduceFlag(false);
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(Centralization);
} // namespace ops
