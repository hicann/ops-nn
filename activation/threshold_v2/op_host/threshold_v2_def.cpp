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
static const std::vector<ge::Format> format = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

static const std::vector<ge::DataType> valueType = {ge::DT_INT64,   ge::DT_INT32, ge::DT_INT16, ge::DT_FLOAT,
                                                    ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_INT8,  ge::DT_UINT8};

class ThresholdV2 : public OpDef {
public:
    explicit ThresholdV2(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType(valueType).Format(format).UnknownShapeFormat(format);
        this->Input("threshold").ParamType(REQUIRED).DataType(valueType).Format(format).UnknownShapeFormat(format);
        this->Input("value").ParamType(OPTIONAL).DataType(valueType).Format(format).UnknownShapeFormat(format);
        this->Output("y").ParamType(REQUIRED).DataType(valueType).Format(format).UnknownShapeFormat(format);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend950", aicore_config);
    }
};

OP_ADD(ThresholdV2);
} // namespace ops