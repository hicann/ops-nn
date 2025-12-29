/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
class MaxPool3D : public OpDef {
public:
    const std::vector<ge::DataType> maxPool3DXDataType = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
    const std::vector<ge::Format> maxPool3DXFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
    explicit MaxPool3D(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(maxPool3DXDataType)
            .Format(maxPool3DXFormat)
            .UnknownShapeFormat(maxPool3DXFormat)
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(maxPool3DXDataType)
            .Format(maxPool3DXFormat)
            .UnknownShapeFormat(maxPool3DXFormat)
            .AutoContiguous();
        this->Attr("ksize").AttrType(REQUIRED).ListInt();
        this->Attr("strides").AttrType(REQUIRED).ListInt();
        this->Attr("padding").AttrType(REQUIRED).String();
        this->Attr("pads").AttrType(OPTIONAL).ListInt({0, 0, 0, 0, 0, 0});
        this->Attr("dilation").AttrType(OPTIONAL).ListInt({1, 1, 1, 1, 1});
        this->Attr("ceil_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("data_format").AttrType(OPTIONAL).String("NDHWC");
        OpAICoreConfig aiCoreConfig;
        aiCoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "max_pool3_d_apt");
        this->AICore().AddConfig("ascend910_95", aiCoreConfig);
        this->AICore().AddConfig("mc62cm12a", aiCoreConfig);
    }
};

OP_ADD(MaxPool3D);
} // namespace ops