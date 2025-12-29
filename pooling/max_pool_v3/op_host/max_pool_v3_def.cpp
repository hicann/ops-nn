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
 * \file max_pool_v3.cpp
 * \brief imply for max_pool
 */

#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
class MaxPoolV3 : public OpDef {
public:
    const std::vector<ge::DataType> maxPoolV3XDataType = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,
                                                          ge::DT_INT32,   ge::DT_INT64, ge::DT_UINT8,
                                                          ge::DT_INT16,   ge::DT_INT8,  ge::DT_UINT16};
    const std::vector<ge::Format> maxPoolV3XFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                      ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                      ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
    explicit MaxPoolV3(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(maxPoolV3XDataType)
            .Format(maxPoolV3XFormat)
            .UnknownShapeFormat(maxPoolV3XFormat)
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(maxPoolV3XDataType)
            .Format(maxPoolV3XFormat)
            .UnknownShapeFormat(maxPoolV3XFormat)
            .AutoContiguous();
        this->Attr("ksize").AttrType(REQUIRED).ListInt();
        this->Attr("strides").AttrType(REQUIRED).ListInt();
        this->Attr("padding_mode").AttrType(OPTIONAL).String("CALCULATED");
        this->Attr("pads").AttrType(OPTIONAL).ListInt({0, 0, 0, 0});
        this->Attr("data_format").AttrType(OPTIONAL).String("NCHW");
        this->Attr("global_pooling").AttrType(OPTIONAL).Bool(false);
        this->Attr("ceil_mode").AttrType(OPTIONAL).Bool(false);
        OpAICoreConfig aiCoreConfig;
        aiCoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "max_pool_v3_apt");
        this->AICore().AddConfig("ascend910_95", aiCoreConfig);
        this->AICore().AddConfig("mc62cm12a", aiCoreConfig);
    }
};

OP_ADD(MaxPoolV3);
} // namespace ops