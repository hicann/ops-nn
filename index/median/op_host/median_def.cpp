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
static const std::vector<ge::DataType> MedianDataTypes = {ge::DT_FLOAT16, ge::DT_FLOAT,  ge::DT_INT16, ge::DT_INT8,
                                                          ge::DT_UINT8,   ge::DT_INT32,  ge::DT_INT64, ge::DT_BF16,
                                                          ge::DT_UINT32,  ge::DT_UINT16, ge::DT_UINT64};
static const std::vector<ge::DataType> MedianIndexTypes(MedianDataTypes.size(), ge::DT_INT64);
static const std::vector<ge::Format> MedianFormats(MedianDataTypes.size(), ge::FORMAT_ND);

class Median : public OpDef {
public:
    explicit Median(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(MedianDataTypes)
            .Format(MedianFormats)
            .UnknownShapeFormat(MedianFormats);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(MedianDataTypes)
            .Format(MedianFormats)
            .UnknownShapeFormat(MedianFormats);
        this->Output("indices")
            .ParamType(REQUIRED)
            .DataType(MedianIndexTypes)
            .Format(MedianFormats)
            .UnknownShapeFormat(MedianFormats);
        this->Attr("dim").AttrType(OPTIONAL).Int(-1);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "median")
            .ExtendCfgInfo("opInterface.value", "median");
        this->AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(Median);
} // namespace ops
