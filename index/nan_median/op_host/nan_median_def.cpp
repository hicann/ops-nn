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
// Keep the public dtype contract aligned with KthValue. For integer inputs,
// NanMedian is identical to Median and the tiling layer selects the unchanged
// static-k path, so no NaN scan is introduced.
static const std::vector<ge::DataType> NanMedianDataTypes = {ge::DT_FLOAT16, ge::DT_FLOAT,  ge::DT_INT16, ge::DT_INT8,
                                                             ge::DT_UINT8,   ge::DT_INT32,  ge::DT_INT64, ge::DT_BF16,
                                                             ge::DT_UINT32,  ge::DT_UINT16, ge::DT_UINT64};
static const std::vector<ge::DataType> NanMedianIndexTypes(NanMedianDataTypes.size(), ge::DT_INT64);
static const std::vector<ge::Format> NanMedianFormats(NanMedianDataTypes.size(), ge::FORMAT_ND);

class NanMedian : public OpDef {
public:
    explicit NanMedian(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(NanMedianDataTypes)
            .Format(NanMedianFormats)
            .UnknownShapeFormat(NanMedianFormats);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(NanMedianDataTypes)
            .Format(NanMedianFormats)
            .UnknownShapeFormat(NanMedianFormats);
        this->Output("indices")
            .ParamType(REQUIRED)
            .DataType(NanMedianIndexTypes)
            .Format(NanMedianFormats)
            .UnknownShapeFormat(NanMedianFormats);
        this->Attr("dim").AttrType(OPTIONAL).Int(-1);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "nan_median")
            .ExtendCfgInfo("opInterface.value", "nan_median");
        this->AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(NanMedian);
} // namespace ops
