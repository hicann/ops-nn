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
namespace {
const std::vector<ge::DataType> VALUE_TYPES = {ge::DT_FLOAT,  ge::DT_FLOAT16, ge::DT_BF16,   ge::DT_INT8,
                                               ge::DT_UINT8,  ge::DT_INT16,   ge::DT_UINT16, ge::DT_INT32,
                                               ge::DT_UINT32, ge::DT_INT64,   ge::DT_UINT64};
const std::vector<ge::Format> FORMATS(VALUE_TYPES.size(), ge::FORMAT_ND);
const std::vector<ge::DataType> INDEX_TYPES(VALUE_TYPES.size(), ge::DT_INT64);

ge::graphStatus CheckSupport(const ge::Operator&, ge::AscendString& result)
{
    // This implementation has not yet integrated compute-dependent output
    // allocation and shape updates into GE's AICore execution path. Keep GE
    // (including the ONNX Unique mapping) on the existing AICPU implementation.
    // ACLNN allocates that buffer explicitly and selects AICore independently
    // in LaunchValuesOnly; this GE selection guard does not disable that path.
    result = ge::AscendString(
        R"({"ret_code":1,"reason":"GE AICore dynamic-output integration is not implemented; retain AICPU"})");
    return ge::GRAPH_FAILED;
}
} // namespace

class UniqueWithCountsAndSorting : public OpDef {
public:
    explicit UniqueWithCountsAndSorting(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType(VALUE_TYPES).Format(FORMATS).UnknownShapeFormat(FORMATS);
        this->Output("y")
            .OutputShapeDependOnCompute()
            .ParamType(REQUIRED)
            .DataType(VALUE_TYPES)
            .Format(FORMATS)
            .UnknownShapeFormat(FORMATS);
        this->Output("indices")
            .OutputShapeDependOnCompute()
            .ParamType(REQUIRED)
            .DataType(INDEX_TYPES)
            .Format(FORMATS)
            .UnknownShapeFormat(FORMATS);
        this->Output("counts")
            .OutputShapeDependOnCompute()
            .ParamType(REQUIRED)
            .DataType(INDEX_TYPES)
            .Format(FORMATS)
            .UnknownShapeFormat(FORMATS);
        // Keep the existing GE prototype's attribute order and defaults.
        this->Attr("return_inverse").AttrType(OPTIONAL).Bool(false);
        this->Attr("return_counts").AttrType(OPTIONAL).Bool(false);
        this->Attr("sorted").AttrType(OPTIONAL).Bool(true);
        this->Attr("out_idx").AttrType(OPTIONAL).Int(ge::DT_INT64);
        this->AICore().SetCheckSupport(CheckSupport);
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "unique_with_counts_and_sorting");
        this->AICore().AddConfig("ascend950", config);
        this->AICore().AddConfig("ascend350", config);
    }
};
OP_ADD(UniqueWithCountsAndSorting);
} // namespace ops
