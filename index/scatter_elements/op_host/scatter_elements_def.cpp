/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_elements_def.cpp
 * \brief scatter_elements_def
 */
#include <vector>
#include "register/op_def_registry.h"

namespace ops
{
class ScatterElements : public OpDef
{
public:
    explicit ScatterElements(const char* name) : OpDef(name)
    {
        const std::vector<ge::DataType> commonDataTypes = {ge::DT_INT64, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8, ge::DT_FLOAT,
                       ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT64, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8,
                       ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};

        const std::vector<ge::Format> commonFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

        const std::vector<ge::DataType> indicesTypes = {ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                       ge::DT_INT32, ge::DT_INT32};

        const std::vector<ge::Format> commonUnknownShapeFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND};

        this->Input("data").ParamType(REQUIRED).DataType(commonDataTypes).Format(commonFormat).UnknownShapeFormat(commonUnknownShapeFormat);

        this->Input("indices").ParamType(REQUIRED).DataType(indicesTypes).Format(commonFormat).UnknownShapeFormat(commonUnknownShapeFormat);

        this->Input("updates").ParamType(REQUIRED).DataType(commonDataTypes).Format(commonFormat).UnknownShapeFormat(commonUnknownShapeFormat);

        this->Output("y").ParamType(REQUIRED).DataType(commonDataTypes).Format(commonFormat).UnknownShapeFormat(commonUnknownShapeFormat);

        this->Attr("axis").AttrType(OPTIONAL).Int(0);
        this->Attr("reduction").AttrType(OPTIONAL).String("none");
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "scatter_elements_apt");
        this->AICore().AddConfig("ascend910_95", aicoreConfig);
        this->AICore().AddConfig("mc62cm12a", aicoreConfig);
    }
};

OP_ADD(ScatterElements);
}  // namespace ops