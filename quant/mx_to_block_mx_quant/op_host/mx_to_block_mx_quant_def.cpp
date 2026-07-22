/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mx_to_block_mx_quant_def.cpp
 * \brief
 */

#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
static constexpr int32_t DEFAULT_DST_TYPE = 36;

static const std::vector<ge::DataType> mxToBlockMxQuantXDataType = {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2,
                                                                    ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2};

static const std::vector<ge::DataType> mxToBlockMxQuantScaleDataType = {ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                                                                        ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0};

static const std::vector<ge::DataType> mxToBlockMxQuantYDataType = {ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2,
                                                                    ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN};

static const std::vector<ge::Format> mxToBlockMxQuantNDFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                                 ge::FORMAT_ND};

class MxToBlockMxQuant : public OpDef {
public:
    explicit MxToBlockMxQuant(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(mxToBlockMxQuantXDataType)
            .Format(mxToBlockMxQuantNDFormat)
            .UnknownShapeFormat(mxToBlockMxQuantNDFormat)
            .AutoContiguous();
        this->Input("mxscale")
            .ParamType(REQUIRED)
            .DataType(mxToBlockMxQuantScaleDataType)
            .Format(mxToBlockMxQuantNDFormat)
            .UnknownShapeFormat(mxToBlockMxQuantNDFormat)
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(mxToBlockMxQuantYDataType)
            .Format(mxToBlockMxQuantNDFormat)
            .UnknownShapeFormat(mxToBlockMxQuantNDFormat);
        this->Output("scale1")
            .ParamType(REQUIRED)
            .DataType(mxToBlockMxQuantScaleDataType)
            .Format(mxToBlockMxQuantNDFormat)
            .UnknownShapeFormat(mxToBlockMxQuantNDFormat);
        this->Output("scale2")
            .ParamType(REQUIRED)
            .DataType(mxToBlockMxQuantScaleDataType)
            .Format(mxToBlockMxQuantNDFormat)
            .UnknownShapeFormat(mxToBlockMxQuantNDFormat);
        this->Attr("dst_type").AttrType(OPTIONAL).Int(DEFAULT_DST_TYPE);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "mx_to_block_mx_quant");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(MxToBlockMxQuant);
} // namespace ops
