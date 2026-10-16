/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file add_rms_norm_hi_f4_quant_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"
#include "../../../common/inc/op_host/op_const_def_nn.h"

namespace ops {

static const std::vector<ge::DataType> xDataType = {Ops::NN::DT_FLOAT16, Ops::NN::DT_FLOAT16, Ops::NN::DT_BF16,
                                                    Ops::NN::DT_BF16};

static const std::vector<ge::DataType> gammaDataType = {Ops::NN::DT_FLOAT16, Ops::NN::DT_FLOAT, Ops::NN::DT_BF16,
                                                        Ops::NN::DT_FLOAT};

static const std::vector<ge::DataType> yDataType = {Ops::NN::DT_HIFLOAT4, Ops::NN::DT_HIFLOAT4, Ops::NN::DT_HIFLOAT4,
                                                    Ops::NN::DT_HIFLOAT4};

static const std::vector<ge::DataType> scaleDataType = {Ops::NN::DT_HIFLOAT4_SCALE, Ops::NN::DT_HIFLOAT4_SCALE,
                                                        Ops::NN::DT_HIFLOAT4_SCALE, Ops::NN::DT_HIFLOAT4_SCALE};

static const std::vector<ge::DataType> rstdDataType = {Ops::NN::DT_FLOAT, Ops::NN::DT_FLOAT, Ops::NN::DT_FLOAT,
                                                       Ops::NN::DT_FLOAT};

static const std::vector<ge::Format> formatND = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class AddRmsNormHiF4Quant : public OpDef {
public:
    explicit AddRmsNormHiF4Quant(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType(xDataType)
            .Format(formatND)
            .UnknownShapeFormat(formatND)
            .AutoContiguous();
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType(xDataType)
            .Format(formatND)
            .UnknownShapeFormat(formatND)
            .AutoContiguous();
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType(gammaDataType)
            .Format(formatND)
            .UnknownShapeFormat(formatND)
            .AutoContiguous();
        this->Input("beta")
            .ParamType(OPTIONAL)
            .DataType(gammaDataType)
            .Format(formatND)
            .UnknownShapeFormat(formatND)
            .AutoContiguous();

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(yDataType)
            .Format(formatND)
            .UnknownShapeFormat(formatND)
            .AutoContiguous();
        this->Output("x")
            .ParamType(REQUIRED)
            .DataType(xDataType)
            .Format(formatND)
            .UnknownShapeFormat(formatND)
            .AutoContiguous();
        this->Output("scale")
            .ParamType(REQUIRED)
            .DataType(scaleDataType)
            .Format(formatND)
            .UnknownShapeFormat(formatND)
            .AutoContiguous();
        this->Output("rstd")
            .ParamType(REQUIRED)
            .DataType(rstdDataType)
            .Format(formatND)
            .UnknownShapeFormat(formatND)
            .AutoContiguous();

        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-6);
        this->Attr("output_rstd").AttrType(OPTIONAL).Bool(false);
        this->Attr("scale_alg").AttrType(OPTIONAL).Int(0);

        OpAICoreConfig config960;
        config960.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "add_rms_norm_hi_f4_quant_apt");
        this->AICore().AddConfig("ascend960dt", config960);
    }
};
OP_ADD(AddRmsNormHiF4Quant);
} // namespace ops
