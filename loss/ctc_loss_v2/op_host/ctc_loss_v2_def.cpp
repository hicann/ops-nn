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
 * \file ctc_loss_v2_def.cpp
 * \brief ctc_loss_v2_def
 */

#include "register/op_def_registry.h"

namespace ops {

static const std::vector<ge::DataType> log_probs = {
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BF16
};

static const std::vector<ge::DataType> targets = {
    ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64,
};

static const std::vector<ge::Format> formats = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
};

class CTCLossV2 : public OpDef {
    public:
        explicit CTCLossV2(const char* name) : OpDef(name)
        {
            this->Input("log_probs")
                    .ParamType(REQUIRED)
                    .DataType(log_probs)
                    .Format(formats);
            this->Input("targets")
                    .ParamType(REQUIRED)
                    .DataType(targets)
                    .Format(formats);
            this->Input("input_lengths")
                    .ParamType(REQUIRED)
                    .DataType(targets)
                    .Format(formats)
                    .ValueDepend(OPTIONAL);
            this->Input("target_lengths")
                    .ParamType(REQUIRED)
                    .DataType(targets)
                    .Format(formats)
                    .ValueDepend(OPTIONAL);
            this->Output("neg_log_likelihood")
                    .ParamType(REQUIRED)
                    .DataType(log_probs)
                    .Format(formats);
            this->Output("log_alpha")
                    .ParamType(REQUIRED)
                    .DataType(log_probs)
                    .Format(formats);
            this->Attr("blank")
                    .AttrType(OPTIONAL)
                    .Int(0);
            this->Attr("reduction")
                    .AttrType(OPTIONAL)
                    .String("none");
            this->Attr("zero_infinity")
                    .AttrType(OPTIONAL)
                    .Bool(false);
            OpAICoreConfig aicoreConfig;
            aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "ctc_loss_v2_apt");
            this->AICore().AddConfig("ascend910_95", aicoreConfig);
        }
    };

    OP_ADD(CTCLossV2);

}