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
 * \file in_training_update_grad_gamma_beta_def.cpp
 * \brief Operator definition for INTrainingUpdateGradGammaBeta.
 */

#include <initializer_list>

#include "register/op_def_registry.h"

namespace ops {
class INTrainingUpdateGradGammaBeta : public OpDef {
public:
    explicit INTrainingUpdateGradGammaBeta(const char* name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> dataTypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                                                               ge::DT_FLOAT};
        const std::initializer_list<ge::Format> formats = {ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NCDHW,
                                                           ge::FORMAT_NDHWC, ge::FORMAT_ND};

        this->Input("res_gamma").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("res_beta").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("pd_gamma").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("pd_beta").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("opFile.value", "in_training_update_grad_gamma_beta");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(INTrainingUpdateGradGammaBeta);
} // namespace ops
