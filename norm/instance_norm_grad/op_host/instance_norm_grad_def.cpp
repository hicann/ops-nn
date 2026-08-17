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
 * \file instance_norm_grad_def.cpp
 * \brief InstanceNormGrad op definition (arch35 / Ascend950 only).
 *
 * Prototype strictly follows ascend910b InstanceNormGrad: 5 inputs (dy, x, variance, mean, gamma),
 * 3 outputs (pd_x, pd_gamma, pd_beta), NO attributes (epsilon is hardcoded 1e-6 in the kernel).
 * dtypes are fp16/fp32 ONLY (no bf16). arch35-only: single AddConfig("ascend950").
 */
#include "register/op_def_registry.h"

namespace ops {
// A2(910B) 权威声明 aic-ascend910b-ops-info.ini 里本算子的 format 是 NDHWC(输入输出皆然)，
// 支持面只能宽不能窄，故 dtype/format 两列并列展开成 4 组合：ND 与 NDHWC 各配 fp32/fp16。
static const std::vector<ge::DataType> dataType = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16};
static const std::vector<ge::Format> dataFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC};

class InstanceNormGrad : public OpDef {
public:
    explicit InstanceNormGrad(const char* name) : OpDef(name)
    {
        this->Input("dy").ParamType(REQUIRED).DataType(dataType).Format(dataFormat).UnknownShapeFormat(dataFormat);
        this->Input("x").ParamType(REQUIRED).DataType(dataType).Format(dataFormat).UnknownShapeFormat(dataFormat);
        this->Input("variance")
            .ParamType(REQUIRED)
            .DataType(dataType)
            .Format(dataFormat)
            .UnknownShapeFormat(dataFormat);
        this->Input("mean").ParamType(REQUIRED).DataType(dataType).Format(dataFormat).UnknownShapeFormat(dataFormat);
        this->Input("gamma").ParamType(REQUIRED).DataType(dataType).Format(dataFormat).UnknownShapeFormat(dataFormat);
        this->Output("pd_x").ParamType(REQUIRED).DataType(dataType).Format(dataFormat).UnknownShapeFormat(dataFormat);
        this->Output("pd_gamma")
            .ParamType(REQUIRED)
            .DataType(dataType)
            .Format(dataFormat)
            .UnknownShapeFormat(dataFormat);
        this->Output("pd_beta")
            .ParamType(REQUIRED)
            .DataType(dataType)
            .Format(dataFormat)
            .UnknownShapeFormat(dataFormat);

        OpAICoreConfig regbaseCfg;
        regbaseCfg.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .PrecisionReduceFlag(false);
        this->AICore().AddConfig("ascend950", regbaseCfg);
    }
};

OP_ADD(InstanceNormGrad);
} // namespace ops
