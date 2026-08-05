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
 * \file l2_normalize_grad_def.cpp
 * \brief L2NormalizeGrad op definition (arch35-only / Ascend950).
 *
 * Prototype identical to the ascend910b baseline (x, y, dy) -> dx, attrs dim (ListInt) / eps (Float),
 * dtypes {float32, float16} (no bf16, following ascend910b check_dtype). No prototype diff vs ascend910b, so no
 * soc isolation: a single ascend950 AICore config is added; the kernel is isolated via
 * op_kernel/arch35 + add_kernel_sources(... ascend950).
 */
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> dataType = {ge::DT_FLOAT, ge::DT_FLOAT16};
static const std::vector<ge::Format> dataFormat = {ge::FORMAT_ND, ge::FORMAT_ND};

class L2NormalizeGrad : public OpDef {
public:
    explicit L2NormalizeGrad(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType(dataType).Format(dataFormat);
        this->Input("y").ParamType(REQUIRED).DataType(dataType).Format(dataFormat);
        this->Input("dy").ParamType(REQUIRED).DataType(dataType).Format(dataFormat);
        this->Output("dx").ParamType(REQUIRED).DataType(dataType).Format(dataFormat);
        this->Attr("dim").AttrType(OPTIONAL).ListInt({1});
        this->Attr("eps").AttrType(OPTIONAL).Float(1e-4f);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .PrecisionReduceFlag(false);
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(L2NormalizeGrad);
} // namespace ops
