/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file smooth_l1_loss_v2_def.cpp
 * \brief smooth_l1_loss_v2 def
 */

#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
class SmoothL1LossV2 : public OpDef {
public:
    explicit SmoothL1LossV2(const char *name) : OpDef(name)
    {
        this->Input("predict")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
        this->Input("label")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
        this->Attr("sigma").AttrType(OPTIONAL).Float(1.0);
        this->Attr("reduction").AttrType(OPTIONAL).String("mean");

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("opFile.value", "smooth_l1_loss_v2_apt");
        this->AICore().AddConfig("ascend910_95", aicoreConfig);
    }
};

OP_ADD(SmoothL1LossV2);
} // namespace ops