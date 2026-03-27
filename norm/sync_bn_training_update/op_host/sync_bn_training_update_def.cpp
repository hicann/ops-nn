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
 * \file sync_bn_training_update.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
constexpr float MOMENTUM_VALUE = 0.1;
class SyncBNTrainingUpdate : public OpDef {
public:
    explicit SyncBNTrainingUpdate(const char *name) : OpDef(name)
    {
        this->Input("mean")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
        this->Input("running_mean")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
        this->Output("running_mean_update")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
        this->Attr("momentum").AttrType(OPTIONAL).Float(MOMENTUM_VALUE);
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "sync_bn_training_update_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(SyncBNTrainingUpdate);
} // namespace ops
