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
 * \file trans_quant_param_v2.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class TransQuantParamV2 : public OpDef {
public:
    explicit TransQuantParamV2(const char* name) : OpDef(name)
    {
        this->Input("scale").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("offset").ParamType(OPTIONAL).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_UINT64}).Format({ge::FORMAT_ND});
        this->Attr("round_mode").AttrType(OPTIONAL).Int(0);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");

        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
        this->AICore().AddConfig("ascend910_95", aicore_config);
        this->AICore().AddConfig("ascend310p", aicore_config);

        OpAICoreConfig config_kirin = GetKirinCoreConfig();
        this->AICore().AddConfig("kirinx90", config_kirin);
    }

private:
    OpAICoreConfig GetKirinCoreConfig() const
    {
        OpAICoreConfig config_kirin;
        config_kirin.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        config_kirin.Input("scale").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        config_kirin.Input("offset").ParamType(OPTIONAL).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        config_kirin.Output("y").ParamType(REQUIRED).DataType({ge::DT_UINT64}).Format({ge::FORMAT_ND});
        return config_kirin;
    }
};

OP_ADD(TransQuantParamV2);
} // namespace ops