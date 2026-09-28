/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_augru_grad_def.cpp
 * \brief DynamicAUGRUGrad算子定义：AUGRU反向（BPTT），融合seq_length掩码生成
 */

#include "register/op_def_registry.h"

namespace ops {
class DynamicAUGRUGrad : public OpDef {
public:
    explicit DynamicAUGRUGrad(const char* name) : OpDef(name)
    {
        RegisterInputs();
        RegisterOutputs();
        RegisterAttributes();
        RegisterPlatformConfig();
    }

private:
    void AddRequiredInput(const char* name)
    {
        this->Input(name)
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
    }

    void RegisterInputs()
    {
        for (const char* name : {"x", "weight_input", "weight_hidden", "weight_att", "y", "init_h", "h", "dy", "dh",
                                 "update", "update_att", "reset", "new", "hidden_new"}) {
            AddRequiredInput(name);
        }
        this->Input("seq_length")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("mask")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_UINT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
    }

    void AddRequiredOutput(const char* name)
    {
        this->Output(name)
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    }

    void RegisterOutputs()
    {
        for (const char* name : {"dw_input", "dw_hidden", "db_input", "db_hidden", "dx", "dh_prev", "dw_att"}) {
            AddRequiredOutput(name);
        }
    }

    void RegisterAttributes()
    {
        this->Attr("direction").AttrType(OPTIONAL).String("UNIDIRECTIONAL");
        this->Attr("cell_depth").AttrType(OPTIONAL).Int(1);
        this->Attr("keep_prob").AttrType(OPTIONAL).Float(-1.0);
        this->Attr("cell_clip").AttrType(OPTIONAL).Float(-1.0);
        this->Attr("num_proj").AttrType(OPTIONAL).Int(0);
        this->Attr("time_major").AttrType(OPTIONAL).Bool(true);
        this->Attr("gate_order").AttrType(OPTIONAL).String("zrh");
        this->Attr("reset_after").AttrType(OPTIONAL).Bool(true);
    }

    void RegisterPlatformConfig()
    {
        OpAICoreConfig aiCoreConfig;
        aiCoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            // 浮点输入契约要求与x同dtype；开启precision_reduce时GE会把混用dtype静默
            // cast后继续匹配执行（binary模式下InferDataType被跳过），语义错误须显式拒绝
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("opFile.value", "dynamic_augru_grad");
        this->AICore().AddConfig("ascend950", aiCoreConfig);
    }
};
OP_ADD(DynamicAUGRUGrad);
} // namespace ops
