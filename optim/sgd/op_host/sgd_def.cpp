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
 * \file sgd_def.cpp
 * \brief SGD OpDef —— 仅适配 Ascend950（arch35 / DAV_3510 / regbase）
 *
 * 支持面对齐 910B/910C 基线（canndev aic-ascend910b-ops-info.ini 的 [SGD] 段），
 * 唯一收窄项是 format：A2 大张量支持 NC1HWC0 / NDC1HWC0 / ND / FRACTAL_Z /
 * FRACTAL_Z_3D 五种，本算子只做 ND。依据是 ops-nn 仓内 arch35 全族 optim 算子
 * （apply_momentum / apply_ftrl / apply_adam_w_v2 / apply_adamax /
 * apply_centered_rms_prop）的 def.cpp 一律只声明 ge::FORMAT_ND，无一例声明私有
 * format —— 跟随本仓 arch35 既定约定，非自行设计。已于 CP1 批准。
 *
 * R2（A5 不碰 A2）天然满足：ops-nn 仓内不存在 SGD 算子（optim/fused_sgd 是语义
 * 不同的另一个算子，且无 arch35 目录），本算子为净新增，只 AddConfig("ascend950")，
 * 不触碰任何 A2 配置（先例：apply_momentum、apply_adamax 亦只有 ascend950 config）。
 *
 * accum / stat 【不声明为 Output】—— 与 A2 形态一致（canndev proto
 * nn_training_ops.h:1431 亦只有一个 OUTPUT），两者靠覆写输入 GM 原地回写。
 * 该"proto 与实现不自洽"是本族常态，已在 README 显式声明。
 */
#include "register/op_def_registry.h"

namespace ops {

class SGD : public OpDef {
public:
    explicit SGD(const char* name) : OpDef(name)
    {
        this->Input("parameters")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("gradient")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("learning_rate")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("accum")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("momentum")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("stat")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("parameters")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        // 属性默认值与 A2 逐字一致（aic-ascend910b-ops-info.ini 的 [SGD] 段）
        this->Attr("dampening").AttrType(OPTIONAL).Float(0.0);
        this->Attr("weight_decay").AttrType(OPTIONAL).Float(0.0);
        this->Attr("nesterov").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)  // 支持 -2（UNKNOWN_RANK）透传
            .DynamicShapeSupportFlag(true) // 支持 -1（UNKNOWN_DIM）
            .PrecisionReduceFlag(false)    // 对齐 A2 precision_reduce.flag=false
            .ExtendCfgInfo("opInterface.value", "sgd");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(SGD);
} // namespace ops
