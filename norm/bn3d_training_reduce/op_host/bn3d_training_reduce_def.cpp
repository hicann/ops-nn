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
 * \file bn3d_training_reduce_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"
#include "bn3d_training_reduce_check_support.h"

namespace ops {
// arch35-only（Ascend950），GE-only：本算子不提供 aclnn 接口。
//
// 公开 origin format 仅支持 NCDHW / NDHWC。当前 CheckSupport 只约束本 custom 候选；
// storage 注册和实现继续保留，平台级多候选拦截需与 built-in 注册联动后才能闭环。
// 输入 x：dtype {fp16, fp32, bf16}，storage format 为 NCDHW / NCHW / NDC1HWC0。
// 输出 sum / square_sum：dtype 恒 DT_FLOAT，format 跟随输入（与 canndev ascend950 op_info 的
// output0.format 逐行等于 input0.format 保持一致）。
//
// 9 行 storage 能力继续与 canndev ascend950 op_info 对齐；该内部能力集合不等于公开 origin 支持面。
class BN3DTrainingReduce : public OpDef {
public:
    explicit BN3DTrainingReduce(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
                     ge::FORMAT_NCHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0})
            .UnknownShapeFormat({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
                                 ge::FORMAT_NCHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0});

        this->Output("sum")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
                     ge::FORMAT_NCHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0})
            .UnknownShapeFormat({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
                                 ge::FORMAT_NCHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0});

        this->Output("square_sum")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
                     ge::FORMAT_NCHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0})
            .UnknownShapeFormat({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
                                 ge::FORMAT_NCHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0});

        // 权威 REG_OP 无属性，此处不得新增 axis / keep_dims / layout / C0 等。

        this->AICore().SetCheckSupport(CheckSupport4BN3DTrainingReduce);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(true)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("opFile.value", "bn3d_training_reduce")
            // OpType "BN3DTrainingReduce" 自动转 snake 会得到 "bn3_d_training_reduce"（BN3D 被切成
            // bn3 + _d），与 Kernel 入口函数名对不上。显式覆盖为 canndev op_info 中登记的
            // opInterface.value=bn3d_training_reduce，与 op_kernel 入口保持一致。
            .ExtendCfgInfo("opInterface.value", "bn3d_training_reduce");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(BN3DTrainingReduce);
} // namespace ops
