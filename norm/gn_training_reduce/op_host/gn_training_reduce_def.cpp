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
 * \file gn_training_reduce_def.cpp
 * \brief GNTrainingReduce operator definition (OpDef).
 */

#include "register/op_def_registry.h"

namespace ops {

class GNTrainingReduce : public OpDef {
public:
    explicit GNTrainingReduce(const char* name) : OpDef(name)
    {
        // —— Input x：dtype {fp16, fp32} × format {NCHW, NHWC} 笛卡尔展开（4 列，与「算子规格」组合列对齐）——
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NCHW, ge::FORMAT_NHWC})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // —— Output sum：恒 fp32、format ND（4 列与 Input 等长）——
        this->Output("sum")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // —— Output square_sum：恒 fp32、format ND，shape 同 sum ——
        this->Output("square_sum")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // —— Attr num_groups：可选 Int，默认 2，须整除 C，须与 GNTrainingUpdate 一致 ——
        this->Attr("num_groups").AttrType(OPTIONAL).Int(2);

        // —— AICore 配置（ascend950 / arch35·DAV_3510）——
        OpAICoreConfig aicoreConfig;
        aicoreConfig
            .DynamicCompileStaticFlag(true) // 静态 shape 统一 kernel，避免 per-shape JIT
            .DynamicFormatFlag(false)       // 不支持动态格式转换（输入固定 NCHW/NHWC）
            .DynamicRankSupportFlag(true)   // rank 由算子校验固定为 4，此处保留框架能力
            .DynamicShapeSupportFlag(true)  // 支持可变 shape（N/C/H/W 可变）
            .NeedCheckSupportFlag(false)    // 跳过框架 support check
            .ExtendCfgInfo("opFile.value", "gn_training_reduce_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};
OP_ADD(GNTrainingReduce);

} // namespace ops
