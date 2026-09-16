/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file update_tensor_desc_def.cpp
 * \brief UpdateTensorDesc 算子定义：x 为占位输入（kernel 不读取其数据），
 *   输出 y 为 128×int64 描述缓冲区，必填属性 shape 决定 y 的 shape，
 *   y 的 dtype 恒为 DT_INT64（由 InferShape / InferDataType 推导强制）。
 */

#include "register/op_def_registry.h"

namespace ops {

class UpdateTensorDesc : public OpDef {
public:
    explicit UpdateTensorDesc(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_DOUBLE, ge::DT_INT8, ge::DT_INT16,
                       ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8, ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("shape").AttrType(REQUIRED).ListInt();

        OpAICoreConfig aicoreConfig;
        aicoreConfig
            .DynamicCompileStaticFlag(true) // kernel 固定按 128×int64 RMW，与 shape 无关
            .DynamicFormatFlag(false)       // 固定 ND
            .DynamicRankSupportFlag(true)   // x rank 0~8，y rank 由 attr shape 决定
            .DynamicShapeSupportFlag(true)  // 动态 shape 场景算子
            .NeedCheckSupportFlag(false)    // dtype/attr 约束由 host 侧 tiling 校验
            .PrecisionReduceFlag(false)     // 纯 int64 元数据搬运，无浮点通路
            .ExtendCfgInfo("opFile.value", "update_tensor_desc_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};
OP_ADD(UpdateTensorDesc);
} // namespace ops
