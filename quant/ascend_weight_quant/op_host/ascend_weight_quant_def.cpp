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
 * \file ascend_weight_quant_def.cpp
 *
 *   Supported (x dtype, offset dtype) -> y dtype on ascend950/ascend960dt/ascend960pr:
 *     (int8, int8) -> int8, format HWCN
 *
 *   Attributes:
 *     dst_type: optional int, defaults to DT_INT8 (DT_INT4 is only declared at the
 *     IR layer in the proto for type-check tolerance, not supported by ops-info).
 *
 *   注意：本文件内不可加预处理宏门控——configure 期 npu_op_code_gen 的独立编译
 *   不继承目录级 add_definitions，宏门控会使注册体被裁空、ops-info 生成丢失。
 */
#include "register/op_def_registry.h"

namespace ops {
class AscendWeightQuant : public OpDef {
public:
    explicit AscendWeightQuant(const char* name) : OpDef(name)
    {
        OpAICoreConfig config;
        config.Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_HWCN})
            .UnknownShapeFormat({ge::FORMAT_HWCN});
        config.Input("offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_HWCN})
            .UnknownShapeFormat({ge::FORMAT_HWCN});
        config.Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_HWCN})
            .UnknownShapeFormat({ge::FORMAT_HWCN});
        this->Attr("dst_type").AttrType(OPTIONAL).Int(ge::DT_INT8);

        config.DynamicCompileStaticFlag(false)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(false)
            .DynamicShapeSupportFlag(false)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend950", config);
        this->AICore().AddConfig("ascend960dt", config);
        this->AICore().AddConfig("ascend960pr", config);
    }
};

OP_ADD(AscendWeightQuant);
} // namespace ops
