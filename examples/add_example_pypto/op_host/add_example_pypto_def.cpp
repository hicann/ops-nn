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
 * \file add_example_pypto_def.cpp
 * \brief AddExamplePypto算子的定义
 *
 * 本文件定义了AddExamplePypto算子的接口。该算子与add_example功能一致，区别在于其kernel由
 * PyPTO DSL(op_kernel/add_example_pypto.py)编写，而非AscendC C++。
 */
#include "register/op_def_registry.h"

namespace ops {
/*!
 * \brief AddExamplePypto算子类定义
 *
 * 逐元素加法：接收两个相同shape的输入张量，输出一个相同shape的张量。
 *
 * 支持的数据类型: FLOAT16
 * 支持的格式: ND (n维格式)
 */
class AddExamplePypto : public OpDef {
public:
    explicit AddExamplePypto(const char* name) : OpDef(name)
    {
        // 定义输入x1的规格
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        // 定义输入x2的规格
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        // 定义输出y的规格
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        // AI Core编译配置
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            // opFile.value必须与PyPTO kernel文件名(op_kernel/add_example_pypto.py)以及
            // enable_pypto_kernel(add_example_pypto)保持一致
            .ExtendCfgInfo("opFile.value", "add_example_pypto");
        // 仅注册ascend950。PyPTO按平台自动推导arch(本toolkit下为a5/dav-c310)，与ascend950匹配；
        // 若需支持a3(ascend910b/ascend910_93)，需在@pl.jit中显式指定arch并确认kernel指令集可用。
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};
OP_ADD(AddExamplePypto); // 注册算子到算子信息库
} // namespace ops
