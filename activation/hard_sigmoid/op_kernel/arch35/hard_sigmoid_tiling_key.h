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
 * \file hard_sigmoid_tiling_key.h
 * \brief HardSigmoid TilingKey 声明。
 *
 * D_T_X 作为单轴 tiling-key，绑定 input0 的 dtype 并枚举各 dtype 的计算路径。这里使用
 * DATATYPE 模板轴而不是自定义 uint 模式轴，保证 binary 构建产物和运行时 tiling 解析能从
 * supportInfo 的输入 dtype 直接选择模板实例。
 */

#ifndef HARD_SIGMOID_TILING_KEY_H
#define HARD_SIGMOID_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

// Kernel UT host 编译时 ASCENDC_CPU_DEBUG 被定义，ASCENDC_TPL_DATATYPE_DECL
// 会展开为引用 C_DT_* 的 ParamStruct 构造；graph/c_types.h 提供这些枚举定义。
// 真机 kernel 编译路径不需要该头文件。
#ifdef ASCENDC_CPU_DEBUG
#include "graph/c_types.h"
#endif

ASCENDC_TPL_ARGS_DECL(HardSigmoid, ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT, C_DT_FLOAT16, C_DT_BF16, C_DT_INT32,
                                                             ASCENDC_TPL_INPUT(0)));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_BF16)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_INT32)), );

#endif // HARD_SIGMOID_TILING_KEY_H
