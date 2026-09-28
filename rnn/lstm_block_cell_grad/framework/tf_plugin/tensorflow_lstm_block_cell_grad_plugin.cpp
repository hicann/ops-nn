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
 * \file tensorflow_lstm_block_cell_grad_plugin.cpp
 * \brief TensorFlow -> CANN 映射插件. LSTMBlockCellGrad, 直通.
 *
 * 同名认领: 本算子实现 TF 原生算子 tf.raw_ops.LSTMBlockCellGrad, 交付目标
 * 就是认领 TF 图里的 LSTMBlockCellGrad 节点并路由到本算子——
 * REGISTER_CUSTOM_OP("LSTMBlockCellGrad").OriginOpType("LSTMBlockCellGrad").
 * adapter 的白名单按 TF 节点名查表、表按 CANN OpType 生成, TF 节点名与
 * CANN OpType 必须逐字一致——不同名则节点静默留在 CPU 上, 不报错.
 *
 * AutoMappingByOpFn 按位置匹配输入、按名字匹配属性, CANN 侧原型与 TF OpDef
 * 是同一份签名 (16 进 5 出 1 bool attr), 无须 ParseOpToGraphFn 建子图.
 *
 * ⚠ 本文件只 include CANN 的 register/register.h, 不 include 任何 TensorFlow
 * 头. 编译打包不需要机器上装 TensorFlow; 只有造图与运行才需要 TF.
 *
 * ⚠ 扩展名 .cpp 是构建强制的, 不是风格选择. 本仓 TF 插件由仓级
 * cmake/func.cmake add_tf_plugin_sources 统一 glob 收源:
 *   file(GLOB TF_PLUGIN_SRCS ${FRAMEWORK_DIR}/*_tf_plugin.cpp
 *        ${FRAMEWORK_DIR}/tf_plugin/*.cpp)
 * 只收 .cpp——.cc 会被静默漏掉: 构建照样 RC=0, 包照样装得上, 只是插件不在
 * 里面, 症状是「TF 图节点无人认领」. 仓内其余 26 个 *_tf_plugin.cpp 同规.
 *
 * ImplyType::TVM: 本算子按 tbe 树布局打包 (op_impl/ai_core/tbe/...),
 * TF parser/OME 据此归类执行路线; 与仓内全部 *_tf_plugin.cpp 注册写法一致
 * (26/26), 缺省不写则 imply_type 依赖库内默认值, 归类不确定.
 */

#include "register/register.h"

namespace domi {

REGISTER_CUSTOM_OP("LSTMBlockCellGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("LSTMBlockCellGrad")
    .ParseParamsByOperatorFn(AutoMappingByOpFn)
    .ImplyType(ImplyType::TVM);

} // namespace domi
