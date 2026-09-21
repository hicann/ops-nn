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
 * \file lamb_update_with_lr_tiling_key.h
 * \brief lamb_update_with_lr_tiling_key head file
 */

#ifndef ADAM_APPLY_ONE_STRUCT_H
#define ADAM_APPLY_ONE_STRUCT_H

#include "atvoss/broadcast/broadcast_base_struct.h"

using namespace Ops::Base;
// 算子自定义的tiling key字段
// 只声明 NDDMA 系模式: UB 广播模式(101~109)要求 DAG 含 Vec::Brc 节点(OpDag::VecBrcSize > 0),
// 而 atvoss/util/vec.h 里 Vec::Brc 的实现体是注释掉的; 本族算子的广播输入一律用 Vec::CopyInBrc
// (NDDMA 系)。实测把内核类型放开成 KERNEL_TYPE_BOTH 后, 框架会给**不需要广播的常规档**也选
// UB 模式, 命中率直接掉到 0%(lamb_next_right 108->0, v2 104->0), 故必须锁定 NDDMA。
// 代价: rank1((5,)->(2,3,4,5)) / tail1((4,8,1)->(4,8,16)) 这两类广播形态当前不被支持。
ASCENDC_TPL_ARGS_DECL(LambUpdateWithLr, BRC_NDDMA_SCH_MODE_KEY_DECL(schMode));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(BRC_NDDMA_SCH_MODE_KEY_SEL(schMode)));

#endif // ADAM_APPLY_ONE_STRUCT_H
