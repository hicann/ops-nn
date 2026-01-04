/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adam_apply_one_with_decay_tiling_key.h
 * \brief adam_apply_one_with_decay_tiling_key head file
 */

#ifndef ADAM_APPLY_ONE_WICH_DECAY_STRUCT_H
#define ADAM_APPLY_ONE_WICH_DECAY_STRUCT_H

#include "atvoss/broadcast/broadcast_base_struct.h"

using namespace Ops::Base;
// 算子自定义的tiling key字段
ASCENDC_TPL_ARGS_DECL(AdamApplyOneWithDecay, BRC_NDDMA_SCH_MODE_KEY_DECL(schMode));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(BRC_NDDMA_SCH_MODE_KEY_SEL(schMode)));

#endif // ADAM_APPLY_ONE_WICH_DECAY_STRUCT_H