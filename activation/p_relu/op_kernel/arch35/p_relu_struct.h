/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file p_relu_struct.h
 * \brief prelu tiling struct
 */

#ifndef OP_NN_ACTIVATION_P_RELU_OP_KERNEL_PRELU_STRUCT_H_
#define OP_NN_ACTIVATION_P_RELU_OP_KERNEL_PRELU_STRUCT_H_

#include "atvoss/broadcast/broadcast_base_struct.h"

namespace PreluOp {
ASCENDC_TPL_ARGS_DECL(PRelu, BRC_TEMP_SCH_MODE_KEY_DECL(schMode));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(BRC_TEMP_SCH_MODE_KEY_SEL(schMode)));
} // namespace PreluOp
#endif //  OP_NN_ACTIVATION_P_RELU_OP_KERNEL_PRELU_STRUCT_H_