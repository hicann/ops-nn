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
 * \file log_sigmoid_grad_struct.h
 * \brief log_sigmoid_grad_struct
 */
#ifndef LOGSIGMOID_GRAD_STRUCT_H_
#define LOGSIGMOID_GRAD_STRUCT_H_

#include "ascendc/host_api/tiling/template_argument.h"

namespace LogSigmoidGradOp {

ASCENDC_TPL_ARGS_DECL(LogSigmoidGrad, BRC_TEMP_SCH_MODE_KEY_DECL(schMode));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(BRC_TEMP_SCH_MODE_KEY_SEL(schMode)));
} // namespace LogSigmoidGradOp

#endif // LOGSIGMOID_GRAD_STRUCT_H_
