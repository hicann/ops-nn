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
 * \file in_training_update_grad_gamma_beta_tiling_key.h
 * \brief Kernel template selector declaration for Ascend 950.
 */

#ifndef IN_TRAINING_UPDATE_GRAD_GAMMA_BETA_TILING_KEY_H_
#define IN_TRAINING_UPDATE_GRAD_GAMMA_BETA_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(INTrainingUpdateGradGammaBeta, ASCENDC_TPL_UINT_DECL(DTYPE_MODE, 8, ASCENDC_TPL_UI_LIST, 0));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(DTYPE_MODE, ASCENDC_TPL_UI_LIST, 0)), );

#endif // IN_TRAINING_UPDATE_GRAD_GAMMA_BETA_TILING_KEY_H_
