/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "lstm_backward_plan_spy.h"

// Give this copy private link symbols while preserving __func__ for DFX checks.
// The production API body and its validation are compiled unchanged.
extern "C" {
decltype(aclnnLstmBackwardGetWorkspaceSize) aclnnLstmBackwardGetWorkspaceSize asm("LstmBackwardPlanGetWorkspaceSize");
decltype(aclnnLstmBackward) aclnnLstmBackward asm("LstmBackwardPlan");
}
#define SingleLayerLstmGrad LstmBackwardPlanGrad
#define ResetAndReshapeTensor LstmBackwardPlanResetAndReshapeTensor
#define PrepareLSTMBackwardNoneInputs LstmBackwardPlanPrepareNoneInputs
#include "../../../op_api/aclnn_lstm_backward.cpp"
#undef PrepareLSTMBackwardNoneInputs
#undef ResetAndReshapeTensor
#undef SingleLayerLstmGrad
