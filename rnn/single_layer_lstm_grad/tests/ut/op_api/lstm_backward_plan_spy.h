/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <array>
#include <cstddef>
#include "../../../op_api/aclnn_lstm_backward.h"
#include "../../../op_api/single_layer_lstm_grad.h"

namespace lstm_test {
// Scoped to one planning assertion, never shared with unrelated operator tests.
// This observes the public API -> l0 boundary, not actual device execution.
//
// The op_api objects are built -fvisibility=hidden, so the out-of-line members below would
// not leave the library the shim is compiled into. The aclnn entry points get their default
// visibility from ACLNN_API; this type has to ask for it.
struct __attribute__((visibility("default"))) GradPlanSpy {
    GradPlanSpy();
    ~GradPlanSpy();
    GradPlanSpy(const GradPlanSpy&) = delete;
    GradPlanSpy& operator=(const GradPlanSpy&) = delete;

    size_t calls = 0;
    bool homogeneousInputs = true;
    int64_t biasElements = 0;
    int64_t biasGradientElements = 0;
};
} // namespace lstm_test

// Test-only link symbols; the API body retains its function names for DFX.
// No ELF interposition or public stub hooks.
extern "C" {
decltype(aclnnLstmBackwardGetWorkspaceSize) LstmBackwardPlanGetWorkspaceSize;
decltype(aclnnLstmBackward) LstmBackwardPlan;
}
namespace l0op {
decltype(SingleLayerLstmGrad) LstmBackwardPlanGrad;
} // namespace l0op
