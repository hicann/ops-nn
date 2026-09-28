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
#include <stdexcept>

namespace {
thread_local lstm_test::GradPlanSpy* activeSpy = nullptr;
}

lstm_test::GradPlanSpy::GradPlanSpy()
{
    if (activeSpy != nullptr) {
        throw std::logic_error("Nested LSTM planning spies are not supported");
    }
    activeSpy = this;
}

lstm_test::GradPlanSpy::~GradPlanSpy() { activeSpy = nullptr; }

namespace l0op {
const std::array<const aclTensor*, 5> LstmBackwardPlanGrad(
    const aclTensor* x, const aclTensor* w, const aclTensor* b, const aclTensor* y, const aclTensor* initH,
    const aclTensor* initC, const aclTensor* h, const aclTensor* c, const aclTensor* dy, const aclTensor* dh,
    const aclTensor* dc, const aclTensor* i, const aclTensor* j, const aclTensor* f, const aclTensor* o,
    const aclTensor* tanhc, const aclTensor* seqLength, const char* direction, const char* gateOrder,
    aclOpExecutor* executor)
{
    if (activeSpy == nullptr) {
        throw std::logic_error("LSTM planning entry used without a scoped spy");
    }
    ++activeSpy->calls;
    for (const auto* tensor : {x, initH, initC, dy, dh, dc, i, j, f, o, h, c, tanhc}) {
        activeSpy->homogeneousInputs &= tensor != nullptr && tensor->GetDataType() == w->GetDataType();
    }
    activeSpy->biasElements = b == nullptr ? 0 : b->GetViewShape().GetShapeSize();
    auto result = SingleLayerLstmGrad(x, w, b, y, initH, initC, h, c, dy, dh, dc, i, j, f, o, tanhc, seqLength,
                                      direction, gateOrder, executor);
    if (result[1] != nullptr) {
        activeSpy->biasGradientElements = result[1]->GetViewShape().GetShapeSize();
    }
    return result;
}
} // namespace l0op
