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
 * \file single_layer_lstm_grad.cpp
 * \brief
 */
#include "single_layer_lstm_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(SingleLayerLstmGrad);

// SingleLayerLstmGrad AICORE算子kernel
const std::array<const aclTensor *, 5> SingleLayerLstmGrad(
    const aclTensor *x, 
    const aclTensor *w, 
    const aclTensor *b, 
    const aclTensor *y,
    const aclTensor *initH, 
    const aclTensor *initC,
    const aclTensor *h,
    const aclTensor *c,
    const aclTensor *dy,
    const aclTensor *dh,
    const aclTensor *dc,
    const aclTensor *i,
    const aclTensor *j,
    const aclTensor *f,
    const aclTensor *o,
    const aclTensor *tanhc,
    const aclTensor *seqLength,
    const char *direction,
    const char *gateOrder,
    aclOpExecutor *executor)
{
    L0_DFX(SingleLayerLstmGrad, x, w, b, initH, initC, dy, dh, dc, y, h, c, i, j, f, o, tanhc, seqLength, direction,
           gateOrder);
    auto dwShape = w->GetViewShape();
    auto dxShape = x->GetViewShape();
    auto dhPrevShape = initC->GetViewShape();

    const aclTensor *dw = executor->AllocTensor(dwShape, x->GetDataType(), op::Format::FORMAT_ND);
    const aclTensor *dx = executor->AllocTensor(dxShape, x->GetDataType(), op::Format::FORMAT_ND);
    const aclTensor *dhPrev = executor->AllocTensor(dhPrevShape, x->GetDataType(), op::Format::FORMAT_ND);
    const aclTensor *dcPrev = executor->AllocTensor(dhPrevShape, x->GetDataType(), op::Format::FORMAT_ND);
    const aclTensor *db = nullptr;
    if (b == nullptr) {
        b = executor->AllocTensor(x->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
        db = executor->AllocTensor(x->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    } else {
        auto dbShape = b->GetViewShape();
        db = executor->AllocTensor(dbShape, x->GetDataType(), op::Format::FORMAT_ND);
    }
    if (seqLength == nullptr) {
        seqLength = executor->AllocTensor(x->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (y == nullptr) {
        y = executor->AllocTensor(x->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(SingleLayerLstmGrad,
                                           OP_INPUT(x, w, b, y, initH, initC, h, c, dy, dh, dc, i, j, f, o, tanhc,
                                                    seqLength),
                                           OP_OUTPUT(dw, db, dx, dhPrev, dcPrev),
                                           OP_ATTR(direction, gateOrder));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SingleLayerLstmGradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr};
    }
    return {dw, db, dx, dhPrev, dcPrev};
}
}  // namespace l0op
