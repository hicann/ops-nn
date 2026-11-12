/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANN_OPS_SOFT_MARGIN_LOSS_REDUCE_H
#define CANN_OPS_SOFT_MARGIN_LOSS_REDUCE_H

#include "atvoss/reduce/reduce_operator.h"

namespace SoftMarginLoss {
using namespace AscendC;
using namespace Ops::Base;

template <typename T>
class PairwiseReduceSumOp : public Vec::ReduceSumOp<T> {
public:
    __aicore__ inline PairwiseReduceSumOp() {}
};

} // namespace SoftMarginLoss

#endif // CANN_OPS_SOFT_MARGIN_LOSS_REDUCE_H
