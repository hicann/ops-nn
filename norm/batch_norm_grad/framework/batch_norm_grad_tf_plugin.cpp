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
 * \file batch_norm_grad_tf_plugin.cpp
 * \brief
 */
#include "register/register.h"

namespace domi {
// BatchNormGrad maps directly from the TensorFlow op of the same name; auto operator mapping suffices.
REGISTER_CUSTOM_OP("BatchNormGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("FusedBatchNormGradV3")
    .ParseParamsByOperatorFn(AutoMappingByOpFn)
    .DelInputWithOriginalType(5, "FusedBatchNormGradV3")
    .ImplyType(ImplyType::TVM);
} // namespace domi
