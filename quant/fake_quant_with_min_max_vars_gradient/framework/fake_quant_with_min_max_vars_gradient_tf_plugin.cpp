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
 * \file fake_quant_with_min_max_vars_gradient_tf_plugin.cpp
 * \brief
 */
#include "register/register.h"

namespace domi {
// FakeQuantWithMinMaxVarsGradient maps directly from the TF op of the same name; auto mapping suffices.
REGISTER_CUSTOM_OP("FakeQuantWithMinMaxVarsGradient")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("FakeQuantWithMinMaxVarsGradient")
    .ParseParamsByOperatorFn(AutoMappingByOpFn)
    .ImplyType(ImplyType::TVM);
} // namespace domi
