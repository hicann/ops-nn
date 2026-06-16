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
 * \file lamb_apply_optimizer_assign_tf_plugin.cpp
 * \brief
 */
#include "register/register.h"

namespace domi {
// LambApplyOptimizerAssign has only tensor inputs (no attributes), so the auto operator mapping suffices.
REGISTER_CUSTOM_OP("LambApplyOptimizerAssign")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("LambApplyOptimizerAssign")
    .ParseParamsByOperatorFn(AutoMappingByOpFn)
    .ImplyType(ImplyType::CUSTOM);
} // namespace domi
