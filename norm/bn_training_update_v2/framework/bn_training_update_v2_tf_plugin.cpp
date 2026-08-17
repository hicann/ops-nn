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
 * \file bn_training_update_v2_tf_plugin.cpp
 * \brief BNTrainingUpdateV2 tf plugin（逐字 mirror canndev
 *        ops/built-in/framework/tf_plugin/bn_training_update_v2_plugin.cc，
 *        与 A2 既有已评审实现保持一致）
 */
#include "register/register.h"

namespace domi {
REGISTER_CUSTOM_OP("BNTrainingUpdateV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("BNTrainingUpdateV2")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::TVM);
} // namespace domi
