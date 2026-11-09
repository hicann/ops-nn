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
 * \file inplace_apply_adagrad_tf_plugin.cpp
 * \brief InplaceApplyAdagrad TensorFlow plugin mapping.
 */
#include "register/register.h"
#include "log/log.h"
#include "framework/plugin_util.h" /* ShouldUseInplaceOpMapping */

namespace domi {
// Repo-standard inplace mapping: Ascend950 uses the inplace operator, other products keep the built-in op.
// ShouldUseInplaceOpMapping() queries aclrtGetSocName() directly and does not require GEInitialize; the
// custom OPP plugin is loaded before the built-in TF plugins, so the Ascend950 mapping is deterministic.
static bool InitInplaceOpMapping()
{
    const bool useInplace = ShouldUseInplaceOpMapping();
    if (!useInplace) {
        OP_LOGW("InplaceApplyAdagrad",
                "SoC is not Ascend950 or SoC query failed; TF ApplyAdagrad/ResourceApplyAdagrad falls back to "
                "the built-in ApplyAdagrad operator.");
    }
    return useInplace;
}

static bool gUseInplaceOpMapping = InitInplaceOpMapping();

static OpReceiver registerApplyOp = OpRegistrationData(gUseInplaceOpMapping ? "InplaceApplyAdagrad" : "ApplyAdagrad")
                                        .FrameworkType(TENSORFLOW)
                                        .OriginOpType(
                                            std::vector<ge::AscendString>{"ApplyAdagrad", "ResourceApplyAdagrad"})
                                        .ParseParamsByOperatorFn(AutoMappingByOpFn)
                                        .ImplyType(ImplyType::TVM);
} // namespace domi
