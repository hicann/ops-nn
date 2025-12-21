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
 * \file conv_bp_register.h
 * \brief
 */

#ifndef CONV_BP_REGISTER_H
#define CONV_BP_REGISTER_H

#include "conv_bp_config_base.h"
#include "conv_bp_impl_base.h"
#include "conv_bp_intf_base.h"

namespace ConvolutionBackprop {
// 注册，通过别名定义用户接口
#define REGISTER_DW_IMPL(name, context, impl, intf)             \
    template <class X_T, class W_TYPE, class DEDY_T, class Y_T> \
    using name = intf<context<X_T, W_TYPE, DEDY_T, Y_T>, impl>
}  // namespace ConvolutionBackprop
#endif
