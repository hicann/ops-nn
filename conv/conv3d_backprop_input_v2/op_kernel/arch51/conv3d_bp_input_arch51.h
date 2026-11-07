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
 * \file conv3d_bp_input_arch51.h
 * \brief
 */

#ifndef CONV3D_BP_INPUT_ARCH51_H
#define CONV3D_BP_INPUT_ARCH51_H

#include "conv3d_bp_config_arch51.h"
#include "conv3d_bp_impl_base_arch51.h"
#include "conv3d_bp_intf_arch51.h"

namespace Convolution3DBackprop {

REGISTER_DX_IMPL(Conv3DBackpropInput, Conv3DBpInputCfg, Conv3DBpInputImpl, Conv3DBpInputIntf);

} // namespace Convolution3DBackprop
#endif
