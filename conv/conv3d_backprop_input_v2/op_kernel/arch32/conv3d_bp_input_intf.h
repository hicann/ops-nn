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
 * \file conv3d_bp_input_intf.h
 * \brief
 */

#ifndef CONV3D_BP_INPUT_INTF_H
#define CONV3D_BP_INPUT_INTF_H

#include "./conv3d_backprop_input_impl/conv3d_bp_func.h"
#include "./conv3d_backprop_input_impl/conv3d_bp_util.h"

namespace Convolution3DBackprop {
template <class Config_, template <typename, class> class Impl>
struct Conv3DBpInputIntf : public ConvBpIntf<Config_, Impl> {
public:
    __aicore__ inline Conv3DBpInputIntf()
    {}
};

} // namespace Convolution3DBackprop

#endif
