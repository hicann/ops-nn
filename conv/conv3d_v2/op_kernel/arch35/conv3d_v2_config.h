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
 * \file conv3d_v2_config.h
 * \brief
 */

#ifndef CONV3D_V2_CONFIG_H
#define CONV3D_V2_CONFIG_H

#include "../../common/arch35/conv_config.h"

namespace conv3d {
using namespace conv;

struct Conv3dParam : public ConvParam {
    __aicore__ inline Conv3dParam()
    {};
};

template <class ConvDataType>
struct Conv3dCfg : public ConvConfig<ConvDataType> {
public:
    __aicore__ inline Conv3dCfg()
    {}

    using ContextData = struct _ : public ConvConfig<ConvDataType>::ContextData {
        __aicore__ inline _()
        {}
    };
};

}  // namespace conv3d

#endif // CONV3D_V2_CONFIG_H