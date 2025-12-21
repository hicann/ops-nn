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
 * \file conv2d_v2_config.h
 * \brief
 */

#ifndef CONV2D_V2_CONFIG_H
#define CONV2D_V2_CONFIG_H

#include "../../common/arch35/conv_config.h"

namespace conv2d {
using namespace conv;

struct Conv2dParam : public ConvParam {
    __aicore__ inline Conv2dParam() {};
    constexpr static int8_t enableSmallChannel = 0;
    constexpr static int8_t weightUbTrans = 0;
    constexpr static int8_t fmapCopyMode = 0;
    constexpr static int8_t isExtendConv2d = 0;
};

template <class ConvDataType>
struct Conv2dCfg : public ConvConfig<ConvDataType> {
public:
    __aicore__ inline Conv2dCfg() {}

    using ContextData = struct _ : public ConvConfig<ConvDataType>::ContextData {
        __aicore__ inline _() {}
    };
};

}  // namespace conv2d

#endif // CONV2D_V2_CONFIG_H