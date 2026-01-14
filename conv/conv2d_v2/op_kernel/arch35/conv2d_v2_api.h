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
 * \file conv2d_v2_api.h
 * \brief
 */

#ifndef CONV2D_V2_API_H
#define CONV2D_V2_API_H

#include "conv2d_v2_api_impl.h"
#include "conv2d_v2_config.h"
#include "conv2d_v2_intf.h"

namespace conv2d {
using namespace conv;

template <class Config,
          template<typename, class> class Impl = Conv2dApiImpl,
          template<class, template<typename, class> class>class Intf = Conv2dIntf>
struct Conv2dIntfExt : public Intf<Config, Impl> {
    __aicore__ inline Conv2dIntfExt() {}
};

#define REGISTER_CONV_API(name, Config, Impl, Intf)                                                                   \
    template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE,               \
              class CONV_CFG=Conv2dParam>                                                                             \
    using name = Conv2dIntfExt<                                                                                       \
        Config<ConvDataType<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>>, Impl, Intf>

REGISTER_CONV_API(Conv2d, Conv2dCfg, Conv2dApiImpl, Conv2dIntf);

}  // namespace conv2d

#endif // CONV2D_V2_API_H