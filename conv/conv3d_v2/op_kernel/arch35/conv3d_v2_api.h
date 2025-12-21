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
 * \file conv3d_v2_api.h
 * \brief
 */

#ifndef CONV3D_V2_API_H
#define CONV3D_V2_API_H

#include "conv3d_v2_api_impl.h"
#include "conv3d_v2_config.h"
#include "conv3d_v2_intf.h"

namespace conv3d {
using namespace conv;

template <class Config,
          template <typename, class> class Impl = Conv3dApiImpl,
          template <class, template <typename, class> class> class Intf = Conv3dIntf>
struct Conv3dIntfExt : public Intf<Config, Impl> {
    __aicore__ inline Conv3dIntfExt()
    {}
};

#define REGISTER_CONV3D_API(name, Config, Impl, Intf)                                                                \
    template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class CONV_CFG = Conv3dParam>  \
    using name =                                                                                                     \
        Conv3dIntfExt<Config<ConvDataType<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, CONV_CFG>>, Impl, Intf>

REGISTER_CONV3D_API(Conv3d, Conv3dCfg, Conv3dApiImpl, Conv3dIntf);

}  // namespace conv3d

#endif // CONV3D_V2_API_H