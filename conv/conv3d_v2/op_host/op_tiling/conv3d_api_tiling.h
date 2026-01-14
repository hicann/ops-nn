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
 * \file conv3d_tiling.h
 * \brief
 */

#ifndef ASCENDC_TILING_CONV3D_TILING_H
#define ASCENDC_TILING_CONV3D_TILING_H

#include "conv3d_api_tiling_base.h"

namespace Conv3dApiTiling {

class __attribute__((visibility("default"))) Conv3dTiling : public Conv3dTilingBase {
public:
    Conv3dTiling() = default;
    explicit Conv3dTiling(const PlatformInfo& platform) : Conv3dTilingBase(platform) {};
    ~Conv3dTiling() override = default;
    int64_t GetTiling(Ops::NN::Conv3dV2::TConv3DTiling &tiling) override;
protected:
    int64_t Compute() override;
};

} // namespace Conv3dApiTiling

#endif // ASCENDC_TILING_CONV3D_TILING_H
