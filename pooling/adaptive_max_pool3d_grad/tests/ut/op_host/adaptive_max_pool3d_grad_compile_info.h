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
 * \file adaptive_max_pool3d_grad_compile_info.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_COMPILE_INFO_H__
#define __OP_HOST_MATMUL_V3_COMPILE_INFO_H__
#include <cstdint>
#include <string>
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

struct Tiling4AdaptiveMaxPool3DGradCompileInfo {
    platform_ascendc::SocVersion curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
    uint64_t totalCoreNum = 0;
    uint64_t maxUbSize = 0;
};
}
#endif // __OP_HOST_MATMUL_V3_COMPILE_INFO_H__