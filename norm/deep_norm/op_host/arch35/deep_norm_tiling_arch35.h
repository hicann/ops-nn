/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file deep_norm_tiling_arch35.h
 * \brief DeepNorm arch35 tiling compile-info declaration.
 */

#ifndef __DEEP_NORM_TILING_ARCH35_H__
#define __DEEP_NORM_TILING_ARCH35_H__

#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"

namespace optiling {

struct DeepNormCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

} // namespace optiling

#endif // __DEEP_NORM_TILING_ARCH35_H__
