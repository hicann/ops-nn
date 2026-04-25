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
 * \file masked_scatter_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __MASKED_SCATTER_TILLING_H__
#define __MASKED_SCATTER_TILLING_H__

#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "util/math_util.h"

namespace optiling {
struct MaskedScatterV1CompileInfo {
    uint32_t vectorCoreNum = 0;
    uint64_t ubSize = 0;
    uint32_t sysWorkspace = 0;
};
}  // namespace optiling
#endif
