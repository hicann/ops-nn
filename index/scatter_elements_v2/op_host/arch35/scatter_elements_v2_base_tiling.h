/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_elements_v2_base_tiling.h
 * \brief
 */
#ifndef SCATTER_ELEMENTS_V2_BASE_TILING_H
#define SCATTER_ELEMENTS_V2_BASE_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
struct ScatterElementsV2CompileInfoArch35 {
  int32_t totalCoreNum = 0;
  uint64_t ubSizePlatForm = 0;
  uint32_t workspaceSize = 0;
};
}
#endif // SCATTER_ELEMENTS_V2_BASE_TILING_H
