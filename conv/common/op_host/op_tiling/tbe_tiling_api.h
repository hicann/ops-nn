/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tbe_tiling_api.h
 * \brief
 */
#ifndef TBE_TILING_API_H
#define TBE_TILING_API_H

#include <cstdint>
#include <exe_graph/runtime/tiling_context.h>
#include <tiling/platform/platform_ascendc.h>
#include "graph/utils/type_utils.h"
#include "platform/platform_infos_def.h"

namespace optiling {
struct Conv3dBackpropV2TBETilingData {
    // L0 tiling parameters
    int32_t m_l0;          // Base M dimension at L0
    int32_t k_l0;          // Base K dimension at L0
    int32_t n_l0;          // Base N dimension at L0

    // L1 tiling parameters
    int32_t m_al1;         // Step M dimension at L1
    int32_t n_bl1;         // Step N dimension at L1
    int32_t k_al1;         // Step K dimension A at L1
    int32_t k_bl1;         // Step K dimension B at L1

    // Buffer parameters
    int32_t db_l0c;        // L0C buffer size
    int32_t db_al1;        // AL1 buffer size
    int32_t db_bl1;        // BL1 buffer size

    // Dimension parameters
    int32_t batch_dim;     // Batch dimension
    int32_t d_dim;         // Depth dimension
    int32_t group_dim;     // Group dimension
    int32_t m_dim;         // M dimension
    int32_t n_dim;         // N dimension
    int32_t k_dim;         // K dimension
};

enum OpTypeV2 : size_t {
  kConv3DBackpropFilterV2,
  kConv3DBackpropInputV2,
  kConv3DTransposeV2,
};

bool GetTbeTiling(gert::TilingContext* context, Conv3dBackpropV2TBETilingData& tbeTilingForV2, const optiling::OpTypeV2 opType);
}
#endif  // TBE_TILING_API_H