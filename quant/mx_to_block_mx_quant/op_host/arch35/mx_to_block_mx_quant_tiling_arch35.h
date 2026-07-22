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
 * \file mx_to_block_mx_quant_tiling_arch35.h
 * \brief Compile info and tiling param for MxToBlockMxQuant host tiling.
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MX_TO_BLOCK_MX_QUANT_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MX_TO_BLOCK_MX_QUANT_H

#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_util.h"
#include "graph/types.h"
#include "register/op_impl_registry.h"
#include "quant/mx_to_block_mx_quant/op_kernel/arch35/mx_to_block_mx_quant_tilingdata.h"

namespace optiling {
struct MxToBlockMxQuantCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

struct MxToBlockMxQuantTilingParam {
    int64_t totalCoreNum{0};
    int64_t ubSize{0};
    uint32_t vfLen{0};
    int64_t dstType{0};
    int64_t usedCoreNum{0};
    int64_t batchNum{1};
    int64_t rowNum{1}; // 单 batch 行数 M
    int64_t colNum{1}; // 列数 K
    int64_t colScaleNum{1};
    int64_t tilingKey{0};
    int64_t rowMode{0};             // 行切分模式：0=ALIGNED 1=NOT_ALIGNED
    int64_t rowBlockNumPerBatch{0}; // 单 batch 行方向基本块数 CeilDiv(M, 64)
    int64_t colBlockNumPerBatch{0}; // 单 batch 列方向基本块数 CeilDiv(K, 512)
    int64_t rowTailLenPerBatch{0};  // 单 batch 行尾块行数
    int64_t colTailLenPerBatch{0};  // 单 batch 列尾块列数
    int64_t totalBlockNum{0};       // 总块数 B*rowBlockNumPerBatch*colBlockNumPerBatch
    int64_t headCoreBlockNum{0};    // 头核处理块数
    int64_t tailCoreBlockNum{0};    // 尾核处理块数
    int64_t headCoreNum{0};         // 头核数量
    int64_t tailCoreNum{0};         // 尾核数量
};

} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_MX_TO_BLOCK_MX_QUANT_H
