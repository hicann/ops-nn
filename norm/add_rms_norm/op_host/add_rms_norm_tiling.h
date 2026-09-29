/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_H_
#include "register/tilingdata_base.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AddRMSNormTilingData)
TILING_DATA_FIELD_DEF(uint32_t, num_row);
TILING_DATA_FIELD_DEF(uint32_t, num_col);
TILING_DATA_FIELD_DEF(uint32_t, block_factor);
TILING_DATA_FIELD_DEF(uint32_t, row_factor);
TILING_DATA_FIELD_DEF(uint32_t, ub_factor);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, avg_factor);
TILING_DATA_FIELD_DEF(uint32_t, num_col_align);
TILING_DATA_FIELD_DEF(uint32_t, last_block_factor);
TILING_DATA_FIELD_DEF(uint32_t, row_loop);
TILING_DATA_FIELD_DEF(uint32_t, last_block_row_loop);
TILING_DATA_FIELD_DEF(uint32_t, row_tail);
TILING_DATA_FIELD_DEF(uint32_t, last_block_row_tail);
TILING_DATA_FIELD_DEF(uint32_t, mul_loop_fp32);
TILING_DATA_FIELD_DEF(uint32_t, mul_tail_fp32);
TILING_DATA_FIELD_DEF(uint32_t, dst_rep_stride_fp32);
TILING_DATA_FIELD_DEF(uint32_t, mul_loop_fp16);
TILING_DATA_FIELD_DEF(uint32_t, mul_tail_fp16);
TILING_DATA_FIELD_DEF(uint32_t, dst_rep_stride_fp16);
TILING_DATA_FIELD_DEF(uint32_t, is_performance);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(AddRMSNormRegbaseTilingData)
TILING_DATA_FIELD_DEF(uint32_t, numRow);
TILING_DATA_FIELD_DEF(uint32_t, numCol);
TILING_DATA_FIELD_DEF(uint32_t, numColAlign);
TILING_DATA_FIELD_DEF(uint32_t, blockFactor);
TILING_DATA_FIELD_DEF(uint32_t, rowFactor);
TILING_DATA_FIELD_DEF(uint32_t, ubFactor);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, avgFactor);
TILING_DATA_FIELD_DEF(uint32_t, ubLoop);
TILING_DATA_FIELD_DEF(uint32_t, colBuferLength);
TILING_DATA_FIELD_DEF(uint32_t, multiNNum);
TILING_DATA_FIELD_DEF(uint32_t, isNddma);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(AddRMSNormRegbaseRFullLoadTilingData)
TILING_DATA_FIELD_DEF(uint64_t, numRow);
TILING_DATA_FIELD_DEF(uint64_t, numCol);
TILING_DATA_FIELD_DEF(uint64_t, numColAlign);
TILING_DATA_FIELD_DEF(uint64_t, blockFactor);
TILING_DATA_FIELD_DEF(uint64_t, rowFactor);
TILING_DATA_FIELD_DEF(uint64_t, binAddQuotient);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, avgFactor);
END_TILING_DATA_DEF;

// key=3000：按 R 分核，每核遍历全部 A 行。
// 前 8 个字段是已有 key=3000 ABI；后续字段只能追加，不能插入到旧字段中间。
BEGIN_TILING_DATA_DEF(AddRMSNormRegbaseSplitARTilingData)
TILING_DATA_FIELD_DEF(uint64_t, numRow);
TILING_DATA_FIELD_DEF(uint64_t, numCol);
TILING_DATA_FIELD_DEF(uint64_t, rBlockFactor);
TILING_DATA_FIELD_DEF(uint64_t, rBlockNum);
TILING_DATA_FIELD_DEF(uint64_t, tailR);
TILING_DATA_FIELD_DEF(uint64_t, binAddQuotient);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, avgFactor);
// 保留上述 key=3000 旧字段的顺序，新字段只能追加，避免改变旧 ABI 偏移。
TILING_DATA_FIELD_DEF(uint64_t, ubFactor);
TILING_DATA_FIELD_DEF(uint64_t, blockTailBinAddQuotient);
TILING_DATA_FIELD_DEF(uint64_t, tailBinAddQuotient);
// 跨 UB tile 使用二进制 cache 归并。
TILING_DATA_FIELD_DEF(uint64_t, basicBlockLoop);
TILING_DATA_FIELD_DEF(uint64_t, mainFoldCount);
TILING_DATA_FIELD_DEF(uint64_t, tailBasicBlockLoop);
TILING_DATA_FIELD_DEF(uint64_t, tailMainFoldCount);
TILING_DATA_FIELD_DEF(uint64_t, numRowAlign);
END_TILING_DATA_DEF;

// key=4000：物理转置后沿 A 向量化。
BEGIN_TILING_DATA_DEF(AddRMSNormRegbaseTransTilingData)
TILING_DATA_FIELD_DEF(uint64_t, numRow);
TILING_DATA_FIELD_DEF(uint64_t, numCol);
TILING_DATA_FIELD_DEF(uint64_t, rAligned);
TILING_DATA_FIELD_DEF(uint64_t, rTileBase);
TILING_DATA_FIELD_DEF(uint64_t, totalTiles);
TILING_DATA_FIELD_DEF(uint64_t, tilesPerCore);
TILING_DATA_FIELD_DEF(uint64_t, tileALen);
TILING_DATA_FIELD_DEF(uint64_t, tileATail);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, avgFactor);
END_TILING_DATA_DEF;

// key=5000：R 为空时仅切分并写出 rstd。
BEGIN_TILING_DATA_DEF(AddRMSNormRegbaseReduceEmptyTilingData)
TILING_DATA_FIELD_DEF(uint64_t, numRow);
TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, rowsPerCore);
TILING_DATA_FIELD_DEF(uint64_t, rowsPerTailCore);
TILING_DATA_FIELD_DEF(uint64_t, tailCoreStartIndex);
TILING_DATA_FIELD_DEF(uint64_t, rowsPerLoop);
END_TILING_DATA_DEF;

struct AddRmsNormCompileInfo {
    uint32_t totalCoreNum = 0;
    uint64_t totalUbSize = 0;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
};

namespace addRmsNormRegbase {
ge::graphStatus TilingAddRmsNormRegbase(gert::TilingContext* context);
ge::graphStatus TilingTranspose(gert::TilingContext* context, uint64_t numRow, uint64_t numCol, uint32_t numCore,
                                uint64_t ubSize, uint32_t curElementByte, uint64_t ubBlockSize, uint64_t vlfp32,
                                float epsilon, float avgFactor);
ge::graphStatus TilingSplitD(gert::TilingContext* context, uint64_t numRow, uint64_t numCol, uint32_t numCore,
                             uint64_t ubSize, ge::DataType dataType, uint32_t curElementByte, uint64_t vlfp32,
                             float epsilon, float avgFactor);
ge::graphStatus TilingSplitAR(gert::TilingContext* context, uint64_t numRow, uint64_t numCol, uint32_t numCore,
                              uint64_t ubSize, uint32_t curElementByte, uint32_t dataPerBlock, uint64_t ubBlockSize,
                              uint64_t vlfp32, uint64_t binaryAddElementMaxLen, float epsilon, float avgFactor);
ge::graphStatus TilingReduceEmpty(gert::TilingContext* context, uint64_t numRow, uint32_t numCore, uint64_t ubSize,
                                  uint64_t ubBlockSize, float epsilon);
} // namespace addRmsNormRegbase

REGISTER_TILING_DATA_CLASS(AddRmsNorm, AddRMSNormTilingData)
REGISTER_TILING_DATA_CLASS(InplaceAddRmsNorm, AddRMSNormTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNorm_1000, AddRMSNormRegbaseRFullLoadTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNorm_2000, AddRMSNormRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNorm_3000, AddRMSNormRegbaseSplitARTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNorm_4000, AddRMSNormRegbaseTransTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNorm_5000, AddRMSNormRegbaseReduceEmptyTilingData)
REGISTER_TILING_DATA_CLASS(InplaceAddRmsNorm_1000, AddRMSNormRegbaseRFullLoadTilingData)
REGISTER_TILING_DATA_CLASS(InplaceAddRmsNorm_2000, AddRMSNormRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(InplaceAddRmsNorm_3000, AddRMSNormRegbaseSplitARTilingData)
REGISTER_TILING_DATA_CLASS(InplaceAddRmsNorm_4000, AddRMSNormRegbaseTransTilingData)
REGISTER_TILING_DATA_CLASS(InplaceAddRmsNorm_5000, AddRMSNormRegbaseReduceEmptyTilingData)
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_H_
