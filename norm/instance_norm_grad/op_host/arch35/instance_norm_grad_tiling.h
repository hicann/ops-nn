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
 * \file instance_norm_grad_tiling.h
 * \brief TilingData definitions + CompileInfo for InstanceNormGrad (arch35).
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_INSTANCE_NORM_GRAD_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_INSTANCE_NORM_GRAD_H
#include "register/tilingdata_base.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "op_host/tiling_base.h"
#include "op_common/op_host/util/platform_util.h"
#include "error_util.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
// Main RegBase tiling data (shared by full_load / recompute keys 101/102/301/302).
// Geometry: logical [N, M, C], M = D*H*W, reduce over M keeping C. Core task = (n, cTileIdx).
BEGIN_TILING_DATA_DEF(InstanceNormGradTilingData)
TILING_DATA_FIELD_DEF(int64_t, N);                   // batch (instances)
TILING_DATA_FIELD_DEF(int64_t, C);                   // channels (innermost)
TILING_DATA_FIELD_DEF(int64_t, M);                   // spatial D*H*W (reduce axis size per instance)
TILING_DATA_FIELD_DEF(int64_t, cTile);               // channels handled per task
TILING_DATA_FIELD_DEF(int64_t, cTileNum);            // ceil(C / cTile)
TILING_DATA_FIELD_DEF(int64_t, taskNum);             // N * cTileNum
TILING_DATA_FIELD_DEF(uint32_t, taskNumPerCore);     // stage1 tasks per (non-tail) core
TILING_DATA_FIELD_DEF(uint32_t, taskNumPerTailCore); // stage1 tasks per tail core
TILING_DATA_FIELD_DEF(uint32_t, tailCore);           // number of front cores carrying taskNumPerCore
TILING_DATA_FIELD_DEF(uint32_t, stage1CoreUsed);     // cores used in stage1
TILING_DATA_FIELD_DEF(uint32_t, mUbTile);            // M rows per UB tile (== M for full_load)
TILING_DATA_FIELD_DEF(uint32_t, mUbIterNum);         // ceil(M / mUbTile)
TILING_DATA_FIELD_DEF(uint32_t, mUbTailNum);         // rows in the last M tile
TILING_DATA_FIELD_DEF(int64_t, reduceNCnt);          // N (rows in dgamma/dbeta workspace)
TILING_DATA_FIELD_DEF(int64_t, workSpaceSize);       // elements per workspace copy = reduceNCnt * C
TILING_DATA_FIELD_DEF(uint32_t, stage2CoreUsed);     // cores used in stage2 cross-N reduce
TILING_DATA_FIELD_DEF(int64_t, cBlockFactor);        // channels per stage2 core
TILING_DATA_FIELD_DEF(int64_t, cTailBlockFactor);    // channels on the last stage2 core
TILING_DATA_FIELD_DEF(uint32_t, stage2SubCap);       // channels per stage2 UB round (host-computed from ubSize)
// 以下三项为 stage1 各缓冲的字节数,一律由 host 依芯片 UB/向量长度算定后下发;
// 内核只按值 InitBuffer,不再自行推导尺寸(避免 host 记账与内核实占两套公式各算各的)。
TILING_DATA_FIELD_DEF(uint32_t, paramBufBytes);     // 每个 fp32 参数缓冲的字节数
TILING_DATA_FIELD_DEF(uint32_t, tmpParamBufBytes);  // 输入 dtype 的临时参数缓冲字节数
TILING_DATA_FIELD_DEF(uint32_t, tileBytes);         // 每个流水缓冲(x/dy/pd_x,各双缓冲)的字节数
TILING_DATA_FIELD_DEF(uint32_t, stage2BufBytes);    // stage2 每个 fp32 缓冲的字节数
TILING_DATA_FIELD_DEF(uint32_t, stage2OutBufBytes); // stage2 输出 dtype 缓冲的字节数
END_TILING_DATA_DEF;

// Keys 101/102/301/302 all share the op's default tiling struct, so only the optype-level
// registration is needed. Registering per-key classes with the same struct name would make the
// tiling generator emit a duplicate definition. Key 500 uses a distinct empty struct below.
REGISTER_TILING_DATA_CLASS(InstanceNormGrad, InstanceNormGradTilingData)

// Empty-tensor tiling data (key 500): only pd_gamma/pd_beta are zeroed.
BEGIN_TILING_DATA_DEF(InstanceNormGradEmptyTilingData)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNumDG);
TILING_DATA_FIELD_DEF(uint64_t, colsPerCoreDG);
TILING_DATA_FIELD_DEF(uint64_t, colsPerUBDG);
TILING_DATA_FIELD_DEF(uint64_t, coreUbBlockCount);
TILING_DATA_FIELD_DEF(uint64_t, tailUbCols);
TILING_DATA_FIELD_DEF(uint64_t, lastCoreBlockCount);
TILING_DATA_FIELD_DEF(uint64_t, lastCoreTailUbCols);
TILING_DATA_FIELD_DEF(uint64_t, colsLastCoreDG);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(InstanceNormGrad_500, InstanceNormGradEmptyTilingData)

struct InstanceNormGradCompileInfo {
    int32_t totalCoreNum = 0;
    uint32_t sysWorkspaceSize = 0;
    uint64_t ubSizePlatForm = 0;
    uint32_t vectorLen = 0;
    uint32_t blockSize = 0;
    bool isRegBase{false};
};
} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_INSTANCE_NORM_GRAD_H
