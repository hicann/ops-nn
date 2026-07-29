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
 * \file median_tiling.cpp
 * \brief Median 算子 tiling：计算 batch/redLen/mid/nSeg + 路径选择(pathCode)/Pad + 核切分 + workspace
 */

#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "experimental/index/median/op_kernel/median_tiling_data.h"
#include "experimental/index/median/op_kernel/median_tiling_key.h"

namespace optiling {

static const gert::Shape g_vec_1_shape = {1};

struct MedianCompileInfo {};

inline const gert::Shape& EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 支持的 dtype 判定（kernel 侧由编译期 DTYPE_INPUT 决定实际类型，故 host 不再下发 dtype 码，仅用于
// 路径选择与核切分）
static bool IsFpDtype(ge::DataType dataType)
{
    return dataType == ge::DT_FLOAT16 || dataType == ge::DT_FLOAT || dataType == ge::DT_BF16;
}
static bool IsValidDtype(ge::DataType dataType)
{
    return IsFpDtype(dataType) || dataType == ge::DT_INT32 || dataType == ge::DT_INT64 || dataType == ge::DT_INT16 ||
           dataType == ge::DT_UINT8 || dataType == ge::DT_INT8;
}

// MAIN 路径 Pad 档位（原 kernel 内计算，按需求下沉到 host）：
//   Pad ∈ {32,64,128,256,512,1024,2048,4096,8192}，小 L 用更小 Pad 让 K 翻倍提高 batch 密度。
static int64_t ComputeMainPad(int64_t redLen)
{
    return (redLen <= 32)   ? 32 :
           (redLen <= 64)   ? 64 :
           (redLen <= 128)  ? 128 :
           (redLen <= 256)  ? 256 :
           (redLen <= 512)  ? 512 :
           (redLen <= 1024) ? 1024 :
           (redLen <= 2048) ? 2048 :
           (redLen <= 4096) ? 4096 :
                              8192;
}

// 核切分：fp 大 L 分 BigSel(≤24576) / BigSort(单行) / MedianBig(多行) 三路（并设 nSeg）；
// 其余（redLen≤8192 / int64 / int 大 L）行间独立占满核。返回核数，并写 tiling->nSeg。
static int64_t ComputeCoreSplit(bool fp, bool isInt64, int64_t redLen, int64_t batch, int64_t maxCore,
                                MedianTilingData* tiling)
{
    int64_t cores;
    if (fp && redLen > 8192 && batch > 0) {
        if (redLen <= 24576) { // BigSel：整行常驻 UB 向量化二分计数（快）
            tiling->nSeg = 0;
            cores = batch < maxCore ? batch : maxCore;
        } else if (batch == 1) { // 单行大 L：BigSort 多核值二分计数
            // L 小（blocks≤40，barrier 主导）：用 blocks 个核，每核 1 个对齐 8192 块 → 减核数、barrier 更快；
            // L 大（blocks>40，计算主导）：占满 40 核（非对齐 ceil(L/40)），并行度优先。
            int64_t blocks = (redLen + 8191) / 8192;
            int64_t ns = blocks < maxCore ? blocks : maxCore;
            if (ns < 1)
                ns = 1;
            tiling->nSeg = ns;
            cores = ns;
        } else { // 多行大 L：MedianBig 每核几行（分块二分计数）
            tiling->nSeg = 0;
            cores = batch < maxCore ? batch : maxCore;
        }
    } else {
        // 多行 redLen≤8192（MedianMain）/ int64 Heap / int 大 L：行间独立，占满核并行
        //   （旧版固定 8 核 → batch=1023 时每核串行 ~128 行；放开到 maxCore 后 ~26 行/核 ≈ 5× 提速）
        cores = batch < maxCore ? (batch == 0 ? 1 : batch) : maxCore;
        if (redLen > 8192 && batch < 8 && !isInt64)
            cores = 8; // int 大 L 少行：8 核切分
    }
    if (cores < 1)
        cores = 1;
    return cores;
}

static ge::graphStatus MedianTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto xs = EnsureNotScalar(inputX->GetStorageShape());
    int64_t numel = xs.GetShapeSize();

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    OP_CHECK_IF(!IsValidDtype(dtype), OP_LOGE(context, "invalid dtype"), return ge::GRAPH_FAILED);
    bool fp = IsFpDtype(dtype);
    bool isInt64 = (dtype == ge::DT_INT64);

    int dn = xs.GetDimNum();
    int64_t redLen = dn > 0 ? xs.GetDim(dn - 1) : 1;
    if (redLen <= 0)
        redLen = 1;
    int64_t batch = numel / redLen;

    MedianTilingData* tiling = context->GetTilingData<MedianTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(MedianTilingData), 0, sizeof(MedianTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->batch = batch;
    tiling->redLen = redLen;
    tiling->mid = (redLen - 1) / 2;
    tiling->nSeg = 0;

    int64_t cores = ComputeCoreSplit(fp, isInt64, redLen, batch, coreNum, tiling);
    context->SetBlockDim(cores);

    // 路径选择下沉到 host（与原 kernel 分发逻辑逐一对应），并作为 tilingkey(schMode) 下发：
    //   nSeg>0 → BIGSORT；int64 → HEAP；redLen<=8192 → SMALL(fp 且 L<=32) / MAIN；
    //   fp 且 redLen<=24576 → BIGSEL；否则 BIG。kernel 侧按此 tilingkey 编译期 if constexpr 分发，无运行期分支。
    int64_t pathCode;
    if (tiling->nSeg > 0) {
        pathCode = MEDIAN_PATH_BIGSORT;
    } else if (isInt64) {
        pathCode = MEDIAN_PATH_HEAP;
    } else if (redLen <= 8192) {
        pathCode = (redLen <= 32 && fp) ? MEDIAN_PATH_SMALL : MEDIAN_PATH_MAIN;
    } else if (fp && redLen <= 24576) {
        pathCode = MEDIAN_PATH_BIGSEL;
    } else {
        pathCode = MEDIAN_PATH_BIG;
    }
    tiling->pad = (pathCode == MEDIAN_PATH_MAIN) ? ComputeMainPad(redLen) : 0;

    size_t* ws = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    ws[0] = 32ull * 1024 * 1024 + (size_t)numel * 32; // 32MB 系统保留 + BigSort 有序 pair 余量

    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(pathCode));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForMedian([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Median).Tiling(MedianTilingFunc).TilingParse<MedianCompileInfo>(TilingParseForMedian);
} // namespace optiling
