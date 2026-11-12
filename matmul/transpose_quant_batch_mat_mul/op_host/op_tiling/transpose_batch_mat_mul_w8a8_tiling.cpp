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
 * \file transpose_batch_mat_mul_w8a8_tiling.cpp
 * \brief W8A8 分支 host tiling 骨架实现（在原 TransposeBatchMatMul OpType 下）。
 *
 * 骨架阶段职责：
 *   1. 校验 shape / dtype / perm / batch_split_factor / K-N 对齐 / A2 B*K 上限
 *   2. 填充 TQBMMW8A8TilingData 的问题规模（matmulTiling 字段）+ w8a8* 扩展字段
 *   3. 预留 workspace（ping-pong 2-slot int32 tile）
 *   4. 不计算真正的 Cube matmul tiling（Step C 补齐）
 *
 * 决策依据: docs/TBMM_W8A8_A2_A3实现方案_20261010.md Sections 11-14。
 */
#include "transpose_batch_mat_mul_w8a8_tiling.h"
#include "../../op_kernel/transpose_quant_batch_mat_mul_w8a8_tiling_key.h"

#include <algorithm>
#include <cstring>

#include "error_util.h"
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_platform_common.h"
#include "tiling/matrix/matmul_tiling.h"
#include "tiling/matrix/bmm_tiling.h"

namespace optiling {
namespace transpose_batch_mat_mul {

namespace {

#define W8A8_CHECK(cond, log_func, return_expr) \
    do {                                        \
        if (cond) {                             \
            log_func;                           \
            return_expr;                        \
        }                                       \
    } while (0)

inline uint32_t CeilAlign(uint32_t x, uint32_t base) { return base == 0 ? x : ((x + base - 1) / base) * base; }

} // namespace

bool IsW8A8Mode(gert::TilingContext* context)
{
    if (context == nullptr) {
        return false;
    }
    auto x1_desc = context->GetInputDesc(0);
    if (x1_desc == nullptr) {
        return false;
    }
    return x1_desc->GetDataType() == ge::DT_INT8;
}

ge::graphStatus TransposeBatchMatMulW8A8DoTiling(gert::TilingContext* context)
{
    OP_TILING_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("TBMM_W8A8", "context is null"), return ge::GRAPH_FAILED);
    auto name_op = context->GetNodeName();

    // --- shape ---
    // 关键: 用 OriginShape（逻辑 shape），不是 StorageShape。
    // FRACTAL_NZ 下 x2 的 storage shape 是 [B, N/16, K/32, 16, 32]（INT8 fractal），
    // rank=5 不是 3。只有 origin shape 保持 [B, K, N]。
    auto shape_x1_desc = context->GetInputShape(0);
    auto shape_x2_desc = context->GetInputShape(1);
    // x1_scale / x2_scale are optional IR inputs. GetInputShape returns null for them.
    auto shape_ts_desc = context->GetOptionalInputShape(3); // x1_scale, already transposed to [B, M]
    auto shape_cs_desc = context->GetOptionalInputShape(4); // x2_scale [B, N]
    W8A8_CHECK(context->GetOptionalInputShape(2) != nullptr,
               CUBE_INNER_ERR_REPORT(name_op, "W8A8 does not support bias"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        shape_x1_desc == nullptr || shape_x2_desc == nullptr || shape_ts_desc == nullptr || shape_cs_desc == nullptr,
        CUBE_INNER_ERR_REPORT(name_op, "W8A8 shape desc null"), return ge::GRAPH_FAILED);
    const auto& sx1 = shape_x1_desc->GetOriginShape(); // [M, B, K]
    const auto& sx2 = shape_x2_desc->GetOriginShape(); // [B, K, N]
    const auto& sts = shape_ts_desc->GetOriginShape(); // [M, B]
    const auto& scs = shape_cs_desc->GetOriginShape(); // [B, N]
    W8A8_CHECK(sx1.GetDimNum() != 3 || sx2.GetDimNum() != 3,
               CUBE_INNER_ERR_REPORT(name_op, "W8A8 x1/x2 origin rank must be 3 (got %zu, %zu)", sx1.GetDimNum(),
                                     sx2.GetDimNum()),
               return ge::GRAPH_FAILED);
    W8A8_CHECK(sts.GetDimNum() != 2 || scs.GetDimNum() != 2,
               CUBE_INNER_ERR_REPORT(name_op, "W8A8 scale rank must be 2 (got ts=%zu, cs=%zu)", sts.GetDimNum(),
                                     scs.GetDimNum()),
               return ge::GRAPH_FAILED);
    // 单维先收窄到 int32，再校验 M*B*K / B*K*N，避免 kernel 32 位乘法溢出。
    constexpr int64_t INT32_SAFE_MAX = 2147483647LL;
    auto mulOverflow = [](int64_t a, int64_t b) -> bool {
        return a > 0 && b > 0 && a > (std::numeric_limits<int64_t>::max() / b);
    };
    for (size_t i = 0; i < 3; ++i) {
        W8A8_CHECK(sx1.GetDim(i) <= 0 || sx2.GetDim(i) <= 0,
                   CUBE_INNER_ERR_REPORT(name_op, "W8A8 unknown/zero dim in x1/x2 not supported"),
                   return ge::GRAPH_FAILED);
        W8A8_CHECK(sx1.GetDim(i) > INT32_SAFE_MAX || sx2.GetDim(i) > INT32_SAFE_MAX,
                   CUBE_INNER_ERR_REPORT(name_op, "W8A8 x1/x2 dim exceeds int32 limit"), return ge::GRAPH_FAILED);
    }
    for (size_t i = 0; i < 2; ++i) {
        W8A8_CHECK(sts.GetDim(i) <= 0 || scs.GetDim(i) <= 0,
                   CUBE_INNER_ERR_REPORT(name_op, "W8A8 unknown/zero dim in scales not supported"),
                   return ge::GRAPH_FAILED);
        W8A8_CHECK(sts.GetDim(i) > INT32_SAFE_MAX || scs.GetDim(i) > INT32_SAFE_MAX,
                   CUBE_INNER_ERR_REPORT(name_op, "W8A8 scale dim exceeds int32 limit"), return ge::GRAPH_FAILED);
    }
    // --- attr ---
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, CUBE_INNER_ERR_REPORT(name_op, "W8A8 attrs null"), return ge::GRAPH_FAILED);
    // TQBMM attr order: dtype, group_size, perm_x1, perm_x2, perm_y, batch_split_factor
    const auto* perm_x1 = attrs->GetAttrPointer<gert::ContinuousVector>(2);
    const auto* perm_x2 = attrs->GetAttrPointer<gert::ContinuousVector>(3);
    const auto* perm_y = attrs->GetAttrPointer<gert::ContinuousVector>(4);
    const auto* batch_split_factor = attrs->GetAttrPointer<int32_t>(5);
    W8A8_CHECK(perm_x1 == nullptr || perm_x2 == nullptr || perm_y == nullptr || batch_split_factor == nullptr,
               CUBE_INNER_ERR_REPORT(name_op, "W8A8 attr pointer null"), return ge::GRAPH_FAILED);
    // perm 长度必须为 3 才能解引用（图模式可能绕过 ACLNN）
    W8A8_CHECK(perm_x1->GetSize() != 3 || perm_x2->GetSize() != 3 || perm_y->GetSize() != 3,
               CUBE_INNER_ERR_REPORT(name_op, "W8A8 perm size must be 3 (got x1=%zu, x2=%zu, y=%zu)",
                                     perm_x1->GetSize(), perm_x2->GetSize(), perm_y->GetSize()),
               return ge::GRAPH_FAILED);
    const auto* px1 = reinterpret_cast<const int64_t*>(perm_x1->GetData());
    const auto* px2 = reinterpret_cast<const int64_t*>(perm_x2->GetData());
    const auto* py = reinterpret_cast<const int64_t*>(perm_y->GetData());
    // K-C only allows perm_x2 [0, 1, 2]: x2 is stored as [B, K, N].
    const bool bTrans = false;
    // int64 域校验 shape 一致性，再缩窄
    const int64_t M64 = sx1.GetDim(0), B64 = sx1.GetDim(1), K64 = sx1.GetDim(2);
    const int64_t B2_64 = sx2.GetDim(0), K2_64 = sx2.GetDim(bTrans ? 2 : 1), N64 = sx2.GetDim(bTrans ? 1 : 2);
    W8A8_CHECK(B64 != B2_64 || K64 != K2_64,
               CUBE_INNER_ERR_REPORT(name_op, "W8A8 batch/K mismatch: x1 [%ld,%ld,%ld] vs x2 [%ld,%ld,%ld]", M64, B64,
                                     K64, B2_64, K2_64, N64),
               return ge::GRAPH_FAILED);
    W8A8_CHECK(sts.GetDim(0) != B64 || sts.GetDim(1) != M64,
               CUBE_INNER_ERR_REPORT(name_op, "W8A8 token_scale must be [B=%ld, M=%ld] (post-transpose)", B64, M64),
               return ge::GRAPH_FAILED);
    W8A8_CHECK(scs.GetDim(0) != B64 || scs.GetDim(1) != N64,
               CUBE_INNER_ERR_REPORT(name_op, "W8A8 channel_scale must be [B=%ld, N=%ld]", B64, N64),
               return ge::GRAPH_FAILED);
    W8A8_CHECK(mulOverflow(M64, B64) || mulOverflow(M64 * B64, K64),
               CUBE_INNER_ERR_REPORT(name_op, "W8A8 M*B*K overflow"), return ge::GRAPH_FAILED);
    W8A8_CHECK(mulOverflow(B64, K64) || mulOverflow(B64 * K64, N64),
               CUBE_INNER_ERR_REPORT(name_op, "W8A8 B*K*N overflow"), return ge::GRAPH_FAILED);
    W8A8_CHECK(mulOverflow(M64, B64) || mulOverflow(M64 * B64, N64),
               CUBE_INNER_ERR_REPORT(name_op, "W8A8 M*B*N overflow"), return ge::GRAPH_FAILED);
    const uint32_t M = static_cast<uint32_t>(M64);
    const uint32_t B = static_cast<uint32_t>(B64);
    const uint32_t K = static_cast<uint32_t>(K64);
    const uint32_t N = static_cast<uint32_t>(N64);

    W8A8_CHECK(!(px1[0] == 1 && px1[1] == 0 && px1[2] == 2),
               CUBE_INNER_ERR_REPORT(name_op, "W8A8 requires perm_x1==[1,0,2]"), return ge::GRAPH_FAILED);
    W8A8_CHECK(!(px2[0] == 0 && px2[1] == 1 && px2[2] == 2),
               CUBE_INNER_ERR_REPORT(name_op, "W8A8 K-C requires perm_x2==[0,1,2]"), return ge::GRAPH_FAILED);
    W8A8_CHECK(!(py[0] == 1 && py[1] == 0 && py[2] == 2),
               CUBE_INNER_ERR_REPORT(name_op, "W8A8 requires perm_y==[1,0,2]"), return ge::GRAPH_FAILED);
    W8A8_CHECK(*batch_split_factor != 1,
               CUBE_INNER_ERR_REPORT(name_op, "W8A8 requires batch_split_factor==1 (got %d)", *batch_split_factor),
               return ge::GRAPH_FAILED);

    // --- K/N 对齐 ---
    W8A8_CHECK(K % W8A8_INT8_K_ALIGN != 0,
               CUBE_INNER_ERR_REPORT(name_op, "W8A8 requires K %% %u == 0 (K=%u)", W8A8_INT8_K_ALIGN, K),
               return ge::GRAPH_FAILED);
    W8A8_CHECK(N % W8A8_INT8_N_ALIGN != 0,
               CUBE_INNER_ERR_REPORT(name_op, "W8A8 requires N %% %u == 0 (N=%u)", W8A8_INT8_N_ALIGN, N),
               return ge::GRAPH_FAILED);

    // --- A2 Nd2NzParams.srcDValue uint16 上限 ---
    auto platformInfo = context->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, CUBE_INNER_ERR_REPORT(name_op, "W8A8 platform info null"),
                    return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    const auto socVer = ascendcPlatform.GetSocVersion();
    W8A8_CHECK(
        socVer != platform_ascendc::SocVersion::ASCEND910B && socVer != platform_ascendc::SocVersion::ASCEND910_93,
        CUBE_INNER_ERR_REPORT(name_op, "W8A8 INT8 only supports Ascend910B / Ascend910_93"), return ge::GRAPH_FAILED);
    if (socVer == platform_ascendc::SocVersion::ASCEND910B) {
        const uint64_t batchK = static_cast<uint64_t>(B) * K;
        W8A8_CHECK(batchK >= W8A8_A2_BATCH_K_LIMIT,
                   CUBE_INNER_ERR_REPORT(name_op, "W8A8 A2 requires B*K < %u (got %lu)", W8A8_A2_BATCH_K_LIMIT, batchK),
                   return ge::GRAPH_FAILED);
    }

    // --- 填 TQBMMW8A8TilingData 的 qbmm 子结构 (QuantBatchMatmulV3TilingData) ---
    // 策略: 直接按 QBMM pertoken_basic 的字段要求填, 让 kernel 侧的 BmmDequantPertokenBasic
    // 完全照搬 QBMM 行为 (TBMM delta 已内嵌到 CalcGMOffset/UpdateBatchOffset/epilogue 中).
    //
    // TBMM 的 "等效 QBMM" 问题规模:
    //   QBMM M = TBMM M  (QBMM A[1,M,K], TBMM A[M,B,K] 的 M 维)
    //   QBMM N = TBMM N
    //   QBMM Ka = TBMM K
    //   QBMM Kb = TBMM K
    //   QBMM batchA = batchB = batchC = TBMM B
    //   QBMM 的 M 维跨步在 kernel 的 CalcGMOffset 处 ×batch_, Y/pertoken 同理
    auto raw_tiling = context->GetRawTilingData();
    OP_TILING_CHECK(raw_tiling == nullptr, CUBE_INNER_ERR_REPORT(name_op, "W8A8 raw tiling null"),
                    return ge::GRAPH_FAILED);
    TQBMMW8A8TilingData* tiling = reinterpret_cast<TQBMMW8A8TilingData*>(raw_tiling->GetData());
    OP_TILING_CHECK(tiling == nullptr, CUBE_INNER_ERR_REPORT(name_op, "W8A8 tiling data cast null"),
                    return ge::GRAPH_FAILED);
    memset(tiling, 0, sizeof(TQBMMW8A8TilingData));
    tiling->batchSplitFactor = 1;
    tiling->matmulTiling.matmulRunInfo.transB = bTrans;

    const uint32_t aicNum = std::max<uint32_t>(1, ascendcPlatform.GetCoreNumAic());
    const bool usePp = true;

    matmul_tiling::MultiCoreMatmulTiling mm(ascendcPlatform);
    mm.SetDim(aicNum);
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8,
                /*isTrans=*/false);
    mm.SetBType(matmul_tiling::TPosition::GM,
                context->GetInputDesc(1)->GetStorageFormat() == ge::FORMAT_FRACTAL_NZ ? matmul_tiling::CubeFormat::NZ :
                                                                                        matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_INT8, bTrans);
    mm.SetCType(matmul_tiling::TPosition::VECIN, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);
    mm.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);
    mm.SetShape(static_cast<int32_t>(M), static_cast<int32_t>(N), static_cast<int32_t>(K));
    mm.SetOrgShape(static_cast<int32_t>(M), static_cast<int32_t>(N), static_cast<int32_t>(K));
    mm.SetBias(false);
    OP_TILING_CHECK(!usePp && mm.GetTiling(tiling->qbmm.matmulTiling) < 0,
                    CUBE_INNER_ERR_REPORT(name_op, "W8A8 MultiCoreMatmulTiling GetTiling failed"),
                    return ge::GRAPH_FAILED);

    auto& mt = tiling->qbmm.matmulTiling;
    mt.dbL0C = 1;
    const bool x2Nd = context->GetInputDesc(1)->GetStorageFormat() != ge::FORMAT_FRACTAL_NZ;
    uint32_t selectedCores = 0;
    if (usePp) {
        mt.M = M;
        mt.N = N;
        mt.Ka = K;
        mt.Kb = K;
        mt.baseK = PpW8A8::L0_K;
        mt.dbL0A = 2;
        mt.dbL0B = 2;
        PpW8A8::Capacity cap{aicNum, 0, 0, 0, 0, 0};
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, cap.l1);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, cap.l0a);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, cap.l0b);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, cap.l0c);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, cap.ub);
        // ND and NZ use different B layouts, so they do not share a tile.
        const auto selected = x2Nd ? PpW8A8::SelectNd(M, B, K, N, cap) : PpW8A8::Select(M, B, K, N, cap);
        W8A8_CHECK(selected.m == 0, CUBE_INNER_ERR_REPORT(name_op, "No feasible Pp W8A8 tile"),
                   return ge::GRAPH_FAILED);
        mt.baseM = selected.m;
        mt.baseN = selected.n;
        mt.dbL0C = selected.l0cBuffers;
        selectedCores = selected.cores;
        const bool ndPair = x2Nd && selected.l0cBuffers > 1U;
        mt.stepKa = (ndPair ? PpW8A8::L1_K_ND : PpW8A8::L1_K) / PpW8A8::L0_K;
        mt.stepKb = mt.stepKa;
        OP_LOGI(name_op, "Pp %s tile: M=%u N=%u UB_M=%u L0C_BUFFERS=%u score=%.0f", x2Nd ? "ND" : "NZ", mt.baseM,
                mt.baseN, selected.ubM, selected.l0cBuffers, selected.cost);
    } else if (mt.baseK > 0) {
        uint64_t l1Size = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
        const uint32_t tileM = static_cast<uint32_t>(mt.baseM);
        const uint32_t tileN = static_cast<uint32_t>(mt.baseN);
        uint32_t panel = static_cast<uint32_t>(mt.baseK);
        while (panel <= K / 2U && 2ULL * (static_cast<uint64_t>(tileM) + tileN) * (panel * 2U) + 4096ULL <= l1Size) {
            panel *= 2U;
        }
        if (panel > 512U) {
            panel = 512U;
        }
        const uint32_t step = panel / static_cast<uint32_t>(mt.baseK);
        if (step >= 1U && panel % static_cast<uint32_t>(mt.baseK) == 0U) {
            mt.stepKa = step;
            mt.stepKb = step;
            mt.depthA1 = step * 2U;
            mt.depthB1 = step * 2U;
            mt.dbL0A = 2;
            mt.dbL0B = 2;
        }
    }
    mt.singleCoreM = std::min<uint32_t>(M, mt.baseM);
    mt.singleCoreN = std::min<uint32_t>(N, mt.baseN);
    if (usePp || bTrans) {
        uint64_t l0cSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize);
        const uint64_t requiredL0c = static_cast<uint64_t>(mt.baseM) * mt.baseN * sizeof(int32_t) *
                                     std::max<uint32_t>(1, static_cast<uint32_t>(mt.dbL0C));
        OP_TILING_CHECK(requiredL0c > l0cSize,
                        CUBE_INNER_ERR_REPORT(name_op, "W8A8 TransB L0C buffers require %lu bytes, available %lu",
                                              requiredL0c, l0cSize),
                        return ge::GRAPH_FAILED);
    }
    mt.usedCoreNum = aicNum;
    const uint32_t singleCoreM = static_cast<uint32_t>(mt.singleCoreM);
    const uint32_t singleCoreN = static_cast<uint32_t>(mt.singleCoreN);
    const uint32_t mTotalCnt = (M + singleCoreM - 1) / singleCoreM;
    const uint32_t nTotalCnt = (N + singleCoreN - 1) / singleCoreN;
    uint32_t nGroups = nTotalCnt;
    if (usePp && x2Nd && mt.dbL0C > 1) {
        nGroups = (nTotalCnt + static_cast<uint32_t>(mt.dbL0C) - 1U) / static_cast<uint32_t>(mt.dbL0C);
    }
    uint32_t usedCoreNum = static_cast<uint32_t>(std::min<uint64_t>(
        aicNum, std::max<uint64_t>(1, static_cast<uint64_t>(mTotalCnt) * nGroups * (usePp ? B : 1))));
    if (usePp && x2Nd && selectedCores > 0U && selectedCores <= aicNum) {
        usedCoreNum = selectedCores;
    }
    if (usePp) {
        mt.usedCoreNum = usedCoreNum;
    }

    auto& pp = tiling->qbmm.params;
    pp.batchA = B;
    pp.batchB = B;
    pp.batchC = B;
    pp.isPerTensor = 0;
    pp.isPertoken = 1;
    pp.isDoubleScale = 0;
    pp.biasThreeDim = 0;
    pp.ubCalcM = std::min<uint32_t>(singleCoreM, 16);
    pp.ubCalcN = singleCoreN;
    pp.needUbBuffer = 32 * 1024;
    pp.realSingleCoreM = singleCoreM;
    pp.realSingleCoreN = singleCoreN;
    pp.biasDtype = static_cast<uint32_t>(ge::DT_INT32);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    pp.ubSize = static_cast<uint32_t>(ubSize);
    if (usePp) {
        pp.ubCalcM = PpW8A8::VectorRows(singleCoreM, singleCoreN, ubSize);
    }
    if (!usePp && bTrans && singleCoreN <= 96) {
        const uint32_t candidateM = std::min<uint32_t>(singleCoreM, 64);
        const uint64_t alignedM = (candidateM + 7U) / 8U * 8U;
        const uint64_t requiredUb = 20ULL * candidateM * singleCoreN + pp.needUbBuffer + 8ULL * singleCoreN +
                                    8ULL * alignedM + 2048ULL;
        if (requiredUb <= ubSize) {
            pp.ubCalcM = candidateM;
        }
    }

    // --- 填 tileL2cacheTiling (L2cacheTileParam) ---
    auto& l2 = tiling->qbmm.tileL2cacheTiling;
    l2.mTileCntL2 = 1; // 不切 L2 M
    l2.nTileCntL2 = 1; // 不切 L2 N
    l2.mTileBlock = mTotalCnt;
    l2.nTileBlock = nTotalCnt;
    l2.calOrder = 0;
    l2.isBasicTiling = 1;

    // --- workspace ---
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, CUBE_INNER_ERR_REPORT(name_op, "W8A8 workspace null"),
                    return ge::GRAPH_FAILED);
    uint64_t libApi = 0;
    try {
        libApi = ascendcPlatform.GetLibApiWorkSpaceSize();
    } catch (const std::exception& e) {
        OP_LOGW(name_op, "W8A8 GetLibApiWorkSpaceSize threw '%s'; fallback to 0", e.what());
    }
    // QBMM pertoken_basic 的 workspace: BUFFER_NUM * usedCoreNum * baseM * baseN * sizeof(int32_t)
    // (pertoken_basic.h:190 offsetWorkspaceC_ = BUFFER_NUM * blockIdx * baseM * baseN)
    uint64_t userBytes = static_cast<uint64_t>(2 /*BUFFER_NUM*/) * usedCoreNum * static_cast<uint32_t>(mt.baseM) *
                         static_cast<uint32_t>(mt.baseN) * sizeof(int32_t);
    const uint64_t total = libApi + userBytes;
    W8A8_CHECK(total >= W8A8_WORKSPACE_LIMIT,
               CUBE_INNER_ERR_REPORT(name_op, "W8A8 workspace %lu exceeds %u (user=%lu lib=%lu aic=%u)", total,
                                     W8A8_WORKSPACE_LIMIT, userBytes, libApi, usedCoreNum),
               return ge::GRAPH_FAILED);
    workspaces[0] = static_cast<size_t>(total);

    context->SetBlockDim(usedCoreNum);
    // tiling_key: W8A8 分支，QUANT_MODE=W8A8 对应 ASCENDC_TPL_SEL 的 MIX_AIC_1_1 变体。
    const uint64_t key = GET_TPL_TILING_KEY(static_cast<uint64_t>(TRANSPOSE_BATCH_MAT_MUL_BATCH_SPLIT_FALSE),
                                            static_cast<uint64_t>(TRANSPOSE_BATCH_MAT_MUL_PP_MAT_MUL_EINSUM_MODE_FALSE),
                                            static_cast<uint64_t>(TRANSPOSE_BATCH_MAT_MUL_PERM_X1_1_0_2),
                                            static_cast<uint64_t>(TRANSPOSE_BATCH_MAT_MUL_PERM_X2_0_1_2),
                                            static_cast<uint64_t>(TRANSPOSE_BATCH_MAT_MUL_QUANT_MODE_W8A8));
    context->SetTilingKey(key);

    raw_tiling->SetDataSize(sizeof(TQBMMW8A8TilingData));
    OP_LOGI(name_op, "W8A8 implementation=%s", usePp ? "Pp" : "QBMM");
    OP_LOGI(name_op, "W8A8 L0 buffers: dbL0A=%u dbL0B=%u dbL0C=%u", static_cast<uint32_t>(mt.dbL0A),
            static_cast<uint32_t>(mt.dbL0B), static_cast<uint32_t>(mt.dbL0C));
    OP_LOGI(name_op, "W8A8 tiling: M=%u B=%u K=%u N=%u aic=%u base=%ux%ux%u step=%ux%u ws=%lu", M, B, K, N, usedCoreNum,
            static_cast<uint32_t>(mt.baseM), static_cast<uint32_t>(mt.baseN), static_cast<uint32_t>(mt.baseK),
            static_cast<uint32_t>(mt.stepKa), static_cast<uint32_t>(mt.stepKb), total);
    return ge::GRAPH_SUCCESS;
}

} // namespace transpose_batch_mat_mul
} // namespace optiling
