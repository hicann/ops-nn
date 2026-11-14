/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 300
#include "arch22/union_sparse_gen_tiling_key.h"
#include "arch22/union_sparse_gen_simd.h"
#else
#include "arch35/union_sparse_gen_simt.h"
#endif

extern "C" __global__ __aicore__ void union_sparse_gen(GM_ADDR sparseBlockIdx, GM_ADDR sparseBlockCount,
                                                       GM_ADDR cuSqlenQ, GM_ADDR cuSqlenKv, GM_ADDR seqUsedQ,
                                                       GM_ADDR seqUsedKv, GM_ADDR unionIdx, GM_ADDR unionCount,
                                                       GM_ADDR exactMask, GM_ADDR workspace, GM_ADDR tiling)
{
    // The generated AscendC entry initializes the library system workspace.
    // No backend here needs a user-workspace allocation.
#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 300
    REGISTER_TILING_DEFAULT(UnionSparseGenTilingData);
    GET_TILING_DATA_WITH_STRUCT(UnionSparseGenTilingData, tilingData, tiling);
#else
    REGISTER_TILING_DEFAULT(UnionSparseGenA5TilingData);
    GET_TILING_DATA_WITH_STRUCT(UnionSparseGenA5TilingData, tilingData, tiling);
#endif
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 300
    if (TILING_KEY_IS(UNION_SPARSE_GEN_A2_KEY_ORDERED_GENERAL)) {
        UnionSparseGenArch22::UnionSparseGen<> op;
        op.Init(sparseBlockIdx, sparseBlockCount, cuSqlenQ, cuSqlenKv, seqUsedQ, seqUsedKv, unionIdx, unionCount,
                exactMask, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(UNION_SPARSE_GEN_A2_KEY_ORDERED_BATCH_512)) {
        UnionSparseGenArch22::UnionSparseGen<UnionSparseGenA2Plan::kMinGroupWidth,
                                             UnionSparseGenArch22::kAdaptiveVectorWidth>
            op;
        op.Init(sparseBlockIdx, sparseBlockCount, cuSqlenQ, cuSqlenKv, seqUsedQ, seqUsedKv, unionIdx, unionCount,
                exactMask, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(UNION_SPARSE_GEN_A2_KEY_ORDERED_BATCH_256)) {
        UnionSparseGenArch22::UnionSparseGen<UnionSparseGenA2Plan::kMinGroupWidth,
                                             (UnionSparseGenArch22::kAdaptiveVectorWidth >
                                                      UnionSparseGenA2Plan::kDefaultVectorLanes ?
                                                  UnionSparseGenA2Plan::kDefaultVectorLanes :
                                                  UnionSparseGenArch22::kAdaptiveVectorWidth)>
            op;
        op.Init(sparseBlockIdx, sparseBlockCount, cuSqlenQ, cuSqlenKv, seqUsedQ, seqUsedKv, unionIdx, unionCount,
                exactMask, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(UNION_SPARSE_GEN_A2_KEY_MEMBERSHIP_TABLE)) {
        UnionSparseGenArch22::UnionSparseGen<0, 0, UnionSparseGenA2Plan::Execution::MembershipTable> op;
        op.Init(sparseBlockIdx, sparseBlockCount, cuSqlenQ, cuSqlenKv, seqUsedQ, seqUsedKv, unionIdx, unionCount,
                exactMask, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(UNION_SPARSE_GEN_A2_KEY_MEMBERSHIP_ADAPTIVE)) {
        UnionSparseGenArch22::UnionSparseGen<0, 0, UnionSparseGenA2Plan::Execution::MembershipAdaptive> op;
        op.Init(sparseBlockIdx, sparseBlockCount, cuSqlenQ, cuSqlenKv, seqUsedQ, seqUsedKv, unionIdx, unionCount,
                exactMask, workspace, tilingData);
        op.Process();
    }
#else
    if (TILING_KEY_IS(UNION_SPARSE_GEN_A5_KEY_ORDERED_SUBGROUP_8)) {
        UnionSparseGenNs::UnionSparseGen<UnionSparseGenA5Algorithm::WarpMerge, UnionSparseGenA5Plan::kMinGroupWidth> op;
        op.Init(sparseBlockIdx, sparseBlockCount, cuSqlenQ, cuSqlenKv, seqUsedQ, seqUsedKv, unionIdx, unionCount,
                exactMask, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(UNION_SPARSE_GEN_A5_KEY_ORDERED_SUBGROUP_16)) {
        UnionSparseGenNs::UnionSparseGen<UnionSparseGenA5Algorithm::WarpMerge, UnionSparseGenA5Plan::kHalfWarpLanes> op;
        op.Init(sparseBlockIdx, sparseBlockCount, cuSqlenQ, cuSqlenKv, seqUsedQ, seqUsedKv, unionIdx, unionCount,
                exactMask, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(UNION_SPARSE_GEN_A5_KEY_ORDERED_SUBGROUP_32)) {
        UnionSparseGenNs::UnionSparseGen<UnionSparseGenA5Algorithm::WarpMerge, UnionSparseGenA5Plan::kWarpLanes> op;
        op.Init(sparseBlockIdx, sparseBlockCount, cuSqlenQ, cuSqlenKv, seqUsedQ, seqUsedKv, unionIdx, unionCount,
                exactMask, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(UNION_SPARSE_GEN_A5_KEY_PACKED_TABLE)) {
        UnionSparseGenNs::UnionSparseGen<UnionSparseGenA5Algorithm::PackedMembership> op;
        op.Init(sparseBlockIdx, sparseBlockCount, cuSqlenQ, cuSqlenKv, seqUsedQ, seqUsedKv, unionIdx, unionCount,
                exactMask, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(UNION_SPARSE_GEN_A5_KEY_RADIX_SORT)) {
        UnionSparseGenNs::UnionSparseGen<UnionSparseGenA5Algorithm::RadixMembership> op;
        op.Init(sparseBlockIdx, sparseBlockCount, cuSqlenQ, cuSqlenKv, seqUsedQ, seqUsedKv, unionIdx, unionCount,
                exactMask, workspace, tilingData);
        op.Process();
    }
#endif
}
