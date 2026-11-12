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
 * \file transpose_quant_batch_mat_mul_w8a8.cpp
 * \brief A2/A3 INT8 W8A8 K-C entry. Compiled only for ascend910b / ascend910_93.
 *
 * GM order matches TransposeQuantBatchMatMul:
 *   x1, x2, bias(unused), x1_scale [B, M], x2_scale [B, N], y, workspace, tiling
 * x1_scale is transposed from the public [M, B] layout by aclnn before launch.
 */

#include "transpose_quant_batch_mat_mul_w8a8_tiling_key.h"
#include "tbmm_qbmm_pertoken_basic.h"
#include "transpose_batch_mat_mul_tiling_data.h"

using namespace AscendC;

namespace TransposeBatchMatmulW8A8Space {
template <class A_T, class B_T, class ScaleT, class OutT, int X1_FORMAT, int X2_FORMAT, bool aTrans, bool bTrans>
class TransposeBatchMatMulW8A8Kernel {
public:
    __aicore__ inline TransposeBatchMatMulW8A8Kernel() = default;

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR scalesGM, GM_ADDR cGM,
                                GM_ADDR workspaceGM, const TQBMMW8A8TilingData* tilingData, AscendC::TPipe* pipe)
    {
        op_.Init(aGM, bGM, scalesGM, nullptr, biasGM, cGM, workspaceGM, &tilingData->qbmm, pipe);
    }

    __aicore__ inline void Process() { op_.Process(); }

private:
    AscendC::BmmDequantPertokenBasic<A_T, B_T, ScaleT, OutT, X1_FORMAT, X2_FORMAT, aTrans, bTrans,
                                     AscendC::QuantBatchMatmulV3Update>
        op_;
};
} // namespace TransposeBatchMatmulW8A8Space

#ifndef FORMAT_FRACTAL_NZ
#define FORMAT_FRACTAL_NZ
#endif

#define TQBMM_W8A8_IMPL(B_TRANS)                                                                                     \
    do {                                                                                                             \
        TPipe pipe;                                                                                                  \
        TransposeBatchMatmulW8A8Space::TransposeBatchMatMulW8A8Kernel<DTYPE_X1, DTYPE_X2, float, DTYPE_Y, FORMAT_X1, \
                                                                      FORMAT_X2, false, B_TRANS>                     \
            op;                                                                                                      \
        op.Init(aGM, bGM, x1_scaleGM, x2_scaleGM, cGM, user, &tilingData, &pipe);                                    \
        op.Process();                                                                                                \
    } while (0)

template <int BATCH_SPLIT, int PP_MAT_MUL_EINSUM_MODE, int PERM_X1, int PERM_X2, int QUANT_MODE>
__global__ __aicore__ void transpose_quant_batch_mat_mul(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR x1_scaleGM,
                                                         GM_ADDR x2_scaleGM, GM_ADDR cGM, GM_ADDR workspaceGM,
                                                         GM_ADDR tilingGM)
{
    // IR bias is unused. Host tiling returns GRAPH_FAILED if bias is present.
    (void)biasGM;
    (void)BATCH_SPLIT;
    (void)PP_MAT_MUL_EINSUM_MODE;
    (void)PERM_X1;
    (void)PERM_X2;
    __gm__ uint8_t* user = GetUserWorkspace(workspaceGM);
    REGISTER_TILING_DEFAULT(TQBMMW8A8TilingData);
    if constexpr (QUANT_MODE == TRANSPOSE_BATCH_MAT_MUL_QUANT_MODE_W8A8) {
        GET_TILING_DATA_WITH_STRUCT(TQBMMW8A8TilingData, tilingData, tilingGM);
        if (tilingData.matmulTiling.matmulRunInfo.transB) {
            TQBMM_W8A8_IMPL(true);
        } else {
            TQBMM_W8A8_IMPL(false);
        }
    }
}
