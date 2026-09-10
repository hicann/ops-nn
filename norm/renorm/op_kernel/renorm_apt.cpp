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
 * \file renorm_apt.cpp
 * \brief A5 (Ascend950) APT kernel entry for Renorm.
 */

#include "arch35/renorm.h"
#include "arch35/renorm_sm_tl.h"
#include "arch35/renorm_sm_cr.h"
#include "arch35/renorm_sm_st.h"
#include "arch35/renorm_sm_st_pipelined.h"
#include "arch35/renorm_bm_vd.h"
#include "arch35/renorm_bm_vg.h"
#include "arch35/renorm_global.h"
#include "arch35/renorm_sm_tl_stable.h"
#include "arch35/renorm_sm_cr_packed.h"
#include "arch35/renorm_global_high_p.h"
#include "arch35/renorm_global_long_p.h"
#include "arch35/renorm_inner_split.h"
#include "arch35/renorm_bm_cr_tiled.h"

// 模板编号
// 0: Template A (SM-CT) - Slice-Major Continuous Single-Level
// 1: Template B (SM-TL) - Slice-Major Continuous Two-Level
// 2: Template C (SM-CR) - Slice-Major Cross-Core Reduction
// 3: Template D (SM-ST) - Slice-Major Stride
// 4: Template E (BM-VD) - Block-Major Vector Direct
// 5: Template F (BM-VG) - Block-Major Vector Grouped
// 7: Template B2 (SM-TL-STABLE) - isolated stable Template B variant
// 8: Template C2 (SM-CR-PACKED) - isolated packed Template C variant
// 9: Template G2 (GLOBAL-HP) - isolated global high-p variant
// 10: Template G3 (GLOBAL-LONG-P) - isolated long single-slice p route
// 11: Template C3 (SM-CR-CONTIGUOUS) - dense logical-block cross-core route
// 12: Template D2 (SM-ST-PIPELINED) - isolated long-slice ping-pong route
// 13: Template H (INNER-SPLIT) - split huge blockSize across cores for numBlocks=1
// 14: Template C4 (SM-CR-UNALIGNED) - aligned packed groups for odd fp16 blocks
// 15: Template C5 (BM-CR-TILED) - block-owned slice tiles for oversized blocks
// 16: Template C6 (SM-CR-PINF) - isolated p=inf packed max reduction
// 58: Template A2 (SM-CT-STABLE) - generic numerically stable p reduction

template <typename D_T_X, uint32_t TEMPLATE>
__global__ __aicore__ void renorm(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(RenormTilingData);
    GET_TILING_DATA_WITH_STRUCT(RenormTilingData, tilingData, tiling);

    if constexpr (TEMPLATE == 0) {
        // Template A: SM-CT (Slice-Major Continuous Single-Level)
        NsRenorm::Renorm<D_T_X> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (TEMPLATE == 1) {
        // Template B: SM-TL (Slice-Major Continuous Two-Level)
        // ARA HighPrecision BigDim Workspace: Pass1 累加→workspace, Pass2 聚合→scale, Pass3 应用
        // KERNEL_TASK_TYPE_DEFAULT required for SyncAll() (nll_loss_grad pattern)
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        // TPipe 外置: 在类外创建 TPipe，触发 Scalar 常量折叠/传播编译优化
        AscendC::TPipe pipe;
        NsRenormSmTl::RenormSmTl<D_T_X> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 2) {
        // Template C: SM-CR (Slice-Major Cross-Core Reduction)
        // 多核沿 R 轴分核, workspace 聚合 partial norm, SyncAll 跨核同步
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        NsRenormSmCr::RenormSmCr<D_T_X> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 3) {
        // Template D: SM-ST (Slice-Major Stride)
        NsRenormSmSt::RenormSmSt<D_T_X> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (TEMPLATE == 4) {
        // Template E: BM-VD (Block-Major Vector Direct)
        NsRenormBmVd::RenormBmVd<D_T_X> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (TEMPLATE == 5) {
        // Template F: BM-VG (Block-Major Vector Grouped)
        NsRenormBmVg::RenormBmVg<D_T_X> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (TEMPLATE == 6) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        NsRenormGlobal::RenormGlobal<D_T_X> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 7) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        NsRenormSmTlStable::RenormSmTlStable<D_T_X> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 8) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        NsRenormSmCrPacked::RenormSmCrPacked<D_T_X> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 9) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        NsRenormGlobalHighP::RenormGlobalHighP<D_T_X> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 10) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        NsRenormGlobalLongP::RenormGlobalLongP<D_T_X> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 11) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        NsRenormSmCrPacked::RenormSmCrPacked<D_T_X, true> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 12) {
        NsRenormSmStPipelined::RenormSmStPipelined op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (TEMPLATE == 13) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        NsRenormInnerSplit::RenormInnerSplit<D_T_X> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 14) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        NsRenormSmCrPacked::RenormSmCrPacked<D_T_X, true, true> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 15) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        NsRenormBmCrTiled::RenormBmCrTiled<D_T_X> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 16) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        // C6 keeps the [slice, aligned block] UB layout.  The dense packed
        // layout is only valid when every slice row is vector aligned.
        NsRenormSmCrPacked::RenormSmCrPacked<D_T_X> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 17) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        // Case 3832 follows the reference's direct FP32 pow overflow result.
        NsRenormSmCrPacked::RenormSmCrPacked<D_T_X, true, true, true> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 18) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        NsRenormSmCrPacked::RenormSmCrPacked<D_T_X, true, true, false, true> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 19) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        NsRenormSmCrPacked::RenormSmCrPacked<D_T_X, true, true, false, false, true> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 20) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        NsRenormSmCrPacked::RenormSmCrPacked<D_T_X, true, true, false, false, false, true> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 21) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        NsRenormSmCrPacked::RenormSmCrPacked<D_T_X, true, true, false, false, false, false, true> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 22) {
        // F2 keeps each logical [slice, block] row dense in UB.  This avoids
        // the per-row DataCopyPad fallback used by template 5 on 30/34B FP16 rows.
        NsRenormBmVg::RenormBmVg<D_T_X, true> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (TEMPLATE == 23) {
        NsRenormBmVg::RenormBmVg<D_T_X, true, true> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (TEMPLATE == 24) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        NsRenormSmCrPacked::RenormSmCrPacked<D_T_X, true, true, false, true, false, false, false, true> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 25) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        NsRenormSmCrPacked::RenormSmCrPacked<D_T_X, true, true, false, true, false, false, false, false, true> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 26) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        NsRenormInnerSplit::RenormInnerSplit<D_T_X, true> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 27) {
        NsRenormBmVg::RenormBmVg<D_T_X, true> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (TEMPLATE == 28) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        NsRenormSmCrPacked::RenormSmCrPacked<D_T_X, true, true, false, true, false, false, false, true, false, false>
            op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 30) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        NsRenormGlobal::RenormGlobal<D_T_X> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 31) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        NsRenormGlobal::RenormGlobal<D_T_X, true> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 32) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        NsRenormInnerSplit::RenormInnerSplit<D_T_X, true, true> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 33) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        NsRenormSmCrPacked::RenormSmCrPacked<D_T_X, true, true, false, true, true, false, false, true, false, true,
                                             true>
            op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 34) {
        // F5 keeps the established packed-row output pass, but performs the
        // p=inf reduction directly in FP16 for the isolated long-row shape.
        NsRenormBmVg::RenormBmVg<D_T_X, true, false, true> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (TEMPLATE == 35) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        NsRenormSmCrPacked::RenormSmCrPacked<D_T_X, true, true, false, false, false, false, true> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 36) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        // Keep the original per-block AR reduction for the unaligned dense
        // row, but use the compact UB allocation selected by host tiling.
        NsRenormSmCrPacked::RenormSmCrPacked<D_T_X, true, true, false, false, true, false, false, false, false, false,
                                             false, true>
            op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 37) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        // Keep the legacy B3 key on the ordinary two-level implementation.
        NsRenormSmTl::RenormSmTl<D_T_X> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 38) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        NsRenormGlobalHighP::RenormGlobalHighP<D_T_X> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 39) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        // Batch RA reduction with the reference's direct FP32 pow semantics.
        NsRenormSmCrPacked::RenormSmCrPacked<D_T_X, true, true, false, false, false, false, true, false, false, false,
                                             false, true, true>
            op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 40) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        // C19 uses C18's arithmetic but verifies an inevitable direct-pow
        // overflow before touching the full reduction stream.
        NsRenormSmCrPacked::RenormSmCrPacked<D_T_X, true, true, false, false, false, true, true, false, false, false,
                                             false, true, true>
            op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 41) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        // Oversized [slice, inner] rows: tile the slice axis and accumulate
        // one partial vector per core before the cross-core atomic merge.
        NsRenormBmCrTiled::RenormBmCrTiled<D_T_X, true> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 42) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        NsRenormInnerSplit::RenormInnerSplit<D_T_X, false, false, true> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 43) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        NsRenormInnerSplit::RenormInnerSplit<D_T_X, false, false, false, true> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 44) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        // Dense finite integer-p route: replace Log/Muls/Exp with a
        // vectorized exponentiation-by-squaring sequence.  It is isolated
        // from the established C16/C17 keys and only selected for exact
        // integer p values by host tiling.
        NsRenormSmCrPacked::RenormSmCrPacked<D_T_X, true, true, false, false, false, false, true, false, false, false,
                                             false, false, false, true>
            op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 45) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        // C18 packed B=1 layout with integer-p arithmetic. Keep the direct
        // packed-row reduction and only replace the expensive Log/Exp pair.
        NsRenormSmCrPacked::RenormSmCrPacked<D_T_X, true, true, false, false, false, false, true, false, false, false,
                                             false, false, true, true>
            op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 46) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        // Compact C21 layout for exact dense integer-p rows. p=90 also uses
        // the shorter addition chain; other C21 routes remain unchanged.
        NsRenormSmCrPacked::RenormSmCrPacked<D_T_X, true, true, false, false, false, false, true, false, false, false,
                                             false, true, false, true, true>
            op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 47) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        // C22 arithmetic with contiguous short rows. RAW_CONTIGUOUS_B1_RA
        // removes padding from both reduction and scale-application passes.
        NsRenormSmCrPacked::RenormSmCrPacked<D_T_X, true, true, false, false, false, false, true, true, false, false,
                                             false, false, true, true>
            op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 48) {
        // Generic scalar reduction. The host selects this key only from
        // reduction geometry and arithmetic risk.
        NsRenorm::Renorm<D_T_X, true> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (TEMPLATE == 59) {
        // High-order positive-p scalar reduction with max-normalized
        // accumulation, selected from arithmetic risk rather than shape.
        NsRenorm::Renorm<D_T_X, true, true> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (TEMPLATE == 49) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        // C49 retains C18 direct-pow arithmetic but stops local reduction
        // once each p-norm accumulator has reached positive infinity.
        NsRenormSmCrPacked::RenormSmCrPacked<D_T_X, true, true, false, false, false, false, true, false, false, false,
                                             false, false, true, false, false, true>
            op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 50) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        // Exact long-row route: prove direct-pow overflow from a conservative
        // max bound before falling back to H4's complete positive-p path.
        NsRenormInnerSplit::RenormInnerSplit<D_T_X, false, false, true, true> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 51) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        // C51 is C18's direct-pow packed reduction with a per-core rebased
        // GM view. The host selects it only for the >4GB BF16 row, where a
        // single GlobalTensor offset exceeds A5's DMA addressable range.
        NsRenormSmCrPacked::RenormSmCrPacked<D_T_X, true, true, false, false, false, false, true, true, false, false,
                                             false, true, true, false, false, false, true>
            op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 52) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        // C52 keeps the flat high-p arithmetic but rebases each core's GM
        // view before any DMA, which is required for the >4GB BF16 row.
        NsRenormGlobalHighP::RenormGlobalHighP<D_T_X, true> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 55) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
        AscendC::TPipe pipe;
        NsRenormInnerSplit::RenormInnerSplit<D_T_X, false, false, false, true, true> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (TEMPLATE == 58) {
        // Isolated scalar Template-A entry. It retains the reference direct
        // p-power arithmetic while using an independent launch key.
        NsRenorm::Renorm<D_T_X, true> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}
