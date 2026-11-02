/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gru_grad_transpose_arch35.h
 * \brief 转置物化（条带 Gather）、输出回投与偏置规约的类外定义。
 */
#ifndef _GRU_GRAD_TRANSPOSE_ARCH35_
#define _GRU_GRAD_TRANSPOSE_ARCH35_

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::InitTransGatherOffsets()
{
    auto offView = this->ubTmp2.template ReinterpretCast<uint32_t>();
    for (int64_t r = 0; r < TRANS_TILE_COLS; ++r) {
        for (int64_t m = 0; m < TRANS_STRIP_COLS; ++m) {
            // 字节粒度偏移: staged[r*SC+m] = loaded[m*TRANS_TILE_COLS+r]
            offView.SetValue(r * TRANS_STRIP_COLS + m, static_cast<uint32_t>((m * TRANS_TILE_COLS + r) * FLOAT_BYTES));
        }
    }
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::TransposeOnePlane(GlobalTensor<DTYPE>& srcGm, GlobalTensor<DTYPE>& dstGm,
                                                               int64_t rows, int64_t cols, int64_t colStart,
                                                               int64_t colEnd, int64_t localIdx, int64_t pairStride,
                                                               int64_t dstRowPitch)
{
    // 列带归属：本簇仅处理源列 [colStart, colEnd)（= dst 行带），生产/消费同簇闭环。
    // dstRowPitch: dst 平面行距——默认等于 rows(整面转置); D6 分段直读时传全量 TS
    // (dst 为 [cols, TS] 大平面的列偏移视图, 行距与段内行数不同)。
    const int64_t dstPitch = (dstRowPitch > 0) ? dstRowPitch : rows;
    const int64_t groups = (rows + TRANS_GROUP - 1) / TRANS_GROUP;
    const int64_t bandCols = colEnd - colStart;
    // ---- fp32 标量混合转置：[16 行 x 8 列] tile -------------------------------------------
    // 原 per-column 路径为大量微搬运+事件，MTE3 成为瓶颈。
    // 新路径：1 次多块 MTE2 载入（8 列 = 32B 块，快路径，srcStride 为间隙语义 = (cols-8)*4）
    //         + 片内标量转置（目的行 16 元连续，标量 store 写合并；标量访存按 4-8 周期/元素
    //           估算 ~153K 元素/AIV ≈ 1ms，对比 MTE 微事务 ~17ms）。
    // fp16 保持原路径（CopyInRow 多行 half 暂存存在 pitch 错配，标量 SetValue 路径后续可平移）。
    if constexpr (sizeof(DTYPE) == FLOAT_BYTES) {
        // ⚠ rows 非 16 整除(I=520 实证: Gather 批次残留 6-9 行 [0,8) 位零元素, 根因未定位)——
        // 整体回退逐列 PROVEN 路径(慢但正确); 16 对齐 shape 保持 v4+Gather。
        if (rows % TRANS_GROUP != 0) {
            const int64_t totalCols = colEnd - colStart;
            for (int64_t task = localIdx; task < groups * totalCols; task += pairStride) {
                const int64_t g = task / totalCols;
                const int64_t m = colStart + task % totalCols;
                const int64_t r0 = g * TRANS_GROUP;
                const int64_t rCnt = (rows - r0 < TRANS_GROUP) ? (rows - r0) : TRANS_GROUP;
                CopyInRow(srcGm, this->ubTmp, r0 * cols + m, rCnt, 1, this->alignPara, cols - 1);
                SyncM2toV();
                CopyOutRow(dstGm, this->ubTmp, m * dstPitch + r0, rCnt, 1, 0, this->alignPara);
                SyncM3toM2();
            }
            GlobalTensor<DTYPE> dstView;
            dstView.SetGlobalBuffer(dstGm.GetPhyAddr(colStart * dstPitch), (colEnd - colStart) * dstPitch);
            DataCacheCleanAndInvalid<DTYPE, AscendC::CacheLine::ENTIRE_DATA_CACHE>(dstView);
            return;
        }
        constexpr int64_t TILE_C = TRANS_TILE_COLS; // 8 列 tile（fp32 恰为 32B 块）
        constexpr int64_t STAGE_OFF = 2048;
        constexpr int64_t SC = TRANS_STRIP_COLS;
        // A3: 条带化转置——[8 列 × SC 行] run: 1 MTE2 + 1 Gather + 1 M3。
        // 消融实证 v4 每-tile 微事务结构使转置占内核 49.2%(3.8ms, 有效带宽仅 ~10GB/s);
        // 条带合并后微事务数 /31(xT) ~ /16(hPrevT), 且 8 对齐列带分配消除 4B 尾列中继
        // (bench: xT band=5 列全中继 / hPrevT band=10 列 2 列中继——微事务风暴主源)。
        // 同步链: [load → M2→V → Gather → V→M3 → M3 → M3→M2], 逐 run 的 M3→M2 经
        // [MTE3→MTE2→V] 传递序护下 run Gather 对暂存区的覆写（与 v4 批结构同构）。
        const int64_t fullSub = bandCols / TILE_C;
        const int64_t tailCols = bandCols % TILE_C;
        const int64_t runsPerSub = (rows + SC - 1) / SC;
        const int64_t totalRuns = fullSub * runsPerSub;
        for (int64_t run = localIdx; run < totalRuns; run += pairStride) {
            const int64_t sb = run / runsPerSub;
            const int64_t rc = run % runsPerSub;
            const int64_t c0 = colStart + sb * TILE_C;
            const int64_t r0 = rc * SC;
            const int64_t rowsThis = (rows - r0 < SC) ? (rows - r0) : SC;
            // 载入 [rowsThis × 8]（块长 32B 多行快路径，单 MTE2），行距 8 落 ubTmp[0,2048)
            CopyInRow(srcGm, this->ubTmp, r0 * cols + c0, rowsThis, TILE_C, TILE_C, cols - TILE_C);
            SyncM2toV();
            // 条带置换: staged[r*SC+m] = loaded[m*8+r]（表固定; 尾 run 垃圾尾 M3 截断）
            Gather(this->ubTmp[STAGE_OFF], this->ubTmp, this->ubTmp2.template ReinterpretCast<uint32_t>(),
                   static_cast<uint32_t>(0), static_cast<uint32_t>(TILE_C * SC));
            SyncVtoM3();
            // 条带写出: dst 行 [c0, c0+8) × 列 [r0, r0+rowsThis)——单 M3
            // (UB 侧 srcStride 单位 32B 块——CopyOutRow 同款; GM 侧字节)
            DataCopyExtParams cp(static_cast<uint16_t>(TILE_C), static_cast<uint32_t>(rowsThis * FLOAT_BYTES),
                                 static_cast<int64_t>((SC - rowsThis) * FLOAT_BYTES / BLOCK_BYTES),
                                 static_cast<int64_t>(dstPitch - rowsThis) * FLOAT_BYTES, 0);
            DataCopyPad(dstGm[c0 * dstPitch + r0], this->ubTmp[STAGE_OFF], cp);
            SyncM3toM2();
        }
        // 尾列（band 非 8 整除时）：MTE3 写出(SetValue 标量 GM 写实证不可靠——H=42/85/86/129/
        // 170/171 + TS%16==0 尾列 16 元窗口零化/错位, dwHh 错而 dwIh/dx/dh0 全对)。
        constexpr int64_t TAIL_GRP = 8; // 尾列中继槽位数（组内 8 列各占独立槽，与 TRANS_TILE_COLS 同宽）
        const int64_t slotElems = TRANS_GROUP * this->alignPara;
        int64_t pendCnt[TAIL_GRP];
        int64_t pendR0[TAIL_GRP];
        int64_t pendM[TAIL_GRP];
        int64_t staged = 0;
        for (int64_t task = localIdx; task < groups * tailCols; task += pairStride) {
            const int64_t g = task / tailCols;
            const int64_t m = colStart + fullSub * TILE_C + task % tailCols;
            const int64_t r0 = g * TRANS_GROUP;
            const int64_t rCnt = (rows - r0 < TRANS_GROUP) ? (rows - r0) : TRANS_GROUP;
            // [性能] 批量中继: 组内 8 列各占独立槽位(staged 顺序发射裸 MTE2), 一次 MTE2->MTE3
            // 直通窄事件后顺序发射裸 MTE3 写出, 组尾一次 MTE3->MTE2 排空(护下组槽位复用)。
            // 原泛化期逐列 CopyInRow+SyncM2toV+CopyOutRow(SyncVtoM3)+SyncM3toM2 = 3 事件/列,
            // hPrevT 尾列跨簇×多行组场景的尾列微搬运
            // (bench 7.3→11.7ms 主因); SetValue 旧版快但标量 GM 写实证不可靠(H=42/85/86/
            // 129/170/171 尾列 16 元窗口零化/错位)——保持 MTE3 写出语义, 仅重组事件结构。
            if (staged == TAIL_GRP) {
                // 纯 MTE 中继: MTE2->MTE3 直通窄事件。V 标量紧致化(GetValue/SetValue)两式实证
                // 均非确定竞态(492 rel 0.2-0.8 / 219 rel 7e+13, 加 PIPE_ALL 无效——标量写 UB
                // 对 MTE3 读疑似可见性问题非定序问题), 弃用; 16 条 4B 微事务/列为已证安全代价。
                SyncM2toM3();
                for (int64_t i = 0; i < staged; ++i) {
                    const LocalTensor<float> slot = this->ubTmp3[i * slotElems];
                    for (int64_t r = 0; r < pendCnt[i]; ++r) {
                        DataCopyExtParams cp(1, static_cast<uint32_t>(FLOAT_BYTES), 0, 0, 0);
                        DataCopyPad(dstGm[pendM[i] * dstPitch + pendR0[i] + r], slot[r * this->alignPara], cp);
                    }
                }
                SyncM3toM2();
                staged = 0;
            }
            {
                const LocalTensor<float> slot = this->ubTmp3[staged * slotElems];
                for (int64_t r = 0; r < rCnt; ++r) {
                    DataCopyExtParams cp(1, static_cast<uint32_t>(FLOAT_BYTES), 0, 0, 0);
                    DataCopyPadExtParams<float> pp(true, 0, 0, 0);
                    DataCopyPad(slot[r * this->alignPara], srcGm[r0 * cols + m + r * cols], cp, pp);
                }
                pendCnt[staged] = rCnt;
                pendR0[staged] = r0;
                pendM[staged] = m;
                ++staged;
            }
        }
        if (staged > 0) {
            SyncM2toM3();
            for (int64_t i = 0; i < staged; ++i) {
                const LocalTensor<float> slot = this->ubTmp3[i * slotElems];
                for (int64_t r = 0; r < pendCnt[i]; ++r) {
                    DataCopyExtParams cp(1, static_cast<uint32_t>(FLOAT_BYTES), 0, 0, 0);
                    DataCopyPad(dstGm[pendM[i] * dstPitch + pendR0[i] + r], slot[r * this->alignPara], cp);
                }
            }
            SyncM3toM2();
        }
        // 纪律 5：本核写过的 dst 行带（[colStart, colEnd) x [0, rows) 连续区），终轮 V2C 前一次刷新
        if (bandCols > 0) {
            GlobalTensor<DTYPE> dstView;
            dstView.SetGlobalBuffer(dstGm.GetPhyAddr(colStart * rows), bandCols * rows);
            DataCacheCleanAndInvalid<DTYPE, AscendC::CacheLine::ENTIRE_DATA_CACHE>(dstView);
        }
        return;
    }
    // ---- fp16：原 per-column 路径（MTE 微事务，待标量路径平移） ---------------------------
    const int64_t totalTasks = groups * bandCols;
    for (int64_t task = localIdx; task < totalTasks; task += pairStride) {
        const int64_t g = task / bandCols;
        const int64_t m = colStart + task % bandCols;
        const int64_t r0 = g * TRANS_GROUP;
        const int64_t rCnt = (rows - r0 < TRANS_GROUP) ? (rows - r0) : TRANS_GROUP;
        CopyInRow(srcGm, this->ubTmp, r0 * cols + m, rCnt, 1, this->alignPara, cols - 1);
        SyncM2toV();
        CopyOutRow(dstGm, this->ubTmp, m * dstPitch + r0, rCnt, 1, 0, this->alignPara);
        SyncM3toM2();
        GlobalTensor<DTYPE> dstView;
        dstView.SetGlobalBuffer(dstGm.GetPhyAddr(m * rows + r0), rCnt);
        DataCacheCleanAndInvalid<DTYPE, AscendC::CacheLine::ENTIRE_DATA_CACHE>(dstView);
    }
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::TransposeGiGhT()
{
    const int64_t ts = this->totalSteps_;
    const int64_t cluster = static_cast<int64_t>(GetBlockIdx()) / AIV_PER_AIC;
    const int64_t localIdx = static_cast<int64_t>(GetBlockIdx()) % AIV_PER_AIC;
    const int64_t numClusters = static_cast<int64_t>(GetBlockNum());
    this->InitTransGatherOffsets();
    // dW^T 重排: 只转置 x[TS,I]->xT[I,TS] 与 hPrevWs[TS,H]->hPrevT[H,TS](体积 6x/3x↓)
    // A3: 8 列子带对齐分配——消非对齐列带的 4B 尾列中继（旧任意宽 band 使 bench 的 xT
    // band=5 列全走中继 / hPrevT band=10 列 2 列中继，为微事务风暴主源）。子带数
    // subBands=cols/8 均衡分布到活跃簇；真尾列（cols%8）归最后活跃簇走中继路径。
    {
        const int64_t cols = this->I;
        const int64_t subBands = cols / TRANS_TILE_COLS; // 8 列子带
        int64_t activeClusters = (numClusters < subBands) ? numClusters : subBands;
        activeClusters = (activeClusters < 1) ? 1 : activeClusters;
        const int64_t per = subBands / activeClusters;
        const int64_t rem = subBands % activeClusters;
        if (cluster < activeClusters) {
            const int64_t sb0 = cluster * per + (cluster < rem ? cluster : rem);
            const int64_t sbCnt = per + (cluster < rem ? 1 : 0);
            const int64_t c0 = sb0 * TRANS_TILE_COLS;
            int64_t c1 = c0 + sbCnt * TRANS_TILE_COLS;
            if (cluster == activeClusters - 1) {
                c1 = cols; // 真尾列归最后活跃簇
            }
            if (c0 < c1) {
                this->TransposeOnePlane(this->inputGm.xGm, this->xTGm, ts, cols, c0, c1, localIdx, AIV_PER_AIC);
            }
        }
    }
    {
        const int64_t cols = this->H;
        const int64_t subBands = cols / TRANS_TILE_COLS; // 8 列子带
        int64_t activeClusters = (numClusters < subBands) ? numClusters : subBands;
        activeClusters = (activeClusters < 1) ? 1 : activeClusters;
        const int64_t per = subBands / activeClusters;
        const int64_t rem = subBands % activeClusters;
        if (cluster < activeClusters) {
            const int64_t sb0 = cluster * per + (cluster < rem ? cluster : rem);
            const int64_t sbCnt = per + (cluster < rem ? 1 : 0);
            const int64_t c0 = sb0 * TRANS_TILE_COLS;
            int64_t c1 = c0 + sbCnt * TRANS_TILE_COLS;
            if (cluster == activeClusters - 1) {
                c1 = cols;
            }
            if (c0 < c1) {
                this->TransposeOnePlane(this->workGm.hPrevWsGm, this->hPrevTGm, ts, cols, c0, c1, localIdx,
                                        AIV_PER_AIC);
            }
        }
    }
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::ConvertOnePlane(GlobalTensor<float>& srcGm, GlobalTensor<DTYPE>& dstGm,
                                                             int64_t rows, int64_t cols, int64_t rowStart,
                                                             int64_t rowEnd, int64_t localIdx, int64_t pairStride)
{
    // 行带归属：与 RunBatchedGmms 的 C 平面行带一致，同簇生产同簇消费。
    // fp32 行分批：每块 rowsPerBlock 行整块读写（多行 32B 对齐快路径，事件数随块数而非行数下降；
    // 原 fp32 逐行 3 事件 × ~12.8K 行/核是尾部大头之一）。
    // ⚠ 仅 fp32：fp16 的 CopyInRow/CopyOutRow 多行 staging 存在 pitch 错配（half 暂存行距
    // pitchBytes/dSize=2*cols 与 Cast 的连续读模式不符），保持逐行。
    // ⚠ 宽行(cols>allocLength, 即 I>4096)：rowsPerBlock=allocLength/cAl=0 → 首块 rCnt<0 直接
    // break, 平面从未回投(I=)；fp16 逐行同宽越界 ubTmp。
    // 修复: 列方向按 chunk(allocLength 内 alignPara 倍数)分块, 单行单块逐块回投。
    const int64_t cAl = (cols + this->alignMask) / this->alignPara * this->alignPara;
    if (cAl > this->allocLength) {
        const int64_t chunk = (this->allocLength / this->alignPara) * this->alignPara;
        for (int64_t r = rowStart + localIdx; r < rowEnd; r += pairStride) {
            for (int64_t c0 = 0; c0 < cols; c0 += chunk) {
                const int64_t cCnt = (cols - c0 < chunk) ? (cols - c0) : chunk;
                const int64_t cAligned = ((cCnt + this->alignMask) / this->alignPara) * this->alignPara;
                CopyInRow(srcGm, this->ubTmp, r * cols + c0, 1, cCnt, cAligned, 0);
                SyncM2toV();
                CopyOutRow(dstGm, this->ubTmp, r * cols + c0, 1, cCnt, 0, cAligned);
                SyncM3toM2();
            }
        }
        return;
    }
    if constexpr (sizeof(DTYPE) == FLOAT_BYTES) {
        // ⚠ UB 行距必须 32B 对齐: 原rowAlign=cols 在 cols%8≠0 时多行慢路径 ub[cols*r] 未对齐
        // → MTE 未对齐 UB 访问 → AIV 陷阱 → AIC 等待超时(
        const int64_t rowsPerBlock = this->allocLength / (cAl > 0 ? cAl : 1);
        const int64_t blockStride = rowsPerBlock * pairStride;
        for (int64_t r0 = rowStart + localIdx * rowsPerBlock; r0 < rowEnd; r0 += blockStride) {
            const int64_t rCnt = (rowEnd - r0 < rowsPerBlock) ? (rowEnd - r0) : rowsPerBlock;
            if (rCnt <= 0) {
                break;
            }
            CopyInRow(srcGm, this->ubTmp, r0 * cols, rCnt, cols, cAl, 0);
            SyncM2toV();
            CopyOutRow(dstGm, this->ubTmp, r0 * cols, rCnt, cols, 0, cAl);
            SyncM3toM2();
        }
    } else {
        for (int64_t r = rowStart + localIdx; r < rowEnd; r += pairStride) {
            CopyInRow(srcGm, this->ubTmp, r * cols, 1, cols, cols, 0);
            SyncM2toV();
            CopyOutRow(dstGm, this->ubTmp, r * cols, 1, cols, 0, cols);
            // ⚠ 下一迭代 CopyInRow（MTE2 写 ubTmp）与本迭代 CopyOutRow（MTE3 读 ubTmp）分属不同 pipe，
            // SyncM3toV 只拦 V 管线、拦不住 MTE2 队列——确定性 WAR（）。补 MTE3->MTE2。
            SyncM3toM2();
        }
    }
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::ConvertTransposedPlane(GlobalTensor<float>& srcGm,
                                                                    GlobalTensor<DTYPE>& dstGm, int64_t srcRows,
                                                                    int64_t srcCols, int64_t rowStart, int64_t rowEnd,
                                                                    int64_t localIdx, int64_t pairStride)
{
    // 逐列分块转置回投：fp32 src [srcRows, srcCols] -> DTYPE dst [srcCols, srcRows]。
    // ⚠ 原批量行分块版缺陷（已废弃）：staging 基址 ubTmp[k*srcRows] 与元素行距 alignPara(8) 不匹配，
    //   列 k 驻留区间 [k*srcRows, k*srcRows + srcRows*8) 必然覆写列 k+1 的基址区（r≥srcRows/8 全被踩），
    //   且 rCnt 列 × pitch8 总跨度超 ubTmp 。
    //   改为 HEAD TransposeOnePlane 同构：逐列 + 块高 stageRows，单列驻留 [0, stageRows*8) 恰满预算。
    const int64_t stageRows = this->allocLength / this->alignPara; // pitch8 下单列驻留上限（=512）
    for (int64_t j = rowStart + localIdx; j < rowEnd; j += pairStride) {
        for (int64_t r0 = 0; r0 < srcRows; r0 += stageRows) {
            const int64_t rCnt = (srcRows - r0 < stageRows) ? (srcRows - r0) : stageRows;
            // 按列收集：src 第 j 列的 [r0, r0+rCnt)（fp32 直读，fp16 由 CopyOutRow 出向 Cast）
            CopyInRow(srcGm, this->ubTmp, j + r0 * srcCols, rCnt, 1, this->alignPara, srcCols - 1);
            SyncM2toV();
            // 按行散布：dst 第 j 行的 [r0, r0+rCnt)，GM 行距 1（连续）
            CopyOutRow(dstGm, this->ubTmp, j * srcRows + r0, rCnt, 1, 0, this->alignPara);
            SyncM3toM2();
        }
    }
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::ConvertOutputs()
{
    const int64_t cols3H = this->H * GRU_GATE_SIZE;
    const int64_t cluster = static_cast<int64_t>(GetBlockIdx()) / AIV_PER_AIC;
    const int64_t localIdx = static_cast<int64_t>(GetBlockIdx()) % AIV_PER_AIC;
    const int64_t numClusters = static_cast<int64_t>(GetBlockNum());
    // A6: 8 列子带对齐分配——消 dW 转置回投的尾列 4B 微事务中继
    // (原 band3H=ceil(768/28)=28 非 8 倍数 → 4 尾列/簇 × 28 簇 × 2 面 ≈ 682 微 MTE3, ESL 实证)
    const int64_t subBands3H = cols3H / TRANS_TILE_COLS; // 8 列子带
    int64_t activeClusters = (numClusters < subBands3H) ? numClusters : subBands3H;
    activeClusters = (activeClusters < 1) ? 1 : activeClusters;
    const int64_t per = subBands3H / activeClusters;
    const int64_t rem = subBands3H % activeClusters;
    if (cluster < activeClusters) {
        const int64_t sb0 = cluster * per + (cluster < rem ? cluster : rem);
        const int64_t sbCnt = per + (cluster < rem ? 1 : 0);
        const int64_t r0 = sb0 * TRANS_TILE_COLS;
        int64_t r1 = r0 + sbCnt * TRANS_TILE_COLS;
        if (cluster == activeClusters - 1) {
            r1 = cols3H; // 真尾列(cols3H%8) 归最后活跃簇
        }
        if (r0 < r1) {
            if constexpr (sizeof(DTYPE) == FLOAT_BYTES) {
                this->InitTransGatherOffsets(); // ubTmp2 偏移表可能被 BiasReduce 覆盖, 重初始化
                this->TransposeOnePlane(this->dwIF32Gm, this->outputGm.dwInputGm, this->I, cols3H, r0, r1, localIdx,
                                        AIV_PER_AIC);
                this->TransposeOnePlane(this->dwHF32Gm, this->outputGm.dwHiddenGm, this->H, cols3H, r0, r1, localIdx,
                                        AIV_PER_AIC);
            } else {
                this->ConvertTransposedPlane(this->dwIF32Gm, this->outputGm.dwInputGm, this->I, cols3H, r0, r1,
                                             localIdx, AIV_PER_AIC);
                this->ConvertTransposedPlane(this->dwHF32Gm, this->outputGm.dwHiddenGm, this->H, cols3H, r0, r1,
                                             localIdx, AIV_PER_AIC);
            }
        }
    }
    // fp32 时 dx 由 GEMM 直写输出(RunDxGemm), 跳过回投
    if constexpr (sizeof(DTYPE) != FLOAT_BYTES) {
        const int64_t TS = this->totalSteps_;
        const int64_t bandTS = (TS + numClusters - 1) / numClusters;
        const int64_t d0 = cluster * bandTS;
        if (d0 < TS) {
            const int64_t d1 = (TS - d0 < bandTS) ? TS : (d0 + bandTS);
            this->ConvertOnePlane(this->dxF32Gm, this->outputGm.dxGm, TS, this->I, d0, d1, localIdx, AIV_PER_AIC);
        }
    }
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::ProcessBiasReduce(GlobalTensor<DTYPE>& srcGm, GlobalTensor<DTYPE>& dstGm,
                                                               int64_t rows, int64_t cols)
{
    SyncM3toV();
    int64_t nReduceCnt = this->tiling->nReduceCnt;
    int64_t singleCoreReduceN = this->tiling->singleCoreReduceN;
    if (nReduceCnt <= 0)
        return;
    if (static_cast<int64_t>(GetBlockIdx()) >= nReduceCnt)
        return;
    int64_t nIdx = GetBlockIdx();
    int64_t nStart = nIdx * singleCoreReduceN;
    int64_t nCnt = (nIdx == nReduceCnt - 1) ? this->tiling->singleCoreReduceNTail : singleCoreReduceN;
    if (nStart >= cols || nCnt <= 0)
        return;
    if (nStart + nCnt > cols)
        nCnt = cols - nStart;

    int64_t maxNOnce = allocLength;

    for (int64_t cStart = 0; cStart < nCnt; cStart += maxNOnce) {
        int64_t cCnt = (cStart + maxNOnce > nCnt) ? (nCnt - cStart) : maxNOnce;
        int64_t cAligned = ((cCnt + this->alignMask) / this->alignPara) * this->alignPara;
        int64_t cGlobalOff = nStart + cStart;

        Duplicate(this->ubTmp, 0.0f, cAligned);
        SyncVtoM2();
        // fp32 行分批：每批 ROW_BATCH 行整块载入（多行快路径）后 UB 内 V 管逐行累加
        // （950PR 同管 RAW 互锁免事件），事件数降 ROW_BATCH 倍——原逐行 2 事件
        // × 2 平面是尾部 scalar 开销大头。fp16 受 CopyInRow 多行 staging pitch 限制保持逐行。
        if constexpr (sizeof(DTYPE) == FLOAT_BYTES) {
            const int64_t rowBatch = (this->allocLength / cAligned) > 0 ? (this->allocLength / cAligned) : 1;
            for (int64_t r0 = 0; r0 < rows; r0 += rowBatch) {
                const int64_t rCnt = (rows - r0 < rowBatch) ? (rows - r0) : rowBatch;
                // ⚠ 快路径 srcStride 为【间隙】语义(字节): 列切片行距 cols, 块长 cCnt,
                // 间隙 = (cols - cCnt) 元素。gmStride=cols(全 pitch)或 0(连续)皆错——
                // dx 回披(整行连续, 间隙 0)与转置(慢路径, 元素 pitch)是两套语义, 勿混。
                CopyInRow(srcGm, this->ubTmp2, r0 * cols + cGlobalOff, rCnt, cCnt, cAligned, cols - cCnt);
                SyncM2toV();
                for (int64_t r = 0; r < rCnt; ++r) {
                    Add(this->ubTmp, this->ubTmp, this->ubTmp2[r * cAligned], cAligned);
                }
                SyncVtoM2();
            }
        } else {
            for (int64_t r = 0; r < rows; ++r) {
                CopyInRow(srcGm, this->ubTmp2, r * cols + cGlobalOff, 1, cCnt, cAligned, 0);
                SyncM2toV();
                Add(this->ubTmp, this->ubTmp, this->ubTmp2, cAligned);
                SyncVtoM2();
            }
        }
        SyncVtoM3();
        CopyOutRow(dstGm, this->ubTmp, cGlobalOff, 1, cCnt, 0, cAligned);
        SyncM3toV();
    }
}

#endif // _GRU_GRAD_TRANSPOSE_ARCH35_
