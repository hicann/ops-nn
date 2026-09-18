/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file conv_bp_dload_compute.h
 * \brief DLoad（d 轴 MTE2 载入）模板 L0 搬运与 Mmad 计算循环——模板参 SrcT 仅为
 *        输入类型；L0C/输出 y 恒 fp32（非 fp32 输出由外部通路转换）
 *
 * 布局与协议：
 *   S1. L0A/L0B(SrcT)、L0C(fp32) 线性布局（M/N/K = mmad.m/n/k）：
 *       addrA(m,k) = (k/8)*(8M) + 8m + (k%8)
 *       addrB(k,n) = (k/8)*(8N) + 8n + (k%8)
 *       addrC(m,n) = (n/16)*(16M) + 16m + (n%16)
 *   S2. B1(fmap) 单 batch 驻留 [c1g][d][h][w][c0]，半区内 addr = (ci/C0)*(dhwin*C0)
 *       + d*(hwIn*C0) + hw*C0 + (ci%C0)；每 batch 一条 Dn2Nz{dnNum=1}。
 *       L1 两 bank（ping@0/pong@L1/2）：MTE1/MTE2 落不同 bank，双流预载零 bank 冲突
 *   S3. A1(dy) 单 batch 驻留 [co1g][paddedDhowo][co0]，addr = (co/C0)*(paddedDhowo*C0)
 *       + pt*C0 + (co%C0)，paddedDhowo = CeilAlign(dhowo,16)
 *   S4. load3d(B1→L0B)：每 (窗,dk,tap) 一条命令，rt=cinAlign16/C0、rs=din*hkwk、
 *       dst=l0b[128*cin16G*tap]；L0C 块序 B = cin16G*tap + g
 *   S5. load2d(A1→L0A)：单命令转置整窗，mStep=Ceil(窗宽,16)、kStep=M/8、
 *       srcStride=paddedDhowo/16、dstStride=kStep/2；尾窗 k∈[howoLen,16) garbage 不被
 *       Mmad（k=howoLen）消费
 *   S6. Mmad：A(co,howo)×B(howo,ci 展开)→C(co,ci)，n=cinAlign16*hkwk、k=howoLen；
 *       dk 段 L0C 基址 = cinAlign16*coutAlign16*hkwk*dk 元素（段间 n 区不相交免背压）；
 *       cmatrixInitVal 仅段首（批0/首有效dout/窗0）true；尾组 garbage 由 fixpipe
 *       nSize=真实 cinLength 截断
 *   S7. Mmad 目的必须 TPosition::CO1；LocalTensor(pos, offset, size) 的 offset 单位
 *       字节、size 单位元素
 *   S8. fixpipe L0C→GM 直出 NZ2DN：src = 16*(MS*dn + CS*n + SS*(d/16)) + d%16（C0 单位）、
 *       dst = DM*dn + n + DS*d；MS=1、CS=alignedCout、SS=alignedCout*hkwk、
 *       DM=cinTotal*dhwK、DS=dhwK；y = [cout][cin][dhwK] 行主序
 *   S9. 事件链：成对 id 不复用、释放向 Init 预置、End 消费残留（Set/Wait 逐 id 配平）。
 *       id 分配：MTE2_MTE1/MTE1_MTE2 各 2（L1 半区 0/1）、M_MTE1/MTE1_M 的 L0B 池
 *       0..bBufCnt-1(≤4) + L0A 6/7、FIX_M/M_FIX 半区 id——各通道 ≤ QUE_MAX_EVENT(8)。
 *       事件链每段恒定（含空段），守卫只落数据操作；dk 全 pad 空段跳过 fixpipe（y 保
 *       host 清零值）
 *   S10. LocalTensor 索引/偏移入参必须 u32（L1/L0 偏移恒 < 2^32；u64 传入会在内联
 *        优化下 S64 溢出）；GM 侧偏移用 u64
 */

#ifndef CONV_BP_DLOAD_COMPUTE_H
#define CONV_BP_DLOAD_COMPUTE_H

#include "basic_api/kernel_basic_intf.h"
#include "op_kernel/math_util.h"
#include "utils/std/algorithm.h"

#include "../util/conv_bp_common_util.h"
#include "conv_bp_dload_config.h"

namespace BpDLoad {

using namespace AscendC;
using BpUtils::C0;
using BpUtils::CoutCinRange;

template <typename SrcT>
class DLoadCompute {
public:
    // TOTAL_L1_SIZE/TOTAL_L0C_SIZE 基的半区常量须函数内 constexpr：device 编译器对
    // 模板类作用域初始化的两阶段查找差异下仅函数体可见（L0A 可见于类作用域）
    static constexpr uint32_t L0A_HALF_BYTES = TOTAL_L0A_SIZE / 2; // L0A 双 buf 半区
    // L0B buf 池深度上限（QUE_MAX_EVENT=8 − L0A 固定 2 id(6/7) − 余量）
    static constexpr uint32_t MAX_L0B_BUF_CNT = 4;

    // fmap = 正向输入 NCDHW [batch][cin][din][hin][win]，dy = 反向梯度 NCDHW
    // [batch][cout][dout][hout][wout]
    __aicore__ inline void Init(const GlobalTensor<SrcT>& fmap, const GlobalTensor<SrcT>& dy, const DLoadConfig& config)
    {
        fmap_ = fmap;
        dy_ = dy;
        // L0B buf 池：tile 步长 = kl0HoWo·aCin·hkwk·sizeof（kl0HoWo/aCin 由 tiling 上层
        // 保证 16 对齐，恒为 512B 倍数）；跨块恒定的 buf 物理边界是池背压链的基础
        const ShapeAttribute& shape = config.shape;
        const DLoadTiling& tiling = config.tiling;
        const uint32_t dhwk = shape.dk * shape.hk * shape.wk;
        bBufTileBytes_ = tiling.kl0HoWo * tiling.singleShapeAligned16Cin * (shape.hk * shape.wk) * sizeof(SrcT);
        uint32_t bufCnt = TOTAL_L0B_SIZE / bBufTileBytes_;
        bBufCnt_ = static_cast<uint8_t>(Std::min(bufCnt, MAX_L0B_BUF_CNT));
        // L0C ping-pong 门：单视图占用（aCout×aCin×dhwk 上界口径）≤ L0C/2 则开门——
        // 上界口径保证门与半区几何跨块恒定；半区从 L0C 中间切（ping@0/pong@L0C/2）
        const uint32_t l0cHalfElems = tiling.singleShapeAligned16Cout * tiling.singleShapeAligned16Cin * dhwk;
        constexpr uint32_t l0cHalfBytes = TOTAL_L0C_SIZE / 2;
        l0cPingpong_ = static_cast<uint64_t>(l0cHalfElems) * sizeof(float) <= l0cHalfBytes;

        // 释放向背压预置位（各链首用免等，End 对称消费）：L1 半区 0/1、FIX_M 0/1、
        // L0B 池 0..bBufCnt-1、L0A 6/7
        SetFlag<HardEvent::MTE1_MTE2>(0);
        SetFlag<HardEvent::MTE1_MTE2>(1);
        SetFlag<HardEvent::FIX_M>(0);
        SetFlag<HardEvent::FIX_M>(1);
        // hf32 按 tiling 标志开关（End 关闭——残留开启污染同核后续 kernel）
        SetHF32Mode(tiling.hf32Flag);
        for (uint8_t i = 0; i < bBufCnt_; ++i) {
            SetFlag<HardEvent::M_MTE1>(MTE1FlagFmap(i));
        }
        SetFlag<HardEvent::M_MTE1>(MTE1FlagDy(false));
        SetFlag<HardEvent::M_MTE1>(MTE1FlagDy(true));
    }

    // 全部块迭代后调用：消费跨半区背压残留 Set（与 Init 预置逐项对称）——漏调则残留
    // flag 污染同核后续 kernel
    __aicore__ inline void End()
    {
        WaitFlag<HardEvent::MTE1_MTE2>(0);
        WaitFlag<HardEvent::MTE1_MTE2>(1);
        WaitFlag<HardEvent::FIX_M>(0);
        WaitFlag<HardEvent::FIX_M>(1);
        SetHF32Mode(false);
        for (uint8_t i = 0; i < bBufCnt_; ++i) {
            WaitFlag<HardEvent::M_MTE1>(MTE1FlagFmap(i));
        }
        WaitFlag<HardEvent::M_MTE1>(MTE1FlagDy(false));
        WaitFlag<HardEvent::M_MTE1>(MTE1FlagDy(true));
    }

    // 单个基本块 [coutRange × cinRange] 的 K 全载计算：L0C 全 dhw 驻留单视图（n 轴
    // [dk][hkwk][cin16] 块序），dk 内移循环 batch×dout×howo 窗段内累加，块尾 fixpipe
    // 整块一次直出。y = [cout][cin][dhwK] 行主序，本块基址
    //   yBase = coutIdx*cinTotal*dhwK + cinIdx*dhwK（入口计算），
    //   y[yBase + co*cinTotal*dhwK + ci*dhwK + dk*hwK + tap] = C(co, ci, tap)。
    // B1+A1 双流预载（单 l1pong 链）：预载 batch0 → 循环{装载 i → 计算 i-1} → 末批
    // 循环外计算；装载/计算各 batch 次，配平。
    // 事件链（半区 id p = l1pong）：装载前 Wait<MTE1_MTE2>(p)（等 2 段前同半区 MTE1
    // 排空）→ 装载 → Set<MTE2_MTE1>(p)；计算前 Wait<MTE2_MTE1>(p) → IterateKL0 →
    // Set<MTE1_MTE2>(p)。跨块背压由半区链覆盖（pong 跨块持续，相邻块半区错开）。
    __aicore__ inline void IterateK(const DLoadConfig& config, const CoutCinRange& cRange, GlobalTensor<float>& y,
                                    uint64_t yBase)
    {
        const ShapeAttribute& shape = config.shape;
        const DLoadTiling& tiling = config.tiling;
        const uint32_t alignedCout = tiling.singleShapeAligned16Cout;
        const uint32_t alignedCin = tiling.singleShapeAligned16Cin;
        const uint32_t hkwk = shape.hk * shape.wk;
        const uint32_t dhwk = shape.dk * hkwk;

        // 通道量按本块 16 对齐：cinAlign16 = CeilAlign(cinLength,16)（B1/L0/L0C 视图
        // 单轨贯穿）；coutAlign16 同式（尾块长 ∈ (16k-16,16k]，恒 = alignedCout）
        const uint32_t cinAlign16 = Ops::Base::CeilAlign<uint32_t>(cRange.cinLength, BLOCK_CUBE);
        const uint32_t coutAlign16 = Ops::Base::CeilAlign<uint32_t>(cRange.coutLength, BLOCK_CUBE);

        // howo 窗宽（k 轴），tiling 上层保证 16 倍数（load3d mStartPt 16 对齐）
        const uint32_t howoWin = tiling.kl0HoWo;

        // L1 两 bank：B1 段在前（dhwin×aCin，上界口径跨块恒定）+ A1 段紧随
        // （bl1HalfBytes = dhwin×aCin×sizeof，aCin 16 对齐 ⇒ 64B 倍数天然对齐）
        const uint32_t dhwin = shape.din * shape.hin * shape.win;
        const uint32_t bl1HalfElems = dhwin * alignedCin;
        const uint32_t al1Elems = alignedCout * shape.dout *
                                  Ops::Base::CeilAlign<uint32_t>(shape.hout * shape.wout, BLOCK_CUBE);
        const uint32_t bl1HalfBytes = bl1HalfElems * sizeof(SrcT);

        // L0C 恒 fp32；开门时半区化（基址 = l0cId × L0C/2），关门全区单视图
        LocalTensor<float> l0c(TPosition::CO1, 0, TOTAL_L0C_SIZE / sizeof(float));
        const uint8_t l0cId = l0cPingpong_ ? l0cPong_ : 0;
        if (l0cPingpong_) {
            l0cPong_ = !l0cPong_;
            // size 取 L0C 剩余容量（≥ 实占——size 精确=用量的仿真器容量校验陷阱规避）
            constexpr uint32_t l0cHalfBytes = TOTAL_L0C_SIZE / 2;
            l0c = LocalTensor<float>(TPosition::CO1, l0cId * l0cHalfBytes,
                                     (TOTAL_L0C_SIZE - l0cId * l0cHalfBytes) / sizeof(float));
        }

        // load3d 状态外提：Fmatrix/padding 仅依赖 shape，块首设置一次（L3D_NO_RESET 免重设）
        const uint8_t fmatrixPadList[4] = {static_cast<uint8_t>(shape.wPad), static_cast<uint8_t>(shape.wPad),
                                           static_cast<uint8_t>(shape.hPad), static_cast<uint8_t>(shape.hPad)};
        SetFmatrix(static_cast<uint16_t>(shape.hin), static_cast<uint16_t>(shape.win), fmatrixPadList,
                   FmatrixMode::FMATRIX_LEFT);
        SetLoadDataPaddingValue(static_cast<SrcT>(0));

        // batch 装载/计算合并为单循环一份代码（AscendC 全内联下多份文本放大目标码）：
        //   i=0 仅装载（预载）；i∈[1,batch) 装载 i → 计算 i-1；i=batch 仅计算末批。
        // 计算游标须取装载翻转前值（l1LoadedPong）——i=batch 无装载不更新
        uint32_t l1ComputePong = l1pong_;
        uint32_t l1LoadedPong = l1pong_;

        WaitFlag<HardEvent::FIX_M>(l0cId);
        for (uint32_t loadIdx = 0; loadIdx <= shape.batch; loadIdx++) {
            if (loadIdx < shape.batch) {
                const uint32_t l1LoadPong = l1pong_;
                l1pong_ = !l1pong_;
                l1LoadedPong = l1LoadPong;
                {
                    LocalTensor<SrcT> bl1;
                    LocalTensor<SrcT> al1;
                    GetL1Buf(l1LoadPong != 0, bl1HalfBytes, bl1HalfElems, al1Elems, bl1, al1);
                    // 半区背压：等 2 段前同半区 MTE1 流排空
                    WaitFlag<HardEvent::MTE1_MTE2>(l1LoadPong);
                    LoadL1Fmap(bl1, config, cRange, cinAlign16, loadIdx);
                    LoadA1Dy(al1, config, cRange, loadIdx);
                    SetFlag<HardEvent::MTE2_MTE1>(l1LoadPong);
                }
            }
            if (loadIdx > 0) {
                LocalTensor<SrcT> bl1Compute;
                LocalTensor<SrcT> al1Compute;
                GetL1Buf(l1ComputePong != 0, bl1HalfBytes, bl1HalfElems, al1Elems, bl1Compute, al1Compute);
                WaitFlag<HardEvent::MTE2_MTE1>(l1ComputePong);
                IterateKL0(shape, loadIdx - 1, cinAlign16, coutAlign16, howoWin, al1Compute, bl1Compute, l0c);
                SetFlag<HardEvent::MTE1_MTE2>(l1ComputePong);
                l1ComputePong = l1LoadedPong;
            }
        }

        // 整块一次 fixpipe（M_FIX 即时对；FIX_M 释放跨块/End 收尾）
        SetFlag<HardEvent::M_FIX>(l0cId);
        WaitFlag<HardEvent::M_FIX>(l0cId);
        DirectOutL0C(l0c, y, yBase, alignedCout, cinAlign16, dhwk, cRange, shape.cin);
        SetFlag<HardEvent::FIX_M>(l0cId);
    }

private:
    // Fmatrix/padding 块首设置后，load3d 命令以此配置跳过每命令重设（减指令）
    static constexpr IsResetLoad3dConfig L3D_NO_RESET = {false, false};

    // L0A(dy) 双 buf 固定 id 6/7（低 id 段留给 L0B 池）
    __aicore__ inline uint8_t MTE1FlagDy(bool pingPong) const { return pingPong + 6; }

    // L0B(fmap) buf 池 id 0..bBufCnt_-1（与 L0A 的 6/7 不重叠）
    __aicore__ inline uint8_t MTE1FlagFmap(uint32_t bufIdx) const { return static_cast<uint8_t>(bufIdx); }

    // L1 两 bank 取段：bank 内 B1 段在前（+0）、A1 紧随（+bl1HalfBytes）
    __aicore__ inline void GetL1Buf(bool pong, uint32_t bl1HalfBytes, uint32_t bl1HalfElems, uint32_t al1Elems,
                                    LocalTensor<SrcT>& bl1, LocalTensor<SrcT>& al1) const
    {
        constexpr uint32_t l1BankBytes = TOTAL_L1_SIZE / 2; // L1 单 bank
        const uint32_t bankBytes = static_cast<uint32_t>(pong) * l1BankBytes;
        bl1 = LocalTensor<SrcT>(TPosition::A1, bankBytes, bl1HalfElems);
        al1 = LocalTensor<SrcT>(TPosition::A1, bankBytes + bl1HalfBytes, al1Elems);
    }

    // 单 batch 的 L0 搬运与 Mmad 循环（dout×howo 窗×dk 内层）：
    //   L0A 双 buf（a0Pong_ 窗尾翻转）；L0B buf 池（bBuf_ 装载后轮转 (bBuf_+1)%bBufCnt_，
    //   装载第 i+bufCnt 个 tile 前等第 i 个 Mmad 完成）；L0C 单视图 l0c 由调用方传入
    __aicore__ inline void IterateKL0(const ShapeAttribute& shape, uint32_t batchIdx, uint32_t cinAlign16,
                                      uint32_t coutAlign16, uint32_t kl0HoWoAlign16, const LocalTensor<SrcT>& al1,
                                      const LocalTensor<SrcT>& bl1, const LocalTensor<float>& l0c)
    {
        constexpr uint32_t l0aHalfElems = L0A_HALF_BYTES / sizeof(SrcT);

        const uint32_t howoTotal = shape.hout * shape.wout;

        for (uint32_t doutIdx = 0; doutIdx < shape.dout; doutIdx++) {
            for (uint32_t howoIdx = 0; howoIdx < howoTotal; howoIdx += kl0HoWoAlign16) {
                const uint32_t howoLen = Std::min(kl0HoWoAlign16, howoTotal - howoIdx);

                LocalTensor<SrcT> l0a(TPosition::A2, a0Pong_ * L0A_HALF_BYTES, l0aHalfElems);

                const uint8_t flagA = MTE1FlagDy(a0Pong_);
                WaitFlag<HardEvent::M_MTE1>(flagA);

                LoadL0Dy(shape, al1, l0a, coutAlign16, doutIdx, howoIdx, howoLen);

                SetFlag<HardEvent::MTE1_M>(flagA);
                WaitFlag<HardEvent::MTE1_M>(flagA);

                for (uint32_t dk = 0; dk < shape.dk; dk++) {
                    // 反算当前 dk 对应 fmap 的 dIn 索引，越界 = 该 dk 平面全 pad 跳过
                    const int32_t dIn = static_cast<int32_t>(doutIdx + dk) - static_cast<int32_t>(shape.dPad);
                    if (dIn < 0 || dIn >= static_cast<int32_t>(shape.din)) {
                        continue;
                    }

                    const uint32_t bBuf = bBuf_;
                    const uint8_t flagB = MTE1FlagFmap(bBuf);
                    // tensor size 取 L0B 剩余容量（≥ 实占；精确=用量的仿真器容量校验陷阱规避）
                    LocalTensor<SrcT> l0b(TPosition::B2, bBuf * bBufTileBytes_,
                                          (TOTAL_L0B_SIZE - bBuf * bBufTileBytes_) / sizeof(SrcT));

                    // 池深度背压：等第 i+bufCnt 次（回到本 buf）前第 i 次 Mmad 完成
                    WaitFlag<HardEvent::M_MTE1>(flagB);

                    LoadL0Fmap(shape, bl1, l0b, cinAlign16, static_cast<uint32_t>(dIn), howoIdx, howoLen);
                    bBuf_ = (bBuf_ + 1) % bBufCnt_;

                    SetFlag<HardEvent::MTE1_M>(flagB);
                    WaitFlag<HardEvent::MTE1_M>(flagB);

                    MmadParams mmad;
                    mmad.m = coutAlign16;
                    mmad.n = cinAlign16 * shape.hk * shape.wk;
                    mmad.k = howoLen;
                    const uint32_t firstDout = shape.dPad > dk ? shape.dPad - dk : 0;
                    mmad.cmatrixInitVal = (batchIdx == 0) && (howoIdx == 0) && (doutIdx == firstDout);
                    // L0C 段基址 = cinAlign16*coutAlign16*hkwk*dk 元素（dk 段 n 区不相交）；
                    // l0c 按 [dk,hwk,cin1,cout1,cout0,cin0] 排布（非常规 FZ）
                    Mmad(l0c[cinAlign16 * coutAlign16 * shape.hk * shape.wk * dk], l0a, l0b, mmad);
                    SetFlag<HardEvent::M_MTE1>(flagB);
                }
                SetFlag<HardEvent::M_MTE1>(flagA);
                a0Pong_ = !a0Pong_;
            }
        }
    }

    // dy GM → A1 单 batch 半区（每 batch 一条 Dn2Nz{dnNum=1}，半区承载 batch 维），
    // 布局见 S3
    __aicore__ inline void LoadA1Dy(const LocalTensor<SrcT>& al1, const DLoadConfig& config, const CoutCinRange& cRange,
                                    uint32_t batchIdx)
    {
        const ShapeAttribute& shape = config.shape;
        const uint32_t alignedCout = config.tiling.singleShapeAligned16Cout;
        const uint32_t dhowo = shape.dout * shape.hout * shape.wout;
        const uint32_t paddedDhowo = Ops::Base::CeilAlign<uint32_t>(dhowo, BLOCK_CUBE); // A1 行距
        Dn2NzParams dn2nz;
        dn2nz.dnNum = 1;
        dn2nz.dValue = cRange.coutLength; // 真实 co 段长
        dn2nz.nValue = dhowo;             // 真实行数，pad 行不写
        dn2nz.srcDnMatrixStride = dhowo;  // dnNum=1 不生效
        dn2nz.srcDValue = dhowo;          // co 行距
        dn2nz.dstNzC0Stride = paddedDhowo;
        dn2nz.dstNzNStride = 1;
        dn2nz.dstNzMatrixStride = paddedDhowo * alignedCout; // dnNum=1 不生效
        const uint64_t offset = static_cast<uint64_t>(batchIdx) * shape.cout * dhowo +
                                static_cast<uint64_t>(cRange.coutIdx) * dhowo;
        DataCopy(al1, dy_[offset], dn2nz);
    }

    // load2d 单命令转置整窗 [co 段][howo 窗]（参数式见 S5）；尾窗 garbage 不被消费
    __aicore__ inline void LoadL0Dy(const ShapeAttribute& shape, const LocalTensor<SrcT>& al1,
                                    const LocalTensor<SrcT>& l0a, uint32_t coutAlign16, uint32_t doutIdx,
                                    uint32_t howoIdx, uint32_t howoLen)
    {
        const uint32_t dhowo = shape.dout * shape.hout * shape.wout;
        LoadData2DParamsV2 load2d;
        load2d.ifTranspose = 1;
        load2d.mStartPosition = 0;
        load2d.kStartPosition = 0;
        load2d.mStep = Ops::Base::CeilDiv<uint32_t>(howoLen, BLOCK_CUBE);
        load2d.kStep = static_cast<uint16_t>(coutAlign16 / C0<SrcT>());
        load2d.srcStride = Ops::Base::CeilDiv<uint32_t>(dhowo, BLOCK_CUBE);
        // dstStride 半单位语义：kStep 恒偶（M 16 对齐 ⇒ kStep=M/8 偶），折半无损
        constexpr uint32_t LOAD2D_DST_STRIDE_DIVISOR = 2;
        load2d.dstStride = static_cast<uint16_t>(load2d.kStep / LOAD2D_DST_STRIDE_DIVISOR);
        const uint32_t srcOff = (doutIdx * shape.hout * shape.wout + howoIdx) * BpUtils::C0<SrcT>();
        LoadData(l0a, al1[srcOff], load2d);
    }

    // B1 → L0B（load3d）：每 tap 一条命令（hkwk 条/窗为下界——fixpipe 块序要求
    // [c1][tap][c0] 落位，repeat 等差游走无法跨 tap 跳步，单命令/窗不可行）；
    // rt=cinAlign16/C0 配对 (2g,2g+1) 落块 g；L0C 块序 B = cin16G*tap + g（消费端
    // fixpipe CS/SS 相应映射，见 DirectOutL0C）；通道视图与量纲见 S2/S4
    __aicore__ inline void LoadL0Fmap(const ShapeAttribute& shape, const LocalTensor<SrcT>& bl1,
                                      const LocalTensor<SrcT>& l0b, uint32_t cinAlign16, uint32_t dIn, uint32_t howoIdx,
                                      uint32_t howoLen)
    {
        const uint32_t hwIn = shape.hin * shape.win;
        const uint32_t hkwk = shape.hk * shape.wk;

        LoadData3DParamsV2<SrcT> load3d;
        load3d.l1H = shape.hin;
        load3d.l1W = shape.win;
        // pad 序 [left,right,top,bottom]：left/right=wPad、top/bottom=hPad
        const uint8_t padList[PAD_SIZE] = {static_cast<uint8_t>(shape.wPad), static_cast<uint8_t>(shape.wPad),
                                           static_cast<uint8_t>(shape.hPad), static_cast<uint8_t>(shape.hPad)};
        for (int32_t i = 0; i < PAD_SIZE; ++i) {
            load3d.padList[i] = padList[i];
        }
        load3d.channelSize = shape.din * cinAlign16; // 合轴通道视图 [c1g][d]
        load3d.kExtension = C0<SrcT>();              // 单 tap 窗（多 tap 的 k 轴序与 FZ 不兼容）
        load3d.kStartPt = 0;
        load3d.mStartPt = howoIdx;
        load3d.mExtension = howoLen; // 真实窗宽
        load3d.strideW = 1;
        load3d.strideH = 1;
        load3d.filterW = shape.wk;
        load3d.filterH = shape.hk;
        load3d.dilationFilterW = 1;
        load3d.dilationFilterH = 1;
        load3d.enTranspose = false;

        // rs=din*hkwk（repeat r 跨 c1g 进位）；rt=cinAlign16/C0（走满全视图）
        LoadDataRepeatParamWithStride rep;
        rep.repeatStride = shape.din * hkwk;
        rep.repeatTime = static_cast<uint8_t>(cinAlign16 / C0<SrcT>());
        rep.repeatMode = 1;
        rep.dstStride = static_cast<uint16_t>(hkwk * (cinAlign16 / BLOCK_CUBE));
        SetLoadDataRepeatWithStride(rep);

        // 源偏移 = dIn 平面基址（u32——S10；batch 维由半区基址承载）
        const uint32_t srcOff = dIn * hwIn * C0<SrcT>();
        const LocalTensor<SrcT> bl1T = bl1[srcOff];
        for (uint32_t tap = 0; tap < hkwk; tap++) {
            load3d.kStartPt = static_cast<uint16_t>(tap * BpUtils::C0<SrcT>());
            // tile 独占一个 L0B 半区，tap 块紧堆于起点；mExt 跨 8 行时 k-row1 自动落
            // 基址+8N（N = 段宽 16*cin16G*hkwk）——tile 实占 [半区, +16N)
            LoadDataWithStride<SrcT, L3D_NO_RESET>(l0b[C0<SrcT>() * cinAlign16 * tap], bl1T, load3d);
        }
    }

    // fmap GM → B1 单 batch 半区（与 LoadA1Dy 同构）：每 batch 一条
    // Dn2Nz{dnNum=1, dValue=cinLength, nValue=dhwin}
    __aicore__ inline void LoadL1Fmap(const LocalTensor<SrcT>& bl1, const DLoadConfig& config,
                                      const CoutCinRange& cRange, uint32_t cinAlign16, uint32_t batchIdx)
    {
        const ShapeAttribute& shape = config.shape;
        const uint64_t dhwin = static_cast<uint64_t>(shape.din) * shape.hin * shape.win;

        Dn2NzParams dn2nz;
        dn2nz.dnNum = 1;
        dn2nz.dValue = cRange.cinLength;                    // 真实段长
        dn2nz.nValue = static_cast<uint16_t>(dhwin);        // din×hin×win 合轴
        dn2nz.srcDnMatrixStride = dhwin;                    // dnNum=1 不生效
        dn2nz.srcDValue = dhwin;                            // cin 行距
        dn2nz.dstNzC0Stride = static_cast<uint16_t>(dhwin); // 组内行距
        dn2nz.dstNzNStride = 1;
        dn2nz.dstNzMatrixStride = static_cast<uint32_t>(dhwin * cinAlign16); // dnNum=1 不生效
        const uint64_t offset = static_cast<uint64_t>(batchIdx) * shape.cin * dhwin +
                                static_cast<uint64_t>(cRange.cinIdx) * dhwin;
        DataCopy(bl1, fmap_[offset], dn2nz);
    }

    // L0C → GM 直出（fixpipe NZ2DN，参数式见 S8）：dnNum/nSize/mSize 取真实段长，
    // 尾组 garbage 由 nSize 截断
    __aicore__ inline void DirectOutL0C(const LocalTensor<float>& l0c, GlobalTensor<float>& y, uint64_t yBase,
                                        uint32_t alignedCout, uint32_t cinAlign16, uint32_t dhwk,
                                        const CoutCinRange& cRange, uint32_t cinTotal)
    {
        FixpipeParamsArch3510<CO2Layout::COLUMN_MAJOR> fp;
        fp.params.dnNum = static_cast<uint16_t>(cRange.coutLength);
        fp.mSize = static_cast<uint16_t>(dhwk);
        fp.nSize = static_cast<uint16_t>(cRange.cinLength);
        fp.params.srcNzMatrixStride = 1; // C0 单位：co 步进 16 元素
        // 块序 B = cin16G*tap + g → CS = M*cin16G（tap 步进）、SS = M（g 步进）
        fp.params.srcNzC0Stride = static_cast<uint16_t>(alignedCout * (cinAlign16 / BLOCK_CUBE));
        fp.srcStride = static_cast<uint16_t>(alignedCout); // cin16 组步进（C0 单位）
        fp.dstStride = dhwk;                               // 元素：DN 行长
        fp.params.dstDnMatrixStride = cinTotal * dhwk;     // 元素：相邻 co 步进
        fp.quantPre = QuantMode_t::NoQuant;
        fp.unitFlag = 0;
        Fixpipe<float, float, CFG_COLUMN_MAJOR>(y[yBase], l0c, fp);
    }

    // 半区/池游标（跨块持续不重置——相邻块 buf 序错开使跨块背压自然衔接）：
    //   a0Pong_（(dout,win) 级 L0A）、bBuf_（(win,dk) 级 L0B 池轮转）、l1pong_（batch 级
    //   L1 归一）、l0cPong_（块级 L0C 半区，开门路径）
    bool l1pong_ = 0;
    bool a0Pong_ = 0;
    uint8_t bBufCnt_ = 0;
    uint8_t bBuf_ = 0;
    uint32_t bBufTileBytes_ = 0; // Init 一次计算（tiling 上界口径，跨块恒定）
    bool l0cPingpong_ = 0;       // Init 门判定一次（跨块恒定 → 两路径不混用）
    bool l0cPong_ = 0;
    GlobalTensor<SrcT> fmap_;
    GlobalTensor<SrcT> dy_;
};

} // namespace BpDLoad

#endif // CONV_BP_DLOAD_COMPUTE_H
