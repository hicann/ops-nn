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
 * \file block_mmad_a16w8_fixpipe_antiquant.h
 * \brief wqbmmv2 ASW 路径的 pingpong without que block mmad，原生内建 antiquant。
 *        A=fp16 x W=int8 混合 Mmad，int32 累加，Fixpipe 随路反量化；
 *        per-channel 用 scale 向量（L1 双缓冲），per-tensor 用 deqScalar 标量；bias 经 L1->BT 随 Mmad 加载。
 *        L1 布局（字节）：A x l1BufNum | B x l1BufNum | scale x 2 | bias x 2。
 */
#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "cmct/utils/arch.h"
#include "cmct/utils/common_utils.h"
#include "cmct/utils/quant_batch_matmul_constant.h"
#include "cmct/utils/tuple_utils.h"
#include "cmct/policy/dispatch_policy.h"

namespace Cmct::Gemm::Block {

// 数据类型 + 转置标记（ASW 仅 ND 格式，format 恒为 ND，不再携带）
template <class T_, bool IsTrans_>
struct WqbmmTensorType {
    using T = T_;
    static constexpr bool isTrans = IsTrans_;
};

// Fixpipe 随路量化/反量化模式映射：per-channel 用向量模式（V 前缀），per-tensor 用标量模式
template <bool PerChannel, typename C>
struct FixpipeQuantMode {
    static constexpr QuantMode_t value = QuantMode_t::NoQuant;
};
template <>
struct FixpipeQuantMode<true, int8_t> {
    static constexpr QuantMode_t value = QuantMode_t::VREQ8;
};
template <>
struct FixpipeQuantMode<true, half> {
    static constexpr QuantMode_t value = QuantMode_t::VDEQF16;
};
template <>
struct FixpipeQuantMode<true, bfloat16_t> {
    static constexpr QuantMode_t value = QuantMode_t::VQS322BF16_PRE;
};
template <>
struct FixpipeQuantMode<false, int8_t> {
    static constexpr QuantMode_t value = QuantMode_t::REQ8;
};
template <>
struct FixpipeQuantMode<false, half> {
    static constexpr QuantMode_t value = QuantMode_t::DEQF16;
};
template <>
struct FixpipeQuantMode<false, bfloat16_t> {
    static constexpr QuantMode_t value = QuantMode_t::QS322BF16_PRE;
};

template <class DispatchPolicy_, class L1TileShape_, class L0TileShape_, class AType_, class BType_, class CType_,
          class BiasType_>
class WqbmmBlockMmad {
public:
    using DispatchPolicy = DispatchPolicy_;
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BiasType = BiasType_;
    using A_T = typename AType::T;
    using B_T = typename BType::T;
    using C_T = typename CType::T;
    using Bias_T = typename BiasType::T;
    using L0cType = int32_t; // A16W8 混合 Mmad 定点累加
    static constexpr bool transA = AType::isTrans;
    static constexpr bool transB = BType::isTrans;
    static constexpr WqbmmAntiQuantMode antiQuantMode = DispatchPolicy::antiQuantMode;
    static constexpr bool perChannelScale = (antiQuantMode == WqbmmAntiQuantMode::PER_CHANNEL);

    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using TupleL1L0Shape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;

    // L0/L0C 双缓冲容量（元素数）
    constexpr static uint64_t HALF_L0A_ELEMS = L0A_SIZE / DOUBLE_BUFFER_COUNT / sizeof(A_T);
    constexpr static uint64_t HALF_L0B_ELEMS = L0B_SIZE / DOUBLE_BUFFER_COUNT / sizeof(B_T);
    constexpr static uint64_t HALF_L0C_ELEMS = AscendC::TOTAL_L0C_SIZE / DOUBLE_BUFFER_COUNT / sizeof(L0cType);
    constexpr static int32_t C0_SIZE_A = AscendC::AuxGetC0Size<A_T>();
    constexpr static int32_t C0_SIZE_W = AscendC::AuxGetC0Size<B_T>();
    constexpr static int32_t C0_SIZE_BIAS = AscendC::AuxGetC0Size<Bias_T>();
    // 硬事件 flag id
    constexpr static uint16_t L1_BUF_FLAG_BASE = 0;    // MTE1_MTE2 / MTE2_MTE1: 0~3 为 A/B L1 buffer
    constexpr static uint16_t BIAS_L1_FLAG_BASE = 4;   // MTE1_MTE2: 4/5 为 bias L1 双缓冲
    constexpr static uint16_t L0_M_MTE1_FLAG = 6;      // M_MTE1: 6/7 为 L0 双缓冲
    constexpr static uint16_t L0_MTE1_M_FLAG = 6;      // MTE1_M: 6/7 为 L0 双缓冲
    constexpr static uint16_t SCALE_FIX_MTE2_FLAG = 0; // FIX_MTE2 / MTE2_FIX: 0/1 为 scale L1 双缓冲
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
    constexpr static uint8_t FIX_SHIFT_VAL_LEN_A16W8 = 29;
#endif

    __aicore__ inline WqbmmBlockMmad()
    {
        if ASCEND_IS_NOT_AIV {
            // 预设"资源空闲"flag，使首个 WaitFlag 直接通过
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1_BUF_FLAG_BASE);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1_BUF_FLAG_BASE + 1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1_BUF_FLAG_BASE + 2); // 2: 第3个L1 buffer
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1_BUF_FLAG_BASE + 3); // 3: 第4个L1 buffer
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(BIAS_L1_FLAG_BASE);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(BIAS_L1_FLAG_BASE + 1);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0_M_MTE1_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0_M_MTE1_FLAG + 1);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);
            AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(SCALE_FIX_MTE2_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(SCALE_FIX_MTE2_FLAG + 1);
        }
    }

    __aicore__ inline ~WqbmmBlockMmad()
    {
        if ASCEND_IS_NOT_AIV {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1_BUF_FLAG_BASE);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1_BUF_FLAG_BASE + 1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1_BUF_FLAG_BASE + 2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1_BUF_FLAG_BASE + 3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(BIAS_L1_FLAG_BASE);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(BIAS_L1_FLAG_BASE + 1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0_M_MTE1_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0_M_MTE1_FLAG + 1);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(SCALE_FIX_MTE2_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(SCALE_FIX_MTE2_FLAG + 1);
        }
    }

public:
    __aicore__ inline void Init(const TupleShape& shape, const TupleShape& tileL1, const TupleShape& tileL0,
                                bool isBias, uint64_t l1BufNum, bool l0cDB, uint8_t shiftValue)
    {
        m_ = static_cast<uint64_t>(Get<MNK_M>(shape));
        n_ = static_cast<uint64_t>(Get<MNK_N>(shape));
        k_ = static_cast<uint64_t>(Get<MNK_K>(shape));
        mL1_ = static_cast<uint64_t>(Get<MNK_M>(tileL1));
        nL1_ = static_cast<uint64_t>(Get<MNK_N>(tileL1));
        kL1_ = static_cast<uint64_t>(Get<MNK_K>(tileL1));
        baseM_ = static_cast<uint64_t>(Get<MNK_M>(tileL0));
        baseN_ = static_cast<uint64_t>(Get<MNK_N>(tileL0));
        baseK_ = static_cast<uint64_t>(Get<MNK_K>(tileL0));
        isBias_ = isBias;
        l1BufNum_ = l1BufNum;
        enableL0cPingPong_ = l0cDB;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
        shiftValue_ = shiftValue;
#endif
        kL1Iter_ = CeilDiv(k_, kL1_);
        // L1 静态布局（字节）：A x l1BufNum | B x l1BufNum | scale x 2 | bias x 2
        aL1Bytes_ = mL1_ * kL1_ * sizeof(A_T);
        // NZ 格式下 C0 内轴须按 C0 对齐分配：transB 时 B 为 (N, K)（K 为内轴），否则为 (K, N)（N 为内轴）
        bL1Bytes_ = transB ? nL1_ * Align(kL1_, static_cast<uint64_t>(C0_SIZE_W)) * sizeof(B_T) :
                             Align(nL1_, static_cast<uint64_t>(C0_SIZE_W)) * kL1_ * sizeof(B_T);
        bL1Base_ = aL1Bytes_ * l1BufNum_;
        scaleL1BufBytes_ = perChannelScale ? nL1_ * sizeof(uint64_t) : 0;
        scaleL1Base_ = bL1Base_ + bL1Bytes_ * l1BufNum_;
        biasL1BufBytes_ = isBias_ ? nL1_ * sizeof(Bias_T) : 0;
        biasL1Base_ = scaleL1Base_ + scaleL1BufBytes_ * DOUBLE_BUFFER_COUNT;
    }

    __aicore__ inline void CacheQuantScalar(uint64_t quantScalar) { quantScalar_ = quantScalar; }

    // 每个 L1 tile 调用一次：K 方向循环累加后 Fixpipe 搬出
    __aicore__ inline void operator()(const AscendC::GlobalTensor<C_T>& cGlobal,
                                      const AscendC::GlobalTensor<A_T>& aGlobal,
                                      const AscendC::GlobalTensor<B_T>& bGlobal,
                                      const AscendC::GlobalTensor<Bias_T>& biasGlobal,
                                      const AscendC::GlobalTensor<uint64_t>& scaleGlobal,
                                      const TupleL1L0Shape& tileShape, bool isFirstTile)
    {
        uint64_t curML1 = static_cast<uint64_t>(Get<MNK_M>(tileShape));
        uint64_t curNL1 = static_cast<uint64_t>(Get<MNK_N>(tileShape));
        uint64_t curML0 = static_cast<uint64_t>(Get<MNK_M0>(tileShape));
        uint64_t curNL0 = static_cast<uint64_t>(Get<MNK_N0>(tileShape));
        // 等待上一轮该半区 L0C 的 fixpipe 完成，L0C 可复用
        uint64_t l0cOffset = (l0cPingPong_ & 0x1) * HALF_L0C_ELEMS;
        uint16_t l0cFlag = enableL0cPingPong_ ? static_cast<uint16_t>(l0cPingPong_ & 0x1) : 0;
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cFlag);
        // per-channel scale：随 tile 的 nOffset 滚动加载（GM 地址已带 nOffset），双缓冲
        uint16_t scaleBufId = scaleLoopCnt_ & 0x1;
        if constexpr (perChannelScale) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(SCALE_FIX_MTE2_FLAG + scaleBufId);
            CopyScaleL1(scaleGlobal, curNL1, scaleBufId);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_FIX>(SCALE_FIX_MTE2_FLAG + scaleBufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_FIX>(SCALE_FIX_MTE2_FLAG + scaleBufId);
        }
        // bias：每 tile 加载一次到 L1，双缓冲
        uint16_t biasBufId = biasLoopCnt_ & 0x1;
        if (isBias_) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(BIAS_L1_FLAG_BASE + biasBufId);
            CopyBiasL1(biasGlobal, curNL1, biasBufId);
        }

        uint64_t kL1 = Min(k_, kL1_);
        uint64_t curKL1Iter = kL1Iter_;
        // 若 stepK>=2，首个 tile 前两轮将搬运量减半，提前启动 mmad
        bool isFirstLoopKL1Half = false;
        if (isFirstTile && kL1 / baseK_ >= DOUBLE_BUFFER_COUNT) {
            isFirstLoopKL1Half = true;
            curKL1Iter++;
        }
        uint64_t kL1OffsetLength = 0;
        for (uint64_t iter0 = 0; iter0 < curKL1Iter; ++iter0) {
            uint64_t curKL1 = (iter0 + 1 == curKL1Iter) ? (k_ - kL1OffsetLength) : kL1;
            if (isFirstLoopKL1Half) {
                if (iter0 == 0) {
                    // 半载粒度须对齐到 baseK_，保证切分后的 curK0 仍是 baseK_ 整数倍
                    // （int8 权重时 mmad 的 K 粒度为 32，16 对齐会切出非法 K 块）
                    curKL1 = CeilAlign(kL1 / DOUBLE_BUFFER_COUNT, baseK_);
                } else if (iter0 == 1) {
                    curKL1 = kL1 - kL1OffsetLength;
                }
            }
            uint64_t l1BufId = abL1LoopCnt_ & (l1BufNum_ - 1);
            uint64_t aL1ByteOffset = aL1Bytes_ * l1BufId;
            uint64_t bL1ByteOffset = bL1Base_ + bL1Bytes_ * l1BufId;
            // GM -> L1（Nd2Nz），l1BufNum 级 buffer
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1_BUF_FLAG_BASE + l1BufId);
            uint64_t offsetA = transA ? kL1OffsetLength * m_ : kL1OffsetLength;
            uint64_t offsetB = transB ? kL1OffsetLength : kL1OffsetLength * n_;
            CopyInA1(aGlobal[offsetA], aL1ByteOffset, curML1, curKL1);
            CopyInB1(bGlobal[offsetB], bL1ByteOffset, curNL1, curKL1);
            // 同 pipe 顺序保证 bias L1 加载也在此 flag 之前完成
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(L1_BUF_FLAG_BASE + l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(L1_BUF_FLAG_BASE + l1BufId);
            // L1 -> L0 -> Mmad，L0 双缓冲
            uint64_t kL0Iter = CeilDiv(curKL1, baseK_);
            uint64_t kL1Offset = 0;
            for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                uint64_t curK0 = (iter1 + 1 == kL0Iter) ? (curKL1 - kL1Offset) : baseK_;
                uint64_t l0Parity = l0PingPong_ & 0x1;
                uint16_t mte1Flag = L0_M_MTE1_FLAG + l0Parity;
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mte1Flag);
                CopyInA2(l0aLocal_[l0Parity * HALF_L0A_ELEMS], aL1ByteOffset, kL1Offset, curML1, curKL1, curML0, curK0);
                bool needBias = isBias_ && iter0 == 0 && iter1 == 0;
                if (needBias) {
                    CopyBiasBt(biasBufId, Align(curNL0, static_cast<uint64_t>(AscendC::BLOCK_CUBE)));
                    // bias 在 L1 上的空间随 MTE1 读完成即可释放
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(BIAS_L1_FLAG_BASE + biasBufId);
                }
                CopyInB2(l0bLocal_[l0Parity * HALF_L0B_ELEMS], bL1ByteOffset, kL1Offset, curNL1, curKL1, curNL0, curK0);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(L0_MTE1_M_FLAG + l0Parity);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0_MTE1_M_FLAG + l0Parity);
                AscendC::MmadParams mmadParams;
                mmadParams.m = curML0;
                mmadParams.n = curNL0;
                mmadParams.k = curK0;
                mmadParams.disableGemv = true;
                mmadParams.unitFlag = 0; // 不使用 unitflag，通过 M_FIX 显式同步
                mmadParams.cmatrixInitVal = (iter0 == 0 && iter1 == 0 && !isBias_);
                mmadParams.cmatrixSource = needBias;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
                mmadParams.fixShiftVal = shiftValue_;
#endif
                MmadCompute(mmadParams, l0cOffset, l0Parity, biasBufId, needBias);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(mte1Flag);
                l0PingPong_++;
                kL1Offset += curK0;
            }
            // L1 buffer 数据已全部读入 L0，释放给下一轮 MTE2
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1_BUF_FLAG_BASE + l1BufId);
            abL1LoopCnt_++;
            kL1OffsetLength += curKL1;
        }
        // 等待全部 Mmad 完成
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0cFlag);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0cFlag);
        // Fixpipe L0C -> GM，随路反量化
        CopyOut(cGlobal, l0cOffset, curML0, curNL0, scaleBufId);
        if constexpr (perChannelScale) {
            // scale 已被 fixpipe 读取，释放 L1 空间
            AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(SCALE_FIX_MTE2_FLAG + scaleBufId);
            scaleLoopCnt_++;
        }
        // 标记该半区 L0C 空闲
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0cFlag);
        if (enableL0cPingPong_) {
            l0cPingPong_++;
        }
        if (isBias_) {
            biasLoopCnt_++;
        }
    }

private:
    __aicore__ inline void CopyInA1(const AscendC::GlobalTensor<A_T>& aGlobal, uint64_t aL1ByteOffset, uint64_t curML1,
                                    uint64_t curKL1)
    {
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        uint64_t nDim = transA ? curKL1 : curML1;
        uint64_t dDim = transA ? curML1 : curKL1;
        nd2nzParams.nValue = nDim;
        nd2nzParams.dValue = dDim;
        nd2nzParams.srcNdMatrixStride = 1;
        nd2nzParams.srcDValue = transA ? m_ : k_;
        nd2nzParams.dstNzC0Stride = Align(nDim, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 1;
        AscendC::DataCopy(aL1Local_[aL1ByteOffset / sizeof(A_T)], aGlobal, nd2nzParams);
    }

    __aicore__ inline void CopyInB1(const AscendC::GlobalTensor<B_T>& bGlobal, uint64_t bL1ByteOffset, uint64_t curNL1,
                                    uint64_t curKL1)
    {
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        uint64_t nDim = transB ? curNL1 : curKL1;
        uint64_t dDim = transB ? curKL1 : curNL1;
        nd2nzParams.nValue = nDim;
        nd2nzParams.dValue = dDim;
        nd2nzParams.srcNdMatrixStride = 1;
        nd2nzParams.srcDValue = transB ? k_ : n_;
        nd2nzParams.dstNzC0Stride = Align(nDim, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 1;
        AscendC::DataCopy(bL1Local_[bL1ByteOffset / sizeof(B_T)], bGlobal, nd2nzParams);
    }

    __aicore__ inline void CopyScaleL1(const AscendC::GlobalTensor<uint64_t>& scaleGlobal, uint64_t curNL1,
                                       uint16_t scaleBufId)
    {
        AscendC::DataCopyPadParams padParams;
        AscendC::DataCopyParams scaleParam{1, static_cast<uint16_t>(curNL1 * sizeof(uint64_t)), 0, 0};
        uint64_t scaleL1ByteOffset = scaleL1Base_ + scaleBufId * scaleL1BufBytes_;
        AscendC::DataCopyPad(scaleL1Local_[scaleL1ByteOffset / sizeof(uint64_t)], scaleGlobal, scaleParam, padParams);
    }

    __aicore__ inline void CopyBiasL1(const AscendC::GlobalTensor<Bias_T>& biasGlobal, uint64_t curNL1,
                                      uint16_t biasBufId)
    {
        AscendC::DataCopyPadParams padParams;
        // blockLen 单位为 Byte
        AscendC::DataCopyParams biasParam{1, static_cast<uint16_t>(curNL1 * sizeof(Bias_T)), 0, 0};
        uint64_t biasL1ByteOffset = biasL1Base_ + biasBufId * biasL1BufBytes_;
        AscendC::DataCopyPad(biasL1Local_[biasL1ByteOffset / sizeof(Bias_T)], biasGlobal, biasParam, padParams);
    }

    __aicore__ inline void CopyInA2(const AscendC::LocalTensor<A_T>& a2Local, uint64_t aL1ByteOffset,
                                    uint64_t kL1Offset, uint64_t curML1, uint64_t curKL1, uint64_t curML0,
                                    uint64_t curK0)
    {
        uint64_t mL1Align = Align(curML1, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        uint64_t aL1Offset = aL1ByteOffset / sizeof(A_T);
        if constexpr (!transA) {
            // (M, K)
            loadDataParams.mStep = CeilDiv(curML0, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
            loadDataParams.kStep = CeilDiv(curK0, static_cast<uint64_t>(C0_SIZE_A));
            loadDataParams.srcStride = CeilDiv(curML1, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
            loadDataParams.dstStride = loadDataParams.mStep;
            loadDataParams.ifTranspose = false;
            aL1Offset += kL1Offset * mL1Align;
        } else {
            // (K, M)
            loadDataParams.mStep = CeilDiv(curK0, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
            loadDataParams.kStep = CeilDiv(curML0, static_cast<uint64_t>(C0_SIZE_A));
            loadDataParams.srcStride = CeilDiv(curKL1, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
            loadDataParams.dstStride = loadDataParams.kStep;
            loadDataParams.ifTranspose = true;
            aL1Offset += kL1Offset * C0_SIZE_A;
        }
        AscendC::LoadData<A_T>(a2Local, aL1Local_[aL1Offset], loadDataParams);
    }

    __aicore__ inline void CopyInB2(const AscendC::LocalTensor<B_T>& b2Local, uint64_t bL1ByteOffset,
                                    uint64_t kL1Offset, uint64_t curNL1, uint64_t curKL1, uint64_t curNL0,
                                    uint64_t curK0)
    {
        uint64_t nL1Align = Align(curNL1, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        uint64_t bL1Offset = bL1ByteOffset / sizeof(B_T);
        if constexpr (transB) {
            // (N, K)
            loadDataParams.mStep = CeilDiv(curNL0, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
            loadDataParams.kStep = CeilDiv(curK0, static_cast<uint64_t>(C0_SIZE_W));
            loadDataParams.srcStride = CeilDiv(curNL1, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
            loadDataParams.dstStride = loadDataParams.mStep;
            loadDataParams.ifTranspose = false;
            bL1Offset += kL1Offset * nL1Align;
            AscendC::LoadData<B_T>(b2Local, bL1Local_[bL1Offset], loadDataParams);
        } else {
            // (K, N)
            loadDataParams.kStep = CeilDiv(curNL0, static_cast<uint64_t>(C0_SIZE_W));
            loadDataParams.srcStride = CeilDiv(curKL1, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
            loadDataParams.dstStride = CeilDiv(curNL0, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
            loadDataParams.ifTranspose = true;
            uint16_t fullMStep = static_cast<uint16_t>(CeilDiv(curK0, static_cast<uint64_t>(AscendC::BLOCK_CUBE)));
            if constexpr (AscendC::IsSameType<B_T, int8_t>::value || AscendC::IsSameType<B_T, uint8_t>::value) {
                // int8 转置加载 mStep 最小为 2，按 2 个分型为步长循环加载
                constexpr uint16_t M_STEP_MIN_VAL_B8 = 2;
                uint16_t l0BLoop = CeilDiv(static_cast<uint64_t>(fullMStep), static_cast<uint64_t>(M_STEP_MIN_VAL_B8));
                loadDataParams.mStep = M_STEP_MIN_VAL_B8;
                uint64_t dstOffset = 0;
                uint64_t dstAddrStride = Align(curNL0, static_cast<uint64_t>(AscendC::BLOCK_CUBE)) * BLOCK_BYTE_SIZE;
                uint16_t oriMstartPos = static_cast<uint16_t>(
                    CeilDiv(kL1Offset, static_cast<uint64_t>(AscendC::BLOCK_CUBE)));
                for (uint16_t idx = 0; idx < l0BLoop; ++idx) {
                    loadDataParams.mStartPosition = oriMstartPos + M_STEP_MIN_VAL_B8 * idx;
                    AscendC::LoadData<B_T>(b2Local[dstOffset], bL1Local_[bL1Offset], loadDataParams);
                    dstOffset += dstAddrStride;
                }
            } else {
                loadDataParams.mStep = fullMStep;
                bL1Offset += kL1Offset * C0_SIZE_W;
                AscendC::LoadData<B_T>(b2Local, bL1Local_[bL1Offset], loadDataParams);
            }
        }
    }

    __aicore__ inline void CopyBiasBt(uint16_t biasBufId, uint64_t nL0Align)
    {
        // s32 场景要对齐到 2，因此是 align(n / biasC0, 16 / biasC0)
        constexpr uint64_t btAlign = static_cast<uint64_t>(AscendC::BLOCK_CUBE) / C0_SIZE_BIAS;
        uint16_t burstLen = Align(nL0Align / C0_SIZE_BIAS, btAlign);
        AscendC::DataCopyParams biasParam{1, burstLen, 0, 0};
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
        biasParam.fixShiftVal = FIX_SHIFT_VAL_LEN_A16W8 - shiftValue_;
#endif
        uint64_t biasL1ByteOffset = biasL1Base_ + biasBufId * biasL1BufBytes_;
        AscendC::DataCopy(biasBtLocal_[baseN_ * biasBufId], biasL1Local_[biasL1ByteOffset / sizeof(Bias_T)], biasParam);
    }

    __aicore__ inline void MmadCompute(const AscendC::MmadParams& mmadParams, uint64_t l0cOffset, uint64_t l0Parity,
                                       uint16_t biasBufId, bool needBias)
    {
        if (needBias) {
            AscendC::Mmad(cL0Local_[l0cOffset], l0aLocal_[l0Parity * HALF_L0A_ELEMS],
                          l0bLocal_[l0Parity * HALF_L0B_ELEMS], biasBtLocal_[baseN_ * biasBufId], mmadParams);
        } else {
            AscendC::Mmad(cL0Local_[l0cOffset], l0aLocal_[l0Parity * HALF_L0A_ELEMS],
                          l0bLocal_[l0Parity * HALF_L0B_ELEMS], mmadParams);
        }
    }

    __aicore__ inline void CopyOut(const AscendC::GlobalTensor<C_T>& cGlobal, uint64_t l0cOffset, uint64_t curML0,
                                   uint64_t curNL0, uint16_t scaleBufId)
    {
        AscendC::FixpipeParamsC310 fixpipeParams;
        fixpipeParams.nSize = curNL0;
        fixpipeParams.mSize = curML0;
        fixpipeParams.dstStride = n_;
        fixpipeParams.srcStride = Align(curML0, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
        fixpipeParams.quantPre = FixpipeQuantMode<perChannelScale, C_T>::value;
        if constexpr (!perChannelScale) {
            fixpipeParams.deqScalar = quantScalar_;
        }
        fixpipeParams.unitFlag = 0;
        fixpipeParams.params = {1, 1, 1};
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
        if constexpr (AscendC::IsSameType<A_T, half>::value && AscendC::IsSameType<B_T, half>::value) {
            fixpipeParams.fixShiftVal = FIX_SHIFT_VAL_LEN_A16W8 - shiftValue_;
        }
#endif
        if constexpr (perChannelScale) {
            uint64_t scaleL1ByteOffset = scaleL1Base_ + scaleBufId * scaleL1BufBytes_;
            AscendC::Fixpipe(cGlobal, cL0Local_[l0cOffset], scaleL1Local_[scaleL1ByteOffset / sizeof(uint64_t)],
                             fixpipeParams);
        } else {
            AscendC::Fixpipe(cGlobal, cL0Local_[l0cOffset], fixpipeParams);
        }
    }

private:
    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};
    uint64_t mL1_{1};
    uint64_t nL1_{1};
    uint64_t kL1_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};
    uint64_t kL1Iter_{0};
    uint64_t l1BufNum_{1};
    bool isBias_{false};
    bool enableL0cPingPong_{false};
    uint64_t quantScalar_{0};
    uint64_t aL1Bytes_{0};
    uint64_t bL1Bytes_{0};
    uint64_t bL1Base_{0};
    uint64_t scaleL1BufBytes_{0};
    uint64_t scaleL1Base_{0};
    uint64_t biasL1BufBytes_{0};
    uint64_t biasL1Base_{0};
    uint64_t abL1LoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    uint64_t scaleLoopCnt_{0};
    uint64_t biasLoopCnt_{0};
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
    uint8_t shiftValue_{13}; // 与 host 侧默认 shiftValue 一致
#endif

    // 静态偏移视图，无 allocator；L1 上 A/B/scale/bias 共用 A1 空间，按字节偏移索引
    AscendC::LocalTensor<A_T> aL1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE / sizeof(A_T)};
    AscendC::LocalTensor<B_T> bL1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE / sizeof(B_T)};
    AscendC::LocalTensor<uint64_t> scaleL1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE / sizeof(uint64_t)};
    AscendC::LocalTensor<Bias_T> biasL1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE / sizeof(Bias_T)};
    AscendC::LocalTensor<A_T> l0aLocal_{AscendC::TPosition::A2, 0, L0A_SIZE / sizeof(A_T)};
    AscendC::LocalTensor<B_T> l0bLocal_{AscendC::TPosition::B2, 0, L0B_SIZE / sizeof(B_T)};
    AscendC::LocalTensor<L0cType> cL0Local_{AscendC::TPosition::CO1, 0, AscendC::TOTAL_L0C_SIZE / sizeof(L0cType)};
    AscendC::LocalTensor<Bias_T> biasBtLocal_{AscendC::TPosition::C2, 0, QuantBatchMatmul::BT_SIZE / sizeof(Bias_T)};
};

} // namespace Cmct::Gemm::Block
