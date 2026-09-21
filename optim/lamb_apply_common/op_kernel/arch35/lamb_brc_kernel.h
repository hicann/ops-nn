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
 * \file lamb_brc_kernel.h
 * \brief LAMB 族「多入多出 + 任意 numpy 广播」逐元素算子的共用搬运/分核/广播骨架
 *
 * 为什么不走 ATVOSS: 本族算子入参 12~13 个, DAGSch 在
 * (mte2Count + mte3Count) * BUF_PING_PONG + tempBufCount <= 32 上编不过;
 * 且 ATVOSS 的 Vec::Brc 未实现, UB 广播档不可用, 只剩 NDDMA 档, 铺不平
 * 「尾轴 1->n」与「低秩补维」两类广播。
 *
 * 本骨架:
 *   - 广播在搬入阶段铺平, 计算段只看逐元素数据; 铺平只用平台已验证的 2D
 *     AscendC::Broadcast 由内向外迭代展开, 因此支持任意 numpy 广播形态。
 *   - 分片长度/行块轴由 host 从 ubSize 反解下发, 内核不做"算完再跟 UB 比大小"的判断,
 *     也没有经验阈值; 行块轴退化到最内层必然成立, 故不存在因尺寸拒收的分支。
 *   - 计算段由算子自己以 Compute::Run(in[], out[], count) 提供。
 */

#ifndef LAMB_BRC_KERNEL_H
#define LAMB_BRC_KERNEL_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "lamb_brc_tiling_data.h"

namespace LambBrc {
using namespace AscendC;

static constexpr Reg::CastTrait LAMB_CAST_UP = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
static constexpr Reg::CastTrait LAMB_CAST_DOWN = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                  Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

constexpr uint32_t LAMB_VF_LEN = Ops::Base::GetVRegSize() / sizeof(float);

// host 侧无法调用 __aicore__ 的平台接口, 只能在共用头里写常量; 这里回连校验, 防止两侧走偏。
static_assert(Ops::Base::GetUbBlockSize() == LAMB_BRC_GM_BLOCK_BYTES, "UB block size mismatch");
static_assert(Ops::Base::GetVRegSize() == LAMB_BRC_VREG_BYTES, "vector register size mismatch");

// 精度档: 默认的 DivAlgo/SqrtAlgo::INTRINSIC 是硬件近似实现, 且按 FTZ(非规格数归零)处理 ——
// 非规格中间量被冲刷成 ±0 后, sqrt(-tiny) 应得的 NaN 会退化成 sqrt(-0)=-0, 再除即成 inf,
// 与 CPU 参照定性不符。取仓内通行的 0ULP + FTZ_FALSE 档(见 index/scatter_reduce_common、
// loss/soft_margin_loss_grad 等), 由库完成非规格数的完整处理。
static constexpr AscendC::Reg::DivSpecificMode LAMB_PRECISE_DIV = {AscendC::Reg::MaskMergeMode::ZEROING, false,
                                                                   AscendC::DivAlgo::PRECISION_0ULP_FTZ_FALSE};
static constexpr AscendC::Reg::SqrtSpecificMode LAMB_PRECISE_SQRT = {AscendC::Reg::MaskMergeMode::ZEROING, false,
                                                                     AscendC::SqrtAlgo::PRECISION_0ULP_FTZ_FALSE};
static constexpr AscendC::Reg::ExpSpecificMode LAMB_PRECISE_EXP = {AscendC::Reg::MaskMergeMode::ZEROING,
                                                                   AscendC::ExpAlgo::PRECISION_1ULP_FTZ_FALSE};
static constexpr AscendC::Reg::LnSpecificMode LAMB_PRECISE_LN = {AscendC::Reg::MaskMergeMode::ZEROING,
                                                                 AscendC::LnAlgo::PRECISION_1ULP_FTZ_FALSE};
// FTZ_TRUE 档: exp 的下溢结果按冲刷到 ±0 处理。ln(b)*steps 小于 fp32 exp 的下溢边界时,
// b^steps 的数学真值本就小于最小非规格数, 冲刷到 0 即是正确结果(b_corr=1)。
static constexpr AscendC::Reg::ExpSpecificMode LAMB_PRECISE_EXP_FTZT = {AscendC::Reg::MaskMergeMode::ZEROING,
                                                                        AscendC::ExpAlgo::PRECISION_1ULP_FTZ_TRUE};
static constexpr AscendC::Reg::LnSpecificMode LAMB_PRECISE_LN_FTZT = {AscendC::Reg::MaskMergeMode::ZEROING,
                                                                      AscendC::LnAlgo::PRECISION_1ULP_FTZ_TRUE};

// corr = 1 - exp(x)。x -> 0 时 exp(x) -> 1, 直写 1-exp(x) 是灾难性抵消(fp32 在 1.0 附近
// 间距 6e-8), 与 exp 实现多准无关。故 |x| < 0.1 走 5 项 Horner 多项式直算 expm1(x) 绕开该减法,
// 否则直算 exp(x)-1。同式同阈值见 activation/elu。
__aicore__ inline void OneMinusExp(Reg::RegTensor<float>& dst, Reg::RegTensor<float>& x, Reg::RegTensor<float>& t1,
                                   Reg::RegTensor<float>& t2, Reg::MaskReg& cmp, Reg::MaskReg& preg)
{
    Reg::Muls(t1, x, 1.0f / 120.0f, preg);
    Reg::Adds(t1, t1, 1.0f / 24.0f, preg);
    Reg::Mul(t1, t1, x, preg);
    Reg::Adds(t1, t1, 1.0f / 6.0f, preg);
    Reg::Mul(t1, t1, x, preg);
    Reg::Adds(t1, t1, 0.5f, preg);
    Reg::Mul(t1, t1, x, preg);
    Reg::Adds(t1, t1, 1.0f, preg);
    Reg::Mul(t1, t1, x, preg); // t1 = expm1(x), 多项式支
    Reg::Exp<float, &LAMB_PRECISE_EXP_FTZT>(t2, x, preg);
    Reg::Adds(t2, t2, -1.0f, preg); // t2 = expm1(x), exp 支
    Reg::Abs(dst, x, preg);         // x 之后不再用, dst 可与 x 同寄存器
    Reg::Compares<float, CMPMODE::LT>(cmp, dst, 0.1f, preg);
    Reg::Select<float>(dst, t1, t2, cmp);
    Reg::Muls(dst, dst, -1.0f, preg); // 1 - exp(x) = -expm1(x)
}

template <typename T>
__aicore__ inline void Load(__local_mem__ T* base, Reg::RegTensor<float>& dst, Reg::MaskReg& preg, uint32_t offset)
{
    if constexpr (std::is_same_v<T, float>) {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(dst, base + offset);
    } else {
        Reg::RegTensor<T> narrow;
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(narrow, base + offset);
        Reg::Cast<float, T, LAMB_CAST_UP>(dst, narrow, preg);
    }
}

template <typename T>
__aicore__ inline void Store(__local_mem__ T* base, Reg::RegTensor<float>& src, Reg::MaskReg& preg, uint32_t offset)
{
    if constexpr (std::is_same_v<T, float>) {
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(base + offset, src, preg);
    } else {
        Reg::RegTensor<T> narrow;
        Reg::Cast<T, float, LAMB_CAST_DOWN>(narrow, src, preg);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(base + offset, narrow, preg);
    }
}

template <typename T, uint32_t NIN, uint32_t NOUT, class Compute>
class BrcElementwiseKernel {
public:
    using Tiling = LambBrcTilingData<NIN, NOUT>;
    static constexpr uint32_t SLOT_NUM = NIN + NOUT + 1;
    static constexpr uint32_t TMP_SLOT = SLOT_NUM - 1;

    __aicore__ inline BrcElementwiseKernel() {}

    __aicore__ inline void Init(GM_ADDR (&inAddr)[NIN], GM_ADDR (&outAddr)[NOUT], const Tiling* tiling, TPipe* pipe)
    {
        tiling_ = tiling;
        blockIdx_ = GetBlockIdx();
        for (uint32_t i = 0; i < NIN; i++) {
            inGm_[i].SetGlobalBuffer((__gm__ T*)inAddr[i]);
        }
        for (uint32_t o = 0; o < NOUT; o++) {
            outGm_[o].SetGlobalBuffer((__gm__ T*)outAddr[o]);
        }
        // 槽位 + fp32 暂存区(4*tileLen): 计算段过长时可拆成两个 __VEC_SCOPE__,
        // 中间量用 fp32 暂存传递, 与单段版数值完全一致(不经 T 的舍入)。
        pipe->InitBuffer(ubBuf_, SLOT_NUM * tiling_->tileLen * sizeof(T) + 4U * tiling_->tileLen * sizeof(float));
        LocalTensor<T> all = ubBuf_.Get<T>();
        for (uint32_t s = 0; s < SLOT_NUM; s++) {
            slot_[s] = all[s * tiling_->tileLen];
        }
        scratch_ = (__local_mem__ float*)((__local_mem__ uint8_t*)all.GetPhyAddr() +
                                          SLOT_NUM * tiling_->tileLen * sizeof(T));
        FillScalarSlots();
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= tiling_->usedCoreNum) {
            return;
        }
        if (tiling_->tilingKey == LAMB_BRC_KEY_FLAT) {
            ProcessFlat();
        } else {
            ProcessBlock();
        }
    }

private:
    // 单元素输入是常量, 只铺一次, 后续分片不再重复搬运。
    __aicore__ inline void FillScalarSlots()
    {
        event_t e2s = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_S>());
        event_t s2v = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::S_V>());
        event_t v2m2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        for (uint32_t i = 0; i < NIN; i++) {
            if (tiling_->inKind[i] != LAMB_BRC_KIND_SCALAR) {
                continue;
            }
            DataCopyExtParams cp{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> pad{false, 0, 0, 0};
            DataCopyPad(slot_[TMP_SLOT], inGm_[i], cp, pad);
            SetFlag<HardEvent::MTE2_S>(e2s);
            WaitFlag<HardEvent::MTE2_S>(e2s);
            T v = slot_[TMP_SLOT].GetValue(0);
            SetFlag<HardEvent::S_V>(s2v);
            WaitFlag<HardEvent::S_V>(s2v);
            Duplicate(slot_[i], v, static_cast<int32_t>(tiling_->tileLen));
            // 暂存槽下一轮会被 MTE2 覆写, 需等本轮标量读取(S)与 Duplicate(V) 完成。
            SetFlag<HardEvent::V_MTE2>(v2m2);
            WaitFlag<HardEvent::V_MTE2>(v2m2);
        }
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_S>(e2s);
        GetTPipePtr()->ReleaseEventID<HardEvent::S_V>(s2v);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(v2m2);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessFlat()
    {
        uint64_t total = tiling_->totalNum;
        uint64_t perCore = tiling_->perCoreElems;
        uint64_t begin = perCore * blockIdx_;
        uint64_t end = (begin + perCore < total) ? (begin + perCore) : total;
        for (uint64_t off = begin; off < end; off += tiling_->tileLen) {
            uint32_t len = static_cast<uint32_t>(((end - off) < tiling_->tileLen) ? (end - off) : tiling_->tileLen);
            for (uint32_t i = 0; i < NIN; i++) {
                if (tiling_->inKind[i] == LAMB_BRC_KIND_SAME) {
                    CopyInRun(i, off, len);
                }
            }
            ComputeAndOut(off, len);
        }
    }

    __aicore__ inline void ProcessBlock()
    {
        uint64_t rows = tiling_->totalRows;
        uint64_t perCore = tiling_->rowsPerCore;
        uint64_t begin = perCore * blockIdx_;
        uint64_t end = (begin + perCore < rows) ? (begin + perCore) : rows;
        uint32_t blockLen = tiling_->blockLen;
        uint32_t rowsPerTile = tiling_->rowsPerTile;

        for (uint64_t r0 = begin; r0 < end; r0 += rowsPerTile) {
            uint64_t batch = ((end - r0) < rowsPerTile) ? (end - r0) : rowsPerTile;
            for (uint64_t j = 0; j < batch; j++) {
                uint32_t dstOff = static_cast<uint32_t>(j) * blockLen;
                for (uint32_t i = 0; i < NIN; i++) {
                    uint32_t kind = tiling_->inKind[i];
                    if (kind == LAMB_BRC_KIND_SCALAR) {
                        continue; // 标量槽已整片铺好, 各行共用
                    }
                    uint64_t srcOff = RowSrcOffset(i, r0 + j);
                    if (kind == LAMB_BRC_KIND_SAME) {
                        CopyInRun(i, srcOff, blockLen, dstOff);
                    } else {
                        CopyInBrc(i, srcOff, dstOff);
                    }
                }
            }
            ComputeAndOut(r0 * blockLen, static_cast<uint32_t>(batch) * blockLen);
        }
    }

    // 行索引按输出前导维分解, 再乘各输入的有效步长(广播轴步长为 0)。
    __aicore__ inline uint64_t RowSrcOffset(uint32_t i, uint64_t r)
    {
        uint64_t off = 0;
        uint64_t rem = r;
        if (tiling_->innerChunkCnt > 1) { // 行索引最低位是最内轴的段号
            uint64_t seg = rem % tiling_->innerChunkCnt;
            rem /= tiling_->innerChunkCnt;
            if (tiling_->inShape[i * LAMB_BRC_MAX_DIM + tiling_->collapsedRank - 1] != 1) {
                off += seg * tiling_->innerChunk;
            }
        }
        for (int32_t j = static_cast<int32_t>(tiling_->splitAxis) - 1; j >= 0; j--) {
            uint64_t d = tiling_->outShape[j];
            off += (rem % d) * tiling_->effStride[i * LAMB_BRC_MAX_DIM + j];
            rem /= d;
        }
        return off;
    }

    __aicore__ inline void CopyInRun(uint32_t i, uint64_t srcOff, uint32_t len, uint32_t dstOff = 0)
    {
        DataCopyExtParams cp{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(slot_[i][dstOff], inGm_[i][srcOff], cp, pad);
    }

    // 块内只可能是"尾轴 1->n"广播(host 的 splitFloor 保证), 或完全不需要展开。
    __aicore__ inline void CopyInBrc(uint32_t i, uint64_t srcOff, uint32_t dstOff = 0)
    {
        uint32_t blockLen = tiling_->blockLen;
        uint32_t srcLen = static_cast<uint32_t>(tiling_->srcBlockLen[i]);
        if (srcLen == blockLen) { // 块内无广播: 直接搬入, 不经暂存
            CopyInRun(i, srcOff, blockLen, dstOff);
            return;
        }
        // 上一次(同一行的前一个广播输入, 或前一行)的 Broadcast 还在读 TMP_SLOT, 必须等它读完
        // 再往里搬下一份 —— 否则 MTE2 的写越过 V 的读(WAR), 结果随机少量出错。
        event_t v2m2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        SetFlag<HardEvent::V_MTE2>(v2m2);
        WaitFlag<HardEvent::V_MTE2>(v2m2);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(v2m2);

        DataCopyExtParams cp{1, static_cast<uint32_t>(srcLen * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(slot_[TMP_SLOT], inGm_[i][srcOff], cp, pad);

        event_t e2v = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        SetFlag<HardEvent::MTE2_V>(e2v);
        WaitFlag<HardEvent::MTE2_V>(e2v);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(e2v);

        // [srcLen, 1] -> [srcLen, d]: 尾轴放大 d 倍
        uint32_t d = blockLen / srcLen;
        uint32_t dstShape[2] = {srcLen, d};
        uint32_t srcShape[2] = {srcLen, 1};
        Broadcast<T, 2, 1>(slot_[i][dstOff], slot_[TMP_SLOT], dstShape, srcShape);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeAndOut(uint64_t dstOff, uint32_t len)
    {
        event_t e2v = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        SetFlag<HardEvent::MTE2_V>(e2v);
        WaitFlag<HardEvent::MTE2_V>(e2v);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(e2v);

        __local_mem__ T* inPtr[NIN];
        __local_mem__ T* outPtr[NOUT];
        for (uint32_t i = 0; i < NIN; i++) {
            inPtr[i] = (__local_mem__ T*)slot_[i].GetPhyAddr();
        }
        for (uint32_t o = 0; o < NOUT; o++) {
            outPtr[o] = (__local_mem__ T*)slot_[NIN + o].GetPhyAddr();
        }
        // 暂存区按 tileLen 跨距切分: tileLen 已对齐到向量寄存器长度, 保证 Load/StoreAlign 的
        // 地址对齐要求; 若按当前分片长度 len 切分, len=1 这类用例会得到 4 字节偏移而触发对齐错误。
        __local_mem__ float* sc[4] = {scratch_, scratch_ + tiling_->tileLen, scratch_ + 2U * tiling_->tileLen,
                                      scratch_ + 3U * tiling_->tileLen};
        Compute::template Run<T>(inPtr, outPtr, sc, len);

        event_t v2m3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        SetFlag<HardEvent::V_MTE3>(v2m3);
        WaitFlag<HardEvent::V_MTE3>(v2m3);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(v2m3);

        DataCopyExtParams cp{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
        for (uint32_t o = 0; o < NOUT; o++) {
            DataCopyPad(outGm_[o][dstOff], slot_[NIN + o], cp);
        }

        // 搬出未完成前不得让下轮的 V/S 覆写输出槽(MTE3->MTE2 挡不住 V), 也不得让 MTE2 覆写输入槽。
        event_t m32v = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
        SetFlag<HardEvent::MTE3_V>(m32v);
        WaitFlag<HardEvent::MTE3_V>(m32v);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(m32v);
        event_t m32m2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
        SetFlag<HardEvent::MTE3_MTE2>(m32m2);
        WaitFlag<HardEvent::MTE3_MTE2>(m32m2);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(m32m2);
    }

    const Tiling* tiling_ = nullptr;
    uint32_t blockIdx_ = 0;
    GlobalTensor<T> inGm_[NIN];
    GlobalTensor<T> outGm_[NOUT];
    TBuf<TPosition::VECCALC> ubBuf_;
    LocalTensor<T> slot_[SLOT_NUM];
    __local_mem__ float* scratch_ = nullptr;
};
} // namespace LambBrc
#endif // LAMB_BRC_KERNEL_H
