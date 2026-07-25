/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONV_BP_WINO_DATA_QUEUE_H
#define CONV_BP_WINO_DATA_QUEUE_H

#include "conv_bp_wino_util.h"

// 正变换后在gm上的排布[N,k1(TileH/SingleShapeTileH * TileW/SingleShapeTileW),C1,k0(16,SingleShapeTileHW),C0]
namespace NK1C1K0C0 {
template <typename T>
struct Shape {
    template <typename TilingT>
    __aicore__ inline static Shape Create(const uint32_t c, const uint32_t tileH, const uint32_t tileW)
    {
        constexpr uint32_t singleShapeTileH = BlockConfig::SingleShapeTileH<TilingT>();
        constexpr uint32_t singleShapeTileW = BlockConfig::SingleShapeTileW<TilingT>();
        uint32_t k1 = Ops::Base::CeilDiv(tileH, singleShapeTileH) * Ops::Base::CeilDiv(tileW, singleShapeTileW);

        constexpr uint32_t k0 = singleShapeTileH * singleShapeTileW * F23_TRANSFORM_TILE_ELEMENTS_16;

        uint32_t c1 = Ops::Base::CeilDiv(c, C0<T>());

        return Shape(k1, c1, k0);
    }

    __aicore__ inline Shape(const uint32_t k1, const uint32_t c1, const uint32_t k0) : k1(k1), c1(c1), k0(k0) {}

    __aicore__ inline uint64_t GetOffset(uint32_t nIdx, uint32_t k1Idx, uint32_t c1Idx) const
    {
        uint64_t k0c0 = static_cast<uint64_t>(k0) * c0;
        uint64_t c1k0c0 = static_cast<uint64_t>(c1) * k0c0;
        uint64_t k1c1k0c0 = static_cast<uint64_t>(k1) * c1k0c0;

        return nIdx * k1c1k0c0 + k1Idx * c1k0c0 + c1Idx * k0c0;
    }

    const uint32_t k1;
    const uint32_t c1;
    const uint32_t k0;
    static constexpr uint8_t c0 = C0<T>();
};

struct CopyK0Params {
    uint32_t batchIdx = 0;
    uint32_t k1Idx = 0;
    uint32_t tiles = 0;
    uint32_t srcBufWidthBlockStride = 0;
    uint32_t c1Idx = 0;
    uint32_t c1Length = 0;
};

template <typename T>
__aicore__ inline void CopyK0UB2GM(const CopyK0Params& p, const AscendC::LocalTensor<T>& ub,
                                   const AscendC::GlobalTensor<T>& gm, const Shape<T>& shape)
{
    uint64_t gmOffset = shape.GetOffset(p.batchIdx, p.k1Idx, p.c1Idx);

    AscendC::DataCopyParams params;
    params.blockCount = F23_TRANSFORM_TILE_ELEMENTS_16;
    params.blockLen = p.tiles;
    params.srcGap = p.srcBufWidthBlockStride - p.tiles;
    params.dstGap = 0;

    constexpr uint8_t c0Byte = Shape<T>::c0 * sizeof(T);
    AscendC::LoopModeParams loop;
    loop.loop1Size = p.c1Length;
    loop.loop1SrcStride = F23_TRANSFORM_TILE_ELEMENTS_16 * p.srcBufWidthBlockStride * c0Byte;
    loop.loop1DstStride = shape.k0 * c0Byte;
    loop.loop2Size = 1;

    AscendC::SetLoopModePara(loop, AscendC::DataCopyMVType::UB_TO_OUT);
    AscendC::DataCopy(gm[gmOffset], ub, params);
    AscendC::ResetLoopModePara(AscendC::DataCopyMVType::UB_TO_OUT);
}

template <typename T>
__aicore__ inline void CopyK0GM2L1(const CopyK0Params& p, const AscendC::GlobalTensor<T>& gm,
                                   const AscendC::LocalTensor<T>& l1, const Shape<T>& shape)
{
    uint64_t gmOffset = shape.GetOffset(p.batchIdx, p.k1Idx, p.c1Idx);

    AscendC::DataCopyParams params;
    params.blockCount = p.c1Length;
    params.blockLen = p.tiles * F23_TRANSFORM_TILE_ELEMENTS_16;
    params.srcGap = shape.k0 - params.blockLen;
    params.dstGap = 0;

    AscendC::DataCopy(l1, gm[gmOffset], params);
}

template <typename T>
__aicore__ inline void CopyK0UB2L1(const CopyK0Params& p, const AscendC::LocalTensor<T>& ub,
                                   const AscendC::LocalTensor<T>& l1)
{
    for (uint32_t c1 = 0; c1 < p.c1Length; c1++) {
        AscendC::DataCopyParams params;
        params.blockCount = F23_TRANSFORM_TILE_ELEMENTS_16;
        params.blockLen = p.tiles;
        params.srcGap = p.srcBufWidthBlockStride - p.tiles;
        params.dstGap = 0;

        uint32_t ubOffset = p.srcBufWidthBlockStride * F23_TRANSFORM_TILE_ELEMENTS_16 * C0<T>() * c1;
        uint32_t l1Offset = p.tiles * F23_TRANSFORM_TILE_ELEMENTS_16 * C0<T>() * c1;
        AscendC::DataCopy(l1[l1Offset], ub[ubOffset], params);
    }
}
} // namespace NK1C1K0C0

// CrossCoreSetFlag内计数器上限不能超过15
// 这里设置连续EnQue12次就要等待DeQue通知，防止计数器超限
static constexpr uint8_t DEFAULT_FREE_SLOTS = 12;
// 纯PingPong写入,只允许连续EnQue2次就要等DeQue通知
static constexpr uint8_t PINGPONG_FREE_SLOTS = 2;
// 没有pingpong,只允许一个写入
static constexpr uint8_t SINGLE_FREE_SLOTS = 1;

// SrcPipe 输出数据的pipe
// DST_PIPE 数据输出后触发执行的pipe
// POP_PIPE 将数据搬出的pipe，该pipe执行完后即代表存在空闲空间
//
// 典型场景:
// fixpipe搬入ub后做InPlace计算在搬出:
// FIXPIPE(SrcPipe)->V(DST_PIPE)->MTE3(POP_PIPE)
// ub计算后cube搬入
// MTE3(SrcPipe)->MTE1(DST_PIPE&POP_PIPE)
//

template <pipe_t Src, pipe_t Dst, pipe_t Pop, uint8_t PushFlag, uint8_t PopFlag, uint8_t FreeSlots, bool C2v>
struct CVSyncQueConfig {
    static constexpr pipe_t SRC_PIPE = Src;
    static constexpr pipe_t DST_PIPE = Dst;
    static constexpr pipe_t POP_PIPE = Pop;
    static constexpr uint8_t PUSH_FLAG = PushFlag;
    static constexpr uint8_t POP_FLAG = PopFlag;
    static constexpr uint8_t FREE_SLOTS = FreeSlots;
    static constexpr bool C2V = C2v;
};

template <typename Config>
class CVSyncQue {
public:
    static constexpr pipe_t SRC_PIPE = Config::SRC_PIPE;
    static constexpr pipe_t DST_PIPE = Config::DST_PIPE;
    static constexpr pipe_t POP_PIPE = Config::POP_PIPE;
    static constexpr uint8_t PUSH_FLAG = Config::PUSH_FLAG;
    static constexpr uint8_t POP_FLAG = Config::POP_FLAG;
    static constexpr uint8_t FREE_SLOTS = Config::FREE_SLOTS;
    static constexpr bool C2V = Config::C2V;
    static constexpr uint8_t SYNC_MODE_4 = 4;
    static constexpr uint8_t MODE_4_AIV_FLAG_STRIDE = 16;

    __aicore__ inline void WaitSlot()
    {
        if (freeSlots_ == 0) {
            if constexpr (C2V) {
                // 整个队列是按模式2实现的，但是模式2跑仿真时有bug,会产生多余的set
                // 先用模式4模拟模式2
#pragma unroll
                for (uint8_t i = 0; i < AivNumInBlock(); i++) {
                    AscendC::CrossCoreWaitFlag<SYNC_MODE_4, SRC_PIPE>(POP_FLAG + MODE_4_AIV_FLAG_STRIDE * i);
                }
            } else {
                AscendC::CrossCoreWaitFlag<SYNC_MODE_4, SRC_PIPE>(POP_FLAG);
            }
        }
    }

    __aicore__ inline void EnQue()
    {
        if constexpr (C2V) {
#pragma unroll
            for (uint8_t i = 0; i < AivNumInBlock(); i++) {
                AscendC::CrossCoreSetFlag<SYNC_MODE_4, SRC_PIPE>(PUSH_FLAG + MODE_4_AIV_FLAG_STRIDE * i);
            }
        } else {
            AscendC::CrossCoreSetFlag<SYNC_MODE_4, SRC_PIPE>(PUSH_FLAG);
        }
        if (freeSlots_ > 0) {
            freeSlots_--;
        }
    }

    template <pipe_t SYNC_DST_PIPE = DST_PIPE>
    __aicore__ inline void WaitData()
    {
        if constexpr (C2V) {
            AscendC::CrossCoreWaitFlag<SYNC_MODE_4, SYNC_DST_PIPE>(PUSH_FLAG);
        } else {
#pragma unroll
            for (uint8_t i = 0; i < AivNumInBlock(); i++) {
                AscendC::CrossCoreWaitFlag<SYNC_MODE_4, SYNC_DST_PIPE>(PUSH_FLAG + MODE_4_AIV_FLAG_STRIDE * i);
            }
        }
    }

    __aicore__ inline void DeQue()
    {
        if constexpr (C2V) {
            AscendC::CrossCoreSetFlag<SYNC_MODE_4, POP_PIPE>(POP_FLAG);
        } else {
#pragma unroll
            for (uint8_t i = 0; i < AivNumInBlock(); i++) {
                AscendC::CrossCoreSetFlag<SYNC_MODE_4, POP_PIPE>(POP_FLAG + MODE_4_AIV_FLAG_STRIDE * i);
            }
        }
    }

    __aicore__ inline void End()
    {
        // 如果CrossCoreSetFlag是最后的指令可能因为一执行完核就退出导致没能成功set,整个核结束前加个全量等待
        AscendC::PipeBarrier<PIPE_ALL>();
    }

protected:
    using QueT = CVSyncQue;

private:
    uint8_t freeSlots_ = FREE_SLOTS;
};

// ub正变换到l1的队列,分成ub->l1和ub->gm->l1这2类
// 使用接口:
//   Init()初始化
//   1.ub写出
//     q.WaitSlot() 等待队列空间
//     q.Write() 写入数据
//     q.EnQue() 完成写入,执行CrossCore通知可读
//  2. l1读出
//     q.WaitData() 等待队列数据
//     LoadL1 读取数据到L1
//     q.DeQue() 释放队列空间
//  End()释放资源
//

template <typename T, uint8_t PUSH_FLAG, uint8_t POP_FLAG>
class UB2L1Queue
    : public CVSyncQue<
          CVSyncQueConfig<PIPE_MTE3, PIPE_MTE1, PIPE_MTE1, PUSH_FLAG, POP_FLAG, PINGPONG_FREE_SLOTS, false> > {
public:
    __aicore__ inline void Init(AscendC::LocalTensor<T> (&l1FmapBuf)[2], AscendC::LocalTensor<T> (&l1DyBuf)[2])
    {
        l1Fmap_[0] = l1FmapBuf[0];
        l1Fmap_[1] = l1FmapBuf[1];
        l1Dy_[0] = l1DyBuf[0];
        l1Dy_[1] = l1DyBuf[1];
    }

    __aicore__ inline void WriteFmap(const NK1C1K0C0::CopyK0Params& p, const AscendC::LocalTensor<T>& ub,
                                     const uint32_t l1Offset)
    {
        Write<true>(p, ub, l1Offset);
    }

    __aicore__ inline void WriteDy(const NK1C1K0C0::CopyK0Params& p, const AscendC::LocalTensor<T>& ub,
                                   const uint32_t l1Offset)
    {
        Write<false>(p, ub, l1Offset);
    }

    __aicore__ inline void EnQue()
    {
        if ASCEND_IS_AIV {
            UB2L1Queue::QueT::EnQue();
            writeFmapPingPongFlag_ = !writeFmapPingPongFlag_;
            writeDyPingPongFlag_ = !writeDyPingPongFlag_;
        }
    }

private:
    template <bool WriteFmap>
    __aicore__ inline void Write(const NK1C1K0C0::CopyK0Params& p, const AscendC::LocalTensor<T>& ub,
                                 const uint32_t l1Offset)
    {
        if ASCEND_IS_AIV {
            if constexpr (WriteFmap) {
                NK1C1K0C0::CopyK0UB2L1(p, ub, this->l1Fmap_[writeFmapPingPongFlag_][l1Offset]);
            } else {
                NK1C1K0C0::CopyK0UB2L1(p, ub, this->l1Dy_[writeDyPingPongFlag_][l1Offset]);
            }
        }
    }

    AscendC::LocalTensor<T> l1Fmap_[2];
    AscendC::LocalTensor<T> l1Dy_[2];
    bool writeFmapPingPongFlag_ = false;
    bool writeDyPingPongFlag_ = false;
};

template <typename T, uint8_t PUSH_FLAG, uint8_t POP_FLAG, uint8_t AIC_MTE2_SYNC_FLAG>
class GM2L1Queue
    : public CVSyncQue<
          CVSyncQueConfig<PIPE_MTE3, PIPE_MTE2, PIPE_MTE2, PUSH_FLAG, POP_FLAG, DEFAULT_FREE_SLOTS, false> > {
public:
    __aicore__ inline GM2L1Queue(__gm__ T* gm, const NK1C1K0C0::Shape<T>& shape) : shape_(shape)
    {
        gm_.SetGlobalBuffer(gm);
    }

    __aicore__ inline void Write(const NK1C1K0C0::CopyK0Params& p, const AscendC::LocalTensor<T>& ub)
    {
        if ASCEND_IS_AIV {
            NK1C1K0C0::CopyK0UB2GM(p, ub, gm_, shape_);
        }
    }

    __aicore__ inline void WaitData()
    {
        if ASCEND_IS_AIC {
            // 所有cube核接收到aiv发送的通知后才表示这一轮数据都准备好了
            // 但整个矩阵都变换完后就不需要对cube做全局同步额外等通知

            // 这里aiv通知后trigger的pipe为Fixpipe而非MTE2
            // 如果直接触发mte2,后续全核同步就需要在mte2做SetWaitFlag，
            // 这需要等待全核当前所有mte2搬运操作完成，从而大大降低mte2的并行度
            // Fixpipe负责L0C的搬出，一版只有在k轴完成累加后才触发，执行频率相对mte2低不少
            // 因此这里借用执行Fixpipe作为中转流水,通过Fixpipe做全核同步降低对整体并行度的影响
            GM2L1Queue::QueT::template WaitData<PIPE_FIX>();
            AscendC::CrossCoreSetFlag<0, PIPE_FIX>(AIC_MTE2_SYNC_FLAG);
            AscendC::CrossCoreWaitFlag<0, PIPE_MTE2>(AIC_MTE2_SYNC_FLAG);
        }
    }

    __aicore__ inline void WaitSlot()
    {
        if ASCEND_IS_AIV {
            GM2L1Queue::QueT::WaitSlot();
        }
    }

    __aicore__ inline void DeQue()
    {
        if ASCEND_IS_AIC {
            GM2L1Queue::QueT::DeQue();
        }
    }

    __aicore__ inline void EnQue()
    {
        if ASCEND_IS_AIV {
            GM2L1Queue::QueT::EnQue();
        }
    }

    __aicore__ inline const AscendC::GlobalTensor<T>& GetGlobalTensor() const { return gm_; }

    __aicore__ inline const NK1C1K0C0::Shape<T>& GetGMShape() const { return shape_; }

private:
    AscendC::GlobalTensor<T> gm_;
    const NK1C1K0C0::Shape<T> shape_;
};

#endif // CONV_BP_WINO_DATA_QUEUE_H
