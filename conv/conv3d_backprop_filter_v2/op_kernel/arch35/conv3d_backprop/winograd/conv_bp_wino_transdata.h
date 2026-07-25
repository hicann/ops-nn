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
 * \file conv_bp_wino_transdata.h
 * \brief
 */

#ifndef CONV_BP_WINO_TRANSDATA_H
#define CONV_BP_WINO_TRANSDATA_H

#include "conv_bp_wino_data_queue.h"

static constexpr uint8_t CROSS_CORE_AIC2AIV_SEND_DN2NZ_FLAG = 7;
static constexpr uint8_t CROSS_CORE_AIV2AIC_RECV_DN2NZ_FLAG = 8;
static constexpr uint8_t CROSS_CORE_AIV_TRANS_END_SYNC_FLAG = 9;
static constexpr uint8_t CROSS_CORE_AIV2AIC_SEND_TRANS_END_FLAG = 10;

template <typename T>
class WinoPreTransData {
public:
    using CVQue = CVSyncQue<CVSyncQueConfig<PIPE_MTE1, PIPE_MTE3, PIPE_MTE3, CROSS_CORE_AIC2AIV_SEND_DN2NZ_FLAG,
                                            CROSS_CORE_AIV2AIC_RECV_DN2NZ_FLAG, PINGPONG_FREE_SLOTS, true> >;

    __aicore__ inline void Init()
    {
        TPipe* pipe = GetTPipePtr();
        if ASCEND_IS_AIC {
            mte22mte1_[0] = pipe->AllocEventID<HardEvent::MTE2_MTE1>();
            mte22mte1_[1] = pipe->AllocEventID<HardEvent::MTE2_MTE1>();
            mte12mte2_[0] = pipe->AllocEventID<HardEvent::MTE1_MTE2>();
            mte12mte2_[1] = pipe->AllocEventID<HardEvent::MTE1_MTE2>();

            SetFlag<HardEvent::MTE1_MTE2>(mte12mte2_[0]);
            SetFlag<HardEvent::MTE1_MTE2>(mte12mte2_[1]);
        }
    }

    __aicore__ inline void TransData2NC1HWC0(__gm__ T* in, __gm__ T* out, uint32_t n, uint32_t c, uint32_t h,
                                             uint32_t w, bool disableInputL2Cache)
    {
        GlobalTensor<T> src, dst;
        src.SetGlobalBuffer(in);
        if (disableInputL2Cache) {
            src.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        dst.SetGlobalBuffer(out);

        uint32_t blockC0, blockHW;
        InitDN2NZTiling(h, w, blockC0, blockHW);
        const uint16_t coreNum = GetBlockNum();

        const uint32_t hw = h * w;
        const uint32_t blockC = blockC0 * C0<T>();
        const uint32_t cBlockCnt = Ops::Base::CeilDiv(c, blockC);
        const uint32_t hwBlockCnt = Ops::Base::CeilDiv(hw, blockHW);
        const uint32_t totalTasks = cBlockCnt * hwBlockCnt * n;

        for (uint32_t i = AicCoreId(); i < totalTasks; i += coreNum) {
            uint32_t hwBlockIdx = i % hwBlockCnt;
            uint32_t cBlockIdx = (i / hwBlockCnt) % cBlockCnt;
            uint32_t nIdx = i / (hwBlockCnt * cBlockCnt);

            uint32_t hwIdx = hwBlockIdx * blockHW;
            uint32_t cIdx = cBlockIdx * blockC;

            uint32_t hwLen = Std::min(blockHW, hw - hwIdx);
            uint32_t cLen = Std::min(blockC, c - cIdx);

            LocalTensor<T> l1(TPosition::A1, pingPongFlag_ * L1_HALF_BYTES, L1_HALF_BYTES);
            LocalTensor<T> ub(TPosition::VECIN, pingPongFlag_ * UB_HALF_BYTES, UB_HALF_BYTES);

            if ASCEND_IS_AIC {
                // dn2nz拷贝进l1没法直接考出去，得传到ub
                CopyInDN2NZ2UB(src, l1, ub, c, hw, nIdx, cIdx, hwIdx, cLen, hwLen);
            }
            if ASCEND_IS_AIV {
                CopyOutInAiv(dst, ub, c, hw, nIdx, cIdx, hwIdx, cLen, hwLen);
            }

            pingPongFlag_ = !pingPongFlag_;
        }
    }

    __aicore__ inline void End()
    {
        TPipe* pipe = GetTPipePtr();
        if ASCEND_IS_AIC {
            WaitFlag<HardEvent::MTE1_MTE2>(mte12mte2_[0]);
            WaitFlag<HardEvent::MTE1_MTE2>(mte12mte2_[1]);

            pipe->ReleaseEventID<HardEvent::MTE1_MTE2>(mte12mte2_[0]);
            pipe->ReleaseEventID<HardEvent::MTE1_MTE2>(mte12mte2_[1]);
            pipe->ReleaseEventID<HardEvent::MTE2_MTE1>(mte22mte1_[0]);
            pipe->ReleaseEventID<HardEvent::MTE2_MTE1>(mte22mte1_[1]);

            // 后续所有mte2操作等aiv搬运结束后在进行
            CrossCoreWaitFlag<4, PIPE_MTE2>(CROSS_CORE_AIV2AIC_SEND_TRANS_END_FLAG);
        }

        if ASCEND_IS_AIV {
            CrossCoreSetFlag<0, PIPE_MTE3>(CROSS_CORE_AIV_TRANS_END_SYNC_FLAG);
            CrossCoreWaitFlag<0, PIPE_MTE2>(CROSS_CORE_AIV_TRANS_END_SYNC_FLAG);
            if (GetSubBlockIdx() == 0) {
                CrossCoreSetFlag<4, PIPE_MTE2>(CROSS_CORE_AIV2AIC_SEND_TRANS_END_FLAG);
            }
        }
    }

private:
    __aicore__ static inline void InitDN2NZTiling(uint32_t h, uint32_t w, uint32_t& outBlockC0, uint32_t& outBlockHW)
    {
        uint32_t hw = h * w;
        constexpr uint32_t maxHWCapacity = BUF_SIZE / DN2NZ_BASE_D;
        constexpr uint32_t maxHWCapacityAligned = ConstexprMaths::AlignDown(maxHWCapacity, DN2NZ_BASE_N);

        if (hw > maxHWCapacity) {
            uint32_t numSplits = Ops::Base::CeilDiv(hw, maxHWCapacityAligned);
            uint32_t avgBlockHW = Ops::Base::CeilDiv(hw, numSplits);
            outBlockHW = Ops::Base::CeilAlign(avgBlockHW, DN2NZ_BASE_N);
        } else {
            // 优先整个hw放进来
            outBlockHW = hw;
        }

        outBlockC0 = Ops::Base::FloorAlign(BUF_SIZE / outBlockHW, DN2NZ_BASE_D) / C0<T>();
    }

    __aicore__ inline void CopyInDN2NZ2UB(const GlobalTensor<T>& src, const LocalTensor<T>& l1,
                                          const LocalTensor<T>& ub, uint32_t c, uint32_t hw, uint32_t nIdx,
                                          uint32_t cIdx, uint32_t hwIdx, uint32_t cLen, uint32_t hwLen)
    {
        const TEventID mte12mte2 = mte12mte2_[pingPongFlag_];
        const TEventID mte22mte1 = mte22mte1_[pingPongFlag_];

        WaitFlag<HardEvent::MTE1_MTE2>(mte12mte2);

        Dn2NzParams dn2nz;
        dn2nz.dnNum = 1;
        dn2nz.dValue = cLen;
        dn2nz.nValue = hwLen;
        dn2nz.srcDValue = hw;
        dn2nz.dstNzC0Stride = hwLen;
        dn2nz.dstNzNStride = 1;

        uint64_t gmOffset = static_cast<uint64_t>(nIdx) * c * hw + static_cast<uint64_t>(cIdx) * hw +
                            static_cast<uint64_t>(hwIdx);

        DataCopy(l1, src[gmOffset], dn2nz);

        SetFlag<HardEvent::MTE2_MTE1>(mte22mte1);
        WaitFlag<HardEvent::MTE2_MTE1>(mte22mte1);

        __ubuf__ T* ubAddr = reinterpret_cast<__ubuf__ T*>(ub.GetPhyAddr());
        __cbuf__ T* l1Addr = reinterpret_cast<__cbuf__ T*>(l1.GetPhyAddr());

        syncQue_.WaitSlot();
        CopyCbufToUbuf(ubAddr, l1Addr, 1, hwLen * Ops::Base::CeilDiv(cLen, C0<T>()), 0, 0);

        SetFlag<HardEvent::MTE1_MTE2>(mte12mte2);
        syncQue_.EnQue();
    }

    __aicore__ inline void CopyOutInAiv(const GlobalTensor<T>& dst, const LocalTensor<T>& ub, uint32_t c, uint32_t hw,
                                        uint32_t nIdx, uint32_t cIdx, uint32_t hwIdx, uint32_t cLen, uint32_t hwLen)
    {
        syncQue_.WaitData();
        // l1->ub当前只会写到0核上
        if (GetSubBlockIdx() == 0) {
            const uint32_t c1 = Ops::Base::CeilDiv(c, C0<T>());
            const uint32_t c1Idx = cIdx / C0<T>();
            const uint64_t gmOffset = static_cast<uint64_t>(nIdx) * c1 * hw * C0<T>() +
                                      static_cast<uint64_t>(c1Idx) * hw * C0<T>() +
                                      static_cast<uint64_t>(hwIdx) * C0<T>();

            DataCopyParams p;
            p.blockCount = Ops::Base::CeilDiv(cLen, C0<T>());
            p.blockLen = hwLen;
            p.srcStride = 0;
            p.dstStride = hw - hwLen;
            DataCopy(dst[gmOffset], ub, p);
        }
        syncQue_.DeQue();
    }

    static constexpr uint32_t L1_HALF_BYTES = AscendC::TOTAL_L1_SIZE / 2;
    static constexpr uint32_t UB_HALF_BYTES = AscendC::TOTAL_UB_SIZE / 2;
    static constexpr uint32_t BUF_SIZE = ConstexprMaths::Min(L1_HALF_BYTES, UB_HALF_BYTES) / sizeof(T);
    // DN尽量贴合C0*256Byte
    static constexpr uint32_t DN2NZ_BASE_D = C0<T>();
    static constexpr uint32_t DN2NZ_BASE_N = 256 / sizeof(T);

    CVQue syncQue_;
    bool pingPongFlag_ = false;
    TEventID mte12mte2_[2] = {0, 0};
    TEventID mte22mte1_[2] = {0, 0};
};

#endif // CONV_BP_WINO_TRANSDATA_H
