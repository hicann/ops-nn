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
 * \file block_mmad_to_multi_mul.h
 * \brief
 */

#pragma once
#include "./block_mmad.h"
#include "../utils/layout_utils.h"
#include "../utils/tuple_utils.h"
#include "../policy/dispatch_policy.h"
#include "adv_api/reduce/reduce.h"

namespace Cmct {
namespace Gemm {
namespace Block {
template <class L1TileShape_, class L0TileShape_, class AType_, class BType_, class CType_, class BiasType_,
          class TileCopy_>
class BlockMmad<MatmulToVector<>, L1TileShape_, L0TileShape_, AType_, BType_, CType_, BiasType_, TileCopy_> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using A_T = typename AType::T;
    using B_T = typename BType::T;
    using C_T = typename CType::T;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    uint64_t m_;
    uint64_t n_;
    uint64_t k_;
    uint64_t baseM_{0};
    uint64_t baseN_{0};
    uint64_t alignN_{0};
    uint64_t baseK_{64};
    uint64_t tailM_{0};
    uint64_t tailN_{0};
    uint64_t tailK_{0};
    uint64_t currentK_{64};
    uint64_t loopK_{0};
    bool hasBias_{false};

    __aicore__ inline BlockMmad()
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0x0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0x1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0x0);
    }

    __aicore__ inline ~BlockMmad()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0x0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0x1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0x0);
    }

public:
    __aicore__ inline void Init(const TupleShape& shape, const TupleShape& blockInfo, uint64_t tailK, uint64_t loopK,
                                bool hasBias)
    {
        m_ = Get<DIMENSION_M>(shape);
        n_ = Get<DIMENSION_N>(shape);
        k_ = Get<DIMENSION_K>(shape);
        baseM_ = Get<0>(blockInfo);
        baseN_ = Get<1>(blockInfo);
        alignN_ = CeilAlign(baseN_, ALIGN_NUM);
        tailM_ = baseM_;
        tailN_ = baseN_;
        tailK_ = tailK;
        loopK_ = loopK;
        hasBias_ = hasBias;
    }

    __aicore__ inline void SetTailM(int64_t tailM) { tailM_ = tailM; }

    __aicore__ inline void SetTailN(int64_t tailN) { tailN_ = tailN; }

    __aicore__ inline void CopyIn(const AscendC::GlobalTensor<float>& srcGlobal,
                                  const AscendC::LocalTensor<float>& dstLocal, uint64_t blockCount)
    {
        uint32_t alignK = AscendC::CeilAlign(currentK_, ALIGN_NUM);
        uint8_t rightPadding = static_cast<uint8_t>(alignK - currentK_);
        AscendC::DataCopyPadExtParams<float> copyPadParams{true, 0, rightPadding, 0};
        uint32_t srcStride = static_cast<uint32_t>((k_ - currentK_) * sizeof(float));
        uint32_t dstStride = static_cast<uint32_t>((baseK_ - alignK) / ALIGN_NUM);
        AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(blockCount),
                                              static_cast<uint32_t>(currentK_ * sizeof(float)), srcStride, dstStride,
                                              0};
        AscendC::DataCopyPad(dstLocal, srcGlobal, copyParams, copyPadParams);
    }

    static __simd_vf__ inline void MulsVf(__ubuf__ float* dstPtr, __ubuf__ float* bubPtr, __ubuf__ float* aubPtr,
                                          uint64_t mAub, uint64_t nBub, uint64_t baseK, uint64_t alignN)
    {
        AscendC::MicroAPI::RegTensor<float> vSrcAReg0;
        AscendC::MicroAPI::RegTensor<float> vSrcBReg0;
        AscendC::MicroAPI::RegTensor<float> vDstReg0;
        AscendC::MicroAPI::RegTensor<float> vDstReg1;
        AscendC::MicroAPI::UnalignRegForStore unReg0;
        AscendC::MicroAPI::MaskReg maskReg0;
        uint32_t tmpCount0 = static_cast<uint32_t>(baseK);
        maskReg0 = AscendC::MicroAPI::UpdateMask<float>(tmpCount0);
        for (uint16_t mIdx = 0; mIdx < static_cast<uint16_t>(mAub); ++mIdx) {
            // 搬入A矩阵一行
            auto addrReg0 = AscendC::MicroAPI::CreateAddrReg<float>(mIdx, baseK);
            AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vSrcAReg0, aubPtr, addrReg0);
            // B矩阵baseN行去乘A矩阵同一行
            for (uint16_t nIdx = 0; nIdx < static_cast<uint16_t>(nBub); ++nIdx) {
                auto addrReg1 = AscendC::MicroAPI::CreateAddrReg<float>(nIdx, baseK);
                AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vSrcBReg0, bubPtr,
                                                                                            addrReg1);
                AscendC::MicroAPI::Mul(vDstReg0, vSrcAReg0, vSrcBReg0, maskReg0);
                // 乘出来的64个数直接做reduce成1个数
                AscendC::MicroAPI::Reduce<AscendC::MicroAPI::ReduceType::SUM>(vDstReg1, vDstReg0, maskReg0);
                AscendC::MicroAPI::StoreUnAlign(dstPtr, vDstReg1, unReg0, 1);
            }
            AscendC::MicroAPI::StoreUnAlignPost(dstPtr, unReg0, 0);
            dstPtr += alignN - nBub;
        }
    }

    __aicore__ inline void Process(uint64_t ubOffsetA, uint64_t ubOffsetB, uint64_t ubOffsetC)
    {
        AscendC::LocalTensor<float> aubLocal = ubLocal_[ubOffsetA];
        AscendC::LocalTensor<float> bubLocal = ubLocal_[ubOffsetB];
        AscendC::LocalTensor<float> cubLocal = ubLocal_[ubOffsetC];
        __ubuf__ float* dstPtr = (__ubuf__ float*)cubLocal.GetPhyAddr();
        __ubuf__ float* bubPtr = (__ubuf__ float*)bubLocal.GetPhyAddr();
        __ubuf__ float* aubPtr = (__ubuf__ float*)aubLocal.GetPhyAddr();
        AscendC::VF_CALL<MulsVf>(dstPtr, bubPtr, aubPtr, tailM_, tailN_, baseK_, alignN_);
    }

    __aicore__ inline void CopyOut(const AscendC::GlobalTensor<float>& cGlobal,
                                   const AscendC::LocalTensor<float>& outubLocal)
    {
        uint16_t blockCount = static_cast<uint16_t>(tailM_);
        uint32_t blockLen = static_cast<uint32_t>(tailN_ * sizeof(float));
        uint32_t srcStride = static_cast<uint32_t>((alignN_ - tailN_) / ALIGN_NUM);
        uint32_t dstStride = static_cast<uint32_t>((n_ - tailN_) * sizeof(float));
        AscendC::DataCopyExtParams copyParams{blockCount, blockLen, srcStride, dstStride, 0};
        AscendC::DataCopyPad(cGlobal, outubLocal, copyParams);
    }

    __aicore__ inline void operator()(const AscendC::GlobalTensor<float>& cGlobal,
                                      const AscendC::GlobalTensor<float>& aGlobal,
                                      const AscendC::GlobalTensor<float>& bGlobal)
    {
        uint64_t ubOffsetAPing = 0;
        uint64_t ubOffsetBPing = baseM_ * baseK_ + ubOffsetAPing;
        // for mul and reduceSum ping
        uint64_t ubOffsetCPing = baseN_ * baseK_ + ubOffsetBPing;
        // for add ping
        uint64_t ubOffsetDPing = baseM_ * alignN_ + ubOffsetCPing;

        uint64_t ubOffsetAPong = baseM_ * alignN_ + ubOffsetDPing;
        uint64_t ubOffsetBPong = baseM_ * baseK_ + ubOffsetAPong;
        // for mul and reduceSum pong
        uint64_t ubOffsetCPong = baseN_ * baseK_ + ubOffsetBPong;
        // for add pong
        uint64_t ubOffsetDPong = baseM_ * alignN_ + ubOffsetCPong;

        // for add out ping + pong
        uint64_t ubOffsetOut = baseM_ * alignN_ + ubOffsetDPong;

        uint64_t ubOffsetA[] = {ubOffsetAPing, ubOffsetAPong};
        uint64_t ubOffsetB[] = {ubOffsetBPing, ubOffsetBPong};
        uint64_t ubOffsetC[] = {ubOffsetCPing, ubOffsetCPong};
        uint64_t ubOffsetD[] = {ubOffsetDPing, ubOffsetDPong};
        constexpr bool isReuse = true;

        // reduceSum结果累加地址ubOffsetD需要清0
        AscendC::Duplicate<float>(ubLocal_[ubOffsetDPing], 0, static_cast<int32_t>(baseM_ * alignN_));
        AscendC::Duplicate<float>(ubLocal_[ubOffsetDPong], 0, static_cast<int32_t>(baseM_ * alignN_));
        for (uint64_t j = 0; j < loopK_; ++j) {
            currentK_ = (j + 1 == loopK_) ? tailK_ : baseK_;
            if (j == loopK_ - 1 && currentK_ != baseK_) {
                // 尾轮不允许脏数据污染
                AscendC::Duplicate<float>(ubLocal_[ubOffsetA[j & 0x1]], 0, static_cast<int32_t>(baseM_ * baseK_));
                AscendC::Duplicate<float>(ubLocal_[ubOffsetB[j & 0x1]], 0, static_cast<int32_t>(baseN_ * baseK_));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_FLAG);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_FLAG);
            }
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(j & 0x1);
            CopyIn(aGlobal[j * baseK_], ubLocal_[ubOffsetA[j & 0x1]], tailM_);
            CopyIn(bGlobal[j * baseK_], ubLocal_[ubOffsetB[j & 0x1]], tailN_);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(j & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(j & 0x1);
            // 微指令完成(1,baseK) * (baseN, baseK) -> reduceSum -> baseN
            Process(ubOffsetA[j & 0x1], ubOffsetB[j & 0x1], ubOffsetC[j & 0x1]);
            // A矩阵和B矩阵使用完毕，可以搬新的进来了
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(j & 0x1);
            // 把算好的(baseM, baseN)加到对应的out
            AscendC::Add(ubLocal_[ubOffsetD[j & 0x1]], ubLocal_[ubOffsetD[j & 0x1]], ubLocal_[ubOffsetC[j & 0x1]],
                         static_cast<int32_t>(baseM_ * alignN_));
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0x0);
        AscendC::Add(ubLocal_[ubOffsetOut], ubLocal_[ubOffsetDPing], ubLocal_[ubOffsetDPong],
                     static_cast<int32_t>(baseM_ * alignN_));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0x0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0x0);
        CopyOut(cGlobal, ubLocal_[ubOffsetOut]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0x0);
    }

private:
    constexpr static uint16_t DIMENSION_M = 0;
    constexpr static uint16_t DIMENSION_N = 1;
    constexpr static uint16_t DIMENSION_K = 2;
    constexpr static uint64_t ALIGN_NUM = 8;
    constexpr static uint16_t SYNC_FLAG = 2;
    AscendC::LocalTensor<float> ubLocal_{AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE};
};
} // namespace Block
} // namespace Gemm
} // namespace Cmct