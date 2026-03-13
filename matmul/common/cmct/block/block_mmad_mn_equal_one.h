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
 * \file block_mmad_mn_equal_one.h
 * \brief
 */

#ifndef MATMUL_BLOCK_BLOCK_MN_EQUAL_ONE_H
#define MATMUL_BLOCK_BLOCK_MN_EQUAL_ONE_H
#include "./block_mmad.h"
#include "../utils/layout_utils.h"
#include "../utils/tuple_utils.h"
#include "../policy/dispatch_policy.h"
#include "adv_api/reduce/reduce.h"

namespace Cmct {
namespace Gemm {
namespace Block {
template <
    class L1TileShape_, class L0TileShape_, class AType_, class BType_, class CType_, class BiasType_, class TileCopy_>
class BlockMmad<MatmulMNEqualOne<>, L1TileShape_, L0TileShape_, AType_, BType_, CType_, BiasType_, TileCopy_> {
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
    uint64_t baseK_{256};
    uint64_t baseMN_{32};
    uint64_t loopK_{1};
    uint64_t loopMN_{1};
    uint64_t shapeMN_{n_};
    bool hasBias_{false};
    bool dataCopyMode_{false};

    __aicore__ inline BlockMmad()
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0x0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0x1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0x0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0x1);
    }

    __aicore__ inline ~BlockMmad()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0x0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0x1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0x0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0x1);
    }

public:
    __aicore__ inline void Init(const TupleShape& shape, int64_t loopK, bool hasBias, bool dataCopyMode)
    {
        m_ = Get<DIMENSION_M>(shape);
        n_ = Get<DIMENSION_N>(shape);
        k_ = Get<DIMENSION_K>(shape);
        loopK_ = loopK * 2;
        hasBias_ = hasBias;
        dataCopyMode_ = dataCopyMode;
        shapeMN_ = m_ == 1 ? n_ : m_;
    }

    __aicore__ inline void CopyInA(
        const AscendC::GlobalTensor<float>& aGlobal, const AscendC::LocalTensor<float>& aubLocal)
    {
        if (dataCopyMode_) {
            AscendC::MultiCopyParams<float, 2> ndDmaParams;
            ndDmaParams.loopInfo.loopSrcStride[0] = 0;
            ndDmaParams.loopInfo.loopSrcStride[1] = 1;
            ndDmaParams.loopInfo.loopDstStride[0] = 1;
            ndDmaParams.loopInfo.loopDstStride[1] = static_cast<uint32_t>(baseMN_);
            ndDmaParams.loopInfo.loopSize[0] = static_cast<uint32_t>(baseMN_);
            ndDmaParams.loopInfo.loopSize[1] = static_cast<uint32_t>(baseK_);
            ndDmaParams.loopInfo.loopLpSize[0] = 0;
            ndDmaParams.loopInfo.loopLpSize[1] = 0;
            ndDmaParams.loopInfo.loopRpSize[0] = 0;
            ndDmaParams.loopInfo.loopRpSize[1] = 0;
            AscendC::DataCopy(aubLocal, aGlobal, ndDmaParams);
        } else {
            AscendC::MultiCopyParams<float, 2> ndDmaParams;
            ndDmaParams.loopInfo.loopSrcStride[0] = 1;
            ndDmaParams.loopInfo.loopSrcStride[1] = 0;
            ndDmaParams.loopInfo.loopDstStride[0] = 1;
            ndDmaParams.loopInfo.loopDstStride[1] = static_cast<uint32_t>(baseK_);
            ndDmaParams.loopInfo.loopSize[0] = static_cast<uint32_t>(baseK_);
            ndDmaParams.loopInfo.loopSize[1] = static_cast<uint32_t>(baseMN_);
            ndDmaParams.loopInfo.loopLpSize[0] = 0;
            ndDmaParams.loopInfo.loopLpSize[1] = 0;
            ndDmaParams.loopInfo.loopRpSize[0] = 0;
            ndDmaParams.loopInfo.loopRpSize[1] = 0;
            AscendC::DataCopy(aubLocal, aGlobal, ndDmaParams);
        }
    }

    __aicore__ inline void CopyInB(
        const AscendC::GlobalTensor<float>& bGlobal, const AscendC::LocalTensor<float>& bubLocal)
    {
        if (dataCopyMode_) {
            AscendC::DataCopyPadExtParams<float> copyPadParams{false, 0, 0, 0};
            uint32_t srcStride = static_cast<uint32_t>((shapeMN_ - baseMN_) * sizeof(float));
            AscendC::DataCopyExtParams copyParams{
                static_cast<uint16_t>(baseK_), static_cast<uint32_t>(baseMN_ * sizeof(float)), srcStride, 0, 0};
            AscendC::DataCopyPad(bubLocal, bGlobal, copyParams, copyPadParams);
        } else {
            AscendC::DataCopyPadExtParams<float> copyPadParams{false, 0, 0, 0};
            uint32_t srcStride = static_cast<uint32_t>((k_ - baseK_) * sizeof(float));
            AscendC::DataCopyExtParams copyParams{
                static_cast<uint16_t>(baseMN_), static_cast<uint32_t>(baseK_ * sizeof(float)), srcStride, 0, 0};
            AscendC::DataCopyPad(bubLocal, bGlobal, copyParams, copyPadParams);
        }
    }

    __aicore__ inline void AivProcess(uint64_t ubOffsetA, uint64_t ubOffsetB, uint64_t ubOffsetC)
    {
        AscendC::LocalTensor<float> aubLocal = ubLocal_[ubOffsetA];
        AscendC::LocalTensor<float> bubLocal = ubLocal_[ubOffsetB];
        AscendC::LocalTensor<float> cubLocal = ubLocal_[ubOffsetC];
        int32_t calCount = static_cast<int32_t>(baseMN_ * baseK_);
        AscendC::MulAddDst(cubLocal, aubLocal, bubLocal, calCount);
    }

    __aicore__ inline void CopyOut(
        const AscendC::GlobalTensor<float>& cGlobal, const AscendC::LocalTensor<float>& outubLocal)
    {
        uint32_t blockLen = static_cast<uint32_t>(baseMN_ * sizeof(float));
        AscendC::DataCopyExtParams copyParams{1, blockLen, 0, 0, 0};
        AscendC::DataCopyPad(cGlobal, outubLocal, copyParams);
    }

    __aicore__ inline void operator()(
        const AscendC::GlobalTensor<float>& cGlobal, const AscendC::GlobalTensor<float>& aGlobal,
        const AscendC::GlobalTensor<float>& bGlobal)
    {
        // 这些地址偏移可以在tiling侧算好
        uint64_t ubOffsetAPing = 0;
        uint64_t ubOffsetBPing = baseMN_ * baseK_ + ubOffsetAPing;
        uint64_t ubOffsetAPong = baseMN_ * baseK_ + ubOffsetBPing;
        uint64_t ubOffsetBPong = baseMN_ * baseK_ + ubOffsetAPong;
        uint64_t ubOffsetC = baseMN_ * baseK_ + ubOffsetBPong;
        uint64_t ubOffsetOut = baseMN_ * baseK_ + ubOffsetC;
        uint64_t ubOffsetA[] = {ubOffsetAPing, ubOffsetAPong};
        uint64_t ubOffsetB[] = {ubOffsetBPing, ubOffsetBPong};
        constexpr bool isReuse = true;

        // k累加地址 ubOffsetC 需要清0
        AscendC::Duplicate<float>(ubLocal_[ubOffsetC], 0, static_cast<int32_t>(baseMN_ * baseK_));
        for (uint64_t j = 0; j < loopK_; ++j) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(j & 0x1);
            if (m_ == 1) {
                CopyInA(aGlobal[j * baseK_], ubLocal_[ubOffsetA[j & 0x1]]);
            } else {
                CopyInA(bGlobal[j * baseK_], ubLocal_[ubOffsetA[j & 0x1]]);
            }
            uint64_t blockOffsetB = dataCopyMode_ ? j * shapeMN_ * baseK_ : j * baseK_;
            if (m_ == 1) {
                CopyInB(bGlobal[blockOffsetB], ubLocal_[ubOffsetB[j & 0x1]]);
            } else {
                CopyInB(aGlobal[blockOffsetB], ubLocal_[ubOffsetB[j & 0x1]]);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(j & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(j & 0x1);
            AivProcess(ubOffsetA[j & 0x1], ubOffsetB[j & 0x1], ubOffsetC);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(j & 0x1);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0x0);
        if (dataCopyMode_) {
            uint32_t shape[] = {static_cast<uint32_t>(baseK_), static_cast<uint32_t>(baseMN_)};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, isReuse>(
                ubLocal_[ubOffsetOut], ubLocal_[ubOffsetC], shape, true);
        } else {
            uint32_t shape[] = {static_cast<uint32_t>(baseMN_), static_cast<uint32_t>(baseK_)};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, isReuse>(
                ubLocal_[ubOffsetOut], ubLocal_[ubOffsetC], shape, true);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0x0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0x0);
        CopyOut(cGlobal, ubLocal_[ubOffsetOut]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0x0);
    }

private:
    constexpr static uint16_t DIMENSION_M = 0;
    constexpr static uint16_t DIMENSION_N = 1;
    constexpr static uint16_t DIMENSION_K = 2;
    constexpr static uint16_t DIMENSION_BLOCK_K = 0;
    constexpr static uint16_t DIMENSION_BLOCK_MN = 1;
    constexpr static uint16_t DIMENSION_BASE_K = 2;
    constexpr static uint16_t DIMENSION_BASE_MN = 3;
    AscendC::LocalTensor<float> ubLocal_{AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE};
};
} // namespace Block
} // namespace Gemm
} // namespace Cmct
#endif
