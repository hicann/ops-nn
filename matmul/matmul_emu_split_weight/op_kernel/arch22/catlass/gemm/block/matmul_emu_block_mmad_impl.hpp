/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_EMU_CATLASS_GEMM_BLOCK_BLOCK_MMAD_IMPL_HPP
#define MATMUL_EMU_CATLASS_GEMM_BLOCK_BLOCK_MMAD_IMPL_HPP

CATLASS_DEVICE
void CheckBiasL1Space()
{
    if constexpr (HAS_BIAS) {
        static constexpr uint32_t BIAS_BUF_SIZE = L0_TILE_N * sizeof(ElementAccumulator);
        static constexpr uint32_t L1BIAS_SIZE = L1_TILE_N * sizeof(ElementBias);
        static_assert(BIAS_BUF_SIZE <= ArchTag::BIAS_SIZE, "BIAS_BUF_SIZE exceeding the BT space! Reduce L0_TILE_N");
        static_assert(L1A_TILE_SIZE * L1A_STAGES + L1B_TILE_SIZE * L1B_STAGES + L1BIAS_SIZE <= ArchTag::L1_SIZE,
                      "L1TileShape exceeding the L1 space!");
    }
}

CATLASS_DEVICE
uint32_t AdjustML1Actual(uint32_t mBlockActual)
{
    uint32_t mL1Actual = mBlockActual;
    if constexpr (std::is_same_v<ArchTag, Arch::AtlasA2>) {
        if (mL1Actual == 1) {
            mL1Actual = 16;
        }
    }
    return mL1Actual;
}

CATLASS_DEVICE
void InitAicBuffers(Arch::Resource<ArchTag>& resource, uint32_t l1BufAddrStart)
{
    uint32_t l1AOffset = l1BufAddrStart;
    uint32_t l1BOffset = l1BufAddrStart + L1A_TILE_SIZE * L1A_STAGES;
    for (uint32_t i = 0; i < L1A_STAGES; i++) {
        l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + L1A_TILE_SIZE * i);
        l1AEventList[i] = i;
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
    }
    for (uint32_t i = 0; i < L1B_STAGES; i++) {
        l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + L1B_TILE_SIZE * i);
        l1BEventList[i] = i + L1A_STAGES;
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
    }
    for (uint32_t i = 0; i < L0A_STAGES; i++) {
        l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_TILE_SIZE * i);
        l0AEventList[i] = i;
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
    }
    for (uint32_t i = 0; i < L0B_STAGES; i++) {
        l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_TILE_SIZE * i);
        l0BEventList[i] = i + L0A_STAGES;
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
    }
    if constexpr (!ENABLE_UNIT_FLAG) {
        for (uint32_t i = 0; i < L0C_STAGES; i++) {
            l0CTensorList[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_TILE_SIZE * i);
            l0CEventList[i] = i;
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
        }
    } else {
        l0CTensorList[0] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
    }
    if constexpr (HAS_BIAS) {
        uint32_t l1BiasOffset = l1BOffset + L1B_TILE_SIZE * L1B_STAGES;
        l1BiasTensor = resource.l1Buf.template GetBufferByByte<uint8_t>(l1BiasOffset);
        l0BiasTensor = resource.btBuf.template GetBufferByByte<ElementAccumulator>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_STAGES + L1B_STAGES);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
    }
}

template <class Element, class CopyFn, class TensorL1, class TensorTile>
CATLASS_DEVICE void CopyGmToL1MaybeResident(CopyFn& copyFn, TensorL1 const& dst, TensorTile const& src,
                                            __gm__ typename AscendC::GlobalTensor<Element>::PrimType** lastAddr,
                                            MatrixCoord* lastCoord, uint32_t listId)
{
    if constexpr (ENABLE_L1_RESIDENT) {
        if (lastAddr[listId] != src.data().GetPhyAddr() || tla::get<0>(src.coord()) != lastCoord[listId].row() ||
            tla::get<1>(src.coord()) != lastCoord[listId].column()) {
            copyFn(dst, src);
            lastCoord[listId] = MatrixCoord{tla::get<0>(src.coord()), tla::get<1>(src.coord())};
            lastAddr[listId] = const_cast<__gm__ typename AscendC::GlobalTensor<Element>::PrimType*>(
                src.data().GetPhyAddr());
        }
    } else {
        copyFn(dst, src);
    }
}

template <class CopyGmToL1A, class CopyGmToL1B, class TensorA, class TensorB>
CATLASS_DEVICE void LoadFirstL1Tiles(CopyGmToL1A& copyGmToL1A, CopyGmToL1B& copyGmToL1B, TensorA& tensorA,
                                     TensorB& tensorB, uint32_t mBlockActual, uint32_t nBlockActual, uint32_t kL1Actual)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);
    auto tensorL1A = tla::MakeTensor(l1ATensorList[l1AListId], L1A_LAYOUT, Arch::PositionL1{});
    auto tensorTileA = GetTileA(tensorA, 0, 0, mBlockActual, kL1Actual);
    CopyGmToL1MaybeResident<ElementA>(copyGmToL1A, tensorL1A, tensorTileA, lastAddrA, lastCoordA, l1AListId);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);

    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListId]);
    auto tensorL1B = tla::MakeTensor(l1BTensorList[l1BListId], L1B_LAYOUT, Arch::PositionL1{});
    auto tensorTileB = GetTile(tensorB, tla::MakeCoord(0, 0), tla::MakeShape(kL1Actual, nBlockActual));
    CopyGmToL1MaybeResident<ElementB>(copyGmToL1B, tensorL1B, tensorTileB, lastAddrB, lastCoordB, l1BListId);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);
}

template <class TensorBias>
CATLASS_DEVICE void LoadFirstBias(TensorBias const& tensorBias)
{
    if constexpr (HAS_BIAS && !std::is_same_v<TensorBias, EmptyClass>) {
        using CopyGmToL1Bias = typename TileCopy::template CopyGmToL1Bias<TensorBias>;
        CopyGmToL1Bias copyGmToL1Bias;
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_STAGES + L1B_STAGES);
        auto l1Bias = l1BiasTensor.template ReinterpretCast<ElementBias>();
        auto tensorL1Bias = tla::MakeTensor(l1Bias, L1BIAS_LAYOUT, Arch::PositionL1{});
        copyGmToL1Bias(tensorL1Bias, tensorBias);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(L1A_STAGES + L1B_STAGES);
    }
}

template <class CopyGmToL1A, class CopyGmToL1B, class TensorA, class TensorB>
CATLASS_DEVICE uint32_t PreloadNextL1(CopyGmToL1A& copyGmToL1A, CopyGmToL1B& copyGmToL1B, TensorA& tensorA,
                                      TensorB& tensorB, uint32_t kL1Idx, uint32_t kL1Loop, uint32_t kBlockActual,
                                      uint32_t mBlockActual, uint32_t nBlockActual, uint32_t l1AListIdNext,
                                      uint32_t l1BListIdNext)
{
    uint32_t kL1IdxNext = kL1Idx + 1;
    uint32_t kL1ActualNext = (kL1IdxNext < kL1Loop - 1) ? L1_TILE_K : (kBlockActual - kL1IdxNext * L1_TILE_K);

    auto tensorL1A = tla::MakeTensor(l1ATensorList[l1AListIdNext], L1A_LAYOUT, Arch::PositionL1{});
    auto tensorL1B = tla::MakeTensor(l1BTensorList[l1BListIdNext], L1B_LAYOUT, Arch::PositionL1{});
    auto tensorTileA = GetTileA(tensorA, 0, kL1IdxNext * L1_TILE_K, mBlockActual, kL1ActualNext);
    auto tensorTileB = GetTile(tensorB, tla::MakeCoord(kL1IdxNext * L1_TILE_K, 0),
                               tla::MakeShape(kL1ActualNext, nBlockActual));

    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListIdNext]);
    CopyGmToL1MaybeResident<ElementA>(copyGmToL1A, tensorL1A, tensorTileA, lastAddrA, lastCoordA, l1AListIdNext);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListIdNext]);

    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListIdNext]);
    CopyGmToL1MaybeResident<ElementB>(copyGmToL1B, tensorL1B, tensorTileB, lastAddrB, lastCoordB, l1BListIdNext);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListIdNext]);
    return kL1ActualNext;
}

template <class TensorL0C, class TensorL0Bias, class TensorBias>
CATLASS_DEVICE void ComputeCurrentKL1(TensorL0C& tensorL0C, TensorL0Bias& tensorL0Bias, TensorBias const& tensorBias,
                                      uint32_t mL1Actual, uint32_t nL1Actual, uint32_t kL1Actual, uint32_t kL1Idx,
                                      uint32_t kL1Loop, uint32_t mL0Loop, uint32_t nL0Loop)
{
    auto tensorL1A = tla::MakeTensor(l1ATensorList[l1AListId], L1A_LAYOUT, Arch::PositionL1{});
    auto tensorL1B = tla::MakeTensor(l1BTensorList[l1BListId], L1B_LAYOUT, Arch::PositionL1{});
    uint32_t kL0Loop = CeilDiv<L0_TILE_K>(kL1Actual);
    for (int mL0Idx = 0; mL0Idx < mL0Loop; mL0Idx++) {
        uint32_t mL0Actual = (mL0Idx < mL0Loop - 1) ? L0_TILE_M : (mL1Actual - mL0Idx * L0_TILE_M);
        ComputeKL0Loop(tensorL0C, tensorL0Bias, tensorBias, tensorL1A, tensorL1B, mL0Idx, mL0Actual, mL0Loop, kL0Loop,
                       kL1Actual, kL1Idx, kL1Loop, nL1Actual, nL0Loop);
    }
}

template <class TensorL0C, class TensorL0Bias, class TensorBias, class TensorL1A, class TensorL1B>
CATLASS_DEVICE void ComputeKL0Loop(TensorL0C& tensorL0C, TensorL0Bias& tensorL0Bias, TensorBias const& tensorBias,
                                   TensorL1A& tensorL1A, TensorL1B& tensorL1B, int mL0Idx, uint32_t mL0Actual,
                                   uint32_t mL0Loop, uint32_t kL0Loop, uint32_t kL1Actual, uint32_t kL1Idx,
                                   uint32_t kL1Loop, uint32_t nL1Actual, uint32_t nL0Loop)
{
    for (int kL0Idx = 0; kL0Idx < kL0Loop; kL0Idx++) {
        uint32_t kL0Actual = (kL0Idx < kL0Loop - 1) ? L0_TILE_K : (kL1Actual - kL0Idx * L0_TILE_K);
        auto l0ATile = l0ATensorList[l0AListId];
        auto layoutAInL0 = tla::MakeLayout<ElementA, LayoutTagL0A>(mL0Actual, kL0Actual);
        auto tensorL0A = tla::MakeTensor(l0ATile, layoutAInL0, Arch::PositionL0A{});
        auto tensorTileL1A = GetTileA(tensorL1A, mL0Idx * L0_TILE_M, kL0Idx * L0_TILE_K, mL0Actual, kL0Actual);

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
        if ((mL0Idx == 0) && (kL0Idx == 0)) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);
        }
        copyL1ToL0A(tensorL0A, tensorTileL1A);
        if ((mL0Idx == mL0Loop - 1) && (kL0Idx == kL0Loop - 1)) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);
        }

        bool initC = ((kL1Idx == 0) && (kL0Idx == 0));
        ComputeNL0Loop(tensorL0C, tensorL0Bias, tensorBias, tensorL0A, tensorL1B, mL0Idx, mL0Actual, mL0Loop, kL0Idx,
                       kL0Actual, kL0Loop, kL1Idx, kL1Loop, nL1Actual, nL0Loop, initC);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
        l0AListId = (l0AListId + 1 < L0A_STAGES) ? (l0AListId + 1) : 0;
    }
}

template <class TensorL0C, class TensorL0Bias, class TensorBias, class TensorL0A, class TensorL1B>
CATLASS_DEVICE void ComputeNL0Loop(TensorL0C& tensorL0C, TensorL0Bias& tensorL0Bias, TensorBias const& tensorBias,
                                   TensorL0A& tensorL0A, TensorL1B& tensorL1B, int mL0Idx, uint32_t mL0Actual,
                                   uint32_t mL0Loop, int kL0Idx, uint32_t kL0Actual, uint32_t kL0Loop, uint32_t kL1Idx,
                                   uint32_t kL1Loop, uint32_t nL1Actual, uint32_t nL0Loop, bool initC)
{
    (void)tensorBias;
    for (int nL0Idx = 0; nL0Idx < nL0Loop; nL0Idx++) {
        uint32_t nL0Actual = (nL0Idx < nL0Loop - 1) ? L0_TILE_N : (nL1Actual - nL0Idx * L0_TILE_N);
        auto l0BTile = l0BTensorList[l0BListId];
        auto layoutBInL0 = tla::MakeLayout<ElementB, LayoutTagL0B>(kL0Actual, nL0Actual);
        auto tensorL0B = tla::MakeTensor(l0BTile, layoutBInL0, Arch::PositionL0B{});
        auto tensorTileL1B = GetTile(tensorL1B, tla::MakeCoord(kL0Idx * L0_TILE_K, nL0Idx * L0_TILE_N),
                                     tla::MakeShape(kL0Actual, nL0Actual));

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
        if ((mL0Idx == 0) && (kL0Idx == 0) && (nL0Idx == 0)) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);
        }
        copyL1ToL0B(tensorL0B, tensorTileL1B);
        if ((mL0Idx == mL0Loop - 1) && (kL0Idx == kL0Loop - 1) && (nL0Idx == nL0Loop - 1)) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListId]);
        }

        LoadBiasToL0<TensorBias>(tensorL0Bias, nL0Idx, nL0Actual, nL0Loop, initC);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0CEventList[l0CListId]);
        auto tensorTileL0C = GetTile(tensorL0C, tla::MakeCoord(mL0Idx * L0_TILE_M, nL0Idx * L0_TILE_N),
                                     tla::MakeShape(mL0Actual, nL0Actual));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0CEventList[l0CListId]);

        uint8_t unitFlag = 0b00;
        if constexpr (ENABLE_UNIT_FLAG) {
            if ((kL1Idx == kL1Loop - 1) && (mL0Idx == mL0Loop - 1) && (kL0Idx == kL0Loop - 1) &&
                (nL0Idx == nL0Loop - 1)) {
                unitFlag = 0b11;
            } else {
                unitFlag = 0b10;
            }
        }
        IssueMmad<TensorBias>(tensorTileL0C, tensorL0A, tensorL0B, tensorL0Bias, mL0Actual, nL0Actual, kL0Actual, initC,
                              unitFlag);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
        l0BListId = (l0BListId + 1 < L0B_STAGES) ? (l0BListId + 1) : 0;
    }
}

template <class TensorBias, class TensorL0Bias>
CATLASS_DEVICE void LoadBiasToL0(TensorL0Bias& tensorL0Bias, int nL0Idx, uint32_t nL0Actual, uint32_t nL0Loop,
                                 bool initC)
{
    if constexpr (HAS_BIAS && !std::is_same_v<TensorBias, EmptyClass>) {
        if (initC) {
            if (nL0Idx == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(L1A_STAGES + L1B_STAGES);
            }
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
            auto l1Bias = l1BiasTensor.template ReinterpretCast<ElementBias>();
            auto tensorL1Bias = tla::MakeTensor(l1Bias, L1BIAS_LAYOUT, Arch::PositionL1{});
            auto tensorTileL1Bias = GetTile(tensorL1Bias, tla::MakeCoord(nL0Idx * L0_TILE_N),
                                            tla::MakeShape(nL0Actual));
            copyL1ToBT(tensorL0Bias, tensorTileL1Bias);
            if (nL0Idx == nL0Loop - 1) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_STAGES + L1B_STAGES);
            }
        }
    }
}

template <class TensorBias, class TensorTileL0C, class TensorL0A, class TensorL0B, class TensorL0Bias>
CATLASS_DEVICE void IssueMmad(TensorTileL0C const& tensorTileL0C, TensorL0A const& tensorL0A,
                              TensorL0B const& tensorL0B, TensorL0Bias const& tensorL0Bias, uint32_t mL0Actual,
                              uint32_t nL0Actual, uint32_t kL0Actual, bool initC, uint8_t unitFlag)
{
    if constexpr (HAS_BIAS && !std::is_same_v<TensorBias, EmptyClass>) {
        if (initC) {
            tileMmad(tensorTileL0C, tensorL0A, tensorL0B, tensorL0Bias, mL0Actual, nL0Actual, kL0Actual, initC,
                     unitFlag);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
        } else {
            tileMmad(tensorTileL0C, tensorL0A, tensorL0B, mL0Actual, nL0Actual, kL0Actual, initC, unitFlag);
        }
    } else {
        tileMmad(tensorTileL0C, tensorL0A, tensorL0B, mL0Actual, nL0Actual, kL0Actual, initC, unitFlag);
    }
}

template <class CopyL0CToDst, class TensorC, class TensorL0C>
CATLASS_DEVICE void StoreBlockOut(CopyL0CToDst& copyL0CToDst, TensorC& tensorC, TensorL0C& tensorL0C)
{
    if constexpr (!ENABLE_UNIT_FLAG) {
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
        copyL0CToDst(tensorC, tensorL0C);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
        l0CListId = (l0CListId + 1 < L0C_STAGES) ? (l0CListId + 1) : 0;
    } else {
        copyL0CToDst(tensorC, tensorL0C, 0b11);
    }
}

#endif // MATMUL_EMU_CATLASS_GEMM_BLOCK_BLOCK_MMAD_IMPL_HPP
