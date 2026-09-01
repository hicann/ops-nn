/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_EMU_CATLASS_GEMM_BLOCK_BLOCK_MMAD_HPP
#define MATMUL_EMU_CATLASS_GEMM_BLOCK_BLOCK_MMAD_HPP

#include "../../matmul_emu_catlass.hpp"
#include "../matmul_emu_dispatch_policy.hpp"
#include "../tile/matmul_emu_tile_copy.hpp"
#include "../../matmul_emu_gemm_coord.hpp"
#include "../../../tla/matmul_emu_layout.hpp"

namespace Catlass::Gemm::Block {

template <
    class DispatchPolicy, class L1TileShape, class L0TileShape, class ElementA, class ElementB, class ElementC,
    class ElementBias = void,
    class TileCopy = Gemm::Tile::PackedTileCopyTla<typename DispatchPolicy::ArchTag, ElementA, layout::RowMajor,
                                                   ElementB, layout::RowMajor, ElementC, layout::RowMajor, ElementBias>,
    class TileMmad = Gemm::Tile::TileMmadTla<typename DispatchPolicy::ArchTag, ElementA,
                                             typename TileCopy::LayoutTagL1A> >
struct BlockMmadTla {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadTla is not implemented for this DispatchPolicy");
};

template <class ArchTag_, bool ENABLE_UNIT_FLAG_, bool USE_HF32_MODE_, uint32_t L0C_STAGES_, bool ENABLE_L1_RESIDENT_,
          uint32_t L1A_STAGES_, uint32_t L1B_STAGES_, uint32_t L0A_STAGES_, uint32_t L0B_STAGES_, class L1TileShape_,
          class L0TileShape_, class ElementA_, class ElementB_, class ElementC_, class ElementBias_, class TileCopy_,
          class TileMmad_>
struct BlockMmadTla<MmadPingpong<ArchTag_, ENABLE_UNIT_FLAG_, USE_HF32_MODE_, L0C_STAGES_, ENABLE_L1_RESIDENT_,
                                 L1A_STAGES_, L1B_STAGES_, L0A_STAGES_, L0B_STAGES_>,
                    L1TileShape_, L0TileShape_, ElementA_, ElementB_, ElementC_, ElementBias_, TileCopy_, TileMmad_> {
public:
    // Type Aliases
    using DispatchPolicy = MmadPingpong<ArchTag_, ENABLE_UNIT_FLAG_, USE_HF32_MODE_, L0C_STAGES_, ENABLE_L1_RESIDENT_,
                                        L1A_STAGES_, L1B_STAGES_, L0A_STAGES_, L0B_STAGES_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = ElementA_;
    using LayoutA = typename TileCopy::LayoutA;
    using ElementB = ElementB_;
    using LayoutB = typename TileCopy::LayoutB;
    using ElementC = ElementC_;
    using LayoutC = typename TileCopy::LayoutC;
    using ElementBias = ElementBias_;

    using TileMmad = TileMmad_;

    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using CopyL1ToBT = typename TileCopy::CopyL1ToBT;

    using ElementAccumulator = typename TileCopy::ElementAccumulator;

    static constexpr bool HAS_BIAS = TileCopy::HAS_BIAS;

    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    using L1AAlignHelper = typename TileCopy_::L1AAlignHelper;
    using L1BAlignHelper = typename TileCopy_::L1BAlignHelper;

    static_assert(tla::is_tuple<L1TileShape>::value && tla::is_static<L1TileShape>::value,
                  "L1TileShape must be tla::tuple and static!");
    static_assert(tla::is_tuple<L0TileShape>::value && tla::is_static<L0TileShape>::value,
                  "L0TileShape must be tla::tuple and static!");

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr bool USE_HF32_MODE = DispatchPolicy::USE_HF32_MODE;
    static constexpr bool ENABLE_L1_RESIDENT = DispatchPolicy::ENABLE_L1_RESIDENT;
    static constexpr uint32_t L1A_STAGES = DispatchPolicy::L1A_STAGES;
    static constexpr uint32_t L1B_STAGES = DispatchPolicy::L1B_STAGES;
    static constexpr uint32_t L0A_STAGES = DispatchPolicy::L0A_STAGES;
    static constexpr uint32_t L0B_STAGES = DispatchPolicy::L0B_STAGES;
    static constexpr uint32_t L0C_STAGES = DispatchPolicy::L0C_STAGES;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});

    // L1 tile size
    static constexpr uint32_t L1A_TILE_SIZE = L1_TILE_M * L1_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L1B_TILE_SIZE = L1_TILE_N * L1_TILE_K * sizeof(ElementB);
    // L0 tile size
    static constexpr uint32_t L0A_TILE_SIZE = L0_TILE_M * L0_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0_TILE_K * L0_TILE_N * sizeof(ElementB);
    static constexpr uint32_t L0C_TILE_SIZE = L1_TILE_M * L1_TILE_N * sizeof(ElementAccumulator);

    // Check HF32_MODE
    static_assert(!USE_HF32_MODE ||
                      (USE_HF32_MODE && std::is_same_v<ElementA, float> && std::is_same_v<ElementB, float>),
                  "HF32 MODE only supports in float!");

    // Check L0C_STAGES
    static_assert(!(ENABLE_UNIT_FLAG && L0C_STAGES != 1), "L0C_STAGES must be 1 when UnitFlag is true!");

    // Check LayoutC
    static_assert(tla::detail::isRowMajor<LayoutC>::value ||
                      ((std::is_same_v<ElementC, half> || std::is_same_v<ElementC, bfloat16_t> ||
                        std::is_same_v<ElementC, float>) &&
                       tla::detail::iszN<ElementC, LayoutC>::value),
                  "LayoutC only supports zN in half or bfloat16 or float, RowMajor in all dtype yet!");

    // Check L1TileShape
    static_assert(L1A_TILE_SIZE * L1A_STAGES + L1B_TILE_SIZE * L1B_STAGES <= ArchTag::L1_SIZE,
                  "L1TileShape exceeding the L1 space!");

    // Check L0TileShape
    static_assert(L0A_TILE_SIZE * L0A_STAGES <= ArchTag::L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert(L0B_TILE_SIZE * L0B_STAGES <= ArchTag::L0B_SIZE, "L0TileShape exceeding the L0B space!");
    static_assert(L0C_TILE_SIZE * L0C_STAGES <= ArchTag::L0C_SIZE, "L0TileShape exceeding the L0C space!");

    static constexpr uint32_t _32B = 32 * 8; // in bits
    static_assert(L1_TILE_M == L0_TILE_M && L1_TILE_N == L0_TILE_N,
                  "The situation where the basic blocks of L1 and L0 differ on the m and n axes is not supported yet");
    static_assert(L0_TILE_K <= L1_TILE_K, "L0TileShape::K cannot exceed L1TileShape::K");
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 2201)
    static_assert(L1_TILE_M * SizeOfBits<ElementA>::value % _32B == 0, "L1TileShape::M must be 32B aligned.");
    static_assert(L1_TILE_K * SizeOfBits<ElementA>::value % _32B == 0, "L1TileShape::K must be 32B aligned.");
    static_assert(L1_TILE_K * SizeOfBits<ElementB>::value % _32B == 0, "L1TileShape::K must be 32B aligned.");
    static_assert(L1_TILE_N * SizeOfBits<ElementB>::value % _32B == 0, "L1TileShape::N must be 32B aligned.");
    static_assert(L0_TILE_K * SizeOfBits<ElementB>::value % _32B == 0, "L0TileShape::K must be 32B aligned.");
#endif

    static_assert((!HAS_BIAS && (L1A_STAGES + L1B_STAGES) <= 8) || (HAS_BIAS && (L1A_STAGES + L1B_STAGES) <= 7),
                  "L1 Buffer overflow: Exceeds the supported range of EVENT(0~7)");

    static_assert((!HAS_BIAS && (L0A_STAGES + L0B_STAGES) <= 8) || (HAS_BIAS && (L0A_STAGES + L0B_STAGES) <= 7),
                  "L0 Buffer overflow: Exceeds the supported range of EVENT_ID(0~7)");

    static constexpr auto L1A_LAYOUT = tla::MakeLayout<ElementA, LayoutTagL1A>(tla::Int<L1_TILE_M>{},
                                                                               tla::Int<L1_TILE_K>{});
    static constexpr auto L1B_LAYOUT = tla::MakeLayout<ElementB, LayoutTagL1B>(tla::Int<L1_TILE_K>{},
                                                                               tla::Int<L1_TILE_N>{});
    static constexpr auto L1BIAS_LAYOUT = tla::MakeLayout(tla::Int<L1_TILE_N>{});
    static constexpr auto L0BIAS_LAYOUT = tla::MakeLayout(tla::Int<L0_TILE_N>{});

    // When enabling L1 resident mode, restore the pointer and coordinates that record the last state
    // to the initial state. if two blockmmad instances need to be consecutively invoked at the kernel layer,
    // RestoreStatus() must be inserted between them.
    CATLASS_DEVICE
    void RestoreStatus()
    {
        for (int i = 0; i < L1A_STAGES; ++i) {
            lastAddrA[i] = nullptr;
            lastCoordA[i] = MatrixCoord{0U, 0U};
        }
        for (int i = 0; i < L1B_STAGES; ++i) {
            lastAddrB[i] = nullptr;
            lastCoordB[i] = MatrixCoord{0U, 0U};
        }
    }

    /// Construct
    CATLASS_DEVICE
    BlockMmadTla(Arch::Resource<ArchTag>& resource, uint32_t l1BufAddrStart = 0)
    {
        if ASCEND_IS_AIC {
            if constexpr (USE_HF32_MODE) {
                AscendC::SetHF32Mode(true);
            } else {
                AscendC::SetHF32Mode(false);
            }
            if constexpr (ENABLE_UNIT_FLAG && tla::detail::isRowMajor<LayoutC>::value) {
                AscendC::SetMMLayoutTransform(true);
            }
            InitAicBuffers(resource, l1BufAddrStart);
            if constexpr (ENABLE_L1_RESIDENT) {
                RestoreStatus();
            }
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockMmadTla()
    {
        if ASCEND_IS_AIC {
            if constexpr (USE_HF32_MODE) {
                AscendC::SetHF32Mode(false);
            }
            if constexpr (ENABLE_UNIT_FLAG && tla::detail::isRowMajor<LayoutC>::value) {
                AscendC::SetMMLayoutTransform(false);
            }
            for (uint32_t i = 0; i < L1A_STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            }
            for (uint32_t i = 0; i < L1B_STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            }
            for (uint32_t i = 0; i < L0A_STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            }
            for (uint32_t i = 0; i < L0B_STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
            }
            if constexpr (!ENABLE_UNIT_FLAG) {
                for (uint32_t i = 0; i < L0C_STAGES; i++) {
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
                }
            }
            if constexpr (HAS_BIAS) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_STAGES + L1B_STAGES);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
            }
        }
    }

    /// Perform a block-scoped matrix multiply-accumulate
    template <class TensorA, class TensorB, class TensorC, class TensorBias = EmptyClass>
    CATLASS_DEVICE void operator()(TensorA& tensorA, TensorB& tensorB, TensorC& tensorC, GemmCoord const& actualShape,
                                   TensorBias const& tensorBias = {})
    {
        CheckBiasL1Space();
        using CopyGmToL1A = typename TileCopy_::template CopyGmToL1A<TensorA>;
        using CopyGmToL1B = typename TileCopy_::template CopyGmToL1B<TensorB>;
        CopyGmToL1A copyGmToL1A;
        CopyGmToL1B copyGmToL1B;
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 2201)
        using CopyL0CToGm = typename TileCopy_::template CopyL0CToGm<TensorC>;
        CopyL0CToGm copyL0CToDst;
#endif

        uint32_t mBlockActual = actualShape.m();
        uint32_t kBlockActual = actualShape.k();
        uint32_t nBlockActual = actualShape.n();
        uint32_t mL1Actual = AdjustML1Actual(mBlockActual);
        uint32_t nL1Actual = nBlockActual;

        auto layoutInL0C = tla::MakeLayoutL0C(mL1Actual, nL1Actual);
        auto tensorL0C = tla::MakeTensor(l0CTensorList[l0CListId], layoutInL0C, Arch::PositionL0C{});
        auto tensorL0Bias = tla::MakeTensor(l0BiasTensor, L0BIAS_LAYOUT, Arch::PositionBias{});

        uint32_t kL1Actual = Min(kBlockActual, L1_TILE_K);
        LoadFirstL1Tiles(copyGmToL1A, copyGmToL1B, tensorA, tensorB, mBlockActual, nBlockActual, kL1Actual);
        LoadFirstBias(tensorBias);
        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
        }

        uint32_t mL0Loop = CeilDiv<L0_TILE_M>(mL1Actual);
        uint32_t nL0Loop = CeilDiv<L0_TILE_N>(nL1Actual);
        uint32_t kL1Loop = CeilDiv<L1_TILE_K>(kBlockActual);
        for (uint32_t kL1Idx = 0; kL1Idx < kL1Loop; kL1Idx++) {
            uint32_t l1AListIdNext = (l1AListId + 1 < L1A_STAGES) ? (l1AListId + 1) : 0;
            uint32_t l1BListIdNext = (l1BListId + 1 < L1B_STAGES) ? (l1BListId + 1) : 0;
            uint32_t kL1ActualNext = 0;
            if (kL1Idx < kL1Loop - 1) {
                kL1ActualNext = PreloadNextL1(copyGmToL1A, copyGmToL1B, tensorA, tensorB, kL1Idx, kL1Loop, kBlockActual,
                                              mBlockActual, nBlockActual, l1AListIdNext, l1BListIdNext);
            }
            ComputeCurrentKL1(tensorL0C, tensorL0Bias, tensorBias, mL1Actual, nL1Actual, kL1Actual, kL1Idx, kL1Loop,
                              mL0Loop, nL0Loop);
            l1AListId = l1AListIdNext;
            l1BListId = l1BListIdNext;
            kL1Actual = kL1ActualNext;
        }

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 2201)
        StoreBlockOut(copyL0CToDst, tensorC, tensorL0C);
#endif
    }

protected:
#include "matmul_emu_block_mmad_impl.hpp"

    template <class TensorA>
    CATLASS_DEVICE auto GetTileA(TensorA& tensorA, uint32_t mIndex, uint32_t kIndex, uint32_t mSize, uint32_t kSize)
    {
        if constexpr (tla::detail::isVector<LayoutA>::value) {
            return GetTile(tensorA, tla::MakeCoord(kIndex), tla::MakeShape(kSize));
        } else {
            return GetTile(tensorA, tla::MakeCoord(mIndex, kIndex), tla::MakeShape(mSize, kSize));
        }
    }

    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> l1ATensorList[L1A_STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList[L1B_STAGES];
    AscendC::LocalTensor<ElementA> l0ATensorList[L0A_STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList[L0B_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensorList[L0C_STAGES];
    AscendC::LocalTensor<uint8_t> l1BiasTensor;
    AscendC::LocalTensor<ElementAccumulator> l0BiasTensor;

    // Multi-stage event id list
    int32_t l1AEventList[L1A_STAGES];
    int32_t l1BEventList[L1B_STAGES];
    int32_t l0AEventList[L0A_STAGES];
    int32_t l0BEventList[L0B_STAGES];
    int32_t l0CEventList[L0C_STAGES];

    __gm__ typename AscendC::GlobalTensor<ElementA>::PrimType* lastAddrA[L1A_STAGES];
    __gm__ typename AscendC::GlobalTensor<ElementB>::PrimType* lastAddrB[L1B_STAGES];
    MatrixCoord lastCoordA[L1A_STAGES];
    MatrixCoord lastCoordB[L1B_STAGES];

    // The id of current stage
    uint32_t l1AListId{0};
    uint32_t l1BListId{0};
    uint32_t l0AListId{0};
    uint32_t l0BListId{0};
    uint32_t l0CListId{0};

    TileMmad tileMmad;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL1ToBT copyL1ToBT;
};

} // namespace Catlass::Gemm::Block

#endif // MATMUL_EMU_CATLASS_GEMM_BLOCK_BLOCK_MMAD_HPP
