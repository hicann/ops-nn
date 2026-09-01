/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_DECOMPOSITION_GM_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_DECOMPOSITION_GM_HPP

#include "../../matmul_emu_catlass.hpp"
#include "../../arch/matmul_emu_arch.hpp"
#include "../../gemm/matmul_emu_gemm_type.hpp"
#include "../../matmul_emu_gemm_coord.hpp"
#include "../../../tla/matmul_emu_layout.hpp"

namespace Catlass::Epilogue::Tile {

template <class ArchTag_, class ComputeType_, uint32_t COMPUTE_LENGTH_>
struct TileElemWiseMuls {
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;
    static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;

    CATLASS_DEVICE
    TileElemWiseMuls() {}

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<ElementCompute> dstLocal, AscendC::LocalTensor<ElementCompute> srcTensor,
                    ElementCompute scalar)
    {
        AscendC::Muls(dstLocal, srcTensor, scalar, COMPUTE_LENGTH);
    }
};

template <class ArchTag_, class ComputeType_, uint32_t COMPUTE_LENGTH_>
struct TileElemWiseAdd {
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;
    static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;

    CATLASS_DEVICE
    TileElemWiseAdd() {}

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<ElementCompute> const& ubOut,
                    AscendC::LocalTensor<ElementCompute> const& ubIn0,
                    AscendC::LocalTensor<ElementCompute> const& ubIn1)
    {
        AscendC::Add(ubOut, ubIn0, ubIn1, COMPUTE_LENGTH);
    }
};

template <class ArchTag, class TensorSrc, class TensorDst, class Enable = void>
struct CopyGm2UbTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported CopyGm2UbTla, can not find the specialization.");
};

template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyGm2UbTla<
    Arch::AtlasA2, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::VECCALC>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::isRowMajor<LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(ElementSrc);

    CATLASS_DEVICE
    CopyGm2UbTla() = default;

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        AscendC::DataCopyExtParams dataCopyParams(
            tla::get<0>(srcTensor.originShape()), tla::get<1>(srcTensor.originShape()) * sizeof(ElementSrc),
            (tla::get<0>(srcTensor.stride()) - tla::get<1>(srcTensor.originShape())) * sizeof(ElementSrc),
            (tla::get<0>(dstTensor.stride()) - tla::get<1>(srcTensor.originShape())) / ELE_NUM_PER_BLK, 0);
        AscendC::DataCopyPadExtParams<ElementSrc> padParams(false, 0, 0, 0);
        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        AscendC::DataCopyPad(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], dataCopyParams, padParams);
    };
};

template <class ArchTag, class TensorSrc, class TensorDst, class Enable = void>
struct CopyUb2GmTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported CopyUb2GmTla, can not find the specialization.");
};

template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct CopyUb2GmTla<
    Arch::AtlasA2, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::GlobalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::GM>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::isRowMajor<LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);

    CATLASS_DEVICE
    CopyUb2GmTla() = default;

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        AscendC::DataCopyExtParams dataCopyParams(
            tla::get<0>(dstTensor.originShape()), tla::get<1>(dstTensor.originShape()) * sizeof(ElementSrc),
            (tla::get<0>(srcTensor.stride()) - tla::get<1>(dstTensor.originShape())) / ELE_NUM_PER_C0,
            (tla::get<0>(dstTensor.stride()) - tla::get<1>(dstTensor.originShape())) * sizeof(ElementSrc), 0);
        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        AscendC::DataCopyPad(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], dataCopyParams);
    };
};

} // namespace Catlass::Epilogue::Tile

namespace Catlass::Epilogue::Block {

template <class ArchTag_, class TileM_, class TileN_, class ElementC_, class LayoutC_, class LayoutTagA_ = void,
          class LayoutTagB_ = void>
struct BlockEpilogueDecompositionGm {
    static constexpr bool USE_GM_WORKSPACE = true;
    using ArchTag = ArchTag_;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;

    static constexpr uint32_t L1_TILE_M = TileM_::value;
    static constexpr uint32_t L1_TILE_N = TileN_::value;
    // Default scale follows the aclnnMatmulEmuSplitWeight constraint (wLowScale = 1/256);
    // may be overridden per-call through Params::scale.
    static constexpr float DEFAULT_SCALE = 1.0f / 256.0f;

    // Each AIV subcore (dual AIV) physically holds half of the tile rows.
    static constexpr uint32_t SUB_M = L1_TILE_M / 2u;
    static constexpr uint32_t STRIDE_C = RoundUp<Catlass::BYTE_PER_C0>(L1_TILE_N);
    static constexpr uint32_t SUBBLOCK_ELEMS = SUB_M * STRIDE_C;
    static constexpr uint32_t PER_ACCUM_BYTES = SUBBLOCK_ELEMS * sizeof(ElementC);

    // Chunk length for the in-place vector fusion; must divide the full subblock.
    static constexpr uint32_t COMPUTE_LENGTH = 8192u;

    static_assert(COMPUTE_LENGTH % Catlass::BYTE_PER_C0 == 0, "COMPUTE_LENGTH must be divisible by BYTE_PER_C0");
    static_assert(SUBBLOCK_ELEMS % COMPUTE_LENGTH == 0, "Full L1 subblock elems must be a multiple of COMPUTE_LENGTH");
    // UB: two full-size GM staging accumulators (ubY0 | ubY1), in-place fusion (no ubOut).
    // Full-size staging lets the whole Y0 subblock be pulled GM->UB while the AIC runs the
    // second mmad (Y1): 2 * SUBBLOCK_ELEMS * 4 = 128KB for <128,256> on 192KB UB.
    static_assert(2u * PER_ACCUM_BYTES <= ArchTag::UB_SIZE,
                  "UB budget overflow: 2 full-size accumulators exceed UB_SIZE. Fallback to smaller L1 tile.");

    using ComputeType = Catlass::Gemm::GemmType<ElementC, Catlass::layout::RowMajor>;
    using TileMuls = Catlass::Epilogue::Tile::TileElemWiseMuls<ArchTag, ComputeType, COMPUTE_LENGTH>;
    using TileAdd = Catlass::Epilogue::Tile::TileElemWiseAdd<ArchTag, ComputeType, COMPUTE_LENGTH>;

    struct Params {
        GM_ADDR ptrY;
        LayoutC layoutY;
        float scale;

        CATLASS_HOST_DEVICE
        Params() : scale(DEFAULT_SCALE) {}

        CATLASS_HOST_DEVICE
        Params(GM_ADDR ptrY_, LayoutC const& layoutY_, float scale_ = DEFAULT_SCALE)
            : ptrY(ptrY_), layoutY(layoutY_), scale(scale_)
        {}
    };

    CATLASS_DEVICE
    BlockEpilogueDecompositionGm(Catlass::Arch::Resource<ArchTag>& resource, Params const& params)
        : params_(params), resource_(resource)
    {
        if ASCEND_IS_AIV {
            // Prime the event semaphores: no pending V, MTE3, nor MTE3->MTE2 dependency yet.
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

    CATLASS_DEVICE
    ~BlockEpilogueDecompositionGm()
    {
        if ASCEND_IS_AIV {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

    // Phase 1 (MTE2): Y0 GM->UB. Issued as soon as the AIC stages Y0 into the GM workspace
    // (flagY0Ready), so this copy overlaps with the AIC's second mmad (Y1).
    CATLASS_DEVICE
    void LoadY0(Catlass::GemmCoord const& actualBlockShapeMNK, AscendC::GlobalTensor<ElementC> const& gmBlockY0,
                LayoutC const& layoutBlock)
    {
        SubblockLayout sub = MakeSubblockLayout(actualBlockShapeMNK, layoutBlock);
        if (!sub.valid) {
            return;
        }
        AscendC::LocalTensor<ElementC> ubY0 = resource_.ubBuf.template GetBufferByByte<ElementC>(0);

        // Gate MTE2: the previous tile's UB->GM writeback (MTE3) must be complete and the
        // in-place vector fusion must be done before reusing ubY0.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);

        auto gmY0Layout = tla::MakeLayout(tla::MakeShape(sub.rows, sub.cols),
                                          tla::MakeStride(sub.gmRowStride, tla::Int<1>{}));
        auto ubY0Layout = tla::MakeLayout(tla::MakeShape(sub.rows, sub.cols),
                                          tla::MakeStride(sub.strideC, tla::Int<1>{}));
        auto gmY0Tile = tla::MakeTensor(gmBlockY0[sub.subblockRow * sub.gmRowStride], gmY0Layout,
                                        Catlass::Arch::PositionGM{});
        auto ubY0Tile = tla::MakeTensor(ubY0, ubY0Layout, Catlass::Arch::PositionUB{});

        Catlass::Epilogue::Tile::CopyGm2UbTla<ArchTag, decltype(gmY0Tile), decltype(ubY0Tile)> copyGm2Ub;
        copyGm2Ub(ubY0Tile, gmY0Tile);
    }

    // Phase 2 (MTE2): Y1 GM->UB.
    CATLASS_DEVICE
    void LoadY1(Catlass::GemmCoord const& actualBlockShapeMNK, AscendC::GlobalTensor<ElementC> const& gmBlockY1,
                LayoutC const& layoutBlock)
    {
        SubblockLayout sub = MakeSubblockLayout(actualBlockShapeMNK, layoutBlock);
        if (!sub.valid) {
            return;
        }
        AscendC::LocalTensor<ElementC> ubY1 = resource_.ubBuf.template GetBufferByByte<ElementC>(PER_ACCUM_BYTES);

        auto gmY1Layout = tla::MakeLayout(tla::MakeShape(sub.rows, sub.cols),
                                          tla::MakeStride(sub.gmRowStride, tla::Int<1>{}));
        auto ubY1Layout = tla::MakeLayout(tla::MakeShape(sub.rows, sub.cols),
                                          tla::MakeStride(sub.strideC, tla::Int<1>{}));
        auto gmY1Tile = tla::MakeTensor(gmBlockY1[sub.subblockRow * sub.gmRowStride], gmY1Layout,
                                        Catlass::Arch::PositionGM{});
        auto ubY1Tile = tla::MakeTensor(ubY1, ubY1Layout, Catlass::Arch::PositionUB{});

        Catlass::Epilogue::Tile::CopyGm2UbTla<ArchTag, decltype(gmY1Tile), decltype(ubY1Tile)> copyGm2Ub;
        copyGm2Ub(ubY1Tile, gmY1Tile);
        // Both Y0/Y1 are now in UB: release the vector pipe.
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    }

    // Phase 3 (V + MTE3): in-place Y = Y0 + scale * Y1 on the padded row-major buffers,
    // then UB->GM writeback of the (rows x cols) subblock.
    CATLASS_DEVICE
    void FuseAndStore(Catlass::GemmCoord const& blockShapeMNK, Catlass::GemmCoord const& blockCoordMNK,
                      Catlass::GemmCoord const& actualBlockShapeMNK, LayoutC const& layoutBlock)
    {
        using Catlass::MatrixCoord;

        SubblockLayout sub = MakeSubblockLayout(actualBlockShapeMNK, layoutBlock);
        if (!sub.valid) {
            return;
        }

        AscendC::LocalTensor<ElementC> ubY0 = resource_.ubBuf.template GetBufferByByte<ElementC>(0);
        AscendC::LocalTensor<ElementC> ubY1 = resource_.ubBuf.template GetBufferByByte<ElementC>(PER_ACCUM_BYTES);

        MatrixCoord blockShape = blockShapeMNK.GetCoordMN();
        MatrixCoord blockCoord = blockCoordMNK.GetCoordMN();
        MatrixCoord blockOffset = blockCoord * blockShape;

        AscendC::GlobalTensor<ElementC> gmY;
        gmY.SetGlobalBuffer((__gm__ ElementC*)params_.ptrY);
        auto tensorY = tla::MakeTensor(gmY, params_.layoutY, Catlass::Arch::PositionGM{});
        auto gmTileY = GetTile(tensorY, tla::MakeCoord(blockOffset.row() + sub.subblockRow, blockOffset.column()),
                               tla::MakeShape(sub.rows, sub.cols));

        TileMuls tileMuls;
        TileAdd tileAdd;

        // Wait for both GM->UB loads (MTE2) and the previous UB->GM writeback (MTE3).
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        // In-place fusion: Y1 = scale * Y1, Y0 = Y0 + Y1. Vector tiles step by COMPUTE_LENGTH
        // (may process padding past totalElems, still inside the full-size subblock buffers).
        for (uint32_t off = 0; off < sub.totalElems; off += COMPUTE_LENGTH) {
            tileMuls(ubY1[off], ubY1[off], params_.scale);
            tileAdd(ubY0[off], ubY0[off], ubY1[off]);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        auto ubY0Layout = tla::MakeLayout(tla::MakeShape(sub.rows, sub.cols),
                                          tla::MakeStride(sub.strideC, tla::Int<1>{}));
        auto ubY0Tile = tla::MakeTensor(ubY0, ubY0Layout, Catlass::Arch::PositionUB{});

        Catlass::Epilogue::Tile::CopyUb2GmTla<ArchTag, decltype(ubY0Tile), decltype(gmTileY)> copyUb2Gm;
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        copyUb2Gm(gmTileY, ubY0Tile);
        // Writeback done: release MTE3->V (next tile's fusion) and MTE3->MTE2 (next tile's load).
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

private:
    struct SubblockLayout {
        uint32_t rows = 0;
        uint32_t cols = 0;
        uint32_t strideC = 0;
        uint32_t totalElems = 0;
        uint32_t gmRowStride = 0;
        uint32_t subblockRow = 0;
        bool valid = false;
    };

    CATLASS_DEVICE SubblockLayout MakeSubblockLayout(Catlass::GemmCoord const& actualBlockShapeMNK,
                                                     LayoutC const& layoutBlock) const
    {
        using Catlass::MatrixCoord;
        MatrixCoord actualBlockShape = actualBlockShapeMNK.GetCoordMN();
        uint32_t subBlockNum = static_cast<uint32_t>(AscendC::GetSubBlockNum());
        MatrixCoord subblockShape{CeilDiv(actualBlockShape.row(), subBlockNum), actualBlockShape.column()};
        MatrixCoord subblockCoord{static_cast<uint32_t>(AscendC::GetSubBlockIdx()), 0};
        MatrixCoord actualSubblockShape = MatrixCoord::Min(subblockShape,
                                                           actualBlockShape - subblockCoord * subblockShape);
        MatrixCoord subblockOffset = subblockCoord * subblockShape;

        SubblockLayout sub;
        if (actualSubblockShape.row() == 0 || actualSubblockShape.column() == 0) {
            return sub; // valid == false
        }
        sub.rows = actualSubblockShape.row();
        sub.cols = actualSubblockShape.column();
        sub.strideC = RoundUp<Catlass::BYTE_PER_C0>(sub.cols);
        sub.totalElems = sub.rows * sub.strideC;
        sub.gmRowStride = static_cast<uint32_t>(tla::get<0>(layoutBlock.stride()));
        sub.subblockRow = subblockOffset.row();
        sub.valid = true;
        return sub;
    }

    static constexpr int32_t EVENT_ID0 = 0;

    Params params_;
    Catlass::Arch::Resource<ArchTag>& resource_;
};

} // namespace Catlass::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_DECOMPOSITION_GM_HPP
