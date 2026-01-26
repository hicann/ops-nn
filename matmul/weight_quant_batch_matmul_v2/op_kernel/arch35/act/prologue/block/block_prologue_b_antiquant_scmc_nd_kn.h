/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PROLOGUE_BLOCK_BLOCK_PROLOGUE_B_ANTIQUANT_SCMC_ND_KN_H
#define PROLOGUE_BLOCK_BLOCK_PROLOGUE_B_ANTIQUANT_SCMC_ND_KN_H
#include "../constant.h"
#include "../../tile/tile_copy.h"
#include "../tile/tile_antiquant.h"
#include "../tile/tile_cast.h"
#include "../dispatch_policy.h"
#include "../../utils/underscore.h"
#include "act/utils/tuple_utils.h"
#include "../../utils/tensor_utils.h"
#include "../../utils/arch.h"
#include "../../utils/math_utils.h"
#include "../../utils/tensor_traits.h"

namespace WeightQuantBatchMatmulV2::Arch35::Act::Prologue::Block {

using Act::_;
using Act::CeilAlign;
using Act::CeilDiv;
using ::Act::Gemm::_1;
using ::Act::Gemm::_16;
using ::Act::Gemm::_256;
using ::Act::Gemm::Get;
using ::Act::Gemm::Int;
using AscendC::MakeCoord;
using AscendC::Shape;
using AscendC::Stride;

// nd-kn场景下，gm和L1的生产消费比为1:1
template <
    typename ArchTag_, typename HighBitType, typename PrologueScaleType, uint64_t AivNum, bool HasOffset_,
    uint64_t UbInBufNum, uint64_t InnerSize, uint64_t VfN, uint64_t VfK, uint64_t UbOutBufNum, uint64_t UbInSize,
    uint64_t UbOutSize, uint64_t ScaleSize, uint64_t OffsetSize, uint64_t AntiQuantScaleAfterCastSize, class BType_,
    class ScaleType_, class TileShapeL1_>
class BlockPrologue<
    BAntiquantScmc<
        ArchTag_, HighBitType, PrologueScaleType, false, false, AivNum, HasOffset_, UbInBufNum, InnerSize, VfN, VfK,
        UbOutBufNum, UbInSize, UbOutSize, ScaleSize, OffsetSize, AntiQuantScaleAfterCastSize>,
    BType_, ScaleType_, TileShapeL1_> {
public:
    using BType = BType_;
    using ElementIn = typename BType::Element;
    using LayoutIn = typename BType::Layout;
    using ScaleType = ScaleType_;
    using ElementScale = typename ScaleType::Element;
    using LayoutScale = typename ScaleType::Layout;
    using TileShapeL1 = TileShapeL1_;
    using ElementOut = HighBitType;
    using DispatchPolicy = BAntiquantScmc<
        Arch::Ascend910_95, HighBitType, PrologueScaleType, false, false, AivNum, HasOffset_, UbInBufNum, InnerSize,
        VfN, VfK, UbOutBufNum, UbInSize, UbOutSize, ScaleSize, OffsetSize, AntiQuantScaleAfterCastSize>;
    using ArchTag = Arch::Ascend910_95;

    static constexpr bool HAS_OFFSET = DispatchPolicy::HAS_OFFSET;
    static constexpr uint8_t AIV_NUM = DispatchPolicy::AIV_NUM;
    // VecAntiQuantConfig
    static constexpr uint64_t UB_MTE2_BUF_NUM = DispatchPolicy::UB_MTE2_BUF_NUM;
    static constexpr uint64_t UB_MTE2_INNER_SIZE = DispatchPolicy::UB_MTE2_INNER_SIZE;
    // VfConfig
    static constexpr uint64_t VF_N_STANDARD_LEN = DispatchPolicy::VF_N_STANDARD_LEN;
    static constexpr uint64_t VF_K_STANDARD_LEN = DispatchPolicy::VF_K_STANDARD_LEN;
    // UB_BUFFER_INFO
    static constexpr uint64_t UB_WEIGHT_OUTPUT_HIGH_BIT_BUFFER_NUM =
        DispatchPolicy::UB_WEIGHT_OUTPUT_HIGH_BIT_BUFFER_NUM;
    static constexpr uint64_t WEIGHT_INPUT_LOW_BIT_UB_TOTAL_SIZE = DispatchPolicy::WEIGHT_INPUT_LOW_BIT_UB_TOTAL_SIZE;
    static constexpr uint64_t HIGH_BIT_DATA_UB_TOTAL_SIZE = DispatchPolicy::HIGH_BIT_DATA_UB_TOTAL_SIZE;
    static constexpr uint64_t ANTIQUANT_SCALE_UB_TOTAL_SIZE = DispatchPolicy::ANTIQUANT_SCALE_UB_TOTAL_SIZE;
    static constexpr uint64_t ANTIQUANT_OFFSET_UB_TOTAL_SIZE = DispatchPolicy::ANTIQUANT_OFFSET_UB_TOTAL_SIZE;
    static constexpr uint64_t ANTIQUANT_SCALE_AFTER_CAST_UB_SIZE =
        DispatchPolicy::ANTIQUANT_SCALE_AFTER_CAST_UB_TOTAL_SIZE;
    static constexpr uint64_t WEIGHT_INPUT_LOW_BIT_UB_SINGLE_BUFFER_SIZE =
        DispatchPolicy::WEIGHT_INPUT_LOW_BIT_UB_SINGLE_BUFFER_SIZE;
    static constexpr uint64_t ANTIQUANT_SCALE_UB_SINGLE_BUFFER_SIZE =
        DispatchPolicy::ANTIQUANT_SCALE_UB_SINGLE_BUFFER_SIZE;
    static constexpr uint64_t ANTIQUANT_OFFSET_UB_SINGLE_BUFFER_SIZE =
        DispatchPolicy::ANTIQUANT_OFFSET_UB_SINGLE_BUFFER_SIZE;
    static constexpr uint64_t HIGH_BIT_DATA_UB_SINGLE_BUFFER_SIZE = DispatchPolicy::HIGH_BIT_DATA_UB_SINGLE_BUFFER_SIZE;
    static constexpr uint64_t ANTIQUANT_SCALE_AFTER_CAST_UB_TOTAL_SIZE =
        DispatchPolicy::ANTIQUANT_SCALE_AFTER_CAST_UB_TOTAL_SIZE;
    static constexpr uint64_t ANTIQUANT_SCALE_AFTER_CAST_UB_SINGLE_BUFFER_SIZE =
        DispatchPolicy::ANTIQUANT_SCALE_AFTER_CAST_UB_SINGLE_BUFFER_SIZE;
    struct Arguments {
        GM_ADDR ptrB;
        GM_ADDR ptrScale;
        GM_ADDR ptrOffset;
        LayoutIn layoutB;
        LayoutScale layoutScale;
        uint64_t antiQuantGroupSize;
    };

    struct Params {
        GM_ADDR ptrB;
        GM_ADDR ptrScale;
        GM_ADDR ptrOffset;
        LayoutIn layoutB;
        LayoutScale layoutScale;
        TileShapeL1 tileShapeL1;
        uint64_t antiQuantGroupSize;
        uint32_t weightL2Cacheable;
    };

    template <typename ProblemShape, typename Tiling>
    __aicore__ inline static Params ToUnderlyingArguments(
        const ProblemShape& problemShape, const Arguments& args, Tiling const* tiling)
    {
        auto baseM = static_cast<uint32_t>(tiling->matmulTiling.baseM);
        auto baseN = static_cast<uint32_t>(tiling->matmulTiling.baseN);
        auto baseK = static_cast<uint32_t>(tiling->matmulTiling.baseK);
        auto stepKa = static_cast<uint32_t>(tiling->matmulTiling.stepKa);
        auto stepKb = static_cast<uint32_t>(tiling->matmulTiling.stepKb);
        return {
            .ptrB = args.ptrB,
            .ptrScale = args.ptrScale,
            .ptrOffset = args.ptrOffset,
            .layoutB = args.layoutB,
            .layoutScale = args.layoutScale,
            .tileShapeL1 = AscendC::MakeShape(baseM, baseN, stepKa * baseK, stepKb * baseK),
            .antiQuantGroupSize = tiling->groupSize,
            .weightL2Cacheable = tiling->weightL2Cacheable};
    }

    __aicore__ inline BlockPrologue(const Params& params)
    {
        // 3 L1kb
        kbL1Size_ = Get<3>(params.tileShapeL1);
        if constexpr (AIV_NUM == 2) {
            // 此时c:v为1：2, cube所需数据由两个v提供
            kMte2BaseSize_ = kbL1Size_ >> 1;
        } else {
            // 此时c:v为1：1, cube所需数据由一个v提供
            kMte2BaseSize_ = kbL1Size_;
        }
        kSize_ = Get<1>(params.layoutB.GetShape());
        ubBInB8_ = Act::CreateTensor<UbInB8TensorTrait>(0);
        ubBOut_ = Act::CreateTensor<UbOutTensorTrait>(WEIGHT_INPUT_LOW_BIT_UB_TOTAL_SIZE);
        ubScale_ = Act::CreateTensor<UbScaleTensorTrait>(
            WEIGHT_INPUT_LOW_BIT_UB_TOTAL_SIZE + HIGH_BIT_DATA_UB_TOTAL_SIZE * sizeof(ElementOut));
        if constexpr (ANTIQUANT_SCALE_AFTER_CAST_UB_SIZE == 0) {
            ubOffset_ = Act::CreateTensor<UbScaleTensorTrait>(
                WEIGHT_INPUT_LOW_BIT_UB_TOTAL_SIZE + HIGH_BIT_DATA_UB_TOTAL_SIZE * sizeof(ElementOut) +
                ANTIQUANT_SCALE_UB_TOTAL_SIZE * sizeof(ElementScale));
        } else {
            ubScaleAfterCast_ = Act::CreateTensor<UbScaleAfterCastTensorTrait>(
                WEIGHT_INPUT_LOW_BIT_UB_TOTAL_SIZE + HIGH_BIT_DATA_UB_TOTAL_SIZE * sizeof(ElementOut) +
                ANTIQUANT_SCALE_UB_TOTAL_SIZE * sizeof(ElementScale));
        }
        uint64_t weightL1Space = Get<1>(params.tileShapeL1) * kbL1Size_; // weight单块大小
        weightF16L1_ = Act::CreateTensor<L1TensorTrait>(0);
        weightF16L1DbOffset_ = L1_SIZE * KB_ELEM<half> - weightL1Space;
        antiQuantGroupSize_ = params.antiQuantGroupSize;
        weightL2Cacheable_ = params.weightL2Cacheable;
    }

    __aicore__ inline ~BlockPrologue()
    {
        if (cvLoopIdx_ > 0) {
            WaitAicToAiv();
        }
        if (cvLoopIdx_ > 1) {
            WaitAicToAiv();
        }

        for (uint16_t idx = 0; idx < ubComputeLoopIdx_ && idx < UB_WEIGHT_OUTPUT_HIGH_BIT_BUFFER_NUM; idx++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(VEC_EVENT_ID_MTE3_TO_V[idx]);
        }

        for (uint16_t idx = 0; idx < ubMte2LoopIdx_ && idx < UB_MTE2_BUF_NUM; idx++) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(VEC_EVENT_ID_V_TO_MTE2[idx]);
        }
    }

    template <class TensorB, class TensorScale, class TensorOffset, class ActualBlockShape>
    __aicore__ inline void operator()(
        const TensorB& gB, const TensorScale& gScale, const TensorOffset& gOffset,
        const ActualBlockShape& actualBlockShape)
    {
        // nd-kn场景下，搬运消费比为1:1
        auto nL1Size = Get<0>(actualBlockShape);
        if constexpr (ANTIQUANT_TYPE == QuantType::PER_CHANNEL) {
            SetStride(ubScale_, AscendC::MakeStride(_1{}, CeilAlign(nL1Size, static_cast<uint64_t>(VECTOR_REG_WIDTH))));
            if constexpr (HasOffset_) {
                SetStride(ubOffset_, ubScale_.GetStride());
            }
        }
        for (uint64_t kMte2Offset = 0; kMte2Offset < kSize_; kMte2Offset += kbL1Size_, cvLoopIdx_++) {
            // l1K的实际大小
            l1RealExternalLen_ = CalcRealLen(kSize_, kMte2Offset, kbL1Size_);
            /*
             * 场景1：当前core为v0，mte2实际搬运的k值为mte2方向搬运标准值(kMte2BaseSize_)和l1上k实际值的最小值
             * 场景2: 当前core为v1, 且l1实际的k值比core v1搬运值大，则mte2实际搬运的k值为两者之差
             * 场景3：当前core为v1, 且l1实际的k值比core v1搬运值小，则mte2无需搬运,设置为0即可
             */
            // l1k对于aiv来说需要处理的实际大小
            uint64_t mte2RealK = AscendC::GetSubBlockIdx() == 0      ? min(kMte2BaseSize_, l1RealExternalLen_) :
                                 l1RealExternalLen_ > kMte2BaseSize_ ? l1RealExternalLen_ - kMte2BaseSize_ :
                                                                       0;
            uint64_t kAivOffset = kMte2Offset + AscendC::GetSubBlockIdx() * kMte2BaseSize_;
            auto gBTile = gB[gB.GetLayout()(AscendC::MakeCoord(_, kAivOffset))];
            WaitVToMTE2();
            if constexpr (ANTIQUANT_TYPE != QuantType::MX) {
                CopyGmToUb(gBTile, gScale, gOffset, nL1Size, mte2RealK);
            } else {
                uint64_t offset = gScale.GetLayout()(MakeCoord(_, CeilDiv(kAivOffset, antiQuantGroupSize_)));
                CopyGmToUb(gBTile, gScale[offset], gOffset, nL1Size, mte2RealK);
            }

            if (cvLoopIdx_ > 1) {
                WaitAicToAiv();
            }
            uint64_t mte2BufIdx = (ubMte2LoopIdx_ - 1) & (UB_MTE2_BUF_NUM - 1);
            auto ubBInB8 = ubBInB8_[mte2BufIdx * WEIGHT_INPUT_LOW_BIT_UB_SINGLE_BUFFER_SIZE];
            auto ubScale = ubScale_[mte2BufIdx * ANTIQUANT_SCALE_UB_SINGLE_BUFFER_SIZE];
            auto ubOffset = ubOffset_[mte2BufIdx * ANTIQUANT_SCALE_UB_SINGLE_BUFFER_SIZE];
            auto ubScaleAfterCast = ubScaleAfterCast_[mte2BufIdx * ANTIQUANT_SCALE_AFTER_CAST_UB_SINGLE_BUFFER_SIZE];
            SetStride(
                weightF16L1_, AscendC::MakeStride(
                                  AscendC::MakeStride(_1{}, CeilAlign(l1RealExternalLen_, 16UL) * 16UL),
                                  AscendC::MakeStride(_16{}, _256{})));
            if constexpr (ANTIQUANT_TYPE == QuantType::MX) {
                WeightAntiQuantCompute(
                    weightF16L1_[(cvLoopIdx_ & 1) * weightF16L1DbOffset_], ubBInB8, ubScaleAfterCast, ubOffset, nL1Size,
                    mte2RealK);
            } else {
                WeightAntiQuantCompute(
                    weightF16L1_[(cvLoopIdx_ & 1) * weightF16L1DbOffset_], ubBInB8, ubScale, ubOffset, nL1Size,
                    mte2RealK);
            }
            SetAivToAic();
            SetVToMTE2();
        }
    }

private:
    __aicore__ inline void WaitVToMTE2()
    {
        if (likely(ubMte2LoopIdx_ > UB_MTE2_BUF_NUM - 1)) {
            if constexpr (UB_MTE2_BUF_NUM == 2 || UB_MTE2_BUF_NUM == 4) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
                    VEC_EVENT_ID_V_TO_MTE2[ubMte2LoopIdx_ & (UB_MTE2_BUF_NUM - 1)]);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(VEC_EVENT_ID_V_TO_MTE2[ubMte2LoopIdx_ % UB_MTE2_BUF_NUM]);
            }
        }
    }

    __aicore__ inline void SetVToMTE2()
    {
        if constexpr (UB_MTE2_BUF_NUM == 2 || UB_MTE2_BUF_NUM == 4) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                VEC_EVENT_ID_V_TO_MTE2[(ubMte2LoopIdx_ - 1) & (UB_MTE2_BUF_NUM - 1)]);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                VEC_EVENT_ID_V_TO_MTE2[(ubMte2LoopIdx_ - 1) % UB_MTE2_BUF_NUM]);
        }
    }

    __aicore__ inline void WaitAicToAiv()
    {
        AscendC::CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
    }

    __aicore__ inline void SetAivToAic()
    {
        AscendC::CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
    }

    template <class TensorTraitB, class TensorScaleOffset>
    __aicore__ inline void CopyGmToUb(
        const AscendC::GlobalTensor<TensorTraitB>& gB, const TensorScaleOffset& gScale,
        const TensorScaleOffset& gOffset, uint64_t ubMte2NSize, uint64_t ubMte2KSize)
    {
        using TensorB = AscendC::GlobalTensor<TensorTraitB>;
        // ubMte2NSize和ubMte2KSize为实际MTE2搬运到UB的有效数据，
        // 其按照ubMte2InnerSize进行跳写，垃圾数据无需操作，搬出的时搬运有效数据即可。
        if (ubMte2NSize == 0 || ubMte2KSize == 0) {
            ubMte2LoopIdx_++; // 避免当前核无任务时，SetVToMTE2()对同一个flagID重复SetFlag的问题
            return;
        }
        if (!weightL2Cacheable_) {
            Act::Tile::Copy<ArchTag, Act::Tile::CacheMode::NOT_ALLOC_KEEP>(
                Act::ReinerpretCast<UbInTensorTrait>(
                    ubBInB8_[(ubMte2LoopIdx_ % UB_MTE2_BUF_NUM) * WEIGHT_INPUT_LOW_BIT_UB_SINGLE_BUFFER_SIZE]),
                gB, AscendC::MakeShape(ubMte2NSize, ubMte2KSize));
        } else {
            Act::Tile::Copy<ArchTag>(
                Act::ReinerpretCast<UbInTensorTrait>(
                    ubBInB8_[(ubMte2LoopIdx_ % UB_MTE2_BUF_NUM) * WEIGHT_INPUT_LOW_BIT_UB_SINGLE_BUFFER_SIZE]),
                gB, AscendC::MakeShape(ubMte2NSize, ubMte2KSize));
        }
        if constexpr (ANTIQUANT_TYPE == QuantType::PER_CHANNEL) {
            Act::Tile::Copy<ArchTag>(
                ubScale_[(ubMte2LoopIdx_ % UB_MTE2_BUF_NUM) * ANTIQUANT_SCALE_UB_SINGLE_BUFFER_SIZE], gScale,
                AscendC::MakeShape(ubMte2NSize, 1));
            if constexpr (HAS_OFFSET) {
                Act::Tile::Copy<ArchTag>(
                    ubOffset_[(ubMte2LoopIdx_ % UB_MTE2_BUF_NUM) * ANTIQUANT_OFFSET_UB_SINGLE_BUFFER_SIZE], gOffset,
                    AscendC::MakeShape(ubMte2NSize, 1));
            }
        } else if constexpr (ANTIQUANT_TYPE == QuantType::PER_TENSOR) {
            GlobalTensor<ElementScale> tmpScale;
            tmpScale.address_ = gScale.address_;
            scaleValue_ = tmpScale.GetValue(0);
            if constexpr (HAS_OFFSET) {
                tmpScale.address_ = gOffset.address_;
                offsetValue_ = tmpScale.GetValue(0);
            }
        } else if constexpr (ANTIQUANT_TYPE == QuantType::MX || ANTIQUANT_TYPE == QuantType::PER_GROUP) {
            if constexpr (HAS_OFFSET) {
                static_assert(
                    AscendC::Std::always_false_v<decltype(HAS_OFFSET)>, "Not support offset in pergroup mode");
            }
            Act::Tile::Copy<ArchTag>(
                ubScale_[(ubMte2LoopIdx_ % UB_MTE2_BUF_NUM) * ANTIQUANT_SCALE_UB_SINGLE_BUFFER_SIZE], gScale,
                AscendC::MakeShape(ubMte2NSize, CeilDiv(ubMte2KSize, antiQuantGroupSize_)));
            // MX场景scale需要进行数据类型转化
            if constexpr (ANTIQUANT_TYPE == QuantType::MX) {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
                Prologue::Tile::TileCast<ArchTag>(
                    ubScaleAfterCast_
                        [(ubMte2LoopIdx_ % UB_MTE2_BUF_NUM) * ANTIQUANT_SCALE_AFTER_CAST_UB_SINGLE_BUFFER_SIZE],
                    ubScale_[(ubMte2LoopIdx_ % UB_MTE2_BUF_NUM) * ANTIQUANT_SCALE_UB_SINGLE_BUFFER_SIZE],
                    AscendC::MakeShape(ubMte2NSize, CeilDiv(ubMte2KSize, antiQuantGroupSize_), antiQuantGroupSize_));
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        ubMte2LoopIdx_++;
    }

    template <class TensorL1, class TensorInB, class TensorScale, class TensorOffset>
    __aicore__ inline void WeightAntiQuantCompute(
        const TensorL1& tensorL1, const TensorInB& ubBInB8, const TensorScale& ubScale, const TensorOffset& ubOffset,
        uint64_t ubMte2NSize, uint64_t ubMte2KSize)
    {
        uint64_t weightHighBitL1Offset;
        uint64_t nRealLen;
        uint64_t kRealLen;
        static constexpr uint64_t OUTER_SIZE = 64;
        using PolicyAntiquant = Prologue::Tile::AntiquantFixTile<UB_MTE2_INNER_SIZE, OUTER_SIZE, HAS_OFFSET>;
        for (uint64_t kOffset = 0; kOffset < ubMte2KSize; kOffset += VF_K_STANDARD_LEN) {
            for (uint64_t nOffset = 0; nOffset < ubMte2NSize; nOffset += VF_N_STANDARD_LEN) {
                if (likely(ubComputeLoopIdx_ > UB_WEIGHT_OUTPUT_HIGH_BIT_BUFFER_NUM - 1)) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                        VEC_EVENT_ID_MTE3_TO_V[ubComputeLoopIdx_ & (UB_WEIGHT_OUTPUT_HIGH_BIT_BUFFER_NUM - 1)]);
                }
                nRealLen = CalcRealLen(ubMte2NSize, nOffset, VF_N_STANDARD_LEN);
                kRealLen = CalcRealLen(ubMte2KSize, kOffset, VF_K_STANDARD_LEN);
                auto regBOut = ubBOut_[(ubComputeLoopIdx_ & 1) * HIGH_BIT_DATA_UB_SINGLE_BUFFER_SIZE];
                auto regBIn = Act::ReinerpretCast<UbInTensorTrait>(
                    ubBInB8[ElemToByte<ElementIn>(ubBInB8.GetLayout()(AscendC::MakeCoord(nOffset, kOffset)))]);
                if constexpr (ANTIQUANT_TYPE == QuantType::PER_TENSOR) {
                    Prologue::Tile::Antiquant<ArchTag, PolicyAntiquant>(
                        regBOut, regBIn, scaleValue_, offsetValue_, AscendC::MakeShape(nRealLen, kRealLen));
                } else if constexpr (ANTIQUANT_TYPE == QuantType::PER_CHANNEL) {
                    auto regScale = ubScale[ubScale.GetLayout()(AscendC::MakeCoord(nOffset, _))];
                    auto regOffset = ubOffset[ubOffset.GetLayout()(AscendC::MakeCoord(nOffset, _))];
                    Prologue::Tile::Antiquant<ArchTag, PolicyAntiquant>(
                        regBOut, regBIn, regScale, regOffset, AscendC::MakeShape(nRealLen, kRealLen));
                } else if constexpr (ANTIQUANT_TYPE == QuantType::MX) {
                    auto regScale =
                        ubScale[ubScale.GetLayout()(MakeCoord(nOffset, CeilDiv(kOffset, antiQuantGroupSize_)))];
                    Prologue::Tile::Antiquant<ArchTag, PolicyAntiquant>(
                        regBOut, regBIn, regScale, ubOffset,
                        AscendC::MakeShape(nRealLen, kRealLen, antiQuantGroupSize_));
                } else {
                    static_assert(AscendC::Std::always_false_v<decltype(ANTIQUANT_TYPE)>, "not support per group");
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                uint64_t offset = tensorL1.GetLayout()(AscendC::MakeCoord(
                    AscendC::MakeCoord(_, CeilDiv(nOffset, 16UL)),
                    AscendC::MakeCoord(_, CeilDiv(kOffset + AscendC::GetSubBlockIdx() * kMte2BaseSize_, 16UL))));
                Act::Tile::Copy<ArchTag>(tensorL1[offset], regBOut, AscendC::MakeShape(nRealLen, kRealLen));
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(
                    VEC_EVENT_ID_MTE3_TO_V[ubComputeLoopIdx_ & (UB_WEIGHT_OUTPUT_HIGH_BIT_BUFFER_NUM - 1)]);
                ubComputeLoopIdx_++;
            }
        }
    }

    // mte2搬运计数，用于控制weight输入的buffer和 mte2&&V间同步控制
    uint64_t ubMte2LoopIdx_ = 0;
    // vf中标准计算单元(VF_N_STANDARD_LEN,
    // VF_K_STANDARD_LEN)的计数，用于控制weight反量化后输出的buffer和V&&mte3间同步控制
    uint64_t ubComputeLoopIdx_ = 0;
    static constexpr AscendC::TEventID VEC_EVENT_ID_V_TO_MTE2[QUADRUPLE_BUFFER_NUM] = {0, 1, 2, 3};
    static constexpr AscendC::TEventID VEC_EVENT_ID_MTE3_TO_V[QUADRUPLE_BUFFER_NUM] = {0, 1, 2, 3};
    uint64_t cvLoopIdx_ = 0;
    ElementScale scaleValue_;
    ElementScale offsetValue_;
    // (n1,k1,k0,n0)
    using L1TensorTrait = AscendC::TensorTrait<
        ElementOut, AscendC::TPosition::B1,
        AscendC::Layout<
            Shape<Shape<_16, uint64_t>, Shape<_16, uint64_t>>, Stride<Stride<_1, uint64_t>, Stride<_16, _256>>>>;
    AscendC::LocalTensor<L1TensorTrait> weightF16L1_;
    uint64_t weightF16L1DbOffset_;
    uint64_t kMte2BaseSize_;
    uint64_t kSize_;
    uint64_t kbL1Size_;
    uint64_t l1RealExternalLen_;
    uint64_t antiQuantGroupSize_;
    int32_t weightL2Cacheable_;
    // ND
    // (k,n) layout (UB_MTE2_INNER_SIZE, uint64_t):(1, UB_MTE2_INNER_SIZE)
    // (n,k) layout (uint64_t, UB_MTE2_INNER_SIZE):(UB_MTE2_INNER_SIZE, 1)
    // Nz
    // (n1,k1,k0,n0) layout ((),()):((),())
    using UbInB8TensorTrait = AscendC::TensorTrait<
        int8_t, AscendC::TPosition::VECIN,
        AscendC::Layout<Shape<Int<UB_MTE2_INNER_SIZE>, uint64_t>, Stride<_1, Int<UB_MTE2_INNER_SIZE>>>>;
    using UbInTensorTrait = AscendC::TensorTrait<
        ElementIn, AscendC::TPosition::VECIN,
        AscendC::Layout<Shape<Int<UB_MTE2_INNER_SIZE>, uint64_t>, Stride<_1, Int<UB_MTE2_INNER_SIZE>>>>;
    // nd (k,n) (n1,k1,k0,n0) layout ((n0,n1),(k0,k1)):((1,k1*k0*n0),(n0,k0*n0))
    using UbOutTensorTrait = AscendC::TensorTrait<
        ElementOut, AscendC::TPosition::VECIN,
        AscendC::Layout<
            Shape<Shape<_16, Int<CeilDiv(VF_N_STANDARD_LEN, 16UL)>>, Shape<_16, Int<CeilDiv(VF_K_STANDARD_LEN, 16UL)>>>,
            Stride<Stride<_1, Int<(VF_K_STANDARD_LEN + 1) * BLOCK_CUBE>>, Stride<_16, _256>>>>;
    using UbScaleTensorTrait = AscendC::Std::conditional_t<
        QUANT_TYPE<decltype(LayoutScale{}.GetShape()), ElementScale> != QuantType::MX,
        AscendC::TensorTrait<
            ElementScale, AscendC::TPosition::VECIN, AscendC::Layout<Shape<uint64_t, _1>, Stride<_1, uint64_t>>>,
        AscendC::TensorTrait<
            ElementScale, AscendC::TPosition::VECIN,
            AscendC::Layout<Shape<uint64_t, uint64_t>, Stride<_1, Int<UB_MTE2_INNER_SIZE>>>>>;
    using UbScaleAfterCastTensorTrait = AscendC::TensorTrait<
        ElementOut, AscendC::TPosition::VECIN,
        AscendC::Layout<Shape<uint64_t, uint64_t>, Stride<_1, Int<UB_MTE2_INNER_SIZE>>>>;
    AscendC::LocalTensor<UbInB8TensorTrait> ubBInB8_;
    AscendC::LocalTensor<UbOutTensorTrait> ubBOut_;
    AscendC::LocalTensor<UbScaleTensorTrait> ubScale_;
    AscendC::LocalTensor<UbScaleTensorTrait> ubOffset_;
    AscendC::LocalTensor<UbScaleAfterCastTensorTrait> ubScaleAfterCast_;

    static constexpr QuantType ANTIQUANT_TYPE = QUANT_TYPE<decltype(LayoutScale{}.GetShape()), ElementScale>;
};
} // namespace WeightQuantBatchMatmulV2::Arch35::Act::Prologue::Block
#endif