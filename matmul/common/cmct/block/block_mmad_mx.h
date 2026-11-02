/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_mmad_mx.h
 * \brief
 */

#pragma once
#include "../utils/layout_utils.h"
#include "../utils/common_utils.h"
#include "../utils/quant_batch_matmul_constant.h"
#include "../utils/tuple_utils.h"
#include "../policy/dispatch_policy.h"
#include "../tile/tile_copy.h"

namespace Cmct {
namespace Gemm {
namespace Block {
using namespace AscendC;
using namespace Cmct::Gemm::QuantBatchMatmul;

struct TileL1L0Param {
    uint64_t curM = 0;
    uint64_t curN = 0;
    uint64_t curAlignM = 0;
    uint64_t curAlignN = 0;
    uint64_t curGmAKL1 = 0;
    uint64_t curGmBKL1 = 0;
    uint64_t curPadAKL1 = 0; // pad to 64 align
    uint64_t curPadBKL1 = 0; // pad to 64 align
    uint64_t curKL0 = 0;
};

template <class DispatchPolicy_, class L1TileShape_, class L0TileShape_, class AType_, class LayoutA_, class BType_,
          class LayoutB_, class CType_, class LayoutC_, class BiasType_, class LayoutBias_, class TileCopy_,
          class Enable = void>
class BlockMmadMx {
    static_assert(AscendC::Std::always_false_v<DispatchPolicy_>, "Should not be here!");
};

template <class DispatchPolicy_, class L1TileShape_, class L0TileShape_, class AType_, class LayoutA_, class BType_,
          class LayoutB_, class CType_, class LayoutC_, class BiasType_, class LayoutBias_, class TileCopy_>
class BlockMmadMx<DispatchPolicy_, L1TileShape_, L0TileShape_, AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_,
                  BiasType_, LayoutBias_, TileCopy_,
                  AscendC::Std::enable_if_t<
                      AscendC::Std::is_base_of_v<MatmulWithScale<>, DispatchPolicy_> ||
                      AscendC::Std::is_base_of_v<MatmulWithScale<AscendC::Shape<_0, _0, _0, _0>, A_FULL_LOAD_MODE>,
                                                 DispatchPolicy_>>> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
#if CMCT_HIF4_SUPPORT
    using ScaleAType = typename AscendC::Conditional<IsSameType<AType, hifloat4x2_t>::value, hif4_scale,
                                                     AscendC::fp8_e8m0_t>::type;
    using ScaleBType = typename AscendC::Conditional<IsSameType<AType, hifloat4x2_t>::value, hif4_scale,
                                                     AscendC::fp8_e8m0_t>::type;
#else
    using ScaleAType = AscendC::fp8_e8m0_t;
    using ScaleBType = AscendC::fp8_e8m0_t;
#endif
    using ScaleType = ScaleAType;
    using MxL0AType = typename GetL0DataType<AType, true>::Type;
    using MxL0BType = typename GetL0DataType<BType, true>::Type;
    using BiasType = BiasType_;
    using DispatchPolicy = DispatchPolicy_;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    uint64_t m_;
    uint64_t n_;
    uint64_t k_;
    uint64_t l1BufNum_{1};
    uint64_t kL1Iter_{0};
    uint64_t kL1_{1};
    uint64_t scaleKL1_{1};
    uint64_t scaleKL1Num_{1}; // scale多载后，K方向上实际的scale元素个数
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};
    bool isBias_{false};
#if CMCT_HIF4_SUPPORT
    static constexpr bool IS_FP4 = AscendC::IsSameType<AType, fp4x2_e2m1_t>::value ||
                                   AscendC::IsSameType<AType, hifloat4x2_t>::value;
#else
    static constexpr bool IS_FP4 = AscendC::IsSameType<AType, fp4x2_e2m1_t>::value;
#endif
    static constexpr uint64_t MIN_STEP = IS_FP4 ? B4_MIN_STEP : B8_MIN_STEP;
    // 元素数折算为存储字节数的右移位数：B8 为 0，B4 为 1（右移参数1表示除2）
    static constexpr uint64_t ELEMENT_SIZE_RATIO = (AscendC::IsSameType<AType, fp8_e5m2_t>::value ||
                                                    AscendC::IsSameType<AType, fp8_e4m3fn_t>::value) ?
                                                       0 :
                                                       1;
#if CMCT_HIF4_SUPPORT
    static constexpr bool IS_HIF4 = AscendC::IsSameType<AType, hifloat4x2_t>::value;
#else
    static constexpr bool IS_HIF4 = false;
#endif
    static constexpr CubeFormat formatB = TagToFormat<LayoutB>::format;
    static constexpr bool transA = TagToTrans<LayoutA>::value;
    static constexpr bool transB = TagToTrans<LayoutB>::value;
    constexpr static uint64_t HALF_L0_SIZE = (L0A_SIZE / DOUBLE_BUFFER_COUNT) << (IS_FP4 ? 1 : 0);
    constexpr static uint64_t HALF_L0C_SIZE = AscendC::TOTAL_L0C_SIZE / DOUBLE_BUFFER_COUNT / sizeof(float);
    constexpr static int32_t C0_SIZE = AscendC::AuxGetC0Size<AType>();
    constexpr static int32_t BIAS_C0 = AscendC::AuxGetC0Size<BiasType>();
    constexpr static uint64_t BLOCK_CUBE = 16UL;
    constexpr static uint64_t BLOCK_REDUCE_CUBE = 32UL;
    constexpr static uint64_t MXFP_GROUP_SIZE = 32UL;
    constexpr static uint64_t MXFP_DIVISOR_SIZE = 64UL;
    constexpr static uint64_t MXFP_MULTI_BASE_SIZE = 2;
    constexpr static uint64_t HIF4_SCALE_NUM_HALF_RATIO = 1UL;
    constexpr static uint64_t HIF4_GROUP_SIZE = 64UL;
    constexpr static uint64_t SCALE_BUFFER_NUM = 2;
    constexpr static uint64_t EVEN_FACTOR = 2UL;
    // Set unitflag state: 3 = final accumulation, 2 = non-final accumulation
    constexpr static uint32_t FINAL_ACCUMULATION = 3;
    constexpr static uint32_t NON_FINAL_ACCUMULATION = 2;
    uint64_t abL1LoopCnt_{0};
    uint64_t scaleLoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0InnerPingPong_{0};
    uint64_t l0cPingPong_{0};
    bool enableL0cPingPong_{false};

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR pertokenScaleGmAddr{nullptr};
        GM_ADDR scaleGmAddr{nullptr};
    };

    struct L1Params {
        uint64_t kL1;
        uint64_t scaleKL1;
        uint64_t l1BufNum;
    };

    enum class MTE1SplitMode : uint32_t {
        DEFAULT = 0x0U,
        SPLIT_M = 0x1U,
        SPLIT_N = 0x1U << 1,
    };

    __aicore__ inline BlockMmadMx()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_1);
        AscendC::SetMMLayoutTransform(true); // true means column first when fixpipe_l0c2out

        // flag 2 的初值由 TPipe（M_MTE1 事件 id 0~2）预置，此处仅显式预置 flag 3 并在析构时回收。
        // flag 2/3 供 MTE1 split 内层乒乓使用（split 仅在 DAV_9202 运行），该 set/wait 成对执行、无副作用。
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(MTE1_WAIT_MMAD_FLAG_3);
    }

    __aicore__ inline ~BlockMmadMx()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_1);
        AscendC::SetMMLayoutTransform(false); // false means row first when fixpipe_l0c2out

        // 与构造中的 flag3 成对；flag 2 由 TPipe 析构时统一等待回收
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(MTE1_WAIT_MMAD_FLAG_3);
    }

public:
    __aicore__ inline void Init(const TupleShape& problemShape, const BlockShape& l0TileShape, const L1Params& l1Params,
                                bool isBias, bool dbL0C)
    {
        m_ = Get<IDX_M_IDX>(problemShape);
        n_ = Get<IDX_N_IDX>(problemShape);
        k_ = Get<IDX_K_IDX>(problemShape);
        kL1_ = l1Params.kL1;
        scaleKL1_ = l1Params.scaleKL1;
        if constexpr (IS_HIF4) {
            scaleKL1Num_ = Cmct::Gemm::CeilDiv(scaleKL1_, HIF4_GROUP_SIZE);
        } else {
            scaleKL1Num_ = Cmct::Gemm::CeilDiv(scaleKL1_, MXFP_GROUP_SIZE);
        }
        baseM_ = Get<IDX_M_IDX>(l0TileShape);
        baseN_ = Get<IDX_N_IDX>(l0TileShape);
        baseK_ = Get<IDX_K_IDX>(l0TileShape);
        isBias_ = isBias;
        l1BufNum_ = l1Params.l1BufNum;
        enableL0cPingPong_ = dbL0C;
        bL1OneBuffer_ = baseN_ * kL1_;
        if constexpr (IS_HIF4) {
            scaleBL1OneBuffer_ = baseN_ * scaleKL1Num_;
        } else {
            scaleBL1OneBuffer_ = baseN_ * Cmct::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        }
        if (isBias_) {
            biasL1OneBuffer_ = baseN_ * sizeof(BiasType);
        }
        if constexpr (DispatchPolicy::fullLoadMode == 0) {
            aL1OneBuffer_ = baseM_ * Cmct::Gemm::Align(kL1_, MXFP_DIVISOR_SIZE);
            if constexpr (IS_HIF4) {
                scaleAL1OneBuffer_ = baseM_ * scaleKL1Num_;
            } else {
                scaleAL1OneBuffer_ = baseM_ * Cmct::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
            }
            for (int32_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
                // 2 buffer: L1 space is : A0|B0|AScale0|BScale0|bias0|...|A1|B1|AScale1|BScale1|bias1|...
                // 4 buffer: L1 space is : A0A2|B0B2|AScale0|BScale0|bias0|...|A1A3|B1B3|AScale1|BScale1|bias1|...
                // For B8, l1Offset represents byte count; for B4, l1Offset represents element count
                uint64_t l1Offset = IS_FP4 ? AscendC::TOTAL_L1_SIZE * (bufferId & 1) :
                                             (AscendC::TOTAL_L1_SIZE >> 1) * (bufferId & 1);
                l1BufferAOffset_[bufferId] = l1Offset + aL1OneBuffer_ * (bufferId >> 1);
                l1BufferBOffset_[bufferId] = l1Offset + aL1OneBuffer_ * (l1BufNum_ >> 1) +
                                             bL1OneBuffer_ * (bufferId >> 1);
            }
            for (int32_t bufferId = 0; bufferId < SCALE_BUFFER_NUM; bufferId++) {
                // l1BufferScaleAOffset_[bufferId]: byte count when B8, element count when B4
                if constexpr (IS_HIF4) {
                    // hif4_scale 为 4Byte，L1 上按 ScaleAType 元素寻址；size 用元素个数，偏移换算为元素
                    uint64_t scaleAByteOffset = (l1BufferBOffset_[bufferId] + bL1OneBuffer_ * (l1BufNum_ >> 1)) >> 1;
                    l1BufferScaleAOffset_[bufferId] = scaleAByteOffset / sizeof(ScaleAType);
                    l1BufferScaleBOffset_[bufferId] = l1BufferScaleAOffset_[bufferId] + scaleAL1OneBuffer_;
                    l1BufferBiasOffset_[bufferId] = (l1BufferScaleBOffset_[bufferId] + scaleBL1OneBuffer_) *
                                                    sizeof(ScaleAType);
                } else {
                    l1BufferScaleAOffset_[bufferId] = IS_FP4 ? ((l1BufferBOffset_[bufferId] +
                                                                 bL1OneBuffer_ * (l1BufNum_ >> 1)) >>
                                                                1) :
                                                               (l1BufferBOffset_[bufferId] +
                                                                bL1OneBuffer_ * (l1BufNum_ >> 1));
                    l1BufferScaleBOffset_[bufferId] = l1BufferScaleAOffset_[bufferId] + scaleAL1OneBuffer_;
                    l1BufferBiasOffset_[bufferId] = l1BufferScaleBOffset_[bufferId] + scaleBL1OneBuffer_;
                }
            }
        } else {
            uint64_t mAlign = Cmct::Gemm::Align(baseM_, transA ? C0_SIZE : BLOCK_CUBE);
            uint64_t kAlign = Cmct::Gemm::Align(k_, MXFP_DIVISOR_SIZE);
            aL1OneBuffer_ = mAlign * kAlign;
            if constexpr (IS_HIF4) {
                scaleAL1OneBuffer_ = baseM_ * Cmct::Gemm::CeilDiv(k_, HIF4_GROUP_SIZE);
            } else {
                scaleAL1OneBuffer_ = baseM_ * Cmct::Gemm::CeilDiv(k_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
            }
            // 2 buffer: L1 space is : B0|BScale0|bias0|A|AScale|...|B1|BScale1|bias1|
            // 4 buffer: L1 space is : B0B2|BScale0|bias0|A|AScale|...|B1B3|BScale1|bias1|...
            // l1BufferAOffset_[0]: byte count when B8, element count when B4
            // l1BufferScaleAOffset_[0]: byte count when B8, element count when B4
            if constexpr (IS_HIF4) {
                // hif4_scale 为 4Byte，scale 偏移按元素计算，bias 偏移按 Byte 计算
                l1BufferAOffset_[0] = bL1OneBuffer_ * (l1BufNum_ >> 1) +
                                      ((scaleBL1OneBuffer_ * sizeof(ScaleBType) + biasL1OneBuffer_) << 1);
                l1BufferScaleAOffset_[0] = ((l1BufferAOffset_[0] + aL1OneBuffer_) >> 1) / sizeof(ScaleAType);
            } else {
                l1BufferAOffset_[0] = IS_FP4 ?
                                          (bL1OneBuffer_ * (l1BufNum_ >> 1) +
                                           ((scaleBL1OneBuffer_ + biasL1OneBuffer_) << 1)) :
                                          (bL1OneBuffer_ * (l1BufNum_ >> 1) + scaleBL1OneBuffer_ + biasL1OneBuffer_);
                l1BufferScaleAOffset_[0] = IS_FP4 ? ((l1BufferAOffset_[0] + aL1OneBuffer_) >> 1) :
                                                    (l1BufferAOffset_[0] + aL1OneBuffer_);
            }
            // b1Offset: byte count when B8, element count when B4
            uint64_t scaleAEndOffset = IS_HIF4 ?
                                           ((l1BufferScaleAOffset_[0] + scaleAL1OneBuffer_) * sizeof(ScaleAType)) :
                                           (l1BufferScaleAOffset_[0] + scaleAL1OneBuffer_);
            uint64_t b1Offset = scaleAEndOffset >= (AscendC::TOTAL_L1_SIZE >> 1) ? scaleAEndOffset :
                                                                                   (AscendC::TOTAL_L1_SIZE >> 1);
            if constexpr (IS_FP4) {
                b1Offset <<= 1;
            }
            for (int32_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
                l1BufferBOffset_[bufferId] = b1Offset * (bufferId & 1) + bL1OneBuffer_ * (bufferId >> 1);
            }
            for (int32_t bufferId = 0; bufferId < SCALE_BUFFER_NUM; bufferId++) {
                // l1BufferScaleBOffset_[bufferId]: byte count when B8, element count when B4
                if constexpr (IS_HIF4) {
                    uint64_t scaleBByteOffset = (l1BufferBOffset_[bufferId] + bL1OneBuffer_ * (l1BufNum_ >> 1)) >> 1;
                    l1BufferScaleBOffset_[bufferId] = scaleBByteOffset / sizeof(ScaleBType);
                    l1BufferBiasOffset_[bufferId] = (l1BufferScaleBOffset_[bufferId] + scaleBL1OneBuffer_) *
                                                    sizeof(ScaleBType);
                } else {
                    l1BufferScaleBOffset_[bufferId] = IS_FP4 ? ((l1BufferBOffset_[bufferId] +
                                                                 bL1OneBuffer_ * (l1BufNum_ >> 1)) >>
                                                                1) :
                                                               (l1BufferBOffset_[bufferId] +
                                                                bL1OneBuffer_ * (l1BufNum_ >> 1));
                    l1BufferBiasOffset_[bufferId] = l1BufferScaleBOffset_[bufferId] + scaleBL1OneBuffer_;
                }
            }
        }
        kL1Iter_ = CeilDiv(k_, kL1_);
    }

    __aicore__ inline void CopyInA1(const AscendC::GlobalTensor<AType>& aGlobal,
                                    const AscendC::LocalTensor<AType>& al1Local, TileL1L0Param& tileL1L0Param)
    {
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        uint64_t nDim = transA ? tileL1L0Param.curGmAKL1 : tileL1L0Param.curM;
        uint64_t dDim = transA ? tileL1L0Param.curM : tileL1L0Param.curGmAKL1;

        nd2nzParams.nValue = nDim;
        nd2nzParams.dValue = dDim;
        if constexpr (IS_FP4) {
            nd2nzParams.dValue = dDim >> 1;
        }
        nd2nzParams.srcNdMatrixStride = 1;
        nd2nzParams.srcDValue = transA ? m_ : k_;
        if constexpr (IS_FP4) {
            nd2nzParams.srcDValue = nd2nzParams.srcDValue >> 1;
        }
        nd2nzParams.dstNzC0Stride = transA ? tileL1L0Param.curPadAKL1 : tileL1L0Param.curAlignM;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 1;
        AscendC::DataCopy(al1Local, aGlobal, nd2nzParams, true);
    }

    __aicore__ inline void CopyInA1Dn2NZ(const AscendC::GlobalTensor<AType>& aGlobal,
                                         const AscendC::LocalTensor<AType>& al1Local, TileL1L0Param& tileL1L0Param)
    {
        // 基础api dn2nz 对hifloat4/fp4 等类型还不支持, 不加过滤会有编译报错。
        // 调用点已由 dn2nzMTE2TransA_ 的 SupportType 条件保证只有 fp8 会进入此函数。
        if constexpr (!SupportType<AType, fp8_e5m2_t, fp8_e4m3fn_t>()) {
            return;
        }

        AscendC::Dn2NzParams dn2nzParams;
        dn2nzParams.dnNum = 1;
        dn2nzParams.nValue = tileL1L0Param.curM;
        dn2nzParams.dValue = tileL1L0Param.curGmAKL1;

        dn2nzParams.srcDnMatrixStride = 0;
        dn2nzParams.srcDValue = m_; // 源Dn 矩阵每行的所含元素个数
        dn2nzParams.dstNzC0Stride = Cmct::Gemm::CeilAlign(tileL1L0Param.curM,
                                                          AscendC::BLOCK_CUBE); // 单位:c0_size, 即32B
        dn2nzParams.dstNzNStride = 1;                                           // 单位:c0_size
        dn2nzParams.dstNzMatrixStride = 0;
        AscendC::DataCopy(al1Local, aGlobal, dn2nzParams, true);
    }

    __aicore__ inline void CopyInB1(const AscendC::GlobalTensor<BType>& bGlobal,
                                    const AscendC::LocalTensor<BType>& bl1Local, TileL1L0Param& tileL1L0Param)
    {
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        uint64_t nDim = transB ? tileL1L0Param.curN : tileL1L0Param.curGmBKL1;
        uint64_t dDim = transB ? tileL1L0Param.curGmBKL1 : tileL1L0Param.curN;

        nd2nzParams.nValue = nDim;
        nd2nzParams.dValue = dDim;
        if constexpr (IS_FP4) {
            nd2nzParams.dValue = dDim >> 1;
        }
        nd2nzParams.srcNdMatrixStride = 1;
        nd2nzParams.srcDValue = transB ? k_ : n_;
        if constexpr (IS_FP4) {
            nd2nzParams.srcDValue = nd2nzParams.srcDValue >> 1;
        }
        nd2nzParams.dstNzC0Stride = transB ? tileL1L0Param.curAlignN : tileL1L0Param.curPadBKL1;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 1;
        AscendC::DataCopy(bl1Local, bGlobal, nd2nzParams, true);
    }

    __aicore__ inline void CopyInB1Dn2NZ(const AscendC::GlobalTensor<BType>& bGlobal,
                                         const AscendC::LocalTensor<BType>& bl1Local, TileL1L0Param& tileL1L0Param)
    {
        // 基础api dn2nz 对hifloat4/fp4 等类型还不支持, 不加过滤会有编译报错。
        // 调用点已由 dn2nzMTE2TransB_ 的 SupportType 条件保证只有 fp8 会进入此函数。
        if constexpr (!SupportType<BType, fp8_e5m2_t, fp8_e4m3fn_t>()) {
            return;
        }

        AscendC::Dn2NzParams dn2nzParams;
        dn2nzParams.dnNum = 1;
        dn2nzParams.nValue = tileL1L0Param.curN;
        dn2nzParams.dValue = tileL1L0Param.curGmBKL1;
        dn2nzParams.srcDnMatrixStride = 0;
        dn2nzParams.srcDValue = n_;
        dn2nzParams.dstNzC0Stride = Cmct::Gemm::CeilAlign(tileL1L0Param.curN,
                                                          AscendC::BLOCK_CUBE); // Nz矩阵间的stride需要cube_block对齐
        dn2nzParams.dstNzNStride = 1;
        dn2nzParams.dstNzMatrixStride = 0;
        AscendC::DataCopy(bl1Local, bGlobal, dn2nzParams, true);
    }

    __aicore__ inline void CopyInB1WeightNz(const AscendC::GlobalTensor<BType>& bGlobal,
                                            const AscendC::LocalTensor<BType>& bl1Local,
                                            const TileL1L0Param& tileL1L0Param)
    {
        AscendC::GlobalTensor<uint8_t> bGlobalUInt8;
        bGlobalUInt8.SetGlobalBuffer((__gm__ uint8_t*)bGlobal.GetPhyAddr());
        auto bl1LocalUint8 = bl1Local.template ReinterpretCast<uint8_t>();

        AscendC::DataCopyExtParams dataCopyParams;
        AscendC::DataCopyPadExtParams<uint8_t> padParams;
        if constexpr (transB) {
            dataCopyParams.blockCount = Cmct::Gemm::CeilDiv(tileL1L0Param.curGmBKL1, C0_SIZE);
            dataCopyParams.blockLen = tileL1L0Param.curAlignN * C0_SIZE;
            if constexpr (IS_FP4) {
                dataCopyParams.blockLen = dataCopyParams.blockLen >> 1;
            }
            dataCopyParams.srcStride = (Cmct::Gemm::CeilAlign(n_, AscendC::BLOCK_CUBE) - tileL1L0Param.curAlignN) *
                                       C0_SIZE;
            if constexpr (IS_FP4) {
                dataCopyParams.srcStride = dataCopyParams.srcStride >> 1;
            }
            dataCopyParams.dstStride = 0;
        } else {
            int64_t curGmBKL1NZ = Cmct::Gemm::CeilAlign(tileL1L0Param.curGmBKL1, AscendC::BLOCK_CUBE);
            dataCopyParams.blockCount = tileL1L0Param.curAlignN / C0_SIZE;
            dataCopyParams.blockLen = curGmBKL1NZ * C0_SIZE;
            if constexpr (IS_FP4) {
                dataCopyParams.blockLen = dataCopyParams.blockLen >> 1;
            }
            dataCopyParams.srcStride = (Cmct::Gemm::CeilAlign(k_, AscendC::BLOCK_CUBE) - curGmBKL1NZ) * C0_SIZE;
            if constexpr (IS_FP4) {
                dataCopyParams.srcStride = dataCopyParams.srcStride >> 1;
            }
            dataCopyParams.dstStride = tileL1L0Param.curPadBKL1 - curGmBKL1NZ;
        }
        AscendC::DataCopyPad(bl1LocalUint8, bGlobalUInt8, dataCopyParams, padParams, true);
    }

    __aicore__ inline void InitA1(const AscendC::LocalTensor<AType>& al1Local, TileL1L0Param& tileL1L0Param)
    {
        AscendC::LocalTensor<half> al1LocalHalf = al1Local.template ReinterpretCast<half>();
        AscendC::InitConstValueParams<half> initConstValueParams;
        uint64_t offset = 0;
        if constexpr (!transA) {
            // nd2nz pading to 64 align
            if (!SetInitParamsMTE1NotTransA(tileL1L0Param, initConstValueParams, offset)) {
                return;
            }
        } else {
            if (dn2nzMTE2TransA_) {
                if (!SetInitParamsMTE1NotTransA(tileL1L0Param, initConstValueParams, offset)) {
                    return;
                }
            } else {
                if (tileL1L0Param.curPadAKL1 == tileL1L0Param.curGmAKL1) {
                    return;
                }
                uint64_t m1 = Cmct::Gemm::CeilDiv(tileL1L0Param.curAlignM, C0_SIZE);
                offset = tileL1L0Param.curGmAKL1 * AscendC::BLOCK_CUBE;
                initConstValueParams.repeatTimes = m1;
                initConstValueParams.blockNum = tileL1L0Param.curPadAKL1 - tileL1L0Param.curGmAKL1;
                initConstValueParams.dstGap = tileL1L0Param.curGmAKL1;
                initConstValueParams.initValue = 0;
            }
        }
        AscendC::InitConstValue(al1LocalHalf[offset], initConstValueParams);
    }

    __aicore__ inline bool SetInitParamsMTE1NotTransA(const TileL1L0Param& tileL1L0Param,
                                                      AscendC::InitConstValueParams<half>& initConstValueParams,
                                                      uint64_t& offset)
    {
        if (tileL1L0Param.curPadAKL1 - tileL1L0Param.curGmAKL1 < BLOCK_REDUCE_CUBE) {
            return false;
        }
        // pad m1, m0, 16 for half
        initConstValueParams.repeatTimes = 1;
        initConstValueParams.blockNum = tileL1L0Param.curAlignM;
        initConstValueParams.dstGap = 0;
        initConstValueParams.initValue = 0;
        uint64_t kAlign = Cmct::Gemm::CeilDiv(tileL1L0Param.curGmAKL1, C0_SIZE) * AscendC::BLOCK_CUBE;
        offset = tileL1L0Param.curAlignM * kAlign;
        return true;
    }

    __aicore__ inline void InitB1(const AscendC::LocalTensor<BType>& bl1Local, TileL1L0Param& tileL1L0Param)
    {
        // Equivalent to curKL1 % MXFP_DIVISOR_SIZE
        AscendC::LocalTensor<half> bl1LocalHalf = bl1Local.template ReinterpretCast<half>();
        AscendC::InitConstValueParams<half> initConstValueParams;
        uint64_t offset = 0;
        if constexpr (transB) {
            // nd2nz pading to 64 align
            if (!SetInitParamsMTE1NotTransB(tileL1L0Param, initConstValueParams, offset)) {
                return;
            }
        } else {
            if (dn2nzMTE2TransB_) {
                if (!SetInitParamsMTE1NotTransB(tileL1L0Param, initConstValueParams, offset)) {
                    return;
                }
            } else {
                if constexpr (formatB == CubeFormat::NZ) {
                    if (tileL1L0Param.curPadBKL1 ==
                        Cmct::Gemm::CeilAlign(tileL1L0Param.curGmBKL1, AscendC::BLOCK_CUBE)) {
                        return;
                    }
                } else {
                    if (tileL1L0Param.curPadBKL1 == tileL1L0Param.curGmBKL1) {
                        return;
                    }
                }
                // when format of B is NZ, we reuse code for ND even though we initialize unnecessary extra space
                uint64_t n1 = Cmct::Gemm::CeilDiv(tileL1L0Param.curAlignN, C0_SIZE);
                offset = tileL1L0Param.curGmBKL1 * AscendC::BLOCK_CUBE;
                initConstValueParams.repeatTimes = n1;
                initConstValueParams.blockNum = tileL1L0Param.curPadBKL1 - tileL1L0Param.curGmBKL1;
                initConstValueParams.dstGap = tileL1L0Param.curGmBKL1;
                initConstValueParams.initValue = 0;
            }
        }
        AscendC::InitConstValue(bl1LocalHalf[offset], initConstValueParams);
    }

    __aicore__ inline bool SetInitParamsMTE1NotTransB(const TileL1L0Param& tileL1L0Param,
                                                      AscendC::InitConstValueParams<half>& initConstValueParams,
                                                      uint64_t& offset)
    {
        if (tileL1L0Param.curPadBKL1 - tileL1L0Param.curGmBKL1 < BLOCK_REDUCE_CUBE) {
            return false;
        }
        // pad n1, n0, 16 for half
        initConstValueParams.repeatTimes = 1;
        initConstValueParams.blockNum = tileL1L0Param.curAlignN;
        initConstValueParams.dstGap = 0;
        initConstValueParams.initValue = 0;
        uint64_t kAlign = Cmct::Gemm::CeilDiv(tileL1L0Param.curGmBKL1, C0_SIZE) * AscendC::BLOCK_CUBE;
        offset = tileL1L0Param.curAlignN * kAlign;
        return true;
    }

    __aicore__ inline void CopyInBias(const AscendC::GlobalTensor<BiasType>& biasGlobal,
                                      const AscendC::LocalTensor<BiasType>& cl1Local, uint64_t curNL1)
    {
        // No need to add sync flag for bias L1 loading because bias loading operation can be covered by
        // A/B/ScaleA/ScaleB load.
        AscendC::DataCopyPadParams padParams;
        // 单位为Byte
        AscendC::DataCopyParams biasParam{1, static_cast<uint16_t>(curNL1 * sizeof(BiasType)), 0, 0};
        AscendC::DataCopyPad(cl1Local, biasGlobal, biasParam, padParams, true);
    }

    __aicore__ inline void CopyInScaleA(const GlobalTensor<ScaleAType>& aScaleGlobal,
                                        const LocalTensor<ScaleAType>& aScaleL1Local, uint64_t curML1, uint64_t curKL1,
                                        uint64_t kL1Offset)
    {
        if (DispatchPolicy::fullLoadMode != 0 && kL1Offset != 0) {
            return;
        }
        uint64_t curScaleKL1 = curKL1;
        if (kL1Offset + curScaleKL1 > k_) {
            curScaleKL1 = k_ - kL1Offset;
        }
        uint64_t nDim = transA ? Cmct::Gemm::CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE) : curML1;
        uint64_t dDim = transA ? curML1 : Cmct::Gemm::CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE);

        uint64_t offsetScaleAGM = !transA ? kL1Offset / MXFP_DIVISOR_SIZE : (kL1Offset / MXFP_DIVISOR_SIZE) * m_;

        GlobalTensor<half> aScaleGlobalB16;
        aScaleGlobalB16.SetGlobalBuffer(((__gm__ half*)(aScaleGlobal.GetPhyAddr())));
        auto aScaleL1LocalImpl = aScaleL1Local.template ReinterpretCast<half>();

        if constexpr (!transA) {
            AscendC::Dn2NzParams dn2nzParams;
            dn2nzParams.dnNum = 1;
            if constexpr (IS_HIF4) {
                offsetScaleAGM = offsetScaleAGM << HIF4_SCALE_NUM_HALF_RATIO;
                dn2nzParams.dValue = nDim;
                dn2nzParams.nValue = dDim << HIF4_SCALE_NUM_HALF_RATIO;
                dn2nzParams.srcDnMatrixStride = 0;
                dn2nzParams.srcDValue = Cmct::Gemm::CeilDiv(k_, HIF4_GROUP_SIZE) << HIF4_SCALE_NUM_HALF_RATIO;
                dn2nzParams.dstNzC0Stride = Cmct::Gemm::CeilDiv(curKL1, HIF4_GROUP_SIZE) << HIF4_SCALE_NUM_HALF_RATIO;
            } else {
                dn2nzParams.dValue = nDim;
                dn2nzParams.nValue = dDim;
                dn2nzParams.srcDnMatrixStride = 0;
                dn2nzParams.srcDValue = Cmct::Gemm::CeilDiv(k_, MXFP_DIVISOR_SIZE);
                dn2nzParams.dstNzC0Stride = Cmct::Gemm::CeilDiv(curKL1, MXFP_DIVISOR_SIZE);
            }
            dn2nzParams.dstNzNStride = 1;
            dn2nzParams.dstNzMatrixStride = 0;
            AscendC::DataCopy(aScaleL1LocalImpl, aScaleGlobalB16[offsetScaleAGM], dn2nzParams, true);
        } else {
            AscendC::Nd2NzParams nd2nzParams;
            nd2nzParams.ndNum = 1;
            if constexpr (IS_HIF4) {
                offsetScaleAGM = offsetScaleAGM << HIF4_SCALE_NUM_HALF_RATIO;
                nd2nzParams.nValue = nDim << HIF4_SCALE_NUM_HALF_RATIO;
                nd2nzParams.dValue = dDim;
                nd2nzParams.srcNdMatrixStride = 0;
                nd2nzParams.srcDValue = m_;
                nd2nzParams.dstNzC0Stride = Cmct::Gemm::CeilDiv(curKL1, MXFP_DIVISOR_SIZE) << HIF4_SCALE_NUM_HALF_RATIO;
            } else {
                nd2nzParams.nValue = nDim;
                nd2nzParams.dValue = dDim;
                nd2nzParams.srcNdMatrixStride = 0;
                nd2nzParams.srcDValue = m_;
                nd2nzParams.dstNzC0Stride = Cmct::Gemm::CeilDiv(curKL1, MXFP_DIVISOR_SIZE);
            }
            nd2nzParams.dstNzNStride = 1;
            nd2nzParams.dstNzMatrixStride = 0;
            AscendC::DataCopy(aScaleL1LocalImpl, aScaleGlobalB16[offsetScaleAGM], nd2nzParams, true);
        }
    }

    __aicore__ inline void CopyInScaleB(const GlobalTensor<ScaleBType>& bScaleGlobal,
                                        const LocalTensor<ScaleBType>& bScaleL1Local, uint64_t curNL1,
                                        uint64_t kL1Offset)
    {
        uint64_t curScaleKL1 = scaleKL1_;
        if (kL1Offset + curScaleKL1 > k_) {
            curScaleKL1 = k_ - kL1Offset;
        }
        uint64_t nDim = transB ? curNL1 : Cmct::Gemm::CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE);
        uint64_t dDim = transB ? Cmct::Gemm::CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE) : curNL1;

        GlobalTensor<half> bScaleGlobalB16;
        bScaleGlobalB16.SetGlobalBuffer(((__gm__ half*)(bScaleGlobal.GetPhyAddr())));
        auto bScaleL1LocalImpl = bScaleL1Local.template ReinterpretCast<half>();

        uint64_t offsetScaleBGM = transB ? kL1Offset / MXFP_DIVISOR_SIZE : (kL1Offset / MXFP_DIVISOR_SIZE) * n_;

        if constexpr (transB) {
            AscendC::Dn2NzParams dn2nzParams;
            dn2nzParams.dnNum = 1;
            if constexpr (IS_HIF4) {
                offsetScaleBGM = offsetScaleBGM << HIF4_SCALE_NUM_HALF_RATIO;
                dn2nzParams.dValue = nDim;
                dn2nzParams.nValue = dDim << HIF4_SCALE_NUM_HALF_RATIO;
                dn2nzParams.srcDnMatrixStride = 0;
                dn2nzParams.srcDValue = Cmct::Gemm::CeilDiv(k_, HIF4_GROUP_SIZE) << HIF4_SCALE_NUM_HALF_RATIO;
                dn2nzParams.dstNzC0Stride = scaleKL1Num_ << HIF4_SCALE_NUM_HALF_RATIO;
            } else {
                dn2nzParams.dValue = nDim;
                dn2nzParams.nValue = dDim;
                dn2nzParams.srcDnMatrixStride = 0;
                dn2nzParams.srcDValue = Cmct::Gemm::CeilDiv(k_, MXFP_DIVISOR_SIZE);
                dn2nzParams.dstNzC0Stride = Cmct::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE);
            }
            dn2nzParams.dstNzNStride = 1;
            dn2nzParams.dstNzMatrixStride = 0;
            AscendC::DataCopy(bScaleL1LocalImpl, bScaleGlobalB16[offsetScaleBGM], dn2nzParams, true);
        } else {
            AscendC::Nd2NzParams nd2nzParams;
            nd2nzParams.ndNum = 1;
            if constexpr (IS_HIF4) {
                offsetScaleBGM = offsetScaleBGM << HIF4_SCALE_NUM_HALF_RATIO;
                nd2nzParams.nValue = nDim << HIF4_SCALE_NUM_HALF_RATIO;
                nd2nzParams.dValue = dDim;
                nd2nzParams.srcNdMatrixStride = 0;
                nd2nzParams.srcDValue = n_;
                nd2nzParams.dstNzC0Stride = scaleKL1Num_ << HIF4_SCALE_NUM_HALF_RATIO;
            } else {
                nd2nzParams.nValue = nDim;
                nd2nzParams.dValue = dDim;
                nd2nzParams.srcNdMatrixStride = 0;
                nd2nzParams.srcDValue = n_;
                nd2nzParams.dstNzC0Stride = Cmct::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE);
            }
            nd2nzParams.dstNzNStride = 1;
            nd2nzParams.dstNzMatrixStride = 0;
            AscendC::DataCopy(bScaleL1LocalImpl, bScaleGlobalB16[offsetScaleBGM], nd2nzParams, true);
        }
    }

    __aicore__ inline void CopyInC2(const AscendC::LocalTensor<BiasType>& biasL1Local,
                                    const AscendC::LocalTensor<float>& biasBt, uint64_t nl1Align, bool needBias)
    {
        if (!needBias) {
            return;
        }
        // s32场景要对齐到2 因此是align(nl1Align / 8, 2)
        uint64_t btAlign = AscendC::BLOCK_CUBE / BIAS_C0;
        uint16_t bustLenth = Cmct::Gemm::Align(nl1Align / BIAS_C0, btAlign);
        AscendC::DataCopyParams biasParam{1, static_cast<uint16_t>(bustLenth), 0, 0};
        // 当dstlocal位于C2时，C2中至少为fp32*16
        AscendC::DataCopy(biasBt, biasL1Local, biasParam);
    }

    __aicore__ inline void CopyInL0A(const AscendC::LocalTensor<MxL0AType>& l0aLocal,
                                     const AscendC::LocalTensor<AType>& al1Local,
                                     const AscendC::LocalTensor<ScaleAType>& scaleAl1Local, uint64_t iter,
                                     TileL1L0Param& tileL1L0Param, uint64_t curScaleKL1, uint64_t realMte1M)
    {
        AscendC::LoadData2DParamsV2 loadDataParams;
        AscendC::LoadData2DMxParams loadData2DMxParams;
        uint64_t realMStep = Cmct::Gemm::CeilDiv(realMte1M, AscendC::BLOCK_CUBE);
        uint64_t srcStrideA = Cmct::Gemm::CeilDiv(tileL1L0Param.curM, AscendC::BLOCK_CUBE);
        if constexpr (!transA) {
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = Cmct::Gemm::CeilDiv(iter * baseK_, C0_SIZE);
            loadDataParams.mStep = realMStep;
            loadDataParams.kStep = Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, C0_SIZE);
            loadDataParams.srcStride = srcStrideA;
            loadDataParams.dstStride = loadDataParams.mStep;
            loadDataParams.ifTranspose = false;
        } else if (dn2nzMTE2TransA_) {
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = Cmct::Gemm::CeilDiv(iter * baseK_, C0_SIZE);
            loadDataParams.mStep = realMStep;
            loadDataParams.kStep = Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, C0_SIZE);
            loadDataParams.srcStride = srcStrideA;
            loadDataParams.dstStride = loadDataParams.mStep;
            loadDataParams.ifTranspose = false;
        } else {
            loadDataParams.mStartPosition = Cmct::Gemm::CeilDiv(iter * baseK_, AscendC::BLOCK_CUBE);
            loadDataParams.kStartPosition = 0;
            if (((realMStep & 1) == 0 && !(IS_FP4)) ||
                (IS_FP4 && Cmct::Gemm::CeilDiv(tileL1L0Param.curM, AscendC::BLOCK_CUBE) % B4_MIN_STEP == 0)) {
                loadDataParams.mStep = Cmct::Gemm::CeilAlign(
                    Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, AscendC::BLOCK_CUBE), MXFP_MULTI_BASE_SIZE);
            } else {
                loadDataParams.mStep = MIN_STEP;
            }
            loadDataParams.kStep = Cmct::Gemm::CeilDiv(realMte1M, C0_SIZE);
            loadDataParams.srcStride = Cmct::Gemm::CeilDiv(tileL1L0Param.curPadAKL1, AscendC::BLOCK_CUBE);
            loadDataParams.dstStride = realMStep;
            loadDataParams.ifTranspose = true;
        }
        loadData2DMxParams.xStartPosition = 0;
        loadData2DMxParams.xStep = realMStep;
        if constexpr (IS_HIF4) {
            // ystartPosition, ystep 的单位为32B
            // hifp4 scale的相邻元素k方向的offset为16*4B = 64B(16是m方向分型大小), ystep 的单位为32B,
            // 所以scale的 y方向offset 数值是 元素个数的2倍
            loadData2DMxParams.yStartPosition = Cmct::Gemm::CeilDiv(iter * baseK_, HIF4_GROUP_SIZE)
                                                << HIF4_SCALE_NUM_HALF_RATIO;
            loadData2DMxParams.yStep = Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, MXFP_DIVISOR_SIZE)
                                       << HIF4_SCALE_NUM_HALF_RATIO;
            loadData2DMxParams.srcStride = Cmct::Gemm::CeilDiv(curScaleKL1, HIF4_GROUP_SIZE)
                                           << HIF4_SCALE_NUM_HALF_RATIO;
            loadData2DMxParams.dstStride = loadData2DMxParams.yStep;
        } else {
            loadData2DMxParams.yStartPosition = Cmct::Gemm::CeilDiv(iter * baseK_, MXFP_DIVISOR_SIZE);
            loadData2DMxParams.yStep = Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, MXFP_DIVISOR_SIZE);
            loadData2DMxParams.srcStride = Cmct::Gemm::CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE);
            loadData2DMxParams.dstStride = loadData2DMxParams.yStep;
        }
        AscendC::LoadData(l0aLocal, al1Local, scaleAl1Local, loadDataParams, loadData2DMxParams);
        if constexpr (transA) {
            // ifTranspose 为 true 即走了 MTE1 转置分支（B8 奇数步由 MTE2 转置分支覆盖），
            // 仅该分支存在按 MIN_STEP 拆指令补齐剩余行的场景
            if (((realMStep & 1) != 0 && loadDataParams.ifTranspose) ||
                (IS_FP4 && Cmct::Gemm::CeilDiv(tileL1L0Param.curM, AscendC::BLOCK_CUBE) % B4_MIN_STEP != 0)) {
                PipeBarrier<PIPE_MTE1>();
                LocalTensor<AType> l0a = l0aLocal.template ReinterpretCast<AType>();
                uint64_t loadTimes = Cmct::Gemm::CeilDiv(Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, BLOCK_CUBE),
                                                         MIN_STEP);
                if constexpr (IS_FP4) {
                    loadData2DMxParams.xStartPosition = 0;
                    loadData2DMxParams.yStartPosition = 0;
                    loadData2DMxParams.xStep = 0;
                    loadData2DMxParams.yStep = 0;
                    loadData2DMxParams.srcStride = 0;
                    loadData2DMxParams.dstStride = 0;
                    for (uint64_t i = 1; i < loadTimes; i++) {
                        loadDataParams.mStartPosition = MIN_STEP * i + Cmct::Gemm::CeilDiv(iter * baseK_, BLOCK_CUBE);
                        AscendC::LoadData(l0a[i * realMStep * BLOCK_CUBE * C0_SIZE], al1Local, scaleAl1Local,
                                          loadDataParams, loadData2DMxParams);
                        PipeBarrier<PIPE_MTE1>();
                    }
                } else {
                    for (uint64_t i = 1; i < loadTimes; i++) {
                        loadDataParams.mStartPosition = MIN_STEP * i + Cmct::Gemm::CeilDiv(iter * baseK_, BLOCK_CUBE);
                        AscendC::LoadData(l0a[i * realMStep * BLOCK_CUBE * C0_SIZE], al1Local, loadDataParams);
                        PipeBarrier<PIPE_MTE1>();
                    }
                }
            }
        }
    }

    __aicore__ inline void CopyInL0B(const AscendC::LocalTensor<MxL0BType>& l0bLocal,
                                     const AscendC::LocalTensor<BType>& bl1Local,
                                     const AscendC::LocalTensor<ScaleBType>& scaleBl1Local, uint64_t iter,
                                     TileL1L0Param& tileL1L0Param, uint64_t realMte1N)
    {
        AscendC::LoadData2DParamsV2 loadDataParams;
        AscendC::LoadData2DMxParams loadData2DMxParams;
        uint64_t realNStep = Cmct::Gemm::CeilDiv(realMte1N, AscendC::BLOCK_CUBE);
        uint64_t srcStrideB = Cmct::Gemm::CeilDiv(tileL1L0Param.curN, AscendC::BLOCK_CUBE);
        if constexpr (transB) {
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = Cmct::Gemm::CeilDiv(iter * baseK_, C0_SIZE);
            loadDataParams.mStep = realNStep;
            loadDataParams.kStep = Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, C0_SIZE);
            loadDataParams.srcStride = srcStrideB;
            loadDataParams.dstStride = loadDataParams.mStep;
            loadDataParams.ifTranspose = false;
        } else if (dn2nzMTE2TransB_) {
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = Cmct::Gemm::CeilDiv(iter * baseK_, C0_SIZE);
            loadDataParams.mStep = realNStep;
            loadDataParams.kStep = Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, C0_SIZE);
            loadDataParams.srcStride = srcStrideB;
            loadDataParams.dstStride = loadDataParams.mStep;
            loadDataParams.ifTranspose = false;
        } else {
            loadDataParams.mStartPosition = Cmct::Gemm::CeilDiv(iter * baseK_, AscendC::BLOCK_CUBE);
            loadDataParams.kStartPosition = 0;
            if (((realNStep & 1) == 0 && !(IS_FP4)) ||
                (IS_FP4 && Cmct::Gemm::CeilDiv(tileL1L0Param.curN, AscendC::BLOCK_CUBE) % B4_MIN_STEP == 0)) {
                loadDataParams.mStep = Cmct::Gemm::CeilAlign(
                    Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, AscendC::BLOCK_CUBE), EVEN_FACTOR);
            } else {
                loadDataParams.mStep = MIN_STEP;
            }
            loadDataParams.kStep = Cmct::Gemm::CeilDiv(realMte1N, C0_SIZE);
            loadDataParams.srcStride = Cmct::Gemm::CeilDiv(tileL1L0Param.curPadBKL1, AscendC::BLOCK_CUBE);
            loadDataParams.dstStride = realNStep;
            loadDataParams.ifTranspose = true;
        }
        loadData2DMxParams.xStartPosition = 0;
        loadData2DMxParams.xStep = realNStep;
        if constexpr (IS_HIF4) {
            // ystartPosition, ystep 的单位为32B
            // hifp4 scale的相邻元素k方向的offset为16*4B = 64B(16是m方向分型大小), ystep 的单位为32B,
            // 所以scale的 y方向offset 数值是 元素个数的2倍
            loadData2DMxParams.yStartPosition = Cmct::Gemm::CeilDiv(iter * baseK_, HIF4_GROUP_SIZE)
                                                << HIF4_SCALE_NUM_HALF_RATIO;
            loadData2DMxParams.yStep = Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, MXFP_DIVISOR_SIZE)
                                       << HIF4_SCALE_NUM_HALF_RATIO;
            loadData2DMxParams.srcStride = scaleKL1Num_ << HIF4_SCALE_NUM_HALF_RATIO;
            loadData2DMxParams.dstStride = loadData2DMxParams.yStep;
        } else {
            loadData2DMxParams.yStartPosition = Cmct::Gemm::CeilDiv(iter * baseK_, MXFP_DIVISOR_SIZE);
            loadData2DMxParams.yStep = Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, MXFP_DIVISOR_SIZE);
            loadData2DMxParams.srcStride = Cmct::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE);
            loadData2DMxParams.dstStride = loadData2DMxParams.yStep;
        }
        AscendC::LoadData(l0bLocal, bl1Local, scaleBl1Local, loadDataParams, loadData2DMxParams);
        if constexpr (!transB) {
            // ifTranspose 为 true 即走了 MTE1 转置分支（B8 奇数步由 MTE2 转置分支覆盖），
            // 仅该分支存在按 MIN_STEP 拆指令补齐剩余行的场景
            if (((realNStep & 1) != 0 && loadDataParams.ifTranspose) ||
                (IS_FP4 && Cmct::Gemm::CeilDiv(tileL1L0Param.curN, AscendC::BLOCK_CUBE) % B4_MIN_STEP != 0)) {
                PipeBarrier<PIPE_MTE1>();
                LocalTensor<BType> l0b = l0bLocal.template ReinterpretCast<BType>();
                uint64_t loadTimes = Cmct::Gemm::CeilDiv(Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, BLOCK_CUBE),
                                                         MIN_STEP);
                if constexpr (IS_FP4) {
                    loadData2DMxParams.xStartPosition = 0;
                    loadData2DMxParams.yStartPosition = 0;
                    loadData2DMxParams.xStep = 0;
                    loadData2DMxParams.yStep = 0;
                    loadData2DMxParams.srcStride = 0;
                    loadData2DMxParams.dstStride = 0;
                    for (uint64_t i = 1; i < loadTimes; i++) {
                        loadDataParams.mStartPosition = MIN_STEP * i + Cmct::Gemm::CeilDiv(iter * baseK_, BLOCK_CUBE);
                        AscendC::LoadData(l0b[i * realNStep * AscendC::BLOCK_CUBE * C0_SIZE], bl1Local, scaleBl1Local,
                                          loadDataParams, loadData2DMxParams);
                        PipeBarrier<PIPE_MTE1>();
                    }
                } else {
                    for (uint64_t i = 1; i < loadTimes; i++) {
                        loadDataParams.mStartPosition = MIN_STEP * i + Cmct::Gemm::CeilDiv(iter * baseK_, BLOCK_CUBE);
                        AscendC::LoadData(l0b[i * realNStep * AscendC::BLOCK_CUBE * C0_SIZE], bl1Local, loadDataParams);
                        PipeBarrier<PIPE_MTE1>();
                    }
                }
            }
        }
    }

    __aicore__ inline void CopyOut(const AscendC::GlobalTensor<CType>& cGlobal, AscendC::LocalTensor<float>& c1Local,
                                   uint64_t baseM, uint64_t baseN)
    {
        AscendC::DataCopyCO12DstParams intriParams;
        intriParams.nSize = baseN;
        intriParams.mSize = baseM;
        intriParams.dstStride = n_;
        intriParams.srcStride = Cmct::Gemm::Align(baseM, AscendC::BLOCK_CUBE);
        // set mode according to dtype
        if constexpr (AscendC::IsSameType<CType, bfloat16_t>::value) {
            intriParams.quantPre = QuantMode_t::F322BF16;
        } else if (AscendC::IsSameType<CType, half>::value) {
            intriParams.quantPre = QuantMode_t::F322F16;
        } else if (AscendC::IsSameType<CType, float>::value) {
            intriParams.quantPre = QuantMode_t::NoQuant;
        }
        intriParams.nz2ndEn = true;
        intriParams.unitFlag = FINAL_ACCUMULATION; // 3 unitflag
        AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
        AscendC::DataCopy(cGlobal, c1Local, intriParams);
    }

    __aicore__ inline void UpdateKL1(TileL1L0Param& tileL1L0Param, uint64_t iter0)
    {
        tileL1L0Param.curGmBKL1 = (iter0 + 1 == kL1Iter_) ? (k_ - iter0 * kL1_) : kL1_;
        tileL1L0Param.curPadBKL1 = Cmct::Gemm::CeilAlign(tileL1L0Param.curGmBKL1, MXFP_DIVISOR_SIZE);
        tileL1L0Param.curGmAKL1 = tileL1L0Param.curGmBKL1;
        tileL1L0Param.curPadAKL1 = tileL1L0Param.curPadBKL1; // pad to 64 align
    }

    __aicore__ inline void UpdateKL0(TileL1L0Param& tileL1L0Param, uint64_t iter1)
    {
        if (iter1 * baseK_ + baseK_ > tileL1L0Param.curPadBKL1) {
            tileL1L0Param.curKL0 = tileL1L0Param.curPadBKL1 - iter1 * baseK_;
        } else {
            tileL1L0Param.curKL0 = baseK_;
        }
    }

    __aicore__ inline void GetAlignMN(TileL1L0Param& tileL1L0Param)
    {
        if constexpr (transA) {
            tileL1L0Param.curAlignM = Cmct::Gemm::CeilAlign(tileL1L0Param.curM,
                                                            dn2nzMTE2TransA_ ? BLOCK_CUBE : C0_SIZE);
        } else {
            tileL1L0Param.curAlignM = Cmct::Gemm::CeilAlign(tileL1L0Param.curM, BLOCK_CUBE);
        }
        if constexpr (!transB) {
            tileL1L0Param.curAlignN = Cmct::Gemm::CeilAlign(tileL1L0Param.curN,
                                                            dn2nzMTE2TransB_ ? BLOCK_CUBE : C0_SIZE);
        } else {
            tileL1L0Param.curAlignN = Cmct::Gemm::CeilAlign(tileL1L0Param.curN, BLOCK_CUBE);
        }
    }

    __aicore__ inline void CopyScalesInL1(AscendC::GlobalTensor<ScaleAType>& scaleAGlobal,
                                          AscendC::GlobalTensor<ScaleBType>& scaleBGlobal, TileL1L0Param& tileL1L0Param,
                                          uint64_t l1Iter, uint64_t scaleL1BufId)
    {
        uint64_t kL1Offset = l1Iter * kL1_;
        if constexpr (DispatchPolicy::fullLoadMode == 0) {
            if (l1Iter % (scaleKL1_ / kL1_) == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + (scaleL1BufId));
                CopyInScaleA(scaleAGlobal, scaleAL1Local_[l1BufferScaleAOffset_[scaleL1BufId]], tileL1L0Param.curM,
                             scaleKL1_, kL1Offset);
                CopyInScaleB(scaleBGlobal, scaleBL1Local_[l1BufferScaleBOffset_[scaleL1BufId]], tileL1L0Param.curN,
                             kL1Offset);
            }
        } else {
            if (l1Iter % (scaleKL1_ / kL1_) == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + (scaleL1BufId));
                CopyInScaleB(scaleBGlobal, scaleBL1Local_[l1BufferScaleBOffset_[scaleL1BufId]], tileL1L0Param.curN,
                             kL1Offset);
            }
            if (abL1LoopCnt_ == 0) {
                CopyInScaleA(scaleAGlobal, scaleAL1Local_[l1BufferScaleAOffset_[0]], tileL1L0Param.curM, k_, kL1Offset);
            }
        }
    }

    __aicore__ inline void CopyAInL1(AscendC::GlobalTensor<AType> aGlobal, TileL1L0Param tileL1L0Param,
                                     uint64_t offsetA, uint64_t offsetAL1, uint64_t l1Iter)
    {
        if (dn2nzMTE2TransA_) {
            if constexpr (DispatchPolicy::fullLoadMode != 0) {
                offsetAL1 = l1BufferAOffset_[0] + l1Iter * kL1_ * tileL1L0Param.curAlignM;
            }
            InitA1(aL1Local_[offsetAL1], tileL1L0Param);
            CopyInA1Dn2NZ(aGlobal[offsetA], aL1Local_[offsetAL1], tileL1L0Param);
            return;
        }

        if constexpr (DispatchPolicy::fullLoadMode == 0) {
            if constexpr (!(IS_FP4 && !transA)) {
                InitA1(aL1Local_[offsetAL1], tileL1L0Param);
            }
            CopyInA1(aGlobal[offsetA], aL1Local_[offsetAL1], tileL1L0Param);
        } else {
            offsetAL1 = l1BufferAOffset_[0] + l1Iter * kL1_ * tileL1L0Param.curAlignM;
            if (abL1LoopCnt_ < kL1Iter_) {
                if constexpr (!(IS_FP4 && !transA)) {
                    InitA1(aL1Local_[offsetAL1], tileL1L0Param);
                }
                CopyInA1(aGlobal[offsetA], aL1Local_[offsetAL1], tileL1L0Param);
            }
        }
    }

    __aicore__ inline void CopyBInL1(AscendC::GlobalTensor<BType> bGlobal, TileL1L0Param tileL1L0Param,
                                     uint64_t l1BufId, uint64_t l1Iter)
    {
        if constexpr (!(IS_FP4 && transB)) {
            InitB1(bL1Local_[l1BufferBOffset_[l1BufId]], tileL1L0Param);
        }
        if constexpr (formatB == CubeFormat::NZ) {
            uint64_t offsetB = transB ? l1Iter * kL1_ * Cmct::Gemm::CeilAlign(n_, AscendC::BLOCK_CUBE) :
                                        l1Iter * kL1_ * C0_SIZE;
            CopyInB1WeightNz(bGlobal[offsetB], bL1Local_[l1BufferBOffset_[l1BufId]], tileL1L0Param);
        } else {
            uint64_t offsetB = transB ? l1Iter * kL1_ : l1Iter * kL1_ * n_;
            if (dn2nzMTE2TransB_) {
                CopyInB1Dn2NZ(bGlobal[offsetB], bL1Local_[l1BufferBOffset_[l1BufId]], tileL1L0Param);
            } else {
                CopyInB1(bGlobal[offsetB], bL1Local_[l1BufferBOffset_[l1BufId]], tileL1L0Param);
            }
        }
    }

    __aicore__ inline void Iterate(TileL1L0Param& tileL1L0Param, MmadParams& mmadParams, uint64_t l1Iter,
                                   uint64_t l1BufId, uint64_t scaleL1BufId, uint64_t offsetAl1, uint64_t l0cOffset)
    {
        uint64_t kL0Iter = Cmct::Gemm::CeilDiv(tileL1L0Param.curGmBKL1, baseK_);
        uint64_t offsetScaleL1 = ComputeScaleL1Offset(l1Iter);
        for (uint16_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
            UpdateKL0(tileL1L0Param, iter1);
            // Load data to L0 and open DB
            uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
            if constexpr (DispatchPolicy::fullLoadMode == 0) {
                CopyInL0A(l0aLocal_[l0Offset], aL1Local_[offsetAl1],
                          scaleAL1Local_[l1BufferScaleAOffset_[scaleL1BufId] + offsetScaleL1], iter1, tileL1L0Param,
                          scaleKL1_, tileL1L0Param.curM);
            } else {
                offsetAl1 = l1BufferAOffset_[0] + l1Iter * kL1_ * tileL1L0Param.curAlignM;
                uint64_t offsetScaleAL1 = 0;
                if constexpr (IS_HIF4) {
                    offsetScaleAL1 = BLOCK_CUBE * l1Iter * (kL1_ / HIF4_GROUP_SIZE);
                } else {
                    offsetScaleAL1 = BLOCK_CUBE * l1Iter * (kL1_ / MXFP_GROUP_SIZE);
                }
                CopyInL0A(l0aLocal_[l0Offset], aL1Local_[offsetAl1],
                          scaleAL1Local_[l1BufferScaleAOffset_[0] + offsetScaleAL1], iter1, tileL1L0Param, k_,
                          tileL1L0Param.curM);
            }
            // copy bias to bt
            CopyInC2(biasL1Local_[l1BufferBiasOffset_[biasBufId_] / sizeof(BiasType)], biasBt_[baseN_ * biasBufId_],
                     Cmct::Gemm::Align(mmadParams.n, AscendC::BLOCK_CUBE), NeedBias(l1Iter, iter1));
            CopyInL0B(l0bLocal_[l0Offset], bL1Local_[l1BufferBOffset_[l1BufId]],
                      scaleBL1Local_[l1BufferScaleBOffset_[scaleL1BufId] + offsetScaleL1], iter1, tileL1L0Param,
                      tileL1L0Param.curN);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
            mmadParams.k = Cmct::Gemm::CeilAlign(tileL1L0Param.curKL0, MXFP_DIVISOR_SIZE);
            mmadParams.unitFlag = (l1Iter + 1 == kL1Iter_ && iter1 + 1 == kL0Iter) ? FINAL_ACCUMULATION :
                                                                                     NON_FINAL_ACCUMULATION;
            mmadParams.cmatrixInitVal = (l1Iter == 0 && iter1 == 0 && !isBias_);
            Mmad(mmadParams, l0cOffset, l0Offset, l0Offset, baseN_ * biasBufId_, NeedBias(l1Iter, iter1));
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
            l0PingPong_++;
        }
    }

    __aicore__ inline uint64_t ComputeScaleL1Offset(uint64_t l1Iter) const
    {
        uint64_t offsetScaleL1;
        if constexpr (IS_HIF4) {
            offsetScaleL1 = AscendC::BLOCK_CUBE * (l1Iter % (scaleKL1_ / kL1_)) * (kL1_ / HIF4_GROUP_SIZE);
        } else {
            offsetScaleL1 = AscendC::BLOCK_CUBE * (l1Iter % (scaleKL1_ / kL1_)) * (kL1_ / MXFP_GROUP_SIZE);
        }
        return offsetScaleL1;
    }

    struct OffsetsForSplitMN {
        uint64_t offsetABL1;
        uint64_t offsetScaleABL1;
    };

    template <bool isScaleA>
    __aicore__ inline OffsetsForSplitMN ComputeOffsetsForSplitMN(uint64_t kL1Iter, uint64_t curAlignM,
                                                                 uint64_t mnLoopId, uint64_t baseMN0,
                                                                 uint64_t scaleKl1) const
    {
        // MTE1需要拆分M时，MTE1加载 A/B 和scaleA/B的起始地址 相对A和scaleA在L1 上的首地址的offset
        OffsetsForSplitMN result{0, 0};

        if constexpr (isScaleA) {
            if (!transA || dn2nzMTE2TransA_) {
                result.offsetABL1 = mnLoopId * C0_SIZE * baseMN0;
            } else {
                result.offsetABL1 = mnLoopId * baseMN0 * (kL1Iter != kL1Iter_ - 1 ? kL1_ : (k_ - kL1_ * (kL1Iter)));
            }
        } else {
            if (transB || dn2nzMTE2TransB_) {
                result.offsetABL1 = mnLoopId * C0_SIZE * baseMN0;
            } else {
                result.offsetABL1 = mnLoopId * baseMN0 * (kL1Iter != kL1Iter_ - 1 ? kL1_ : (k_ - kL1_ * (kL1Iter)));
            }
        }
        result.offsetScaleABL1 = BLOCK_CUBE * scaleKl1 * (mnLoopId * baseMN0 / BLOCK_CUBE);

        if constexpr (isScaleA && DispatchPolicy::fullLoadMode == Cmct::Gemm::A_FULL_LOAD_MODE) {
            // 全载,MTE2 加载A 是分了l1Iter_次加载， 每次加载kL1_ *
            // curAlignM，起始地址是l1BufferAOffset_[0]加上之前加载的A 加载scaleA 是一次全部加载
            result.offsetABL1 += kL1Iter * kL1_ * curAlignM;
            if constexpr (IS_HIF4) {
                result.offsetScaleABL1 += AscendC::BLOCK_CUBE * kL1Iter * (kL1_ / HIF4_GROUP_SIZE);
            } else {
                result.offsetScaleABL1 += AscendC::BLOCK_CUBE * kL1Iter * (kL1_ / MXFP_GROUP_SIZE);
            }
        }
        return result;
    }

    template <bool isSplitM>
    __aicore__ inline void CopyL0cToGMForSplitMN(uint64_t l0cOffset, uint64_t iterMN, uint64_t curMN, uint64_t baseMN0,
                                                 AscendC::GlobalTensor<CType> cGlobal,
                                                 const TileL1L0Param& tileL1L0Param)
    {
        AscendC::LocalTensor<float> c1Local = c1Local_[l0cOffset];
        uint64_t offsetcGM = 0;
        if (iterMN == SPLIT_NUM_MAX - 1) {
            if constexpr (isSplitM) {
                offsetcGM += baseMN0 * n_;
            } else {
                offsetcGM += baseMN0;
            }
        }

        if constexpr (isSplitM) {
            CopyOut(cGlobal[offsetcGM], c1Local, curMN, tileL1L0Param.curN);
        } else {
            CopyOut(cGlobal[offsetcGM], c1Local, tileL1L0Param.curM, curMN);
        }
    }

    template <bool isSplitM>
    __aicore__ inline void IterateSplitUnified(TileL1L0Param& tileL1L0Param, MmadParams& mmadParams, uint64_t l1Iter,
                                               uint64_t l1BufId, uint64_t scaleL1BufId, uint64_t offsetAl1,
                                               uint64_t l0cOffset, AscendC::GlobalTensor<CType> cGlobal,
                                               uint64_t baseMN0, uint64_t baseMN1)
    {
        auto offsetScaleL1 = ComputeScaleL1Offset(l1Iter);
        uint64_t kL0Iter = Cmct::Gemm::CeilDiv(tileL1L0Param.curGmBKL1, baseK_);

        for (uint16_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
            UpdateKL0(tileL1L0Param, iter1);
            mmadParams.k = Cmct::Gemm::CeilAlign(tileL1L0Param.curKL0, MXFP_DIVISOR_SIZE);
            // copy bias to bt, 全量bias仅搬入一次，两个half按C2偏移取用
            CopyInC2(biasL1Local_[l1BufferBiasOffset_[biasBufId_] / sizeof(BiasType)], biasBt_[baseN_ * biasBufId_],
                     Cmct::Gemm::Align(tileL1L0Param.curN, AscendC::BLOCK_CUBE), NeedBias(l1Iter, iter1));

            if constexpr (isSplitM) {
                uint64_t l0OffsetB = HALF_L0_SIZE * (l0PingPong_ & 0x1);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((l0PingPong_ & 0x1));
                CopyInL0B(l0bLocal_[l0OffsetB], bL1Local_[l1BufferBOffset_[l1BufId]],
                          scaleBL1Local_[l1BufferScaleBOffset_[scaleL1BufId] + offsetScaleL1], iter1, tileL1L0Param,
                          tileL1L0Param.curN);
            } else {
                uint64_t l0OffsetA = HALF_L0_SIZE * (l0PingPong_ & 0x1);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((l0PingPong_ & 0x1));
                if constexpr (DispatchPolicy::fullLoadMode == 0) {
                    CopyInL0A(l0aLocal_[l0OffsetA], aL1Local_[offsetAl1],
                              scaleAL1Local_[l1BufferScaleAOffset_[scaleL1BufId] + offsetScaleL1], iter1, tileL1L0Param,
                              scaleKL1_, tileL1L0Param.curM);
                } else {
                    auto afulloadOffsets = ComputeOffsetsForSplitMN<true>(l1Iter, tileL1L0Param.curAlignM, 0, 0,
                                                                          scaleKL1Num_);
                    CopyInL0A(l0aLocal_[l0OffsetA], aL1Local_[l1BufferAOffset_[0] + afulloadOffsets.offsetABL1],
                              scaleAL1Local_[l1BufferScaleAOffset_[0] + afulloadOffsets.offsetScaleABL1], iter1,
                              tileL1L0Param, k_, tileL1L0Param.curM);
                }
            }

            for (uint16_t innerIter = 0; innerIter < SPLIT_NUM_MAX; innerIter++) {
                uint64_t curMN = baseMN0;
                if (innerIter == SPLIT_NUM_MAX - 1) {
                    curMN = baseMN1;
                }
                uint64_t l0OffsetInner = HALF_L0_SIZE * (l0InnerPingPong_ & 0x1);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((l0InnerPingPong_ & 0x01) + MTE1_WAIT_MMAD_FLAG_2);
                if constexpr (isSplitM) {
                    auto offsetsAForSplitM = ComputeOffsetsForSplitMN<true>(l1Iter, tileL1L0Param.curAlignM, innerIter,
                                                                            baseMN0, scaleKL1Num_);
                    if constexpr (DispatchPolicy::fullLoadMode == 0) {
                        CopyInL0A(l0aLocal_[l0OffsetInner], aL1Local_[offsetAl1 + offsetsAForSplitM.offsetABL1],
                                  scaleAL1Local_[l1BufferScaleAOffset_[scaleL1BufId] + offsetScaleL1 +
                                                 offsetsAForSplitM.offsetScaleABL1],
                                  iter1, tileL1L0Param, scaleKL1_, curMN);
                    } else {
                        CopyInL0A(l0aLocal_[l0OffsetInner],
                                  aL1Local_[l1BufferAOffset_[0] + offsetsAForSplitM.offsetABL1],
                                  scaleAL1Local_[l1BufferScaleAOffset_[0] + offsetsAForSplitM.offsetScaleABL1], iter1,
                                  tileL1L0Param, k_, curMN);
                    }
                    mmadParams.m = curMN;
                    mmadParams.n = tileL1L0Param.curN;
                } else {
                    auto offsetsBForSplitN = ComputeOffsetsForSplitMN<false>(l1Iter, tileL1L0Param.curAlignN, innerIter,
                                                                             baseMN0, scaleKL1Num_);
                    CopyInL0B(l0bLocal_[l0OffsetInner],
                              bL1Local_[l1BufferBOffset_[l1BufId] + offsetsBForSplitN.offsetABL1],
                              scaleBL1Local_[l1BufferScaleBOffset_[scaleL1BufId] + offsetScaleL1 +
                                             offsetsBForSplitN.offsetScaleABL1],
                              iter1, tileL1L0Param, curMN);

                    mmadParams.m = tileL1L0Param.curM;
                    mmadParams.n = curMN;
                }

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0InnerPingPong_ & 0x1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0InnerPingPong_ & 0x1);
                mmadParams.unitFlag = (l1Iter + 1 == kL1Iter_ && iter1 + 1 == kL0Iter) ? FINAL_ACCUMULATION :
                                                                                         NON_FINAL_ACCUMULATION;
                mmadParams.cmatrixInitVal = (l1Iter == 0 && iter1 == 0 && !isBias_);
                uint64_t l0cOffsetupdate = (l0InnerPingPong_ & 0x1) * HALF_L0C_SIZE;
                uint64_t l0OffsetOuter = HALF_L0_SIZE * (l0PingPong_ & 0x1);
                uint64_t l0OffsetInnerVal = HALF_L0_SIZE * (l0InnerPingPong_ & 0x1);

                uint64_t biasOffset = baseN_ * biasBufId_;
                if constexpr (!isSplitM) {
                    // N方向拆分的第二个half，bias从N拆分点开始取
                    biasOffset += (innerIter == SPLIT_NUM_MAX - 1) ? baseMN0 : 0;
                }
                if constexpr (isSplitM) {
                    Mmad(mmadParams, l0cOffsetupdate, l0OffsetInnerVal, l0OffsetOuter, biasOffset,
                         NeedBias(l1Iter, iter1));
                } else {
                    Mmad(mmadParams, l0cOffsetupdate, l0OffsetOuter, l0OffsetInnerVal, biasOffset,
                         NeedBias(l1Iter, iter1));
                }

                if (mmadParams.unitFlag == FINAL_ACCUMULATION) {
                    CopyL0cToGMForSplitMN<isSplitM>(l0cOffsetupdate, innerIter, curMN, baseMN0, cGlobal, tileL1L0Param);
                    // CopyOut 完成后释放该 half 的 L0C，供后续 tile 的 Mmad 复用
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0InnerPingPong_ & 0x1);
                }

                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((l0InnerPingPong_ & 0x1) + MTE1_WAIT_MMAD_FLAG_2);
                l0InnerPingPong_++;
            }
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((l0PingPong_ & 0x1));
            l0PingPong_++;
        }
    }

    template <bool isSplitM>
    __aicore__ inline void IterateInnerLoopKUnified(TileL1L0Param& tileL1L0Param, MmadParams& mmadParams,
                                                    uint64_t l1Iter, uint64_t l1BufId, uint64_t scaleL1BufId,
                                                    uint64_t offsetAl1, uint64_t l0cOffset,
                                                    AscendC::GlobalTensor<CType> cGlobal, uint64_t baseMN0,
                                                    uint64_t baseMN1)
    {
        auto offsetScaleL1 = ComputeScaleL1Offset(l1Iter);
        uint64_t kL0Iter = Cmct::Gemm::CeilDiv(tileL1L0Param.curGmBKL1, baseK_);

        for (uint16_t outerIter = 0; outerIter < SPLIT_NUM_MAX; outerIter++) {
            uint64_t curMN = baseMN0;
            if (outerIter == SPLIT_NUM_MAX - 1) {
                curMN = baseMN1;
            }

            for (uint16_t innerIterK = 0; innerIterK < kL0Iter; ++innerIterK) {
                UpdateKL0(tileL1L0Param, innerIterK);
                mmadParams.k = Cmct::Gemm::CeilAlign(tileL1L0Param.curKL0, MXFP_DIVISOR_SIZE);
                // copy bias to bt, 全量bias仅在首个half搬入一次，两个half按C2偏移取用
                CopyInC2(biasL1Local_[l1BufferBiasOffset_[biasBufId_] / sizeof(BiasType)], biasBt_[baseN_ * biasBufId_],
                         Cmct::Gemm::Align(tileL1L0Param.curN, AscendC::BLOCK_CUBE),
                         NeedBias(l1Iter, innerIterK) && outerIter == 0);
                uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x01);

                OffsetsForSplitMN boffsetsForSplitM{0, 0};
                OffsetsForSplitMN aoffsetsForSplitM{0, 0};
                if constexpr (isSplitM) {
                    aoffsetsForSplitM = ComputeOffsetsForSplitMN<true>(l1Iter, tileL1L0Param.curAlignM, outerIter,
                                                                       baseMN0, scaleKL1Num_);
                    boffsetsForSplitM = ComputeOffsetsForSplitMN<false>(l1Iter, tileL1L0Param.curAlignM, 0, 0,
                                                                        scaleKL1Num_);
                    mmadParams.m = curMN;
                    mmadParams.n = tileL1L0Param.curN;
                } else {
                    aoffsetsForSplitM = ComputeOffsetsForSplitMN<true>(l1Iter, tileL1L0Param.curAlignM, 0, 0,
                                                                       scaleKL1Num_);
                    boffsetsForSplitM = ComputeOffsetsForSplitMN<false>(l1Iter, tileL1L0Param.curAlignM, outerIter,
                                                                        baseMN0, scaleKL1Num_);
                    mmadParams.m = tileL1L0Param.curM;
                    mmadParams.n = curMN;
                }

                if constexpr (DispatchPolicy::fullLoadMode == 0) {
                    CopyInL0A(l0aLocal_[l0Offset], aL1Local_[offsetAl1 + aoffsetsForSplitM.offsetABL1],
                              scaleAL1Local_[l1BufferScaleAOffset_[scaleL1BufId] + offsetScaleL1 +
                                             aoffsetsForSplitM.offsetScaleABL1],
                              innerIterK, tileL1L0Param, scaleKL1_, isSplitM ? curMN : tileL1L0Param.curM);
                } else {
                    CopyInL0A(l0aLocal_[l0Offset], aL1Local_[l1BufferAOffset_[0] + aoffsetsForSplitM.offsetABL1],
                              scaleAL1Local_[l1BufferScaleAOffset_[0] + aoffsetsForSplitM.offsetScaleABL1], innerIterK,
                              tileL1L0Param, k_, isSplitM ? curMN : tileL1L0Param.curM);
                }

                CopyInL0B(l0bLocal_[l0Offset], bL1Local_[l1BufferBOffset_[l1BufId] + boffsetsForSplitM.offsetABL1],
                          scaleBL1Local_[l1BufferScaleBOffset_[scaleL1BufId] + offsetScaleL1 +
                                         boffsetsForSplitM.offsetScaleABL1],
                          innerIterK, tileL1L0Param, isSplitM ? tileL1L0Param.curN : curMN);

                mmadParams.unitFlag = (l1Iter + 1 == kL1Iter_ && innerIterK + 1 == kL0Iter) ? FINAL_ACCUMULATION :
                                                                                              NON_FINAL_ACCUMULATION;
                mmadParams.cmatrixInitVal = (l1Iter == 0 && innerIterK == 0 && !isBias_);
                uint64_t l0cOffsetupdate = (outerIter & 0x1) * HALF_L0C_SIZE;
                uint64_t biasOffset = baseN_ * biasBufId_;
                if constexpr (!isSplitM) {
                    // N方向拆分的第二个half，bias从N拆分点开始取
                    biasOffset += (outerIter == SPLIT_NUM_MAX - 1) ? baseMN0 : 0;
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
                Mmad(mmadParams, l0cOffsetupdate, l0Offset, l0Offset, biasOffset, NeedBias(l1Iter, innerIterK));

                if (mmadParams.unitFlag == FINAL_ACCUMULATION) {
                    CopyL0cToGMForSplitMN<isSplitM>(l0cOffsetupdate, outerIter, curMN, baseMN0, cGlobal, tileL1L0Param);
                    // CopyOut 完成后释放该 half 的 L0C，供后续 tile 的 Mmad 复用
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(outerIter & 0x1);
                }

                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
                l0PingPong_++;
            }
        }
    }

    __aicore__ inline void operator()(AscendC::GlobalTensor<AType> aGlobal, AscendC::GlobalTensor<BType> bGlobal,
                                      AscendC::GlobalTensor<ScaleAType> scaleAGlobal,
                                      AscendC::GlobalTensor<ScaleBType> scaleBGlobal,
                                      AscendC::GlobalTensor<BiasType> biasGlobal, AscendC::GlobalTensor<CType> cGlobal,
                                      BlockShape singleShape)
    {
        TileL1L0Param tileL1L0Param;
        tileL1L0Param.curM = Get<IDX_M_TILEIDX>(singleShape);
        tileL1L0Param.curN = Get<IDX_N_TILEIDX>(singleShape);
        AscendC::MmadParams mmadParams;
        mmadParams.m = tileL1L0Param.curM;
        mmadParams.n = tileL1L0Param.curN;
        mmadParams.disableGemv = true;

        MTE1SplitMode splitMode = MTE1SplitMode::DEFAULT;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9202)
        uint64_t baseMN0 = 0;
        uint64_t baseMN1 = 0;

        // 潜在条件，MTE1_a/b块超0.5L0a/b 需要进一步拆分，已开启db.
        // MTE1 split 依赖 L0C 512KB（DAV_9202）下两个 half 的 L0C 乒乓，只在 DAV_9202 使能；
        // 其它架构（含 DAV_9201/DAV_3510）保持 DEFAULT，避免走到未验证的拆分路径。
        // curM/curN 不足2个BLOCK_CUBE时保护为不拆分（避免baseMN0为0；当前host约束下不可达）
        if (tileL1L0Param.curM >= AscendC::BLOCK_CUBE * SPLIT_NUM_MAX && tileL1L0Param.curM > tileL1L0Param.curN &&
            (((tileL1L0Param.curM * baseK_) >> ELEMENT_SIZE_RATIO) > SPLIT_THRESHOLD_MIN)) {
            splitMode = MTE1SplitMode::SPLIT_M;
            baseMN0 = (tileL1L0Param.curM / AscendC::BLOCK_CUBE / SPLIT_NUM_MAX) * AscendC::BLOCK_CUBE;
            baseMN1 = tileL1L0Param.curM - baseMN0;
        } else if (tileL1L0Param.curN >= AscendC::BLOCK_CUBE * SPLIT_NUM_MAX &&
                   tileL1L0Param.curM <= tileL1L0Param.curN &&
                   (((tileL1L0Param.curN * baseK_) >> ELEMENT_SIZE_RATIO) > SPLIT_THRESHOLD_MIN)) {
            splitMode = MTE1SplitMode::SPLIT_N;
            baseMN0 = (tileL1L0Param.curN / AscendC::BLOCK_CUBE / SPLIT_NUM_MAX) * AscendC::BLOCK_CUBE;
            baseMN1 = tileL1L0Param.curN - baseMN0;
        }
#endif

        // MTE2 dn2nz 转置：K 为外轴且 M/N 为 16 的奇数倍时，MTE2 搬运时直接完成转置，
        // 避免 MTE1 走转置路径导致指令拆分；该能力与平台无关，3510/9201/9202 均可用。
        // 类型约束与 CopyInA1Dn2NZ/CopyInB1Dn2NZ 保持一致：仅对 fp8_e5m2/fp8_e4m3fn 打开，
        // 避免非 fp8 类型（hif8 等）把 flag 置 true 后走到空实现。
        dn2nzMTE2TransA_ = false;
        dn2nzMTE2TransB_ = false;
        if constexpr (!IS_FP4) {
            if constexpr (SupportType<AType, fp8_e5m2_t, fp8_e4m3fn_t>()) {
                dn2nzMTE2TransA_ = transA && ((Cmct::Gemm::CeilDiv(tileL1L0Param.curM, AscendC::BLOCK_CUBE) & 1) != 0);
            }
            if constexpr (SupportType<BType, fp8_e5m2_t, fp8_e4m3fn_t>()) {
                dn2nzMTE2TransB_ = !transB &&
                                   ((Cmct::Gemm::CeilDiv(tileL1L0Param.curN, AscendC::BLOCK_CUBE) & 1) != 0) &&
                                   formatB != CubeFormat::NZ;
            }
        }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9202)
        // split 后单次 MTE1 搬运的 M/N 可能是 16 的奇数倍，需要 MTE2 提前完成转置
        if constexpr (!IS_FP4) {
            if constexpr (SupportType<AType, fp8_e5m2_t, fp8_e4m3fn_t>()) {
                if (splitMode == MTE1SplitMode::SPLIT_M &&
                    (Cmct::Gemm::CeilDiv(baseMN0, AscendC::BLOCK_CUBE) & 1) != 0) {
                    dn2nzMTE2TransA_ = transA;
                }
            }
            if constexpr (SupportType<BType, fp8_e5m2_t, fp8_e4m3fn_t>()) {
                if (splitMode == MTE1SplitMode::SPLIT_N &&
                    (Cmct::Gemm::CeilDiv(baseMN0, AscendC::BLOCK_CUBE) & 1) != 0 && formatB != CubeFormat::NZ) {
                    dn2nzMTE2TransB_ = !transB;
                }
            }
        }
#endif
        GetAlignMN(tileL1L0Param);
        uint64_t l0cOffset = (l0cPingPong_ & 1) * HALF_L0C_SIZE;
        // FIX_M 握手：每个 tile 开始等上一 tile 的 fixpipe CopyOut 释放 L0C 半区，CopyOut 结束后再
        // SetFlag 归还。MTE1 split（仅 DAV_9202）每个 tile 使用 0/HALF_L0C_SIZE 两个半区，
        // DEFAULT 只使用当前 ping-pong 半区。两个半区的初值由构造函数预置。
        if (splitMode == MTE1SplitMode::DEFAULT) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cPingPong_ & 0x1);
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_1);
        }
        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            // Load data to L1 and open DB
            uint64_t l1BufId = abL1LoopCnt_ & (l1BufNum_ - 1);
            uint64_t scaleL1BufId = scaleLoopCnt_ & 1;
            uint64_t offsetA = transA ? iter0 * kL1_ * m_ : iter0 * kL1_;
            uint64_t offsetAl1 = l1BufferAOffset_[l1BufId];
            CopyScalesInL1(scaleAGlobal, scaleBGlobal, tileL1L0Param, iter0, scaleL1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            biasBufId_ = scaleL1BufId;
            UpdateKL1(tileL1L0Param, iter0);
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9202)
            // isSuitForInnerLoopK 需在 UpdateKL1 赋值 curGmBKL1 后评估当前 k 块的 kL0Iter；
            // 该路径会把非拆分操作数的 L0 加载量翻倍，仅在小 k 时更划算
            bool isSuitForInnerLoopK = kL1Iter_ <= SPLIT_NUM_MAX &&
                                       Cmct::Gemm::CeilDiv(tileL1L0Param.curGmBKL1, baseK_) <= SPLIT_NUM_MAX;
#endif
            CopyAInL1(aGlobal, tileL1L0Param, offsetA, offsetAl1, iter0);
            if (isBias_ && iter0 == 0) {
                CopyInBias(biasGlobal, biasL1Local_[l1BufferBiasOffset_[biasBufId_] / sizeof(BiasType)],
                           tileL1L0Param.curN);
            }
            CopyBInL1(bGlobal, tileL1L0Param, l1BufId, iter0);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9202)
            if (splitMode == MTE1SplitMode::SPLIT_M) {
                if (!isSuitForInnerLoopK) {
                    IterateSplitUnified<true>(tileL1L0Param, mmadParams, iter0, l1BufId, scaleL1BufId, offsetAl1,
                                              l0cOffset, cGlobal, baseMN0, baseMN1);
                } else {
                    IterateInnerLoopKUnified<true>(tileL1L0Param, mmadParams, iter0, l1BufId, scaleL1BufId, offsetAl1,
                                                   l0cOffset, cGlobal, baseMN0, baseMN1);
                }
            } else if (splitMode == MTE1SplitMode::SPLIT_N) {
                if (!isSuitForInnerLoopK) {
                    IterateSplitUnified<false>(tileL1L0Param, mmadParams, iter0, l1BufId, scaleL1BufId, offsetAl1,
                                               l0cOffset, cGlobal, baseMN0, baseMN1);
                } else {
                    IterateInnerLoopKUnified<false>(tileL1L0Param, mmadParams, iter0, l1BufId, scaleL1BufId, offsetAl1,
                                                    l0cOffset, cGlobal, baseMN0, baseMN1);
                }
            }
            if (splitMode == MTE1SplitMode::DEFAULT)
#endif
            {
                Iterate(tileL1L0Param, mmadParams, iter0, l1BufId, scaleL1BufId, offsetAl1, l0cOffset);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            if ((iter0 + 1) % (scaleKL1_ / kL1_) == 0 || iter0 == kL1Iter_ - 1) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + (scaleL1BufId));
                scaleLoopCnt_++;
            }
            abL1LoopCnt_++;
        }
        if (splitMode == MTE1SplitMode::DEFAULT) {
            // Copy out to GM
            AscendC::LocalTensor<float> c1Local = c1Local_[l0cOffset];
            // 数据搬出到GM或ub
            CopyOut(cGlobal, c1Local, mmadParams.m, mmadParams.n);
            // CopyOut 完成后释放当前 L0C 半区，供下一个 tile 的 Mmad 复用
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0cPingPong_ & 0x1);
        }
        // split 路径固定使用 0/HALF_L0C_SIZE 两个半区，不参与 L0C 乒乓轮转
        if (enableL0cPingPong_ && splitMode == MTE1SplitMode::DEFAULT) {
            l0cPingPong_++;
        }
    }

private:
    __aicore__ inline bool NeedBias(uint64_t kIter0, uint64_t kIter1) { return isBias_ && kIter0 == 0 && kIter1 == 0; }

    __aicore__ inline void Mmad(AscendC::MmadParams& mmadParams, uint64_t l0cOffset, uint64_t l0aOffset,
                                uint64_t l0bOffset, uint64_t biasOffset, bool needBias)
    {
        mmadParams.cmatrixSource = needBias;
        if (needBias) {
            AscendC::Mmad(c1Local_[l0cOffset], l0aLocal_[l0aOffset], l0bLocal_[l0bOffset], biasBt_[biasOffset],
                          mmadParams);
        } else {
            mmadParams.cmatrixSource = false;
            AscendC::Mmad(c1Local_[l0cOffset], l0aLocal_[l0aOffset], l0bLocal_[l0bOffset], mmadParams);
        }
    }

private:
    constexpr static uint16_t SCALE_BUFFER_FLAG_0 = 4;
    constexpr static uint16_t SCALE_BUFFER_FLAG_1 = 5;
    constexpr static uint16_t SPLIT_NUM_MAX = 2;
    // MTE1 单次搬运的 M/N 侧数据量（curM*baseK / curN*baseK 经 >> ELEMENT_SIZE_RATIO 折算为字节数）
    // 超过半区 L0 容量（L0A_SIZE/2，B8 下即 HALF_L0_SIZE = 32KB）时，需在 M/N 方向继续拆分
    constexpr static uint64_t SPLIT_THRESHOLD_MIN = 32 * 1024UL;
    bool dn2nzMTE2TransA_ = false;
    bool dn2nzMTE2TransB_ = false;
    uint16_t biasBufId_ = 0;
    uint64_t biasL1OneBuffer_ = 0UL;
    uint64_t aL1OneBuffer_ = 0UL;
    uint64_t bL1OneBuffer_ = 0UL;
    uint64_t scaleAL1OneBuffer_ = 0UL;
    uint64_t scaleBL1OneBuffer_ = 0UL;
    uint64_t l1BufferAOffset_[4] = {0UL};      // default 4 buffer
    uint64_t l1BufferBOffset_[4] = {0UL};      // default 4 buffer
    uint64_t l1BufferScaleAOffset_[2] = {0UL}; // default 2 buffer
    uint64_t l1BufferScaleBOffset_[2] = {0UL}; // default 2 buffer
    uint64_t l1BufferBiasOffset_[2] = {0UL};   // default 2 buffer
    AscendC::LocalTensor<MxL0AType> l0aLocal_{AscendC::TPosition::A2, 0, L0A_SIZE};
    AscendC::LocalTensor<MxL0BType> l0bLocal_{AscendC::TPosition::B2, 0, L0B_SIZE};
    AscendC::LocalTensor<float> c1Local_{AscendC::TPosition::CO1, 0, AscendC::TOTAL_L0C_SIZE};
    AscendC::LocalTensor<float> biasBt_{AscendC::TPosition::C2, 0, BT_SIZE};
    AscendC::LocalTensor<AType> aL1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE};
    AscendC::LocalTensor<BType> bL1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE};
    AscendC::LocalTensor<BiasType> biasL1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE / sizeof(BiasType)};
    AscendC::LocalTensor<ScaleAType> scaleAL1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE};
    AscendC::LocalTensor<ScaleBType> scaleBL1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE};
};
} // namespace Block
} // namespace Gemm
} // namespace Cmct
