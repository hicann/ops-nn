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
 * \file block_mmad_mx_quant.h
 * \brief
 */
#ifndef BLOCK_MMAD_QBMM_MX_STREAMK_H
#define BLOCK_MMAD_QBMM_MX_STREAMK_H
#include "../policy/dispatch_policy.h"
#include "../utils/common_utils.h"
#include "../utils/quant_batch_matmul_constant.h"
#include "../utils/layout_utils.h"
#include "../utils/tuple_utils.h"

namespace {
constexpr static uint64_t BLOCK_REDUCE_CUBE = 32UL;
constexpr static uint64_t MXFP_DIVISOR_SIZE = 64UL;
constexpr static uint64_t HIF4_GROUP_SIZE = 64UL;
constexpr static uint64_t MXFP_MULTI_BASE_SIZE = 2UL;
constexpr static uint64_t EVEN_FACTOR = 2UL;
constexpr static uint64_t HIF4_SCALE_NUM_HALF_RATIO = 1UL; // 位移参数，左移1表示乘2, hif4scale一个元素为4B，等于2个half
constexpr static uint16_t SCALE_BUFFER_FLAG_0 = 4;
constexpr static uint16_t SCALE_BUFFER_FLAG_1 = 5;
} // namespace

// 4bit dtype judgement, kept local because hifloat4x2_t only exists on newer toolkits.
template <typename AType>
constexpr bool IS_FP4 = AscendC::IsSameType<AType, fp4x2_e2m1_t>::value ||
                        AscendC::IsSameType<AType, hifloat4x2_t>::value;

namespace Cmct {
namespace Gemm {
namespace Block {
using namespace AscendC;
using namespace QuantBatchMatmul;

template <class DispatchPolicy_, class L1TileShape_, class L0TileShape_, class AType_, class LayoutA_, class BType_,
          class LayoutB_, class CType_, class LayoutC_, class WorkspaceDtype_, class BiasType_, class LayoutBias_,
          class TileCopy_, class Enable = void>
class BlockQbmmMXStreamk {};

template <class DispatchPolicy_, class L1TileShape_, class L0TileShape_, class AType_, class LayoutA_, class BType_,
          class LayoutB_, class CType_, class LayoutC_, class WorkspaceDtype_, class BiasType_, class LayoutBias_,
          class TileCopy_>
class BlockQbmmMXStreamk<DispatchPolicy_, L1TileShape_, L0TileShape_, AType_, LayoutA_, BType_, LayoutB_, CType_,
                         LayoutC_, WorkspaceDtype_, BiasType_, LayoutBias_, TileCopy_,
                         AscendC::Std::enable_if_t<
                             AscendC::Std::is_base_of_v<MatmulWithScale<>, DispatchPolicy_> ||
                             AscendC::Std::is_base_of_v<
                                 MatmulWithScale<AscendC::Shape<_0, _0, _0, _0>, A_FULL_LOAD_MODE>, DispatchPolicy_>>> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using WorkspaceDtype = WorkspaceDtype_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using MxL0AType = typename GetL0DataType<AType, true>::Type; // 在l0a 上的数据类型，实际上没变 hifp4 对应的还是hifp4
    using MxL0BType = typename GetL0DataType<BType, true>::Type;
    using ScaleAType = typename AscendC::Conditional<IsSameType<AType, hifloat4x2_t>::value, hif4_scale,
                                                     AscendC::fp8_e8m0_t>::type;
    using ScaleBType = typename AscendC::Conditional<IsSameType<AType, hifloat4x2_t>::value, hif4_scale,
                                                     AscendC::fp8_e8m0_t>::type;
    using ScaleType = ScaleAType;

    using BiasType = BiasType_;
    using DispatchPolicy = DispatchPolicy_;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

    static constexpr bool transA = TagToTrans<LayoutA>::value;
    static constexpr bool transB = TagToTrans<LayoutB>::value;
    static constexpr CubeFormat formatB = TagToFormat<LayoutB>::format;
    static constexpr uint64_t MIN_STEP = IS_FP4<AType> ? B4_MIN_STEP : B8_MIN_STEP;

    struct L1Params {
        uint64_t kL1;
        uint64_t scaleKL1;
        uint64_t l1BufNum;
        uint32_t x1QuantMode;
        uint32_t x2QuantMode;
    };

    __aicore__ inline BlockQbmmMXStreamk()
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
    }

    __aicore__ inline ~BlockQbmmMXStreamk()
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
    }

public:
    __aicore__ inline void Init(const TupleShape& problemShape, const BlockShape& l0TileShape, const L1Params& l1Params,
                                bool isBias, bool dbL0C)
    {
        m_ = Get<IDX_M_IDX>(problemShape);
        n_ = Get<IDX_N_IDX>(problemShape);
        k_ = Get<IDX_K_IDX>(problemShape);
        kL1_ = l1Params.kL1; // baseK*stepk
        scaleKL1DP_ = l1Params.scaleKL1;
        scaleKL1_ = scaleKL1DP_;
        scaleKL1Factor_ = scaleKL1_ / kL1_;
        baseM_ = Get<IDX_M_IDX>(l0TileShape);
        baseN_ = Get<IDX_N_IDX>(l0TileShape);
        baseK_ = Get<IDX_K_IDX>(l0TileShape);

        isBias_ = isBias;
        l1BufNum_ = l1Params.l1BufNum;
        enableL0cPingPong_ = dbL0C;

        if constexpr (!SupportType<AType, fp8_e5m2_t, fp8_e4m3fn_t>()) {
            elementSizeRatio_ = 1; // 右移参数1表示除2
        }
        aL1OneBuffer_ = (baseM_ * Cmct::Gemm::Align(kL1_, MXFP_DIVISOR_SIZE)) >> elementSizeRatio_; // 单位B
        bL1OneBuffer_ = (baseN_ * Cmct::Gemm::Align(kL1_, MXFP_DIVISOR_SIZE)) >> elementSizeRatio_;
        halfL0AEleNum_ = HALF_L0A_SIZE << elementSizeRatio_; // halfL0AEleNum_单位从B 转为可存储的元素数

        if constexpr (SupportType<AType, hifloat4x2_t>()) {
            scaleAL1OneBuffer_ = baseM_ * Cmct::Gemm::CeilDiv(scaleKL1_, HIF4_GROUP_SIZE) * sizeof(ScaleType);
            scaleBL1OneBuffer_ = baseN_ * Cmct::Gemm::CeilDiv(scaleKL1_, HIF4_GROUP_SIZE) * sizeof(ScaleType);
        } else {
            scaleBL1OneBuffer_ = baseN_ * Cmct::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE *
                                 sizeof(ScaleType); // 单位B
            scaleAL1OneBuffer_ = baseM_ * Cmct::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE *
                                 sizeof(ScaleType);
        }

        for (int32_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
            // 2 buffer: L1 space is : A0|B0|AScale0|BScale0|bias0|...|A1|B1|AScale1|BScale1|bias1|...
            // 4 buffer: L1 space is : A0A2|B0B2|AScale0|BScale0|bias0|...|A1A3|B1B3|AScale1|BScale1|bias1|...
            uint64_t l1Offset = (AscendC::TOTAL_L1_SIZE >> 1) * (bufferId & 1); // L1 上存储A B 的buffter 的起始地址
            l1BufferAOffset_[bufferId] = (l1Offset + aL1OneBuffer_ * (bufferId >> 1))
                                         << elementSizeRatio_; // 单位elements
            l1BufferBOffset_[bufferId] = (l1Offset + aL1OneBuffer_ * (l1BufNum_ >> 1) + bL1OneBuffer_ * (bufferId >> 1))
                                         << elementSizeRatio_; // 单位elements
        }

        for (int32_t bufferId = 0; bufferId < SCALE_BUFFER_NUM; bufferId++) {
            // LocalTensor的offset 单位是element
            l1BufferScaleAOffset_[bufferId] = ((l1BufferBOffset_[bufferId] >> elementSizeRatio_) +
                                               bL1OneBuffer_ * (l1BufNum_ >> 1)) /
                                              sizeof(ScaleAType);

            l1BufferScaleBOffset_[bufferId] = (l1BufferScaleAOffset_[bufferId] * sizeof(ScaleAType) +
                                               scaleAL1OneBuffer_) /
                                              sizeof(ScaleBType);

            l1BufferBiasOffset_[bufferId] = l1BufferScaleBOffset_[bufferId] * sizeof(ScaleBType) + scaleBL1OneBuffer_;
        }
    }

    __aicore__ inline void operator()(AscendC::GlobalTensor<AType> aGlobal, AscendC::GlobalTensor<BType> bGlobal,
                                      AscendC::GlobalTensor<ScaleAType> scaleAGlobal,
                                      AscendC::GlobalTensor<ScaleBType> scaleBGlobal,
                                      AscendC::GlobalTensor<BiasType> biasGlobal, AscendC::GlobalTensor<CType> cGlobal,
                                      AscendC::GlobalTensor<WorkspaceDtype> workspaceGlobal, BlockShape singleShape,
                                      bool isBias, bool isSkScene)
    {
        TileL1L0Param tileL1L0Param;
        tileL1L0Param.curM = Get<IDX_M_TILE_IDX>(singleShape);
        tileL1L0Param.curN = Get<IDX_N_TILE_IDX>(singleShape);
        singleCoreK_ = Get<IDX_K_IDX>(singleShape); // 可能是K也可能是K轴切分后的数值
        isBias_ = isBias;

        dn2nzMTE2TransA_ = transA && (Cmct::Gemm::CeilDiv(tileL1L0Param.curM, AscendC::BLOCK_CUBE) & 1) != 0 &&
                           !IS_FP4<AType>;
        dn2nzMTE2TransB_ = !transB && (Cmct::Gemm::CeilDiv(tileL1L0Param.curN, AscendC::BLOCK_CUBE) & 1) != 0 &&
                           !IS_FP4<AType> && formatB != CubeFormat::NZ;

        if (isSkScene) {
            kL1_ = min(singleCoreK_, kL1_);
            scaleKL1_ = min(singleCoreK_, scaleKL1_);
            kL1Iter_ = (singleCoreK_ + kL1_ - 1) / kL1_;
        } else {
            // DP+SK 的情况下，SK会提前计算，后边轮次计算DP 时需要将scaleKL1_恢复成tiling中的输入
            scaleKL1_ = scaleKL1DP_;
            kL1Iter_ = (k_ + kL1_ - 1) / kL1_;
        }
        scaleKL1Factor_ = scaleKL1_ / kL1_;

        GetAlignMN(tileL1L0Param);
        AscendC::MmadParams mmadParams;
        mmadParams.m = tileL1L0Param.curM;
        mmadParams.n = tileL1L0Param.curN;
        mmadParams.disableGemv = true;

        uint64_t l0cOffset = (l0cPingPong_ & 1) * HALF_L0C_SIZE;
        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            // Load data to L1 and open DB
            uint64_t l1BufId = abL1LoopCnt_ & (l1BufNum_ - 1);
            uint64_t scaleL1BufId = scaleLoopCnt_ & 1;
            uint64_t offsetA = transA ? iter0 * kL1_ * m_ : iter0 * kL1_;
            uint64_t offsetAl1 = l1BufferAOffset_[l1BufId];

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            biasBufId_ = abL1LoopCnt_ & 1;
            UpdateKL1(tileL1L0Param, iter0);
            CopyAInL1(aGlobal, tileL1L0Param, offsetA, offsetAl1, iter0);

            if (isBias_ && iter0 == 0) {
                CopyInBias(biasGlobal, biasL1Local_[l1BufferBiasOffset_[biasBufId_] / sizeof(BiasType)],
                           tileL1L0Param.curN);
            }
            CopyBInL1(bGlobal, tileL1L0Param, l1BufId, iter0);
            CopyScalesInL1(scaleAGlobal, scaleBGlobal, tileL1L0Param, iter0, scaleL1BufId);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId); // 1

            Iterate(tileL1L0Param, mmadParams, iter0, l1BufId, scaleL1BufId, offsetAl1, l0cOffset);

            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            if ((iter0 + 1) % scaleKL1Factor_ == 0 || iter0 == kL1Iter_ - 1) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + (scaleL1BufId));
                scaleLoopCnt_++;
            }
            abL1LoopCnt_++;
        }

        // Copy out to GM
        AscendC::LocalTensor<float> c1Local = c1Local_[l0cOffset];
        CopyOut(cGlobal, workspaceGlobal, c1Local, mmadParams.m, mmadParams.n, isSkScene);
        if (enableL0cPingPong_) {
            l0cPingPong_++;
        }
    }

    __aicore__ inline void Iterate(TileL1L0Param& tileL1L0Param, MmadParams& mmadParams, uint64_t l1Iter,
                                   uint64_t l1BufId, uint64_t scaleL1BufId, uint64_t offsetAl1, uint64_t l0cOffset)
    {
        uint64_t kL0Iter = Cmct::Gemm::CeilDiv(tileL1L0Param.curGmBKL1, baseK_);
        for (uint16_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
            UpdateKL0(tileL1L0Param, iter1);
            // Load data to L0 and open DB
            uint64_t l0Offset = halfL0AEleNum_ * (l0PingPong_ & 0x1) / sizeof(MxL0AType);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);

            uint64_t offsetScaleL1 = 0;
            if constexpr (SupportType<AType, hifloat4x2_t>()) {
                offsetScaleL1 = AscendC::BLOCK_CUBE * (l1Iter % scaleKL1Factor_) * (kL1_ / HIF4_GROUP_SIZE);
            } else {
                offsetScaleL1 = AscendC::BLOCK_CUBE * (l1Iter % scaleKL1Factor_) * (kL1_ / MXFP_GROUP_SIZE);
            }

            if (!transA || dn2nzMTE2TransA_) {
                CopyInL0ANonTransA(l0aLocal_[l0Offset], aL1Local_[offsetAl1],
                                   scaleAL1Local_[l1BufferScaleAOffset_[scaleL1BufId] + offsetScaleL1], iter1,
                                   tileL1L0Param);
            } else {
                CopyInL0ATransA(l0aLocal_[l0Offset], aL1Local_[offsetAl1],
                                scaleAL1Local_[l1BufferScaleAOffset_[scaleL1BufId] + offsetScaleL1], iter1,
                                tileL1L0Param);
            }

            // copy bias to bt
            CopyInC2(biasL1Local_[l1BufferBiasOffset_[biasBufId_] / sizeof(BiasType)], biasBt_[baseN_ * biasBufId_],
                     Cmct::Gemm::Align(mmadParams.n, AscendC::BLOCK_CUBE), NeedBias(l1Iter, iter1));

            if (transB || dn2nzMTE2TransB_) {
                CopyInL0BTransB(l0bLocal_[l0Offset], bL1Local_[l1BufferBOffset_[l1BufId]],
                                scaleBL1Local_[l1BufferScaleBOffset_[scaleL1BufId] + offsetScaleL1], iter1,
                                tileL1L0Param);
            } else {
                CopyInL0BNonTransB(l0bLocal_[l0Offset], bL1Local_[l1BufferBOffset_[l1BufId]],
                                   scaleBL1Local_[l1BufferScaleBOffset_[scaleL1BufId] + offsetScaleL1], iter1,
                                   tileL1L0Param);
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
            mmadParams.k = Cmct::Gemm::CeilAlign(tileL1L0Param.curKL0, MXFP_DIVISOR_SIZE);
            mmadParams.unitFlag = (l1Iter + 1 == kL1Iter_ && iter1 + 1 == kL0Iter) ? FINAL_ACCUMULATION :
                                                                                     NON_FINAL_ACCUMULATION;
            mmadParams.cmatrixInitVal = (l1Iter == 0 && iter1 == 0 && !isBias_);

            Mmad(mmadParams, l0cOffset, l0Offset, baseN_ * biasBufId_, NeedBias(l1Iter, iter1));
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
            l0PingPong_++;
        }
    }

    __aicore__ inline void CopyInA1(const AscendC::GlobalTensor<AType>& aGlobal,
                                    const AscendC::LocalTensor<AType>& al1Local, TileL1L0Param& tileL1L0Param)
    {
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        uint64_t nDim = transA ? tileL1L0Param.curGmAKL1 : tileL1L0Param.curM;
        uint64_t dDim = transA ? tileL1L0Param.curM : tileL1L0Param.curGmAKL1;

        nd2nzParams.nValue = nDim;
        nd2nzParams.dValue = dDim >> elementSizeRatio_; // 4bit数据，搬运到L1时，2个4bit合成一个8bit 搬
        nd2nzParams.srcNdMatrixStride = 1;
        nd2nzParams.srcDValue = (transA ? m_ : k_) >> elementSizeRatio_;
        nd2nzParams.dstNzC0Stride = transA ? tileL1L0Param.curPadAKL1 : tileL1L0Param.curAlignM;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 1;

        AscendC::DataCopy(al1Local, aGlobal, nd2nzParams, 1); // mte2 prealloc 未开启
    }

    __aicore__ inline void CopyInA1Dn2NZ(const AscendC::GlobalTensor<AType>& aGlobal,
                                         const AscendC::LocalTensor<AType>& al1Local, TileL1L0Param& tileL1L0Param)
    {
        // 基础api dn2nz 对hifloat4 数据还不支持, 不加过滤会有编译报错,
        // 当前hif4还不支持K为外轴，因此也不会走进此函数，不会造成逻辑问题，后继支持了多种trans, 这里需要同步适配
        if constexpr (!SupportType<BType, fp8_e5m2_t, fp8_e4m3fn_t>()) {
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
        AscendC::DataCopy(al1Local, aGlobal, dn2nzParams);
    }

    __aicore__ inline void CopyInB1(const AscendC::GlobalTensor<BType>& bGlobal,
                                    const AscendC::LocalTensor<BType>& bl1Local, TileL1L0Param& tileL1L0Param)
    {
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        uint64_t nDim = transB ? tileL1L0Param.curN : tileL1L0Param.curGmBKL1;
        uint64_t dDim = transB ? tileL1L0Param.curGmBKL1 : tileL1L0Param.curN;

        nd2nzParams.nValue = nDim;
        nd2nzParams.dValue = dDim >> elementSizeRatio_; // 单位B
        nd2nzParams.srcNdMatrixStride = 1;
        nd2nzParams.srcDValue = (transB ? k_ : n_) >> elementSizeRatio_; // 单位B
        nd2nzParams.dstNzC0Stride = transB ? tileL1L0Param.curAlignN : tileL1L0Param.curPadBKL1;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 1;
        AscendC::DataCopy(bl1Local, bGlobal, nd2nzParams, 1);
    }

    __aicore__ inline void CopyInB1Dn2NZ(const AscendC::GlobalTensor<BType>& bGlobal,
                                         const AscendC::LocalTensor<BType>& bl1Local, TileL1L0Param& tileL1L0Param)
    {
        // 基础api dn2nz 对hifloat4 数据还不支持, 不加过滤会有编译报错
        // 当前hif4还不支持K为外轴，因此也不会走进此函数，不会造成逻辑问题，后继支持了多种trans, 这里需要同步适配
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
        AscendC::DataCopy(bl1Local, bGlobal, dn2nzParams);
    }

    __aicore__ inline void CopyInB1WeightNz(const AscendC::GlobalTensor<BType>& bGlobal,
                                            const AscendC::LocalTensor<BType>& bl1Local,
                                            const TileL1L0Param& tileL1L0Param)
    {
        // WeightNz: GM 上 B 已经是 NZ 排布，用 DataCopyPad 按 NZ 分块搬运，无需 ND->NZ 转换
        AscendC::GlobalTensor<uint8_t> bGlobalUInt8;
        bGlobalUInt8.SetGlobalBuffer((__gm__ uint8_t*)bGlobal.GetPhyAddr());
        auto bl1LocalUint8 = bl1Local.template ReinterpretCast<uint8_t>();

        AscendC::DataCopyExtParams dataCopyParams;
        AscendC::DataCopyPadExtParams<uint8_t> padParams;
        if constexpr (transB) {
            // layout::Zn: GM 上 N 为外轴，K 为内轴，每个 NZ 块为 [C0_NUM, C0_SIZE]
            dataCopyParams.blockCount = Cmct::Gemm::CeilDiv(tileL1L0Param.curGmBKL1, C0_SIZE);
            dataCopyParams.blockLen = tileL1L0Param.curAlignN * C0_SIZE;
            if constexpr (IS_FP4<AType>) {
                dataCopyParams.blockLen = dataCopyParams.blockLen >> elementSizeRatio_;
            }
            dataCopyParams.srcStride = (Cmct::Gemm::CeilAlign(n_, AscendC::BLOCK_CUBE) - tileL1L0Param.curAlignN) *
                                       C0_SIZE;
            if constexpr (IS_FP4<AType>) {
                dataCopyParams.srcStride = dataCopyParams.srcStride >> elementSizeRatio_;
            }
            dataCopyParams.dstStride = 0;
        } else {
            // layout::Nz: GM 上 K 为外轴，N 为内轴，每个 NZ 块为 [C0_SIZE, C0_NUM]
            int64_t curGmBKL1NZ = Cmct::Gemm::CeilAlign(tileL1L0Param.curGmBKL1, AscendC::BLOCK_CUBE);
            dataCopyParams.blockCount = tileL1L0Param.curAlignN / C0_SIZE;
            dataCopyParams.blockLen = curGmBKL1NZ * C0_SIZE;
            if constexpr (IS_FP4<AType>) {
                dataCopyParams.blockLen = dataCopyParams.blockLen >> elementSizeRatio_;
            }
            dataCopyParams.srcStride = (Cmct::Gemm::CeilAlign(k_, AscendC::BLOCK_CUBE) - curGmBKL1NZ) * C0_SIZE;
            if constexpr (IS_FP4<AType>) {
                dataCopyParams.srcStride = dataCopyParams.srcStride >> elementSizeRatio_;
            }
            dataCopyParams.dstStride = tileL1L0Param.curPadBKL1 - curGmBKL1NZ;
        }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
        AscendC::DataCopyPad(bl1LocalUint8, bGlobalUInt8, dataCopyParams, padParams, true);
#else
        AscendC::DataCopyPad(bl1LocalUint8, bGlobalUInt8, dataCopyParams, padParams);
#endif
    }

    __aicore__ inline void InitA1(const AscendC::LocalTensor<AType>& al1Local, const TileL1L0Param& tileL1L0Param)
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
                if (tileL1L0Param.curPadBKL1 == tileL1L0Param.curGmBKL1) {
                    return;
                }
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
        AscendC::DataCopyPadParams padParams;
        AscendC::DataCopyParams biasParam{1, static_cast<uint16_t>(curNL1 * sizeof(BiasType)), 0, 0};
        AscendC::DataCopyPad(cl1Local, biasGlobal, biasParam, padParams, 1);
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

    __aicore__ inline void CopyInL0ATransA(const AscendC::LocalTensor<MxL0AType>& l0aLocal,
                                           const AscendC::LocalTensor<AType>& al1Local,
                                           const AscendC::LocalTensor<ScaleAType>& scaleAl1Local, uint64_t iter,
                                           TileL1L0Param& tileL1L0Param)
    {
        uint64_t m1 = Cmct::Gemm::CeilDiv(tileL1L0Param.curM, AscendC::BLOCK_CUBE);
        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = Cmct::Gemm::CeilDiv(iter * baseK_, AscendC::BLOCK_CUBE);
        loadDataParams.kStartPosition = 0;
        if (((m1 & 1) == 0 && !IS_FP4<AType>) || (m1 % B4_MIN_STEP == 0 && IS_FP4<AType>)) {
            loadDataParams.mStep = Cmct::Gemm::CeilAlign(Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, AscendC::BLOCK_CUBE),
                                                         MXFP_MULTI_BASE_SIZE);
        } else {
            loadDataParams.mStep = MIN_STEP;
        }
        loadDataParams.kStep = Cmct::Gemm::CeilDiv(tileL1L0Param.curM, C0_SIZE);
        loadDataParams.srcStride = Cmct::Gemm::CeilDiv(tileL1L0Param.curPadAKL1, AscendC::BLOCK_CUBE);
        loadDataParams.dstStride = m1;
        loadDataParams.ifTranspose = true;

        AscendC::LoadData2DMxParams loadData2DMxParams;
        loadData2DMxParams.xStartPosition = 0;
        loadData2DMxParams.xStep = m1;
        if constexpr (AscendC::SupportType<AType, hifloat4x2_t>()) {
            // ystartPosition, ystep 的单位为32B
            // hifp4 scale的相邻元素k方向的offset为16*4B = 64B(16是m方向分型大小), ystep 的单位为32B,
            // 所以scale的 y方向offset 数值是 元素个数的2倍
            loadData2DMxParams.yStartPosition = Cmct::Gemm::CeilDiv(iter * baseK_, HIF4_GROUP_SIZE)
                                                << HIF4_SCALE_NUM_HALF_RATIO;
            loadData2DMxParams.yStep = Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, MXFP_DIVISOR_SIZE)
                                       << HIF4_SCALE_NUM_HALF_RATIO;
            loadData2DMxParams.srcStride = Cmct::Gemm::CeilDiv(scaleKL1_, HIF4_GROUP_SIZE) << HIF4_SCALE_NUM_HALF_RATIO;
            loadData2DMxParams.dstStride = loadData2DMxParams.yStep;
        } else {
            loadData2DMxParams.yStartPosition = Cmct::Gemm::CeilDiv(iter * baseK_, MXFP_DIVISOR_SIZE);
            loadData2DMxParams.yStep = Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, MXFP_DIVISOR_SIZE);
            loadData2DMxParams.srcStride = Cmct::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE);
            loadData2DMxParams.dstStride = loadData2DMxParams.yStep;
        }
        AscendC::LoadData(l0aLocal, al1Local, scaleAl1Local, loadDataParams, loadData2DMxParams);
        if constexpr (transA) {
            if (((m1 & 1) != 0 && !IS_FP4<AType>) || (m1 % B4_MIN_STEP != 0 && IS_FP4<AType>)) {
                PipeBarrier<PIPE_MTE1>();
                LocalTensor<AType> l0a = l0aLocal.template ReinterpretCast<AType>();
                uint64_t loadTimes = Cmct::Gemm::CeilDiv(Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, AscendC::BLOCK_CUBE),
                                                         MIN_STEP);
                for (uint64_t i = 1; i < loadTimes; i++) {
                    loadDataParams.mStartPosition = MIN_STEP * i +
                                                    Cmct::Gemm::CeilDiv(iter * baseK_, AscendC::BLOCK_CUBE);
                    AscendC::LoadData(l0a[i * m1 * AscendC::BLOCK_CUBE * C0_SIZE], al1Local, loadDataParams);
                    PipeBarrier<PIPE_MTE1>();
                }
            }
        }
    }

    __aicore__ inline void CopyInL0ANonTransA(const AscendC::LocalTensor<MxL0AType>& l0aLocal,
                                              const AscendC::LocalTensor<AType>& al1Local,
                                              const AscendC::LocalTensor<ScaleAType>& scaleAl1Local, uint64_t iter,
                                              TileL1L0Param& tileL1L0Param)
    {
        // transA=false 时，mte1 搬A和 scaleA
        uint64_t m1 = Cmct::Gemm::CeilDiv(tileL1L0Param.curM, AscendC::BLOCK_CUBE);
        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = Cmct::Gemm::CeilDiv(iter * baseK_, C0_SIZE);
        loadDataParams.mStep = m1;
        loadDataParams.kStep = Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, C0_SIZE);
        loadDataParams.srcStride = loadDataParams.mStep;
        loadDataParams.dstStride = loadDataParams.mStep;
        loadDataParams.ifTranspose = false;

        AscendC::LoadData2DMxParams loadData2DMxParams;
        loadData2DMxParams.xStartPosition = 0;
        loadData2DMxParams.xStep = m1;

        if constexpr (SupportType<AType, fp8_e5m2_t, fp8_e4m3fn_t>()) {
            loadData2DMxParams.yStartPosition = Cmct::Gemm::CeilDiv(iter * baseK_, MXFP_DIVISOR_SIZE);
            loadData2DMxParams.yStep = Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, MXFP_DIVISOR_SIZE);
            loadData2DMxParams.srcStride = Cmct::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE);
            loadData2DMxParams.dstStride = loadData2DMxParams.yStep;
        } else if constexpr (SupportType<AType, fp4x2_e2m1_t, fp4x2_e1m2_t>()) {
            loadData2DMxParams.yStartPosition = Cmct::Gemm::CeilDiv(iter * baseK_, MXFP_DIVISOR_SIZE);
            loadData2DMxParams.yStep = Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, C0_SIZE);
            loadData2DMxParams.srcStride = Cmct::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE);
            loadData2DMxParams.dstStride = loadData2DMxParams.yStep;
        } else if constexpr (SupportType<AType, hifloat4x2_t>()) {
            // ystartPosition, ystep 的单位为32B
            // hifp4 scale的相邻元素k方向的offset为16*4B = 64B(16是m方向分型大小), ystep 的单位为32B,
            // 所以scale的 y方向offset 数值是 元素个数的2倍
            loadData2DMxParams.yStartPosition = Cmct::Gemm::CeilDiv(iter * baseK_, HIF4_GROUP_SIZE)
                                                << HIF4_SCALE_NUM_HALF_RATIO;
            loadData2DMxParams.yStep = loadDataParams.kStep << HIF4_SCALE_NUM_HALF_RATIO;
            loadData2DMxParams.srcStride = Cmct::Gemm::CeilDiv(scaleKL1_, HIF4_GROUP_SIZE) << HIF4_SCALE_NUM_HALF_RATIO;
            loadData2DMxParams.dstStride = loadData2DMxParams.yStep;
        }

        AscendC::LoadData(l0aLocal, al1Local, scaleAl1Local, loadDataParams, loadData2DMxParams);
    }

    __aicore__ inline void CopyInL0BNonTransB(const AscendC::LocalTensor<MxL0BType>& l0bLocal,
                                              const AscendC::LocalTensor<BType>& bl1Local,
                                              const AscendC::LocalTensor<ScaleBType>& scaleBl1Local, uint64_t iter,
                                              TileL1L0Param& tileL1L0Param)
    {
        AscendC::LoadData2DParamsV2 loadDataParams;
        AscendC::LoadData2DMxParams loadData2DMxParams;
        uint64_t n1 = Cmct::Gemm::CeilDiv(tileL1L0Param.curN, AscendC::BLOCK_CUBE);

        loadDataParams.mStartPosition = Cmct::Gemm::CeilDiv(iter * baseK_, AscendC::BLOCK_CUBE);
        loadDataParams.kStartPosition = 0;
        if (((n1 & 1) == 0 && !IS_FP4<AType>) || (n1 % B4_MIN_STEP == 0 && IS_FP4<AType>)) {
            loadDataParams.mStep = Cmct::Gemm::CeilAlign(Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, AscendC::BLOCK_CUBE),
                                                         EVEN_FACTOR);
        } else {
            loadDataParams.mStep = MIN_STEP;
        }
        loadDataParams.kStep = Cmct::Gemm::CeilDiv(tileL1L0Param.curN, C0_SIZE);
        loadDataParams.srcStride = Cmct::Gemm::CeilDiv(tileL1L0Param.curPadBKL1, AscendC::BLOCK_CUBE);
        loadDataParams.dstStride = n1;
        loadDataParams.ifTranspose = true;

        loadData2DMxParams.xStartPosition = 0;
        loadData2DMxParams.xStep = n1;
        if (SupportType<BType, hifloat4x2_t>()) {
            // ystartPosition, ystep 的单位为32B
            // hifp4 scale的相邻元素k方向的offset为16*4B = 64B(16是m方向分型大小), ystep 的单位为32B,
            // 所以scale的 y方向offset 数值是 元素个数的2倍
            loadData2DMxParams.yStartPosition = Cmct::Gemm::CeilDiv(iter * baseK_, HIF4_GROUP_SIZE)
                                                << HIF4_SCALE_NUM_HALF_RATIO;
            loadData2DMxParams.yStep = Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, MXFP_DIVISOR_SIZE)
                                       << HIF4_SCALE_NUM_HALF_RATIO;
            loadData2DMxParams.srcStride = Cmct::Gemm::CeilDiv(scaleKL1_, HIF4_GROUP_SIZE) << HIF4_SCALE_NUM_HALF_RATIO;
            loadData2DMxParams.dstStride = loadData2DMxParams.yStep;
        } else {
            loadData2DMxParams.yStartPosition = Cmct::Gemm::CeilDiv(iter * baseK_, MXFP_DIVISOR_SIZE);
            loadData2DMxParams.yStep = Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, MXFP_DIVISOR_SIZE);
            loadData2DMxParams.srcStride = Cmct::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE);
            loadData2DMxParams.dstStride = loadData2DMxParams.yStep;
        }
        AscendC::LoadData(l0bLocal, bl1Local, scaleBl1Local, loadDataParams, loadData2DMxParams);
        if constexpr (!transB) {
            if (((n1 & 1) != 0 && !IS_FP4<AType>) || (n1 % B4_MIN_STEP != 0 && IS_FP4<AType>)) {
                PipeBarrier<PIPE_MTE1>();
                LocalTensor<BType> l0b = l0bLocal.template ReinterpretCast<BType>();
                uint64_t loadTimes = Cmct::Gemm::CeilDiv(Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, BLOCK_CUBE),
                                                         MIN_STEP);
                for (uint64_t i = 1; i < loadTimes; i++) {
                    loadDataParams.mStartPosition = MIN_STEP * i + Cmct::Gemm::CeilDiv(iter * baseK_, BLOCK_CUBE);
                    AscendC::LoadData(l0b[i * n1 * AscendC::BLOCK_CUBE * C0_SIZE], bl1Local, loadDataParams);
                    PipeBarrier<PIPE_MTE1>();
                }
            }
        }
    }

    __aicore__ inline void CopyInL0BTransB(const AscendC::LocalTensor<MxL0BType>& l0bLocal,
                                           const AscendC::LocalTensor<BType>& bl1Local,
                                           const AscendC::LocalTensor<ScaleBType>& scaleBl1Local, uint64_t iter,
                                           TileL1L0Param& tileL1L0Param)
    {
        // 增加对hif4 和mxfp4 的兼容
        uint64_t n1 = Cmct::Gemm::CeilDiv(tileL1L0Param.curN, AscendC::BLOCK_CUBE);

        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = Cmct::Gemm::CeilDiv(iter * baseK_, C0_SIZE);
        loadDataParams.mStep = n1;
        loadDataParams.kStep = Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, C0_SIZE);
        loadDataParams.srcStride = loadDataParams.mStep;
        loadDataParams.dstStride = loadDataParams.mStep;
        loadDataParams.ifTranspose = false;

        AscendC::LoadData2DMxParams loadData2DMxParams;
        loadData2DMxParams.xStartPosition = 0;
        loadData2DMxParams.xStep = n1;
        if constexpr (SupportType<BType, fp8_e5m2_t, fp8_e4m3fn_t>()) {
            loadData2DMxParams.yStartPosition = Cmct::Gemm::CeilDiv(iter * baseK_, MXFP_DIVISOR_SIZE);
            loadData2DMxParams.yStep = Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, MXFP_DIVISOR_SIZE);
            loadData2DMxParams.srcStride = Cmct::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE);
            loadData2DMxParams.dstStride = loadData2DMxParams.yStep;
        } else if constexpr (SupportType<BType, fp4x2_e2m1_t, fp4x2_e1m2_t>()) {
            loadData2DMxParams.yStartPosition = Cmct::Gemm::CeilDiv(iter * baseK_, MXFP_DIVISOR_SIZE);
            loadData2DMxParams.yStep = Cmct::Gemm::CeilDiv(tileL1L0Param.curKL0, C0_SIZE);
            loadData2DMxParams.srcStride = Cmct::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE);
            loadData2DMxParams.dstStride = loadData2DMxParams.yStep;
        } else if constexpr (SupportType<BType, hifloat4x2_t>()) {
            // ystartPosition, ystep 的单位为32B
            // hifp4 scale的相邻元素k方向的offset为16*4B = 64B(16是m方向分型大小), ystep 的单位为32B,
            // 所以scale的 y方向offset 数值是 元素个数的2倍
            loadData2DMxParams.yStartPosition = Cmct::Gemm::CeilDiv(iter * baseK_, HIF4_GROUP_SIZE)
                                                << HIF4_SCALE_NUM_HALF_RATIO;
            loadData2DMxParams.yStep = loadDataParams.kStep << HIF4_SCALE_NUM_HALF_RATIO;
            loadData2DMxParams.srcStride = Cmct::Gemm::CeilDiv(scaleKL1_, HIF4_GROUP_SIZE) << HIF4_SCALE_NUM_HALF_RATIO;
            loadData2DMxParams.dstStride = loadData2DMxParams.yStep;
        }
        AscendC::LoadData(l0bLocal, bl1Local, scaleBl1Local, loadDataParams, loadData2DMxParams);
    }

    __aicore__ inline void CopyOut(const AscendC::GlobalTensor<CType>& cGlobal,
                                   AscendC::GlobalTensor<WorkspaceDtype>& workspaceGlobal,
                                   AscendC::LocalTensor<float>& c1Local, uint64_t baseM, uint64_t baseN, bool isSkScene)
    {
        AscendC::DataCopyCO12DstParams intriParams;
        intriParams.nSize = baseN;
        intriParams.mSize = baseM;
        intriParams.dstStride = n_;
        intriParams.srcStride = Cmct::Gemm::Align(baseM, AscendC::BLOCK_CUBE);

        if (isSkScene) {
            intriParams.quantPre = QuantMode_t::NoQuant;
            intriParams.dstStride = min(baseN, n_);
        } else {
            if constexpr (AscendC::IsSameType<CType, bfloat16_t>::value) {
                intriParams.quantPre = QuantMode_t::F322BF16;
            } else if constexpr (AscendC::IsSameType<CType, half>::value) {
                intriParams.quantPre = QuantMode_t::F322F16;
            } else if constexpr (AscendC::IsSameType<CType, float>::value) {
                intriParams.quantPre = QuantMode_t::NoQuant;
            }
        }

        intriParams.nz2ndEn = true;
        intriParams.unitFlag = FINAL_ACCUMULATION; // 3 unitflag
        AscendC::SetFixpipeNz2ndFlag(1, 1, 1);

        if (!isSkScene) {
            AscendC::DataCopy(cGlobal, c1Local, intriParams);
        } else {
            AscendC::DataCopy(workspaceGlobal, c1Local, intriParams);
        }
    }

    __aicore__ inline void UpdateKL1(TileL1L0Param& tileL1L0Param, uint64_t iter0)
    {
        tileL1L0Param.curGmBKL1 = (iter0 + 1 == kL1Iter_) ? (singleCoreK_ - iter0 * kL1_) : kL1_;
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

    __aicore__ inline void CopyScalesInL1(const AscendC::GlobalTensor<ScaleAType>& scaleAGlobal,
                                          const AscendC::GlobalTensor<ScaleBType>& scaleBGlobal,
                                          const TileL1L0Param& tileL1L0Param, uint64_t l1Iter, uint64_t scaleL1BufId)
    {
        uint64_t kL1Offset = l1Iter * kL1_;
        if constexpr (DispatchPolicy::fullLoadMode == 0) {
            // 加载scaleA, scaleB 到l1
            if (l1Iter % scaleKL1Factor_ == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + (scaleL1BufId));

                if (transA) {
                    CopyTransScaleAMTE2(scaleAGlobal, scaleAL1Local_[l1BufferScaleAOffset_[scaleL1BufId]],
                                        tileL1L0Param.curM, scaleKL1_, kL1Offset);
                } else {
                    CopyNonTransScaleAMTE2(scaleAGlobal, scaleAL1Local_[l1BufferScaleAOffset_[scaleL1BufId]],
                                           tileL1L0Param.curM, scaleKL1_, kL1Offset);
                }

                if (transB) {
                    CopyTransScaleBMTE2(scaleBGlobal, scaleBL1Local_[l1BufferScaleBOffset_[scaleL1BufId]],
                                        tileL1L0Param.curN, kL1Offset);
                } else {
                    CopyNontransScaleBMTE2(scaleBGlobal, scaleBL1Local_[l1BufferScaleBOffset_[scaleL1BufId]],
                                           tileL1L0Param.curN, kL1Offset);
                }
            }
        }
    }

    __aicore__ inline void CopyNonTransScaleAMTE2(const GlobalTensor<ScaleAType>& aScaleGlobal,
                                                  const LocalTensor<ScaleAType>& aScaleL1Local, uint64_t curML1,
                                                  uint64_t curKL1, uint64_t kL1Offset)
    {
        if (DispatchPolicy::fullLoadMode != 0 && kL1Offset != 0) {
            return;
        }

        uint64_t curScaleKL1 = curKL1;
        if (kL1Offset + curScaleKL1 > singleCoreK_) {
            curScaleKL1 = singleCoreK_ - kL1Offset;
        }

        uint64_t nDim = curML1;
        uint64_t dDim = Cmct::Gemm::CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE);
        uint64_t offsetScaleAGM = kL1Offset / MXFP_DIVISOR_SIZE;

        GlobalTensor<half> aScaleGlobalB16;
        aScaleGlobalB16.SetGlobalBuffer(((__gm__ half*)(aScaleGlobal.GetPhyAddr())));
        auto aScaleL1LocalImpl = aScaleL1Local.template ReinterpretCast<half>();

        AscendC::Dn2NzParams dn2nzParams;
        dn2nzParams.dnNum = 1;
        if constexpr (SupportType<BType, hifloat4x2_t>()) {
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

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
        AscendC::DataCopy(aScaleL1LocalImpl, aScaleGlobalB16[offsetScaleAGM], dn2nzParams, true);
#else
        AscendC::DataCopy(aScaleL1LocalImpl, aScaleGlobalB16[offsetScaleAGM], dn2nzParams);
#endif
    }

    __aicore__ inline void CopyTransScaleAMTE2(const GlobalTensor<ScaleAType>& aScaleGlobal,
                                               const LocalTensor<ScaleAType>& aScaleL1Local, uint64_t curML1,
                                               uint64_t curKL1, uint64_t kL1Offset)
    {
        // 当前只有mxfp8支持 transA=true
        if (DispatchPolicy::fullLoadMode != 0 && kL1Offset != 0 &&
            !SupportType<AType, fp8_e5m2_t, fp8_e4m3fn_t, hifloat4x2_t>()) {
            return;
        }

        uint64_t curScaleKL1 = curKL1;
        if (kL1Offset + curScaleKL1 > k_) {
            curScaleKL1 = singleCoreK_ - kL1Offset;
        }

        uint64_t nDim = Cmct::Gemm::CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE);
        uint64_t dDim = curML1;
        uint64_t offsetScaleAGM = (kL1Offset / MXFP_DIVISOR_SIZE) * m_;

        GlobalTensor<half> aScaleGlobalB16;
        aScaleGlobalB16.SetGlobalBuffer(((__gm__ half*)(aScaleGlobal.GetPhyAddr())));
        auto aScaleL1LocalImpl = aScaleL1Local.template ReinterpretCast<half>();

        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        if constexpr (SupportType<BType, hifloat4x2_t>()) {
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

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
        AscendC::DataCopy(aScaleL1LocalImpl, aScaleGlobalB16[offsetScaleAGM], nd2nzParams, true);
#else
        AscendC::DataCopy(aScaleL1LocalImpl, aScaleGlobalB16[offsetScaleAGM], nd2nzParams);
#endif
    }

    __aicore__ inline void CopyNontransScaleBMTE2(const GlobalTensor<ScaleAType>& bScaleGlobal,
                                                  const LocalTensor<ScaleAType>& bScaleL1Local, uint64_t curNL1,
                                                  uint64_t kL1Offset)
    {
        if constexpr (!SupportType<BType, fp8_e5m2_t, fp8_e4m3fn_t, hifloat4x2_t, fp4x2_e2m1_t>()) {
            return;
        }

        uint64_t curScaleKL1 = scaleKL1_;
        if (kL1Offset + curScaleKL1 > k_) {
            curScaleKL1 = singleCoreK_ - kL1Offset;
        }

        uint64_t nDim = Cmct::Gemm::CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE);
        uint64_t dDim = curNL1;

        GlobalTensor<half> bScaleGlobalB16;
        bScaleGlobalB16.SetGlobalBuffer(((__gm__ half*)(bScaleGlobal.GetPhyAddr())));
        auto bScaleL1LocalImpl = bScaleL1Local.template ReinterpretCast<half>();

        uint64_t offsetScaleBGM = (kL1Offset / MXFP_DIVISOR_SIZE) * n_;

        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        if constexpr (AscendC::SupportType<BType, hifloat4x2_t>()) {
            offsetScaleBGM = offsetScaleBGM << HIF4_SCALE_NUM_HALF_RATIO;
            nd2nzParams.nValue = nDim << HIF4_SCALE_NUM_HALF_RATIO;
            nd2nzParams.dValue = dDim;
            nd2nzParams.srcNdMatrixStride = 0;
            nd2nzParams.srcDValue = n_;
            nd2nzParams.dstNzC0Stride = Cmct::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) << HIF4_SCALE_NUM_HALF_RATIO;
        } else {
            nd2nzParams.nValue = nDim;
            nd2nzParams.dValue = dDim;
            nd2nzParams.srcNdMatrixStride = 0;
            nd2nzParams.srcDValue = n_;
            nd2nzParams.dstNzC0Stride = Cmct::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE);
        }

        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 0;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
        AscendC::DataCopy(bScaleL1LocalImpl, bScaleGlobalB16[offsetScaleBGM], nd2nzParams, true);
#else
        AscendC::DataCopy(bScaleL1LocalImpl, bScaleGlobalB16[offsetScaleBGM], nd2nzParams);
#endif
    }

    __aicore__ inline void CopyTransScaleBMTE2(const GlobalTensor<ScaleAType>& bScaleGlobal,
                                               const LocalTensor<ScaleAType>& bScaleL1Local, uint64_t curNL1,
                                               uint64_t kL1Offset)
    {
        uint64_t curScaleKL1 = scaleKL1_;
        if (kL1Offset + curScaleKL1 > k_) {
            curScaleKL1 = singleCoreK_ - kL1Offset;
        }
        uint64_t nDim = curNL1;
        uint64_t dDim = Cmct::Gemm::CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE);

        GlobalTensor<half> bScaleGlobalB16;
        bScaleGlobalB16.SetGlobalBuffer(((__gm__ half*)(bScaleGlobal.GetPhyAddr())));
        auto bScaleL1LocalImpl = bScaleL1Local.template ReinterpretCast<half>();

        uint64_t offsetScaleBGM = kL1Offset / MXFP_DIVISOR_SIZE;

        AscendC::Dn2NzParams dn2nzParams;
        dn2nzParams.dnNum = 1;

        if constexpr (SupportType<BType, hifloat4x2_t>()) {
            offsetScaleBGM = offsetScaleBGM << HIF4_SCALE_NUM_HALF_RATIO;
            dn2nzParams.dValue = nDim;
            dn2nzParams.nValue = dDim << HIF4_SCALE_NUM_HALF_RATIO;
            dn2nzParams.srcDnMatrixStride = 0;
            dn2nzParams.srcDValue = Cmct::Gemm::CeilDiv(k_, MXFP_DIVISOR_SIZE) << HIF4_SCALE_NUM_HALF_RATIO;
            dn2nzParams.dstNzC0Stride = Cmct::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) << HIF4_SCALE_NUM_HALF_RATIO;
        } else {
            dn2nzParams.dValue = nDim;
            dn2nzParams.nValue = dDim;
            dn2nzParams.srcDnMatrixStride = 0;
            dn2nzParams.srcDValue = Cmct::Gemm::CeilDiv(k_, MXFP_DIVISOR_SIZE);
            dn2nzParams.dstNzC0Stride = Cmct::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE);
        }

        dn2nzParams.dstNzNStride = 1;
        dn2nzParams.dstNzMatrixStride = 0;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
        AscendC::DataCopy(bScaleL1LocalImpl, bScaleGlobalB16[offsetScaleBGM], dn2nzParams, true);
#else
        AscendC::DataCopy(bScaleL1LocalImpl, bScaleGlobalB16[offsetScaleBGM], dn2nzParams);
#endif
    }

    __aicore__ inline void CopyAInL1(AscendC::GlobalTensor<AType> aGlobal, TileL1L0Param tileL1L0Param,
                                     uint64_t offsetA, uint64_t offsetAL1, uint64_t l1Iter)
    {
        if constexpr (DispatchPolicy::fullLoadMode == 0) {
            InitA1(aL1Local_[offsetAL1], tileL1L0Param);
            if (dn2nzMTE2TransA_) {
                CopyInA1Dn2NZ(aGlobal[offsetA], aL1Local_[offsetAL1], tileL1L0Param);
            } else {
                CopyInA1(aGlobal[offsetA], aL1Local_[offsetAL1], tileL1L0Param);
            }
        }
    }

    __aicore__ inline void CopyBInL1(AscendC::GlobalTensor<BType> bGlobal, TileL1L0Param tileL1L0Param,
                                     uint64_t l1BufId, uint64_t l1Iter)
    {
        if constexpr (!(IS_FP4<AType> && transB)) {
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

private:
    __aicore__ inline bool NeedBias(uint64_t kIter0, uint64_t kIter1) { return isBias_ && kIter0 == 0 && kIter1 == 0; }

    __aicore__ inline void Mmad(AscendC::MmadParams& mmadParams, uint64_t l0cOffset, uint64_t l0abOffset,
                                uint64_t biasOffset, bool needBias)
    {
        mmadParams.cmatrixSource = needBias;
        if (needBias) {
            AscendC::Mmad(c1Local_[l0cOffset], l0aLocal_[l0abOffset], l0bLocal_[l0abOffset], biasBt_[biasOffset],
                          mmadParams);
        } else {
            mmadParams.cmatrixSource = false;
            AscendC::Mmad(c1Local_[l0cOffset], l0aLocal_[l0abOffset], l0bLocal_[l0abOffset], mmadParams);
        }
    }

private:
    constexpr static uint64_t HALF_L0C_SIZE = AscendC::TOTAL_L0C_SIZE / DOUBLE_BUFFER_COUNT /
                                              sizeof(float);                  // 单位：元素个数
    constexpr static uint64_t HALF_L0A_SIZE = L0A_SIZE / DOUBLE_BUFFER_COUNT; // L0A_SIZE 定义在cmct的arch.h中, 单位B
    constexpr static int32_t C0_SIZE = AscendC::AuxGetC0Size<AType>(); // Atype=8Bit,C0_SIZE=32; Atype=4Bit,C0_SIZE=64;
    constexpr static int32_t BIAS_C0 = AscendC::AuxGetC0Size<BiasType>();
    uint64_t m_;
    uint64_t n_;
    uint64_t k_;

    uint64_t l1BufNum_{1UL};
    uint64_t kL1Iter_{0UL};
    uint64_t kL1_{1UL};
    uint64_t scaleKL1_{1UL};
    uint64_t scaleKL1DP_{1UL};
    uint64_t scaleKL1Factor_{1UL};
    uint64_t baseM_{16UL};
    uint64_t baseN_{16UL};
    uint64_t baseK_{16UL};
    uint64_t singleCoreK_{0UL};
    uint64_t elementSizeRatio_ = 0; // 用于右移的参数，0表示除1.
    bool isBias_{false};
    bool enableL0cPingPong_{false};

    // K为外轴时，是否需要MTE2 dn2nz 搬运A/B 数据到l1同时完成转置，避免在mte1做转置导致指令拆分
    bool dn2nzMTE2TransA_ = false;
    bool dn2nzMTE2TransB_ = false;

    uint16_t biasBufId_{0};
    uint64_t abL1LoopCnt_{0UL};
    uint64_t scaleLoopCnt_{0UL};
    uint64_t l0PingPong_{0UL};
    uint64_t l0cPingPong_{0UL};
    uint64_t aL1OneBuffer_{0UL};
    uint64_t bL1OneBuffer_{0UL};
    uint64_t scaleAL1OneBuffer_{0UL};
    uint64_t scaleBL1OneBuffer_{0UL};
    uint64_t l1BufferAOffset_[4] = {0UL};      // default 4 buffer
    uint64_t l1BufferBOffset_[4] = {0UL};      // default 4 buffer
    uint64_t l1BufferScaleAOffset_[2] = {0UL}; // default 2 buffer
    uint64_t l1BufferScaleBOffset_[2] = {0UL}; // default 2 buffer
    uint64_t l1BufferBiasOffset_[2] = {0UL};   // default 2 buffer
    uint64_t halfL0AEleNum_ = {0UL};

    AscendC::LocalTensor<MxL0AType> l0aLocal_{AscendC::TPosition::A2, 0, L0A_SIZE};
    AscendC::LocalTensor<MxL0BType> l0bLocal_{AscendC::TPosition::B2, 0, L0B_SIZE};

    AscendC::LocalTensor<float> c1Local_{AscendC::TPosition::CO1, 0,
                                         AscendC::TOTAL_L0C_SIZE}; // loc上数据类型，mx/hif4只能是float
    AscendC::LocalTensor<float> biasBt_{AscendC::TPosition::C2, 0, BT_SIZE};
    AscendC::LocalTensor<AType> aL1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE};
    AscendC::LocalTensor<BType> bL1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE};
    AscendC::LocalTensor<BiasType> biasL1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE / sizeof(BiasType)};

    // 偏移单位为Element
    AscendC::LocalTensor<ScaleAType> scaleAL1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE};
    AscendC::LocalTensor<ScaleBType> scaleBL1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE};
};
} // namespace Block
} // namespace Gemm
} // namespace Cmct
#endif
