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
 * \file block_rotate_quant.h
 * \brief
 */

#ifndef MATMUL_BLOCK_BLOCK_ROTATE_QUANT_H
#define MATMUL_BLOCK_BLOCK_ROTATE_QUANT_H
#include "./block_mmad.h"
#include "../utils/common_utils.h"
#include "../utils/layout_utils.h"
#include "../utils/tuple_utils.h"
#include "../policy/dispatch_policy.h"

namespace Cmct {
namespace Gemm {
namespace Block {
template <class DispatchPolicy_, class L1TileShape_, class L0TileShape_, class AType_, class BType_, class CType_,
          class BiasType_, class TileCopy_>
class BlockMmad<DispatchPolicy_, L1TileShape_, L0TileShape_, AType_, BType_, CType_, BiasType_, TileCopy_,
                AscendC::Std::enable_if_t<AscendC::Std::is_base_of_v<MatmulRotateQuant<>, DispatchPolicy_>>> {
public:
    using L0cType = float;
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BiasType = BiasType_;
    using A_T = typename AType::T;
    using B_T = typename BType::T;
    using C_T = typename CType::T;
    using Bias_T = typename BiasType::T;
    using DispatchPolicy = DispatchPolicy_;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using TupleL1L0Shape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};
    uint64_t b_{1};
    uint64_t orgKa_{1};
    uint64_t mL1_{1};
    uint64_t nL1_{1};
    uint64_t kL1_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};
    uint64_t kAlign_{1};
    uint64_t l0PingPong_{1};
    uint64_t subBlockId_{0}; // master原版: 3510侧Fixpipe subBlockId自增(乒乓), v121侧不使用
    constexpr static uint64_t BUFFER_NUM = 2;
    constexpr static uint64_t HALF_L0_SIZE = L0A_SIZE / BUFFER_NUM / sizeof(A_T);
    constexpr static uint64_t HALF_L0C_SIZE = AscendC::TOTAL_L0C_SIZE / BUFFER_NUM / sizeof(L0cType);
    constexpr static int32_t C0_SIZE = AscendC::AuxGetC0Size<typename AType::T>();

    // v121(9201/9202)回退cann_out原版单缓冲(仅单flag); v100(3510)保持ping-pong双flag预置
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
    __aicore__ inline BlockMmad()
    {
        if ASCEND_IS_AIC {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ZERO_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(SIXTH_FLAG);
        }
    }

    __aicore__ inline ~BlockMmad()
    {
        if ASCEND_IS_AIC {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(SIXTH_FLAG);
        }
    }
#else
    __aicore__ inline BlockMmad()
    {
        if ASCEND_IS_AIC {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ZERO_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(FIRST_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FIRST_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(SIXTH_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(SEVENTH_FLAG);
        }
    }

    __aicore__ inline ~BlockMmad()
    {
        if ASCEND_IS_AIC {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(FIRST_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FIRST_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(SIXTH_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(SEVENTH_FLAG);
        }
    }
#endif

public:
    template <uint64_t FULL_LOAD_MODE_ = B_FULL_LOAD_MODE>
    __aicore__ inline void Init(const TupleShape& shape, const TupleShape& tileL1, const TupleShape& tileL0)
    {
        m_ = Get<MKN_M>(shape);
        n_ = Get<MKN_N>(shape);
        k_ = Get<MKN_K>(shape);
        b_ = Get<MKN_B>(shape);
        orgKa_ = b_ * k_;
        mL1_ = Get<MKN_M>(tileL1);
        nL1_ = Get<MKN_N>(tileL1);
        kL1_ = Get<MKN_K>(tileL1);
        baseM_ = Get<MKN_M>(tileL0);
        baseN_ = Get<MKN_N>(tileL0);
        baseK_ = Get<MKN_K>(tileL0);

        uint64_t nl1Align = Cmct::Gemm::Align(nL1_, AscendC::BLOCK_CUBE);
        uint64_t kl1Align = Cmct::Gemm::Align(kL1_, AscendC::BLOCK_CUBE);
        l0PingPong_ = 0;
        subBlockId_ = 0; // master原版(3510侧)
    }

    __aicore__ inline void CopyInA1(const AscendC::GlobalTensor<A_T>& aGlobal,
                                    const AscendC::LocalTensor<A_T>& al1Local, uint64_t curML1, uint64_t curKL1,
                                    uint64_t srcDValue)
    {
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        uint64_t nDim = curML1;
        uint64_t dDim = curKL1;
        nd2nzParams.nValue = nDim;
        nd2nzParams.dValue = dDim;
        nd2nzParams.srcNdMatrixStride = 1;
        nd2nzParams.srcDValue = srcDValue;
        nd2nzParams.dstNzC0Stride = Cmct::Gemm::Align(nDim, AscendC::BLOCK_CUBE);
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = nd2nzParams.dstNzC0Stride * Cmct::Gemm::Align(dDim, AscendC::BLOCK_CUBE);
        AscendC::DataCopy(al1Local, aGlobal, nd2nzParams);
    }

    __aicore__ inline void CopyInB1(const AscendC::GlobalTensor<B_T>& rotGlobal,
                                    const AscendC::LocalTensor<B_T>& bl1Local, uint64_t curNL1, uint64_t curKL1)
    {
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        uint64_t nDim = curKL1;
        uint64_t dDim = curNL1;
        nd2nzParams.nValue = nDim;
        nd2nzParams.dValue = dDim;
        nd2nzParams.srcNdMatrixStride = 1;
        nd2nzParams.srcDValue = k_;
        nd2nzParams.dstNzC0Stride = Cmct::Gemm::Align(nDim, AscendC::BLOCK_CUBE);
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 1;
        AscendC::DataCopy(bl1Local, rotGlobal, nd2nzParams);
    }

    __aicore__ inline void CopyInA2(const AscendC::LocalTensor<A_T>& a2Local, const AscendC::LocalTensor<A_T>& al1Local,
                                    uint64_t curML1, uint64_t mL0, uint64_t kL0, uint64_t mStartPosition)
    {
        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = mStartPosition;
        loadDataParams.kStartPosition = 0;
        loadDataParams.mStep = Cmct::Gemm::CeilDiv(mL0, AscendC::BLOCK_CUBE);
        loadDataParams.kStep = Cmct::Gemm::CeilDiv(kL0, AscendC::BLOCK_CUBE);
        loadDataParams.srcStride = Cmct::Gemm::CeilDiv(curML1, AscendC::BLOCK_CUBE);
        loadDataParams.dstStride = loadDataParams.mStep;
        loadDataParams.ifTranspose = false;
        AscendC::LoadData<A_T>(a2Local, al1Local, loadDataParams);
    }

    __aicore__ inline void CopyInB2(const AscendC::LocalTensor<B_T>& b2Local, const AscendC::LocalTensor<B_T>& bl1Local,
                                    uint64_t curKL1, uint64_t nL0, uint64_t kL0)
    {
        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.mStep = Cmct::Gemm::CeilDiv(kL0, AscendC::BLOCK_CUBE);
        loadDataParams.kStep = Cmct::Gemm::CeilDiv(nL0, AscendC::BLOCK_CUBE);
        loadDataParams.srcStride = Cmct::Gemm::CeilDiv(curKL1, AscendC::BLOCK_CUBE);
        loadDataParams.dstStride = loadDataParams.kStep;
        loadDataParams.ifTranspose = true;
        AscendC::LoadData(b2Local, bl1Local, loadDataParams);
    }

    __aicore__ inline void CopyOut(const AscendC::LocalTensor<C_T>& dstLocal,
                                   const AscendC::LocalTensor<L0cType>& c1Local, uint64_t baseM, uint64_t baseN,
                                   uint64_t ubOffset)
    {
        // FixpipeParams类型为编译期差异(3510用Arch3510+subBlockId自增, v121用C310+subBlockId恒0)
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fixpipeParams;
#else
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> fixpipeParams;
#endif
        uint64_t c0 = AscendC::BLOCK_CUBE;
        fixpipeParams.nSize = Cmct::Gemm::Align(baseN, c0);
        fixpipeParams.mSize = baseM;
        fixpipeParams.srcStride = Cmct::Gemm::Align(baseM, AscendC::BLOCK_CUBE);
        fixpipeParams.dstStride = fixpipeParams.nSize;
        fixpipeParams.quantPre = QuantMode_t::F322BF16;
        fixpipeParams.dualDstCtl = 0;
        fixpipeParams.unitFlag = 0;
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
        fixpipeParams.subBlockId = 0; // v121(1:1): cann_out同款, 无路由恒0
#else
        fixpipeParams.subBlockId = (subBlockId_++) & 0x1; // master原版(1:2): 乒乓自增
#endif
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = baseM * Cmct::Gemm::Align(baseN, c0) / AscendC::BLOCK_CUBE;
        fixpipeParams.params.dstNdStride = baseM * Cmct::Gemm::Align(baseN, c0);
        AscendC::Fixpipe<C_T, L0cType, AscendC::Impl::CFG_ROW_MAJOR_UB>(dstLocal[ubOffset], c1Local, fixpipeParams);
    }

    template <typename T>
    __aicore__ inline void operator()(T cTensor, AscendC::GlobalTensor<A_T> aGlobal,
                                      AscendC::GlobalTensor<B_T> rotGlobal, TupleL1L0Shape tileShape, bool isFirstRound)
    {
        uint64_t curML1 = Get<MNK_M>(tileShape);
        uint64_t curNL1 = Get<MNK_N>(tileShape);
        uint64_t curKL1 = Get<MNK_K>(tileShape);
        uint64_t mL1 = Get<MNK_B>(tileShape);
        uint64_t curML0 = Get<MNK_M0>(tileShape);
        uint64_t curNL0 = Get<MNK_N0>(tileShape);

        aL1OneBuffer_ = Cmct::Gemm::Align(mL1, AscendC::BLOCK_CUBE) * Cmct::Gemm::Align(curKL1, AscendC::BLOCK_CUBE);

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
        // v121: 回退cann_out原版单缓冲流水(无L0 ping-pong偏移, 单flag), 机器验证的正确基线
        AscendC::LocalTensor<L0cType> c1Local = c1Local_;
        uint64_t bl1Offset = aL1OneBuffer_ * NUM_TWO;
        if (b_ > 1) {
            bL1OneBuffer_ = Cmct::Gemm::Align(curKL1, AscendC::BLOCK_CUBE) *
                            Cmct::Gemm::Align(curNL1, AscendC::BLOCK_CUBE);
        }

        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ZERO_FLAG);
        CopyInA1(aGlobal, l1Local_, curML1, curKL1, orgKa_);
        if (isFirstRound || b_ > 1) {
            CopyInB1(rotGlobal, l1Local_[bl1Offset], curNL1, curKL1);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(SIXTH_FLAG);
        CopyInA2(l0aLocal_, l1Local_, curML1, curML1, curKL1, 0);
        CopyInB2(l0bLocal_, l1Local_[bl1Offset], curKL1, curNL0, curKL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ZERO_FLAG);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(SIXTH_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(SIXTH_FLAG);
        AscendC::MmadParams mmadParams;
        mmadParams.m = curML1;
        mmadParams.n = curNL0;
        mmadParams.k = curKL1;
        mmadParams.disableGemv = true;
        mmadParams.unitFlag = 0;
        mmadParams.cmatrixInitVal = true;
        Mmad(mmadParams); // v121贴cann_out: 无偏移参数

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(SIXTH_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(ZERO_FLAG);
        CopyOut(cTensor, c1Local, curML1, curNL0, 0); // v121贴cann_out: 局部c1Local(无偏移), ubOffset=0
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
#else
        uint64_t mte1Flag = ((l0PingPong_ & 0x1) + SIXTH_FLAG);
        uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
        uint64_t l0cOffset = (l0PingPong_ & 0x1) * HALF_L0C_SIZE;
        AscendC::LocalTensor<L0cType> c1Local = c1Local_[l0cOffset];
        uint64_t al1Offset = (l0PingPong_ & 0x1) * aL1OneBuffer_;
        uint64_t bl1Offset = aL1OneBuffer_ * NUM_TWO;
        if (b_ > 1) {
            bL1OneBuffer_ = Cmct::Gemm::Align(curKL1, AscendC::BLOCK_CUBE) *
                            Cmct::Gemm::Align(curNL1, AscendC::BLOCK_CUBE);
            bl1Offset += (l0PingPong_ & 0x1) * bL1OneBuffer_;
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l0PingPong_ & 0x1);
        CopyInA1(aGlobal, l1Local_[al1Offset], curML1, curKL1, orgKa_);
        if (isFirstRound || b_ > 1) {
            CopyInB1(rotGlobal, l1Local_[bl1Offset], curNL1, curKL1);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l0PingPong_ & 0x1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l0PingPong_ & 0x1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(static_cast<uint16_t>(mte1Flag));
        CopyInA2(l0aLocal_[l0Offset], l1Local_[al1Offset], curML1, curML1, curKL1, 0);
        CopyInB2(l0bLocal_[l0Offset], l1Local_[bl1Offset], curKL1, curNL0, curKL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l0PingPong_ & 0x1);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(static_cast<uint16_t>(mte1Flag));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(static_cast<uint16_t>(mte1Flag));
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0PingPong_ & 0x1);
        AscendC::MmadParams mmadParams;
        mmadParams.m = curML1;
        mmadParams.n = curNL0;
        mmadParams.k = curKL1;
        mmadParams.disableGemv = true;
        mmadParams.unitFlag = 0;
        mmadParams.cmatrixInitVal = true;
        Mmad(mmadParams, l0cOffset, l0Offset);

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(static_cast<uint16_t>(mte1Flag));
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0PingPong_ & 0x1);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0PingPong_ & 0x1);
        CopyOut(cTensor, c1Local_[l0cOffset], curML1, curNL0, 0); // master原版(1:2): CopyOut内subBlockId自增
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0PingPong_ & 0x1);
        l0PingPong_++;
#endif
    }

private:
    __aicore__ inline void Mmad(AscendC::MmadParams& mmadParams
#if !(defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202)))
                                ,
                                uint64_t l0cOffset, uint64_t l0abOffset
#endif
    )
    {
        mmadParams.cmatrixSource = false;
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
        AscendC::Mmad(c1Local_, l0aLocal_, l0bLocal_, mmadParams); // v121贴cann_out: 无偏移
#else
        AscendC::Mmad(c1Local_[l0cOffset], l0aLocal_[l0abOffset], l0bLocal_[l0abOffset], mmadParams);
#endif
    }

private:
    constexpr static uint16_t MKN_M = 0;
    constexpr static uint16_t MKN_N = 1;
    constexpr static uint16_t MKN_K = 2;
    constexpr static uint16_t MKN_B = 3;
    constexpr static uint16_t ZERO_FLAG = 0;
    constexpr static uint16_t FIRST_FLAG = 1;
    constexpr static uint16_t SIXTH_FLAG = 6;
    constexpr static uint16_t SEVENTH_FLAG = 7;
    constexpr static uint16_t M_ALIGN = 16;
    constexpr static uint16_t NUM_TWO = 2;
    uint64_t aL1OneBuffer_ = 0;
    uint64_t bL1OneBuffer_ = 0;
    AscendC::LocalTensor<A_T> l0aLocal_{AscendC::TPosition::A2, 0, L0A_SIZE};
    AscendC::LocalTensor<B_T> l0bLocal_{AscendC::TPosition::B2, 0, L0B_SIZE};
    AscendC::LocalTensor<L0cType> c1Local_{AscendC::TPosition::CO1, 0, AscendC::TOTAL_L0C_SIZE};
    AscendC::LocalTensor<A_T> l1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE};
};
} // namespace Block
} // namespace Gemm
} // namespace Cmct
#endif
