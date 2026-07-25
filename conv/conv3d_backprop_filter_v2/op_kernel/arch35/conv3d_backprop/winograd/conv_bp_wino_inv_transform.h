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
 * \file conv_bp_wino_inv_transform.h
 * \brief
 */

#ifndef CONV_BP_WINO_INV_TRANSFORM_H
#define CONV_BP_WINO_INV_TRANSFORM_H

#include "conv_bp_wino_util.h"

using namespace AscendC;

namespace WinoInvBufUtil {
// 需要申请26个CoutCin空间,16个用来放原始数据,9个用来放逆变换转置后的数据
static constexpr uint32_t COUT_CIN_BUF_CNT = 25;
static constexpr uint8_t CROSS_CORE_INTERLEAVE_MTE3_SYNC_FLAG = 11;

template <typename TilingT>
static constexpr __aicore__ inline uint32_t InvTransSinglePointBufSize()
{
    constexpr uint32_t sizeCoutCin = BlockConfig::SingleShapeInvTransformCout<TilingT>() *
                                     BlockConfig::SingleShapeCin<TilingT>();
    return ConstexprMaths::AlignUp(sizeCoutCin, C0<float>());
}

template <typename TilingT>
static constexpr __aicore__ inline uint32_t InvTransBufSize()
{
    return InvTransSinglePointBufSize<TilingT>() * COUT_CIN_BUF_CNT;
}

template <typename TilingT>
static constexpr __aicore__ inline uint32_t GetInvBufTotalSizeInBytes()
{
    constexpr uint8_t BufCnt = BlockConfig::InvTransformBufCnt<TilingT>();
    constexpr uint32_t bufSize = InvTransBufSize<TilingT>() * BufCnt * sizeof(float);
    static_assert(bufSize < TOTAL_UB_SIZE, "illegal buffer size");
    return bufSize;
}
} // namespace WinoInvBufUtil

template <typename DstT, typename TilingT>
class WinoInvTransformer {
public:
    static constexpr uint32_t INV_TRANS_BUF_SIZE = WinoInvBufUtil::InvTransBufSize<TilingT>();
    static constexpr uint32_t INV_TRANS_SINGLE_POINT_BUF_SIZE = WinoInvBufUtil::InvTransSinglePointBufSize<TilingT>();

    __aicore__ inline explicit WinoInvTransformer(__gm__ DstT* yGm, __gm__ float* tailGm)
    {
        yGm_.SetGlobalBuffer(yGm);
        tailGm_.SetGlobalBuffer(tailGm);
    }

    __aicore__ inline void Init()
    {
        TPipe* pipe = GetTPipePtr();
        v2mte3_ = pipe->AllocEventID<HardEvent::V_MTE3>();
        mte32mte2_ = pipe->AllocEventID<HardEvent::MTE3_MTE2>();

        mte22v_ = pipe->AllocEventID<HardEvent::MTE2_V>();
        v2mte2_[0] = pipe->AllocEventID<HardEvent::V_MTE2>();
        v2mte2_[1] = pipe->AllocEventID<HardEvent::V_MTE2>();
    }

    template <bool WriteToTailGM = false, typename QueConfig>
    __aicore__ inline void TransformOutput(CVSyncQue<QueConfig>& l0c2ubSync, const CoutCinRange& localBlock,
                                           const uint32_t cinSrc, const LocalTensor<float>& vBuf,
                                           uint32_t tailKGroupIdx, uint32_t tailKGroups, uint16_t tailBlockId,
                                           bool atomicAdd)
    {
        constexpr uint16_t aivNums = AivNumInBlock();
        constexpr uint16_t singleBlockCout = BlockConfig::SingleShapeInvTransformCout<TilingT>() * aivNums;
        constexpr uint8_t BufCnt = BlockConfig::InvTransformBufCnt<TilingT>();
        const uint16_t aivId = GetSubBlockIdx();
        if (atomicAdd) {
            SetAtomicAdd<DstT>();
        }

        for (uint32_t coutIdxInBlock = 0; coutIdxInBlock < localBlock.coutLength; coutIdxInBlock += singleBlockCout) {
            const uint16_t coutLengthInBlock = Std::min(singleBlockCout, localBlock.coutLength - coutIdxInBlock);

            l0c2ubSync.WaitData();

            const uint16_t localCoutLength = Ops::Base::CeilDiv(coutLengthInBlock, aivNums);
            const uint16_t localCoutOffset = localCoutLength * aivId;
            // 尾轮不逆变换，累加完在做一次逆变换
            if (localCoutOffset < coutLengthInBlock) {
                const uint32_t processCoutLength = Std::min(localCoutLength, coutLengthInBlock - localCoutOffset);
                const uint32_t coutCin = processCoutLength * localBlock.cinLength;

                LocalTensor<float> buf = vBuf[bufIdx_ * INV_TRANS_BUF_SIZE];
                ProcessInvTransform(reinterpret_cast<__ubuf__ float*>(buf.GetPhyAddr()), coutCin,
                                    Ops::Base::CeilDiv(coutCin, VL<float>()));

                const uint32_t coutIdx = localBlock.coutIdx + coutIdxInBlock + localCoutOffset;

                SetFlag<HardEvent::V_MTE3>(v2mte3_);
                WaitFlag<HardEvent::V_MTE3>(v2mte3_);

                if constexpr (WriteToTailGM) {
                    // 尾轮没实现非fp32的输出，当前dw也没必要实现
                    static_assert(Std::is_same_v<DstT, float>, "only support fp32 when enable tail write");

                    DataCopyExtParams params;
                    params.blockCount = processCoutLength;
                    params.blockLen = localBlock.cinLength * KERNEL_3x3 * sizeof(DstT);
                    params.srcStride = 0;
                    params.dstStride = 0;

                    constexpr uint32_t TailBlockSize = BlockConfig::SingleShapeCout<TilingT>() *
                                                       BlockConfig::SingleShapeCin<TilingT>() * KERNEL_3x3;

                    uint64_t gmOffset = (tailBlockId * tailKGroups + tailKGroupIdx) * TailBlockSize +
                                        localBlock.cinLength * KERNEL_3x3 * (coutIdx - localBlock.coutIdx);
                    DataCopyPad<float, PaddingMode::Compact>(
                        tailGm_[gmOffset],
                        buf[F23_TRANSFORM_TILE_ELEMENTS_16 * INV_TRANS_SINGLE_POINT_BUF_SIZE].ReinterpretCast<DstT>(),
                        params);
                } else {
                    CopyOut(
                        localBlock, processCoutLength, coutIdx, cinSrc,
                        buf[F23_TRANSFORM_TILE_ELEMENTS_16 * INV_TRANS_SINGLE_POINT_BUF_SIZE].ReinterpretCast<DstT>());
                }
            }

            l0c2ubSync.DeQue();
            bufIdx_ = (bufIdx_ + 1) % BufCnt;
        }

        if (atomicAdd) {
            SetAtomicNone();
        }
    }

    __aicore__ inline void TailInterleaveWrite(const CoutCinRange& localBlock, const uint32_t cinSrc,
                                               uint32_t tailKGroup, uint32_t tailKGroupIdx, uint16_t tailBlockId)
    {
        CrossCoreSetFlag<0, PIPE_MTE3>(WinoInvBufUtil::CROSS_CORE_INTERLEAVE_MTE3_SYNC_FLAG);
        CrossCoreWaitFlag<0, PIPE_MTE2>(WinoInvBufUtil::CROSS_CORE_INTERLEAVE_MTE3_SYNC_FLAG);

        RemainderDistributionSpliter splitter(localBlock.coutLength, tailKGroup);
        uint16_t coutOffset, coutLength;
        splitter.GetSplit(tailKGroupIdx, coutOffset, coutLength);

        if (coutLength == 0 || GetSubBlockIdx() != 0) {
            return;
        }

        uint32_t bufLength = Ops::Base::CeilAlign(coutLength * localBlock.cinLength * KERNEL_3x3, C0<float>());
        uint32_t availableBufCnt = TOTAL_UB_SIZE / (bufLength * sizeof(float));
        // 切PingPong
        uint32_t inputCnt = Std::min((availableBufCnt - 1) / 2, Ops::Base::CeilDiv(tailKGroup, 2u));

        LocalTensor<float> accumulateBuf = AccumulateTailData(localBlock, coutOffset, coutLength, tailKGroup,
                                                              tailBlockId, bufLength, inputCnt);

        SetFlag<HardEvent::V_MTE3>(v2mte3_);
        WaitFlag<HardEvent::V_MTE3>(v2mte3_);

        CopyOut(localBlock, coutLength, localBlock.coutIdx + coutOffset, cinSrc, accumulateBuf);
    }

    __aicore__ inline void BlockMTE2ByMTE3() const
    {
        SetFlag<HardEvent::MTE3_MTE2>(mte32mte2_);
        WaitFlag<HardEvent::MTE3_MTE2>(mte32mte2_);
    }

private:
    __aicore__ inline LocalTensor<float> AccumulateTailData(const CoutCinRange& localBlock, uint32_t coutOffset,
                                                            uint32_t coutLength, uint32_t tailKGroup,
                                                            uint16_t tailBlockId, uint32_t bufLength,
                                                            uint32_t inputCnt) const
    {
        uint32_t bufLengthInBytes = bufLength * sizeof(float);
        uint32_t inputBufLengthInBytes = inputCnt * bufLengthInBytes;

        constexpr uint32_t PING_PONG_BUF_CNT = 2;
        LocalTensor<float> accumulateBuf(TPosition::VECCALC, inputCnt * PING_PONG_BUF_CNT * bufLengthInBytes,
                                         bufLengthInBytes);

        bool pingPongFlag = false;
        constexpr uint32_t TailBlockSize = BlockConfig::SingleShapeCout<TilingT>() *
                                           BlockConfig::SingleShapeCin<TilingT>() * KERNEL_3x3;

        uint32_t loadCnt = Ops::Base::CeilDiv(tailKGroup, inputCnt);

        SetFlag<HardEvent::V_MTE2>(v2mte2_[0]);
        SetFlag<HardEvent::V_MTE2>(v2mte2_[1]);

        for (uint32_t i = 0; i < loadCnt; i++) {
            LocalTensor<float> inBuf(TPosition::VECCALC, inputBufLengthInBytes * pingPongFlag, inputBufLengthInBytes);

            uint32_t startGroupIdx = i * inputCnt;
            uint32_t loadGroups = Std::min(inputCnt, tailKGroup - startGroupIdx);
            DataCopyExtParams params;
            params.blockCount = loadGroups;
            params.blockLen = coutLength * localBlock.cinLength * KERNEL_3x3 * sizeof(float);
            params.srcStride = TailBlockSize * sizeof(float) - params.blockLen;
            params.dstStride = 0;

            TEventID v2mte2Flag = v2mte2_[pingPongFlag];
            WaitFlag<HardEvent::V_MTE2>(v2mte2Flag);

            DataCopyPad<float, PaddingMode::Normal>(inBuf,
                                                    tailGm_[(tailBlockId * tailKGroup + startGroupIdx) * TailBlockSize +
                                                            localBlock.cinLength * KERNEL_3x3 * coutOffset],
                                                    params, {false, 0, 0, 0});

            SetFlag<HardEvent::MTE2_V>(mte22v_);
            WaitFlag<HardEvent::MTE2_V>(mte22v_);
            if (i == 0) {
                Duplicate(accumulateBuf, 0.0f, static_cast<int32_t>(bufLength));
            }
            Accumulate(reinterpret_cast<__ubuf__ float*>(inBuf.GetPhyAddr()),
                       reinterpret_cast<__ubuf__ float*>(accumulateBuf.GetPhyAddr()), loadGroups, bufLength,
                       Ops::Base::CeilDiv(bufLength, VL<float>()));

            SetFlag<HardEvent::V_MTE2>(v2mte2Flag);
            pingPongFlag = !pingPongFlag;
        }

        WaitFlag<HardEvent::V_MTE2>(v2mte2_[0]);
        WaitFlag<HardEvent::V_MTE2>(v2mte2_[1]);

        return accumulateBuf;
    }

    __aicore__ inline void CopyOut(const CoutCinRange& localBlock, uint32_t processCoutLength, uint32_t coutIdx,
                                   uint32_t cinSrc, const LocalTensor<DstT>& buf)
    {
        DataCopyExtParams params;
        params.blockCount = processCoutLength;
        params.blockLen = localBlock.cinLength * KERNEL_3x3 * sizeof(DstT);
        params.srcStride = 0;
        params.dstStride = (static_cast<uint64_t>(cinSrc) - localBlock.cinLength) * KERNEL_3x3 * sizeof(DstT);
        uint64_t gmOffset = (static_cast<uint64_t>(coutIdx) * cinSrc + localBlock.cinIdx) * KERNEL_3x3;
        DataCopyPad<DstT, PaddingMode::Compact>(yGm_[gmOffset], buf, params);
    }

    static constexpr uint32_t KERNEL_3 = 3;
    static constexpr uint32_t KERNEL_3x3 = 9;

    __simd_vf__ static inline void Accumulate(__ubuf__ float* buf, __ubuf__ float* accBuf, uint16_t blockCnt,
                                              uint32_t blockLength, uint16_t blockLoopCnt)
    {
        uint16_t blockCntOneLess = blockCnt - 1;
        uint32_t maskValue = blockLength;
        int32_t loopStride = static_cast<int32_t>(VL<float>()) - static_cast<int32_t>((blockCnt - 1) * blockLength);

        for (uint16_t i = 0; i < blockLoopCnt; i++) {
            Reg::MaskReg mask = Reg::UpdateMask<float>(maskValue);
            Reg::RegTensor<float> acc;
            Reg::Duplicate(acc, 0, mask);

            for (uint16_t n = 0; n < blockCntOneLess; n++) {
                Reg::RegTensor<float> s;
                Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(s, buf, static_cast<int32_t>(blockLength));
                Reg::Add(acc, acc, s, mask);
            }

            Reg::RegTensor<float> s;
            Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(s, buf, loopStride);
            Reg::Add(acc, acc, s, mask);

            Reg::RegTensor<float> a;
            Reg::LoadAlign(a, accBuf);
            Reg::Add(acc, acc, a, mask);

            Reg::StoreAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(accBuf, acc, VL<float>(), mask);
        }
    }

    __simd_vf__ static inline void ProcessInvTransform(__ubuf__ float* buf, const uint32_t coutCinLength,
                                                       const uint16_t loopCnt)
    {
        using namespace Reg;
        RegTensor<float> value0P5;
        Duplicate(value0P5, 0.5f);

        constexpr uint32_t singlePointSize = INV_TRANS_SINGLE_POINT_BUF_SIZE;
        // 当前正变换结束后点按列优先排列
        __ubuf__ float* src0 = buf;
        __ubuf__ float* src1 = buf + singlePointSize;
        __ubuf__ float* src2 = buf + singlePointSize * 2;
        __ubuf__ float* src3 = buf + singlePointSize * 3;

        RegTensor<uint32_t> seq, tmp9, index;

        Arange(reinterpret_cast<RegTensor<int32_t>&>(seq), 0);
        Duplicate(tmp9, 9);
        MaskReg maskAll = CreateMask<uint32_t, MaskPattern::ALL>();
        Mul(index, seq, tmp9, maskAll);
        __ubuf__ float* transposeBuf = buf + F23_TRANSFORM_TILE_ELEMENTS_16 * singlePointSize;
        __ubuf__ float* dst = transposeBuf;

        uint32_t maskValue = coutCinLength;

        for (uint16_t i = 0; i < loopCnt; i++) {
            MaskReg mask = UpdateMask<float>(maskValue);

            constexpr uint32_t pointRowStride = singlePointSize * F23_TRANSFORM_TILE_SIZE_4;
            RegTensor<float> col0d0, col0d1, col0d2;
            TransformCol(src0, src1, src2, src3, mask, value0P5, col0d0, col0d1, col0d2, pointRowStride);

            RegTensor<float> col1d0, col1d1, col1d2;
            TransformCol(src0, src1, src2, src3, mask, value0P5, col1d0, col1d1, col1d2, pointRowStride);

            RegTensor<float> col2d0, col2d1, col2d2;
            TransformCol(src0, src1, src2, src3, mask, value0P5, col2d0, col2d1, col2d2, pointRowStride);

            RegTensor<float> col3d0, col3d1, col3d2;
            constexpr int32_t nextColStride = -3 * pointRowStride + VL<float>();
            TransformCol(src0, src1, src2, src3, mask, value0P5, col3d0, col3d1, col3d2, nextColStride);

            __ubuf__ float* dst0 = dst;

            TransformRowWithCastAndSetter(dst0, col0d0, col1d0, col2d0, col3d0, value0P5, index, mask);
            TransformRowWithCastAndSetter(dst0, col0d1, col1d1, col2d1, col3d1, value0P5, index, mask);
            TransformRowWithCastAndSetter(dst0, col0d2, col1d2, col2d2, col3d2, value0P5, index, mask);

            dst += VL<float>() * KERNEL_3x3;
        }

        // scatter完后在重新做cast，把float转b16后空的2个字节移除，不能直接用b16做scatter，bank冲突太严重
        if constexpr (!Std::is_same_v<DstT, float>) {
            B32ToB16(transposeBuf, loopCnt, maskAll);
        }
    }

    __simd_callee__ static inline void TransformRowWithCastAndSetter(
        __ubuf__ float*& dst0, Reg::RegTensor<float>& c0, Reg::RegTensor<float>& c1, Reg::RegTensor<float>& c2,
        Reg::RegTensor<float>& c3, Reg::RegTensor<float>& value0P5, Reg::RegTensor<uint32_t>& index, Reg::MaskReg& mask)
    {
        Reg::RegTensor<float> r0, r1, r2;
        TransformRowAndCastInZero(mask, value0P5, c0, c1, c2, c3, r0, r1, r2);

        Scatter(dst0, r0, index, mask);
        ++dst0;
        Scatter(dst0, r1, index, mask);
        ++dst0;
        Scatter(dst0, r2, index, mask);
        ++dst0;
    }

    __simd_callee__ static inline void B32ToB16(__ubuf__ float* transposeBuf, uint16_t loopCnt, Reg::MaskReg& maskAll)
    {
        using namespace Reg;
        if constexpr (!Std::is_same_v<DstT, float>) {
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

            __ubuf__ float* loadSrc = transposeBuf;
            __ubuf__ float* castDst = transposeBuf;
            for (uint16_t i = 0; i < loopCnt; i++) {
                for (uint16_t j = 0; j < KERNEL_3x3; j++) {
                    RegTensor<float> t0;
                    LoadAlign<float, PostLiteral::POST_MODE_UPDATE>(t0, loadSrc, VL<float>());
                    StoreAlign<float, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK_B32>(
                        castDst, t0, VL<float>() / 2, maskAll);
                }
            }
        }
    }

    __simd_callee__ static inline void TransformCol(__ubuf__ float*& src0, __ubuf__ float*& src1, __ubuf__ float*& src2,
                                                    __ubuf__ float*& src3, Reg::MaskReg& mask,
                                                    Reg::RegTensor<float>& value0P5, Reg::RegTensor<float>& d0,
                                                    Reg::RegTensor<float>& d1, Reg::RegTensor<float>& d2,
                                                    const int32_t postUpdateStride)
    {
        Reg::RegTensor<float> s0;
        Reg::RegTensor<float> s1;
        Reg::RegTensor<float> s2;
        Reg::RegTensor<float> s3;

        Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(s0, src0, postUpdateStride);
        Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(s1, src1, postUpdateStride);
        Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(s2, src2, postUpdateStride);
        Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(s3, src3, postUpdateStride);

        TransformVf(value0P5, s0, s1, s2, s3, d0, d1, d2, mask);
    }

    __simd_callee__ static inline void TransformRowAndCastInZero(Reg::MaskReg& mask, Reg::RegTensor<float>& value0P5,
                                                                 Reg::RegTensor<float>& d0, Reg::RegTensor<float>& d1,
                                                                 Reg::RegTensor<float>& d2, Reg::RegTensor<float>& d3,
                                                                 Reg::RegTensor<float>& out0,
                                                                 Reg::RegTensor<float>& out1,
                                                                 Reg::RegTensor<float>& out2)
    {
        if constexpr (Std::is_same_v<DstT, float>) {
            TransformVf(value0P5, d0, d1, d2, d3, out0, out1, out2, mask);
        } else {
            Reg::RegTensor<float> tmp0;
            Reg::RegTensor<float> tmp1;
            Reg::RegTensor<float> tmp2;
            TransformVf(value0P5, d0, d1, d2, d3, tmp0, tmp1, tmp2, mask);

            static_assert(sizeof(DstT) == 2);
            static constexpr Reg::CastTrait castTraitB322B16 = {
                Reg::RegLayout::ZERO,
                Reg::SatMode::NO_SAT,
                Reg::MaskMergeMode::ZEROING,
                RoundMode::CAST_RINT,
            };

            Cast<DstT, float, castTraitB322B16>(reinterpret_cast<Reg::RegTensor<DstT>&>(out0), tmp0, mask);
            Cast<DstT, float, castTraitB322B16>(reinterpret_cast<Reg::RegTensor<DstT>&>(out1), tmp1, mask);
            Cast<DstT, float, castTraitB322B16>(reinterpret_cast<Reg::RegTensor<DstT>&>(out2), tmp2, mask);
        }
    }

    __simd_callee__ static inline void TransformVf(Reg::RegTensor<float>& value0P5, Reg::RegTensor<float>& s0,
                                                   Reg::RegTensor<float>& s1, Reg::RegTensor<float>& s2,
                                                   Reg::RegTensor<float>& s3, Reg::RegTensor<float>& d0,
                                                   Reg::RegTensor<float>& d1, Reg::RegTensor<float>& d2,
                                                   Reg::MaskReg& mask)
    {
        Reg::RegTensor<float> tmpAdd;
        Reg::RegTensor<float> tmpSub;
        Reg::RegTensor<float> tmpAddHalf;

        Reg::Add(tmpAdd, s1, s2, mask);
        Reg::Sub(tmpSub, s1, s2, mask);
        Reg::Mul(tmpAddHalf, tmpAdd, value0P5, mask);
        Reg::Add(d0, s0, tmpAddHalf, mask);
        Reg::Mul(d1, tmpSub, value0P5, mask);
        Reg::Add(d2, tmpAddHalf, s3, mask);
    }

    TEventID mte32mte2_ = 0;
    TEventID v2mte3_ = 0;
    TEventID mte22v_;
    TEventID v2mte2_[2];
    uint8_t bufIdx_ = 0;
    GlobalTensor<DstT> yGm_;
    GlobalTensor<float> tailGm_;
};

#endif // CONV_BP_WINO_INV_TRANSFORM_H
