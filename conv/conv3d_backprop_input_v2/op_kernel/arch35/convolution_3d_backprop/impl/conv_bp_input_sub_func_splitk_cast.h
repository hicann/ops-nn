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
 * \file conv_bp_input_sub_func_splitk_cast.h
 * \brief SplitK/SplitDk workspace-to-UB cast chain for conv3d backprop input
 */

#ifndef CONV3D_BP_INPUT_SUB_FUNC_SPLITK_CAST_H
#define CONV3D_BP_INPUT_SUB_FUNC_SPLITK_CAST_H

#include "conv_bp_input_sub_func_utils.h"

using AscendC::GlobalTensor;
using AscendC::LocalTensor;

namespace Convolution3DBackpropFunc {

template <class Intf>
__aicore__ inline void DataCopyCastVecToOutput(Intf* self, const GlobalTensor<typename Intf::DstT>& output,
                                               uint32_t dinIdx = 0, uint32_t mOffsetInOutput = 0, uint32_t curSegM = 0)
{
    uint64_t dstOffset = 0;
    DataCopyExtParams mte3Param;
    // for Split K
    if (self->ctx.useUbAccumForSplitK_) {
        uint32_t segM = (curSegM > 0) ? curSegM : static_cast<uint32_t>(self->ctx.realMSize_);
        if constexpr (Intf::Config::dType::format == Convolution3DBackprop::CubeFormat::NCDHW) {
            // UB->GM: NCDHW
            dstOffset = static_cast<uint64_t>(self->ctx.curNIdx_) * self->ctx.tiling_->baseN *
                            self->ctx.diHiWi_ +                                                         // cin offset
                        (static_cast<uint64_t>(self->ctx.curDinStartIdx_) + dinIdx) * self->ctx.hiWi_ + // di offset
                        static_cast<uint64_t>(self->ctx.curMIdx_) * self->ctx.tiling_->baseM +
                        mOffsetInOutput; // hi&wi offset
            mte3Param.blockCount = self->ctx.baseUseN_;
            mte3Param.blockLen = segM * sizeof(typename Intf::DstT);
            mte3Param.srcStride = 0;
            mte3Param.dstStride = self->ctx.diHiWi_ * sizeof(typename Intf::DstT) - mte3Param.blockLen;
        } else if constexpr (Intf::Config::dType::format == Convolution3DBackprop::CubeFormat::NDHWC) {
            // UB->GM: NDHWC
            dstOffset = static_cast<uint64_t>(self->ctx.curNIdx_) * self->ctx.tiling_->baseN + // cin offset
                        (static_cast<uint64_t>(self->ctx.curDinStartIdx_) + dinIdx) * self->ctx.hiWi_ *
                            self->ctx.tiling_->cin + // di offset
                        (static_cast<uint64_t>(self->ctx.curMIdx_) * self->ctx.tiling_->baseM + mOffsetInOutput) *
                            self->ctx.tiling_->cin; // hi&wi offset
            mte3Param.blockCount = segM;
            mte3Param.blockLen = self->ctx.baseUseN_ * sizeof(typename Intf::DstT);
            mte3Param.srcStride = 0;
            mte3Param.dstStride = self->ctx.tiling_->cin * sizeof(typename Intf::DstT) - mte3Param.blockLen;
        }
    }
    // deprecated: for Split Dk
    if (self->ctx.enableSplitDk_) {
        dstOffset = static_cast<uint64_t>(self->ctx.curNIdx_) * self->ctx.tiling_->baseN * self->ctx.diHiWi_ +
                    static_cast<uint64_t>(self->ctx.curDinIdx_) * self->ctx.hiWi_ +
                    static_cast<uint64_t>(self->ctx.curMIdx_) * self->ctx.tiling_->baseM;
        mte3Param.blockCount = self->ctx.baseUseN_;
        mte3Param.blockLen = self->ctx.baseUseM_ * sizeof(typename Intf::DstT);
        mte3Param.srcStride = 0;
        mte3Param.dstStride = self->ctx.diHiWi_ * sizeof(typename Intf::DstT) - mte3Param.blockLen;
    }

#if !__FIXED_POINT_ONLY_CUBE_TO_L0C__
    if constexpr (std::is_same<typename Intf::L0cT, int32_t>::value) {
        DataCopyPad<typename Intf::DstT, PaddingMode::Compact>(
            output[dstOffset], self->ctx.castVecTensor_.template ReinterpretCast<typename Intf::DstT>(), mte3Param);
    } else {
        DataCopyPad<typename Intf::DstT, PaddingMode::Compact>(
            output[dstOffset], self->ctx.castVecTensor_.template ReinterpretCast<typename Intf::SrcT>(), mte3Param);
    }
#endif
}

template <class Intf>
__aicore__ inline void CastToDstType(Intf* self, const GlobalTensor<typename Intf::DstT>& output, uint8_t enAtomic = 0,
                                     bool enSequentialWrite = false)
{
    if ASCEND_IS_AIC_SHOULD_RETURN {
        return;
    }
    if (GetSubBlockIdx() > 0) {
        return;
    }
    // 单核切Dk时Iterate接口里DinIdx会超出范围，跳过
    if (self->ctx.curDinIdx_ >= self->ctx.curDinStartIdx_ + self->ctx.singleShapeDin_) {
        return;
    }
    if (!enSequentialWrite) {
        // workspace格式: D singleCoreCin/baseN singleCoreM/baseM baseN baseM
        uint64_t singleCoreCinAlignBaseN = AlignUp(self->ctx.tiling_->singleCoreCin, self->ctx.tiling_->baseN);
        uint64_t singleCoreMAlignBaseM = AlignUp(self->ctx.tiling_->singleCoreM, self->ctx.tiling_->baseM);
        uint64_t singleCoreWorkspaceSize = singleCoreCinAlignBaseN * singleCoreMAlignBaseM *
                                           self->ctx.tiling_->singleCoreDin;
        int64_t srcOffset = (self->ctx.curDinIdx_ - self->ctx.curDinStartIdx_) * singleCoreCinAlignBaseN *
                                singleCoreMAlignBaseM +
                            self->ctx.curNIdx_ * self->ctx.tiling_->baseN * singleCoreMAlignBaseM +
                            self->ctx.curMIdx_ * self->ctx.tiling_->baseN * self->ctx.tiling_->baseM +
                            GetAicBlockIdx() * singleCoreWorkspaceSize;
        self->ctx.castVecTensor_ = self->ctx.vecBuf_.template Get<float>();
        DataCopyExtParams mte2Param;
        mte2Param.blockCount = 1;
        mte2Param.blockLen = self->ctx.baseUseM_ * self->ctx.baseUseN_ * sizeof(float);
        mte2Param.srcStride = 0;
        mte2Param.dstStride = 0;
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad<float, PaddingMode::Compact>(self->ctx.castVecTensor_, self->ctx.l0cOutGm_[srcOffset], mte2Param,
                                                 padParams);

        event_t eventIdMte2ToVec = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToVec);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToVec);

        // fp32 cast to Dst Type
        Cast(self->ctx.castVecTensor_.template ReinterpretCast<typename Intf::SrcT>(), self->ctx.castVecTensor_,
             RoundMode::CAST_RINT, self->ctx.baseUseM_ * self->ctx.baseUseN_);

        event_t eventIdVecToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVecToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVecToMte3);
        DataCopyCastVecToOutput(self, output);
    }
}

template <class Intf>
__aicore__ inline void MoveFromUsrSpaceToUbCast(Intf* self, const GlobalTensor<typename Intf::DstT>& output)
{
    // 计算workspace中当前核切片的偏移
    // workspace格式: [Din][singleCoreCinAlignBaseN][singleCoreMAlignBaseM]
    uint64_t singleCoreCinAlignBaseN = AlignUp(self->ctx.tiling_->singleCoreCin, self->ctx.tiling_->baseN);
    uint64_t singleCoreMAlignBaseM = AlignUp(self->ctx.tiling_->singleCoreM, self->ctx.tiling_->baseM);
    uint64_t singleCoreWorkspaceSize = singleCoreCinAlignBaseN * singleCoreMAlignBaseM *
                                       self->ctx.tiling_->singleCoreDin;
    uint64_t coreOffset = GetAicBlockIdx() * singleCoreWorkspaceSize;
    uint64_t dinStride = singleCoreCinAlignBaseN * singleCoreMAlignBaseM;

    // 获取UB cast缓冲区
    self->ctx.castVecTensor_ = self->ctx.vecBuf_.template Get<float>();

    uint32_t totalM = static_cast<uint32_t>(self->ctx.realMSize_);
    uint32_t baseM = self->ctx.tiling_->baseM;

    event_t eventIdMte2ToVec = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    event_t eventIdVecToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

    bool isFirstSeg = true;
    // note: singleCoreDin在inner product tiling中设置为1，为后续拓展din维度保留din循环
    for (uint32_t dinIdx = 0; dinIdx < self->ctx.singleShapeDin_; dinIdx++) {
        uint64_t dinSrcOffset = coreOffset + static_cast<uint64_t>(dinIdx) * dinStride;
        for (uint32_t mSegIdx = 0; mSegIdx < totalM; mSegIdx += baseM) {
            uint32_t curSegM = (totalM - mSegIdx < baseM) ? (totalM - mSegIdx) : baseM;
            uint64_t srcOffset = dinSrcOffset + static_cast<uint64_t>(mSegIdx) * self->ctx.baseUseN_;

            if (!isFirstSeg) {
                WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            }

            // workspace → UB
            DataCopyExtParams mte2Param;
            mte2Param.blockCount = 1;
            mte2Param.blockLen = curSegM * self->ctx.baseUseN_ * sizeof(float);
            mte2Param.srcStride = 0;
            mte2Param.dstStride = 0;
            DataCopyPad<float, PaddingMode::Compact>(self->ctx.castVecTensor_, self->ctx.l0cOutGm_[srcOffset],
                                                     mte2Param, padParams);

            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToVec);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToVec);

            // fp32 → DstT cast
            Cast(self->ctx.castVecTensor_.template ReinterpretCast<typename Intf::SrcT>(), self->ctx.castVecTensor_,
                 std::is_same<typename Intf::DstT, hifloat8_t>::value ? RoundMode::CAST_ROUND : RoundMode::CAST_RINT,
                 curSegM * self->ctx.baseUseN_);

            SetFlag<HardEvent::V_MTE3>(eventIdVecToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVecToMte3);

            // UB → output
            DataCopyCastVecToOutput<Intf>(self, output, dinIdx, mSegIdx, curSegM);

            bool isLastSeg = (dinIdx + 1 == self->ctx.singleShapeDin_) && (mSegIdx + curSegM >= totalM);
            if (!isLastSeg) {
                SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            }
            isFirstSeg = false;
        }
    }
}

template <class Intf>
__aicore__ inline void AccumulateSegmentOnWorkspace(Intf* self, const GlobalTensor<typename Intf::DstT>& output,
                                                    bool enSequentialWrite = false)
{
    if ASCEND_IS_AIC_SHOULD_RETURN {
        return;
    }

    if (GetSubBlockIdx() > 0) {
        return;
    }

    if (enSequentialWrite) {
        return;
    }

    MoveFromUsrSpaceToUbCast<Intf>(self, output);
}

} // namespace Convolution3DBackpropFunc

#endif
