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
 * \file conv3d_backprop_filter_v2.h
 * \brief
 */
#ifndef CONV3D_BACKPROP_FILTER_H
#define CONV3D_BACKPROP_FILTER_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "conv3d_bp_filter.h"
#include "kernel_type.h"
#include "lib/matmul_intf.h"
#include "conv3d_backprop_filter_v2_tiling_data.h"
namespace AscendC {

__aicore__ inline constexpr ConvolutionBackprop::CubeFormat GetFormat(int format)
{
    if (format == FORMAT_NDC1HWC0) {
        return ConvolutionBackprop::CubeFormat::NDC1HWC0;
    }
    return ConvolutionBackprop::CubeFormat::NCDHW;
}

template <typename xType, int xFormat, typename dedyType, int dedyFormat, typename yType, int yFormat>
class Conv3dDw {
public:
    __aicore__ inline Conv3dDw(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dedy, GM_ADDR y, GM_ADDR workSpace,
                                const Conv3DBackpropFilterV2TilingData* tilingData)
    {
        InitTilingData(tilingData);
        // init global buffer
        xGm_.SetGlobalBuffer((__gm__ xType*)x);
        dedyGm_.SetGlobalBuffer((__gm__ dedyType*)dedy);
        yGm_.SetGlobalBuffer((__gm__ yType*)y);
        dw_.Init(&(tilingData->dwTiling));
#if defined(DETERMINISTIC_MODE) && DETERMINISTIC_MODE == 1
        workspaceGm_.SetGlobalBuffer((__gm__ yType*)workSpace);
        CalMaxIterate();
#endif
    }

    /** main logical function
     */
    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            return;
        }

        if (block_idx >= GetBlockNum()) {
            dw_.End();
            return;
        }

        CalSingleCoreShape();
        for (uint32_t i = 0; i < singleShapeGroup_; ++i) {
            uint32_t groupIdx = groupCoreIndx_ * singleCoreGroup_ + i;
            for (uint64_t j = 0; j < singleShapeBatch_; ++j) {
                uint64_t batchDoutIdx = batchCoreIndx_ * singleCoreBatch_ + j;
                uint64_t batchIdx = batchDoutIdx / dout_;
                uint64_t doutIdx = batchDoutIdx - batchIdx * dout_;
                for (uint32_t t = 0; t < singleShapeDk_; ++t) {
                    uint32_t dkIdx = dkCoreIndx_ * singleCoreDk_ + t;
                    ReCalDkCinSingleCoreShape(batchIdx, doutIdx, groupIdx, dkIdx);
                    if (singleShapeNInCurrentHo_ == 0 || singleShapeMInCurrentHo_ == 0) {
                        continue;
                    }
                    CalcOffset(batchIdx, doutIdx, groupIdx, dkIdx);
                    dw_.SetFmap(xGm_[offsetB_]);
                    dw_.SetOutBackprop(dedyGm_[offsetA_]);
                    dw_.SetSingleShape(singleShapeMInCurrentHo_, singleShapeNInCurrentHo_, singleShapeK_, hoStartIdx_);
                    dw_.IterateAll(yGm_[offsetC_], 1); // 1 means atomic add.
                }
            }
        }
        dw_.End();
    }

#if defined(DETERMINISTIC_MODE) && DETERMINISTIC_MODE == 1
    __aicore__ inline void ProcessWithDeterministic()
    {
        if (block_idx >= GetBlockNum()) {
            return;
        }
        CalSingleCoreShape();
        InitMixCoreBuffer(); // initialize UB buffer for deterministic adding
        for (uint32_t i = 0; i < singleShapeGroup_; ++i) {
            uint32_t groupIdx = groupCoreIndx_ * singleCoreGroup_ + i;
            for (uint64_t j = 0; j < singleCoreBatch_; ++j) {
                uint64_t batchDoutIdx = batchCoreIndx_ * singleCoreBatch_ + j;
                uint64_t batchIdx = batchDoutIdx / dout_;
                uint64_t doutIdx = batchDoutIdx - batchIdx * dout_;
                for (uint32_t t = 0; t < singleShapeDk_; ++t) {
                    uint32_t dkIdx = dkCoreIndx_ * singleCoreDk_ + t;
                    /* 由于确定性计算要求每个核Reduce轴的计算次序都相同，尾块不能提前进入下一个循环，因此
                     * 使用singleCoreBatch进行循环，若尾块超出了有效数据范围，则直接设置
                     * singleShapeNInCurrentHo_和singleShapeMInCurrentHo_为零，不在IterateAll中计算 */
                    if (j < singleShapeBatch_) {
                        ReCalDkCinSingleCoreShape(batchIdx, doutIdx, groupIdx, dkIdx);
                    } else {
                        singleShapeNInCurrentHo_ = 0;
                        singleShapeMInCurrentHo_ = 0;
                    }

                    /* 在确定性计算当中，即使当前核没有有效数据进行计算，
                     *仍然需要进行workspace置零操作和同步操作，因此不能够直接跳过计算 */
                    if (singleShapeNInCurrentHo_ == 0 || singleShapeMInCurrentHo_ == 0) {
                        // 跳过场景只需要用到C矩阵运算, 只传入group和dk偏移即可
                        CalcOffset(0, 0, groupIdx, dkIdx);
                        dw_.SetSingleShape(singleShapeM_, singleShapeN_, singleShapeK_, hoStartIdx_);
                    } else {
                        /* 当且仅当当前核有有效数据进行计算时，需要计算Gm地址中的offset并进行设置 */
                        CalcOffset(batchIdx, doutIdx, groupIdx, dkIdx);
                        dw_.SetFmap(xGm_[offsetB_]);
                        dw_.SetOutBackprop(dedyGm_[offsetA_]);
                        dw_.SetSingleShape(singleShapeMInCurrentHo_, singleShapeNInCurrentHo_, singleShapeK_,
                                           hoStartIdx_);
                    }
                    // Involving VectorCore
                    DeterministicIterateAll();
                }
            }
        }
        ReachMaxIterate();
        dw_.End();
    }
#endif

protected:
    static constexpr ConvolutionBackprop::CubeFormat xCubeFormat = GetFormat(xFormat);
    static constexpr ConvolutionBackprop::CubeFormat dedyCubeFormat = ConvolutionBackprop::CubeFormat::NDC1HWC0;
    static constexpr ConvolutionBackprop::CubeFormat yCubeFormat = ConvolutionBackprop::CubeFormat::FRACTAL_Z_3D;
    static constexpr uint64_t DOUBLE_BUFFER = 2;
    static constexpr uint64_t SYNC_MODE2 = 2;
    static constexpr uint64_t SYNC_AIV_AIC_FLAG = 6;
    static constexpr uint64_t SYNC_AIC_AIV_FLAG = 8;
#if defined(DETERMINISTIC_MODE) && DETERMINISTIC_MODE == 1
    static constexpr uint64_t UB_SIZE = 196608;
    static constexpr uint64_t SYNC_MODE0 = 0;
    static constexpr uint64_t SYNC_AIV_ONLY_ALL_FLAG = 0;
    static constexpr uint64_t SYNC_AIC_ONLY_ALL_FLAG = 4;
#endif
    using xDwType = ConvolutionBackprop::ConvType<TPosition::GM, xCubeFormat, xType>;
    using filterSizeDwType = ConvolutionBackprop::ConvType<TPosition::GM, ConvolutionBackprop::CubeFormat::ND, int32_t>;
    using dedyDwType = ConvolutionBackprop::ConvType<TPosition::GM, dedyCubeFormat, dedyType>;
    using yDwType = ConvolutionBackprop::ConvType<TPosition::GM, yCubeFormat, yType>;
    ConvolutionBackprop::Conv3DBackpropFilter<xDwType, filterSizeDwType, dedyDwType, yDwType> dw_;
    GlobalTensor<xType> xGm_;
    GlobalTensor<xType> dedyGm_;
    GlobalTensor<yType> yGm_;
    GlobalTensor<xType> workspaceXGm_;
    uint64_t offsetA_;
    uint64_t offsetB_;
    uint64_t offsetC_;
    uint32_t batchDim_;
    uint32_t kDim_;
    uint32_t mDim_;
    uint32_t nDim_;
    uint32_t groupDim_;
    uint32_t batchCoreIndx_;
    uint32_t mCoreIndx_;
    uint32_t nCoreIndx_;
    uint32_t kCoreIndx_;
    uint32_t cout1G_;
    uint32_t ho_;
    uint32_t wo_;
    uint32_t hi_;
    uint32_t wi_;
    uint32_t hk_;
    uint32_t wk_;
    uint32_t strideH_;
    uint32_t strideD_;
    uint32_t padUp_;
    uint32_t padFront_;
    uint32_t padBack_;
    uint32_t dilationH_;
    uint32_t dout_;
    uint32_t di_;
    uint32_t dk_;
    uint32_t hoStartIdx_;
    uint32_t batch_;
    uint32_t cin_;
    uint32_t c0_;
    uint64_t m_;
    uint64_t n_;
    uint64_t k_;
    uint64_t singleCoreBatch_;
    uint64_t singleCoreCout_;
    uint64_t singleCoreCin_;
    uint64_t singleCoreHo_;
    uint64_t singleShapeBatch_;
    uint64_t singleShapeM_;
    uint64_t singleShapeN_;
    uint64_t singleShapeNInCurrentHo_;
    uint64_t singleShapeK_;
    uint64_t coutG_;
    uint64_t cinG_;
    uint32_t dkDim_;
    uint32_t groupCoreIndx_;
    uint32_t dkCoreIndx_;
    uint32_t group_;
    uint32_t singleCoreGroup_;
    uint32_t singleCoreDk_;
    uint32_t singleShapeGroup_;
    uint32_t singleShapeDk_;
    uint64_t singleShapeMInCurrentHo_;
    uint32_t tailCinG_;
    uint32_t tailCoutG_;
    uint32_t dilationD_;
    bool seperateDk_;
#if defined(DETERMINISTIC_MODE) && DETERMINISTIC_MODE == 1
    GlobalTensor<yType> workspaceGm_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    uint64_t offsetWorkspaceC_;
    uint32_t maxIterate_;
    uint64_t singleSize_;
    uint64_t dataSize_;
    bool gmPingPongEventId_ = 0;
    uint64_t syncTimes_ = 0;
#endif

    __aicore__ inline void InitTilingData(const Conv3DBackpropFilterV2TilingData* tilingData)
    {
        batchDim_ = tilingData->params.batchDim;
        mDim_ = tilingData->params.mDim;
        nDim_ = tilingData->params.nDim;
        kDim_ = tilingData->params.kDim;
        groupDim_ = tilingData->params.groupDim;
        batch_ = tilingData->dwTiling.batch;
        dk_ = tilingData->dwTiling.dk;
        cout1G_ = tilingData->dwTiling.cout1G;
        c0_ = tilingData->dwTiling.k0;
        cin_ = tilingData->dwTiling.cin;
        coutG_ = static_cast<uint64_t>(cout1G_) * c0_;
        cinG_ = static_cast<uint64_t>(tilingData->dwTiling.cin1G) * c0_;
        k_ = static_cast<uint64_t>(tilingData->dwTiling.ho) * tilingData->dwTiling.wo;
        singleCoreBatch_ = tilingData->dwTiling.singleCoreBatch;
        singleCoreCout_ = tilingData->dwTiling.singleCoreCout;
        singleCoreCin_ = tilingData->dwTiling.singleCoreCin;
        singleCoreHo_ = tilingData->dwTiling.singleCoreHo;
        ho_ = tilingData->dwTiling.ho;
        wo_ = tilingData->dwTiling.wo;
        hi_ = tilingData->dwTiling.hi;
        wi_ = tilingData->dwTiling.wi;
        hk_ = tilingData->dwTiling.hk;
        wk_ = tilingData->dwTiling.wk;
        strideH_ = tilingData->dwTiling.strideH;
        padUp_ = tilingData->dwTiling.padUp;
        padFront_ = tilingData->dwTiling.padFront;
        padBack_ = tilingData->dwTiling.padBack;
        strideD_ = tilingData->dwTiling.strideD;
        dout_ = tilingData->dwTiling.dout;
        di_ = tilingData->dwTiling.di;
        hoStartIdx_ = 0;
        dkDim_ = tilingData->params.dkDim;
        group_ = tilingData->dwTiling.group;
        singleCoreGroup_ = tilingData->dwTiling.singleCoreGroup;
        singleCoreDk_ = tilingData->dwTiling.singleCoreDk;
        dilationD_ = tilingData->dwTiling.dilationD;
        dilationH_ = tilingData->dwTiling.dilationH;
        seperateDk_ = group_ > 1 || dilationD_ > 1;
#if defined(DETERMINISTIC_MODE) && DETERMINISTIC_MODE == 1
        seperateDk_ = true;
        singleSize_ = tilingData->dwTiling.baseM * tilingData->dwTiling.baseN;
        dataSize_ = UB_SIZE / 2 / sizeof(yType); // 2: divide ub into 2 equal-size part for adding
#endif
        if (seperateDk_) {
            m_ = coutG_;
            n_ = cinG_ * tilingData->dwTiling.hk * tilingData->dwTiling.wk;
            tailCinG_ = DivCeil(tilingData->dwTiling.cin, c0_) * c0_ - cinG_ * (group_ - 1);
            tailCoutG_ = DivCeil(tilingData->dwTiling.cout, c0_) * c0_ - coutG_ * (group_ - 1);
        } else {
            m_ = DivCeil(tilingData->dwTiling.cout, c0_) * c0_;
            n_ = dk_ * cinG_ * tilingData->dwTiling.hk * tilingData->dwTiling.wk;
            tailCinG_ = cinG_;
            tailCoutG_ = coutG_;
        }
    }

    __aicore__ inline void CalcOffset(uint64_t batchIdx, uint64_t doutIdx, uint32_t groupIdx, uint32_t dkIdx,
                                      bool isBasicBlock = false)
    {
        if (seperateDk_) {
            CalcOffsetWithGroup(batchIdx, doutIdx, groupIdx, dkIdx);
#if defined(DETERMINISTIC_MODE) && DETERMINISTIC_MODE == 1
            offsetWorkspaceC_ = DOUBLE_BUFFER * block_idx * singleSize_;
#endif
            return;
        }

        uint64_t hoCal = isBasicBlock ? (kCoreIndx_ * singleShapeK_) / wo_ : kCoreIndx_ * singleCoreHo_;
        uint64_t hoOffset = hoCal * wo_ * c0_;
        uint64_t coutOffset = mCoreIndx_ * singleCoreCout_ * ho_ * wo_;
        uint64_t batchOffsetA = (batchIdx * dout_ + doutIdx) * m_ * ho_ * wo_;
        offsetA_ = batchOffsetA + coutOffset + hoOffset;

        uint64_t dinIdx = 0;
        if (doutIdx * strideD_ > padFront_) {
            dinIdx = doutIdx * strideD_ - padFront_;
        }

        uint64_t hiIdx = 0;
        if (hoCal * strideH_ > padUp_) {
            hiIdx = hoCal * strideH_ - padUp_;
        }
        uint64_t hiOffset = hiIdx * wi_ * c0_;
        uint64_t cinOffset = nCoreIndx_ * singleCoreCin_ * hi_ * wi_;
        uint64_t batchOffsetB = (batchIdx * di_ + dinIdx) * cinG_ * hi_ * wi_;
        offsetB_ = batchOffsetB + cinOffset + hiOffset;
        offsetC_ = nCoreIndx_ * singleCoreCin_ * hk_ * wk_ * coutG_ + mCoreIndx_ * singleCoreCout_ * c0_;
        if (doutIdx * strideD_ < padFront_) {
            offsetC_ += (padFront_ - doutIdx * strideD_) * cinG_ * hk_ * wk_ * coutG_;
        }
    }

    __aicore__ inline void CalSingleCoreShape()
    {
        kCoreIndx_ = block_idx % kDim_;
        nCoreIndx_ = (block_idx / kDim_) % nDim_;
        mCoreIndx_ = (block_idx / (kDim_ * nDim_)) % mDim_;
        batchCoreIndx_ = (block_idx / (kDim_ * nDim_ * mDim_)) % batchDim_;
        uint64_t batchRamin = static_cast<uint64_t>(batch_) * dout_ - batchCoreIndx_ * singleCoreBatch_;
        uint64_t mRamin = m_ - mCoreIndx_ * singleCoreCout_;
        uint64_t nRamin = n_ - nCoreIndx_ * singleCoreCin_ * hk_ * wk_;
        uint64_t kRamin = k_ - kCoreIndx_ * singleCoreHo_ * wo_;
        singleShapeBatch_ = batchRamin < singleCoreBatch_ ? batchRamin : singleCoreBatch_;
        singleShapeM_ = mRamin < singleCoreCout_ ? mRamin : singleCoreCout_;
        singleShapeN_ = nRamin < singleCoreCin_ * hk_ * wk_ ? nRamin : singleCoreCin_ * hk_ * wk_;
        singleShapeK_ = kRamin < singleCoreHo_ * wo_ ? kRamin : singleCoreHo_ * wo_;
        hoStartIdx_ = kCoreIndx_ * singleCoreHo_;

        if (seperateDk_) {
            uint64_t totalDim = static_cast<uint64_t>(batchDim_) * kDim_ * nDim_ * mDim_;
            dkCoreIndx_ = (block_idx / totalDim) % dkDim_;
            groupCoreIndx_ = (block_idx / (totalDim * dkDim_)) % groupDim_;
            uint32_t groupRamin = group_ - groupCoreIndx_ * singleCoreGroup_;
            uint32_t dkRamin = dk_ - dkCoreIndx_ * singleCoreDk_;
            singleShapeGroup_ = groupRamin < singleCoreGroup_ ? groupRamin : singleCoreGroup_;
            singleShapeDk_ = dkRamin < singleCoreDk_ ? dkRamin : singleCoreDk_;
        } else {
            dkCoreIndx_ = 0;
            groupCoreIndx_ = 0;
            singleShapeGroup_ = 1;
            singleShapeDk_ = 1;
        }
    }

    __aicore__ inline void ReCalDkCinSingleCoreShape(uint64_t batchIdx, uint64_t doutIdx, uint32_t groupIdx,
                                                     uint32_t dkIdx)
    {
        singleShapeNInCurrentHo_ = singleShapeN_;
        singleShapeMInCurrentHo_ = singleShapeM_;
        if (seperateDk_) {
            ReCalDkGroupSingleCoreShape(batchIdx, doutIdx, groupIdx, dkIdx);
            return;
        }

        uint64_t dinIdx = 0;
        if (doutIdx * strideD_ > padFront_) {
            dinIdx = doutIdx * strideD_ - padFront_;
        }
        uint64_t newDk = dk_;
        if (doutIdx * strideD_ + dk_ > padFront_ + di_) {
            if (doutIdx * strideD_ < padFront_ + di_) {
                newDk = padFront_ + di_ - doutIdx * strideD_;
                if (n_ / dk_ * newDk < nCoreIndx_ * singleCoreCin_ * hk_ * wk_) {
                    singleShapeNInCurrentHo_ = 0;
                    return;
                }
                uint64_t nRamin = n_ / dk_ * newDk - nCoreIndx_ * singleCoreCin_ * hk_ * wk_;
                singleShapeNInCurrentHo_ = nRamin < singleCoreCin_ * hk_ * wk_ ? nRamin : singleCoreCin_ * hk_ * wk_;
            } else {
                singleShapeNInCurrentHo_ = 0;
            }
        }

        if (doutIdx * strideD_ < padFront_) {
            if (doutIdx * strideD_ + newDk > padFront_) {
                newDk = doutIdx * strideD_ + newDk - padFront_;
                if (n_ / dk_ * newDk < nCoreIndx_ * singleCoreCin_ * hk_ * wk_) {
                    singleShapeNInCurrentHo_ = 0;
                    return;
                }
                uint64_t nRamin = n_ / dk_ * newDk - nCoreIndx_ * singleCoreCin_ * hk_ * wk_;
                singleShapeNInCurrentHo_ = nRamin < singleCoreCin_ * hk_ * wk_ ? nRamin : singleCoreCin_ * hk_ * wk_;
            } else {
                singleShapeNInCurrentHo_ = 0;
            }
        }
    }

    __aicore__ inline void CalcOffsetWithGroup(uint64_t batchIdx, uint64_t doutIdx, uint32_t groupIdx, uint32_t dkIdx)
    {
        uint32_t alignedCin = (group_ - 1) * cinG_ + tailCinG_;
        uint32_t alignedCout = (group_ - 1) * coutG_ + tailCoutG_;
        uint64_t hwO = static_cast<uint64_t>(ho_) * wo_;
        uint64_t hwI = static_cast<uint64_t>(hi_) * wi_;

        uint64_t hoOffset = kCoreIndx_ * singleCoreHo_ * static_cast<uint64_t>(wo_) * c0_;
        uint64_t coutOffset = mCoreIndx_ * singleCoreCout_ * hwO;
        uint64_t groupOffsetA = groupIdx * coutG_ * hwO;
        uint64_t batchOffsetA = (batchIdx * dout_ + doutIdx) * alignedCout * hwO;
        offsetA_ = batchOffsetA + groupOffsetA + coutOffset + hoOffset;

        // 小于0的场景已经被跳过
        uint64_t dinIdx = doutIdx * strideD_ - padFront_ + dkIdx * dilationD_;

        uint64_t hiIdx = 0;
        if (kCoreIndx_ * singleCoreHo_ * strideH_ > padUp_) {
            hiIdx = kCoreIndx_ * singleCoreHo_ * strideH_ - padUp_;
        }
        uint64_t hiOffset = hiIdx * static_cast<uint64_t>(wi_) * c0_;
        uint64_t cinOffset = nCoreIndx_ * singleCoreCin_ * hwI;
        uint64_t groupOffsetB = groupIdx * cinG_ * hwI;
        uint64_t depthOffsetB = dinIdx * alignedCin * hwI;
        uint64_t batchOffsetB = batchIdx * di_ * alignedCin * hwI;
        offsetB_ = batchOffsetB + depthOffsetB + groupOffsetB + cinOffset + hiOffset;

        uint64_t coutK = static_cast<uint64_t>(coutG_) * hk_ * wk_;
        uint64_t cinCoutK = cinG_ * coutK;
        uint64_t groupOffsetC = groupIdx * dk_ * cinCoutK;
        uint64_t dkOffsetC = dkIdx * cinCoutK;
        uint64_t cinCoutOffsetC = nCoreIndx_ * singleCoreCin_ * coutK + mCoreIndx_ * singleCoreCout_ * c0_;
        offsetC_ = groupOffsetC + dkOffsetC + cinCoutOffsetC;
    }

    __aicore__ inline void ReCalDkGroupSingleCoreShape(uint64_t batchIdx, uint64_t doutIdx, uint32_t groupIdx,
                                                       uint32_t dkIdx)
    {
        int64_t dinIdx = doutIdx * strideD_ - padFront_ + dkIdx * dilationD_;
        if (dinIdx < 0 || dinIdx >= di_) {
            singleShapeNInCurrentHo_ = 0;
            singleShapeMInCurrentHo_ = 0;
            return;
        }

        if (groupIdx == (group_ - 1)) {
            if (tailCinG_ < cinG_) {
                uint64_t tailN = tailCinG_ * hk_ * wk_;
                uint64_t singleCoreN = singleCoreCin_ * hk_ * wk_;
                uint64_t currentN = nCoreIndx_ * singleCoreN;
                if (tailN <= currentN) {
                    singleShapeNInCurrentHo_ = 0;
                    return;
                } else {
                    uint64_t nRemain = tailN - currentN;
                    singleShapeNInCurrentHo_ = nRemain < singleCoreN ? nRemain : singleCoreN;
                }
            }

            if (tailCoutG_ < coutG_) {
                uint32_t tailM = tailCoutG_;
                uint32_t currentM = mCoreIndx_ * singleCoreCout_;
                if (tailM <= currentM) {
                    singleShapeMInCurrentHo_ = 0;
                    return;
                } else {
                    uint32_t mRemain = tailM - currentM;
                    singleShapeMInCurrentHo_ = mRemain < singleCoreCout_ ? mRemain : singleCoreCout_;
                }
            }
        }
    }

#if defined(DETERMINISTIC_MODE) && DETERMINISTIC_MODE == 1
#include "conv3d_backprop_filter_v2_deterministic.h"
#endif // DETERMINISTIC_MODE
};
} // namespace AscendC

#endif // CONV3D_BACKPROP_FILTER_H
