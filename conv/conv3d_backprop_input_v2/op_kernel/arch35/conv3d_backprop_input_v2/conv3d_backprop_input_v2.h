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
 * \file conv3d_backprop_input_v2.h
 * \brief
 */
#ifndef CONV3D_BACKPROP_INPUT_V2_ADVANCE_H
#define CONV3D_BACKPROP_INPUT_V2_ADVANCE_H

#include "conv3d_bp_input.h"
#include "basic_api/kernel_basic_intf.h"
#include "kernel_type.h"
#include "lib/matmul_intf.h"
#include "conv3d_backprop_input_v2_tiling_data.h"
#include "../../conv3d_backprop_input_v2_arch35_tiling_key.h"
#ifndef DTYPE_BIAS
#define DTYPE_BIAS float
#define FORMAT_BIAS FORMAT_MAX  // FORMAT_MAX意为数据格式不支持，用以表达不带bias输入场景
#endif

namespace AscendC {

using Conv3dConfig = typename Convolution3DBackprop::Conv3dConfig;

__aicore__ inline constexpr Convolution3DBackprop::CubeFormat GetFormat(int format)
{
    if (format == FORMAT_MAX) {
        return Convolution3DBackprop::CubeFormat::UNSUPPORT;
    } else if (format == FORMAT_ND) {
        return Convolution3DBackprop::CubeFormat::ND;
    } else if (format == FORMAT_NDHWC || format == FORMAT_NHWC) {
        return Convolution3DBackprop::CubeFormat::NDHWC;
    } else if (format == FORMAT_DHWCN || format == FORMAT_HWCN) {
        return Convolution3DBackprop::CubeFormat::DHWCN;
    } else {
        return Convolution3DBackprop::CubeFormat::NCDHW;
    }
}

template <typename filterType, int filterFormat, typename dedyType, int dedyFormat, typename yType, int yFormat,
        typename biasType, int biasFormat,
        uint8_t b2Condition, uint8_t kernelSplitMode, uint8_t groupMode,
        uint8_t b1Condition = TPL_GM_TO_L1,
        bool enableC04Flag = false>
class Conv3dDx {
public:
    __aicore__ inline Conv3dDx(){};
    __aicore__ inline void Init(GM_ADDR filter, GM_ADDR dedy, GM_ADDR y, GM_ADDR workSpace,
                                const conv_bp_v2_kernel::Conv3DBackpropInputV2TilingData *tilingData, GM_ADDR bias = nullptr)
    {
        InitTilingData(tilingData);
        enableSplitDk_ = tiling_->singleIterateDk != tiling_->dk;
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
        if constexpr ((kernelSplitMode != TPL_SPLIT_KERNEL_HW) &&
            groupMode == TPL_GROUP_MODE_ORIGIN) {
            if (!enableSplitDk_) {
                if ASCEND_IS_AIV {
                    return;
                }
            }
        }
#else
        if ASCEND_IS_AIV {
            return;
        }
#endif
        // init global buffer
        filterGm_.SetGlobalBuffer((__gm__ filterType *)filter);
        dedyGm_.SetGlobalBuffer((__gm__ dedyType *)dedy);
        yGm_.SetGlobalBuffer((__gm__ yType *)y);
        dedx_.Init(&(tilingData->conv3DDxTiling));
#if defined(__DAV_310R6__)
        if constexpr (biasFormat != FORMAT_MAX) {
            biasGm_.SetGlobalBuffer((__gm__ biasType *)bias);
        }
#endif

#if defined(__DAV_C310__) || defined(__DAV_310R6__)
        InitMixCoreBuffer(workSpace);
#endif
    }

    /** main logical function
     */
    __aicore__ inline void Process()
    {
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
        if constexpr ((kernelSplitMode != TPL_SPLIT_KERNEL_HW) &&
            groupMode == TPL_GROUP_MODE_ORIGIN) {
            if (!enableSplitDk_) {
                if ASCEND_IS_AIV {
                    return;
                }
            }
        }
#else
        if ASCEND_IS_AIV {
            return;
        }
#endif
        CalSingleCoreShape();
        if (curHoStartIdx_ >= static_cast<int32_t>((tiling_->ho - 1) * tiling_->strideH + 1) || singleShapeM_ == 0) {
            dedx_.End();
            return;
        }

        InitBlockStride();
        CalcBlockOffset(batchCoreIdx_ * tiling_->singleCoreBatch, groupCoreIdx_ * tiling_->singleCoreGroup);
        if (likely(tiling_->group == 1)) {
            for (uint32_t batchIdx = 0; batchIdx < singleShapeBatch_; ++batchIdx) {
                dedx_.SetOutBackprop(dedyGm_[offsetA_]);
                dedx_.SetWeight(filterGm_[offsetB_]);
                dedx_.SetFullLoadFlag(this->tiling_->enableFullLoad);
#if defined(__DAV_310R6__)
                if constexpr (biasFormat != FORMAT_MAX) {
                    dedx_.SetBias(biasGm_[offsetBias_]);
                }
#endif
                dedx_.IterateAll(yGm_[offsetC_], 0);  // 1 means atomic add
                CalcBatchOffset();
            }
        } else {
            ProcessGroup();
        }

        dedx_.End();
    }

    __aicore__ inline void ProcessGroup()
    {
        for (uint32_t batchIdx = 0; batchIdx < singleShapeBatch_; ++batchIdx) {
            uint64_t backupOffsetA = offsetA_;
            uint64_t backupOffsetB = offsetB_;
            uint64_t backupOffsetC = offsetC_;
            for (uint32_t groupIdx = 0; groupIdx < singleShapeGroup_; ++groupIdx) {
                CalSingleCoreShapeInGroup(groupIdx);
                if (singleShapeKInGroup_ == 0 || singleShapeNInGroup_ == 0) {
                    continue;
                }
                dedx_.SetOutBackprop(dedyGm_[offsetA_]);
                dedx_.SetWeight(filterGm_[offsetB_]);
                dedx_.SetFullLoadFlag(this->tiling_->enableFullLoad);
#if defined(__DAV_310R6__)
                if constexpr (biasFormat != FORMAT_MAX) {
                    offsetBias_ = nCoreIdx_ * tiling_->singleCoreCin + groupIdx * tiling_->cinG;
                    dedx_.SetBias(biasGm_[offsetBias_]);
                }
#endif
                dedx_.IterateAll(yGm_[offsetC_], 0);  // 1 means atomic add
                CalcGroupOffset();
            }
            CalcBatchOffset(backupOffsetA, backupOffsetB, backupOffsetC);
        }
    }

protected:
    static constexpr Convolution3DBackprop::CubeFormat filterCubeFormat = GetFormat(filterFormat);
    static constexpr Convolution3DBackprop::CubeFormat dedyCubeFormat = GetFormat(dedyFormat);
    static constexpr Convolution3DBackprop::CubeFormat yCubeFormat = GetFormat(yFormat);
    static constexpr Convolution3DBackprop::CubeFormat biasCubeFormat = GetFormat(biasFormat);
    using filterDxType = Convolution3DBackprop::ConvType<TPosition::GM, filterCubeFormat, filterType>;
    using inputSizeDxType =
        Convolution3DBackprop::ConvType<TPosition::GM, Convolution3DBackprop::CubeFormat::ND, int32_t>;
    using dedyDxType = Convolution3DBackprop::ConvType<TPosition::GM, dedyCubeFormat, dedyType>;
    using yDxType = Convolution3DBackprop::ConvType<TPosition::GM, yCubeFormat, yType>;
    using biasDxType = Convolution3DBackprop::ConvType<TPosition::GM, biasCubeFormat, biasType>;
    static constexpr Conv3dConfig conv3dConfig = {b2Condition, kernelSplitMode, groupMode, b1Condition, enableC04Flag};
    Convolution3DBackprop::Conv3DBackpropInput<filterDxType, inputSizeDxType, dedyDxType, yDxType, biasDxType, conv3dConfig> dedx_;

    GlobalTensor<filterType> filterGm_;
    GlobalTensor<filterType> dedyGm_;
    GlobalTensor<yType> yGm_;
#if defined(__DAV_310R6__)
    GlobalTensor<biasType> biasGm_;
#endif

    uint64_t batchStrideA_ = 1;
    uint64_t batchStrideC_ = 1;
    uint64_t groupStrideA_ = 1;
    uint64_t groupStrideB_ = 1;
    uint64_t groupStrideC_ = 1;
    uint64_t coutStrideA_ = 1;
    uint64_t cinStrideB_ = 1;
    uint64_t cinStrideC_ = 1;

    uint64_t offsetA_ = 0;
    uint64_t offsetB_ = 0;
    uint64_t offsetC_ = 0;
#if defined(__DAV_310R6__)
    uint64_t offsetBias_ = 0;
#endif

    uint32_t kSCoreIdx_ = 0;
    uint32_t batchCoreIdx_ = 0;
    uint32_t mCoreIdx_ = 0;
    uint32_t nCoreIdx_ = 0;
    uint32_t kCoreIdx_ = 0;
    uint32_t dCoreIdx_ = 0;
    uint32_t groupCoreIdx_ = 0;
    int32_t curHoStartIdx_ = 0;

    uint64_t singleShapeBatch_ = 0;
    uint64_t singleShapeM_ = 0;
    uint32_t singleShapeN_ = 0;
    uint64_t singleShapeK_ = 0;
    uint32_t singleShapeDin_ = 0;
    uint32_t singleShapeGroup_ = 0;
    uint32_t singleShapeNInGroup_ = 0;
    uint64_t singleShapeKInGroup_ = 0;
    bool enableSplitDk_ = false;
    bool enableVecTrans_ = false;

    const conv_bp_v2_kernel::TConv3DInputV2Tiling* tiling_;
    const conv_bp_v2_kernel::Conv3DBackpropInputV2Params* params_;

    __aicore__ inline void InitTilingData(const conv_bp_v2_kernel::Conv3DBackpropInputV2TilingData *tilingData)
    {
        tiling_ = &(tilingData->conv3DDxTiling);
        params_ = &(tilingData->params);
    }

    __aicore__ inline void InitBlockStride()
    {
        if constexpr (dedyCubeFormat == Convolution3DBackprop::CubeFormat::NCDHW) {
            coutStrideA_ = static_cast<uint64_t>(tiling_->dout) * tiling_->ho * tiling_->wo;
            batchStrideA_ = coutStrideA_ * tiling_->cout;
        } else {
            batchStrideA_ = static_cast<uint64_t>(tiling_->dout) * tiling_->ho * tiling_->wo * tiling_->cout;
        }
        if (this->enableVecTrans_) {
            cinStrideB_ = 1;
        } else {
            if constexpr (filterCubeFormat == Convolution3DBackprop::CubeFormat::NCDHW) {
                cinStrideB_ = static_cast<uint64_t>(tiling_->dk) * tiling_->hk * tiling_->wk;
            } else if constexpr (filterCubeFormat == Convolution3DBackprop::CubeFormat::NDHWC) {
                cinStrideB_ = 1;
            } else {  // DHWCN
                cinStrideB_ = static_cast<uint64_t>(tiling_->cout);
            }
        }

        if constexpr (yCubeFormat == Convolution3DBackprop::CubeFormat::NCDHW) {
            cinStrideC_ = static_cast<uint64_t>(tiling_->di) * tiling_->hi * tiling_->wi;
            batchStrideC_ = cinStrideC_ * tiling_->cin;
        } else {
            batchStrideC_ = static_cast<uint64_t>(tiling_->di) * tiling_->hi * tiling_->wi * tiling_->cin;
        }

        if (unlikely(tiling_->group > 1)) {
            groupStrideA_ = coutStrideA_ * tiling_->coutG;
            if constexpr (filterCubeFormat == Convolution3DBackprop::CubeFormat::NCDHW) {
                if constexpr (groupMode == TPL_GROUP_MODE_ENLARGE) {
                    groupStrideB_ = static_cast<uint64_t>(tiling_->coutG) * tiling_->cinG / tiling_->enlarge * cinStrideB_;
                } else {
                    groupStrideB_ = static_cast<uint64_t>(tiling_->coutG) * tiling_->cinG * cinStrideB_;
                }
            } else if constexpr (filterCubeFormat == Convolution3DBackprop::CubeFormat::NDHWC) {
                if constexpr (groupMode == TPL_GROUP_MODE_ENLARGE) {
                    groupStrideB_ = static_cast<uint64_t>(tiling_->coutG) * tiling_->dk * tiling_->hk * tiling_->wk *
                        tiling_->cinG / tiling_->enlarge;
                } else {
                    groupStrideB_ = static_cast<uint64_t>(tiling_->coutG) * tiling_->dk * tiling_->hk * tiling_->wk *
                        tiling_->cinG;
                }
            } else {  // DHWCN
                groupStrideB_ = static_cast<uint64_t>(tiling_->coutG);
            }
            groupStrideC_ = cinStrideC_ * tiling_->cinG;
        }
    }

    __aicore__ inline void CalcBlockOffsetA(uint32_t batchIdx)
    {
        uint32_t alignedHoStartIdx = 0;
        if constexpr (kernelSplitMode == TPL_NO_SPLIT_KERNEL) {
            alignedHoStartIdx = curHoStartIdx_ < 0 ? 0 : ((curHoStartIdx_ + tiling_->strideH - 1) / tiling_->strideH);
        } else {
            alignedHoStartIdx = curHoStartIdx_ < 0 ? 0 : curHoStartIdx_;
        }

        uint64_t mOffsetA = 0;
        if constexpr (kernelSplitMode != TPL_SPLIT_KERNEL_H) {
            if constexpr (dedyCubeFormat == Convolution3DBackprop::CubeFormat::NCDHW) {
                mOffsetA = static_cast<uint64_t>(alignedHoStartIdx) * tiling_->wo;
            } else {
                mOffsetA = (static_cast<uint64_t>(alignedHoStartIdx) * tiling_->wo) * tiling_->cout;
            }
        }
        uint64_t batchOffsetA = batchStrideA_ * batchIdx;
        uint64_t coutOffsetA = coutStrideA_ * kCoreIdx_ * tiling_->singleCoreCout;
        offsetA_ = batchOffsetA + coutOffsetA + mOffsetA;
    }

    __aicore__ inline void CalcBlockOffsetB()
    {
        if constexpr (groupMode == TPL_GROUP_MODE_ENLARGE) {
            offsetB_ = 0;
        } else {
            offsetB_ = cinStrideB_ * nCoreIdx_ * tiling_->singleCoreCin;
        }
    }

    __aicore__ inline void CalcBlockOffsetC(uint32_t batchIdx)
    {
        uint64_t batchOffsetC = batchStrideC_ * batchIdx;
        uint64_t cinOffsetC = cinStrideC_ * nCoreIdx_ * tiling_->singleCoreCin;
        uint64_t mOffsetC = mCoreIdx_ * tiling_->singleCoreM;
        if constexpr (kernelSplitMode == TPL_SPLIT_KERNEL_H) {
            // singleCoreM 不一定会对齐wi，当不对齐时，需要计算非整行时输出地址
            uint64_t curSkipMSize = (mOffsetC / tiling_->wi) *
                static_cast<uint64_t>(tiling_->wi) * (tiling_->strideH - 1);
            mOffsetC += curSkipMSize;
        } else if constexpr (kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
            uint64_t splitWi = DivCeil(tiling_->wi, tiling_->strideW);
            uint64_t curSingleHi = tiling_->singleCoreM / splitWi;
            uint64_t curSingleCoreM = curSingleHi * tiling_->strideH * tiling_->wi;
            mOffsetC = mCoreIdx_ * curSingleCoreM;
        }

        if constexpr (yCubeFormat == Convolution3DBackprop::CubeFormat::NDHWC) {
            mOffsetC *= tiling_->cin;
        }
        offsetC_ = batchOffsetC + cinOffsetC + mOffsetC;
    }

    __aicore__ inline void CalcGroupBlockOffset(uint32_t groupIdx)
    {
        if (likely(groupIdx == 0)) {
            return;
        }
        offsetA_ += groupStrideA_ * groupIdx;
        offsetB_ += groupStrideB_ * groupIdx;
        offsetC_ += groupStrideC_ * groupIdx;
    }

#if defined(__DAV_310R6__)
    __aicore__ inline void CalcBiasOffset()
    {
        if constexpr (biasFormat != FORMAT_MAX) {
            offsetBias_ = static_cast<uint64_t>(nCoreIdx_) * tiling_->singleCoreCin;
        }
    }
#endif

    __aicore__ inline void CalcBlockOffset(uint32_t batchIdx, uint32_t groupIdx)
    {
        CalcBlockOffsetA(batchIdx);
        CalcBlockOffsetB();
        CalcBlockOffsetC(batchIdx);
        CalcGroupBlockOffset(groupIdx);
#if defined(__DAV_310R6__)
        CalcBiasOffset();
#endif
    }

    __aicore__ inline void CalcBatchOffset()
    {
        offsetA_ += batchStrideA_;
        offsetC_ += batchStrideC_;
    }

    __aicore__ inline void CalcBatchOffset(uint64_t &backupOffsetA, uint64_t backupOffsetB,
        uint64_t backupOffsetC)
    {
        offsetA_ = backupOffsetA + batchStrideA_;
        offsetB_ = backupOffsetB;
        offsetC_ = backupOffsetC + batchStrideC_;
    }

    __aicore__ inline void CalcBatchOffsetForKernelSplit(uint64_t &curOffsetA, uint64_t &curOffsetC)
    {
        curOffsetA += batchStrideA_;
        curOffsetC += batchStrideC_;
    }

    __aicore__ inline void CalcGroupOffset()
    {
        offsetA_ += groupStrideA_;
        offsetB_ += groupStrideB_;
        offsetC_ += groupStrideC_;
    }

    __aicore__ inline void ReCalSingleCoreShapeForOutputPad(uint64_t &mTail, uint32_t &dTail)
    {
        int32_t diRemain = static_cast<int32_t>(tiling_->backpropPadTail - (tiling_->dilationD * (tiling_->dk - 1)));
        if (dTail > 0 && dTail > diRemain) {
            dTail = dTail - diRemain;
        }

        int32_t hiRemain = static_cast<int32_t>(tiling_->backpropPadDown - (tiling_->dilationH * (tiling_->hk - 1)));
        if (mTail > 0 && mTail >= hiRemain * tiling_->wi) {
            int32_t hiLen = hiRemain * tiling_->wi;
            mTail = (mTail > hiLen) ? mTail - hiLen : 0;
        }
    }

    __aicore__ inline void CalSingleCoreShape()
    {
        uint64_t blockIdx = GetAicBlockIdx();
        kCoreIdx_ = blockIdx % params_->kDim;
        nCoreIdx_ = (blockIdx / params_->kDim) % params_->nDim;
        mCoreIdx_ = (blockIdx / (params_->kDim * params_->nDim)) % params_->mDim;
        batchCoreIdx_ = (blockIdx / (params_->kDim * params_->nDim * params_->mDim)) % params_->batchDim;
        dCoreIdx_ = (blockIdx / (params_->kDim * params_->nDim * params_->mDim * params_->batchDim)) % params_->dDim;
        groupCoreIdx_ = (blockIdx / (params_->kDim * params_->nDim * params_->mDim * params_->batchDim * params_->dDim)) % params_->groupDim;
        // 放大后的绝对坐标
        uint32_t backpropPadUp = tiling_->backpropPadUp;
        if constexpr (kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
            backpropPadUp >>= 1;
            curHoStartIdx_ = static_cast<int32_t>(mCoreIdx_ * (tiling_->singleCoreM / (tiling_->wi * tiling_->strideH))) - backpropPadUp;
        } else {
            curHoStartIdx_ = static_cast<int32_t>(mCoreIdx_ * tiling_->singleCoreM / tiling_->wi) - backpropPadUp;
        }
        uint64_t batchTail = tiling_->batch - batchCoreIdx_ * tiling_->singleCoreBatch;
        uint64_t mTail = tiling_->hi * tiling_->wi - mCoreIdx_ * tiling_->singleCoreM;
        uint32_t dTail = tiling_->di - dCoreIdx_ * tiling_->singleCoreDin;
        uint32_t groupTail = tiling_->group - groupCoreIdx_ * tiling_->singleCoreGroup;
        if constexpr (kernelSplitMode != TPL_SPLIT_KERNEL_H) {
            ReCalSingleCoreShapeForOutputPad(mTail, dTail);
        }

        singleShapeBatch_ = Min(batchTail, tiling_->singleCoreBatch);
        singleShapeM_ = Min(mTail, tiling_->singleCoreM);
        singleShapeDin_ = Min(dTail, tiling_->singleCoreDin);
        singleShapeGroup_ = Min(groupTail, tiling_->singleCoreGroup);
        uint32_t curDinStartIdx = dCoreIdx_ * tiling_->singleCoreDin;
        uint32_t curCinStartIdx = nCoreIdx_ * tiling_->singleCoreCin;
        uint32_t curCoutStartIdx = 0;

        if (likely(tiling_->group == 1)) {
            uint32_t nTail = tiling_->cin - nCoreIdx_ * tiling_->singleCoreCin;
            singleShapeN_ = Min(nTail, tiling_->singleCoreCin);
            singleShapeK_ = tiling_->cout * tiling_->hk * tiling_->wk;
        } else {
            uint32_t nTail = tiling_->cin - groupCoreIdx_ * tiling_->singleCoreGroup * tiling_->cinG;
            uint32_t coutTail = tiling_->cout - groupCoreIdx_ * tiling_->singleCoreGroup * tiling_->coutG;
            singleShapeN_ = Min(nTail, tiling_->singleCoreGroup * tiling_->cinG);
            singleShapeK_ = Min(coutTail, tiling_->singleCoreGroup * tiling_->coutG);
        }

        dedx_.SetSingleShape(singleShapeM_, singleShapeK_, singleShapeN_, singleShapeDin_);
        dedx_.SetStartIdx(curDinStartIdx, this->mCoreIdx_ * this->tiling_->singleCoreM, curCinStartIdx, curCoutStartIdx);
    }

    __aicore__ inline void CalSingleCoreShapeInGroup(uint32_t groupIdx)
    {
        uint32_t coutTail = singleShapeK_ - groupIdx * tiling_->coutG;
        uint32_t nTail = singleShapeN_ - groupIdx * tiling_->cinG;
        if (coutTail == 0) {
            singleShapeKInGroup_ = 0;
            return;
        }

        uint64_t blockIdx = GetAicBlockIdx();
        nCoreIdx_ = (blockIdx / params_->kDim) % params_->nDim;
        if (nTail <= nCoreIdx_ * tiling_->singleCoreCin) {
            singleShapeNInGroup_ = 0;
            return;
        }
        uint32_t singleShapeCoutInGroup = Min(coutTail, tiling_->coutG);
        singleShapeKInGroup_ = static_cast<uint64_t>(singleShapeCoutInGroup) * tiling_->hk * tiling_->wk;
        nTail = Min(nTail, tiling_->cinG);
        nTail -= nCoreIdx_ * tiling_->singleCoreCin;
        singleShapeNInGroup_ = Min(nTail, tiling_->singleCoreCin);
        dedx_.SetSingleShape(singleShapeM_, singleShapeKInGroup_, singleShapeNInGroup_, singleShapeDin_);
    }

    __aicore__ inline void InitMixCoreBuffer(GM_ADDR workSpace)
    {
        dedx_.ctx.l0cOutGm_.SetGlobalBuffer((__gm__ float *)workSpace);
    }

    template <typename T>
    __aicore__ inline T Max(T a, T b)
    {
        return a > b ? a : b;
    }

    template <typename T>
    __aicore__ inline T Min(T a, T b)
    {
        return a < b ? a : b;
    }
};
}  // namespace AscendC

#endif  // CONV3D_BACKPROP_INPUT_V2_ADVANCE_H
