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
 * \file conv3d_dx_block_base.h
 * \brief a basic block move strategy that reuse overlapping sliding window cache
 */
#ifndef CONV3D_DX_BLOCK_BASE_H
#define CONV3D_DX_BLOCK_BASE_H

#include "conv3d_bp_input.h"
#include "basic_api/kernel_basic_intf.h"
#include "kernel_type.h"
#include "../../conv3d_backprop_input_v2_arch35_tiling_key.h"
#include "lib/matmul_intf.h"
#include "conv3d_backprop_input_v2_tiling_data.h"

#ifndef DTYPE_BIAS
#define DTYPE_BIAS int32_t
#define FORMAT_BIAS FORMAT_MAX  // FORMAT_MAX意为数据格式不支持，用以表达不带bias输入场景
#endif

#ifndef DTYPE_SCALE
#define DTYPE_SCALE uint64_t
#define FORMAT_SCALE FORMAT_MAX  // FORMAT_MAX意为数据格式不支持，用以表达不带scale输入场景
#endif

namespace AscendC {
constexpr uint8_t LOOP_DNM = 1;
constexpr uint8_t LOOP_DMN = 2;
constexpr uint8_t LOOP_MDN = 3;
static constexpr uint8_t SYNC_MODE2 = 2;
static constexpr uint16_t SYNC_AIV_AIC_DET_FLAG = 6;

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
    } else if (format == FORMAT_FRACTAL_Z) {
        return Convolution3DBackprop::CubeFormat::FRACTALZ;
    } else {
        return Convolution3DBackprop::CubeFormat::NCDHW;
    }
}

__aicore__ inline constexpr Convolution3DBackprop::CubeFormat GetScaleFormat(int format)
{
    if (format == FORMAT_ND || format == FORMAT_NCHW) {
        return Convolution3DBackprop::CubeFormat::ND;
    } else {
        return Convolution3DBackprop::CubeFormat::UNSUPPORT;
    }
}

template <typename filterType, int filterFormat, typename dedyType, int dedyFormat, typename yType, int yFormat,
    typename biasType, int biasFormat,
    uint8_t b2Condition, uint8_t kernelSplitMode, uint8_t groupMode,
    uint8_t b1Condition = TPL_GM_TO_L1,
    bool enableC04Flag = false, typename scaleType = uint64_t, int scaleFormat = FORMAT_MAX>
class Conv3dDxBase {
protected:
    uint8_t loopDirect_ = LOOP_DMN;
    uint64_t mCnt_ = 0;
    uint64_t mCoreTail_ = 0;
    uint64_t nCnt_ = 0;
    uint64_t nTailCnt_ = 0;
    uint64_t nCoreTail_ = 0;
    uint64_t nGroupCoreTail_ = 0;
    uint64_t dinCnt_ = 0;
    uint64_t dinCoreTail_ = 0;
    uint64_t coutGroupTail_ = 0;
    uint64_t totalCnt_ = 0;
    uint64_t tailCnt_ = 0;
    uint64_t calRound_ = 0;
    uint64_t usedCoreNum_ = 0;
    uint64_t preOffsetB_ = 0;
    uint8_t preEnableFullLoad = 0;
    uint8_t useUbAccumForSplitK_ = 0;

    static constexpr Convolution3DBackprop::CubeFormat filterCubeFormat = GetFormat(filterFormat);
    static constexpr Convolution3DBackprop::CubeFormat dedyCubeFormat = GetFormat(dedyFormat);
    static constexpr Convolution3DBackprop::CubeFormat yCubeFormat = GetFormat(yFormat);
    static constexpr Convolution3DBackprop::CubeFormat biasCubeFormat = GetFormat(biasFormat);
    static constexpr Convolution3DBackprop::CubeFormat scaleCubeFormat = GetScaleFormat(scaleFormat);
    using filterDxType = Convolution3DBackprop::ConvType<TPosition::GM, filterCubeFormat, filterType>;
    using inputSizeDxType =
        Convolution3DBackprop::ConvType<TPosition::GM, Convolution3DBackprop::CubeFormat::ND, int32_t>;
    using dedyDxType = Convolution3DBackprop::ConvType<TPosition::GM, dedyCubeFormat, dedyType>;
    using yDxType = Convolution3DBackprop::ConvType<TPosition::GM, yCubeFormat, yType>;
    using biasDxType = Convolution3DBackprop::ConvType<TPosition::GM, biasCubeFormat, biasType>;
    using scaleDxType = Convolution3DBackprop::ConvType<TPosition::GM, scaleCubeFormat, scaleType>;
    static constexpr Conv3dConfig conv3dConfig = {b2Condition, kernelSplitMode, groupMode,b1Condition, enableC04Flag};
    Convolution3DBackprop::Conv3DBackpropInput<filterDxType, inputSizeDxType, dedyDxType, yDxType, biasDxType,
        scaleDxType, conv3dConfig> dedx_;

    GlobalTensor<filterType> filterGm_;
    GlobalTensor<dedyType> dedyGm_;
    GlobalTensor<yType> yGm_;
    GlobalTensor<biasType> biasGm_;
    GlobalTensor<scaleType> scaleGm_;

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
    uint64_t offsetBias_ = 0;
    int64_t preOffsetBias_ = -1;
    uint64_t offsetScale_ = 0;
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
    bool enableVecTrans_ = false;
    bool hasBias_ = false;
    bool fullLoadBiasFlag_ = false;
    bool freeBiasFlag_ = false;

    const conv_bp_v2_kernel::TConv3DInputV2Tiling* tiling_;

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
            } else if constexpr (filterCubeFormat == Convolution3DBackprop::CubeFormat::FRACTALZ) {
                cinStrideB_ = static_cast<uint64_t>(tiling_->c0);
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
            } else if constexpr (filterCubeFormat == Convolution3DBackprop::CubeFormat::FRACTALZ) {
                groupStrideB_ = static_cast<uint64_t>(tiling_->coutG) * tiling_->hk * tiling_->wk * tiling_->cinG;
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

    __aicore__ inline void CalcBiasOffset(uint32_t groupIdx)
    {
        if (unlikely(hasBias_)) {
            // bias需要叠加额外的group偏移
            offsetBias_ = static_cast<uint64_t>(nCoreIdx_) * tiling_->singleCoreCin + groupIdx * tiling_->cinG;
        }
    }

    __aicore__ inline void CalcBiasFullLoadFlag()
    {
        if (hasBias_ && (offsetBias_ != preOffsetBias_)) {
            // bias按照group全载
            fullLoadBiasFlag_ = true;
        }
        preOffsetBias_ = offsetBias_;
    }

    __aicore__ inline void CalcScaleOffset(uint32_t groupIdx)
    {
        if constexpr (GetScaleFormat(scaleFormat) != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
            if (tiling_->quantMode == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
                offsetScale_ = static_cast<uint64_t>(nCoreIdx_) * tiling_->singleCoreCin + groupIdx * tiling_->cinG;
            } else if (tiling_->quantMode == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::SCALAR_QUANT)) {
                offsetScale_ = 0;
            }            
        }
    }

    __aicore__ inline void CalcBlockOffset(uint32_t batchIdx, uint32_t groupIdx)
    {
        CalcBlockOffsetA(batchIdx);
        CalcBlockOffsetB();
        CalcBlockOffsetC(batchIdx);
        CalcGroupBlockOffset(groupIdx);

        CalcBiasOffset(groupIdx);
        CalcScaleOffset(groupIdx);
    }
};
}

#endif // CONV3D_BACKPROP_INPUT_ROWC_BLOCK_ADVANCE_H