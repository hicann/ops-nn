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
 * \file conv3d_dx_block_arch51.h
 * \brief
 */

#ifndef CONV3D_DX_BLOCK_ARCH51_H
#define CONV3D_DX_BLOCK_ARCH51_H

#include "conv3d_bp_input_arch51.h"
#include "basic_api/kernel_basic_intf.h"
#include "kernel_type.h"
#include "lib/matmul_intf.h"

#ifndef DTYPE_BIAS
#define DTYPE_BIAS int32_t
#define FORMAT_BIAS FORMAT_MAX // FORMAT_MAX意为数据格式不支持，用以表达不带bias输入场景
#endif

#ifndef DTYPE_SCALE
#define DTYPE_SCALE uint64_t
#define FORMAT_SCALE FORMAT_MAX // FORMAT_MAX意为数据格式不支持，用以表达不带scale输入场景
#endif

namespace AscendC {
constexpr uint8_t LOOP_DNM = 1;
constexpr uint8_t LOOP_DMN = 2;
constexpr uint8_t LOOP_MDN = 3;

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
          typename biasType, int biasFormat, typename scale0Type = uint64_t, int scale0Format = FORMAT_MAX>
class Conv3dDxBase {
protected:
    static constexpr Convolution3DBackprop::CubeFormat filterCubeFormat = GetFormat(filterFormat);
    static constexpr Convolution3DBackprop::CubeFormat dedyCubeFormat = GetFormat(dedyFormat);
    static constexpr Convolution3DBackprop::CubeFormat yCubeFormat = GetFormat(yFormat);
    static constexpr Convolution3DBackprop::CubeFormat biasCubeFormat = GetFormat(biasFormat);
    using filterDxType = Convolution3DBackprop::ConvType<TPosition::GM, filterCubeFormat, filterType>;
    using inputSizeDxType = Convolution3DBackprop::ConvType<TPosition::GM, Convolution3DBackprop::CubeFormat::ND,
                                                            int32_t>;
    using dedyDxType = Convolution3DBackprop::ConvType<TPosition::GM, dedyCubeFormat, dedyType>;
    using yDxType = Convolution3DBackprop::ConvType<TPosition::GM, yCubeFormat, yType>;
    using biasDxType = Convolution3DBackprop::ConvType<TPosition::GM, biasCubeFormat, biasType>;
    using scale0DxType = Convolution3DBackprop::ConvType<TPosition::GM, GetScaleFormat(scale0Format), scale0Type>;
    Convolution3DBackprop::Conv3DBackpropInput<filterDxType, inputSizeDxType, dedyDxType, yDxType, biasDxType,
                                               scale0DxType>
        dedx_;

    GlobalTensor<filterType> filterGm_;
    GlobalTensor<dedyType> dedyGm_;
    GlobalTensor<yType> yGm_;
    GlobalTensor<biasType> biasGm_;
    GlobalTensor<scale0Type> scale0Gm_;

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
    uint32_t dCoreIdx_ = 0;
    uint32_t groupCoreIdx_ = 0;
    int32_t curHoStartIdx_ = 0;

    uint64_t singleShapeBatch_ = 0;
    uint64_t singleShapeM_ = 0;
    uint32_t singleShapeN_ = 0;
    uint64_t singleShapeK_ = 0;
    uint32_t singleShapeDin_ = 0;
    bool hasBias_ = false;
    bool fullLoadBiasFlag_ = false;
    bool freeBiasFlag_ = false;

    const Conv3DBackpropInputArch35TilingData* tiling_;

    __aicore__ inline void InitBlockStride()
    {
        const auto* tiling = tiling_;
        coutStrideA_ = static_cast<uint64_t>(tiling->dout) * tiling->ho * tiling->wo;
        batchStrideA_ = coutStrideA_ * tiling->cout;
        cinStrideB_ = 1;
        cinStrideC_ = static_cast<uint64_t>(tiling->di) * tiling->hi * tiling->wi;
        batchStrideC_ = cinStrideC_ * tiling->cin;

        if (unlikely(tiling->group > 1)) {
            groupStrideA_ = coutStrideA_ * tiling->coutG;
            groupStrideB_ = static_cast<uint64_t>(tiling->coutG) * tiling->dk * tiling->hk * tiling->wk * tiling->cinG;
            groupStrideC_ = cinStrideC_ * tiling->cinG;
        }
    }

    __aicore__ inline void CalcBlockOffsetA(uint32_t batchIdx)
    {
        uint32_t alignedHoStartIdx = curHoStartIdx_ < 0 ? 0 :
                                                          ((curHoStartIdx_ + tiling_->strideH - 1) / tiling_->strideH);

        uint64_t mOffsetA = static_cast<uint64_t>(alignedHoStartIdx) * tiling_->wo;
        uint64_t batchOffsetA = batchStrideA_ * batchIdx;
        uint64_t coutOffsetA = coutStrideA_ * kSCoreIdx_ * tiling_->singleCoreCout;
        offsetA_ = batchOffsetA + coutOffsetA + mOffsetA;
    }

    __aicore__ inline void CalcBlockOffsetB()
    {
        // A16W8 fractal_z 保留 N 方向核内偏移。
        offsetB_ = cinStrideB_ * nCoreIdx_ * tiling_->singleCoreCin;
    }

    __aicore__ inline void CalcBlockOffsetC(uint32_t batchIdx)
    {
        uint64_t batchOffsetC = batchStrideC_ * batchIdx;
        uint64_t cinOffsetC = cinStrideC_ * nCoreIdx_ * tiling_->singleCoreCin;
        uint64_t mOffsetC = 0;
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
        if (unlikely(hasBias_ && (offsetBias_ != preOffsetBias_))) {
            // bias按照group全载
            fullLoadBiasFlag_ = true;
        }
        preOffsetBias_ = offsetBias_;
    }

    __aicore__ inline void CalcScaleOffset(uint32_t groupIdx)
    {
        if constexpr (GetScaleFormat(scale0Format) != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
            if (tiling_->quantMode0 == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
                offsetScale_ = static_cast<uint64_t>(nCoreIdx_) * tiling_->singleCoreCin + groupIdx * tiling_->cinG;
            } else {
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

template <typename filterType, int filterFormat, typename dedyType, int dedyFormat, typename yType, int yFormat,
          typename biasType, int biasFormat, typename scale0Type = uint64_t, int scale0Format = FORMAT_MAX>
class Conv3dDxOswBlock : public Conv3dDxBase<filterType, filterFormat, dedyType, dedyFormat, yType, yFormat, biasType,
                                             biasFormat, scale0Type, scale0Format> {
public:
    __aicore__ inline Conv3dDxOswBlock(){};
    __aicore__ inline void Init(GM_ADDR filter, GM_ADDR dedy, GM_ADDR y, GM_ADDR workSpace,
                                const Conv3DBackpropInputArch35TilingData& tilingData, GM_ADDR bias = nullptr,
                                GM_ADDR scale0 = nullptr)
    {
        InitTilingData(tilingData);

        this->filterGm_.SetGlobalBuffer((__gm__ filterType*)filter);
        this->dedyGm_.SetGlobalBuffer((__gm__ dedyType*)dedy);
        this->yGm_.SetGlobalBuffer((__gm__ yType*)y);
        if constexpr (GetScaleFormat(scale0Format) != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
            this->scale0Gm_.SetGlobalBuffer((__gm__ scale0Type*)scale0);
        }

        if (unlikely(bias != nullptr)) {
            this->hasBias_ = true;
            this->biasGm_.SetGlobalBuffer((__gm__ biasType*)bias);
        }
        this->dedx_.Init(tilingData, this->hasBias_);
    }

    __aicore__ inline void Process()
    {
        if (GetAicBlockIdx() >= this->usedCoreNum_) {
            this->dedx_.End();
            return;
        }

        CalBasicBlock();
        this->dedx_.End();
    }

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

    __aicore__ inline void CalBasicBlockCnt()
    {
        const auto* tiling = this->tiling_;
        uint64_t m = static_cast<uint64_t>(tiling->hi) * tiling->wi;
        this->mCnt_ = DivCeil(m, this->singleShapeM_);
        this->mCoreTail_ = m - (this->mCnt_ - 1) * this->singleShapeM_;

        uint64_t n = tiling->cinG;
        uint64_t tailN = tiling->cin - n * (tiling->group - 1);
        this->nCnt_ = DivCeil(n, this->singleShapeN_);
        this->nTailCnt_ = DivCeil(tailN, this->singleShapeN_);
        this->nCoreTail_ = n - (this->nCnt_ - 1) * this->singleShapeN_;

        this->nGroupCoreTail_ = tailN - (this->nTailCnt_ - 1) * this->singleShapeN_;

        uint64_t k = static_cast<uint64_t>(tiling->cout) * tiling->hk * tiling->wk;
        // enlarge场景，cout可能无法被group整除，因此需要计算最后一组实际参与计算的k
        this->coutGroupTail_ = k - (tiling->group - 1) * this->singleShapeK_;

        if (this->singleShapeDin_ > 1) {
            this->dinCnt_ = DivCeil(tiling->di, this->singleShapeDin_);
            this->dinCoreTail_ = tiling->di - (this->dinCnt_ - 1) * this->singleShapeDin_;
        } else {
            this->dinCnt_ = tiling->di;
            this->dinCoreTail_ = 1;
        }

        // 记录基本块的位置
        this->totalCnt_ = static_cast<uint64_t>(tiling->batch) * static_cast<uint64_t>(tiling->group) * this->dinCnt_ *
                          this->mCnt_ * this->nCnt_;

        uint64_t blockNum = GetBlockNum();
        if (this->totalCnt_ < blockNum) {
            this->usedCoreNum_ = this->totalCnt_;
        } else {
            this->usedCoreNum_ = blockNum;
        }

        this->calRound_ = this->totalCnt_ / this->usedCoreNum_;
        this->tailCnt_ = this->totalCnt_ - this->calRound_ * this->usedCoreNum_;
    }

    __aicore__ inline void InitBasicBlockLoopDirect()
    {
        const auto* tiling = this->tiling_;
        // 1.Kernel>1时右矩阵格式转换Bound,有效带宽不到1T,先沿着N走位;Kerenel=1沿着窄的方向走位
        // 2.M方向尽可能按照滑窗叠加的方向来分基本块，优先复用D的滑窗OverLap, 其次时H滑窗在L1边界的叠加
        if (tiling->dk > 1) {
            this->loopDirect_ = LOOP_MDN;
        } else if (tiling->hk > 1) {
            this->loopDirect_ = LOOP_DMN;
        } else if (this->mCnt_ > this->nCnt_) {
            this->loopDirect_ = LOOP_DMN;
        } else {
            this->loopDirect_ = LOOP_DNM;
        }
    }

    __aicore__ inline void CalBasicBlockIdx(uint64_t basicBlockIdx)
    {
        const auto* tiling = this->tiling_;
        uint64_t mnCnt = this->mCnt_ * this->nCnt_;
        uint64_t depthMNCnt = this->dinCnt_ * mnCnt;
        if (unlikely(tiling->group > 1)) {
            uint64_t groupDepthMNCnt = static_cast<uint64_t>(tiling->group) * depthMNCnt;
            this->batchCoreIdx_ = basicBlockIdx / groupDepthMNCnt;
            basicBlockIdx -= this->batchCoreIdx_ * groupDepthMNCnt;
            this->groupCoreIdx_ = basicBlockIdx / depthMNCnt;
            basicBlockIdx -= this->groupCoreIdx_ * depthMNCnt;
        } else {
            this->batchCoreIdx_ = basicBlockIdx / depthMNCnt;
            basicBlockIdx -= this->batchCoreIdx_ * depthMNCnt;
        }

        if (this->loopDirect_ == LOOP_MDN) {
            uint64_t depthNcnt = this->dinCnt_ * this->nCnt_;
            this->mCoreIdx_ = basicBlockIdx / depthNcnt;
            basicBlockIdx -= this->mCoreIdx_ * depthNcnt;
            if (this->dinCnt_ > 1) {
                this->dCoreIdx_ = basicBlockIdx / this->nCnt_;
                basicBlockIdx -= this->dCoreIdx_ * this->nCnt_;
            } else {
                this->dCoreIdx_ = 0;
            }
            this->nCoreIdx_ = basicBlockIdx;
        } else if (this->loopDirect_ == LOOP_DMN) {
            if (this->dinCnt_ > 1) {
                this->dCoreIdx_ = basicBlockIdx / mnCnt;
                basicBlockIdx -= this->dCoreIdx_ * mnCnt;
            } else {
                this->dCoreIdx_ = 0;
            }
            this->mCoreIdx_ = basicBlockIdx / this->nCnt_;
            basicBlockIdx -= this->mCoreIdx_ * this->nCnt_;
            this->nCoreIdx_ = basicBlockIdx;
        } else if (this->loopDirect_ == LOOP_DNM) {
            if (this->dinCnt_ > 1) {
                this->dCoreIdx_ = basicBlockIdx / mnCnt;
                basicBlockIdx -= this->dCoreIdx_ * mnCnt;
            } else {
                this->dCoreIdx_ = 0;
            }
            this->nCoreIdx_ = basicBlockIdx / this->mCnt_;
            basicBlockIdx -= this->nCoreIdx_ * this->mCnt_;
            this->mCoreIdx_ = basicBlockIdx;
        }
    }

    __aicore__ inline void InitTilingData(const Conv3DBackpropInputArch35TilingData& tilingData)
    {
        this->tiling_ = &(tilingData);
        this->dedx_.ctx.curEnableFullLoad_ = this->tiling_->enableFullLoad;
        const auto* tiling = this->tiling_;
        this->singleShapeM_ = tiling->singleCoreM;
        if (unlikely(tiling->group > 1)) {
            this->singleShapeK_ = static_cast<uint64_t>(tiling->coutG) * tiling->hk * tiling->wk;
        } else {
            this->singleShapeK_ = static_cast<uint64_t>(tiling->cout) * tiling->hk * tiling->wk;
        }
        this->singleShapeN_ = tiling->singleCoreCin;
        this->singleShapeDin_ = tiling->singleCoreDin;
        this->CalBasicBlockCnt();
        this->InitBasicBlockLoopDirect();
        this->InitBlockStride();
    }

    __aicore__ inline void CalBasicBlockOffset()
    {
        // 当前A的偏移仍然要用到ho，在切w模板拓展后可以从API外部剥离对ho的感知
        this->curHoStartIdx_ = this->dedx_.ctx.curHoStartIdx_;
        this->CalcBlockOffset(this->batchCoreIdx_, this->groupCoreIdx_);
    }

    __aicore__ inline void CheckFullLoadEnable()
    {
        if (this->offsetB_ == this->preOffsetB_) {
            this->preEnableFullLoad = this->dedx_.ctx.curEnableFullLoad_;
            return;
        }
        this->preOffsetB_ = this->offsetB_;
        // 代表B矩阵偏移变化且上一轮是全载
        if (this->dedx_.ctx.curEnableFullLoad_ == 1 && this->preEnableFullLoad == this->dedx_.ctx.curEnableFullLoad_) {
            // 释放上一轮全载且后续不全载
            this->dedx_.FreeB1Tensor();
            this->dedx_.ctx.curEnableFullLoad_ = 0;
        }
        this->preEnableFullLoad = this->dedx_.ctx.curEnableFullLoad_;
    }

    __aicore__ inline void IterateAllForBias(bool& firstloadbias)
    {
        this->CalcBiasFullLoadFlag();
        this->freeBiasFlag_ = this->fullLoadBiasFlag_ && firstloadbias;
        this->dedx_.IterateAll(this->yGm_[this->offsetC_], 0, this->fullLoadBiasFlag_, this->freeBiasFlag_);
        if (this->fullLoadBiasFlag_) {
            firstloadbias = true;
        }
        this->fullLoadBiasFlag_ = false;
    }

    __aicore__ inline void CalBasicBlockCore(uint64_t blockIdx, uint64_t blockNum)
    {
        bool firstloadbias = false;
        uint64_t basicIdx = blockIdx;

        const auto* tiling = this->tiling_;
        const uint32_t group = tiling->group;
        const uint64_t singleCoreM = tiling->singleCoreM;
        const uint32_t singleCoreCin = tiling->singleCoreCin;

        for (uint64_t j = 0; j < this->calRound_; ++j) {
            this->CalBasicBlockIdx(basicIdx);
            basicIdx += blockNum;

            uint64_t mCoreUse = (this->mCoreIdx_ == (this->mCnt_ - 1)) ? this->mCoreTail_ : this->singleShapeM_;
            uint64_t nCoreUse = (this->nCoreIdx_ == (this->nCnt_ - 1)) ? this->nCoreTail_ : this->singleShapeN_;
            uint64_t dinCoreUse = (this->dCoreIdx_ == (this->dinCnt_ - 1)) ? this->dinCoreTail_ : this->singleShapeDin_;
            uint64_t coutCoreUse = this->singleShapeK_;
            if (unlikely(group > 1)) {
                coutCoreUse = (this->groupCoreIdx_ == (group - 1)) ? this->coutGroupTail_ : coutCoreUse;
                if (unlikely(this->tiling_->cin % this->tiling_->cinG != 0 && this->groupCoreIdx_ == (group - 1))) {
                    if (this->nCoreIdx_ == this->nTailCnt_ - 1) {
                        nCoreUse = this->nGroupCoreTail_;
                    } else if (this->nCoreIdx_ > this->nTailCnt_ - 1) {
                        continue;
                    }
                }
            }
            this->dedx_.SetSingleShape(mCoreUse, coutCoreUse, nCoreUse, dinCoreUse);
            this->dedx_.SetStartIdx(this->dCoreIdx_ * this->singleShapeDin_, this->mCoreIdx_ * singleCoreM,
                                    this->nCoreIdx_ * singleCoreCin);
            this->CalBasicBlockOffset();
            this->dedx_.SetOutBackprop(this->dedyGm_[this->offsetA_]);
            this->dedx_.SetWeight(this->filterGm_[this->offsetB_]);

            this->CheckFullLoadEnable();

            if constexpr (GetScaleFormat(scale0Format) != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
                this->dedx_.SetScale(this->scale0Gm_[this->offsetScale_]);
            }

            if (unlikely(this->hasBias_)) {
                this->dedx_.SetBias(this->biasGm_[this->offsetBias_]);
                this->IterateAllForBias(firstloadbias);
            } else {
                this->dedx_.IterateAll(this->yGm_[this->offsetC_], 0, false, false);
            }
        }
    }

    __aicore__ inline void CalBasicBlock()
    {
        uint64_t blockIdx = GetAicBlockIdx();
        // 拖尾的部分依次分配到前面的核计算，这些核会多算一轮
        if (blockIdx < this->tailCnt_) {
            ++this->calRound_;
        }

        uint64_t blockNum = this->usedCoreNum_;
        CalBasicBlockCore(blockIdx, blockNum);

        // 当b1全载且dk=1时，只需要加载一次b1，在循环结束后释放
        this->dedx_.FreeB1Tensor();
        // Release bias after full load.
        // Note: Currently only the bias full-load scenario exists.
        if (unlikely(this->tiling_->isBiasFullLoad && this->hasBias_)) {
            this->dedx_.FreeBiasTensor();
        }
    }
};
} // namespace AscendC

#include "conv3d_backprop_input_v2_init_output_arch51.h"

#endif // CONV3D_DX_BLOCK_ARCH51_H
