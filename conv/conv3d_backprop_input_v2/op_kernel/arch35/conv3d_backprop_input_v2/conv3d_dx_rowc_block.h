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
 * \file conv3d_dx_rowc_block.h
 * \brief a basic block move strategy that reuse overlapping sliding window cache
 */
#ifndef CONV3D_DX_ROWC_BLOCK_ADVANCE_H
#define CONV3D_DX_ROWC_BLOCK_ADVANCE_H

#include "conv3d_dx_block_base.h"
#include "../../../inc/macro.h"

namespace AscendC {
constexpr uint8_t LOOP_DNM = 1;
constexpr uint8_t LOOP_DMN = 2;
constexpr uint8_t LOOP_MDN = 3;
static constexpr uint8_t SYNC_MODE2 = 2;
static constexpr uint16_t SYNC_AIV_AIC_DET_FLAG = 6;
constexpr int BLOCK_CUBE_ALIGN_BITS = 4;

template <typename filterType, int filterFormat, typename dedyType, int dedyFormat, typename yType, int yFormat,
          typename biasType, int biasFormat, uint8_t b2Condition, uint8_t kernelSplitMode, uint8_t groupMode,
          uint8_t b1Condition = TPL_GM_TO_L1, bool enableC04Flag = false, typename scale0Type = uint64_t,
          int scale0Format = FORMAT_MAX, typename y1Type = yType, typename scale1Type = scale0Type,
          int scale1Format = scale0Format>
class Conv3dDxOswBlock
    : public Conv3dDxBase<filterType, filterFormat, dedyType, dedyFormat, yType, yFormat, biasType, biasFormat,
                          b2Condition, kernelSplitMode, groupMode, b1Condition, enableC04Flag, scale0Type, scale0Format,
                          y1Type, scale1Type, scale1Format> {
public:
    __aicore__ inline Conv3dDxOswBlock(){};
    __aicore__ inline void Init(GM_ADDR filter, GM_ADDR dedy, GM_ADDR y, GM_ADDR workSpace,
                                const Conv3DBackpropInputArch35TilingData& tilingData, GM_ADDR bias = nullptr,
                                GM_ADDR scale0 = nullptr, GM_ADDR y1 = nullptr, GM_ADDR scale1 = nullptr)
    {
        InitTilingData(tilingData);

        if (!enableC04Flag && !groupMode && !this->useUbAccumForSplitK_) {
            if ASCEND_IS_AIV_SHOULD_RETURN {
                return;
            }
        }

        if (this->useUbAccumForSplitK_ && GetSubBlockIdx() != 0) {
            return;
        }

        if (!this->enableVecTrans_) {
            this->filterGm_.SetGlobalBuffer((__gm__ filterType*)filter);
        } else {
            this->filterGm_.SetGlobalBuffer((__gm__ filterType*)workSpace);
            // 开启前置transpose同时使用累加轴特性需要分段使用
            if (this->useUbAccumForSplitK_) {
                workSpace += static_cast<uint64_t>(this->tiling_->coutG) * this->tiling_->dk * this->tiling_->hk *
                             this->tiling_->wk *
                             (((this->tiling_->cinG + BLOCK_CUBE - 1) >> BLOCK_CUBE_ALIGN_BITS)
                              << BLOCK_CUBE_ALIGN_BITS) *
                             sizeof(filterType); // 4 : 2的4次方
            }
        }
        this->dedyGm_.SetGlobalBuffer((__gm__ dedyType*)dedy);
        this->yGm_.SetGlobalBuffer((__gm__ yType*)y);
#ifdef DTYPE_Y1
        if (tilingData.dualOutput != 0 && y1 != nullptr) {
            this->y1Gm_.SetGlobalBuffer((__gm__ y1Type*)y1);
            this->hasSecondOutput_ = true;
        }
#endif
        if constexpr (GetScaleFormat(scale0Format) != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
            this->scale0Gm_.SetGlobalBuffer((__gm__ scale0Type*)scale0);
        }
#ifdef DTYPE_Y1
        if constexpr (GetScaleFormat(scale1Format) != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
            if (scale1 != nullptr) {
                this->scale1Gm_.SetGlobalBuffer((__gm__ scale1Type*)scale1);
            }
        }
#endif

        if (unlikely(bias != nullptr)) {
            this->hasBias_ = true;
            this->biasGm_.SetGlobalBuffer((__gm__ biasType*)bias);
        }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510) || __DAV_35_FAMILY__
        InitMixCoreBuffer(workSpace);
#endif
#ifdef DTYPE_Y1
        this->dedx_.Init(tilingData, this->hasBias_, this->hasSecondOutput_);
#else
        this->dedx_.Init(tilingData, this->hasBias_);
#endif
    }

    __aicore__ inline void Process()
    {
        if (!enableC04Flag && !groupMode && !this->useUbAccumForSplitK_) {
            if ASCEND_IS_AIV_SHOULD_RETURN {
                return;
            }
        }

        if (this->useUbAccumForSplitK_ && GetSubBlockIdx() != 0) {
            return;
        }

        if (GetAicBlockIdx() >= this->usedCoreNum_) {
            ProcessForUnUsedCore();
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
    uint8_t useUbAccumForSplitK_ = 0;

    __aicore__ inline void CrossCoreWaitVecTrans()
    {
        if (this->enableVecTrans_) {
            if ASCEND_IS_AIC_SCALAR {
#if __CUBE_VECTOR_FUSION_ONLY__
                AscendC::TQueSync<PIPE_MTE3, PIPE_MTE2> sync;
                sync.WaitFlag((event_t)SYNC_AIV_AIC_DET_FLAG);
#else
                CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE2>(SYNC_AIV_AIC_DET_FLAG);
#endif
            }
        }
    }

    __aicore__ inline void ProcessForUnUsedCore()
    {
        CrossCoreWaitVecTrans();
        this->dedx_.End();
    }

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
        this->nGroupCoreTail_ = tailN % this->singleShapeN_;

        uint64_t k = static_cast<uint64_t>(tiling->cout) * tiling->hk * tiling->wk;
        if constexpr (b1Condition == TPL_GM_TO_L1_NO_HK) {
            k = static_cast<uint64_t>(tiling->cout) * tiling->wk;
        } else if constexpr (b1Condition == TPL_GM_TO_L1_NO_HK_WK) {
            k = static_cast<uint64_t>(tiling->cout);
        }
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
            if constexpr (b1Condition == TPL_GM_TO_L1) {
                this->singleShapeK_ = static_cast<uint64_t>(tiling->coutG) * tiling->hk * tiling->wk;
            } else if constexpr (b1Condition == TPL_GM_TO_L1_NO_HK) {
                this->singleShapeK_ = static_cast<uint64_t>(tiling->coutG) * tiling->wk;
            } else if constexpr (b1Condition == TPL_GM_TO_L1_NO_HK_WK) {
                this->singleShapeK_ = static_cast<uint64_t>(tiling->coutG);
            }
        } else {
            if constexpr (b1Condition == TPL_GM_TO_L1) {
                this->singleShapeK_ = static_cast<uint64_t>(tiling->cout) * tiling->hk * tiling->wk;
            } else if constexpr (b1Condition == TPL_GM_TO_L1_NO_HK) {
                this->singleShapeK_ = static_cast<uint64_t>(tiling->cout) * tiling->wk;
            } else if constexpr (b1Condition == TPL_GM_TO_L1_NO_HK_WK) {
                this->singleShapeK_ = static_cast<uint64_t>(tiling->cout);
            }
        }
        // 开启 split cout 同时 split Hk 或 split HkWk 时 Cout 置为 1
        if (tiling->enableSplitK) {
            if constexpr (b1Condition == TPL_GM_TO_L1) {
                this->singleShapeK_ = static_cast<uint64_t>(tiling->kSegment) * tiling->hk * tiling->wk;
            } else if constexpr (b1Condition == TPL_GM_TO_L1_NO_HK) {
                this->singleShapeK_ = static_cast<uint64_t>(tiling->kSegment) * tiling->wk;
            } else if constexpr (b1Condition == TPL_GM_TO_L1_NO_HK_WK) {
                this->singleShapeK_ = static_cast<uint64_t>(tiling->kSegment);
            }
        }
        this->singleShapeN_ = tiling->singleCoreCin;
        this->singleShapeDin_ = tiling->singleCoreDin;
        this->enableVecTrans_ = tiling->enableVecTrans;
        this->useUbAccumForSplitK_ = tiling->useUbAccumForSplitK;
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
#ifdef DTYPE_Y1
        if (this->hasSecondOutput_) {
            this->dedx_.IterateAll(this->yGm_[this->offsetC_], this->y1Gm_[this->offsetC_], 0, this->fullLoadBiasFlag_,
                                   this->freeBiasFlag_);
        } else {
#endif
            this->dedx_.IterateAll(this->yGm_[this->offsetC_], 0, this->fullLoadBiasFlag_, this->freeBiasFlag_);
#ifdef DTYPE_Y1
        }
#endif
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
            this->dedx_.SetBatchCoreIdx(this->batchCoreIdx_);
            this->dedx_.SetSingleShape(mCoreUse, coutCoreUse, nCoreUse, dinCoreUse);
            this->dedx_.SetStartIdx(this->dCoreIdx_ * this->singleShapeDin_, this->mCoreIdx_ * singleCoreM,
                                    this->nCoreIdx_ * singleCoreCin, 0);
            this->CalBasicBlockOffset();
            this->dedx_.SetOutBackprop(this->dedyGm_[this->offsetA_]);
            this->dedx_.SetWeight(this->filterGm_[this->offsetB_]);

            this->CheckFullLoadEnable();

            if constexpr (GetScaleFormat(scale0Format) != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
                this->dedx_.SetScale(this->scale0Gm_[this->offsetScale_]);
            }
#ifdef DTYPE_Y1
            if constexpr (GetScaleFormat(scale1Format) != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
                if (this->hasSecondOutput_) {
                    this->dedx_.SetScale1(this->scale1Gm_[this->offsetScale_]);
                }
            }
#endif

            if (j == 0) {
                this->CrossCoreWaitVecTrans();
            }

            if (unlikely(this->hasBias_)) {
                this->dedx_.SetBias(this->biasGm_[this->offsetBias_]);
                this->IterateAllForBias(firstloadbias);
            } else {
#ifdef DTYPE_Y1
                if (this->hasSecondOutput_) {
                    this->dedx_.IterateAll(this->yGm_[this->offsetC_], this->y1Gm_[this->offsetC_], 0, false, false);
                } else {
#endif
                    this->dedx_.IterateAll(this->yGm_[this->offsetC_], 0, false, false);
#ifdef DTYPE_Y1
                }
#endif
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

        if ASCEND_IS_AIC_SCALAR {
            // 当b1全载且dk=1时，只需要加载一次b1，在循环结束后释放
            this->dedx_.FreeB1Tensor();
            // Release bias after full load.
            // Note: Currently only the bias full-load scenario exists.
            if (unlikely(this->tiling_->isBiasFullLoad && this->hasBias_)) {
                this->dedx_.FreeBiasTensor();
            }
        }
    }

    __aicore__ inline void InitMixCoreBuffer(GM_ADDR workSpace)
    {
        this->dedx_.ctx.l0cOutGm_.SetGlobalBuffer((__gm__ float*)workSpace);
    }
};
} // namespace AscendC

#endif // CONV3D_BACKPROP_INPUT_ROWC_BLOCK_ADVANCE_H
