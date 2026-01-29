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
 * \file conv3d_dx_kernel_split_block.h
 * \brief a basic block move strategy that reuse overlapping sliding window cache
 */
#ifndef CONV3D_BACKPROP_INPUT_KERNEL_SPLIT_BLOCK_H
#define CONV3D_BACKPROP_INPUT_KERNEL_SPLIT_BLOCK_H

#include "conv3d_bp_input.h"
#include "basic_api/kernel_basic_intf.h"
#include "kernel_type.h"
#include "conv3d_dx_rowc_block.h"
#include "../../conv3d_backprop_input_v2_arch35_tiling_key.h"

namespace AscendC {
template <typename filterType, int filterFormat, typename dedyType, int dedyFormat, typename yType, int yFormat,
    typename biasType, int biasFormat,
    uint8_t b2Condition, uint8_t kernelSplitMode, uint8_t groupMode,
    uint8_t b1Condition = TPL_GM_TO_L1,
    bool enableC04Flag = false>
class Conv3dDxKsBlock : public Conv3dDxOswBlock<filterType, filterFormat, dedyType, dedyFormat, yType, yFormat, biasType, biasFormat,
    b2Condition, kernelSplitMode, groupMode, b1Condition, enableC04Flag> {
public:
    __aicore__ inline Conv3dDxKsBlock() {};
    __aicore__ inline void Init(GM_ADDR filter, GM_ADDR dedy, GM_ADDR y, GM_ADDR workSpace,
                                const conv_bp_v2_kernel::Conv3DBackpropInputV2TilingData *tilingData)
    {
        if constexpr (kernelSplitMode != TPL_SPLIT_KERNEL_HW) {
            if ASCEND_IS_AIV {
                return;
            }
        }

        InitTilingData(tilingData);
        if (this->enableVecTrans_) {
            this->filterGm_.SetGlobalBuffer((__gm__ filterType *)workSpace);
            workSpace += static_cast<uint64_t>(this->tiling_->cout) *
                this->tiling_->dk * this->tiling_->hk * this->tiling_->wk *
                (((this->tiling_->cin + BLOCK_CUBE - 1) >> 4) << 4) * sizeof(filterType); // 4 : 2的4次方
        } else {
            this->filterGm_.SetGlobalBuffer((__gm__ filterType *)filter);
        }
        this->dedyGm_.SetGlobalBuffer((__gm__ dedyType *)dedy);
        this->yGm_.SetGlobalBuffer((__gm__ yType *)y);
        this->dedx_.Init(&(tilingData->conv3DDxTiling));

#if defined(__DAV_C310__) || defined(__DAV_310R6__)
        InitMixCoreBuffer(workSpace);
#endif
    }

    __aicore__ inline void Process() {
        if constexpr (kernelSplitMode != TPL_SPLIT_KERNEL_HW) {
            if ASCEND_IS_AIV {
                return;
            }
        }

        if (GetAicBlockIdx() >= this->usedCoreNum_) {
            this->ProcessForUnUsedCore();
            return;
        }

        CalBasicBlock();
        this->dedx_.End();
    }

protected:
    int64_t kernelSplitStrideB_ = 0;
    uint64_t kSTailMCnt_ = 0;
    uint32_t kSUseWorkSpace_ = 0;
    uint64_t kSM_ = 0;
    uint64_t kSTailM_ = 0;

    __aicore__ inline void InitMixCoreBuffer(GM_ADDR workSpace)
    {
        if constexpr (kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
            if (this->kSUseWorkSpace_) {
                this->dedx_.ctx.l0cOutWorkspace_.SetGlobalBuffer((__gm__ yType *)workSpace);
            }
        }
    }

    __aicore__ inline void CalBasicBlockCnt() {
        uint64_t kSCnt = static_cast<uint64_t>(this->tiling_->strideH);
        this->kSM_ = DivCeil(static_cast<uint64_t>(this->tiling_->hi), this->tiling_->strideH);
        if constexpr (kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
            kSCnt = 1;
            this->kSM_ *= DivCeil(static_cast<uint64_t>(this->tiling_->wi), this->tiling_->strideW);
        } else {	
            this->kSM_ *= this->tiling_->wi;	
        }
        this->mCnt_ = DivCeil(this->kSM_, this->singleShapeM_);
        this->mCoreTail_ = this->kSM_ - (this->mCnt_ - 1) * this->singleShapeM_;

        uint64_t n = this->tiling_->cin;
        this->nCnt_ = DivCeil(n, this->singleShapeN_);
        this->nCoreTail_ = n - (this->nCnt_ - 1) * this->singleShapeN_;

        if (this->singleShapeDin_ > 1) {
            this->dinCnt_ = DivCeil(this->tiling_->di, this->singleShapeDin_);
            this->dinCoreTail_ = this->tiling_->di - (this->dinCnt_ - 1) * this->singleShapeDin_;
        } else {
            this->dinCnt_ = this->tiling_->di;
            this->dinCoreTail_ = 1;
        }

        // 记录基本块的位置
        this->totalCnt_ = kSCnt * static_cast<uint64_t>(this->tiling_->batch) *
            this->dinCnt_ * this->mCnt_ * this->nCnt_;

        uint64_t blockNum = GetBlockNum();
        if (this->totalCnt_ < blockNum) {
            this->usedCoreNum_ = this->totalCnt_;
        } else {
            this->usedCoreNum_ = blockNum;
        }

        this->calRound_ = this->totalCnt_ / this->usedCoreNum_;
        this->tailCnt_ = this->totalCnt_ - this->calRound_ * this->usedCoreNum_;
    }

    __aicore__ inline void InitBasicBlockLoopDirect() {
        // 1.Kernel>1时右矩阵格式转换Bound,有效带宽不到1T,先沿着N走位;Kerenl=1沿着窄的方向走位
        // 2.M方向尽可能按照滑窗叠加的方向来分基本块，优先复用D的滑窗OverLap, 其次时H滑窗在L1边界的叠加
        if (this->tiling_->dk > 1) {
            this->loopDirect_ = LOOP_MDN;
        } else if (this->tiling_->hk / this->tiling_->strideH > 1) {
            this->loopDirect_ = LOOP_DMN;
        } else if (this->mCnt_ > this->nCnt_) {
            this->loopDirect_ = LOOP_DMN;
        } else {
            this->loopDirect_ = LOOP_DNM;
        }
    }

    __aicore__ inline void CalBasicBlockIdx(uint64_t basicBlockIdx) {
        uint64_t mnCnt = this->mCnt_ * this->nCnt_;
        uint64_t depthMNCnt = this->dinCnt_ * mnCnt;
        if constexpr (kernelSplitMode == TPL_SPLIT_KERNEL_H) {
            uint64_t batchDepthMNCnt = static_cast<uint64_t>(this->tiling_->batch) * depthMNCnt;
            this->kSCoreIdx_ = basicBlockIdx / batchDepthMNCnt;
            basicBlockIdx -= this->kSCoreIdx_ * batchDepthMNCnt;
        }
        this->batchCoreIdx_ = basicBlockIdx / depthMNCnt;
        basicBlockIdx -= this->batchCoreIdx_ * depthMNCnt;
        if (this->loopDirect_ == LOOP_MDN) {
            uint64_t depthNcnt = this->dinCnt_ * this->nCnt_;
            this->mCoreIdx_ = basicBlockIdx / depthNcnt;
            basicBlockIdx -= this->mCoreIdx_ * depthNcnt;
            this->dCoreIdx_ = basicBlockIdx / this->nCnt_;
            basicBlockIdx -= this->dCoreIdx_ * this->nCnt_;
            this->nCoreIdx_ = basicBlockIdx;
        } else if (this->loopDirect_ == LOOP_DMN) {
            this->dCoreIdx_ = basicBlockIdx / mnCnt;
            basicBlockIdx -= this->dCoreIdx_ * mnCnt;
            this->mCoreIdx_ = basicBlockIdx / this->nCnt_;
            basicBlockIdx -= this->mCoreIdx_ * this->nCnt_;
            this->nCoreIdx_ = basicBlockIdx;
        } else if (this->loopDirect_ == LOOP_DNM) {
            this->dCoreIdx_ = basicBlockIdx / mnCnt;
            basicBlockIdx -= this->dCoreIdx_ * mnCnt;
            this->nCoreIdx_ = basicBlockIdx / this->mCnt_;
            basicBlockIdx -= this->nCoreIdx_ * this->mCnt_;
            this->mCoreIdx_ = basicBlockIdx;
        }
        this->groupCoreIdx_ = 0;
    }

    static __aicore__ inline int32_t CalcSubKernelBackPadUp(int32_t kernelSize, int32_t stride,
        int32_t backPad, int32_t idx)
    {
        int32_t subKernelSize = (kernelSize + stride - idx - 1) / stride;
        int32_t subKernelBackPadUpMax = subKernelSize - 1;
        // 先去除第一行的影响，然后在此坐标上，计算错过了多少有效ho
        int32_t missPointNum = 0;
        int32_t curIdx = kernelSize - idx - backPad - 1;
        if (curIdx > 0) {
            missPointNum = (curIdx - 1) / stride + 1;
        }
        int32_t subKernelBackPadUp = subKernelBackPadUpMax - missPointNum;
        return subKernelBackPadUp;
    }

    __aicore__ inline void InitTilingData(const conv_bp_v2_kernel::Conv3DBackpropInputV2TilingData* tilingData) {
        Conv3dDx<filterType, filterFormat, dedyType, dedyFormat, yType, yFormat, biasType, biasFormat,
            b2Condition, kernelSplitMode, groupMode, b1Condition, enableC04Flag>::InitTilingData(tilingData);
        this->kSUseWorkSpace_ = tilingData->conv3DDxKSTiling.kSUseWorkSpace;
        this->dedx_.SetKernelSplitParams(tilingData->conv3DDxKSTiling.kSCoutFullLoad, this->kSUseWorkSpace_);
        this->singleShapeM_ = this->tiling_->singleCoreM;
        this->singleShapeK_ = static_cast<uint64_t>(this->tiling_->cout) * this->tiling_->hk * this->tiling_->wk;
        this->singleShapeN_ = this->tiling_->singleCoreCin;
        this->singleShapeDin_ = this->tiling_->singleCoreDin;
        this->enableVecTrans_ = this->tiling_->enableVecTrans;
        CalBasicBlockCnt();
        InitBasicBlockLoopDirect();
        this->InitBlockStride();
    }

    __aicore__ inline void CalcBlockKSOffset(int32_t kernelIdx)
    {
        uint32_t alignedHoStartIdx = this->curHoStartIdx_ < 0 ? 0 : this->curHoStartIdx_;
        this->offsetA_ += static_cast<uint64_t>(alignedHoStartIdx) * this->tiling_->wo;
        this->offsetB_ += static_cast<uint64_t>(this->kSCoreIdx_) * kernelSplitStrideB_;
        this->offsetC_ += static_cast<uint64_t>(kernelIdx) * this->tiling_->wi;
    }

    __aicore__ inline void InitKernelSplitStrideB()
    {
        if (this->enableVecTrans_) {
            kernelSplitStrideB_ = this->tiling_->wk *
                (((this->tiling_->cin + BLOCK_CUBE - 1) >> 4) << 4); // 4 : 2的4次方
        } else if (this->filterCubeFormat == Convolution3DBackprop::CubeFormat::NDHWC) {
            kernelSplitStrideB_ = this->tiling_->wk * this->tiling_->cin;
        } else {
            kernelSplitStrideB_ = this->tiling_->wk;
        }
    }

    __aicore__ inline void ReCalcMInfo(uint32_t kSTailIdx, uint32_t kernelIdx)
    {
        if (kSTailIdx != 0 && kernelIdx >= kSTailIdx) {
            this->kSTailMCnt_ = DivCeil(this->kSTailM_, this->singleShapeM_);
            this->mCoreTail_ = this->kSTailM_ - (this->kSTailMCnt_ - 1) * this->singleShapeM_;
        } else {
            this->mCoreTail_ = this->kSM_ - (this->mCnt_ - 1) * this->singleShapeM_;
        }
    }

    __aicore__ inline void SetIterateParams(uint64_t mCoreUse, uint32_t splitBaseHk,
        uint32_t baseHkIter, uint32_t splitTailHk, uint32_t kernelIdx)
    {
        uint64_t nCoreUse = (this->nCoreIdx_ == (this->nCnt_ - 1)) ? this->nCoreTail_ : this->singleShapeN_;
        uint64_t dinCoreUse = (this->dCoreIdx_ == (this->dinCnt_ - 1)) ? this->dinCoreTail_ : this->singleShapeDin_;
        uint32_t curSplitHk = this->kSCoreIdx_ < baseHkIter ? splitBaseHk : splitTailHk;
        int32_t curBackpropPadUp = CalcSubKernelBackPadUp(this->tiling_->hk, this->tiling_->strideH, this->tiling_->backpropPadUp, this->kSCoreIdx_);

        this->dedx_.SetSingleShapeParams(curSplitHk, curBackpropPadUp);
        this->dedx_.SetSingleShape(mCoreUse, this->singleShapeK_, nCoreUse, dinCoreUse);
        this->dedx_.SetStartIdx(this->dCoreIdx_ * this->singleShapeDin_, this->mCoreIdx_ * this->tiling_->singleCoreM,
            this->nCoreIdx_ * this->tiling_->singleCoreCin, 0);
        this->CalBasicBlockOffset();
        CalcBlockKSOffset(kernelIdx);
        this->dedx_.SetOutBackprop(this->dedyGm_[this->offsetA_]);
        this->dedx_.SetWeight(this->filterGm_[this->offsetB_]);
    }

    __aicore__ inline void CalBasicBlockCoreForSplitH(uint64_t blockIdx, uint64_t blockNum)
    {
        uint32_t splitBaseHk = DivCeil(this->tiling_->hk, this->tiling_->strideH);
        uint32_t splitTailHk = this->tiling_->hk / this->tiling_->strideH;
        uint32_t baseHkIter = splitTailHk == splitBaseHk ?
            this->tiling_->strideH : this->tiling_->hk - splitTailHk * this->tiling_->strideH;
        uint32_t kernelStartIdx = this->tiling_->padUp % this->tiling_->strideH;
        kernelStartIdx = kernelStartIdx == 0 ? 0 : this->tiling_->strideH - kernelStartIdx;
        uint32_t kSTailIdx = this->tiling_->hi - (this->tiling_->hi / this->tiling_->strideH) * this->tiling_->strideH;
        uint64_t mCoreUse = 0;
        this->kSTailM_ = (static_cast<uint64_t>(this->tiling_->hi) / this->tiling_->strideH) * this->tiling_->wi;
        this->CrossCoreWaitVecTrans();
        for (uint64_t j = 0; j < this->calRound_; ++j) {
            CalBasicBlockIdx(j * blockNum + blockIdx);
            uint32_t kernelIdx = this->kSCoreIdx_ + kernelStartIdx;
            if (kernelIdx >= this->tiling_->strideH) {
                // 等于strideH时，说明此时输出位置在首地址处
                kernelIdx = kernelIdx - (kernelIdx / this->tiling_->strideH) * this->tiling_->strideH;
            }
            ReCalcMInfo(kSTailIdx, kernelIdx);

            if (kSTailIdx != 0 && kernelIdx >= kSTailIdx) {
                if (this->mCoreIdx_ >= this->kSTailMCnt_) {
                    continue;   // 当切换子kernel后，mcnt可能会变小
                }
                mCoreUse = (this->mCoreIdx_ == (this->kSTailMCnt_ - 1)) ? this->mCoreTail_ : this->singleShapeM_;
            } else {
                mCoreUse = (this->mCoreIdx_ == (this->mCnt_ - 1)) ? this->mCoreTail_ : this->singleShapeM_;
            }
            this->SetIterateParams(mCoreUse, splitBaseHk, baseHkIter, splitTailHk, kernelIdx);
            this->dedx_.SetFullLoadFlag(this->tiling_->enableFullLoad);
            this->dedx_.IterateAll(this->yGm_[this->offsetC_], 0);  // 1 means atomic add
        }
    }

    __aicore__ inline void CalBasicBlock()
    {
        uint64_t blockIdx = GetAicBlockIdx();
        if (blockIdx < this->tailCnt_) {     // 拖尾的部分依次分配到前面的核计算，这些核会多算一轮
            ++this->calRound_;
        }

        uint64_t blockNum = this->usedCoreNum_;
        if constexpr (kernelSplitMode == TPL_SPLIT_KERNEL_H) {
            InitKernelSplitStrideB();
            CalBasicBlockCoreForSplitH(blockIdx, blockNum);
        } else {
            this->CalBasicBlockCore(blockIdx, blockNum);
            if ASCEND_IS_AIC {
                // dk等于1且B矩阵全载，则整个循环过程中B矩阵仅需要加载一次，在循环结束后释放
                this->dedx_.FreeB1Tensor();
            }
        }
    }
};
}

#endif // CONV3D_BACKPROP_INPUT_KERNEL_SPLIT_BLOCK_H