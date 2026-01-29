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
#ifndef CONV3D_BACKPROP_INPUT_ROWC_BLOCK_ADVANCE_H
#define CONV3D_BACKPROP_INPUT_ROWC_BLOCK_ADVANCE_H

#include "conv3d_bp_input.h"
#include "basic_api/kernel_basic_intf.h"
#include "kernel_type.h"
#include "conv3d_backprop_input_v2.h"
#include "../../conv3d_backprop_input_v2_arch35_tiling_key.h"

namespace AscendC {
constexpr uint8_t LOOP_DNM = 1;
constexpr uint8_t LOOP_DMN = 2;
constexpr uint8_t LOOP_MDN = 3;
static constexpr uint8_t SYNC_MODE2 = 2;
static constexpr uint16_t SYNC_AIV_AIC_DET_FLAG = 6;

template <typename filterType, int filterFormat, typename dedyType, int dedyFormat, typename yType, int yFormat,
    typename biasType, int biasFormat,
    uint8_t b2Condition, uint8_t kernelSplitMode, uint8_t groupMode,
    uint8_t b1Condition = TPL_GM_TO_L1,
    bool enableC04Flag = false>
class Conv3dDxOswBlock : public Conv3dDx<filterType, filterFormat, dedyType, dedyFormat, yType, yFormat, biasType, biasFormat,
    b2Condition, kernelSplitMode, groupMode, b1Condition, enableC04Flag> {
public:
    __aicore__ inline Conv3dDxOswBlock() {};
    __aicore__ inline void Init(GM_ADDR filter, GM_ADDR dedy, GM_ADDR y, GM_ADDR workSpace,
                                const conv_bp_v2_kernel::Conv3DBackpropInputV2TilingData *tilingData)
    {
        if constexpr (!enableC04Flag && !groupMode) {
            if ASCEND_IS_AIV {
                return;
            }
        }

        InitTilingData(tilingData);
        if (this->enableVecTrans_) {
            this->filterGm_.SetGlobalBuffer((__gm__ filterType *)workSpace);
        } else {
            this->filterGm_.SetGlobalBuffer((__gm__ filterType *)filter);
        }
        this->dedyGm_.SetGlobalBuffer((__gm__ dedyType *)dedy);
        this->yGm_.SetGlobalBuffer((__gm__ yType *)y);
        this->dedx_.Init(&(tilingData->conv3DDxTiling));
    }

    __aicore__ inline void Process() {
        if constexpr (!enableC04Flag && !groupMode) {
            if ASCEND_IS_AIV {
                return;
            }
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

     __aicore__ inline void CrossCoreWaitVecTrans()
    {
        if (this->enableVecTrans_) {
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE2>(SYNC_AIV_AIC_DET_FLAG);
            }
        }
    }

    __aicore__ inline void ProcessForUnUsedCore()
    {
        CrossCoreWaitVecTrans();
        this->dedx_.End();
    }

    __aicore__ inline void CalBasicBlockCnt() {
        uint64_t m = static_cast<uint64_t>(this->tiling_->hi) * this->tiling_->wi;
        this->mCnt_ = DivCeil(m, this->singleShapeM_);
        this->mCoreTail_ = m - (this->mCnt_ - 1) * this->singleShapeM_;

        uint64_t n = this->tiling_->cinG;
        uint64_t tailN =  this->tiling_->cin - n * (this->tiling_->group - 1);
        this->nCnt_ = DivCeil(n, this->singleShapeN_);
        this->nTailCnt_ = DivCeil(tailN, this->singleShapeN_);
        this->nCoreTail_ = n - (this->nCnt_ - 1) * this->singleShapeN_;
        this->nGroupCoreTail_ = tailN % this->singleShapeN_;

        uint64_t k = static_cast<uint64_t>(this->tiling_->cout) * this->tiling_->hk * this->tiling_->wk;
        if constexpr (b1Condition == TPL_GM_TO_L1_NO_HK) {
            k = static_cast<uint64_t>(this->tiling_->cout) * this->tiling_->wk;
        } else if constexpr (b1Condition == TPL_GM_TO_L1_NO_HK_WK) {
            k = static_cast<uint64_t>(this->tiling_->cout);
        }
        // enlarge场景，cout可能无法被group整除，因此需要计算最后一组实际参与计算的k
        this->coutGroupTail_ = k - (this->tiling_->group - 1) * this->singleShapeK_;

        if (this->singleShapeDin_ > 1) {
            this->dinCnt_ = DivCeil(this->tiling_->di, this->singleShapeDin_);
            this->dinCoreTail_ = this->tiling_->di - (this->dinCnt_ - 1) * this->singleShapeDin_;
        } else {
            this->dinCnt_ = this->tiling_->di;
            this->dinCoreTail_ = 1;
        }

        // 记录基本块的位置
        this->totalCnt_ = static_cast<uint64_t>(this->tiling_->batch) * static_cast<uint64_t>(this->tiling_->group) * this->dinCnt_ * this->mCnt_ * this->nCnt_;

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
        } else if (this->tiling_->hk > 1) {
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
        if(unlikely(this->tiling_->group > 1)) {
            uint64_t groupDepthMNCnt = static_cast<uint64_t>(this->tiling_->group) * depthMNCnt;
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
    }

    __aicore__ inline void InitTilingData(const conv_bp_v2_kernel::Conv3DBackpropInputV2TilingData* tilingData) {
        Conv3dDx<filterType, filterFormat, dedyType, dedyFormat, yType, yFormat, biasType, biasFormat,
            b2Condition, kernelSplitMode, groupMode, b1Condition, enableC04Flag>::InitTilingData(tilingData);
        this->singleShapeM_ = this->tiling_->singleCoreM;
        if(unlikely(this->tiling_->group > 1)) {
            if constexpr (b1Condition == TPL_GM_TO_L1) {
                this->singleShapeK_ = static_cast<uint64_t>(this->tiling_->coutG) * this->tiling_->hk * this->tiling_->wk;
            } else if constexpr (b1Condition == TPL_GM_TO_L1_NO_HK) {
                this->singleShapeK_ = static_cast<uint64_t>(this->tiling_->coutG) * this->tiling_->wk;
            } else if constexpr (b1Condition == TPL_GM_TO_L1_NO_HK_WK) {
                this->singleShapeK_ = static_cast<uint64_t>(this->tiling_->coutG);
            }
        } else {
            if constexpr (b1Condition == TPL_GM_TO_L1) {
                this->singleShapeK_ = static_cast<uint64_t>(this->tiling_->cout) * this->tiling_->hk * this->tiling_->wk;
            } else if constexpr (b1Condition == TPL_GM_TO_L1_NO_HK) {
                this->singleShapeK_ = static_cast<uint64_t>(this->tiling_->cout) * this->tiling_->wk;
            } else if constexpr (b1Condition == TPL_GM_TO_L1_NO_HK_WK) {
                this->singleShapeK_ = static_cast<uint64_t>(this->tiling_->cout);
            }
        }
        this->singleShapeN_ = this->tiling_->singleCoreCin;
        this->singleShapeDin_ = this->tiling_->singleCoreDin;
        this->enableVecTrans_ = this->tiling_->enableVecTrans;
        this->CalBasicBlockCnt();
        this->InitBasicBlockLoopDirect();
        this->InitBlockStride();
    }

    __aicore__ inline void CalBasicBlockOffset() {
        // 当前A的偏移仍然要用到ho，在切w模板拓展后可以从API外部剥离对ho的感知
        this->curHoStartIdx_ = this->dedx_.ctx.curHoStartIdx_;
        this->CalcBlockOffset(this->batchCoreIdx_, this->groupCoreIdx_);
    }

    __aicore__ inline void CheckFullLoadEnable() {
         if (this->offsetB_ == this->preOffsetB_) {
             this->preEnableFullLoad = this->tiling_->enableFullLoad;
             return;
         }
         this->preOffsetB_ = this->offsetB_;
         // 代表B矩阵偏移变化且上一轮是全载
         if (this->tiling_->enableFullLoad == 1 && this->preEnableFullLoad == this->tiling_->enableFullLoad) {
             // 释放上一轮全载且后续不全载
             this->dedx_.FreeB1Tensor();
             const_cast<conv_bp_v2_kernel::TConv3DInputV2Tiling*>(this->tiling_)->enableFullLoad = 0;
         }
         this->preEnableFullLoad = this->tiling_->enableFullLoad;
     }

    __aicore__ inline void CalBasicBlockCore(uint64_t blockIdx, uint64_t blockNum) {
        for (uint64_t j = 0; j < this->calRound_; ++j) {
            this->CalBasicBlockIdx(j * blockNum + blockIdx);
            uint64_t mCoreUse = (this->mCoreIdx_ == (this->mCnt_ - 1)) ? this->mCoreTail_ : this->singleShapeM_;
            uint64_t nCoreUse = (this->nCoreIdx_ == (this->nCnt_ - 1)) ? this->nCoreTail_ : this->singleShapeN_;
            uint64_t dinCoreUse = (this->dCoreIdx_ == (this->dinCnt_ - 1)) ? this->dinCoreTail_ : this->singleShapeDin_;
            uint64_t coutCoreUse = this->singleShapeK_;
            if (this->tiling_->group > 1) {
                coutCoreUse = (this->groupCoreIdx_ == (this->tiling_->group - 1)) ? this->coutGroupTail_ : coutCoreUse;
                if (unlikely(this->tiling_->cin % this->tiling_->cinG != 0 && this->groupCoreIdx_ == (this->tiling_->group - 1))) {
                    if (this->nCoreIdx_ == this->nTailCnt_ - 1) {
                        nCoreUse = this->nGroupCoreTail_;
                    } else if (this->nCoreIdx_ > this->nTailCnt_ - 1) {
                        return;
                    }

                }
            }
            this->dedx_.SetSingleShape(mCoreUse, coutCoreUse, nCoreUse, dinCoreUse);
            this->dedx_.SetStartIdx(this->dCoreIdx_ * this->singleShapeDin_, this->mCoreIdx_ * this->tiling_->singleCoreM,
                this->nCoreIdx_ * this->tiling_->singleCoreCin, 0);
            this->CalBasicBlockOffset();
            this->dedx_.SetOutBackprop(this->dedyGm_[this->offsetA_]);
            this->dedx_.SetWeight(this->filterGm_[this->offsetB_]);

            this->CheckFullLoadEnable();
            this->dedx_.SetFullLoadFlag(this->tiling_->enableFullLoad);

            if (j == 0) {
                this->CrossCoreWaitVecTrans();
            }
            this->dedx_.IterateAll(this->yGm_[this->offsetC_], 0);  // 1 means atomic
        }
    }

    __aicore__ inline void CalBasicBlock() {
        uint64_t blockIdx = GetAicBlockIdx();
        // 拖尾的部分依次分配到前面的核计算，这些核会多算一轮
        if (blockIdx < tailCnt_) {
            ++calRound_;
        }

        uint64_t blockNum = this->usedCoreNum_;
        CalBasicBlockCore(blockIdx, blockNum);

        if ASCEND_IS_AIC {
            // 当b1全载且dk=1时，只需要加载一次b1，在循环结束后释放
            this->dedx_.FreeB1Tensor();
        }
    }
};
}

#endif // CONV3D_BACKPROP_INPUT_ROWC_BLOCK_ADVANCE_H