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
 * \file conv3d_dw_v2_basic_block.h
 * \brief
 */
#ifndef CONV3D_BACKPROP_FILTER_BASIC_BLOCK_H
#define CONV3D_BACKPROP_FILTER_BASIC_BLOCK_H

#include "basic_api/kernel_basic_intf.h"
#include "utils/std/algorithm.h"
#include "conv3d_bp_filter.h"
#include "kernel_type.h"
#include "conv3d_backprop_filter_v2.h"

constexpr uint32_t NO_STREAMK_CALC = 0;
constexpr uint32_t STREAMK_BATCHDOUT = 1;
constexpr uint32_t STREAMK_HWOUT = 2;
// DW累加轴大小阈值，支持按需修改大小，当前大小 256*256
constexpr uint64_t SINGLE_BLOCK_ADD_THRESHOLD = 65536;
constexpr uint64_t HUGE_PADDING_HO_THRESHOLD = 256;

namespace AscendC {
using Conv3ddwConfig = typename ConvolutionBackprop::Conv3ddwConfig;

template <typename xType, int xFormat, typename dedyType, int dedyFormat, typename yType, int yFormat, bool isSplitKernelHW>
class Conv3dDwBasicBlockMNStreamK : public Conv3dDw<xType, xFormat, dedyType, dedyFormat, yType, yFormat, isSplitKernelHW> {
public:
    __aicore__ inline Conv3dDwBasicBlockMNStreamK()
    {
    };
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dedy, GM_ADDR y, GM_ADDR workSpace,
        const conv_bp_v2_kernel::Conv3DBackpropFilterV2TilingData* tilingData)
    {
        InitCommonTilingData(tilingData);
        // init global buffer
        this->xGm_.SetGlobalBuffer((__gm__ xType*)x);
        this->dedyGm_.SetGlobalBuffer((__gm__ dedyType*)dedy);
        this->yGm_.SetGlobalBuffer((__gm__ yType*)y);
        this->dw_.Init(&(tilingData->dwTiling));
        // streamk场景，由于需要从l0c搬入GM，在从GM搬入UB,所以需要申请额外GM空间
        if (streamkType_ != NO_STREAMK_CALC) {
            this->workspaceGm_.SetGlobalBuffer((__gm__ float*)workSpace);
        }
    }

    __aicore__ inline void Process()
    {
        if (block_idx >= usedCoreNum_) {
            // vector GetBlockIdx() 与 Cube不同，此处注意使用全局变量准确
            return;
        }
        CalBasicBlock();
        this->dw_.End();
    }

protected:
    uint64_t mCnt_;
    uint64_t mCoreTail_;
    uint64_t nCnt_;
    uint64_t nCoreTail_;
    uint64_t ciCoreTail_;
    uint64_t batchDoutNcnt_;
    uint64_t totalCnt_;
    uint64_t tailCnt_;
    uint64_t calRound_;
    uint32_t deterAddCoreNum_ = 0;
    uint64_t usedCoreNum_;
    uint32_t streamkType_;
    uint64_t singleCoreBatch_; // 在基本块场景，singleCoreBatch用于表示单核batch
    uint64_t singleShapeK_;
    static constexpr uint64_t SYNC_MODE0 = 0;
    static constexpr uint64_t SYNC_MODE2 = 2;
    static constexpr uint16_t SYNC_AIC_ONLY_ALL_DET_FLAG = 4;
    static constexpr uint16_t SYNC_AIC_AIV_DET_FLAG = 8;
    static constexpr uint64_t CUBE_WORKSPACE = AscendC::TOTAL_L0C_SIZE >> 2;

    __aicore__ inline void CalBasicBlockCnt()
    {
        mCnt_ = Ceil(this->m_, this->singleShapeM_);
        mCoreTail_ = this->m_ - (mCnt_ - 1) * this->singleShapeM_;
        nCnt_ = Ceil(this->n_, this->singleShapeN_);
        nCoreTail_ = this->n_ - (nCnt_ - 1) * this->singleShapeN_;
        ciCoreTail_ = this->tiling_->cin1G - (nCnt_ - 1) * this->tiling_->singleCoreCin;
        uint64_t mnCnt = mCnt_ * nCnt_;
        totalCnt_ = mnCnt * this->tiling_->dk * this->tiling_->realGroup;
        batchDoutNcnt_ = nCnt_ * this->tiling_->dk * this->tiling_->realGroup;

        if (streamkType_ == NO_STREAMK_CALC) {
            calRound_ = Ceil(totalCnt_, usedCoreNum_);
            tailCnt_ = 0;
        } else {
            calRound_ = totalCnt_ / usedCoreNum_;
            tailCnt_ = totalCnt_ - calRound_ * usedCoreNum_;
        }
    }

    __aicore__ void InitCommonTilingData(const conv_bp_v2_kernel::Conv3DBackpropFilterV2TilingData* tilingData)
    {
        Conv3dDw<xType, xFormat, dedyType, dedyFormat, yType, yFormat, isSplitKernelHW>::InitTilingData(tilingData);
        usedCoreNum_ = tilingData->dwTiling.usedCoreNum;
        streamkType_ = tilingData->dwTiling.streamkType;
        singleCoreBatch_ = tilingData->dwTiling.singleCoreBatchDout;
        singleShapeK_ = tilingData->dwTiling.singleCoreK;
        CalBasicBlockCnt();
    }

    __aicore__ inline void SetDeterAddCore4StreamK(uint64_t batchDoutIdx, bool isNoDeter)
    {
        uint32_t batchCoreInx = batchDoutIdx / singleCoreBatch_;
        uint32_t deterAddCoreInx = this->kCoreIndx_ + batchCoreInx;
        uint32_t coreInx = block_idx / deterAddCoreNum_;
        uint32_t addCoreStartInx = coreInx * deterAddCoreNum_;
        this->offsetCubeWorkSpaceC_ = (deterAddCoreInx + coreInx * deterAddCoreNum_) * CUBE_WORKSPACE;
        this->dw_.SetDeterministicCoreInfo(deterAddCoreNum_, deterAddCoreInx, addCoreStartInx, isNoDeter);
    }

    __aicore__ inline void CalcStreamK(uint64_t basicBlockIdx)
    {
        if (tailCnt_ == 0) { return; }
        // 获取基本块的位置
        uint64_t batchDoutCnt = Ceil(static_cast<uint64_t>(this->tiling_->batch) * this->tiling_->dout, singleCoreBatch_);
        uint64_t batchDoutIdx = 0;
        uint64_t batchIdx = 0;
        uint64_t doutIdx = 0;
        uint64_t kCnt = Ceil(this->k_, singleShapeK_);
        this->kCoreIndx_ = 0;
        deterAddCoreNum_ = kCnt * batchDoutCnt;
        if (streamkType_ == STREAMK_HWOUT) {
            this->kCoreIndx_ = block_idx % deterAddCoreNum_;
        } else if (streamkType_ == STREAMK_BATCHDOUT) {
            batchDoutIdx = (block_idx % deterAddCoreNum_) * singleCoreBatch_;
            batchIdx = batchDoutIdx / this->tiling_->dout;
            doutIdx = batchDoutIdx - batchIdx * this->tiling_->dout;
        }
        uint64_t tailBlockIdx = basicBlockIdx / usedCoreNum_ * usedCoreNum_ + basicBlockIdx % usedCoreNum_ / deterAddCoreNum_;
        uint32_t groupIdx = tailBlockIdx % this->tiling_->realGroup;
        uint32_t dkIdx = (tailBlockIdx / this->tiling_->realGroup) % this->tiling_->dk;
        this->nCoreIndx_ = (tailBlockIdx / this->tiling_->realGroup / this->tiling_->dk) % nCnt_;
        this->mCoreIndx_ = (tailBlockIdx) / batchDoutNcnt_;
        this->cinCoreIndx_ = this->nCoreIndx_;
        // singleShapeBatch用于表示基本块的batch
        uint64_t batchDoutTail = static_cast<uint64_t>(this->tiling_->batch) * this->tiling_->dout - (batchDoutCnt - 1) * singleCoreBatch_;
        uint64_t batchdoutCurrentCore = ((batchDoutIdx / singleCoreBatch_) == (batchDoutCnt - 1)) ? batchDoutTail : singleCoreBatch_;

        this->singleShapeBatch_ = batchdoutCurrentCore;

        this->CalcOffset(batchIdx, doutIdx, groupIdx, dkIdx, (this->kCoreIndx_ * singleShapeK_) / this->tiling_->wo);
        this->ReCalDkCinSingleCoreShape(doutIdx, groupIdx, dkIdx);
        // 尾块的streamk场景，启动核可能比确定性计算的核多，因此在确定性计算中将不使用的核进行计算跳过，计算跳过也需要发送同步信号
        // isCompute代表该核不参与计算的基本块，isNoDeter代表该核跳过确定性计算
        bool isCompute = true;
        bool isNoDeter = block_idx >= kCnt * batchDoutCnt * tailCnt_;
        if (this->singleShapeNInCurrentHo_ == 0 || this->singleShapeMInCurrentHo_ == 0 || isNoDeter) {
            isCompute = false;
        } else {
            this->dw_.SetOutBackprop(this->dedyGm_[this->offsetA_]);
            this->dw_.SetStartIdx(batchDoutIdx, this->kCoreIndx_ * this->tiling_->singleCoreHo, dkIdx);
            this->dw_.SetFmap(this->xGm_[this->offsetB_]);
        }
        CalcSingleShape(kCnt, isCompute);
        CalcIterate(batchDoutIdx, batchdoutCurrentCore, groupIdx, dkIdx, isCompute, isNoDeter, kCnt);
    }

    __aicore__ inline void CalcSingleShape(uint64_t kCnt, bool isCompute)
    {
        uint32_t ciCoreUse = (this->nCoreIndx_ == (nCnt_ - 1)) ? ciCoreTail_ : this->tiling_->singleCoreCin;
        uint64_t mCoreUse = (this->mCoreIndx_ == (mCnt_ - 1)) ? mCoreTail_ : this->singleShapeM_;
        uint64_t nCoreUse = (this->nCoreIndx_ == (nCnt_ - 1)) ? nCoreTail_ : this->singleShapeN_;
        if (isCompute) {
            nCoreUse = nCoreUse > this->singleShapeNInCurrentHo_ ? this->singleShapeNInCurrentHo_ : nCoreUse;
            mCoreUse = mCoreUse > this->singleShapeMInCurrentHo_ ? this->singleShapeMInCurrentHo_ : mCoreUse;
        }
        if (this->tiling_->group > 1) {
            ciCoreUse = nCoreUse / (this->tiling_->hk * this->tiling_->wk);
        }
        uint64_t kCoreTail = this->k_ - (kCnt - 1) * singleShapeK_;
        kCoreTail = kCoreTail > this->tiling_->wo ? kCoreTail : this->tiling_->wo;
        uint64_t kCoreUse = ((this->kCoreIndx_ == (kCnt - 1)) && isCompute) ? kCoreTail : singleShapeK_;
        this->dw_.SetSingleShape(mCoreUse, nCoreUse, kCoreUse, ciCoreUse, this->singleShapeBatch_);
    }

    __aicore__ inline void CalcIterate(uint64_t batchDoutIdx, uint64_t batchdoutCurrentCore, uint32_t groupIdx,
        uint32_t dkIdx, bool isCompute, bool isNoDeter, uint64_t kCnt)
    {
        // 做STREAM K，有确定性计算
        SetDeterAddCore4StreamK(batchDoutIdx, isNoDeter);
        this->dw_.ctx.l0cPingPongFlag_ = 1;
        // 对于singleShape大于基本块场景，会有多轮计算
        uint64_t nCoreTailAlign = Ceil(ciCoreTail_, this->tiling_->n0) * this->tiling_->n0 * this->tiling_->hk * this->tiling_->wk;
        uint64_t maxMIter = Ceil(
            mCnt_ > 1 ? this->singleShapeM_ : mCoreTail_, this->tiling_->baseM);
        uint64_t maxNIter = Ceil(nCnt_ > 1 ? this->singleShapeN_ : nCoreTailAlign, this->tiling_->baseN);

        uint8_t barrierTriggerCnt = 0;
        constexpr uint8_t barrierThreshold = 14;
        for (uint64_t i = 0; i < maxMIter; i++) {
            for (uint64_t j = 0; j < maxNIter; j++) {
                bool isCurIter = (i < this->dw_.ctx.mIter_) && (j < this->dw_.ctx.nIter_);
                if ASCEND_IS_AIC {
                    CubeWaitVector();
                    if (isCompute && isCurIter) {
                        CalcIterateCube(batchDoutIdx, batchdoutCurrentCore, groupIdx, dkIdx, kCnt);
                    } else if (this->tiling_->cl0Pbuffer > 1) {
                        this->dw_.ctx.l0cPingPongFlag_ = !this->dw_.ctx.l0cPingPongFlag_;
                    }
                    if (!isCompute && ((++barrierTriggerCnt) % barrierThreshold == 0)) {
                        //如果全是跳过,那么cube核没有实际逻辑,导致cube核CrossCoreSetFlag过快,
                        //vector核来不及消费Flag，使得CrossCoreSetFlag内部的计数器溢出导致异常
                        //根据文档,CrossCoreSetFlag的计数器最多设置15次,所以每隔14次触发一次Barrier
                        //强制cube等待vector的notify,让两边同步
                        PipeBarrier<PIPE_ALL>();
                        barrierTriggerCnt = 0;
                    }
                    CubeNotifyVector();
                }
                if ASCEND_IS_AIV {
                    this->ClearWorkspace(this->workspaceGm_[this->offsetCubeWorkSpaceC_]);
                    VecNotifyCube();
                    this->dw_.Iterate();
                    VecWaitCube();
                    if (isCurIter) {
                        this->dw_.DeterministicReduceKInUb(this->yGm_[this->offsetC_], this->workspaceGm_);
                    }
                    if (this->tiling_->cl0Pbuffer > 1) {
                        this->dw_.ctx.l0cPingPongFlag_ = !this->dw_.ctx.l0cPingPongFlag_;
                    }
                }
            }
        }
    }

    __aicore__ inline void CalcIterateCube(uint64_t batchDoutIdx, uint64_t batchdoutCurrentCore, uint32_t groupIdx,
        uint32_t dkIdx, uint64_t kCnt)
    {
        uint64_t kCoreTail = this->k_ - (kCnt - 1) * singleShapeK_;
        uint64_t kCoreUse = this->kCoreIndx_ == (kCnt - 1) ? kCoreTail : singleShapeK_;
        uint64_t hoCoreUse = kCoreUse / this->tiling_->wo;

        uint64_t reduceSegmentsH = 0;
        uint64_t reduceSegmentsND = 0;
        CalcReduceSegments(batchdoutCurrentCore, hoCoreUse, this->tiling_->wo, reduceSegmentsH, reduceSegmentsND);

        this->singleShapeBatch_ = reduceSegmentsND;
        this->dw_.ctx.singleShapeBatch_ = this->singleShapeBatch_;
        ConvolutionBackpropFunc::Out2L1ScalarParams out2L1Params;
        ConvolutionBackpropFunc::Out2L1ScalarParams out2L1ParamsFollow;
        out2L1ParamsFollow.isLoad2L1A = true;
        out2L1ParamsFollow.isFreeAL1 = true;
        out2L1ParamsFollow.isLoad2L1B = true;
        out2L1ParamsFollow.isFreeBL1 = true;
        if (!this->dw_.UpdateMNIdx(out2L1Params)) {
            return;
        }

        uint64_t hoStartIdx = this->kCoreIndx_ * this->tiling_->singleCoreHo;

        for (uint64_t batchdout = 0; batchdout < batchdoutCurrentCore; batchdout += reduceSegmentsND) {
            uint64_t curbatchDoutIdx = batchDoutIdx + batchdout;
            uint64_t batchIdx = curbatchDoutIdx / this->tiling_->dout;
            uint64_t doutIdx = curbatchDoutIdx - batchIdx * this->tiling_->dout;
            if (batchdout + reduceSegmentsND > batchdoutCurrentCore) {
                this->singleShapeBatch_ = batchdoutCurrentCore - batchdout;
                this->dw_.ctx.singleShapeBatch_ = this->singleShapeBatch_;
            }

            for (uint64_t hoOffset = 0; hoOffset < hoCoreUse; hoOffset += reduceSegmentsH) {
                uint64_t hoIdx = hoStartIdx + hoOffset;

                this->CalcOffset(batchIdx, doutIdx, groupIdx, dkIdx, hoIdx);
                this->ReCalDkCinSingleCoreShape(doutIdx, groupIdx, dkIdx);
                // isCompute计算了singleShape的整个dout是否在pad内不需要计算
                // isComputeInner 计算了当前dout是否在pad内不需要计算
                bool isComputeInner = true;
                if (this->singleShapeNInCurrentHo_ == 0 || this->singleShapeMInCurrentHo_ == 0) {
                    isComputeInner = false;
                } else {
                    // 每次batchIdx，doutIdx改变，都需要重新计算offsetA，offsetB，但不会影响offsetC
                    this->dw_.SetOutBackprop(this->dedyGm_[this->offsetA_]);
                    this->dw_.SetStartIdx(curbatchDoutIdx, hoIdx, dkIdx);
                    this->dw_.SetFmap(this->xGm_[this->offsetB_]);
                    this->dw_.SetSingleShapeK(AscendC::Std::min(hoCoreUse - hoOffset, reduceSegmentsH) * this->tiling_->wo);
                }
                if (isComputeInner) {
                    this->dw_.Compute(out2L1Params);
                    // 1: enable atomic add; true: enable sequential write
                    this->dw_.GetTensorC(this->workspaceGm_[this->offsetCubeWorkSpaceC_], 1, true);
                }
                // K方向循环时，L1不驻留，每次都要重新搬入，M,N位置均不变，直接开始计算
                out2L1Params = out2L1ParamsFollow;
            }
        }
    }

    __aicore__ inline void CalBasicBlock()
    {
        uint64_t basicBlockIdx = block_idx;
        // 主轮的基本块为完整的singleShapeM、singleShapeN、singleShapeK， BatchDout内移，在L0C内累加
        for (uint64_t j = 0; j < calRound_; ++j) {
            if (basicBlockIdx >= totalCnt_) {
                return;
            }

            uint32_t dkIdx = (basicBlockIdx / this->tiling_->realGroup) % this->tiling_->dk;
            this->nCoreIndx_ = (basicBlockIdx / this->tiling_->realGroup / this->tiling_->dk) % nCnt_;
            this->mCoreIndx_ = basicBlockIdx / batchDoutNcnt_;
            this->cinCoreIndx_ = this->nCoreIndx_;
            uint32_t groupIdx = basicBlockIdx % this->tiling_->realGroup;

            CalSingleBasicBlock(dkIdx, groupIdx);
            basicBlockIdx += usedCoreNum_;
        }
        // streamkType != 0时，才有尾轮的确定性计算
        if (streamkType_ != NO_STREAMK_CALC) {
            CalcStreamK(basicBlockIdx);
        }
    }

    __aicore__ inline void CalSingleBasicBlock(int32_t dkIdx, uint32_t groupIdx)
    {
        uint64_t mCoreUse = (this->mCoreIndx_ == (mCnt_ - 1)) ? mCoreTail_ : this->singleShapeM_;
        uint64_t nCoreUse = (this->nCoreIndx_ == (nCnt_ - 1)) ? nCoreTail_ : this->singleShapeN_;
        uint64_t ciCoreUse = (this->nCoreIndx_ == (nCnt_ - 1)) ? ciCoreTail_ : this->tiling_->singleCoreCin;

        uint64_t totalBatchdout = static_cast<uint64_t>(this->tiling_->batch) * this->tiling_->dout;
        // 由于浮点数累加存在精度丢失，针对累加轴超大场景，对batch*dout做切分，在核内分单次计算singleShapeBatch_,完成后fixpip搬出到GM累加，通过多次搬出以降低累加轴

        uint64_t reduceSegmentsH = 0;
        uint64_t reduceSegmentsND = 0;
        CalcReduceSegments(totalBatchdout, this->tiling_->ho, this->tiling_->wo, reduceSegmentsH, reduceSegmentsND);

        for (uint64_t batchDoutIdx = 0; batchDoutIdx < totalBatchdout; batchDoutIdx += reduceSegmentsND) {
            uint64_t batchIdx = batchDoutIdx / this->tiling_->dout;
            uint64_t doutIdx = batchDoutIdx - batchIdx * this->tiling_->dout;
            this->singleShapeBatch_ = AscendC::Std::min(totalBatchdout - batchDoutIdx, reduceSegmentsND);

            for (uint64_t hoIdx = 0; hoIdx < this->tiling_->ho; hoIdx += reduceSegmentsH) {
                this->CalcOffset(batchIdx, doutIdx, groupIdx, dkIdx, hoIdx);
                this->ReCalDkCinSingleCoreShape(doutIdx, groupIdx, dkIdx);

                // AIV无需后续的计算，只需同步基本块basicBlockIdx
                if ASCEND_IS_AIV {
                    if (this->tiling_->group == this->tiling_->realGroup) {
                        continue;
                    }
                    if (GetSubBlockIdx() > 0) {
                        continue;
                    }
                }

                if (this->singleShapeNInCurrentHo_ == 0 || this->singleShapeMInCurrentHo_ == 0) {
                    continue;
                }

                uint64_t batchCoreUse = this->singleShapeBatch_;

                // group>1情况下的处理
                nCoreUse = nCoreUse > this->singleShapeNInCurrentHo_ ? this->singleShapeNInCurrentHo_ : nCoreUse;
                mCoreUse = mCoreUse > this->singleShapeMInCurrentHo_ ? this->singleShapeMInCurrentHo_ : mCoreUse;
                if (this->tiling_->group > 1) {
                    ciCoreUse = nCoreUse / (this->tiling_->hk * this->tiling_->wk);
                }

                this->dw_.SetOutBackprop(this->dedyGm_[this->offsetA_]);
                uint64_t singleShapeK = AscendC::Std::min(this->tiling_->ho - hoIdx, reduceSegmentsH) * this->tiling_->wo;
                this->dw_.SetSingleShape(mCoreUse, nCoreUse, singleShapeK, ciCoreUse, batchCoreUse);
                this->dw_.SetStartIdx(batchDoutIdx, hoIdx, dkIdx);
                this->dw_.SetFmap(this->xGm_[this->offsetB_]);
                this->dw_.IterateAll(this->yGm_[this->offsetC_], 1);
            }
        }
    }

    __aicore__ inline void CalcReduceSegments(uint64_t nd, uint64_t ho, uint64_t wo, uint64_t& segmentsH, uint64_t& segmentsND)
    {
        const uint64_t segmentsNDH = Ceil(SINGLE_BLOCK_ADD_THRESHOLD, wo);
        segmentsH = AscendC::Std::min(segmentsNDH, ho);
        
        // 对于SplitKernelHW分支，限制H轴切分的大小最大为256，使得padding不会超load3d的限制
        if constexpr (isSplitKernelHW) {
            segmentsH = AscendC::Std::min(segmentsH, HUGE_PADDING_HO_THRESHOLD);
        }

        segmentsND = AscendC::Std::min(Ceil(segmentsNDH, segmentsH), nd);
    }

    __aicore__ inline void CubeNotifyVector()
    {
#ifndef __CCE_KT_TEST__
        CrossCoreSetFlag<SYNC_MODE0, PIPE_FIX>(SYNC_AIC_ONLY_ALL_DET_FLAG);
        CrossCoreWaitFlag(SYNC_AIC_ONLY_ALL_DET_FLAG);
        CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_AIV_DET_FLAG);
#endif
    }

    __aicore__ inline void VecWaitCube()
    {
#ifndef __CCE_KT_TEST__
        CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE2>(SYNC_AIC_AIV_DET_FLAG);
#endif
    }

    __aicore__ inline void CubeWaitVector()
    {
#ifndef __CCE_KT_TEST__
        CrossCoreWaitFlag<SYNC_MODE2, PIPE_M>(ConvolutionBackpropFunc::SYNC_AIV_AIC_DET_FLAG);
#endif
    }

    static __aicore__ inline void VecNotifyCube()
    {
#ifndef __CCE_KT_TEST__
        static constexpr uint16_t SYNC_AIV_ONLY_ALL_DET_FLAG = 2;
        CrossCoreSetFlag<SYNC_MODE0, PIPE_MTE3>(SYNC_AIV_ONLY_ALL_DET_FLAG);
        CrossCoreWaitFlag(SYNC_AIV_ONLY_ALL_DET_FLAG);
        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(ConvolutionBackpropFunc::SYNC_AIV_AIC_DET_FLAG);
#endif
    }
};

template <typename xType, int xFormat, typename dedyType, int dedyFormat, typename yType, int yFormat, bool isSplitKernelHW>
class Conv3dDwBasicBlockStreamK : public Conv3dDwBasicBlockMNStreamK<xType, xFormat, dedyType, dedyFormat, yType, yFormat, isSplitKernelHW> {
public:
    __aicore__ inline Conv3dDwBasicBlockStreamK()
    {
    };

    __aicore__ inline void Process()
    {
        if (block_idx >= this->usedCoreNum_) {
            this->dw_.End();
            return;
        }
        this->CalcStreamK(block_idx);
        this->dw_.End();
    }
};
}

#endif // CONV3D_BACKPROP_FILTER_BASIC_BLOCK_H