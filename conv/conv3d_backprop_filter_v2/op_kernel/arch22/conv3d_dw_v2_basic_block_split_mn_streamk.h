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
 * \file conv3d_dw_v2_basic_block_split_mn_streamk.h
 * \brief
 */

#ifndef CONV3D_BACKPROP_FILTER_BASIC_BLOCK_SPLIT_MN_STREAMK_H
#define CONV3D_BACKPROP_FILTER_BASIC_BLOCK_SPLIT_MN_STREAMK_H

#include "conv3d_dw_v2_basic_block.h"

namespace AscendC {
template <typename xType, int xFormat, typename dedyType, int dedyFormat, typename yType, int yFormat>
class Conv3dDwBasicBlockSplitMNStreamK
    : public Conv3dDwBasicBlock<xType, xFormat, dedyType, dedyFormat, yType, yFormat> {
public:
    __aicore__ inline Conv3dDwBasicBlockSplitMNStreamK(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dedy, GM_ADDR y, GM_ADDR workSpace,
                                const Conv3DBackpropFilterV2TilingData* tilingData)
    {
        this->InitCommonTilingData(tilingData);
        InitSplitTilingData(tilingData);
        // init global buffer
        this->xGm_.SetGlobalBuffer((__gm__ xType*)x);
        this->dedyGm_.SetGlobalBuffer((__gm__ dedyType*)dedy);
        this->yGm_.SetGlobalBuffer((__gm__ yType*)y);
        this->dw_.Init(&(tilingData->dwTiling));
        this->workspaceXGm_.SetGlobalBuffer((__gm__ xType*)workSpace);
#if defined(DETERMINISTIC_MODE) && DETERMINISTIC_MODE == 1
        this->workspaceGm_.SetGlobalBuffer((__gm__ yType*)workSpace);
#endif
    }

    __aicore__ inline void Process()
    {
        if (block_idx >= this->usedCoreNum_) {
            return;
        }
        if ASCEND_IS_AIV {
            if constexpr (xFormat == FORMAT_NCDHW) {
                this->InitTransdataBuffer();
            } else {
#if defined(DETERMINISTIC_MODE) && DETERMINISTIC_MODE == 1
                if (this->streamKType_ == NO_STREAMK_CALC) {
                    return;
                }
#else
                return;
#endif
            }
        }
#if defined(DETERMINISTIC_MODE) && DETERMINISTIC_MODE == 1
        if (this->streamKType_ != NO_STREAMK_CALC) {
            this->InitMixCoreBuffer();
        }
#endif
        CalBasicBlock();
        this->dw_.End();
    }

protected:
    static constexpr uint8_t SYNC_MODE0 = 0;
    static constexpr uint8_t SYNC_MODE2 = 2;
    uint32_t streamKType_ = NO_STREAMK_CALC;
    uint32_t coreStreamK_ = 0;
    uint64_t streamkWorkspaceSize_ = 0;
    uint32_t channelSize_ = 0;

    __aicore__ inline void InitSplitTilingData(const Conv3DBackpropFilterV2TilingData* tilingData)
    {
        this->streamKType_ = tilingData->basicBlockTiling.streamKType;
        this->coreStreamK_ = tilingData->basicBlockTiling.coreStreamK;
        this->streamkWorkspaceSize_ = static_cast<uint64_t>(tilingData->basicBlockTiling.singleCoreM) *
                                      static_cast<uint64_t>(tilingData->basicBlockTiling.singleCoreN);
        this->singleShapeM_ = tilingData->basicBlockTiling.singleCoreM;
        this->singleShapeK_ = this->k_;
        this->singleShapeN_ = tilingData->basicBlockTiling.singleCoreN;
        this->channelSize_ = tilingData->dwTiling.channelSize;
    }

    __aicore__ inline void CalBasicBlock()
    {
        uint32_t mCnt = DivCeil(this->m_, this->singleShapeM_);
        uint32_t mCoreTail = this->m_ - (mCnt - 1) * this->singleShapeM_;

        uint64_t nCnt = DivCeil(this->n_, this->singleShapeN_);
        uint64_t nCoreTail = this->n_ - (nCnt - 1) * this->singleShapeN_;

        // 记录基本块的位置
        uint64_t batchDout = static_cast<uint64_t>(this->batch_) * this->dout_;
        uint64_t groupCnt = this->seperateDk_ ? static_cast<uint64_t>(this->group_) : 1UL;
        uint64_t dkCnt = this->seperateDk_ ? static_cast<uint64_t>(this->dk_) : 1UL;
        uint64_t mnCnt = mCnt * nCnt;
#if defined(DETERMINISTIC_MODE) && DETERMINISTIC_MODE == 1
        uint64_t totalCnt = mnCnt * groupCnt * dkCnt;
        uint64_t calRound = totalCnt / this->usedCoreNum_;
        uint64_t tailCnt = totalCnt - calRound * this->usedCoreNum_;
        uint64_t basicBlockIdx = 0;

        if (this->streamKType_ == NO_STREAMK_CALC) {
            if (block_idx < tailCnt) {
                basicBlockIdx = block_idx * calRound + block_idx;
                ++calRound;
            } else {
                basicBlockIdx = block_idx * calRound + tailCnt;
            }
        } else {
            basicBlockIdx = block_idx * calRound;
        }

        uint64_t batchDoutIndex = 0;
        this->kCoreIndx_ = 0;
        if ASCEND_IS_AIC {
            for (uint64_t j = 0; j < calRound; ++j) {
                if (basicBlockIdx >= totalCnt) {
                    break;
                }
                uint64_t groupDkIdx = basicBlockIdx / mnCnt;
                uint64_t localIdx = basicBlockIdx - groupDkIdx * mnCnt;
                uint32_t groupIdx = static_cast<uint32_t>(groupDkIdx % groupCnt);
                uint32_t dkIdx = static_cast<uint32_t>(groupDkIdx / groupCnt);
                if (this->coreBindOrder_ == ROW_FIRST) {
                    this->mCoreIndx_ = localIdx / nCnt;
                    this->nCoreIndx_ = localIdx - this->mCoreIndx_ * nCnt;
                } else {
                    uint64_t nIndex = localIdx / mCnt;
                    this->mCoreIndx_ = localIdx - nIndex * mCnt;
                    this->nCoreIndx_ = nIndex % nCnt;
                }
                basicBlockIdx++;

                uint64_t mCoreUse = (this->mCoreIndx_ == (mCnt - 1)) ? mCoreTail : this->singleShapeM_;
                uint64_t nCoreUse = (this->nCoreIndx_ == (nCnt - 1)) ? nCoreTail : this->singleShapeN_;

                for (batchDoutIndex = 0; batchDoutIndex < batchDout; ++batchDoutIndex) {
                    uint64_t batchIdx = batchDoutIndex / this->dout_;
                    uint64_t doutIdx = batchDoutIndex - batchIdx * this->dout_;
                    this->CalcOffset(batchIdx, doutIdx, groupIdx, dkIdx, true);
                    this->ReCalDkCinSingleCoreShape(batchIdx, doutIdx, groupIdx, dkIdx);
                    if (this->singleShapeNInCurrentHo_ == 0 || this->singleShapeMInCurrentHo_ == 0) {
                        continue;
                    }
                    uint64_t nCoreUseCur = nCoreUse > this->singleShapeNInCurrentHo_ ? this->singleShapeNInCurrentHo_ :
                                                                                       nCoreUse;
                    this->dw_.SetOutBackprop(this->dedyGm_[this->offsetA_]);
                    this->hoStartIdx_ = 0;
                    this->dw_.SetSingleShape(mCoreUse, nCoreUseCur, this->singleShapeK_, this->hoStartIdx_);
                    this->dw_.SetFmap(this->xGm_[this->offsetB_]);
                    this->dw_.IterateAll(this->yGm_[this->offsetC_], 1);
                }
            }
        }
#else
        uint64_t batchDoutMnCnt = batchDout * mnCnt;
        uint64_t totalCnt = batchDoutMnCnt * groupCnt * dkCnt;
        uint64_t calRound = totalCnt / this->usedCoreNum_;
        uint64_t tailCnt = totalCnt - calRound * this->usedCoreNum_;
        uint64_t basicBlockIdx = 0;

        if (this->streamKType_ == NO_STREAMK_CALC) {
            // 拖尾的部分依次分配到前面的核计算，这些核会多算一轮
            if (block_idx < tailCnt) {
                basicBlockIdx = block_idx * calRound + block_idx;
                ++calRound;
            } else {
                basicBlockIdx = block_idx * calRound + tailCnt;
            }
        } else {
            // streamK 只处理尾轮，主轮只做完整轮次
            basicBlockIdx = block_idx * calRound;
        }

        // 1:M*K 行优先绑核，2:M*K 列优先绑核
        uint64_t batchDoutIndex = 0;
        uint64_t batchDoutNcnt = batchDout * nCnt;
        this->kCoreIndx_ = 0; // 默认不分核
        uint64_t syncTimes = 0;
        for (uint64_t j = 0; j < calRound; ++j) {
            if (basicBlockIdx >= totalCnt) {
                break;
            }
            uint64_t groupDkIdx = this->seperateDk_ ? (basicBlockIdx / batchDoutMnCnt) : 0;
            uint64_t localIdx = this->seperateDk_ ? (basicBlockIdx - groupDkIdx * batchDoutMnCnt) : basicBlockIdx;
            uint32_t groupIdx = this->seperateDk_ ? static_cast<uint32_t>(groupDkIdx % groupCnt) : 0U;
            uint32_t dkIdx = this->seperateDk_ ? static_cast<uint32_t>(groupDkIdx / groupCnt) : 0U;
            if (this->coreBindOrder_ == ROW_FIRST) {
                // 行优先, NDC1HWC0的行方向是C0即N方向
                this->mCoreIndx_ = localIdx / batchDoutNcnt;
                batchDoutIndex = (localIdx - this->mCoreIndx_ * batchDoutNcnt) / nCnt;
                this->nCoreIndx_ = localIdx - this->mCoreIndx_ * batchDoutNcnt - batchDoutIndex * nCnt;
            } else {
                // 列优先, NDC1HWC0的列方向是Cout方向
                uint64_t batchDoutNIndex = localIdx / mCnt;
                this->mCoreIndx_ = localIdx - batchDoutNIndex * mCnt;
                this->nCoreIndx_ = batchDoutNIndex % nCnt;
                batchDoutIndex = localIdx / mnCnt;
            }
            uint64_t batchIdx = batchDoutIndex / this->dout_;
            uint64_t doutIdx = batchDoutIndex - batchIdx * this->dout_;

            basicBlockIdx++;

            // 不可用totalCnt - 1作为尾块, totalCnt包含batch*dout
            uint64_t mCoreUse = (this->mCoreIndx_ == (mCnt - 1)) ? mCoreTail : this->singleShapeM_;
            uint64_t nCoreUse = (this->nCoreIndx_ == (nCnt - 1)) ? nCoreTail : this->singleShapeN_;

            this->CalcOffset(batchIdx, doutIdx, groupIdx, dkIdx, true);
            this->ReCalDkCinSingleCoreShape(batchIdx, doutIdx, groupIdx, dkIdx);
            if (this->singleShapeNInCurrentHo_ == 0 || this->singleShapeMInCurrentHo_ == 0) {
                continue;
            }
            nCoreUse = nCoreUse > this->singleShapeNInCurrentHo_ ? this->singleShapeNInCurrentHo_ : nCoreUse;
            this->dw_.SetOutBackprop(this->dedyGm_[this->offsetA_]);
            this->hoStartIdx_ = this->kCoreIndx_ * this->singleCoreHo_;
            this->dw_.SetSingleShape(mCoreUse, nCoreUse, this->singleShapeK_, this->hoStartIdx_);
            this->dw_.SetFmap(this->xGm_[this->offsetB_]);
            this->dw_.IterateAll(this->yGm_[this->offsetC_], 1);
        }
#endif

        if (this->streamKType_ != NO_STREAMK_CALC) {
            CalStreamK(totalCnt, calRound, tailCnt, batchDout, mCnt, nCnt, mCoreTail, nCoreTail);
        }
    }

    __aicore__ inline void CalStreamK(uint64_t totalCnt, uint64_t calRound, uint64_t tailCnt, uint64_t batchDout,
                                      uint64_t mCnt, uint64_t nCnt, uint64_t mCoreTail, uint64_t nCoreTail)
    {
        if (tailCnt == 0 || this->coreStreamK_ == 0) {
            return;
        }

        if (this->streamKType_ != STREAMK_BATCHDOUT && this->streamKType_ != STREAMK_HWOUT) {
            return;
        }

        uint64_t groupCnt = this->seperateDk_ ? static_cast<uint64_t>(this->group_) : 1UL;
        uint64_t dkCnt = this->seperateDk_ ? static_cast<uint64_t>(this->dk_) : 1UL;
        uint64_t mnCnt = mCnt * nCnt;
        uint64_t streamKCoreNum = static_cast<uint64_t>(tailCnt) * this->coreStreamK_;
        bool isParticipant = block_idx < streamKCoreNum;

        uint64_t tailBlockIdx = static_cast<uint64_t>(block_idx) / this->coreStreamK_;
        uint64_t splitIdx = static_cast<uint64_t>(block_idx) % this->coreStreamK_;
        uint64_t basicBlockIdx = calRound * this->usedCoreNum_ + tailBlockIdx;
        if (basicBlockIdx >= totalCnt) {
            isParticipant = false;
        }

        uint64_t groupDkIdx = basicBlockIdx / mnCnt;
        uint64_t localIdx = basicBlockIdx - groupDkIdx * mnCnt;
        uint32_t groupIdx = static_cast<uint32_t>(groupDkIdx % groupCnt);
        uint32_t dkIdx = static_cast<uint32_t>(groupDkIdx / groupCnt);

        uint64_t batchIdx = 0;
        uint64_t doutIdx = 0;
        uint64_t mCoreUse = 0;
        uint64_t nCoreUse = 0;
        uint64_t kCoreUse = 0;
        uint64_t savedSingleShapeK = this->singleShapeK_;
        uint64_t streamSingleShapeK = 0;
        uint64_t streamSingleShapeBatchDout = 0;
        uint64_t splitCnt = 0;
        uint64_t coreUse = 0;

        if (isParticipant) {
            if (this->coreBindOrder_ == ROW_FIRST) {
                this->mCoreIndx_ = localIdx / nCnt;
                this->nCoreIndx_ = localIdx - this->mCoreIndx_ * nCnt;
            } else {
                uint64_t nIndex = localIdx / mCnt;
                this->mCoreIndx_ = localIdx - nIndex * mCnt;
                this->nCoreIndx_ = nIndex % nCnt;
            }

            mCoreUse = (this->mCoreIndx_ == (mCnt - 1)) ? mCoreTail : this->singleShapeM_;
            nCoreUse = (this->nCoreIndx_ == (nCnt - 1)) ? nCoreTail : this->singleShapeN_;

            if (this->streamKType_ == STREAMK_BATCHDOUT) {
                streamSingleShapeBatchDout = DivCeil(batchDout, static_cast<uint64_t>(this->coreStreamK_));
                splitCnt = DivCeil(batchDout, streamSingleShapeBatchDout);
                if (splitIdx >= splitCnt) {
                    isParticipant = false;
                } else {
                    coreUse = (splitIdx == (splitCnt - 1)) ? (batchDout - (splitCnt - 1) * streamSingleShapeBatchDout) :
                                                             streamSingleShapeBatchDout;
                    if (coreUse == 0) {
                        isParticipant = false;
                    }
                }
                kCoreUse = this->singleShapeK_;
            } else {
                streamSingleShapeK = DivCeil(this->singleShapeK_, static_cast<uint64_t>(this->coreStreamK_));
                streamSingleShapeK = DivCeil(streamSingleShapeK, static_cast<uint64_t>(this->wo_)) * this->wo_;
                if (streamSingleShapeK == 0) {
                    isParticipant = false;
                } else {
                    splitCnt = DivCeil(this->singleShapeK_, streamSingleShapeK);
                    if (splitIdx >= splitCnt) {
                        isParticipant = false;
                    } else {
                        kCoreUse = (splitIdx == (splitCnt - 1)) ?
                                       (this->singleShapeK_ - (splitCnt - 1) * streamSingleShapeK) :
                                       streamSingleShapeK;
                        if (kCoreUse == 0) {
                            isParticipant = false;
                        } else {
                            kCoreUse = kCoreUse > this->wo_ ? kCoreUse : this->wo_;
                        }
                    }
                }
            }
        }

        if (isParticipant) {
            if (this->streamKType_ == STREAMK_BATCHDOUT) {
                this->singleCoreHo_ = this->singleShapeK_ / this->wo_;
                this->kCoreIndx_ = 0;
                this->hoStartIdx_ = 0;
            } else {
                this->singleCoreHo_ = streamSingleShapeK / this->wo_;
                this->singleShapeK_ = streamSingleShapeK;
                this->kCoreIndx_ = static_cast<uint32_t>(splitIdx);
                this->hoStartIdx_ = (splitIdx * streamSingleShapeK) / this->wo_;
            }
#if defined(DETERMINISTIC_MODE) && DETERMINISTIC_MODE == 1
            if ASCEND_IS_AIV {
                this->CalcOffset(0, 0, groupIdx, dkIdx, true);
                InitStreamKWorkspace(this->streamkWorkspaceSize_, static_cast<uint64_t>(block_idx));
                CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(this->SYNC_AIV_AIC_FLAG);
            }
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<SYNC_MODE2, PIPE_FIX>(this->SYNC_AIV_AIC_FLAG);
            }
#endif
            if ASCEND_IS_AIC {
                uint8_t enAtomic = 0;
                uint64_t loopCount = (this->streamKType_ == STREAMK_BATCHDOUT) ? coreUse : batchDout;
                for (uint64_t index = 0; index < loopCount; ++index) {
                    if (this->streamKType_ == STREAMK_BATCHDOUT) {
                        uint64_t globalBatchDoutIndex = splitIdx * streamSingleShapeBatchDout + index;
                        batchIdx = globalBatchDoutIndex / this->dout_;
                        doutIdx = globalBatchDoutIndex - batchIdx * this->dout_;
                    } else {
                        batchIdx = index / this->dout_;
                        doutIdx = index - batchIdx * this->dout_;
                    }
                    this->CalcOffset(batchIdx, doutIdx, groupIdx, dkIdx, true);

                    this->ReCalDkCinSingleCoreShape(batchIdx, doutIdx, groupIdx, dkIdx);
                    if (this->singleShapeNInCurrentHo_ == 0 || this->singleShapeMInCurrentHo_ == 0) {
                        continue;
                    }
                    uint64_t nCoreUseCur = nCoreUse > this->singleShapeNInCurrentHo_ ? this->singleShapeNInCurrentHo_ :
                                                                                       nCoreUse;
                    this->dw_.SetOutBackprop(this->dedyGm_[this->offsetA_]);
                    this->dw_.SetSingleShape(mCoreUse, nCoreUseCur, kCoreUse, this->hoStartIdx_);
                    this->dw_.SetFmap(this->xGm_[this->offsetB_]);
#if defined(DETERMINISTIC_MODE) && DETERMINISTIC_MODE == 1
                    this->offsetWorkspaceC_ = block_idx * this->streamkWorkspaceSize_;
                    this->dw_.IterateAllDeterministic(this->workspaceGm_[this->offsetWorkspaceC_], enAtomic);
                    enAtomic = 1U;
#else
                    this->dw_.IterateAll(this->yGm_[this->offsetC_], 1);
#endif
                }
            }

            this->singleShapeK_ = savedSingleShapeK;
        }

#if defined(DETERMINISTIC_MODE) && DETERMINISTIC_MODE == 1
        if ASCEND_IS_AIC {
            CrossCoreSetFlag<SYNC_MODE0, PIPE_FIX>(this->SYNC_AIC_ONLY_ALL_FLAG);
            CrossCoreWaitFlag<SYNC_MODE0, PIPE_FIX>(this->SYNC_AIC_ONLY_ALL_FLAG);
        }

        if (isParticipant) {
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(this->SYNC_AIC_AIV_FLAG);
            }

            if ASCEND_IS_AIV {
                if (mCnt == 1) {
                    ReduceStreamKToY<false>(mCoreUse, nCoreUse, tailBlockIdx);
                } else {
                    ReduceStreamKToY<true>(mCoreUse, nCoreUse, tailBlockIdx);
                }
            }
        }
#endif
    }

#if defined(DETERMINISTIC_MODE) && DETERMINISTIC_MODE == 1
    __aicore__ inline void InitStreamKWorkspace(uint64_t blockSize, uint64_t coreIdx)
    {
        bool isSingleCore = (blockSize <= MIN_DATA_SIZE);
        if (GetSubBlockIdx() != 0 && isSingleCore) {
            return;
        }

        uint64_t taskDataSize = isSingleCore ? blockSize : blockSize >> 1;
        uint64_t baseOffset = coreIdx * blockSize + GetSubBlockIdx() * taskDataSize;
        taskDataSize += GetSubBlockIdx() * (blockSize & 1);

        LocalTensor<yType> ubSrc = this->tmpBuf_.template Get<yType>();
        uint64_t perDataSize = taskDataSize < this->dataSize_ ? taskDataSize : this->dataSize_;
        uint64_t processed = 0;
        TEventID eventIdVToMte3 = GetTPipePtr()->FetchEventID<HardEvent::V_MTE3>();

        AscendC::Duplicate<yType>(ubSrc, 0.0f, perDataSize);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        while (processed < taskDataSize) {
            uint64_t curSize = (taskDataSize - processed) < perDataSize ? (taskDataSize - processed) : perDataSize;
            DataCopy(this->workspaceGm_[baseOffset + processed], ubSrc, curSize);
            processed += curSize;
        }
    }

    template <bool isSplitM = false>
    __aicore__ inline void ReduceStreamKToY(uint64_t mCoreUse, uint64_t nCoreUse, uint64_t tailBlockIdx)
    {
        if (mCoreUse == 0 || nCoreUse == 0) {
            return;
        }

        uint64_t reducerCore = tailBlockIdx * this->coreStreamK_;
        uint32_t maxVecCore = this->coreStreamK_ * VEC_CUBE_RATIO;
        uint64_t totalSize;
        uint32_t useVecCore;
        uint32_t localVecIdx = (block_idx - reducerCore) * VEC_CUBE_RATIO + GetSubBlockIdx();
        uint64_t alignedM = this->cout1G_ * this->channelSize_;
        uint64_t curOffsetC = this->offsetC_;
        uint64_t vecOffsetC = 0;
        uint32_t offsetBlock = 0;

        uint32_t totalBlockNum = Ceil(nCoreUse, this->channelSize_);
        useVecCore = totalBlockNum > maxVecCore ? maxVecCore : totalBlockNum;
        if (localVecIdx >= useVecCore) {
            return;
        }
        totalSize = totalBlockNum / useVecCore;
        offsetBlock = totalSize * localVecIdx;
        uint32_t tailVecNum = totalBlockNum % useVecCore;
        if (tailVecNum != 0) {
            offsetBlock += localVecIdx < tailVecNum ? localVecIdx : tailVecNum;
            totalSize += localVecIdx < tailVecNum ? 1 : 0;
        }
        totalSize = totalSize * Ceil(mCoreUse, BLOCK_CUBE) * BLOCK_CUBE * this->channelSize_;
        curOffsetC += offsetBlock * alignedM * this->channelSize_;
        vecOffsetC = offsetBlock * mCoreUse * this->channelSize_;

        LocalTensor<yType> ubSrc1 = this->tmpBuf_.template Get<yType>();
        LocalTensor<yType> ubSrc2 = ubSrc1[this->dataSize_];
        uint64_t processed = 0;
        uint64_t baseCore = tailBlockIdx * this->coreStreamK_;
        uint64_t perDataSize = !isSplitM ? this->dataSize_ :
                                           Ceil(mCoreUse, BLOCK_CUBE) * BLOCK_CUBE * this->channelSize_;
        TEventID eventIdMte2ToV = GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>();
        TEventID eventIdVToMte2 = GetTPipePtr()->FetchEventID<HardEvent::V_MTE2>();

        TEventID eventIdMte3ToMte2 = GetTPipePtr()->FetchEventID<HardEvent::MTE3_MTE2>();
        TEventID eventIdVToMte3 = GetTPipePtr()->FetchEventID<HardEvent::V_MTE3>();

        CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE2>(this->SYNC_AIC_AIV_FLAG);

        while (processed < totalSize) {
            uint64_t curSize = (totalSize - processed) < perDataSize ? (totalSize - processed) : perDataSize;
            DataCopy(ubSrc1, this->workspaceGm_[baseCore * this->streamkWorkspaceSize_ + vecOffsetC + processed],
                     curSize);
            for (uint32_t coreIdx = 1; coreIdx < this->coreStreamK_; ++coreIdx) {
                uint64_t srcCore = baseCore + coreIdx;
                uint64_t srcOffset = srcCore * this->streamkWorkspaceSize_ + vecOffsetC + processed;
                DataCopy(ubSrc2, this->workspaceGm_[srcOffset], curSize);
                SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                Add<yType>(ubSrc1, ubSrc1, ubSrc2, curSize);
                SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
                WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            }
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            DataCopy(this->yGm_[curOffsetC], ubSrc1, curSize);

            processed += curSize;
            if constexpr (!isSplitM) {
                curOffsetC += curSize;
            } else {
                curOffsetC += alignedM * this->channelSize_;
            }

            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
    }
#endif
};
} // namespace AscendC

#endif
