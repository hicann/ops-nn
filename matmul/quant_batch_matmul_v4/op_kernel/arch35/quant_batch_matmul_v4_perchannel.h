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
 * \file quant_batch_matmul_v4_perchannel.h
 * \brief
 */

#ifndef QUANT_BATCH_MATMUL_V4_PERCHANNEL_H
#define QUANT_BATCH_MATMUL_V4_PERCHANNEL_H

#include "quant_batch_matmul_v4_reg_base_common.h"
#include "quant_batch_matmul_v4_tiling_data.h"

using matmul::MatmulImpl;

namespace QuantBatchMatmulV4
{

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz = false>
class QuantBatchMatmulV4PerChannelKernel
    : public QuantBatchMatmulV4RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset,
                                                   antiQuantType, scaleType, weightNz>
{
public:
    __aicore__ inline QuantBatchMatmulV4PerChannelKernel(){};
    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale, GM_ADDR x2_scale, GM_ADDR y_scale, GM_ADDR x1_offset,
        GM_ADDR x2_offset, GM_ADDR y_offset, GM_ADDR y, GM_ADDR workspace,
        const qbmmv4_tiling::QuantBatchMatmulV4TilingDataParams* tilingData, TPipe* tPipe);
    __aicore__ inline void CopyND2NZ(int aL1Idx);
    __aicore__ inline void GetAL1(int64_t kFactorIdx, AscendC::TEventID eventIdsMte1ToMte2[MAX_AL1_BUF_NUM]);
    __aicore__ inline void GetBL1(int64_t kFactorIdx);
    __aicore__ inline void IterateMatmul(int64_t kFactorIdx);
    __aicore__ inline void Process();

    using inputXType = MatmulL1GmType<TPosition::TSCM, CubeFormat::NZ, xType, aTrans>;
    using inputWType = MatmulL1GmType<TPosition::TSCM, CubeFormat::NZ, xType, bTrans>;
    using outputYType = MatmulType<TPosition::GM, CubeFormat::ND, yType>;
    using inputBiasType = MatmulType<TPosition::GM, CubeFormat::ND, biasType>;
    MatmulImpl<inputXType, inputWType, outputYType, inputBiasType, CFG_MDL> mmObj_;
};

template <
    typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
    bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline void QuantBatchMatmulV4PerChannelKernel<
    xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType, scaleType, weightNz>::
    Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale, GM_ADDR x2_scale, GM_ADDR y_scale, GM_ADDR x1_offset,
        GM_ADDR x2_offset, GM_ADDR y_offset, GM_ADDR y, GM_ADDR workspace,
        const qbmmv4_tiling::QuantBatchMatmulV4TilingDataParams* tilingData, TPipe* tPipe)
{
    this->tiling_ = tilingData;
    this->pipe_ = tPipe;

    if ASCEND_IS_AIV {
        this->curBlockIdx_ = GetBlockIdx() / GetTaskRation();
    } else {
        this->curBlockIdx_ = GetBlockIdx();
    }
    this->nDimIdx_ = this->curBlockIdx_ / this->tiling_->cubeNumBlocksM;
    this->mDimIdx_ = this->curBlockIdx_ % this->tiling_->cubeNumBlocksM;

    this->InitTilingData();

    this->InitInputs(x1, x2, x1_scale, x2_scale, x2_offset, y_scale, y_offset, bias, y, workspace);
    this->l1Local_ = this->l1Tbuf_.template Get<xType>();

    if ASCEND_IS_AIC {
        if (this->tiling_->BL1Pingpong == this->QUADRUPLE_BUFFER) {
            // L1 布局如下：
            // 1) AL1 使能 4 buffer 时
            //    L1 (0~256KB):   BL1_P0 | BL1_P2 | AL1_P1 | AL1_P3
            //    L1 (257~512KB): BL1_P1 | BL1_P3 | AL1_P0 | AL1_P2
            // 2) AL1 使能 2 buffer 时
            //    L1 (0~256KB):   BL1_P0 | BL1_P2 | AL1_P1
            //    L1 (257~512KB): BL1_P1 | BL1_P3 | AL1_P0
            // 3) AL1 使能 1 buffer 时
            //    L1 (0~512KB):   BL1_P0 | BL1_P2 | AL1_P0 | BL1_P1 | BL1_P3
            int32_t bL1DataSizeTotal = this->DOUBLE_BUFFER * this->bL1DataSize_ * sizeof(xType);
            if (this->tiling_->AL1Pingpong == this->QUADRUPLE_BUFFER) {
                this->bL1LocalBuf0_ = this->l1Tbuf_.template GetWithOffset<xType>(this->bL1DataSize_, 0);
                this->bL1LocalBuf2_ =
                    this->l1Tbuf_.template GetWithOffset<xType>(this->bL1DataSize_, this->bL1DataSize_ * sizeof(xType));
                this->aL1LocalBuf0_ = this->l1Tbuf_.template GetWithOffset<xType>(
                    this->aL1DataSize_, this->L1_BUFFER_HALF_SIZE + bL1DataSizeTotal);
                this->aL1LocalBuf2_ = this->l1Tbuf_.template GetWithOffset<xType>(
                    this->aL1DataSize_, this->L1_BUFFER_HALF_SIZE + bL1DataSizeTotal + this->aL1DataSize_ * sizeof(xType));
                this->aL1LocalBuf1_ = this->l1Tbuf_.template GetWithOffset<xType>(this->aL1DataSize_, bL1DataSizeTotal);
                this->aL1LocalBuf3_ = this->l1Tbuf_.template GetWithOffset<xType>(
                    this->aL1DataSize_, bL1DataSizeTotal + this->aL1DataSize_ * sizeof(xType));

                this->bL1LocalBuf1_ =
                    this->l1Tbuf_.template GetWithOffset<xType>(this->bL1DataSize_, this->L1_BUFFER_HALF_SIZE);
                this->bL1LocalBuf3_ = this->l1Tbuf_.template GetWithOffset<xType>(
                    this->bL1DataSize_, this->L1_BUFFER_HALF_SIZE + this->bL1DataSize_ * sizeof(xType));
            } else if (this->tiling_->AL1Pingpong == this->DOUBLE_BUFFER) {
                this->bL1LocalBuf0_ = this->l1Tbuf_.template GetWithOffset<xType>(this->bL1DataSize_, 0);
                this->bL1LocalBuf2_ =
                    this->l1Tbuf_.template GetWithOffset<xType>(this->bL1DataSize_, this->bL1DataSize_ * sizeof(xType));
                this->aL1LocalBuf0_ = this->l1Tbuf_.template GetWithOffset<xType>(
                    this->aL1DataSize_, this->L1_BUFFER_HALF_SIZE + bL1DataSizeTotal);
                this->aL1LocalBuf1_ = this->l1Tbuf_.template GetWithOffset<xType>(this->aL1DataSize_, bL1DataSizeTotal);

                this->bL1LocalBuf1_ =
                    this->l1Tbuf_.template GetWithOffset<xType>(this->bL1DataSize_, this->L1_BUFFER_HALF_SIZE);
                this->bL1LocalBuf3_ = this->l1Tbuf_.template GetWithOffset<xType>(
                    this->bL1DataSize_, this->L1_BUFFER_HALF_SIZE + this->bL1DataSize_ * sizeof(xType));
            } else {  // this->tiling_->AL1Pingpong == 1
                this->bL1LocalBuf0_ = this->l1Tbuf_.template GetWithOffset<xType>(this->bL1DataSize_, 0);
                this->bL1LocalBuf2_ =
                    this->l1Tbuf_.template GetWithOffset<xType>(this->bL1DataSize_, this->bL1DataSize_ * sizeof(xType));
                this->aL1LocalBuf0_ = this->l1Tbuf_.template GetWithOffset<xType>(this->aL1DataSize_, bL1DataSizeTotal);
                this->bL1LocalBuf1_ = this->l1Tbuf_.template GetWithOffset<xType>(
                    this->bL1DataSize_,
                    this->Max(this->L1_BUFFER_HALF_SIZE, bL1DataSizeTotal + this->aL1DataSize_ * sizeof(xType)));
                this->bL1LocalBuf3_ = this->l1Tbuf_.template GetWithOffset<xType>(
                    this->bL1DataSize_,
                    this->Max(this->L1_BUFFER_HALF_SIZE, bL1DataSizeTotal + this->aL1DataSize_ * sizeof(xType)) +
                        this->bL1DataSize_ * sizeof(xType));
            }
        } else {
            // BL1 为 single buffer 或 double buffer (未考虑 L1 bank 冲突)
            this->bL1LocalBuf0_ = this->l1Tbuf_.template GetWithOffset<xType>(this->bL1DataSize_, 0);
            if (this->tiling_->BL1Pingpong == this->DOUBLE_BUFFER) {
                this->bL1LocalBuf1_ =
                    this->l1Tbuf_.template GetWithOffset<xType>(this->bL1DataSize_, this->bL1DataSize_ * sizeof(xType));
            }
            this->aL1LocalBuf0_ = this->l1Tbuf_.template GetWithOffset<xType>(
                this->aL1DataSize_, this->vecPingpong_ * this->bL1DataSize_ * sizeof(xType));
            if (this->tiling_->AL1Pingpong == this->QUADRUPLE_BUFFER) {
                this->aL1LocalBuf1_ = this->l1Tbuf_.template GetWithOffset<xType>(
                    this->aL1DataSize_,
                    (this->aL1DataSize_ + this->tiling_->BL1Pingpong * this->bL1DataSize_) * sizeof(xType));
                this->aL1LocalBuf2_ = this->l1Tbuf_.template GetWithOffset<xType>(
                    this->aL1DataSize_,
                    (this->aL1DataSize_ * IDX_2 + this->tiling_->BL1Pingpong * this->bL1DataSize_) * sizeof(xType));
                this->aL1LocalBuf3_ = this->l1Tbuf_.template GetWithOffset<xType>(
                    this->aL1DataSize_,
                    (this->aL1DataSize_ * IDX_3 + this->tiling_->BL1Pingpong * this->bL1DataSize_) * sizeof(xType));
            } else if (this->tiling_->AL1Pingpong == this->DOUBLE_BUFFER) {
                this->aL1LocalBuf1_ = this->l1Tbuf_.template GetWithOffset<xType>(
                    this->aL1DataSize_,
                    (this->aL1DataSize_ + this->tiling_->BL1Pingpong * this->bL1DataSize_) * sizeof(xType));
            }
        }
        mmObj_.SetSubBlockIdx(0);
        mmObj_.Init(&this->tiling_->matmulTiling, tPipe);
    } else {
        this->weightInUb_ = this->vecQueWeight_.template Get<wType>();
        this->scaleInUb_ = this->vecQueScale_.template Get<scaleType>();
        this->weightOutUb_ = this->vecQueWeightOut_.template Get<xType>();
        this->scaleMaskUb_ = this->vecQueScaleMask_.template Get<uint64_t>();
        if constexpr (hasAntiQuantOffset) {
            this->offsetInUb_ = this->vecQueOffset_.template Get<xType>();
        }
        this->scaleMaskUb_.SetValue(0, 0x00000000ffffffff);
        this->scaleMaskUb_.SetValue(1, 0x00000000ffffffff);
        this->scaleMaskUb_.SetValue(2, 0x00000000ffffffff);
        this->scaleMaskUb_.SetValue(3, 0x00000000ffffffff);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline void
QuantBatchMatmulV4PerChannelKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                   scaleType, weightNz>::CopyND2NZ(int aL1Idx)
{
    AscendC::Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    if constexpr (aTrans) {
        nd2nzParams.nValue = this->kAL1Len_;
        nd2nzParams.dValue = this->mAL1Len_;
        nd2nzParams.srcDValue = this->tiling_->matmulTiling.M;
    } else {
        nd2nzParams.nValue = this->mAL1Len_;
        nd2nzParams.dValue = this->kAL1Len_;
        nd2nzParams.srcDValue = this->tiling_->matmulTiling.Ka;
    }
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.dstNzC0Stride = this->CeilAlign(nd2nzParams.nValue, BLOCK_CUBE);
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;
    if (aL1Idx == IDX_0) {
        DataCopy(this->aL1LocalBuf0_, this->xGlobal_[this->aGmOffset_], nd2nzParams);
    } else if (aL1Idx == IDX_1) {
        DataCopy(this->aL1LocalBuf1_, this->xGlobal_[this->aGmOffset_], nd2nzParams);
    } else if (aL1Idx == IDX_2) {
        DataCopy(this->aL1LocalBuf2_, this->xGlobal_[this->aGmOffset_], nd2nzParams);
    } else if (aL1Idx == IDX_3) {
        DataCopy(this->aL1LocalBuf3_, this->xGlobal_[this->aGmOffset_], nd2nzParams);
    }
    SetFlag<HardEvent::MTE2_MTE1>(aL1Idx);
    WaitFlag<HardEvent::MTE2_MTE1>(aL1Idx);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline void
QuantBatchMatmulV4PerChannelKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                   scaleType, weightNz>::GetAL1(int64_t kFactorIdx,
                                                      AscendC::TEventID eventIdsMte1ToMte2[MAX_AL1_BUF_NUM])
{
    if ASCEND_IS_AIC {
        this->kAL1Offset_ = kFactorIdx / this->kAl1Factor_ * this->kAL1Size_;
        this->kAL1Len_ = this->Min(this->tiling_->kSize - this->kAL1Offset_, this->kAL1Size_);
        if (kFactorIdx % this->kAl1Factor_ != 0) {
            return;
        }

        // 等待目标buffer可用
        WaitFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2[this->curAL1BufIdx_]);
        this->aGmOffset_ = this->mBlockOffset_ + this->curML0Idx_ * this->tiling_->matmulTiling.baseM;
        if constexpr (!aTrans) {
            this->aGmOffset_ *= this->tiling_->matmulTiling.Ka;
            this->aGmOffset_ += (kFactorIdx / this->kAl1Factor_) * this->kAL1Size_;
        } else {
            this->aGmOffset_ += (kFactorIdx / this->kAl1Factor_) * this->tiling_->mSize;
        }
        // CopyND2NZ 函数内部包含 SetFlag<MTE2_MTE1> / WaitFlag<MTE2_MTE1> 等同步指令
        CopyND2NZ(this->curAL1BufIdx_);
    }
}

/*
 * 该函数作用为通过 IterMatmulOut 每次移动一个 baseM 或 baseN，并循环遍历 KL1，
 * 计算好 AL1 和 BL1 的搬运时刻和 index 后，调用 compute 进行计算
 */
template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline void
QuantBatchMatmulV4PerChannelKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                   scaleType, weightNz>::Process()
{
    uint64_t usedCoreNum = this->tiling_->cubeNumBlocksM * this->tiling_->cubeNumBlocksN;
    if ((this->curBlockIdx_) >= usedCoreNum) {
        return;
    }
    AscendC::TEventID eventIdsMte1ToMte2[MAX_AL1_BUF_NUM];
    this->InitSync(eventIdsMte1ToMte2);

    while (this->IterMatmulOut()) {
        for (int32_t kFactorIdx = 0; kFactorIdx < this->kSingleCoreIterNum_; kFactorIdx++) {
            this->GetAL1(kFactorIdx, eventIdsMte1ToMte2);
            this->GetBL1(kFactorIdx);
            this->IterateMatmul(kFactorIdx);
            this->PostProcess(kFactorIdx, eventIdsMte1ToMte2);
        }
        if ASCEND_IS_AIC {
            uint64_t outOffset =
                (this->mBlockOffset_ + this->curML0Idx_ * this->tiling_->matmulTiling.baseM) * this->tiling_->nSize +
                this->nBlockOffset_ + this->curNL0Idx_ * this->tiling_->matmulTiling.baseN;
#ifndef __CCE_KT_TEST__
            mmObj_.GetTensorC(this->yGlobal_[outOffset]);
#endif
        }
    }
    this->EndSync(eventIdsMte1ToMte2);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline void
QuantBatchMatmulV4PerChannelKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                   scaleType, weightNz>::GetBL1(int64_t kFactorIdx)
{
    this->kBL1Offset_ = kFactorIdx / this->kBl1Factor_ * this->kBL1Size_;
    this->kBL1Len_ = this->Min(this->tiling_->kSize - this->kBL1Offset_, this->kBL1Size_);
    this->nBL1Offset_ = this->nBlockOffset_ + this->curNL1Idx_ * this->tiling_->matmulTiling.baseN;
    if (kFactorIdx % this->kBl1Factor_ != 0) {
        return;
    }

    if ASCEND_IS_AIV {
        // WaitForCube在CopyUb2L1内
        this->CopyUb2L1();
        // NotifyCube在CopyUb2L1内
    } else {
#if IS_MX_QUANT_TYPE
        this->scaleBGmOffset_ = this->nBL1Offset_ * (this->tiling_->matmulTiling.Kb / GROUP_SIZE_32) +
                                (kFactorIdx / this->kBl1Factor_) * this->kBL1Size_ / GROUP_SIZE_32;
        CopyScaleB2L1();
#endif
        this->WaitForVector(this->curBL1BufIdx_);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, QuantType antiQuantType, typename scaleType, bool weightNz>
__aicore__ inline void
QuantBatchMatmulV4PerChannelKernel<xType, wType, biasType, yType, aTrans, bTrans, hasAntiQuantOffset, antiQuantType,
                                   scaleType, weightNz>::IterateMatmul(int64_t kFactorIdx)
{
    if ASCEND_IS_AIC {
        // 一次载入gm到L1载入，多次使用，需要有偏移
        if constexpr (aTrans) {
            // (m1,k1,k0,m0), offsetK(kBL1Size_肯定是16的倍数，所以不需要对齐) + offsetM
            this->aL1Offset_ = (kFactorIdx % this->kAl1Factor_) * this->kBL1Size_ * BLOCK_CUBE +
                               (this->curML0Idx_ % this->tiling_->matmulTiling.stepM) *
                                   this->tiling_->matmulTiling.baseM * this->CeilAlign(this->kAL1Len_, BLOCK_CUBE);
        } else {
            // (k1,m1,m0,k0), offsetM + offsetK
            this->aL1Offset_ =
                (this->curML0Idx_ % this->tiling_->matmulTiling.stepM) * this->tiling_->matmulTiling.baseM *
                    BLOCK_CUBE +
                (kFactorIdx % this->kAl1Factor_) * this->kBL1Size_ * this->CeilAlign(this->mAL1Len_, BLOCK_CUBE);
        }
        if constexpr (aTrans) {
            if constexpr (bTrans) {
                mmObj_.SetOrgShape(this->mAL1Len_, this->CeilAlign(this->nBL1Len_, BLOCK_CUBE),
                                         this->CeilAlign(this->kAL1Len_, BLOCK_CUBE), this->kBL1Len_,
                                         this->tiling_->nSize);
            } else {
                mmObj_.SetOrgShape(this->mAL1Len_, this->nBL1Len_, this->CeilAlign(this->kAL1Len_, BLOCK_CUBE),
                                         this->CeilAlign(this->kBL1Len_, BLOCK_CUBE), this->tiling_->nSize);
            }
        } else {
            if constexpr (bTrans) {
                mmObj_.SetOrgShape(this->CeilAlign(this->mAL1Len_, BLOCK_CUBE),
                                         this->CeilAlign(this->nBL1Len_, BLOCK_CUBE), this->kAL1Len_, this->kBL1Len_,
                                         this->tiling_->nSize);
            } else {
                mmObj_.SetOrgShape(this->CeilAlign(this->mAL1Len_, BLOCK_CUBE), this->nBL1Len_, this->kAL1Len_,
                                         this->CeilAlign(this->kBL1Len_, BLOCK_CUBE), this->tiling_->nSize);
            }
        }
        if (this->curAL1BufIdx_ == IDX_0) {
            mmObj_.SetTensorA(this->aL1LocalBuf0_[this->aL1Offset_], aTrans);
        } else if (this->curAL1BufIdx_ == IDX_1) {
            mmObj_.SetTensorA(this->aL1LocalBuf1_[this->aL1Offset_], aTrans);
        } else if (this->curAL1BufIdx_ == IDX_2) {
            mmObj_.SetTensorA(this->aL1LocalBuf2_[this->aL1Offset_], aTrans);
        } else {
            mmObj_.SetTensorA(this->aL1LocalBuf3_[this->aL1Offset_], aTrans);
        }
        if constexpr (bTrans) {
            // (k1,n1,n0,k0), offsetN + offsetK
            this->bL1Offset_ =
                (this->curNL0Idx_ % this->tiling_->matmulTiling.stepN) * this->tiling_->matmulTiling.baseN *
                    BLOCK_CUBE +
                (kFactorIdx % this->kBl1Factor_) * this->kAL1Size_ * this->CeilAlign(this->nBL1Len_, BLOCK_CUBE);
        } else {
            // (n1,k1,k0,n0), offsetK + offsetN
            this->bL1Offset_ = (kFactorIdx % this->kBl1Factor_) * this->kAL1Size_ * BLOCK_CUBE +
                               (this->curNL0Idx_ % this->tiling_->matmulTiling.stepN) *
                                   this->tiling_->matmulTiling.baseN * this->CeilAlign(this->kBL1Len_, BLOCK_CUBE);
        }
        if (this->curBL1BufIdx_ == IDX_0) {
            mmObj_.SetTensorB(this->bL1LocalBuf0_[this->bL1Offset_], bTrans);
        } else if (this->curBL1BufIdx_ == IDX_1) {
            mmObj_.SetTensorB(this->bL1LocalBuf1_[this->bL1Offset_], bTrans);
        } else if (this->curBL1BufIdx_ == IDX_2) {
            mmObj_.SetTensorB(this->bL1LocalBuf2_[this->bL1Offset_], bTrans);
        } else {
            mmObj_.SetTensorB(this->bL1LocalBuf3_[this->bL1Offset_], bTrans);
        }
        if (this->tiling_->matmulTiling.isBias) {
            mmObj_.SetBias(
                this->biasGlobal_[this->curNL0Idx_ * this->tiling_->matmulTiling.baseN + this->nBlockOffset_]);
        }
        mmObj_.SetQuantVector(
            this->quantScaleGlobal_[this->curNL0Idx_ * this->tiling_->matmulTiling.baseN + this->nBlockOffset_]);
        mmObj_.SetTail(this->baseUseM_, this->baseUseN_, this->Min(this->kAL1Len_, this->kBL1Len_));
        mmObj_.Iterate(kFactorIdx != 0);
    }
}
}  // namespace QuantBatchMatmulV4
#endif  // QUANT_BATCH_MATMUL_V4_PERCHANNEL_H
