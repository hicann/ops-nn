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
 * \file gather_v2_simd_last_gather.h
 * \brief
 */
#ifndef GATHER_V2_SIMD_LAST_GATHER
#define GATHER_V2_SIMD_LAST_GATHER

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"

namespace gatherv2 {
using namespace AscendC;

template <typename T>
struct IndexTypeGet {
    using type = typename std::conditional<sizeof(T) == sizeof(int8_t) || sizeof(T) == sizeof(int16_t), uint16_t, uint32_t>::type;
};

template <typename TARGET_T, typename ORG_T>
__aicore__ inline void LoadIndices(MicroAPI::RegTensor<TARGET_T> &vregIndcie, MicroAPI::MaskReg &gatherMask, __local_mem__ ORG_T *indiceAddr, MicroAPI::MaskReg preg, ORG_T dimsize) 
{
    if constexpr (sizeof(ORG_T) == sizeof(int64_t)) {
        if constexpr (sizeof(TARGET_T) == sizeof(int32_t)) {
            MicroAPI::RegTensor<int64_t, MicroAPI::RegTraitNumTwo> tmpReg;
            MicroAPI::DataCopy(tmpReg, indiceAddr);
            MicroAPI::MaskReg gtPreg;
            MicroAPI::MaskReg ltPreg;
            MicroAPI::CompareScalar<int64_t, CMPMODE::GE>(gtPreg, tmpReg, 0, preg); 
            MicroAPI::CompareScalar<int64_t, CMPMODE::LT>(ltPreg, tmpReg, dimsize, preg); 
            MicroAPI::MaskAnd(gatherMask, gtPreg, ltPreg, preg);
            MicroAPI::Pack((MicroAPI::RegTensor<uint32_t>&)vregIndcie, tmpReg);
        } else {
            constexpr int16_t vfLen = AscendC::VECTOR_REG_WIDTH / sizeof(int32_t);
            MicroAPI::RegTensor<int64_t, MicroAPI::RegTraitNumTwo> tmpReg0, tmpReg1;
            MicroAPI::RegTensor<int32_t> tmpB32Reg0, tmpB32Reg1;
            MicroAPI::MaskReg lowPreg, highPreg;
            MicroAPI::MaskInterleave<int16_t>(lowPreg, highPreg, preg, preg);
            MicroAPI::DataCopy(tmpReg0, indiceAddr);
            MicroAPI::DataCopy(tmpReg1, indiceAddr + vfLen);
            MicroAPI::Pack((MicroAPI::RegTensor<uint32_t>&)tmpB32Reg0, tmpReg0);
            MicroAPI::Pack((MicroAPI::RegTensor<uint32_t>&)tmpB32Reg1, tmpReg1);
            MicroAPI::DeInterleave<int16_t>((MicroAPI::RegTensor<int16_t>&)vregIndcie, (MicroAPI::RegTensor<int16_t>&)tmpB32Reg0, (MicroAPI::RegTensor<int16_t>&)tmpB32Reg0, (MicroAPI::RegTensor<int16_t>&)tmpB32Reg1);
            MicroAPI::MaskReg gtPreg, ltPreg;
            MicroAPI::CompareScalar<int64_t, CMPMODE::GE>(gtPreg, tmpReg0, 0, lowPreg); 
            MicroAPI::CompareScalar<int64_t, CMPMODE::LT>(ltPreg, tmpReg0, dimsize, lowPreg);   
            MicroAPI::MaskAnd(lowPreg, gtPreg, ltPreg, lowPreg);
            MicroAPI::CompareScalar<int64_t, CMPMODE::GE>(gtPreg, tmpReg1, 0, highPreg); 
            MicroAPI::CompareScalar<int64_t, CMPMODE::LT>(ltPreg, tmpReg1, dimsize, highPreg);   
            MicroAPI::MaskAnd(highPreg, gtPreg, ltPreg, highPreg);
            MicroAPI::MaskDeInterleave<int16_t>(gatherMask, ltPreg, lowPreg, highPreg);
        }   
    } else {
        if constexpr (sizeof(TARGET_T) == sizeof(int32_t)) {
            MicroAPI::DataCopy((MicroAPI::RegTensor<int32_t>&)vregIndcie, indiceAddr);
            MicroAPI::MaskReg gtPreg;
            MicroAPI::MaskReg ltPreg;
            MicroAPI::CompareScalar<int32_t, CMPMODE::GE>(gtPreg, (MicroAPI::RegTensor<int32_t>&)vregIndcie, 0, preg); 
            MicroAPI::CompareScalar<int32_t, CMPMODE::LT>(ltPreg,(MicroAPI::RegTensor<int32_t>&) vregIndcie, dimsize, preg); 
            MicroAPI::MaskAnd(gatherMask, gtPreg, ltPreg, preg);
      } else {
            constexpr int16_t vfLen = AscendC::VECTOR_REG_WIDTH / sizeof(int32_t);
            MicroAPI::RegTensor<int32_t> tmpReg0, tmpReg1;
            MicroAPI::RegTensor<int32_t> tmpB32;
            MicroAPI::MaskReg lowPreg, highPreg;
            MicroAPI::MaskInterleave<int16_t>(lowPreg, highPreg, preg, preg);
            MicroAPI::DataCopy(tmpReg0, indiceAddr);
            MicroAPI::DataCopy(tmpReg1, indiceAddr + vfLen);

            MicroAPI::DeInterleave<int16_t>((MicroAPI::RegTensor<int16_t>&)vregIndcie, (MicroAPI::RegTensor<int16_t>&)tmpB32, (MicroAPI::RegTensor<int16_t>&)tmpReg0, (MicroAPI::RegTensor<int16_t>&)tmpReg1);
            MicroAPI::MaskReg gtPreg, ltPreg;
            MicroAPI::CompareScalar<int32_t, CMPMODE::GE>(gtPreg, tmpReg0, 0, lowPreg); 
            MicroAPI::CompareScalar<int32_t, CMPMODE::LT>(ltPreg, tmpReg0, dimsize, lowPreg);   
            MicroAPI::MaskAnd(lowPreg, gtPreg, ltPreg, lowPreg);
            MicroAPI::CompareScalar<int32_t, CMPMODE::GE>(gtPreg, tmpReg1, 0, highPreg); 
            MicroAPI::CompareScalar<int32_t, CMPMODE::LT>(ltPreg, tmpReg1, dimsize, highPreg);   
            MicroAPI::MaskAnd(highPreg, gtPreg, ltPreg, highPreg);
            MicroAPI::MaskDeInterleave<int16_t>(gatherMask, ltPreg, lowPreg, highPreg);
      }        
    }
}

template <typename X_T, typename INDICES_T, bool NEG_INDICE_SUPPORT>
class Gatherv2SimdLastGather {
  public:
        __aicore__ inline Gatherv2SimdLastGather(TPipe *pipe): pipe_(pipe){};
        __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR axis, GM_ADDR y, const GatherV2LastTilingData* tilingData);
        __aicore__ inline void Process();
        __aicore__ inline void CopyInIndices(int64_t offset, uint32_t nBurst, uint32_t copyLen);
        __aicore__ inline void CopyInX(int64_t offset, uint32_t nBurst, uint32_t copyLen);
        __aicore__ inline void CopyOut(int64_t offset, uint32_t nBurst, uint32_t copyLen);
        __aicore__ inline void ConvertNegIndices(const LocalTensor<int32_t> &indicesLocal, int32_t num);
        __aicore__ inline void GatherComputeMultiRows(const LocalTensor<X_T> &xLocal,  const LocalTensor<int32_t> &indicesLocal, int32_t rows, int32_t cols, int64_t yOffset);
        __aicore__ inline void SplitIndicesProcess();
        __aicore__ inline void NoSplitIndicesProcess();
    private:
        GlobalTensor<X_T> xGm_;
        GlobalTensor<INDICES_T> indicesGm_;
        GlobalTensor<X_T> yGm_;
        TPipe *pipe_;
        TQue<QuePosition::VECIN, 1> inQueue_;
        TQue<QuePosition::VECIN, 1> indexQue_;
        TQue<QuePosition::VECOUT, 1> outQueue_;
        TBuf<QuePosition::VECCALC> indexBuf_;
        const GatherV2LastTilingData* tilingData_;
        int32_t blockIdx_ {0};
        int32_t pBlock_ {0};
        int32_t gBlock_ {0};
        int32_t curGatherFactor_ {0};
};

template <typename X_T, typename INDICES_T, bool NEG_INDICE_SUPPORT>
__aicore__ inline void Gatherv2SimdLastGather<X_T, INDICES_T, NEG_INDICE_SUPPORT>::Init(GM_ADDR x, GM_ADDR indices, GM_ADDR axis,
                                                                    GM_ADDR y, const GatherV2LastTilingData* tilingData) {
    tilingData_ = tilingData;
    blockIdx_ = static_cast<int32_t>(GetBlockIdx());
    pBlock_ = blockIdx_ / tilingData->coreInCols;
    gBlock_ = blockIdx_ % tilingData->coreInCols;
    xGm_.SetGlobalBuffer((__gm__ X_T*)x + pBlock_*tilingData->blockFactor*tilingData->gatherDimSize);

    indicesGm_.SetGlobalBuffer((__gm__ INDICES_T*)indices + gBlock_*tilingData->gFactor);
    yGm_.SetGlobalBuffer((__gm__ X_T*)y +  pBlock_*tilingData->blockFactor*tilingData->gatherSize + gBlock_*tilingData->gFactor);
    pipe_->InitBuffer(inQueue_,  tilingData_->inputNum, tilingData_->inputUbSize);
    pipe_->InitBuffer(outQueue_, 2, tilingData_->outUbSize);
    pipe_->InitBuffer(indexQue_, tilingData_->indicesNum, tilingData_->indiceUbSize);
    pipe_->InitBuffer(indexBuf_, tilingData_->indiceCastUbSize);
    curGatherFactor_ = (gBlock_ == tilingData->coreInCols - 1) ? tilingData->gatherSize - (tilingData->coreInCols - 1) * tilingData->gFactor : tilingData->gFactor;
}

template <typename X_T, typename INDICES_T, bool NEG_INDICE_SUPPORT>
__aicore__ inline void Gatherv2SimdLastGather<X_T, INDICES_T, NEG_INDICE_SUPPORT>::CopyInIndices(int64_t offset, uint32_t nBurst, uint32_t copyLen)
{
    DataCopyPadExtParams<INDICES_T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;
    LocalTensor<INDICES_T> indicesLocal = indexQue_.AllocTensor<INDICES_T>();
    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;
    dataCoptExtParams.blockLen = copyLen * sizeof(INDICES_T);
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(indicesLocal, indicesGm_[offset], dataCoptExtParams, dataCopyPadExtParams);
    indexQue_.EnQue<INDICES_T>(indicesLocal);
}

template <typename X_T, typename INDICES_T, bool NEG_INDICE_SUPPORT>
__aicore__ inline void Gatherv2SimdLastGather<X_T, INDICES_T, NEG_INDICE_SUPPORT>::CopyInX(int64_t offset, uint32_t nBurst, uint32_t copyLen)
{
    DataCopyPadExtParams<X_T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;
    LocalTensor<X_T> xLocal = inQueue_.AllocTensor<X_T>();
    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;
    dataCoptExtParams.blockLen = copyLen * sizeof(X_T);
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(xLocal, xGm_[offset], dataCoptExtParams, dataCopyPadExtParams);
    inQueue_.EnQue<X_T>(xLocal);
}

template <typename X_T, typename INDICES_T, bool NEG_INDICE_SUPPORT>
__aicore__ inline void Gatherv2SimdLastGather<X_T, INDICES_T, NEG_INDICE_SUPPORT>::CopyOut(int64_t offset, uint32_t nBurst, uint32_t copyLen)
{
    int32_t ubBlockSize= static_cast<int64_t>(Ops::Base::GetUbBlockSize());
    int32_t srcStride = (tilingData_->ubCols - copyLen)  * sizeof(X_T) / ubBlockSize;
    int32_t dstStride = (tilingData_->gatherSize - copyLen) * sizeof(X_T);
    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;
    dataCoptExtParams.blockLen = copyLen * sizeof(X_T);
    dataCoptExtParams.srcStride = srcStride;
    dataCoptExtParams.dstStride = dstStride;
    LocalTensor<X_T> yLocal = outQueue_.DeQue<X_T>();
    DataCopyPad(yGm_[offset], yLocal, dataCoptExtParams);
    outQueue_.FreeTensor(yLocal);
}

template <typename X_T, typename INDICES_T, bool NEG_INDICE_SUPPORT>
__aicore__ inline void Gatherv2SimdLastGather<X_T, INDICES_T, NEG_INDICE_SUPPORT>::SplitIndicesProcess() {
    int32_t inputFactor = tilingData_->ubRows;
    int64_t curBlockFactor = (GetBlockIdx() == tilingData_->needCoreNum - 1) ? tilingData_->tailBlockFactor : tilingData_->blockFactor;
    int64_t inputLoop = (curBlockFactor + inputFactor - 1) / inputFactor;
    int32_t tailInputFactor = curBlockFactor - (inputLoop - 1) * inputFactor;
    int32_t maxOutCols = tilingData_->ubCols;
    int64_t outLoopNum = (curGatherFactor_ + maxOutCols - 1) / maxOutCols;
    int32_t tailOutCols = curGatherFactor_ - (outLoopNum - 1) * maxOutCols;
    for (int64_t i = 0; i < inputLoop; i++) {
        int32_t curInputFactor = (i == inputLoop - 1) ? tailInputFactor : inputFactor;
        CopyInX(i * inputFactor * tilingData_->gatherDimSize, 1, curInputFactor * tilingData_->gatherDimSize);
        LocalTensor<X_T> xLocal = inQueue_.DeQue<X_T>();
        for (int64_t j = 0; j < outLoopNum; j++) {
            int32_t curOutCols = (j == outLoopNum - 1) ? tailOutCols : maxOutCols;
            CopyInIndices(j * maxOutCols, 1, curOutCols);
            LocalTensor<INDICES_T> indicesLocal = indexQue_.DeQue<INDICES_T>();
            if constexpr (sizeof(INDICES_T) == sizeof(int64_t)) {
                LocalTensor<int32_t> tmpLocal = indexBuf_.Get<int32_t>();
                Cast(tmpLocal, indicesLocal, AscendC::RoundMode::CAST_NONE, curOutCols);
                indexQue_.FreeTensor(indicesLocal);
                ConvertNegIndices(tmpLocal, curOutCols);
                GatherComputeMultiRows(xLocal, tmpLocal, curInputFactor, curOutCols, i * inputFactor * tilingData_->gatherSize + j * maxOutCols);
            } else {
                ConvertNegIndices(indicesLocal, curOutCols);
                GatherComputeMultiRows(xLocal, indicesLocal, curInputFactor, curOutCols, i * inputFactor * tilingData_->gatherSize + j * maxOutCols);
                indexQue_.FreeTensor(indicesLocal);
            }
      }
      inQueue_.FreeTensor(xLocal);
    }
}

template <typename X_T, typename INDICES_T, bool NEG_INDICE_SUPPORT>
__aicore__ inline void Gatherv2SimdLastGather<X_T, INDICES_T, NEG_INDICE_SUPPORT>::NoSplitIndicesProcess() {
    int32_t maxIndcieNum = tilingData_->outUbSize / sizeof(X_T);
    int64_t indiceLoopNum = (curGatherFactor_ + maxIndcieNum - 1) / maxIndcieNum;
    int32_t tailIndicesNum = curGatherFactor_ - (indiceLoopNum - 1) * maxIndcieNum;
    int32_t inputFactor = tilingData_->ubRows;
    int64_t curBlockFactor = (GetBlockIdx() == tilingData_->needCoreNum - 1) ? tilingData_->tailBlockFactor : tilingData_->blockFactor;
    int64_t inputLoop = (curBlockFactor + inputFactor - 1) / inputFactor;
    int32_t tailInputFactor = curBlockFactor - (inputLoop - 1) * inputFactor;
    CopyInIndices(0, 1, curGatherFactor_);
    LocalTensor<INDICES_T> indicesLocal = indexQue_.DeQue<INDICES_T>();
    LocalTensor<int32_t> tmpLocal = indexBuf_.Get<int32_t>();
    if constexpr (sizeof(INDICES_T) == sizeof(int64_t)) {
        Cast(tmpLocal, indicesLocal, AscendC::RoundMode::CAST_NONE, curGatherFactor_);
        ConvertNegIndices(tmpLocal, curGatherFactor_);
    } else {
        ConvertNegIndices(indicesLocal, curGatherFactor_);
    }

    for (int64_t i = 0; i < inputLoop; i++) {
        int32_t curInputFactor = (i == inputLoop - 1) ? tailInputFactor : inputFactor;
        CopyInX(i * inputFactor * tilingData_->gatherDimSize, 1, curInputFactor * tilingData_->gatherDimSize);
        LocalTensor<X_T> xLocal = inQueue_.DeQue<X_T>();
        if constexpr (sizeof(INDICES_T) == sizeof(int64_t)) {
            GatherComputeMultiRows(xLocal, tmpLocal, curInputFactor, curGatherFactor_, i * inputFactor * tilingData_->gatherSize);
        } else {
            GatherComputeMultiRows(xLocal, indicesLocal, curInputFactor, curGatherFactor_, i * inputFactor * tilingData_->gatherSize);
        }
        inQueue_.FreeTensor(xLocal);
    }
    indexQue_.FreeTensor(indicesLocal);
}

template <typename X_T, typename INDICES_T, bool NEG_INDICE_SUPPORT>
__aicore__ inline void Gatherv2SimdLastGather<X_T, INDICES_T, NEG_INDICE_SUPPORT>::GatherComputeMultiRows(const LocalTensor<X_T> &xLocal,  const LocalTensor<int32_t> &indicesLocal, int32_t rows, int32_t cols, int64_t yOffset) {
  
    __local_mem__ X_T *xAddr = (__local_mem__ X_T *)xLocal.GetPhyAddr();
    __local_mem__ int32_t *indiceAddr = (__local_mem__ int32_t *)indicesLocal.GetPhyAddr();
    using indiceType = typename IndexTypeGet<X_T>::type;
    int32_t colInUb = tilingData_->ubCols;
    int32_t loopInCols = (cols + colInUb - 1) / colInUb;
    int32_t tailCols = cols - (loopInCols - 1) * colInUb;

    for (int32_t colIdx = 0; colIdx < loopInCols; colIdx++) {
      LocalTensor<X_T> yLocal = outQueue_.AllocTensor<X_T>();
      __local_mem__ X_T *yAddr = (__local_mem__ X_T *)yLocal.GetPhyAddr();
      int32_t curCols = (colIdx == loopInCols - 1) ? tailCols : colInUb;
      constexpr int16_t vfLen = AscendC::VECTOR_REG_WIDTH / sizeof(indiceType);
      uint16_t vfLoopNum = (curCols + vfLen - 1) / vfLen;
      int32_t dimSize = tilingData_->gatherDimSize;
      uint16_t vfRowsLoop = rows;
      int32_t indicesOffset = colIdx * colInUb;
      __VEC_SCOPE__
      {
          using RegDstT = typename std::conditional<sizeof(X_T) == sizeof(int64_t), MicroAPI::RegTensor<X_T, MicroAPI::RegTraitNumTwo>,
                                              MicroAPI::RegTensor<X_T>>::type;
          RegDstT vd0;
          MicroAPI::RegTensor<indiceType> vregIndcie;
          MicroAPI::MaskReg gatherMask;
          uint32_t size = curCols;
          __local_mem__ int32_t *curIndiceAddr = indiceAddr + indicesOffset;
          for (uint16_t i = 0; i < vfLoopNum; i++) {
              MicroAPI::MaskReg preg0 = AscendC::MicroAPI::UpdateMask<indiceType>(size);
              LoadIndices<indiceType, int32_t>(vregIndcie, gatherMask, curIndiceAddr, preg0, dimSize);
              curIndiceAddr += vfLen;
              __local_mem__ X_T *curyAddr = yAddr + i * vfLen;
              __local_mem__ X_T * curXaddr = xAddr;
              for (uint16_t j = 0; j < vfRowsLoop; j++) {                 
                  if constexpr (sizeof(X_T) == 1) {
                    MicroAPI::DataCopyGather((MicroAPI::RegTensor<int16_t>&)vd0, curXaddr, vregIndcie, gatherMask);
                    MicroAPI::DataCopy<X_T, AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(
                      curyAddr, vd0, preg0);
                  } else {
                      MicroAPI::DataCopyGather(vd0, curXaddr, vregIndcie, gatherMask);
                      MicroAPI::DataCopy(curyAddr, vd0, preg0);
                  }
                curXaddr += dimSize;
                curyAddr += colInUb;
              }
          }
      }

      outQueue_.EnQue(yLocal);
      CopyOut(yOffset + colIdx * colInUb, rows, curCols);
    }
}

template <typename X_T, typename INDICES_T, bool NEG_INDICE_SUPPORT>
__aicore__ inline void Gatherv2SimdLastGather<X_T, INDICES_T, NEG_INDICE_SUPPORT>::ConvertNegIndices(const LocalTensor<int32_t> &indicesLocal, int32_t num) {
    if constexpr (NEG_INDICE_SUPPORT) {
        __local_mem__ int32_t *indiceAddr = (__local_mem__ int32_t *)indicesLocal.GetPhyAddr();
        constexpr int16_t vfLen = AscendC::VECTOR_REG_WIDTH / sizeof(int32_t);
        uint16_t vfLoopNum = (num + vfLen - 1) / vfLen;
        int32_t dimSize = tilingData_->gatherDimSize;
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<int32_t> indice;
            MicroAPI::RegTensor<int32_t> dst;
            MicroAPI::MaskReg ltPreg;
            uint32_t size = num;
            __local_mem__ int32_t *curIndiceAddr = indiceAddr;
            for (uint16_t i = 0; i < vfLoopNum; i++) {
                MicroAPI::MaskReg preg = AscendC::MicroAPI::UpdateMask<int32_t>(size);
                MicroAPI::DataCopy(indice, curIndiceAddr);
                MicroAPI::CompareScalar<int32_t, CMPMODE::LT>(ltPreg, indice, 0, preg); 
                MicroAPI::Adds(dst, indice, dimSize, ltPreg);
                MicroAPI::Copy<int32_t, MicroAPI::MaskMergeMode::MERGING>(indice, dst, ltPreg);
                MicroAPI::DataCopy(curIndiceAddr, indice, preg);
                curIndiceAddr += vfLen;
            }
      }
    }
}

template <typename X_T, typename INDICES_T, bool NEG_INDICE_SUPPORT>
__aicore__ inline void Gatherv2SimdLastGather<X_T, INDICES_T, NEG_INDICE_SUPPORT>::Process() {
  if (blockIdx_ >= tilingData_->needCoreNum) {
     return;
  }
  
  if (tilingData_->splitMode == 0) {
     NoSplitIndicesProcess();
  } else if (tilingData_->splitMode == 1) {
     SplitIndicesProcess();
  }
}

}  // namespace gatherv2
#endif  // GATHER_V2_SIMD_LAST_GATHER
