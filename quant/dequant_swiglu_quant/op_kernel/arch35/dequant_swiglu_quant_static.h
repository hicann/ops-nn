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
 * \file dequant_swiglu_quant_static.h
 * \brief
 */

#ifndef DEQUANT_SWIGLU_QUANT_STATIC_H
#define DEQUANT_SWIGLU_QUANT_STATIC_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "dequant_swiglu_quant_common.h"

namespace DequantSwigluQuantV35Ops {
using namespace AscendC;

template <typename TActScale, typename TQuantScale, typename TGroup, typename TBias, typename TXtype, typename TYtype>
class DequantSwigluQuantBaseStatic {
 public:
  static constexpr bool hasActScale_ = IsSameType<TActScale, float>::value;
  static constexpr bool hasQuantScale_ = IsSameType<TQuantScale, float>::value;
  static constexpr bool hasGroupIndex_ = IsSameType<TGroup, int64_t>::value || IsSameType<TGroup, int32_t>::value;
  // bias标记 bias可支持float，float16，bf16，int32
  static constexpr bool hasBiasIndex_ = IsSameType<TBias, float>::value || IsSameType<TBias, half>::value || IsSameType<TBias, bfloat16_t>::value || IsSameType<TBias, int32_t>::value;
  // bias数据类型为int32标记
  static constexpr bool ifBiasIntIndex_ = IsSameType<TBias, int32_t>::value;
  // bias数据类型为float标记
  static constexpr bool ifBiasFloatIndex_ = IsSameType<TBias, float>::value;
  // bias数据类型为float16标记
  static constexpr bool ifBiasFloat16Index_ = IsSameType<TBias, half>::value;
  // bias数据类型为bfloat16标记
  static constexpr bool ifBiasBfloat16Index_ = IsSameType<TBias, bfloat16_t>::value;
  // x数据类型为int32标记
  static constexpr bool ifXIntIndex_ = IsSameType<TXtype, int32_t>::value;
  // x数据类型为bf16标记
  static constexpr bool ifXBf16Index_ = IsSameType<TXtype, bfloat16_t>::value;
  // x数据类型为float16标记
  static constexpr bool ifXFloat16Index_ = IsSameType<TXtype, half>::value;
  // y数据类型为int8标记
  static constexpr bool ifYInt8Index_ = IsSameType<TYtype, int8_t>::value;
  // y数据类型为float8_e4m3标记
  static constexpr bool ifYFloat8e4m3Index_ = IsSameType<TYtype, fp8_e4m3fn_t>::value;
  // y数据类型为float8_e5m2标记
  static constexpr bool ifYFloat8e5m2Index_ = IsSameType<TYtype, fp8_e5m2_t>::value;
  // y数据类型为float4_e2m1标记
  static constexpr bool ifYFloat4e2m1Index_ = IsSameType<TYtype, fp4x2_e2m1_t>::value;
  // y数据类型为float4_e1m2标记
  static constexpr bool ifYFloat4e1m2Index_ = IsSameType<TYtype, fp4x2_e1m2_t>::value;
  // y数据类型为hifloat8标记
  static constexpr bool ifYHiFloat8Index_ = IsSameType<TYtype, hifloat8_t>::value;

  __aicore__ inline DequantSwigluQuantBaseStatic(TPipe* pipe) {
    pipe_ = pipe;
  };

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR weightScale, GM_ADDR activationScale, GM_ADDR bias, GM_ADDR quantScale,
                              GM_ADDR quantOffset, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR scale,
                              const DequantSwigluQuantV35BaseTilingData* tilingData);

  __aicore__ inline void Process();

 private:
  __aicore__ inline void ComputeReduceMax(const LocalTensor<float>& tempRes, int32_t calCount, float& maxValue);
  __aicore__ inline void ProcessSingleGroup(int64_t groupIndex, int64_t realDimx, int64_t groupOffset);
  __aicore__ inline void ProcessSingleGroupPerCore(int64_t groupIdx, int64_t xDimPerCore, int64_t coreDimxOffset);

 protected:
  /* global memory address */
  // input global mem
  GlobalTensor<TXtype> xGm_; // 当前模板支持 fp16, bf16, 用模板参数去承载输入x的类型
  GlobalTensor<float> weightScaleGm_;
  GlobalTensor<TActScale> activationScaleGm_;
  GlobalTensor<TBias> biasGm_;  // 当前模板支持 float，float16，bf16，int32， 用模板参数去承载bias的类型
  GlobalTensor<TQuantScale> quantScaleGm_;
  GlobalTensor<float> quantOffsetGm_;
  GlobalTensor<int32_t> groupIndexGm_;

  // output global mem
  GlobalTensor<TYtype> yGm_;
  GlobalTensor<float> scaleGm_;

  // bias
  LocalTensor<TBias> biasLocal;
  __local_mem__ TBias* bias1Ptr;
  __local_mem__ TBias* bias2Ptr;

  // weight_scale
  __local_mem__ float* wScale1Ptr;
  __local_mem__ float* wScale2Ptr;
  __local_mem__ float* wScale1Addr;
  __local_mem__ float* wScale2Addr;

  // quant_offset
  __local_mem__ float* qOffsetPtr;

  /* ascendc variable */
  TPipe* pipe_ = nullptr;
  TQue<QuePosition::VECIN, 1> xActQueue_;
  TQue<QuePosition::VECIN, 1> inScaleQueue_;
  TQue<QuePosition::VECIN, 1> biasQueue_;
  TQue<QuePosition::VECOUT, 1> yQueue_;

  TBuf<> tmpBuffer;

  uint32_t blockIdx_ = GetBlockIdx();
  uint32_t realCoreDim_ = 0;
  int64_t realDimx_ = 0;
  int64_t groupOffset_ = 0;
  uint32_t xUbAlignB32_ = 0; // 根据4Byte计算的32B对齐，用于float32类型
  uint32_t xUbAlignB32FullRow_ = 0; // 根据4Byte计算的整行32B对齐，用于float32类型
  uint32_t xTypeUbAlignB32_ = 0; // 根据x的输入类型不同计算的32B对齐
  uint32_t xTypeUbAlignB32FullRow_ = 0; // 根据x的输入类型不同计算的整行32B对齐
  uint32_t yUbAlignB8_ = 0;
  uint32_t yUbAlignB4_ = 0;
  uint32_t aScaleUbAlignB32_ = 0;
  uint32_t biasUbAlign_ = 0; // bias的32B对齐标记
  int64_t roundMode_ = 0; // 溢出模式的标识
  float scalarMaxNum_ = 127.0; //根据不同的输出类型，对应的最大值不同，默认127
  bool hasQuantOffset_ = false;
  bool quantIsOne_ = false;

  const DequantSwigluQuantV35BaseTilingData* tl_ = nullptr;
};
// 公共函数实现

template <typename TActScale, typename TQuantScale, typename TGroup, typename TBias, typename TXtype, typename TYtype>
__aicore__ inline void DequantSwigluQuantBaseStatic<TActScale, TQuantScale, TGroup, TBias, TXtype, TYtype>::Init(
    GM_ADDR x, GM_ADDR weightScale, GM_ADDR activationScale, GM_ADDR bias, GM_ADDR quantScale, GM_ADDR quantOffset,
    GM_ADDR groupIndex, GM_ADDR y, GM_ADDR scale, const DequantSwigluQuantV35BaseTilingData* tilingData)
{
  tl_ = tilingData;
  // 兼容bf16和float16类型，xUbAlign对齐点适配修改；如果是bf16，float16，则是用2B对对齐32B；如果是int32，则是用4B对齐32B
  if constexpr (ifXBf16Index_ || ifXFloat16Index_) {
    uint32_t BLOCK_ELEM_B32_BF = BLOCK_SIZE / sizeof(TXtype);
    xTypeUbAlignB32_ = CeilDivision(tl_->UbFactorDimy, BLOCK_ELEM_B32_BF) * BLOCK_ELEM_B32_BF;
    xTypeUbAlignB32FullRow_ = CeilDivision(tl_->UbFactorDimy * 2, BLOCK_ELEM_B32_BF) * BLOCK_ELEM_B32_BF;
  } else {
    uint32_t BLOCK_ELEM_B32_BF = BLOCK_SIZE / sizeof(TXtype);
    xTypeUbAlignB32_ = CeilDivision(tl_->UbFactorDimy, BLOCK_ELEM_B32) * BLOCK_ELEM_B32;
    xTypeUbAlignB32FullRow_ = CeilDivision(tl_->UbFactorDimy * 2, BLOCK_ELEM_B32) * BLOCK_ELEM_B32;
  }

  // 4B的数据类型对应的32B对齐点
  xUbAlignB32_ = CeilDivision(tl_->UbFactorDimy, BLOCK_ELEM_B32) * BLOCK_ELEM_B32;
  xUbAlignB32FullRow_ = CeilDivision(tl_->UbFactorDimy * 2, BLOCK_ELEM_B32) * BLOCK_ELEM_B32;

  // 兼容int8和fp8类型，两种类型都是1B
  yUbAlignB8_ = CeilDivision(tl_->UbFactorDimy, BLOCK_ELEM_B8) * BLOCK_ELEM_B8;
  // fp4数据类型对齐点，伪1B对齐
  yUbAlignB4_ = CeilDivision(tl_->UbFactorDimy / 2, BLOCK_ELEM_B8) * BLOCK_ELEM_B8;
  // activate_scale对应的对齐点
  aScaleUbAlignB32_ = BLOCK_ELEM_B32;

  // bias不同的数据类型对应的32对齐不同
  uint32_t blockElem = BLOCK_SIZE / sizeof(TBias);
  biasUbAlign_ = CeilDivision(tl_->UbFactorDimy, blockElem) * blockElem;

  // 获取指定输出类型和溢出模式
  roundMode_ = tl_->roundMode;
  //获取指定输出类型对应的最大值
  if constexpr (ifYFloat8e4m3Index_) {
    scalarMaxNum_ = 448.0;
  }
  if constexpr (ifYFloat8e5m2Index_) {
    scalarMaxNum_ = 57344.0;
  }
  if constexpr (ifYFloat4e2m1Index_) {
    scalarMaxNum_ = 6.0;
  }
  if constexpr (ifYFloat4e1m2Index_) {
    scalarMaxNum_ = 1.75;
  }
  if constexpr (ifYHiFloat8Index_) {
    scalarMaxNum_ = 32768.0;
  }

  if (quantOffset != nullptr) {
    hasQuantOffset_ = true;
  }

  if (tl_->quantIsOne == 1) {
    quantIsOne_ = true;
  }

  xGm_.SetGlobalBuffer((__gm__ TXtype*)x);
  weightScaleGm_.SetGlobalBuffer((__gm__ float*)weightScale);
  activationScaleGm_.SetGlobalBuffer((__gm__ TActScale*)activationScale);
  quantScaleGm_.SetGlobalBuffer((__gm__ TQuantScale*)quantScale);
  quantOffsetGm_.SetGlobalBuffer((__gm__ float*)quantOffset);
  if constexpr (hasGroupIndex_) {
    groupIndexGm_.SetGlobalBuffer((__gm__ int32_t*)groupIndex);
  }

  // yGm
  yGm_.SetGlobalBuffer((__gm__ TYtype*)y);
  scaleGm_.SetGlobalBuffer((__gm__ float*)scale);

  // x + activation_scale
  uint32_t tailSupply = CeilDivision(tl_->inDimy, BLOCK_SIZE * 4) * BLOCK_SIZE * 4 -  tl_->inDimy;
  if (tl_->swiGluMode == 0) {
    pipe_->InitBuffer(xActQueue_, DOUBLE_BUFFER,
                  (tl_->UbFactorDimx * xTypeUbAlignB32_ * 2) * sizeof(TXtype) +
                  (tl_->UbFactorDimx * aScaleUbAlignB32_) * sizeof(float));
  } else {
    pipe_->InitBuffer(xActQueue_, DOUBLE_BUFFER,
                  (tl_->UbFactorDimx * xTypeUbAlignB32_ * 2) * sizeof(TXtype) +
                  tailSupply * sizeof(TXtype) +
                  (tl_->UbFactorDimx * aScaleUbAlignB32_) * sizeof(float));
  }
  // weight_scale + quant_scale + quant_offset
  if (hasQuantOffset_) {
    if (tl_->swiGluMode == 0) {
      pipe_->InitBuffer(inScaleQueue_, 1, (xUbAlignB32_ * 2 + xUbAlignB32_ * 2) * sizeof(float));
    } else {
      pipe_->InitBuffer(inScaleQueue_, 1, (xUbAlignB32_ * 2 + xUbAlignB32_ * 2 + tailSupply) * sizeof(float));
    }
  } else {
    if (tl_->swiGluMode == 0) {
      pipe_->InitBuffer(inScaleQueue_, 1, (xUbAlignB32_ * 2 + xUbAlignB32_) * sizeof(float));
    } else {
      pipe_->InitBuffer(inScaleQueue_, 1, (xUbAlignB32_ * 2 + xUbAlignB32_ + tailSupply) * sizeof(float));
    }
  }
  
  // y
  pipe_->InitBuffer(yQueue_, DOUBLE_BUFFER, tl_->UbFactorDimx * yUbAlignB8_ * sizeof(TYtype));

  pipe_->InitBuffer(tmpBuffer, tl_->UbFactorDimx * xUbAlignB32_ * sizeof(float));

  // 如果有bias入参，则给bias进行地址申请
  if constexpr (hasBiasIndex_) {
    biasGm_.SetGlobalBuffer((__gm__ TBias*)bias);
    if (tl_->swiGluMode == 0) {
      // bias (1 * 2H) 申请
      pipe_->InitBuffer(biasQueue_, 1, (biasUbAlign_ * 2) * sizeof(TBias));
    } else {
      uint32_t tailSupply = CeilDivision(biasUbAlign_ * 2, BLOCK_SIZE * 4) * BLOCK_SIZE * 4 -  biasUbAlign_ * 2;
      pipe_->InitBuffer(biasQueue_, 1, (biasUbAlign_ * 2) * sizeof(TBias) + tailSupply * sizeof(TBias));
    }
  }
}

template <typename TActScale, typename TQuantScale, typename TGroup, typename TBias, typename TXtype, typename TYtype>
__aicore__ inline void DequantSwigluQuantBaseStatic<TActScale, TQuantScale, TGroup, TBias, TXtype, TYtype>::Process()
{
  if constexpr (!hasGroupIndex_) {
    realDimx_ = tl_->inDimx;
    if (realDimx_ < 0) {
      realDimx_ = 0;
    }

    ProcessSingleGroup(0, realDimx_, 0);
    return;
  }

  groupOffset_ = 0;
  if (tl_->isSpecialCoreCut == 0) {
    for (int64_t groupIndex = 0; groupIndex < tl_->inGroupNum; groupIndex++) {
      int64_t realGroupIndex = 0;
      if (tl_->speGroupType == 1) {
        if (tl_->groupIndexMode == 1) {//int64/int32
          realDimx_ = static_cast<int64_t>(groupIndexGm_(groupIndex*2+1));
          realGroupIndex = static_cast<int64_t>(groupIndexGm_(groupIndex*2));
        } else {
          realDimx_ = groupIndexGm_.template ReinterpretCast<int64_t>()(groupIndex*2+1);
          realGroupIndex = groupIndexGm_.template ReinterpretCast<int64_t>()(groupIndex*2);
        }
      } else {
        realGroupIndex = groupIndex;
        if (tl_->groupIndexMode == 1) {//int64/int32
          realDimx_ = static_cast<int64_t>(groupIndexGm_(groupIndex));
        } else {
          realDimx_ = groupIndexGm_.template ReinterpretCast<int64_t>()(groupIndex);
        }
      }
      if (realDimx_ <= 0) {
        continue;
      }

      // inDimx x的元素总数 / x的-1维shape，也即按照-1维的循环次数，也即总行数（按照二维理解）
      if (groupOffset_ < tl_->inDimx) {
        ProcessSingleGroup(realGroupIndex, realDimx_, groupOffset_);
        groupOffset_ += realDimx_;
      }
    }
    return;
  } else {
    int64_t cuGroupIdx = blockIdx_;
    for (int64_t groupIndex = 0; groupIndex < tl_->inGroupNum; groupIndex++) {
      int64_t realGroupIndex = 0;
      if (tl_->speGroupType == 1) {
        if (tl_->groupIndexMode == 1) {//int64/int32
          realDimx_ = static_cast<int64_t>(groupIndexGm_(groupIndex*2+1));
          realGroupIndex = static_cast<int64_t>(groupIndexGm_(groupIndex*2));
        } else {
          realDimx_ = groupIndexGm_.template ReinterpretCast<int64_t>()(groupIndex*2+1);
          realGroupIndex = groupIndexGm_.template ReinterpretCast<int64_t>()(groupIndex*2);
        }
      } else {
        realGroupIndex = groupIndex;
        if (tl_->groupIndexMode == 1) {//int64/int32
          realDimx_ = static_cast<int64_t>(groupIndexGm_(groupIndex));
        } else {
          realDimx_ = groupIndexGm_.template ReinterpretCast<int64_t>()(groupIndex);
        }
      }
      
      if (realDimx_ <= 0) {
        continue;
      }

      // inDimx x的元素总数 / x的-1维shape，也即按照-1维的循环次数，也即总行数（按照二维理解）
      if (groupIndex == cuGroupIdx) {
        ProcessSingleGroupPerCore(realGroupIndex, realDimx_, groupOffset_);
        cuGroupIdx += tl_->maxCoreNum;
      }
      groupOffset_ += realDimx_;
    }
    return;
  }
}

template <typename TActScale, typename TQuantScale, typename TGroup, typename TBias, typename TXtype, typename TYtype>
__aicore__ inline void DequantSwigluQuantBaseStatic<TActScale, TQuantScale, TGroup, TBias, TXtype, TYtype>::ProcessSingleGroupPerCore(int64_t groupIdx, int64_t xDimPerCore, int64_t coreDimxOffset)
{
    DataCopyPadParams padParams{false, 0, 0, 0};

    // 激活左/右部分偏移，actRight是表示是否激活右半部分
    int64_t actOffset = (tl_->swiGluMode == 0) ? tl_->actRight * tl_->UbFactorDimy : 0;
    int64_t gateOffset = tl_->UbFactorDimy - actOffset;

    // weight_scale搬入
    LocalTensor<float> inScaleLocal = inScaleQueue_.AllocTensor<float>();
    if constexpr (ifXIntIndex_) {
      if (tl_->swiGluMode == 0) {
        // copy_in: weight_scale(G, H) 激活部分
        DataCopyParams dataCopyWeightScaleParams;
        dataCopyWeightScaleParams.blockCount = 1;
        dataCopyWeightScaleParams.blockLen = tl_->UbFactorDimy * sizeof(float);
        dataCopyWeightScaleParams.srcStride = 0;
        dataCopyWeightScaleParams.dstStride = 0;
        DataCopyPad(inScaleLocal[0], weightScaleGm_[groupIdx * tl_->inDimy + actOffset], dataCopyWeightScaleParams, padParams);

        // copy_in: weight_scale(G, H) 门控部分
        DataCopyPad(inScaleLocal[xUbAlignB32_], weightScaleGm_[groupIdx * tl_->inDimy + gateOffset], dataCopyWeightScaleParams, padParams);
      } else {
        DataCopyParams dataCopyWeightScaleParams;
        dataCopyWeightScaleParams.blockCount = 1;
        dataCopyWeightScaleParams.blockLen = tl_->UbFactorDimy * 2 * sizeof(float);
        dataCopyWeightScaleParams.srcStride = 0;
        dataCopyWeightScaleParams.dstStride = 0;
        DataCopyPad(inScaleLocal[0], weightScaleGm_[groupIdx * tl_->inDimy], dataCopyWeightScaleParams, padParams);
      }
      
    }

    // copy_in: quant_scale(G, H)
    int64_t quantUbFactorDimy = quantIsOne_ ? 1 : tl_->UbFactorDimy;
    if constexpr (hasQuantScale_) {
      DataCopyParams dataCopyQuantScaleParams;
      dataCopyQuantScaleParams.blockCount = 1;
      dataCopyQuantScaleParams.blockLen = quantUbFactorDimy * sizeof(TQuantScale);
      dataCopyQuantScaleParams.srcStride = 0;
      dataCopyQuantScaleParams.dstStride = 0;
      DataCopyPad(inScaleLocal[xUbAlignB32_ * 2], quantScaleGm_[groupIdx * quantUbFactorDimy], dataCopyQuantScaleParams, padParams);
      if (hasQuantOffset_) {
        DataCopyPad(inScaleLocal[xUbAlignB32_ * 3], quantOffsetGm_[groupIdx * quantUbFactorDimy], dataCopyQuantScaleParams, padParams);
      }
    }
    inScaleQueue_.EnQue(inScaleLocal);
    inScaleLocal = inScaleQueue_.DeQue<float>();

    // copy_in: bias(1, 2H)->(1, H), (1, H)
    if constexpr (hasBiasIndex_) {
      if (tl_->swiGluMode == 0) {
        biasLocal = biasQueue_.AllocTensor<TBias>();
        DataCopyParams dataCopyBiasParams;
        dataCopyBiasParams.blockCount = 1;
        dataCopyBiasParams.blockLen = tl_->UbFactorDimy * sizeof(TBias);
        dataCopyBiasParams.srcStride = 0;
        dataCopyBiasParams.dstStride = 0;
        // [G, 2H]
        DataCopyPad(biasLocal[0], biasGm_[groupIdx * tl_->inDimy + actOffset], dataCopyBiasParams, padParams);
        DataCopyPad(biasLocal[biasUbAlign_], biasGm_[groupIdx * tl_->inDimy + gateOffset], dataCopyBiasParams, padParams);
      } else {
        biasLocal = biasQueue_.AllocTensor<TBias>();
        DataCopyParams dataCopyBiasParams;
        dataCopyBiasParams.blockCount = 1;
        dataCopyBiasParams.blockLen = tl_->UbFactorDimy * 2 * sizeof(TBias);
        dataCopyBiasParams.srcStride = 0;
        dataCopyBiasParams.dstStride = 0;
        // [G, 2H]
        DataCopyPad(biasLocal[0], biasGm_[groupIdx * tl_->inDimy], dataCopyBiasParams, padParams);
      }
      
      biasQueue_.EnQue(biasLocal);
      biasLocal = biasQueue_.DeQue<TBias>();
    }

    
    // UbFactorDimX：ub每次能处理的行数，ubDimxLoop: 每个核上需要处理几行数据，然后在单个核上，ub处理这几行数据需要几次循环
    int64_t ubDimxLoop = (xDimPerCore + tl_->UbFactorDimx - 1) / tl_->UbFactorDimx;
    int64_t ubDimxTailFactor = xDimPerCore - tl_->UbFactorDimx * (ubDimxLoop - 1);

    // 核内循环
    for (uint64_t i = 0; i < ubDimxLoop; i++) {
      // 核内循环每次计算时的起始地址：当前核计算时的起始地址 + 第i次核内循环要处理的起始地址 + 已处理过的行偏移(历史已处理过的总行数)
      int64_t xDimOffsetPerLoop = coreDimxOffset + i * tl_->UbFactorDimx ;
      int64_t xDimPerLoop = tl_->UbFactorDimx;
      if (i == ubDimxLoop - 1) {
        xDimPerLoop = ubDimxTailFactor;
      }

      // copy_in: x(xDimPerLoop, H) 激活部分
      LocalTensor<TXtype> xActLocal = xActQueue_.AllocTensor<TXtype>();
      if (tl_->swiGluMode == 0) {
        DataCopyParams dataCopyXParams;
        dataCopyXParams.blockCount = xDimPerLoop;
        dataCopyXParams.blockLen = tl_->UbFactorDimy * sizeof(TXtype);
        dataCopyXParams.srcStride = tl_->UbFactorDimy * sizeof(TXtype);
        dataCopyXParams.dstStride = 0;
        DataCopyPad(xActLocal[0], xGm_[xDimOffsetPerLoop * tl_->inDimy + actOffset], dataCopyXParams, padParams);

        // copy_in: x(xDimPerLoop, H) 门控部分
        DataCopyPad(xActLocal[xTypeUbAlignB32_ * xDimPerLoop], xGm_[xDimOffsetPerLoop * tl_->inDimy + gateOffset], dataCopyXParams, padParams);
      } else {
        // 奇偶搬运，连续搬运左右字块，actOffset和gateOffset不生效
        DataCopyParams dataCopyXParams;
        dataCopyXParams.blockCount = xDimPerLoop;
        dataCopyXParams.blockLen = tl_->UbFactorDimy * 2 * sizeof(TXtype);
        dataCopyXParams.srcStride = 0;
        dataCopyXParams.dstStride = 0;
        DataCopyPad(xActLocal[0], xGm_[xDimOffsetPerLoop * tl_->inDimy], dataCopyXParams, padParams);
      }
      
      // copy_in: activation_scale(BS,)
      LocalTensor<float> xActLocalFp32 = xActLocal.template ReinterpretCast<float>();
      if constexpr (hasActScale_) {
        DataCopyParams dataCopyActScaleParams;
        dataCopyActScaleParams.blockCount = xDimPerLoop;
        dataCopyActScaleParams.blockLen = sizeof(float);
        dataCopyActScaleParams.srcStride = 0;
        dataCopyActScaleParams.dstStride = 0;
        DataCopyPad(xActLocalFp32[xTypeUbAlignB32_ * xDimPerLoop * 2], activationScaleGm_[xDimOffsetPerLoop], dataCopyActScaleParams, padParams);
      }
      xActQueue_.EnQue(xActLocal);
      xActLocal = xActQueue_.DeQue<TXtype>();

      // int8,fp8
      LocalTensor<TYtype> yLocal = yQueue_.AllocTensor<TYtype>();
      // fp4
      LocalTensor<uint8_t> yFp4Local = yLocal.template ReinterpretCast<uint8_t>();

      LocalTensor<float> tmpXLocal = tmpBuffer.Get<float>();

      __local_mem__ float* tmpXPtr = (__local_mem__ float*)tmpXLocal.GetPhyAddr();
      __local_mem__ TYtype* yPtr = (__local_mem__ TYtype*)yLocal.GetPhyAddr();
      __local_mem__ uint8_t* yFp4Ptr = (__local_mem__ uint8_t*)yFp4Local.GetPhyAddr();

      __local_mem__ TXtype* x1Ptr = (__local_mem__ TXtype*)xActLocal.GetPhyAddr(0);
      __local_mem__ TXtype* x2Ptr = (__local_mem__ TXtype*)xActLocal.GetPhyAddr(xTypeUbAlignB32_ * xDimPerLoop);
      __local_mem__ float* aScalePtr = (__local_mem__ float*)xActLocalFp32.GetPhyAddr(xTypeUbAlignB32_ * xDimPerLoop * 2);
      // 当x=int32时，才去获取weight_scale的地址
      if constexpr (ifXIntIndex_) {
        wScale1Ptr = (__local_mem__ float*)inScaleLocal.GetPhyAddr(0);
        wScale2Ptr = (__local_mem__ float*)inScaleLocal.GetPhyAddr(xUbAlignB32_);
      }

      // 增加bias的地址，使用时需要判断biasPtr是否为空指针
      if constexpr (hasBiasIndex_) {
        bias1Ptr = (__local_mem__ TBias*)biasLocal.GetPhyAddr(0);
        bias2Ptr = (__local_mem__ TBias*)biasLocal.GetPhyAddr(biasUbAlign_);
      }

      // int32： 256B / 4B = 64次 ，bf16 or float16：256B / 2B = 128次
      constexpr uint16_t sizePerRepeat = AscendC::GetVecLen() / sizeof(float);
      uint32_t width = tl_->UbFactorDimy; // 输出y的-1轴对应的shape大小，也即H
      uint32_t widthFullRow = tl_->UbFactorDimy * 2; // 输出x的-1轴对应的shape大小，也即2H
      uint16_t repeatTimes = CeilDivision(tl_->UbFactorDimy , sizePerRepeat); // 向上取整
      uint16_t repeatTimesFullRow = CeilDivision(widthFullRow , sizePerRepeat); // 向上取整
      __local_mem__ float* tmpX2Ptr = (__local_mem__ float*)tmpXLocal.GetPhyAddr(xUbAlignB32_ * xDimPerLoop);


      // dequant
      if (tl_->swiGluMode == 1) {
        VF_CALL<DequantSwigluV2<TXtype, TBias, ifXIntIndex_, ifXFloat16Index_, ifXBf16Index_, hasBiasIndex_, hasActScale_, ifBiasIntIndex_, ifBiasFloatIndex_, ifBiasFloat16Index_, ifBiasBfloat16Index_>>(x1Ptr, tmpXPtr, wScale1Ptr, aScalePtr, bias1Ptr,
                                                                        xDimPerLoop, widthFullRow, repeatTimesFullRow, sizePerRepeat,
                                                                        xTypeUbAlignB32FullRow_, xTypeUbAlignB32_, xTypeUbAlignB32_, aScaleUbAlignB32_, 0,
                                                                        tl_->clampLimit, tl_->gluAlpha, tl_->gluBias);
      } else {
        VF_CALL<DequantSwigluV1<TXtype, TBias, ifXIntIndex_, ifXFloat16Index_, ifXBf16Index_, hasBiasIndex_, hasActScale_, ifBiasIntIndex_, ifBiasFloatIndex_, ifBiasFloat16Index_, ifBiasBfloat16Index_>>(x1Ptr, x2Ptr, tmpXPtr, wScale1Ptr, wScale2Ptr, aScalePtr, bias1Ptr, bias2Ptr,
                                                                        xDimPerLoop, width, repeatTimes, sizePerRepeat,
                                                                        xTypeUbAlignB32_, xTypeUbAlignB32_, xTypeUbAlignB32_, aScaleUbAlignB32_, 0);
      }
      

      __local_mem__ float* qScalePtr = (__local_mem__ float*)inScaleLocal.GetPhyAddr(xUbAlignB32_ * 2);
      
      // quantOffset存在的时候，则获取对应的ub地址
      if (hasQuantOffset_) {
        qOffsetPtr = (__local_mem__ float*)inScaleLocal.GetPhyAddr(xUbAlignB32_ * 3);
      }
      

      // biasLocal是否为空的判断
      if constexpr (hasBiasIndex_) {
        biasQueue_.FreeTensor(biasLocal);
      }
      
      // static_quant
      if (tl_->quantIsOne == 1 && hasQuantOffset_) {
        VF_CALL<StaticQuantWithQuantOffsetOne>(tmpXPtr, qScalePtr, qOffsetPtr, tmpXPtr, width, xDimPerLoop, repeatTimes,
                           sizePerRepeat, xTypeUbAlignB32_, static_cast<uint16_t>(tl_->quantIsOne), hasQuantOffset_);
      } else if (tl_->quantIsOne == 1 && !hasQuantOffset_) {
        VF_CALL<StaticQuantWithOne>(tmpXPtr, qScalePtr, qOffsetPtr, tmpXPtr, width, xDimPerLoop, repeatTimes,
                           sizePerRepeat, xTypeUbAlignB32_, static_cast<uint16_t>(tl_->quantIsOne), hasQuantOffset_);
      } else if (tl_->quantIsOne != 1 && hasQuantOffset_) {
        VF_CALL<StaticQuantWithQuantOffset>(tmpXPtr, qScalePtr, qOffsetPtr, tmpXPtr, width, xDimPerLoop, repeatTimes,
                           sizePerRepeat, xTypeUbAlignB32_, static_cast<uint16_t>(tl_->quantIsOne), hasQuantOffset_);
      } else if (tl_->quantIsOne != 1 && !hasQuantOffset_) {
        VF_CALL<StaticQuant>(tmpXPtr, qScalePtr, qOffsetPtr, tmpXPtr, width, xDimPerLoop, repeatTimes,
                           sizePerRepeat, xTypeUbAlignB32_, static_cast<uint16_t>(tl_->quantIsOne), hasQuantOffset_);
      }
      
      xActQueue_.FreeTensor(xActLocal);

      // cast y
      VF_CALL<CastY<TXtype, TYtype, ifYFloat8e4m3Index_, ifYFloat8e5m2Index_, ifYFloat4e2m1Index_, ifYFloat4e1m2Index_, ifYHiFloat8Index_>>(tmpXPtr, yPtr, yFp4Ptr, tl_->UbFactorDimy, xDimPerLoop, repeatTimes, sizePerRepeat, roundMode_, xTypeUbAlignB32_, yUbAlignB8_, yUbAlignB4_);


      inScaleQueue_.FreeTensor(inScaleLocal);

      yQueue_.EnQue<TYtype>(yLocal);
      yLocal = yQueue_.DeQue<TYtype>();

      // copy_out: y
      if constexpr (ifYFloat4e2m1Index_ || ifYFloat4e1m2Index_) {
        DataCopyParams dataCopyYParams;
        dataCopyYParams.blockCount = xDimPerLoop;
        dataCopyYParams.blockLen = tl_->outDimy * sizeof(TYtype) / 2;
        dataCopyYParams.srcStride = 0;
        dataCopyYParams.dstStride = 0;
        DataCopyPad(yGm_.template ReinterpretCast<uint8_t>()[xDimOffsetPerLoop * tl_->outDimy / 2], yFp4Local[0], dataCopyYParams);
        yQueue_.FreeTensor(yLocal);
      } else {
        DataCopyParams dataCopyYParams;
        dataCopyYParams.blockCount = xDimPerLoop;
        dataCopyYParams.blockLen = tl_->outDimy * sizeof(TYtype);
        dataCopyYParams.srcStride = 0;
        dataCopyYParams.dstStride = 0;
        DataCopyPad(yGm_[xDimOffsetPerLoop * tl_->outDimy], yLocal[0], dataCopyYParams);
        yQueue_.FreeTensor(yLocal);
      }
    }
}

template <typename TActScale, typename TQuantScale, typename TGroup, typename TBias, typename TXtype, typename TYtype>
__aicore__ inline void DequantSwigluQuantBaseStatic<TActScale, TQuantScale, TGroup, TBias, TXtype, TYtype>::ProcessSingleGroup(int64_t groupIdx, int64_t realDimx, int64_t groupOffset)
{
  // 计算处理当前的数据时，每个核可以处理几行数据
  int64_t blockDimxFactor = (realDimx + tl_->maxCoreNum - 1) / tl_->maxCoreNum;
  // 在上一步计算每个核可以处理的行数据的前提下，计算处理数据，实际需要多少个核
  realCoreDim_ = (realDimx + blockDimxFactor - 1) / blockDimxFactor;

  // 如果当前核id超过了我实际需要的核数，则说明处理完成了
  if (blockIdx_ < realCoreDim_) {
    // 核间tiling：实际核数计算
    int64_t blockDimxTailFactor = realDimx - blockDimxFactor * (realCoreDim_ - 1); // 最后一个核需要处理的行数
    int64_t xDimPerCore = blockDimxFactor; //blockDimxFactor：每个核上需要处理几行数据
    // 判断是否需要最后一个核处理尾行
    if (blockIdx_ == (realCoreDim_ - 1)) {
        xDimPerCore = blockDimxTailFactor;
    }
    int64_t coreDimxOffset = blockDimxFactor * blockIdx_ + groupOffset;
    ProcessSingleGroupPerCore(groupIdx, xDimPerCore, coreDimxOffset);
  }
}

}  // namespace DequantSwigluQuantV35Ops
#endif  // DEQUANT_SWIGLU_QUANT_STATIC_H
