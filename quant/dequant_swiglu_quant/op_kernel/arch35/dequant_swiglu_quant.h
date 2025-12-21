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
 * \file dequant_swiglu_quant.h
 * \brief
 */

#ifndef DEQUANT_SWIGLU_QUANT_H
#define DEQUANT_SWIGLU_QUANT_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "dequant_swiglu_quant_common.h"

namespace DequantSwigluQuantV35Ops {
using namespace AscendC;

template <typename TActScale, typename TQuantScale, typename TGroup, typename TBias, typename TXtype, typename TYtype>
class DequantSwigluQuantBase {
 public:
  static constexpr bool hasActScale_ = IsSameType<TActScale, float>::value;
  static constexpr bool hasQuantScale_ = IsSameType<TQuantScale, float>::value;
  static constexpr bool hasGroupIndex_ = IsSameType<TGroup, int64_t>::value;
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

  __aicore__ inline DequantSwigluQuantBase(TPipe* pipe) {
    pipe_ = pipe;
  };

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR weightScale, GM_ADDR activationScale, GM_ADDR bias, GM_ADDR quantScale,
                              GM_ADDR quantOffset, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR scale,
                              const DequantSwigluQuantV35BaseTilingData* tilingData);

  __aicore__ inline void Process();

 private:
  __aicore__ inline void ComputeReduceMax(const LocalTensor<float>& tempRes, int32_t calCount, float& maxValue);
  __aicore__ inline void ProcessSingleGroup(int64_t groupIndex);

 protected:
  /* global memory address */
  // input global mem
  GlobalTensor<TXtype> xGm_; // 当前模板支持 fp16, bf16, 用模板参数去承载输入x的类型
  GlobalTensor<float> weightScaleGm_;
  GlobalTensor<TActScale> activationScaleGm_;
  GlobalTensor<TBias> biasGm_;  // 当前模板支持 float，float16，bf16，int32， 用模板参数去承载bias的类型
  GlobalTensor<TQuantScale> quantScaleGm_;
  GlobalTensor<float> quantOffsetGm_; // 本次不支持
  GlobalTensor<TGroup> groupIndexGm_;

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

  /* ascendc variable */
  TPipe* pipe_ = nullptr;
  TQue<QuePosition::VECIN, 1> xActQueue_;
  TQue<QuePosition::VECIN, 1> inScaleQueue_;
  TQue<QuePosition::VECIN, 1> biasQueue_;
  TQue<QuePosition::VECOUT, 1> yQueue_;
  TQue<QuePosition::VECOUT, 1> scaleQueue_;

  TBuf<> tmpBuffer;

  uint32_t blockIdx_ = GetBlockIdx();
  uint32_t realCoreDim_ = 0;
  int64_t realDimx_ = 0;
  int64_t groupOffset_ = 0;
  uint32_t xUbAlignB32_ = 0; // 根据4Byte计算的32B对齐，用于float32类型
  uint32_t xTypeUbAlignB32_ = 0; // 根据x的输入类型不同计算的32B对齐
  uint32_t yUbAlignB8_ = 0;
  uint32_t yUbAlignB4_ = 0;
  uint32_t aScaleUbAlignB32_ = 0;
  uint32_t biasUbAlign_ = 0; // bias的32B对齐标记
  int64_t roundMode_ = 0; // 溢出模式的标识
  float scalarMaxNum_ = 127.0; //根据不同的输出类型，对应的最大值不同，默认127

  const DequantSwigluQuantV35BaseTilingData* tl_ = nullptr;
};
// 公共函数实现

template <typename TActScale, typename TQuantScale, typename TGroup, typename TBias, typename TXtype, typename TYtype>
__aicore__ inline void DequantSwigluQuantBase<TActScale, TQuantScale, TGroup, TBias, TXtype, TYtype>::Init(
    GM_ADDR x, GM_ADDR weightScale, GM_ADDR activationScale, GM_ADDR bias, GM_ADDR quantScale, GM_ADDR quantOffset,
    GM_ADDR groupIndex, GM_ADDR y, GM_ADDR scale, const DequantSwigluQuantV35BaseTilingData* tilingData)
{
  tl_ = tilingData;
  // 兼容bf16和float16类型，xUbAlign对齐点适配修改；如果是bf16，float16，则是用2B对对齐32B；如果是int32，则是用4B对齐32B
  if constexpr (ifXBf16Index_ || ifXFloat16Index_) {
    uint32_t BLOCK_ELEM_B32_BF = BLOCK_SIZE / sizeof(TXtype);
    xTypeUbAlignB32_ = CeilDivision(tl_->UbFactorDimy, BLOCK_ELEM_B32_BF) * BLOCK_ELEM_B32_BF;
  } else {
    xTypeUbAlignB32_ = CeilDivision(tl_->UbFactorDimy, BLOCK_ELEM_B32) * BLOCK_ELEM_B32;
  }

  // 4B的数据类型对应的32B对齐点
  xUbAlignB32_ = CeilDivision(tl_->UbFactorDimy, BLOCK_ELEM_B32) * BLOCK_ELEM_B32;

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

  xGm_.SetGlobalBuffer((__gm__ TXtype*)x);
  weightScaleGm_.SetGlobalBuffer((__gm__ float*)weightScale);
  activationScaleGm_.SetGlobalBuffer((__gm__ TActScale*)activationScale);
  quantScaleGm_.SetGlobalBuffer((__gm__ TQuantScale*)quantScale);
  if constexpr (hasGroupIndex_) {
    groupIndexGm_.SetGlobalBuffer((__gm__ TGroup*)groupIndex);
  }

  // yGm
  yGm_.SetGlobalBuffer((__gm__ TYtype*)y);
  scaleGm_.SetGlobalBuffer((__gm__ float*)scale);

  // x + activation_scale
  pipe_->InitBuffer(xActQueue_, DOUBLE_BUFFER,
                    (tl_->UbFactorDimx * xTypeUbAlignB32_ * 2) * sizeof(TXtype) +
                    (tl_->UbFactorDimx * aScaleUbAlignB32_) * sizeof(float));
  // weight_scale + quant_scale
  pipe_->InitBuffer(inScaleQueue_, 1, (xUbAlignB32_ * 2 + xUbAlignB32_) * sizeof(float));

  // y
  pipe_->InitBuffer(yQueue_, DOUBLE_BUFFER, tl_->UbFactorDimx * yUbAlignB8_ * sizeof(TYtype));

  pipe_->InitBuffer(scaleQueue_, 1, tl_->UbFactorDimx * aScaleUbAlignB32_ * sizeof(float));

  pipe_->InitBuffer(tmpBuffer, tl_->UbFactorDimx * xUbAlignB32_ * sizeof(float));

  // 如果有bias入参，则给bias进行地址申请
  if constexpr (hasBiasIndex_) {
    biasGm_.SetGlobalBuffer((__gm__ TBias*)bias);
    // bias (1 * 2H) 申请
    pipe_->InitBuffer(biasQueue_, 1, (biasUbAlign_ * 2) * sizeof(TBias));
  }
}

template <typename TActScale, typename TQuantScale, typename TGroup, typename TBias, typename TXtype, typename TYtype>
__aicore__ inline void DequantSwigluQuantBase<TActScale, TQuantScale, TGroup, TBias, TXtype, TYtype>::Process()
{
  if constexpr (!hasGroupIndex_) {
    realDimx_ = tl_->inDimx;
    if (realDimx_ < 0) {
      realDimx_ = 0;
    }

    ProcessSingleGroup(0);
    return;
  }

  groupOffset_ = 0;
  for (int64_t groupIndex = 0; groupIndex < tl_->inGroupNum; groupIndex++) {
    realDimx_ = static_cast<int64_t>(groupIndexGm_(groupIndex));
    if (realDimx_ < 0) {
      realDimx_ = 0;
    }

    // inDimx x的元素总数 / x的-1维shape，也即按照-1维的循环次数，也即总行数（按照二维理解）
    if (realDimx_ > 0 && groupOffset_ < tl_->inDimx) {
      ProcessSingleGroup(groupIndex);
      groupOffset_ += realDimx_;
    }
  }
}

template <typename TActScale, typename TQuantScale, typename TGroup, typename TBias, typename TXtype, typename TYtype>
__aicore__ inline void DequantSwigluQuantBase<TActScale, TQuantScale, TGroup, TBias, TXtype, TYtype>::ProcessSingleGroup(int64_t groupIdx)
{
  // 计算处理当前的数据时，每个核可以处理几行数据
  int64_t blockDimxFactor = (realDimx_ + tl_->maxCoreNum - 1) / tl_->maxCoreNum;
  // 在上一步计算每个核可以处理的行数据的前提下，计算处理数据，实际需要多少个核
  realCoreDim_ = (realDimx_ + blockDimxFactor - 1) / blockDimxFactor;

  // 如果当前核id超过了我实际需要的核数，则说明处理完成了
  if (blockIdx_ < realCoreDim_) {
    DataCopyPadParams padParams{false, 0, 0, 0};

    // 激活左/右部分偏移，actRight是表示是否激活右半部分
    int64_t actOffset = tl_->actRight * tl_->UbFactorDimy;
    int64_t gateOffset = tl_->UbFactorDimy - actOffset;

    // weight_scale搬入
    LocalTensor<float> inScaleLocal = inScaleQueue_.AllocTensor<float>();
    if constexpr (ifXIntIndex_) {
      // copy_in: weight_scale(G, H) 激活部分
      DataCopyParams dataCopyWeightScaleParams;
      dataCopyWeightScaleParams.blockCount = 1;
      dataCopyWeightScaleParams.blockLen = tl_->UbFactorDimy * sizeof(float);
      dataCopyWeightScaleParams.srcStride = 0;
      dataCopyWeightScaleParams.dstStride = 0;
      DataCopyPad(inScaleLocal[0], weightScaleGm_[groupIdx * tl_->inDimy + actOffset], dataCopyWeightScaleParams, padParams);

      // copy_in: weight_scale(G, H) 门控部分
      DataCopyPad(inScaleLocal[xUbAlignB32_], weightScaleGm_[groupIdx * tl_->inDimy + gateOffset], dataCopyWeightScaleParams, padParams);
    }

    // copy_in: quant_scale(G, H)
    if constexpr (hasQuantScale_) {
      DataCopyParams dataCopyQuantScaleParams;
      dataCopyQuantScaleParams.blockCount = 1;
      dataCopyQuantScaleParams.blockLen = tl_->UbFactorDimy * sizeof(TQuantScale);
      dataCopyQuantScaleParams.srcStride = 0;
      dataCopyQuantScaleParams.dstStride = 0;
      DataCopyPad(inScaleLocal[xUbAlignB32_ * 2], quantScaleGm_[groupIdx * tl_->UbFactorDimy], dataCopyQuantScaleParams, padParams);
    }
    inScaleQueue_.EnQue(inScaleLocal);
    inScaleLocal = inScaleQueue_.DeQue<float>();

    // copy_in: bias(1, 2H)->(1, H), (1, H)
    if constexpr (hasBiasIndex_) {
      biasLocal = biasQueue_.AllocTensor<TBias>();
      DataCopyParams dataCopyBiasParams;
      dataCopyBiasParams.blockCount = 1;
      dataCopyBiasParams.blockLen = tl_->UbFactorDimy * sizeof(TBias);
      dataCopyBiasParams.srcStride = 0;
      dataCopyBiasParams.dstStride = 0;
      DataCopyPad(biasLocal[0], biasGm_[actOffset], dataCopyBiasParams, padParams);
      DataCopyPad(biasLocal[biasUbAlign_], biasGm_[gateOffset], dataCopyBiasParams, padParams);
      biasQueue_.EnQue(biasLocal);
      biasLocal = biasQueue_.DeQue<TBias>();
    }

    // 核间tiling：实际核数计算
    int64_t blockDimxTailFactor = realDimx_ - blockDimxFactor * (realCoreDim_ - 1); // 最后一个核需要处理的行数
    int64_t xDimPerCore = blockDimxFactor; //blockDimxFactor：每个核上需要处理几行数据
    // 判断是否需要最后一个核处理尾行
    if (blockIdx_ == (realCoreDim_ - 1)) {
        xDimPerCore = blockDimxTailFactor;
    }
    // UbFactorDimX：ub每次能处理的行数，ubDimxLoop: 每个核上需要处理几行数据，然后在单个核上，ub处理这几行数据需要几次循环
    int64_t ubDimxLoop = (xDimPerCore + tl_->UbFactorDimx - 1) / tl_->UbFactorDimx;
    int64_t ubDimxTailFactor = xDimPerCore - tl_->UbFactorDimx * (ubDimxLoop - 1);
    int64_t xDimOffsetCore = blockDimxFactor * blockIdx_; // blockIdx_:当前核的idx，从0开始；xDimOffsetCore：当前核要从第几行开始处理

    // 核内循环
    for (uint64_t i = 0; i < ubDimxLoop; i++) {
      // 核内循环每次计算时的起始地址：当前核计算时的起始地址 + 第i次核内循环要处理的起始地址 + 已处理过的行偏移(历史已处理过的总行数)
      int64_t xDimOffsetPerLoop = xDimOffsetCore + i * tl_->UbFactorDimx + groupOffset_;
      int64_t xDimPerLoop = tl_->UbFactorDimx;
      if (i == ubDimxLoop - 1) {
        xDimPerLoop = ubDimxTailFactor;
      }

      // copy_in: x(xDimPerLoop, H) 激活部分
      LocalTensor<TXtype> xActLocal = xActQueue_.AllocTensor<TXtype>();
      DataCopyParams dataCopyXParams;
      dataCopyXParams.blockCount = xDimPerLoop;
      dataCopyXParams.blockLen = tl_->UbFactorDimy * sizeof(TXtype);
      dataCopyXParams.srcStride = tl_->UbFactorDimy * sizeof(TXtype);
      dataCopyXParams.dstStride = 0;
      DataCopyPad(xActLocal[0], xGm_[xDimOffsetPerLoop * tl_->inDimy + actOffset], dataCopyXParams, padParams);

      // copy_in: x(xDimPerLoop, H) 门控部分
      DataCopyPad(xActLocal[xTypeUbAlignB32_ * xDimPerLoop], xGm_[xDimOffsetPerLoop * tl_->inDimy + gateOffset], dataCopyXParams, padParams);

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

      LocalTensor<float> scaleLocal = scaleQueue_.AllocTensor<float>();
      LocalTensor<float> tmpXLocal = tmpBuffer.Get<float>();

      __local_mem__ float* tmpXPtr = (__local_mem__ float*)tmpXLocal.GetPhyAddr();
      __local_mem__ TYtype* yPtr = (__local_mem__ TYtype*)yLocal.GetPhyAddr();
      __local_mem__ uint8_t* yFp4Ptr = (__local_mem__ uint8_t*)yFp4Local.GetPhyAddr();

      __local_mem__ float* scalePtr = (__local_mem__ float*)scaleLocal.GetPhyAddr();
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

      __VEC_SCOPE__
      {
        AscendC::MicroAPI::RegTensor<TXtype> vreg0, vreg10;
        AscendC::MicroAPI::RegTensor<float> vreg1, vreg2, vreg3, vreg4, vreg5, vreg6;
        AscendC::MicroAPI::RegTensor<float> vreg7, vreg8, vreg9, vreg11, vreg12;
        AscendC::MicroAPI::RegTensor<float> vreg13, vreg14, vreg15;
        AscendC::MicroAPI::RegTensor<int32_t> vreg16, vreg17, verg18, vreg19;
        AscendC::MicroAPI::RegTensor<float> vreg20, vreg21;
        AscendC::MicroAPI::RegTensor<half> vreg24, vreg25;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vreg26, vreg27;
        AscendC::MicroAPI::MaskReg mask;

        // int32： 256B / 4B = 64次 ，bf16 or float16：256B / 2B = 128次
        constexpr uint16_t sizePerRepeat = AscendC::GetVecLen() / sizeof(float);
        uint32_t width = tl_->UbFactorDimy; // 输出y的-1轴对应的shape大小，也即H
        uint16_t repeatTimes = CeilDivision(tl_->UbFactorDimy, sizePerRepeat); // 向上取整
        const float scalarOne = 1.0;

        // 每次可以处理256B的数据，256B / 4B = 64 可以处理多少个float元素，repeatTimes：处理H个float元素需要的循环次数
        for (uint16_t j = 0; j < repeatTimes; j++) {
            mask = AscendC::MicroAPI::UpdateMask<uint32_t>(width);

            // 计算，当x=int32时，weight_scale才参与计算
            if constexpr (ifXIntIndex_) {
              wScale1Addr = wScale1Ptr + j * sizePerRepeat;
              wScale2Addr = wScale2Ptr + j * sizePerRepeat;

              AscendC::MicroAPI::DataCopy(vreg2, wScale1Addr);
              AscendC::MicroAPI::DataCopy(vreg12, wScale2Addr);
            }

            // ub的循环次数，也即ub每次可以处理多少行数据
            for (uint16_t i = 0; i < static_cast<uint16_t>(xDimPerLoop); i++) {
                // x的数据类型变换之后，对齐点变化了，应该用xTypeUb参数
                auto x1Addr = x1Ptr + i * xTypeUbAlignB32_ + j * sizePerRepeat;
                auto x2Addr = x2Ptr + i * xTypeUbAlignB32_ + j * sizePerRepeat;
                auto dstAddr = tmpXPtr + i * xTypeUbAlignB32_ + j * sizePerRepeat;

                // vreg0 -> x1, vreg10 -> x2
                // bf16和float16场景下需要修改，新增搬运模板参数DIST_UNPACK_B16
                if constexpr (ifXFloat16Index_) {
                  AscendC::MicroAPI::DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg0, x1Addr);
                  AscendC::MicroAPI::DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg10, x2Addr);
                }
                if constexpr (ifXBf16Index_) {
                  AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg0, x1Addr);
                  AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg10, x2Addr);
                }
                if constexpr (ifXIntIndex_) {
                  AscendC::MicroAPI::DataCopy(vreg0, x1Addr);
                  AscendC::MicroAPI::DataCopy(vreg10, x2Addr);
                }

                // 如果bias有值，且x=int32，bias=int32，则先将x+bias
                if constexpr (hasBiasIndex_) {
                  // 使用bias的地址指针时，做判空处理
                  if constexpr (ifXIntIndex_ && ifBiasIntIndex_) {
                    auto bias1Addr = bias1Ptr + j * sizePerRepeat;
                    auto bias2Addr = bias2Ptr + j * sizePerRepeat;
                    AscendC::MicroAPI::DataCopy(vreg16, bias1Addr);
                    AscendC::MicroAPI::DataCopy(vreg17, bias2Addr);
                    //x + bias
                    AscendC::MicroAPI::Add(vreg0, vreg0, vreg16, mask);
                    AscendC::MicroAPI::Add(vreg10, vreg10, vreg17, mask);
                  }
                }

                // x:int32->float32
                if constexpr (ifXIntIndex_) {
                  AscendC::MicroAPI::Cast<float, TXtype, CAST_INT32_TO_FP32>(vreg1, vreg0, mask);
                  AscendC::MicroAPI::Cast<float, TXtype, CAST_INT32_TO_FP32>(vreg11, vreg10, mask);
                  // x * weight
                  AscendC::MicroAPI::Mul(vreg3, vreg1, vreg2, mask);
                  AscendC::MicroAPI::Mul(vreg13, vreg11, vreg12, mask);
                }
                // x:float16, bfloat16 -> float32, 不进行x * weight
                if constexpr (ifXBf16Index_ || ifXFloat16Index_) {
                  AscendC::MicroAPI::Cast<float, TXtype, CAST_BF16_FP16_TO_FP32>(vreg3, vreg0, mask);
                  AscendC::MicroAPI::Cast<float, TXtype, CAST_BF16_FP16_TO_FP32>(vreg13, vreg10, mask);
                }

                // x * activation_scale
                if constexpr (hasActScale_) {
                  auto aScaleAddr = aScalePtr + i * aScaleUbAlignB32_;
                  AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vreg4, aScaleAddr);
                  AscendC::MicroAPI::Mul(vreg3, vreg3, vreg4, mask);
                  AscendC::MicroAPI::Mul(vreg13, vreg13, vreg4, mask);
                }

                // 如果bias有值，且x!=int32 && bias!=int32，则先将dequant后的结果加上bias
                if constexpr (ifBiasFloatIndex_ || ifBiasFloat16Index_ || ifBiasBfloat16Index_) {
                  // 确认bias的计算地址
                  auto bias1Addr = bias1Ptr + j * sizePerRepeat;
                  auto bias2Addr = bias2Ptr + j * sizePerRepeat;
                  if constexpr (ifBiasFloatIndex_) {
                    AscendC::MicroAPI::DataCopy(vreg20, bias1Addr);
                    AscendC::MicroAPI::DataCopy(vreg21, bias2Addr);
                  }
                  if constexpr (ifBiasFloat16Index_) {
                    AscendC::MicroAPI::DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg24, bias1Addr);
                    AscendC::MicroAPI::DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg25, bias2Addr);
                    AscendC::MicroAPI::Cast<float, half, CAST_BF16_FP16_TO_FP32>(vreg20, vreg24, mask);
                    AscendC::MicroAPI::Cast<float, half, CAST_BF16_FP16_TO_FP32>(vreg21, vreg25, mask);
                  }
                  if constexpr (ifBiasBfloat16Index_) {
                    AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg26, bias1Addr);
                    AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg27, bias2Addr);
                    AscendC::MicroAPI::Cast<float, bfloat16_t, CAST_BF16_FP16_TO_FP32>(vreg20, vreg26, mask);
                    AscendC::MicroAPI::Cast<float, bfloat16_t, CAST_BF16_FP16_TO_FP32>(vreg21, vreg27, mask);
                  }
                  // 将dequant后的结果加上bias
                  AscendC::MicroAPI::Add(vreg3, vreg3, vreg20, mask);
                  AscendC::MicroAPI::Add(vreg13, vreg13, vreg21, mask);
                }

                // Swish
                AscendC::MicroAPI::Muls(vreg6, vreg3, -(scalarOne), mask);
                AscendC::MicroAPI::Exp(vreg7, vreg6, mask);
                AscendC::MicroAPI::Adds(vreg8, vreg7, scalarOne, mask);
                AscendC::MicroAPI::Div<float, &DIV_MODE>(vreg9, vreg3, vreg8, mask);

                // glu
                AscendC::MicroAPI::Mul(vreg15, vreg9, vreg13, mask);

                // store: reg->ub
                AscendC::MicroAPI::DataCopy(dstAddr, vreg15, mask);
            }
        }
      } // __VEC_SCOPE__
      xActQueue_.FreeTensor(xActLocal);
      // biasLocal是否为空的判断
      if constexpr (hasBiasIndex_) {
        biasQueue_.FreeTensor(biasLocal);
      }

      __local_mem__ float* qScalePtr = (__local_mem__ float*)inScaleLocal.GetPhyAddr(xUbAlignB32_ * 2);

      __VEC_SCOPE__
      {
          AscendC::MicroAPI::RegTensor<float> vreg0, vreg1, vreg2, vreg3, vreg4, vreg5, vreg6, vreg7, vreg8, vregDiv;
          AscendC::MicroAPI::RegTensor<float> vregTmpX;
          AscendC::MicroAPI::RegTensor<float> vreg16;
          AscendC::MicroAPI::RegTensor<int16_t> vreg17;
          AscendC::MicroAPI::RegTensor<half> vreg18;
          AscendC::MicroAPI::RegTensor<int8_t> vreg19;
          AscendC::MicroAPI::MaskReg mask, maskTail, maskOne;

          constexpr uint16_t sizePerRepeat = AscendC::GetVecLen() / sizeof(float);
          uint16_t repeatTimes = CeilDivision(tl_->UbFactorDimy, sizePerRepeat);

          uint32_t block = static_cast<uint32_t>(sizePerRepeat);
          uint32_t tailBlock = tl_->UbFactorDimy - sizePerRepeat * (repeatTimes - 1);
          uint32_t numOne = 1;
          mask = AscendC::MicroAPI::UpdateMask<uint32_t>(block);
          maskTail = AscendC::MicroAPI::UpdateMask<uint32_t>(tailBlock);
          maskOne = AscendC::MicroAPI::UpdateMask<uint32_t>(numOne);  // 用于读写1个元素

          float scalarOne = 1.0;
          float scalarZero = 0;
          // 不同的输出数据类型对应的最大值不同
          float scalarMaxNum = scalarMaxNum_;

          AscendC::MicroAPI::Duplicate(vregDiv, scalarMaxNum);

          for (uint16_t i = 0; i < static_cast<uint16_t>(xDimPerLoop); i++) {
            auto scaleAddr = scalePtr + i * aScaleUbAlignB32_;
            AscendC::MicroAPI::Duplicate(vregTmpX, scalarZero);

            // 先处理尾块
            uint16_t j = repeatTimes - 1;
            auto tmpXAddr = tmpXPtr + i * xTypeUbAlignB32_ + j * sizePerRepeat;
            AscendC::MicroAPI::DataCopy(vreg0, tmpXAddr);

            // x * quant_scale
            if constexpr (hasQuantScale_) {
              auto qScaleAddr = qScalePtr + j * sizePerRepeat;
              AscendC::MicroAPI::DataCopy(vreg1, qScaleAddr);
              AscendC::MicroAPI::Mul(vreg0, vreg0, vreg1, maskTail);
              AscendC::MicroAPI::DataCopy(tmpXAddr, vreg0, maskTail);
            }

            // 循环求行内最大值
            AscendC::MicroAPI::Abs(vreg3, vreg0, maskTail);
            AscendC::MicroAPI::Max(vregTmpX, vregTmpX, vreg3, maskTail);

            // 整块处理
            for (uint16_t j = 0; j < static_cast<uint16_t>(repeatTimes - 1); j++) {
              auto tmpXAddr = tmpXPtr + i * xTypeUbAlignB32_ + j * sizePerRepeat;
              AscendC::MicroAPI::DataCopy(vreg0, tmpXAddr);

              // x * quant_scale
              if constexpr (hasQuantScale_) {
                auto qScaleAddr = qScalePtr + j * sizePerRepeat;
                AscendC::MicroAPI::DataCopy(vreg1, qScaleAddr);
                AscendC::MicroAPI::Mul(vreg0, vreg0, vreg1, mask);
                AscendC::MicroAPI::DataCopy(tmpXAddr, vreg0, mask);
              }

              // 循环求行内最大值
              AscendC::MicroAPI::Abs(vreg3, vreg0, mask);
              AscendC::MicroAPI::Max(vregTmpX, vregTmpX, vreg3, mask);
            }

            // vreg[0]为最大值，vreg[1]为最大值对应索引
            AscendC::MicroAPI::ReduceMax(vreg4, vregTmpX, mask);
            AscendC::MicroAPI::Div(vreg5, vreg4, vregDiv, mask);
            // 拷贝第一个数到UB
            AscendC::MicroAPI::DataCopy(scaleAddr, vreg5, maskOne);
          }
      } // __VEC_SCOPE__

      scaleQueue_.EnQue<float>(scaleLocal);
      scaleLocal = scaleQueue_.DeQue<float>();

      // copy_out: scale
      DataCopyParams dataCopyScaleParams;
      dataCopyScaleParams.blockCount = xDimPerLoop;
      dataCopyScaleParams.blockLen = sizeof(float);
      dataCopyScaleParams.srcStride = 0;
      dataCopyScaleParams.dstStride = 0;
      DataCopyPad(scaleGm_[xDimOffsetPerLoop], scaleLocal[0], dataCopyScaleParams);

      __VEC_SCOPE__
      {
        constexpr uint16_t sizePerRepeat = AscendC::GetVecLen() / sizeof(float);
        uint16_t repeatTimes = CeilDivision(tl_->UbFactorDimy, sizePerRepeat);

        AscendC::MicroAPI::RegTensor<float> vreg6, vreg7, vreg8;
        AscendC::MicroAPI::RegTensor<int16_t> vreg9;
        AscendC::MicroAPI::RegTensor<half> vreg10;
        AscendC::MicroAPI::RegTensor<int8_t> vreg11;
        AscendC::MicroAPI::RegTensor<fp8_e4m3fn_t> vreg12;
        AscendC::MicroAPI::RegTensor<fp8_e5m2_t> vreg13;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vreg14, vreg15;
        AscendC::MicroAPI::RegTensor<fp4x2_e2m1_t> vreg16;
        AscendC::MicroAPI::RegTensor<fp4x2_e1m2_t> vreg17;
        AscendC::MicroAPI::RegTensor<uint16_t> yRegTensor;
        AscendC::MicroAPI::RegTensor<uint8_t> out;

        AscendC::MicroAPI::MaskReg maskFull8;
        AscendC::MicroAPI::MaskReg mask;
        uint32_t fp4Width = 32;

        maskFull8 = AscendC::MicroAPI::UpdateMask<uint32_t>(fp4Width);

        for (uint16_t i = 0; i < static_cast<uint16_t>(xDimPerLoop); i++) {
          auto scaleAddr = scalePtr + i * aScaleUbAlignB32_;
          AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vreg6, scaleAddr);

          uint32_t width = static_cast<uint32_t>(tl_->UbFactorDimy);
          for(uint16_t j = 0; j < repeatTimes; j++) {
            mask = AscendC::MicroAPI::UpdateMask<uint32_t>(width);

            auto tmpXAddr = tmpXPtr + i * xTypeUbAlignB32_ + j * sizePerRepeat;
            auto yAddr = yPtr + i * yUbAlignB8_ + j * sizePerRepeat;
            auto yFp4Addr = yFp4Ptr + i * yUbAlignB4_ + (j * sizePerRepeat / 2);

            AscendC::MicroAPI::DataCopy(vreg7, tmpXAddr);
            AscendC::MicroAPI::Div(vreg8, vreg7, vreg6, mask);

            // 根据输出类型进行不同的cast操作
            if constexpr (ifYFloat8e4m3Index_) {
              // float32 -> float8_e4m3
              AscendC::MicroAPI::Cast<fp8_e4m3fn_t, float, CAST_FP32_TO_FP8>(vreg12, vreg8, mask);
              AscendC::MicroAPI::DataCopy<fp8_e4m3fn_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(yAddr, vreg12, mask);
            } else if constexpr (ifYFloat8e5m2Index_) {
              // float32 -> float8_e5m2
              AscendC::MicroAPI::Cast<fp8_e5m2_t, float, CAST_FP32_TO_FP8>(vreg13, vreg8, mask);
              AscendC::MicroAPI::DataCopy<fp8_e5m2_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(yAddr, vreg13, mask);
            } else if constexpr (ifYFloat4e2m1Index_) {
              // float32 -> bfloat16 -> float4_e2m1
              AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(vreg14, vreg8, mask);
              AscendC::MicroAPI::Pack((AscendC::MicroAPI::RegTensor<uint16_t>&)vreg14, (AscendC::MicroAPI::RegTensor<uint32_t>&)vreg14);
              // 获取对应的CastTrait
              if (roundMode_ == 1) {
                AscendC::MicroAPI::Cast<fp4x2_e2m1_t, bfloat16_t, CAST_BF16_TO_FP4_ROUND>(vreg16, vreg14, mask);
              } else if (roundMode_ == 2) {
                AscendC::MicroAPI::Cast<fp4x2_e2m1_t, bfloat16_t, CAST_BF16_TO_FP4_FLOOR>(vreg16, vreg14, mask);
              } else if (roundMode_ == 3) {
                AscendC::MicroAPI::Cast<fp4x2_e2m1_t, bfloat16_t, CAST_BF16_TO_FP4_CEIL>(vreg16, vreg14, mask);
              } else if (roundMode_ == 4) {
                AscendC::MicroAPI::Cast<fp4x2_e2m1_t, bfloat16_t, CAST_BF16_TO_FP4_TRUNC>(vreg16, vreg14, mask);
              } else {
                AscendC::MicroAPI::Cast<fp4x2_e2m1_t, bfloat16_t, CAST_BF16_TO_FP4_RINT>(vreg16, vreg14, mask);
              }
              // 搬出
              AscendC::MicroAPI::DataCopy<uint8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(yFp4Addr, (AscendC::MicroAPI::RegTensor<uint8_t>&)vreg16, maskFull8);
            } else if constexpr (ifYFloat4e1m2Index_) {
              // float32 -> bfloat16 -> float4_e1m2
              AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(vreg15, vreg8, mask);
              AscendC::MicroAPI::Pack((AscendC::MicroAPI::RegTensor<uint16_t>&)vreg15, (AscendC::MicroAPI::RegTensor<uint32_t>&)vreg15);
              // 获取对应的CastTrait
              if (roundMode_ == 1) {
                AscendC::MicroAPI::Cast<fp4x2_e1m2_t, bfloat16_t, CAST_BF16_TO_FP4_ROUND>(vreg17, vreg15, mask);
              } else if (roundMode_ == 2) {
                AscendC::MicroAPI::Cast<fp4x2_e1m2_t, bfloat16_t, CAST_BF16_TO_FP4_FLOOR>(vreg17, vreg15, mask);
              } else if (roundMode_ == 3) {
                AscendC::MicroAPI::Cast<fp4x2_e1m2_t, bfloat16_t, CAST_BF16_TO_FP4_CEIL>(vreg17, vreg15, mask);
              } else if (roundMode_ == 4) {
                AscendC::MicroAPI::Cast<fp4x2_e1m2_t, bfloat16_t, CAST_BF16_TO_FP4_TRUNC>(vreg17, vreg15, mask);
              } else {
                AscendC::MicroAPI::Cast<fp4x2_e1m2_t, bfloat16_t, CAST_BF16_TO_FP4_RINT>(vreg17, vreg15, mask);
              }
              // 搬出
              AscendC::MicroAPI::DataCopy<uint8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(yFp4Addr, (AscendC::MicroAPI::RegTensor<uint8_t>&)vreg17, maskFull8);
            } else {
              // float32 -> int8
              AscendC::MicroAPI::Cast<int16_t, float, CAST_FP32_TO_INT16>(vreg9, vreg8, mask);
              AscendC::MicroAPI::Cast<half, int16_t, CAST_INT16_TO_FP16>(vreg10, vreg9, mask);
              AscendC::MicroAPI::Cast<int8_t, half, CAST_FP16_TO_INT8>(vreg11, vreg10, mask);

              AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(yAddr, vreg11, mask);
            }
          }
        }
      } // __VEC_SCOPE__
      inScaleQueue_.FreeTensor(inScaleLocal);
      scaleQueue_.FreeTensor(scaleLocal);

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
}

}  // namespace DequantSwigluQuantV35Ops
#endif  // DEQUANT_SWIGLU_QUANT_H
