/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pool_3d_grad_kernel_base.h
 * \brief MaxPool3DGrad/MaxPool3DGradWithArgmax NCDHW kernel 共用的成员字段与 tiling 解析逻辑。
 */

#ifndef POOL_3D_COMMON_ARCH35_POOL_3D_GRAD_KERNEL_BASE_H_
#define POOL_3D_COMMON_ARCH35_POOL_3D_GRAD_KERNEL_BASE_H_

#include <cstdint>

namespace Pool3DGradCommon {

/*
 * 功能：3D 池化梯度 NCDHW kernel 共用基类，收编 46 项公共 tiling 字段对应的成员与解析逻辑。
 * 说明：ParseTilingData 为成员模板，兼容 Pool3DGradNCDHWTilingData 与
 *       MaxPool3DGradWithArgmaxNCDHWTilingData 两种同名字段的 tiling 结构。
 */
class Pool3DGradNcdhwKernelBase {
public:
    template <typename TilingT>
    __aicore__ inline void ParseTilingData(const TilingT& tilingData)
    {
        dArgmax_ = tilingData.dArgmax;
        hArgmax_ = tilingData.hArgmax;
        wArgmax_ = tilingData.wArgmax;

        dOutput_ = tilingData.dOutput;
        hOutput_ = tilingData.hOutput;
        wOutput_ = tilingData.wOutput;

        kernelD_ = tilingData.dKernel;
        kernelH_ = tilingData.hKernel;
        kernelW_ = tilingData.wKernel;

        strideD_ = tilingData.dStride;
        strideH_ = tilingData.hStride;
        strideW_ = tilingData.wStride;

        padD_ = tilingData.padD;
        padH_ = tilingData.padH;
        padW_ = tilingData.padW;

        dilationD_ = tilingData.dilationD;
        dilationH_ = tilingData.dilationH;
        dilationW_ = tilingData.dilationW;

        highAxisInner_ = tilingData.highAxisInner;
        highAxisTail_ = tilingData.highAxisTail;
        highAxisOuter_ = tilingData.highAxisOuter;

        dOutputInner_ = tilingData.dOutputInner;
        dOutputTail_ = tilingData.dOutputTail;
        dOutputOuter_ = tilingData.dOutputOuter;

        hOutputInner_ = tilingData.hOutputInner;
        hOutputTail_ = tilingData.hOutputTail;
        hOutputOuter_ = tilingData.hOutputOuter;

        wOutputInner_ = tilingData.wOutputInner;
        wOutputTail_ = tilingData.wOutputTail;
        wOutputOuter_ = tilingData.wOutputOuter;

        normalCoreProcessNum_ = tilingData.normalCoreProcessNum;
        tailCoreProcessNum_ = tilingData.tailCoreProcessNum;
        usedCoreNum_ = tilingData.usedCoreNum;

        outputBufferSize_ = tilingData.outputBufferSize;
        gradBufferSize_ = tilingData.gradBufferSize;
        argmaxBufferSize_ = tilingData.argmaxBufferSize;

        dProBatchSize_ = tilingData.dProBatchSize;
        hProBatchSize_ = tilingData.hProBatchSize;
        wProBatchSize_ = tilingData.wProBatchSize;
    }

protected:
    uint32_t blockIdx_ = 0;

    int64_t dArgmax_ = 1;
    int64_t hArgmax_ = 1;
    int64_t wArgmax_ = 1;

    int64_t dOutput_ = 1;
    int64_t hOutput_ = 1;
    int64_t wOutput_ = 1;

    int64_t kernelD_ = 1;
    int64_t kernelH_ = 1;
    int64_t kernelW_ = 1;

    int64_t strideD_ = 1;
    int64_t strideH_ = 1;
    int64_t strideW_ = 1;

    int64_t padD_ = 0;
    int64_t padH_ = 0;
    int64_t padW_ = 0;

    int64_t dilationD_ = 1;
    int64_t dilationH_ = 1;
    int64_t dilationW_ = 1;

    int64_t highAxisInner_ = 1;
    int64_t highAxisTail_ = 1;
    int64_t highAxisOuter_ = 1;
    int64_t highAxisActual_ = 1;

    int64_t dOutputInner_ = 1;
    int64_t dOutputTail_ = 1;
    int64_t dOutputOuter_ = 1;
    int64_t dOutputActual_ = 1;

    int64_t hOutputInner_ = 1;
    int64_t hOutputTail_ = 1;
    int64_t hOutputOuter_ = 1;
    int64_t hOutputActual_ = 1;

    int64_t wOutputInner_ = 1;
    int64_t wOutputTail_ = 1;
    int64_t wOutputOuter_ = 1;
    int64_t wOutputActual_ = 1;
    int64_t wOutputAligned_ = 1;

    int64_t normalCoreProcessNum_ = 1;
    int64_t tailCoreProcessNum_ = 1;
    int64_t curCoreProcessNum_ = 1;
    int64_t usedCoreNum_ = 1;

    int64_t outputBufferSize_ = 1;
    int64_t gradBufferSize_ = 1;
    int64_t argmaxBufferSize_ = 1;

    int64_t highAxisIndex_ = 0;
    int64_t hAxisIndex_ = 0;
    int64_t wAxisIndex_ = 0;
    int64_t dAxisIndex_ = 0;

    int64_t hArgmaxActual_ = 0;
    int64_t dArgmaxActual_ = 0;
    int64_t wArgmaxActual_ = 0;
    int64_t wArgmaxAligned_ = 0;

    int64_t highAxisArgmaxOffset_ = 0;
    int64_t hAxisArgmaxOffset_ = 0;
    int64_t dAxisArgmaxOffset_ = 0;
    int64_t wAxisArgmaxOffset_ = 0;

    int64_t argmaxPlaneSize_ = 1;

    int64_t dProBatchSize_ = 1;
    int64_t hProBatchSize_ = 1;
    int64_t wProBatchSize_ = 1;
    int64_t curDProBatchSize_ = 1;
    int64_t curHProBatchSize_ = 1;
    int64_t curWProBatchSize_ = 1;
};

} // namespace Pool3DGradCommon

#endif // POOL_3D_COMMON_ARCH35_POOL_3D_GRAD_KERNEL_BASE_H_
