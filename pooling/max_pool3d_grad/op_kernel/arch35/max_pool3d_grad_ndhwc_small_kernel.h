/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MAX_POOL3D_GRAD_NDHWC_SMALL_KERNEL_H
#define MAX_POOL3D_GRAD_NDHWC_SMALL_KERNEL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "max_pool3d_grad_ndhwc_impl_scatter.h"

namespace MaxPool3DGradNDHWCNameSpace {

using namespace AscendC;
using namespace Pool3DGradNameSpace;

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
class MaxPool3DGradNDHWCSmallKernel : public MaxPool3DGradNDHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE> {
    using Base = MaxPool3DGradNDHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>;

public:
    __aicore__ inline MaxPool3DGradNDHWCSmallKernel() {}

    __aicore__ inline void Init(GM_ADDR origX, GM_ADDR origY, GM_ADDR grad, GM_ADDR y,
                                const Pool3DGradNDHWCTilingData& tilingData);
    __aicore__ inline void Process();

    using Base::argmaxBuff_;
    using Base::argmaxBufferSize_;
    using Base::BLOCK_SIZE;
    using Base::blockIdx_;
    using Base::cAligned_;
    using Base::cAxisGradOffset_;
    using Base::cAxisIndex_;
    using Base::cDim_;
    using Base::cOutputActual_;
    using Base::curCoreProcessNum_;
    using Base::dArgmaxActual_;
    using Base::dArgmaxActualEnd_;
    using Base::dArgmaxActualStart_;
    using Base::dAxisIndex_;
    using Base::dOutputActual_;
    using Base::gradGm_;
    using Base::gradQue_;
    using Base::hArgmaxActual_;
    using Base::hArgmaxActualEnd_;
    using Base::hArgmaxActualStart_;
    using Base::hAxisIndex_;
    using Base::hOutputActual_;
    using Base::isPad_;
    using Base::mte3ToMte2Event_;
    using Base::nAxisIndex_;
    using Base::nOutputActual_;
    using Base::outputQue_;
    using Base::tilingData_;
    using Base::V_REG_SIZE;
    using Base::vlT2_;
    using Base::vToMte2Event_;
    using Base::wArgmaxActual_;
    using Base::wArgmaxActualEnd_;
    using Base::wArgmaxActualStart_;
    using Base::wAxisIndex_;
    using Base::wOutputActual_;
    using Base::yGm_;

private:
    __aicore__ inline void ForwardScalarCompute();
    __aicore__ inline void ForwardCopyIn();
    __aicore__ inline void Forward();
    __aicore__ inline void ForwardComputeTile();

    __simd_callee__ inline void MaxPoolSingleChannelWithArgmax(
        __ubuf__ INDEX_T* argmaxAddr, __ubuf__ T* srcAddr, uint16_t kD, uint16_t kH, uint16_t kW, uint32_t depStride,
        uint32_t rowStride, uint32_t colStride, uint16_t repeatElms, int32_t curInD, int32_t curInH, int32_t curInW,
        int32_t hInput, int32_t wInput, uint32_t dDilation, uint32_t hDilation, uint32_t wDilation);
    __simd_vf__ inline void ProcessBigCPosition(__ubuf__ INDEX_T* argmaxAddr, __ubuf__ T* srcAddr, uint16_t cLoop,
                                                uint16_t tailNum, uint16_t vl, uint16_t dKernel, uint16_t hKernel,
                                                uint16_t wKernel, uint32_t depStride, uint32_t rowStride,
                                                uint32_t colStride, int32_t curInD, int32_t curInH, int32_t curInW,
                                                int32_t hInput, int32_t wInput, uint32_t dDilation, uint32_t hDilation,
                                                uint32_t wDilation);
    __aicore__ inline void ChunkCGather(__ubuf__ T* xAddr, __ubuf__ INDEX_T* argmaxAddr);
    __aicore__ inline void SingleRowGather(__ubuf__ T* xAddr, __ubuf__ INDEX_T* argmaxAddr);
    __aicore__ inline void MultiRowGather(__ubuf__ T* xAddr, __ubuf__ INDEX_T* argmaxAddr);
    __aicore__ inline void MultiDepGather(__ubuf__ T* xAddr, __ubuf__ INDEX_T* argmaxAddr);
    __aicore__ inline void MultiNcGather(__ubuf__ T* xAddr, __ubuf__ INDEX_T* argmaxAddr);

    __simd_vf__ inline void ProcessWNDHWC(__ubuf__ T* xAddr, __ubuf__ INDEX_T* argmaxAddr, uint16_t repeatElem,
                                          uint16_t cChunk, uint16_t dKernel, uint16_t hKernel, uint16_t wKernel,
                                          uint32_t colStride, uint32_t rowStride, uint32_t depStride,
                                          uint32_t wBaseStep, int32_t dOrigin, int32_t hOrigin, int32_t wOrigin,
                                          int32_t sW, int32_t hInput, int32_t wInput, uint32_t dDilation,
                                          uint32_t hDilation, uint32_t wDilation, uint32_t isPad);

    __simd_vf__ inline void ProcessWNDHWC2D(__ubuf__ T* xAddr, __ubuf__ INDEX_T* argmaxAddr, uint16_t repeatElem,
                                            uint16_t cChunk, uint16_t wOut, uint16_t dKernel, uint16_t hKernel,
                                            uint16_t wKernel, uint32_t colStride, uint32_t rowStride,
                                            uint32_t depStride, uint32_t wBaseStep, uint32_t hBaseStep, int32_t dOrigin,
                                            int32_t hOrigin, int32_t wOrigin, int32_t sW, int32_t sH, int32_t hInput,
                                            int32_t wInput, uint32_t dDilation, uint32_t hDilation, uint32_t wDilation,
                                            uint32_t isPad);

    __simd_vf__ inline void ProcessWNDHWC3D(__ubuf__ T* xAddr, __ubuf__ INDEX_T* argmaxAddr, uint16_t repeatElem,
                                            uint16_t cChunk, uint16_t wOut, uint16_t hOut, uint16_t dKernel,
                                            uint16_t hKernel, uint16_t wKernel, uint32_t colStride, uint32_t rowStride,
                                            uint32_t depStride, uint32_t wBaseStep, uint32_t hBaseStep,
                                            uint32_t dBaseStep, int32_t dOrigin, int32_t hOrigin, int32_t wOrigin,
                                            int32_t sD, int32_t sW, int32_t sH, int32_t hInput, int32_t wInput,
                                            uint32_t dDilation, uint32_t hDilation, uint32_t wDilation, uint32_t isPad);

    __simd_vf__ inline void ProcessWNDHWC4D(__ubuf__ T* xAddr, __ubuf__ INDEX_T* argmaxAddr, uint16_t repeatElem,
                                            uint16_t cChunk, uint16_t wOut, uint16_t hOut, uint16_t dOut,
                                            uint16_t dKernel, uint16_t hKernel, uint16_t wKernel, uint32_t colStride,
                                            uint32_t rowStride, uint32_t depStride, uint32_t wBaseStep,
                                            uint32_t hBaseStep, uint32_t dBaseStep, uint32_t ncBaseStep,
                                            int32_t dOrigin, int32_t hOrigin, int32_t wOrigin, int32_t sD, int32_t sW,
                                            int32_t sH, int32_t hInput, int32_t wInput, uint32_t dDilation,
                                            uint32_t hDilation, uint32_t wDilation, uint32_t isPad);

    __simd_callee__ inline void ProcessWNDHWCKernel(
        __ubuf__ T* xAddr, int32_t hOffset, uint32_t colStride, uint32_t rowStride, uint32_t depStride,
        Reg::RegTensor<int32_t>& indexReg, uint16_t dKernel, uint16_t hKernel, uint16_t wKernel, uint16_t repeatElem,
        Reg::RegTensor<INDEX_T>& argmaxDStart, Reg::RegTensor<INDEX_T>& argmaxHStart,
        Reg::RegTensor<INDEX_T>& argmaxWStart, Reg::RegTensor<INDEX_T>& argmaxDRes, Reg::RegTensor<INDEX_T>& argmaxHRes,
        Reg::RegTensor<INDEX_T>& argmaxWRes, uint32_t dDilation, uint32_t hDilation, uint32_t wDilation);

    __simd_callee__ inline void ComposeAndStoreArgmax(Reg::RegTensor<INDEX_T>& argmaxDRes,
                                                      Reg::RegTensor<INDEX_T>& argmaxHRes,
                                                      Reg::RegTensor<INDEX_T>& argmaxWRes, __ubuf__ INDEX_T* argmaxAddr,
                                                      uint16_t repeatElem, int32_t hInput, int32_t wInput,
                                                      uint32_t isPad);

    __simd_vf__ inline void DupCalcNegInfVf(__ubuf__ T* calcAddr, uint64_t totalElems, uint32_t vl);
    __simd_vf__ inline void CopyValidToCalcVf(__ubuf__ T* calcAddr, __ubuf__ T* srcAddr, uint32_t nActual,
                                              uint32_t dNoPad, uint32_t hNoPad, uint32_t wNoPadRowElems, uint32_t dPad,
                                              uint32_t hPad, uint32_t wPadRowElems, uint32_t leftElems, uint32_t front,
                                              uint32_t top, uint32_t vl);

    TPipe pipe_;

    GlobalTensor<T> xGm_;

    TQue<QuePosition::VECIN, BUFFER_NUM> inputQue_;
    TBuf<TPosition::VECCALC> inputCalcBuff_;

    TBufPool<TPosition::VECCALC> forwardBufPool_;
    TBufPool<TPosition::VECCALC> backwardBufPool_;
    int64_t totalStageBufferSize_ = 0;

    bool isBigC_ = false;

    __ubuf__ T* xForwardAddr_ = nullptr;

    int64_t inputStrideW_ = 0;
    int64_t inputStrideH_ = 0;
    int64_t inputStrideD_ = 0;
    int64_t forwardInputOffset_ = 0;

    uint32_t ncBaseStep_ = 0;
    uint32_t dBaseStep_ = 0;
    uint32_t hBaseStep_ = 0;
    int64_t dInputActualPad_ = 0;
    int64_t hInputActualPad_ = 0;
    int64_t wInputActualPad_ = 0;

    int64_t leftOffsetToInputLeft_ = 0;
    int64_t rightOffsetToInputRight_ = 0;
    int64_t topOffsetToInputTop_ = 0;
    int64_t downOffsetToInputDown_ = 0;
    int64_t frontOffsetToInputFront_ = 0;
    int64_t backOffsetToInputBack_ = 0;
    int64_t dInputActualNoPad_ = 0;
    int64_t hInputActualNoPad_ = 0;
    int64_t wInputActualNoPad_ = 0;
};

} // namespace MaxPool3DGradNDHWCNameSpace
#endif
