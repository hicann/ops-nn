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
 * \file cla_gate_quant_single_axis.h
 * \brief Single-axis (tail axis) ClaGate MX quantization implementation.
 *
 * Naming convention: see cla_gate_quant_common.h (`...Tensor` for LocalTensor,
 * `...Addr` / `...StartAddr` / `...WriteAddr` / `...ReadAddr` for `__ubuf__`
 * pointers, `...Num` / `...Count` for sizes, and always name the quantity).
 */

#ifndef OPS_NN_CLA_GATE_QUANT_SINGLE_AXIS_H
#define OPS_NN_CLA_GATE_QUANT_SINGLE_AXIS_H

#include "cla_gate_quant_common.h"

namespace ClaGateQuant {

// Single-axis CLA gate quantization.
template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
class ClaGateQuantTailAxisBase : public ClaGateQuantCommonBase<xDtype, rowDataDtype, roundMode, scaleAlg> {
public:
    using Base = ClaGateQuantCommonBase<xDtype, rowDataDtype, roundMode, scaleAlg>;
    using Base::Base;

    __aicore__ inline void Init(GM_ADDR globalOut, GM_ADDR localOut, GM_ADDR globalGate, GM_ADDR localGate,
                                GM_ADDR rowData, GM_ADDR rowScale, GM_ADDR colData, GM_ADDR colScale)
    {
        Base::template Init<TPL_SINGLE_AXIS>(globalOut, localOut, globalGate, localGate, rowData, rowScale, colData,
                                             colScale);
    }

    __aicore__ inline void Process();

protected:
    using Base::activationInQueue_;
    using Base::blockIdx_;
    using Base::castTraitBF16toFp4;
    using Base::castTraitFp32toBF16;
    using Base::castTraitFp32toYdtype;
    using Base::colDataGm_;
    using Base::colScaleGm_;
    using Base::colScaleReciprocalBuf_;
    using Base::ComputeClaGateFullTile;
    using Base::ComputeRowDataToFp4;
    using Base::CopyInClaGate;
    using Base::FillZero;
    using Base::gateInQueue_;
    using Base::gateSideOffset_;
    using Base::globalGateGm_;
    using Base::globalOutGm_;
    using Base::halfBufferElemCount_;
    using Base::headCount_;
    using Base::headSize_;
    using Base::headsPerTileRow_;
    using Base::localGateGm_;
    using Base::localOutGm_;
    using Base::mergedClaGateBuf_;
    using Base::outputDtypeInvMax_;
    using Base::outputDtypeMaxExp_;
    using Base::pipe_;
    using Base::rowDataGm_;
    using Base::rowDataQueue_;
    using Base::rowLength_;
    using Base::rowScaleGm_;
    using Base::rowScaleQueue_;
    using Base::rowScaleReciprocalBuf_;
    using Base::sigmoidBroadcastBuf_;
    using Base::tileRowCount_;
    using Base::tileRowLength_;
    using Base::tilingData_;
    using Base::ubBlockElemCountB16_;
    using Base::ubBlockElemCountB8_;

    __aicore__ inline void CopyInClaGateBatch(int64_t firstSegment, int64_t segmentCount);
    __aicore__ inline void ComputeSigmoidSingleAxis(__ubuf__ xDtype* globalGateRow, __ubuf__ xDtype* localGateRow,
                                                    uint16_t gateCount);
    __aicore__ inline void ComputeClaGateSingleAxisD256(__ubuf__ xDtype* globalAddr, __ubuf__ xDtype* localAddr,
                                                        __ubuf__ xDtype* outputAddr, uint16_t calcRows);
    __aicore__ inline void ComputeClaGateSingleAxisD128(__ubuf__ xDtype* globalAddr, __ubuf__ xDtype* localAddr,
                                                        __ubuf__ xDtype* outputAddr, uint16_t calcRows);
    template <bool fullWidth>
    __aicore__ inline void ProcessChunk(int64_t firstSegment, int64_t segmentCount, uint16_t validElements);
    template <bool fullWidth>
    __aicore__ inline void QuantizeAndCopyOut(int64_t firstSegment, int64_t segmentCount, uint16_t validElements,
                                              __ubuf__ xDtype* mergedAddr);
    __aicore__ inline void ComputeRowScaleOcpBatchCompact(uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                                          __ubuf__ uint8_t* rowScaleAddr,
                                                          __ubuf__ uint16_t* rowScaleReciprocalAddr,
                                                          __ubuf__ uint16_t* maxExpAddr);
    __aicore__ inline void ComputeRowDataToFp8Compact(uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                                      __ubuf__ uint16_t* rowScaleReciprocalAddr,
                                                      __ubuf__ uint8_t* rowDataAddr);
    __aicore__ inline void ComputeRowScaleCuBLASSingleAxis(uint16_t dataLen, uint16_t blockCount,
                                                           __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* rowDataAddr,
                                                           __ubuf__ uint8_t* rowScaleAddr,
                                                           __ubuf__ uint16_t* rowScaleReciprocalAddr,
                                                           __ubuf__ uint16_t* absMaxAddr);
    __aicore__ inline void CopyOutRowOnly(int64_t rowDataOutOffset, int64_t rowScaleOutOffset, int64_t blockCount,
                                          int64_t dataLen);
    __aicore__ inline void CopyOutRowOnlyBatch(int64_t rowDataOutOffset, int64_t rowScaleOutOffset,
                                               int64_t segmentCount);
};

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateQuantTailAxisBase<xDtype, rowDataDtype, roundMode, scaleAlg>::Process()
{
    int64_t coreNum = tilingData_->usedCoreNum;
    if (blockIdx_ >= coreNum || coreNum <= 0) {
        return;
    }

    int64_t baseSegmentNum = tilingData_->baseTaskCount;
    int64_t extraCoreNum = tilingData_->extraTaskCoreCount;
    int64_t totalSegmentNum = baseSegmentNum * coreNum + extraCoreNum;
    if (totalSegmentNum <= 0) {
        return;
    }

    int64_t coreSegmentNum = baseSegmentNum + (blockIdx_ < extraCoreNum ? 1 : 0);
    int64_t coreFirstSegment = blockIdx_ * baseSegmentNum + (blockIdx_ < extraCoreNum ? blockIdx_ : extraCoreNum);
    if (coreSegmentNum <= 0) {
        return;
    }

    // UB batch capacity is derived by host tiling.  Repeated batches use a
    // complete 64-gate sigmoid-VF quantum (64 segments for D=256, 32 for D=128),
    // while the last batch may contain fewer segments.
    int64_t batchSegmentCapacity = tileRowCount_;

    // Segments are contiguous 256-element slices of the flattened [T, K] stream.
    // Only the final stream segment may be short and uses the bounded path below.
    const int64_t fullSegmentEnd = tilingData_->streamTailSize == ONCE_ROW_LEN ? totalSegmentNum : totalSegmentNum - 1;

    int64_t coreSegmentEnd = coreFirstSegment + coreSegmentNum;
    int64_t batchSegmentEnd = coreSegmentEnd < fullSegmentEnd ? coreSegmentEnd : fullSegmentEnd;
    for (int64_t segmentIdx = coreFirstSegment; segmentIdx < batchSegmentEnd; segmentIdx += batchSegmentCapacity) {
        int64_t batchSegmentCount = batchSegmentEnd - segmentIdx < batchSegmentCapacity ? batchSegmentEnd - segmentIdx :
                                                                                          batchSegmentCapacity;
        ProcessChunk<true>(segmentIdx, batchSegmentCount, static_cast<uint16_t>(tileRowLength_));
    }

    if (coreFirstSegment <= fullSegmentEnd && fullSegmentEnd < coreSegmentEnd) {
        ProcessChunk<false>(fullSegmentEnd, 1, static_cast<uint16_t>(tilingData_->streamTailSize));
    }
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
template <bool fullWidth>
__aicore__ inline void ClaGateQuantTailAxisBase<xDtype, rowDataDtype, roundMode, scaleAlg>::ProcessChunk(
    int64_t firstSegment, int64_t segmentCount, uint16_t validElements)
{
    if constexpr (fullWidth) {
        CopyInClaGateBatch(firstSegment, segmentCount);
    } else {
        // With D in {128, 256}, a short flattened segment can only be the final
        // single D=128 head.  Keep its GM copy bounded while retaining the same
        // 256-element UB frame used by full chunks.
        TileDesc desc;
        desc.rowBlockIdx = firstSegment;
        desc.colBlockIdx = 0;
        desc.absRowStart = (firstSegment * tileRowLength_) / rowLength_;
        desc.colOffset = (firstSegment * tileRowLength_) % rowLength_;
        desc.calcRow = 1;
        desc.calcCol = validElements;
        desc.headsPerRowInBlock = validElements / headSize_;
        CopyInClaGate(desc);
    }

    LocalTensor<xDtype> gateInput = gateInQueue_.template DeQue<xDtype>();
    auto globalGateAddr = (__ubuf__ xDtype*)gateInput.GetPhyAddr();
    auto localGateAddr = (__ubuf__ xDtype*)gateInput[gateSideOffset_].GetPhyAddr();
    if constexpr (fullWidth) {
        // Start the sigmoid as soon as the small gate copy arrives.  The
        // activation queue is dequeued afterwards so its MTE2 can overlap it.
        ComputeSigmoidSingleAxis(globalGateAddr, localGateAddr, static_cast<uint16_t>(segmentCount * headsPerTileRow_));
        gateInQueue_.template FreeTensor(gateInput);
    }

    LocalTensor<xDtype> activationInput = activationInQueue_.template DeQue<xDtype>();
    LocalTensor<xDtype> mergedResult = mergedClaGateBuf_.template Get<xDtype>();
    auto globalAddr = (__ubuf__ xDtype*)activationInput.GetPhyAddr();
    auto localAddr = (__ubuf__ xDtype*)activationInput[halfBufferElemCount_].GetPhyAddr();
    auto mergedAddr = (__ubuf__ xDtype*)mergedResult.GetPhyAddr();

    if constexpr (fullWidth) {
        if (headSize_ == D128_HEAD_ELEMS) {
            ComputeClaGateSingleAxisD128(globalAddr, localAddr, mergedAddr, static_cast<uint16_t>(segmentCount));
        } else {
            ComputeClaGateSingleAxisD256(globalAddr, localAddr, mergedAddr, static_cast<uint16_t>(segmentCount));
        }
    } else {
        // The short chunk contains one D=128 head.  The common one-head VF
        // avoids reading a nonexistent second gate or the unused half-frame.
        ComputeClaGateFullTile(globalAddr, localAddr, globalGateAddr, localGateAddr, mergedAddr, 1);
        gateInQueue_.template FreeTensor(gateInput);
    }

    activationInQueue_.template FreeTensor(activationInput);
    QuantizeAndCopyOut<fullWidth>(firstSegment, segmentCount, validElements, mergedAddr);
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
template <bool fullWidth>
__aicore__ inline void ClaGateQuantTailAxisBase<xDtype, rowDataDtype, roundMode, scaleAlg>::QuantizeAndCopyOut(
    int64_t firstSegment, int64_t segmentCount, uint16_t validElements, __ubuf__ xDtype* mergedAddr)
{
    LocalTensor<uint8_t> rowScaleTensor = rowScaleQueue_.template AllocTensor<uint8_t>();
    LocalTensor<uint8_t> rowDataTensor = rowDataQueue_.template AllocTensor<uint8_t>();
    LocalTensor<uint16_t> rowScaleReciprocalTensor = rowScaleReciprocalBuf_.template Get<uint16_t>();
    auto rowDataAddr = (__ubuf__ uint8_t*)rowDataTensor.GetPhyAddr();
    auto rowScaleAddr = (__ubuf__ uint8_t*)rowScaleTensor.GetPhyAddr();
    auto rowScaleReciprocalAddr = (__ubuf__ uint16_t*)rowScaleReciprocalTensor.GetPhyAddr();

    if constexpr (scaleAlg == TPL_SCALE_ALG_1) {
        LocalTensor<uint16_t> absMaxTensor = colScaleReciprocalBuf_.template Get<uint16_t>();
        auto absMaxAddr = (__ubuf__ uint16_t*)absMaxTensor.GetPhyAddr();
        ComputeRowScaleCuBLASSingleAxis(static_cast<uint16_t>(tileRowLength_), static_cast<uint16_t>(segmentCount),
                                        mergedAddr, rowDataAddr, rowScaleAddr, rowScaleReciprocalAddr, absMaxAddr);
    } else {
        LocalTensor<uint16_t> maxExpTensor = colScaleReciprocalBuf_.template Get<uint16_t>();
        auto maxExpAddr = (__ubuf__ uint16_t*)maxExpTensor.GetPhyAddr();
        ComputeRowScaleOcpBatchCompact(static_cast<uint16_t>(segmentCount), mergedAddr, rowScaleAddr,
                                       rowScaleReciprocalAddr, maxExpAddr);
        if constexpr (IsSameType<rowDataDtype, fp4x2_e2m1_t>::value || IsSameType<rowDataDtype, fp4x2_e1m2_t>::value) {
            ComputeRowDataToFp4(static_cast<uint16_t>(tileRowLength_), static_cast<uint16_t>(segmentCount), mergedAddr,
                                rowScaleReciprocalAddr, rowDataAddr, ONCE_ROW_LEN / BLOCK_SIZE);
        } else {
            ComputeRowDataToFp8Compact(static_cast<uint16_t>(segmentCount), mergedAddr, rowScaleReciprocalAddr,
                                       rowDataAddr);
        }
    }

    rowScaleQueue_.template EnQue(rowScaleTensor);
    rowDataQueue_.template EnQue(rowDataTensor);
    int64_t rowDataOutOffset = firstSegment * tileRowLength_;
    int64_t rowScaleOutOffset = firstSegment * (tileRowLength_ / BLOCK_SIZE);
    if constexpr (fullWidth) {
        CopyOutRowOnlyBatch(rowDataOutOffset, rowScaleOutOffset, segmentCount);
    } else {
        CopyOutRowOnly(rowDataOutOffset, rowScaleOutOffset, 1, validElements);
    }
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateQuantTailAxisBase<xDtype, rowDataDtype, roundMode, scaleAlg>::CopyInClaGateBatch(
    int64_t firstSegment, int64_t segmentCount)
{
    DataCopyPadExtParams<xDtype> pad = {false, 0, 0, 0};
    // Enqueue gates independently so sigmoid can overlap the activation transfer.
    // Each stream segment maps to headsPerTileRow_ contiguous [T, N] gate values.
    LocalTensor<xDtype> gateInput = gateInQueue_.template AllocTensor<xDtype>();
    int64_t firstHead = firstSegment * headsPerTileRow_;
    int64_t gateCount = segmentCount * headsPerTileRow_;
    uint32_t gateBytes = static_cast<uint32_t>(gateCount * sizeof(xDtype));
    DataCopyExtParams gateCopy = {1, gateBytes, 0, 0, 0};
    DataCopyPad(gateInput, globalGateGm_[firstHead], gateCopy, pad);
    DataCopyPad(gateInput[gateSideOffset_], localGateGm_[firstHead], gateCopy, pad);
    gateInQueue_.template EnQue(gateInput);

    LocalTensor<xDtype> activationInput = activationInQueue_.template AllocTensor<xDtype>();
    int64_t srcOffset = firstSegment * tileRowLength_;
    uint32_t totalBytes = static_cast<uint32_t>(segmentCount * tileRowLength_ * sizeof(xDtype));
    DataCopyExtParams contiguousCopy = {1, totalBytes, 0, 0, 0};
    DataCopyPad(activationInput, globalOutGm_[srcOffset], contiguousCopy, pad);
    DataCopyPad(activationInput[halfBufferElemCount_], localOutGm_[srcOffset], contiguousCopy, pad);
    activationInQueue_.template EnQue(activationInput);
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateQuantTailAxisBase<xDtype, rowDataDtype, roundMode, scaleAlg>::ComputeSigmoidSingleAxis(
    __ubuf__ xDtype* globalGateRow, __ubuf__ xDtype* localGateRow, uint16_t gateCount)
{
    asc_vf_call<ComputeSigmoidSingleAxisVF<xDtype, rowDataDtype, roundMode, scaleAlg>>(
        globalGateRow, localGateRow, gateCount,
        (__ubuf__ uint8_t*)sigmoidBroadcastBuf_.template Get<float>().GetPhyAddr(), gateSideOffset_);
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void
ClaGateQuantTailAxisBase<xDtype, rowDataDtype, roundMode, scaleAlg>::ComputeClaGateSingleAxisD256(
    __ubuf__ xDtype* globalAddr, __ubuf__ xDtype* localAddr, __ubuf__ xDtype* outputAddr, uint16_t calcRows)
{
    asc_vf_call<ComputeClaGateSingleAxisD256VF<xDtype, rowDataDtype, roundMode, scaleAlg>>(
        globalAddr, localAddr, outputAddr, calcRows, tileRowLength_,
        (__ubuf__ uint8_t*)sigmoidBroadcastBuf_.template Get<float>().GetPhyAddr(), gateSideOffset_);
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void
ClaGateQuantTailAxisBase<xDtype, rowDataDtype, roundMode, scaleAlg>::ComputeClaGateSingleAxisD128(
    __ubuf__ xDtype* globalAddr, __ubuf__ xDtype* localAddr, __ubuf__ xDtype* outputAddr, uint16_t calcRows)
{
    asc_vf_call<ComputeClaGateSingleAxisD128VF<xDtype, rowDataDtype, roundMode, scaleAlg>>(
        globalAddr, localAddr, outputAddr, calcRows,
        (__ubuf__ uint8_t*)sigmoidBroadcastBuf_.template Get<float>().GetPhyAddr(), gateSideOffset_);
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateQuantTailAxisBase<xDtype, rowDataDtype, roundMode, scaleAlg>::CopyOutRowOnlyBatch(
    int64_t rowDataOutOffset, int64_t rowScaleOutOffset, int64_t segmentCount)
{
    LocalTensor<uint8_t> rowDataLocal = rowDataQueue_.template DeQue<uint8_t>();
    LocalTensor<uint8_t> rowScaleLocal = rowScaleQueue_.template DeQue<uint8_t>();

    int64_t dataBytesPerSegment = tileRowLength_;
    int64_t rowDataOutOffsetNow = rowDataOutOffset;
    if constexpr (IsSameType<rowDataDtype, fp4x2_e2m1_t>::value || IsSameType<rowDataDtype, fp4x2_e1m2_t>::value) {
        dataBytesPerSegment /= DIGIT_TWO;
        rowDataOutOffsetNow /= DIGIT_TWO;
    }

    DataCopyExtParams rowDataCopy = {1, static_cast<uint32_t>(segmentCount * dataBytesPerSegment), 0, 0, 0};
    DataCopyPad(rowDataGm_[rowDataOutOffsetNow], rowDataLocal, rowDataCopy);

    constexpr uint32_t SCALE_BYTES_PER_SEGMENT = ONCE_ROW_LEN / BLOCK_SIZE;
    // Full-width OCP batches produce dense scale bytes for both FP8 and FP4.
    DataCopyExtParams rowScaleCopy = {1, static_cast<uint32_t>(segmentCount * SCALE_BYTES_PER_SEGMENT), 0, 0, 0};
    DataCopyPad(rowScaleGm_[rowScaleOutOffset], rowScaleLocal, rowScaleCopy);
    PipeBarrier<PIPE_MTE3>();

    rowDataQueue_.template FreeTensor(rowDataLocal);
    rowScaleQueue_.template FreeTensor(rowScaleLocal);
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void
ClaGateQuantTailAxisBase<xDtype, rowDataDtype, roundMode, scaleAlg>::ComputeRowScaleOcpBatchCompact(
    uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* rowScaleAddr,
    __ubuf__ uint16_t* rowScaleReciprocalAddr, __ubuf__ uint16_t* maxExpAddr)
{
    asc_vf_call<ComputeRowScaleOcpBatchCompactVF<xDtype, rowDataDtype, roundMode, scaleAlg>>(
        blockCount, xAddr, rowScaleAddr, rowScaleReciprocalAddr, maxExpAddr, outputDtypeMaxExp_);
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateQuantTailAxisBase<xDtype, rowDataDtype, roundMode, scaleAlg>::ComputeRowDataToFp8Compact(
    uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* rowScaleReciprocalAddr,
    __ubuf__ uint8_t* rowDataAddr)
{
    asc_vf_call<ComputeRowDataToFp8CompactVF<xDtype, rowDataDtype, roundMode, scaleAlg>>(
        blockCount, xAddr, rowScaleReciprocalAddr, rowDataAddr);
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void
ClaGateQuantTailAxisBase<xDtype, rowDataDtype, roundMode, scaleAlg>::ComputeRowScaleCuBLASSingleAxis(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* rowDataAddr,
    __ubuf__ uint8_t* rowScaleAddr, __ubuf__ uint16_t* rowScaleReciprocalAddr, __ubuf__ uint16_t* absMaxAddr)
{
    asc_vf_call<ComputeRowScaleCuBLASSingleAxisVF<xDtype, rowDataDtype, roundMode, scaleAlg>>(
        dataLen, blockCount, xAddr, rowDataAddr, rowScaleAddr, rowScaleReciprocalAddr, absMaxAddr, outputDtypeInvMax_);
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateQuantTailAxisBase<xDtype, rowDataDtype, roundMode, scaleAlg>::CopyOutRowOnly(
    int64_t rowDataOutOffset, int64_t rowScaleOutOffset, int64_t blockCount, int64_t dataLen)
{
    uint16_t outBurst = static_cast<uint16_t>(blockCount);
    uint32_t outBlockLen = 0;
    uint32_t srcStride = 0;
    uint32_t dstStride = 0;
    int64_t rowDataOutOffsetNow = rowDataOutOffset;

    if constexpr (IsSameType<rowDataDtype, fp4x2_e2m1_t>::value || IsSameType<rowDataDtype, fp4x2_e1m2_t>::value) {
        outBlockLen = static_cast<uint32_t>(dataLen / DIGIT_TWO * sizeof(uint8_t));
        srcStride = static_cast<uint32_t>((tileRowLength_ - dataLen) / DIGIT_TWO * sizeof(uint8_t) / UB_BLOCK_SIZE);
        dstStride = static_cast<uint32_t>((rowLength_ - dataLen) / DIGIT_TWO * sizeof(uint8_t));
        rowDataOutOffsetNow = rowDataOutOffset / DIGIT_TWO;
    } else {
        outBlockLen = static_cast<uint32_t>(dataLen * sizeof(uint8_t));
        srcStride = static_cast<uint32_t>((tileRowLength_ - dataLen) * sizeof(rowDataDtype) / UB_BLOCK_SIZE);
        dstStride = static_cast<uint32_t>((rowLength_ - dataLen) * sizeof(uint8_t));
    }

    DataCopyExtParams rowDataCopyOutParams = {outBurst, outBlockLen, srcStride, dstStride, 0};
    int64_t dataLenAlign = ops::CeilDiv(dataLen, DOUBLE_BLOCK_SIZE) * DOUBLE_BLOCK_SIZE;
    uint32_t rowScaleOutLen = static_cast<uint32_t>(dataLenAlign / BLOCK_SIZE);
    DataCopyExtParams rowScaleCopyOutParams = {
        outBurst, rowScaleOutLen, static_cast<uint32_t>(0),
        static_cast<uint32_t>(ops::CeilAlign(rowLength_, DOUBLE_BLOCK_SIZE) / BLOCK_SIZE - rowScaleOutLen),
        static_cast<uint32_t>(0)};

    LocalTensor<uint8_t> rowDataLocal = rowDataQueue_.template DeQue<uint8_t>();
    DataCopyPad(rowDataGm_[rowDataOutOffsetNow], rowDataLocal, rowDataCopyOutParams);
    rowDataQueue_.FreeTensor(rowDataLocal);

    LocalTensor<uint8_t> rowScaleLocal = rowScaleQueue_.template DeQue<uint8_t>();
    DataCopyPad(rowScaleGm_[rowScaleOutOffset], rowScaleLocal, rowScaleCopyOutParams);
    rowScaleQueue_.FreeTensor(rowScaleLocal);
}

} // namespace ClaGateQuant

#endif // OPS_NN_CLA_GATE_QUANT_SINGLE_AXIS_H
