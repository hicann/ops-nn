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
 * \file cla_gate_quant_dual_axis.h
 * \brief Dual-axis ClaGate MX quantization implementation.
 *
 * Naming convention: see cla_gate_quant_common.h (`...Tensor` for LocalTensor,
 * `...Addr` / `...StartAddr` / `...WriteAddr` / `...ReadAddr` for `__ubuf__`
 * pointers, `...Num` / `...Count` for sizes, and always name the quantity).
 */

#ifndef OPS_NN_CLA_GATE_QUANT_DUAL_AXIS_H
#define OPS_NN_CLA_GATE_QUANT_DUAL_AXIS_H

#include "cla_gate_quant_common.h"

namespace ClaGateQuant {

// Dual-axis CLA gate quantization.
template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
class ClaGateQuantDualAxisBase : public ClaGateQuantCommonBase<xDtype, rowDataDtype, roundMode, scaleAlg> {
public:
    using Base = ClaGateQuantCommonBase<xDtype, rowDataDtype, roundMode, scaleAlg>;
    using Base::Base;

    __aicore__ inline void Init(GM_ADDR globalOut, GM_ADDR localOut, GM_ADDR globalGate, GM_ADDR localGate,
                                GM_ADDR rowData, GM_ADDR rowScale, GM_ADDR colData, GM_ADDR colScale)
    {
        Base::template Init<TPL_DUAL_AXIS>(globalOut, localOut, globalGate, localGate, rowData, rowScale, colData,
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
    using Base::colDataOffsetInOutQueue_;
    using Base::colScaleGm_;
    using Base::colScaleOffsetInOutQueue_;
    using Base::colScaleReciprocalBuf_;
    using Base::colTailSize_;
    using Base::ComputeClaGateD128TwoHeads;
    using Base::ComputeClaGateFullTile;
    using Base::ComputeRowDataToFp4;
    using Base::ComputeRowDataToFp8;
    using Base::CopyInClaGate;
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
    using Base::rowScaleElementsPerRow_;
    using Base::rowScaleGm_;
    using Base::rowScaleQueue_;
    using Base::rowScaleReciprocalBuf_;
    using Base::sigmoidBroadcastBuf_;
    using Base::tileRowCount_;
    using Base::tileRowLength_;
    using Base::tilingData_;
    using Base::ubBlockElemCountB16_;
    using Base::ubBlockElemCountB8_;

    // Builds the coordinate/geometry descriptor of one tile of the
    // (row block x column block) grid.  Either direction may end in a short tail
    // block, so the descriptor carries the valid extent next to the origin.
    __aicore__ inline TileDesc MakeTileDesc(int64_t rowBlockIdx, int64_t colBlockIdx) const;

    __aicore__ inline void ComputeInterleave(__ubuf__ uint8_t* dstAddr, __ubuf__ uint8_t* src0Addr,
                                             __ubuf__ uint8_t* src1Addr, bool hasSecondSlot);
    __aicore__ inline void ComputeScaleOcp(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                           __ubuf__ uint8_t* rowScaleAddr, __ubuf__ uint16_t* rowScaleReciprocalAddr,
                                           __ubuf__ uint8_t* colScaleAddr, __ubuf__ uint16_t* colScaleReciprocalAddr);
    __aicore__ inline void ComputeScaleCuBLAS(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                              __ubuf__ uint8_t* rowDataAddr, __ubuf__ uint8_t* rowScaleAddr,
                                              __ubuf__ uint16_t* rowScaleReciprocalAddr, __ubuf__ uint8_t* colScaleAddr,
                                              __ubuf__ uint16_t* colScaleReciprocalAddr);
    __aicore__ inline void ComputeScaleCuBLASSecondLast(uint16_t dataLen, uint32_t outputDtypeInvMax,
                                                        __ubuf__ uint16_t* colScaleReciprocalAddr,
                                                        __ubuf__ uint8_t* colScaleAddr);
    __aicore__ inline void ComputeColDataToFp8(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                               __ubuf__ uint16_t* colScaleReciprocalAddr,
                                               __ubuf__ uint8_t* colDataAddr);
    __aicore__ inline void ComputeColDataToFp4(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                               __ubuf__ uint16_t* colScaleReciprocalAddr,
                                               __ubuf__ uint8_t* colDataAddr);
    __aicore__ inline void CopyOut(int64_t rowDataOutOffset, int64_t rowScaleOutOffset, int64_t colScaleOutOffset,
                                   int64_t blockCount, int64_t blockCountAlign, int64_t dataLen, int64_t dataLenAlign);
};

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateQuantDualAxisBase<xDtype, rowDataDtype, roundMode, scaleAlg>::Process()
{
    // K=N*D and D is 128 or 256, so K is always exactly divisible by 64.
    rowScaleElementsPerRow_ = rowLength_ / DOUBLE_BLOCK_SIZE * DIGIT_TWO;
    int64_t totalCoreNum = tilingData_->usedCoreNum;
    if (blockIdx_ >= totalCoreNum) {
        return;
    }

    int64_t blockPerCore = tilingData_->baseTaskCount;
    int64_t tailCore = tilingData_->extraTaskCoreCount;
    int64_t loops = blockPerCore + (blockIdx_ < tailCore ? 1 : 0);
    if (loops == 0) {
        return;
    }
    int64_t firstBlock = blockIdx_ * blockPerCore + (blockIdx_ < tailCore ? blockIdx_ : tailCore);
    int64_t colBlockNum = tilingData_->colTileNum;
    int64_t rowBlockIdx = firstBlock / colBlockNum;
    int64_t colBlockIdx = firstBlock % colBlockNum;
    TileDesc tile = MakeTileDesc(rowBlockIdx, colBlockIdx);

    for (int64_t i = 0; i < loops; ++i) {
        CopyInClaGate(tile);
        LocalTensor<xDtype> gateInput = gateInQueue_.template DeQue<xDtype>();
        LocalTensor<xDtype> activationInput = activationInQueue_.template DeQue<xDtype>();
        LocalTensor<xDtype> mergedResult = mergedClaGateBuf_.template Get<xDtype>();
        auto globalAddr = (__ubuf__ xDtype*)activationInput.GetPhyAddr();
        auto localAddr = (__ubuf__ xDtype*)activationInput[halfBufferElemCount_].GetPhyAddr();
        auto globalGateAddr = (__ubuf__ xDtype*)gateInput.GetPhyAddr();
        auto localGateAddr = (__ubuf__ xDtype*)gateInput[gateSideOffset_].GetPhyAddr();
        auto mergedAddr = (__ubuf__ xDtype*)mergedResult.GetPhyAddr();

        // A full 256-column D=128 tile uses the two-head implementation;
        // a partial-width tail uses the generic one-head path.
        if (headSize_ == D128_HEAD_ELEMS && tile.calcCol == tileRowLength_ && tile.headsPerRowInBlock == DIGIT_TWO) {
            ComputeClaGateD128TwoHeads(globalAddr, localAddr, globalGateAddr, localGateAddr, mergedAddr,
                                       static_cast<uint16_t>(tile.calcRow));
        } else {
            ComputeClaGateFullTile(globalAddr, localAddr, globalGateAddr, localGateAddr, mergedAddr,
                                   static_cast<uint16_t>(tile.calcRow));
        }

        activationInQueue_.template FreeTensor(activationInput);
        gateInQueue_.template FreeTensor(gateInput);

        // Column-scale output keeps two 32-row slots per 64-row frame.
        constexpr int64_t calcRowAligned = DOUBLE_BLOCK_SIZE;
        int64_t calcColAligned = tile.calcCol;

        LocalTensor<uint8_t> rowScaleTensor = rowScaleQueue_.template AllocTensor<uint8_t>();
        LocalTensor<uint8_t> rowDataTensor = rowDataQueue_.template AllocTensor<uint8_t>();
        // Expose column regions of the shared output frames as sub-tensors.
        LocalTensor<uint8_t> colScaleTensor = rowScaleTensor[colScaleOffsetInOutQueue_];
        LocalTensor<uint8_t> colDataTensor = rowDataTensor[colDataOffsetInOutQueue_];
        LocalTensor<uint16_t> rowScaleReciprocalTensor = rowScaleReciprocalBuf_.template Get<uint16_t>();
        LocalTensor<uint16_t> colScaleReciprocalTensor = colScaleReciprocalBuf_.template Get<uint16_t>();
        auto rowDataAddr = (__ubuf__ uint8_t*)rowDataTensor.GetPhyAddr();
        auto colDataAddr = (__ubuf__ uint8_t*)colDataTensor.GetPhyAddr();
        auto rowScaleAddr = (__ubuf__ uint8_t*)rowScaleTensor.GetPhyAddr();
        auto colScaleAddr = (__ubuf__ uint8_t*)colScaleTensor.GetPhyAddr();
        auto rowScaleReciprocalAddr = (__ubuf__ uint16_t*)rowScaleReciprocalTensor.GetPhyAddr();
        auto colScaleReciprocalAddr = (__ubuf__ uint16_t*)colScaleReciprocalTensor.GetPhyAddr();

        int64_t calcBlockLoop = ops::CeilDiv(tile.calcRow, BLOCK_SIZE);
        if constexpr (scaleAlg == TPL_SCALE_ALG_1) {
            ComputeScaleCuBLAS(static_cast<uint16_t>(tileRowLength_), static_cast<uint16_t>(tile.calcRow), mergedAddr,
                               rowDataAddr, rowScaleAddr, rowScaleReciprocalAddr, colScaleAddr, colScaleReciprocalAddr);
        }

        for (int64_t blk = 0; blk < calcBlockLoop; blk++) {
            int64_t blockRowStart = blk * BLOCK_SIZE;
            int64_t blockRowCount = tile.calcRow - blockRowStart;
            blockRowCount = blockRowCount < BLOCK_SIZE ? blockRowCount : BLOCK_SIZE;
            int64_t dataOffset = blk * BLOCK_SIZE * tileRowLength_;
            int64_t rowScaleOutOffset = blk * BLOCK_SIZE *
                                        ops::CeilAlign(tileRowLength_ / BLOCK_SIZE, ubBlockElemCountB8_);
            int64_t colScaleOutOffset = blk * tileRowLength_;
            int64_t rowScaleReciprocalOffset = blk * BLOCK_SIZE *
                                               ops::CeilAlign(tileRowLength_ / BLOCK_SIZE, ubBlockElemCountB16_);
            int64_t colScaleReciprocalOffset = blk * tileRowLength_;

            if constexpr (scaleAlg == TPL_SCALE_ALG_0) {
                ComputeScaleOcp(static_cast<uint16_t>(tileRowLength_), static_cast<uint16_t>(blockRowCount),
                                mergedAddr + dataOffset, rowScaleAddr + rowScaleOutOffset,
                                rowScaleReciprocalAddr + rowScaleReciprocalOffset, colScaleAddr + colScaleOutOffset,
                                colScaleReciprocalAddr + colScaleReciprocalOffset);
            }

            int64_t packedDataOffset = dataOffset;
            if constexpr (IsSameType<rowDataDtype, fp4x2_e2m1_t>::value ||
                          IsSameType<rowDataDtype, fp4x2_e1m2_t>::value) {
                packedDataOffset = dataOffset / DIGIT_TWO;
                ComputeRowDataToFp4(static_cast<uint16_t>(tileRowLength_), static_cast<uint16_t>(blockRowCount),
                                    mergedAddr + dataOffset, rowScaleReciprocalAddr + rowScaleReciprocalOffset,
                                    rowDataAddr + packedDataOffset);
                ComputeColDataToFp4(static_cast<uint16_t>(tileRowLength_), static_cast<uint16_t>(blockRowCount),
                                    mergedAddr + dataOffset, colScaleReciprocalAddr + colScaleReciprocalOffset,
                                    colDataAddr + packedDataOffset);
                ComputeColDataToFp4(static_cast<uint16_t>(tileRowLength_), static_cast<uint16_t>(blockRowCount),
                                    mergedAddr + dataOffset + VF_LEN_B16,
                                    colScaleReciprocalAddr + colScaleReciprocalOffset + VF_LEN_B16,
                                    colDataAddr + packedDataOffset + VF_LEN_B16 / DIGIT_TWO);
            } else {
                if constexpr (scaleAlg == TPL_SCALE_ALG_0) {
                    ComputeRowDataToFp8(static_cast<uint16_t>(tileRowLength_), static_cast<uint16_t>(blockRowCount),
                                        mergedAddr + dataOffset, rowScaleReciprocalAddr + rowScaleReciprocalOffset,
                                        rowDataAddr + packedDataOffset);
                }
                ComputeColDataToFp8(static_cast<uint16_t>(tileRowLength_), static_cast<uint16_t>(blockRowCount),
                                    mergedAddr + dataOffset, colScaleReciprocalAddr + colScaleReciprocalOffset,
                                    colDataAddr + packedDataOffset);
            }
        }

        if constexpr (scaleAlg == TPL_SCALE_ALG_0) {
            constexpr int64_t scaleSlotCount = DOUBLE_BLOCK_SIZE / BLOCK_SIZE;
            for (int64_t blk = 1; blk < scaleSlotCount; blk += 2) {
                auto src0Addr = (__ubuf__ uint8_t*)colScaleTensor[(blk - 1) * tileRowLength_].GetPhyAddr();
                auto src1Addr = (__ubuf__ uint8_t*)colScaleTensor[blk * tileRowLength_].GetPhyAddr();
                auto dstAddr = (__ubuf__ uint8_t*)colScaleTensor[(blk - 1) * tileRowLength_].GetPhyAddr();
                ComputeInterleave(dstAddr, src0Addr, src1Addr, blk < calcBlockLoop);
            }
        }

        rowScaleQueue_.template EnQue(rowScaleTensor);
        rowDataQueue_.template EnQue(rowDataTensor);

        int64_t rowDataOutOffset = tile.absRowStart * rowLength_ + tile.colOffset;
        int64_t rowScaleOutOffset = tile.absRowStart * rowScaleElementsPerRow_ + tile.colOffset / BLOCK_SIZE;
        int64_t colScaleOutOffset = tile.rowBlockIdx * DIGIT_TWO * rowLength_ + tile.colOffset * DIGIT_TWO;

        CopyOut(rowDataOutOffset, rowScaleOutOffset, colScaleOutOffset, tile.calcRow, calcRowAligned, tile.calcCol,
                calcColAligned);

        if (i + 1 < loops) {
            ++colBlockIdx;
            if (colBlockIdx == colBlockNum) {
                colBlockIdx = 0;
                ++rowBlockIdx;
            }
            tile = MakeTileDesc(rowBlockIdx, colBlockIdx);
        }
    }
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline TileDesc ClaGateQuantDualAxisBase<xDtype, rowDataDtype, roundMode, scaleAlg>::MakeTileDesc(
    int64_t rowBlockIdx, int64_t colBlockIdx) const
{
    TileDesc desc;
    desc.colBlockIdx = colBlockIdx;
    desc.rowBlockIdx = rowBlockIdx;
    desc.absRowStart = desc.rowBlockIdx * tileRowCount_;
    desc.colOffset = desc.colBlockIdx * tileRowLength_;

    // The last row block is short when T is not a multiple of tileRowCount_.
    // min(tileRowCount_, rowCount - absRowStart) gives the final row tile's
    // valid height without carrying another host field.
    int64_t rowRemain = tilingData_->rowCount - desc.absRowStart;
    desc.calcRow = rowRemain < tileRowCount_ ? rowRemain : tileRowCount_;
    // The last column block is short when K is not a multiple of tileRowLength_
    // (only D=128 with an odd head count); colTailSize carries its width.
    desc.calcCol = (desc.colBlockIdx == tilingData_->colTileNum - 1) ? colTailSize_ : tileRowLength_;
    // A full tile contains headsPerTileRow_ heads; a short tile contains one.
    desc.headsPerRowInBlock = desc.calcCol == tileRowLength_ ? headsPerTileRow_ : 1;
    return desc;
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateQuantDualAxisBase<xDtype, rowDataDtype, roundMode, scaleAlg>::ComputeInterleave(
    __ubuf__ uint8_t* dstAddr, __ubuf__ uint8_t* src0Addr, __ubuf__ uint8_t* src1Addr, bool hasSecondSlot)
{
    asc_vf_call<ComputeInterleaveVF<xDtype, rowDataDtype, roundMode, scaleAlg>>(dstAddr, src0Addr, src1Addr,
                                                                                hasSecondSlot);
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateQuantDualAxisBase<xDtype, rowDataDtype, roundMode, scaleAlg>::ComputeScaleOcp(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* rowScaleAddr,
    __ubuf__ uint16_t* rowScaleReciprocalAddr, __ubuf__ uint8_t* colScaleAddr,
    __ubuf__ uint16_t* colScaleReciprocalAddr)
{
    asc_vf_call<ComputeScaleOcpVF<xDtype, rowDataDtype, roundMode, scaleAlg>>(
        dataLen, blockCount, xAddr, rowScaleAddr, rowScaleReciprocalAddr, colScaleAddr, colScaleReciprocalAddr,
        outputDtypeMaxExp_);
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void
ClaGateQuantDualAxisBase<xDtype, rowDataDtype, roundMode, scaleAlg>::ComputeScaleCuBLASSecondLast(
    uint16_t dataLen, uint32_t outputDtypeInvMax, __ubuf__ uint16_t* colScaleReciprocalAddr,
    __ubuf__ uint8_t* colScaleAddr)
{
    asc_vf_call<ComputeScaleCuBLASSecondLastVF<xDtype, rowDataDtype, roundMode, scaleAlg>>(
        dataLen, outputDtypeInvMax, colScaleReciprocalAddr, colScaleAddr);
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateQuantDualAxisBase<xDtype, rowDataDtype, roundMode, scaleAlg>::ComputeScaleCuBLAS(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* rowDataAddr,
    __ubuf__ uint8_t* rowScaleAddr, __ubuf__ uint16_t* rowScaleReciprocalAddr, __ubuf__ uint8_t* colScaleAddr,
    __ubuf__ uint16_t* colScaleReciprocalAddr)
{
    asc_vf_call<ComputeScaleCuBLASVF<xDtype, rowDataDtype, roundMode, scaleAlg>>(
        dataLen, blockCount, xAddr, rowDataAddr, rowScaleAddr, rowScaleReciprocalAddr, colScaleAddr,
        colScaleReciprocalAddr, outputDtypeInvMax_);
    ComputeScaleCuBLASSecondLast(dataLen, outputDtypeInvMax_, colScaleReciprocalAddr, colScaleAddr);
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateQuantDualAxisBase<xDtype, rowDataDtype, roundMode, scaleAlg>::ComputeColDataToFp8(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* colScaleReciprocalAddr,
    __ubuf__ uint8_t* colDataAddr)
{
    asc_vf_call<ComputeColDataToFp8VF<xDtype, rowDataDtype, roundMode, scaleAlg>>(
        dataLen, blockCount, xAddr, colScaleReciprocalAddr, colDataAddr, tileRowLength_);
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateQuantDualAxisBase<xDtype, rowDataDtype, roundMode, scaleAlg>::ComputeColDataToFp4(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* colScaleReciprocalAddr,
    __ubuf__ uint8_t* colDataAddr)
{
    asc_vf_call<ComputeColDataToFp4VF<xDtype, rowDataDtype, roundMode, scaleAlg>>(
        dataLen, blockCount, xAddr, colScaleReciprocalAddr, colDataAddr, tileRowLength_);
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateQuantDualAxisBase<xDtype, rowDataDtype, roundMode, scaleAlg>::CopyOut(
    int64_t rowDataOutOffset, int64_t rowScaleOutOffset, int64_t colScaleOutOffset, int64_t blockCount,
    int64_t blockCountAlign, int64_t dataLen, int64_t dataLenAlign)
{
    uint16_t outBurst = static_cast<uint16_t>(blockCount);
    uint32_t outBlockLen = 0;
    uint32_t srcStride = 0;
    uint32_t dstStride = 0;

    int64_t rowDataOutOffsetNow = rowDataOutOffset;

    // axis=-2 two rows interleaved, accounting for 32B alignment
    uint32_t scaleSrcStride = DIGIT_TWO * ops::CeilDiv(dataLen, UB_BLOCK_SIZE) -
                              ops::CeilDiv(DIGIT_TWO * dataLen, UB_BLOCK_SIZE);

    if constexpr (IsSameType<rowDataDtype, fp4x2_e2m1_t>::value || IsSameType<rowDataDtype, fp4x2_e1m2_t>::value) {
        // FP4: two fp4 values packed into one byte
        outBlockLen = static_cast<uint32_t>(dataLen / DIGIT_TWO * sizeof(uint8_t));
        srcStride = static_cast<uint32_t>((tileRowLength_ - dataLen) / DIGIT_TWO * sizeof(uint8_t) / UB_BLOCK_SIZE);
        dstStride = static_cast<uint32_t>((rowLength_ - dataLen) / DIGIT_TWO * sizeof(uint8_t));
        rowDataOutOffsetNow = rowDataOutOffset / DIGIT_TWO;
    } else {
        // FP8: one byte per element
        outBlockLen = static_cast<uint32_t>(dataLen * sizeof(uint8_t));
        srcStride = static_cast<uint32_t>((tileRowLength_ - dataLen) * sizeof(rowDataDtype) / UB_BLOCK_SIZE);
        dstStride = static_cast<uint32_t>((rowLength_ - dataLen) * sizeof(uint8_t));
    }

    DataCopyExtParams rowDataCopyOutParams = {outBurst, outBlockLen, srcStride, dstStride, 0};

    // Row-scale output contains one byte per 32 input elements.
    uint32_t rowScaleOutLen = dataLenAlign / BLOCK_SIZE;

    DataCopyExtParams rowScaleCopyOutParams = {
        outBurst, static_cast<uint32_t>(rowScaleOutLen * sizeof(uint8_t)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(ops::CeilAlign(rowLength_, DOUBLE_BLOCK_SIZE) / BLOCK_SIZE - rowScaleOutLen),
        static_cast<uint32_t>(0)};

    // Column scales are stored as interleaved pairs of 32-row groups.
    DataCopyExtParams colScaleCopyOutParams = {
        static_cast<uint16_t>(blockCountAlign / DOUBLE_BLOCK_SIZE),
        static_cast<uint32_t>(dataLen * DIGIT_TWO * sizeof(uint8_t)), static_cast<uint32_t>(scaleSrcStride),
        static_cast<uint32_t>(DIGIT_TWO * (rowLength_ - dataLen) * sizeof(uint8_t)), static_cast<uint32_t>(0)};

    // Dequeue and copy row_data + col_data; both live in the same double buffer.
    LocalTensor<uint8_t> rowDataLocal = rowDataQueue_.template DeQue<uint8_t>();
    LocalTensor<uint8_t> colDataLocal = rowDataLocal[colDataOffsetInOutQueue_];
    DataCopyPad(rowDataGm_[rowDataOutOffsetNow], rowDataLocal, rowDataCopyOutParams);
    DataCopyPad(colDataGm_[rowDataOutOffsetNow], colDataLocal, rowDataCopyOutParams);
    rowDataQueue_.FreeTensor(rowDataLocal);

    // Dequeue and copy row_scale + col_scale; both live in the same double buffer.
    LocalTensor<uint8_t> rowScaleLocal = rowScaleQueue_.template DeQue<uint8_t>();
    LocalTensor<uint8_t> colScaleLocal = rowScaleLocal[colScaleOffsetInOutQueue_];
    DataCopyPad(rowScaleGm_[rowScaleOutOffset], rowScaleLocal, rowScaleCopyOutParams);
    DataCopyPad(colScaleGm_[colScaleOutOffset], colScaleLocal, colScaleCopyOutParams);
    rowScaleQueue_.FreeTensor(rowScaleLocal);
}

} // namespace ClaGateQuant

#endif // OPS_NN_CLA_GATE_QUANT_DUAL_AXIS_H
