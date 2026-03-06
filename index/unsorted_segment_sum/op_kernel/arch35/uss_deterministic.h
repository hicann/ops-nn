/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file uss_deterministic.h
 * \brief uss_deterministic
 */
#ifndef USS_DETERMINISTIC_H
#define USS_DETERMINISTIC_H

#include <cmath>
#include <cstdint>

#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;


constexpr int32_t VL_SIZE = Ops::Base::GetVRegSize();
constexpr uint32_t DOUBLE = 2;
constexpr uint32_t MAX_THREAD = 1024;
constexpr uint64_t UB_AGLIN_VALUE = 32;
constexpr uint32_t SORT_STAT_PADDING = 64;
constexpr uint32_t FLOAT_BYTES = 4;
constexpr uint32_t USED_THREAD_NUMS = 1024;

static constexpr MicroAPI::CastTrait castTraitFP322INT32 = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::SAT,
                                                            MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

static constexpr MicroAPI::CastTrait castTraitINT322FP32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                            MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

constexpr AscendC::MicroAPI::CastTrait castTraitB162B32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

static constexpr SortConfig sortConfig{SortType::RADIX_SORT, false};

template <typename Tp, Tp v>
struct integral_constant {
    static constexpr Tp value = v;
};
using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;
template <typename, typename>
struct is_same : public false_type {};
template <typename Tp>
struct is_same<Tp, Tp> : public true_type {};

__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD_NUMS) inline void SimtComputeR(uint32_t blockId, uint64_t segmentNum,
    uint32_t dimANum, uint32_t blockNums, __gm__ float* workspaceRValue, __gm__ uint64_t* workspaceValidIdCount)
{
    for (uint32_t i = blockId * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < segmentNum * dimANum;
            i = i + blockNums * Simt::GetThreadNum()) {
        uint32_t row = i / dimANum;
        workspaceRValue[i] = workspaceRValue[i] * workspaceValidIdCount[row];
    }
}

template <typename P>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD_NUMS) inline void SetZero(uint32_t blockId, uint32_t blockNums, __gm__ P* gmBuffer, uint64_t length)
{
    for (uint64_t i = blockId * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < length; i = i + blockNums * Simt::GetThreadNum()) {
        gmBuffer[i] = static_cast<P>(0);
    }
}

template <typename U>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD_NUMS) inline void SimtRemoveRepetitiveId(__local_mem__ U* sortedIndice,
    uint32_t uniqueIdNum, uint32_t shiftOffset, uint32_t positiveStart, __local_mem__ int32_t* idCounts)
{
    for (uint32_t ubIdx = Simt::GetThreadIdx(); ubIdx < uniqueIdNum; ubIdx += Simt::GetThreadNum()) {
        sortedIndice[ubIdx] = sortedIndice[shiftOffset + positiveStart + idCounts[ubIdx]];
    }
}

template <typename U>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD_NUMS) inline void SimtUpdateIdCount(__gm__ uint64_t* workspaceValidIdCount,
    __local_mem__ U* sortedIdTensor, uint32_t uniqueNum, __gm__ uint64_t* workspaceCoreValidIdNum,
    uint64_t currentIdx, __local_mem__ int32_t* idCounts, uint64_t segmentNum)
{
    if (Simt::GetThreadIdx() == 0) {
        Simt::AtomicMax(workspaceValidIdCount + segmentNum, static_cast<uint64_t>(sortedIdTensor[uniqueNum - 1]));
        workspaceCoreValidIdNum[currentIdx] = uniqueNum;
    }
    for (uint32_t ubIdx = Simt::GetThreadIdx(); ubIdx < uniqueNum; ubIdx += Simt::GetThreadNum()) {
        Simt::AtomicAdd(workspaceValidIdCount + sortedIdTensor[ubIdx], static_cast<uint64_t>(idCounts[ubIdx]));
    }
}

template <typename T, typename U>
class KernelUSSDeterministic {
public:
    __aicore__ inline KernelUSSDeterministic(TPipe *pipe) {
        pipe_ = pipe;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output, GM_ADDR workSpace,
                                const UnsortedSegmentSumDetermTilingData* tilingData)
    {
        xGm_.SetGlobalBuffer((__gm__ T*)(x));
        yGm_.SetGlobalBuffer((__gm__ T*)(output));
        intOutGm_.SetGlobalBuffer((__gm__ int32_t*)(output));
        segmentIds_.SetGlobalBuffer((__gm__ U*)(segmentIds));

        vfFloatNum_ = VL_SIZE / sizeof(float);
        vfUNum_ = VL_SIZE / sizeof(U);

        singleCoreSampleNum_ = tilingData->singleCoreSampleNum;
        tailSampleNum_ = tilingData->tailSampleNum;
        dimANum_ = tilingData->innerDim;
        dimAFloatAlign_ = tilingData->row32BAlign;
        maxLoopInCore_ = tilingData->maxLoopInCore;
        usedLogicCore_ = tilingData->usedLogicCore;
        segmentNum_ = tilingData->outputOuterDim;
        totalSampleNum_ = tilingData->inputOuterDim;
        halfUBSize_ = tilingData->halfUBSize;
        secondSampleNum_ = tilingData->secondSampleNum;
        thirdSampleNum_ = tilingData->thirdSampleNum;
        sortTmpSize_ = tilingData->tmpBufferSize;
        usedThread_ = MIN(tilingData->maxThread, MAX_THREAD);

        blockId_ = GetBlockIdx();
        blockNums_ = GetBlockNum();

        pipe_->InitBuffer(inQueue_, DOUBLE, halfUBSize_);

        workspaceValidIdCount_.SetGlobalBuffer((__gm__ uint64_t*)workSpace,
                                               segmentNum_ + 1);  // 预留一个数存放最大id
        workspaceCoreValidIdNum_.SetGlobalBuffer((__gm__ uint64_t*)workSpace + segmentNum_ + 1, usedLogicCore_);
        workspaceRValue_.SetGlobalBuffer((__gm__ float*)workSpace + (segmentNum_ + 1 + usedLogicCore_) * DOUBLE,
                                         segmentNum_ * dimANum_);
        workspaceLogicCoreSumResults_.SetGlobalBuffer(
            (__gm__ float*)workSpace + (segmentNum_ + 1 + usedLogicCore_) * DOUBLE + segmentNum_ * dimANum_,
            totalSampleNum_ * dimAFloatAlign_);
        workspaceLogicCoreIds_.SetGlobalBuffer((__gm__ U*)workSpace + (segmentNum_ + 1 + usedLogicCore_) * DOUBLE +
                                                segmentNum_ * dimANum_ + totalSampleNum_ * dimAFloatAlign_,
                                                totalSampleNum_);
        if constexpr (is_same<bfloat16_t, T>::value || is_same<half, T>::value) {
            workspaceInt32Res_.SetGlobalBuffer(
                (__gm__ int32_t*)workSpace + (segmentNum_ + 1 + usedLogicCore_) * DOUBLE + segmentNum_ * dimANum_ +
                totalSampleNum_ * dimAFloatAlign_ + totalSampleNum_ * sizeof(U) / sizeof(int32_t),
                totalSampleNum_ * dimANum_);
        }
    }

    template <typename P>
    __aicore__ inline P MIN(P x, P y) {
        return x < y ? x : y;
    }

    __aicore__ inline void CopyInData(uint32_t rowNums, uint64_t currentIdx, const LocalTensor<T> inputTensor,
                                      const LocalTensor<U> segmentIdTensor)
    {
        uint64_t inputOffset = currentIdx * singleCoreSampleNum_ * dimANum_;
        uint64_t IdOffset = currentIdx * singleCoreSampleNum_;
        DataCopyExtParams dataCopyParam{(uint16_t)1, (uint32_t)(rowNums * dimANum_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> dataPadParam{false, 0, 0, 0};
        DataCopyPad(inputTensor, xGm_[inputOffset], dataCopyParam, dataPadParam);
        // copy id
        DataCopyExtParams idCopyParam{(uint16_t)1, (uint32_t)(rowNums * sizeof(U)), 0, 0, 0};
        DataCopyPadExtParams<U> idPadParam{false, 0, 0, 0};
        DataCopyPad(segmentIdTensor, segmentIds_[IdOffset], idCopyParam, idPadParam);
    }

    __aicore__ inline void SortIdInCore(LocalTensor<U> segmentIdTensor, LocalTensor<uint8_t> sortUtilTensor,
                                        LocalTensor<U> sortedIdTensor, LocalTensor<uint32_t> sortedIdxTensor,
                                        const uint32_t sortLen)
    {
        uint32_t shiftOffset = UB_AGLIN_VALUE / sizeof(U);
        Duplicate(sortedIdTensor, static_cast<U>(-1), shiftOffset * DOUBLE + singleCoreSampleNum_);
        LocalTensor<U> realData = sortedIdTensor[shiftOffset];
        AscendC::Sort<U, false, sortConfig>(realData, sortedIdxTensor, segmentIdTensor, sortUtilTensor,
                                            static_cast<uint32_t>(sortLen));
    }

    __aicore__ void RemoveDup(__local_mem__ U* sortedIndicesAddr, __local_mem__ int32_t* idCountsAddr, uint32_t counter,
                              const uint32_t& shiftOffset, const uint32_t& loopCnt, const uint32_t& tail)
    {
        uint32_t realLength = vfUNum_;
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<int32_t> orderReg;
            AscendC::MicroAPI::RegTensor<U> indicesReg;
            AscendC::MicroAPI::RegTensor<U> indicesShiftOneReg;
            AscendC::MicroAPI::RegTensor<int32_t> selReg0;
            AscendC::MicroAPI::MaskReg cmpMask;
            AscendC::MicroAPI::MaskReg maskRegUpdate;
            AscendC::MicroAPI::UnalignReg u0;
            AscendC::MicroAPI::UnalignReg u1;
            AscendC::MicroAPI::UnalignReg ureg0;
            AscendC::MicroAPI::ClearSpr<SpecialPurposeReg::AR>();

            for (uint16_t i = 0; i < static_cast<uint16_t>(loopCnt); ++i) {
                if (i == loopCnt - 1) {
                    realLength = tail;
                }
                uint32_t start = i * vfUNum_;
                AscendC::MicroAPI::Arange(orderReg, start);
                maskRegUpdate = AscendC::MicroAPI::UpdateMask<U>(counter);
                auto unalignUpdate = sortedIndicesAddr + i * vfUNum_ + shiftOffset;
                AscendC::MicroAPI::DataCopyUnAlignPre(u0, unalignUpdate);
                AscendC::MicroAPI::DataCopyUnAlign<U>(indicesReg, u0, unalignUpdate);
                AscendC::MicroAPI::DataCopyUnAlignPre(u1, unalignUpdate - 1);
                AscendC::MicroAPI::DataCopyUnAlign<U>(indicesShiftOneReg, u1, unalignUpdate - 1);
                AscendC::MicroAPI::Compare<U, CMPMODE::NE>(cmpMask, indicesReg, indicesShiftOneReg, maskRegUpdate);
                // vSQZ
                AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg0, orderReg,
                                                                                                     cmpMask);
                AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    idCountsAddr, selReg0, ureg0);
            }
            AscendC::MicroAPI::DataCopyUnAlignPost(idCountsAddr, ureg0);
        }
    }

    __aicore__ void UniqueID(const LocalTensor<U>& sortedIndice, uint32_t& uniqueIdNum, LocalTensor<int32_t>& idCounts,
                             const uint32_t& sortLen, uint32_t& positiveStart)
    {
        __local_mem__ U* sortedIndicesAddr = (__ubuf__ U*)sortedIndice.GetPhyAddr();
        __local_mem__ int32_t* idCountsAddr = (__ubuf__ int32_t*)idCounts.GetPhyAddr();
        uint32_t shiftOffset = UB_AGLIN_VALUE / sizeof(U);
        while (positiveStart < sortLen) {
            if (sortedIndice(shiftOffset + positiveStart) >= 0) {
                break;
            }
            ++positiveStart;
        }
        if (positiveStart == sortLen) {
            uniqueIdNum = 0;
            return;
        }
        uint32_t loopCnt = (sortLen + DOUBLE + vfUNum_ - 1 - positiveStart) / vfUNum_;
        uint32_t tail = sortLen + DOUBLE - positiveStart - (loopCnt - 1) * vfUNum_;

        uint32_t counter = sortLen + 1 - positiveStart;
        RemoveDup(sortedIndicesAddr, idCountsAddr, counter, shiftOffset + positiveStart, loopCnt, tail);
        // minus 1 because added -1 was counted
        uniqueIdNum = ((AscendC::MicroAPI::GetSpr<SpecialPurposeReg::AR>()) / sizeof(int32_t)) - 1;
        // remove repetitive id
        Simt::VF_CALL<SimtRemoveRepetitiveId<U>>(Simt::Dim3(USED_THREAD_NUMS), (__local_mem__ U*)(sortedIndice.GetPhyAddr()), uniqueIdNum,
            shiftOffset, positiveStart, (__local_mem__ int32_t*)(idCounts.GetPhyAddr()));
        // compute repeated num of each id
        uint16_t loopCntStatFre = (uniqueIdNum + vfFloatNum_ - 1) / vfFloatNum_;
        uint32_t counterStatFre = uniqueIdNum;
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<int32_t> beginReg;
            AscendC::MicroAPI::RegTensor<int32_t> endReg;
            AscendC::MicroAPI::RegTensor<int32_t> subReg;
            AscendC::MicroAPI::MaskReg maskRegUpdateFre;
            AscendC::MicroAPI::UnalignReg u0Fre;
            for (uint16_t i = 0; i < loopCntStatFre; ++i) {
                auto addrUpdate = idCountsAddr + i * vfFloatNum_ + 1;
                maskRegUpdateFre = AscendC::MicroAPI::UpdateMask<int32_t>(counterStatFre);
                DataCopy(beginReg, idCountsAddr + i * vfFloatNum_);
                AscendC::MicroAPI::DataCopyUnAlignPre(u0Fre, addrUpdate);
                AscendC::MicroAPI::DataCopyUnAlign<int32_t>(endReg, u0Fre, addrUpdate);
                AscendC::MicroAPI::Sub(subReg, endReg, beginReg, maskRegUpdateFre);
                DataCopy(idCountsAddr + i * vfFloatNum_, subReg, maskRegUpdateFre);
            }
        }
    }

    __aicore__ void DupSegmentProcess(const LocalTensor<uint32_t>& sortedIndiceIndex, uint32_t& uniqueIdNum,
                                      const LocalTensor<int32_t>& idCounts, LocalTensor<float>& resBuf,
                                      const LocalTensor<T>& inputLocal, const uint32_t& positiveStart,
                                      const uint32_t& rowsInUB)
    {
         // ignore the samples whose id is negative
        __local_mem__ T* gradLocalAddr =
            (__ubuf__ T*)inputLocal.GetPhyAddr() + positiveStart * dimANum_;
        __local_mem__ float* resBufAddr = (__ubuf__ float*)resBuf.GetPhyAddr();
        __local_mem__ uint32_t* sortedIndiceIndexAddr =
            (__ubuf__ uint32_t*)sortedIndiceIndex.GetPhyAddr() + positiveStart;

        uint32_t loopPerRow = (dimANum_ + vfFloatNum_ - 1) / vfFloatNum_;
        uint32_t segCount = 0;
        uint32_t gradWeightGmindex = positiveStart;
        // compute by float since all types are casted to float
        uint32_t postUpdateStride = dimANum_ <= vfFloatNum_
                                        ? dimANum_
                                        : vfFloatNum_;
        uint32_t ubAddrUpdate = 0;
        // align to VL size
        auto totalLength = (dimAFloatAlign_ * FLOAT_BYTES + VL_SIZE - 1) / VL_SIZE * VL_SIZE / FLOAT_BYTES * rowsInUB;

        __VEC_SCOPE__
        {
            for (uint16_t i = 0; i < static_cast<uint16_t>(uniqueIdNum); ++i) {
                AscendC::MicroAPI::RegTensor<float> sumResReg;
                AscendC::MicroAPI::RegTensor<T> tmpReg;
                AscendC::MicroAPI::RegTensor<T> dstReg1;
                AscendC::MicroAPI::RegTensor<T> dstReg2;
                AscendC::MicroAPI::RegTensor<float> tmpRegB32;
                AscendC::MicroAPI::MaskReg maskReg =
                    AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                MicroAPI::UnalignReg ureg0;
                MicroAPI::UnalignReg ureg1;
                AscendC::MicroAPI::MaskReg maskReg1;

                // frequency of each id
                segCount = idCounts(i);
                // VL align
                for (uint16_t j = 0; j < static_cast<uint16_t>(loopPerRow); ++j) {
                    maskReg1 = AscendC::MicroAPI::UpdateMask<uint32_t>(totalLength);
                    AscendC::MicroAPI::Duplicate(sumResReg, (float)0, maskReg);
                    for (uint16_t k = 0; k < static_cast<uint16_t>(segCount); ++k) {
                        auto unalignAddr =
                            gradLocalAddr + j * vfFloatNum_ + sortedIndiceIndex(gradWeightGmindex + k) * dimANum_;
                        AscendC::MicroAPI::DataCopyUnAlignPre(ureg0, unalignAddr);
                        AscendC::MicroAPI::DataCopyUnAlign<T>(tmpReg, ureg0, unalignAddr);
                        if constexpr (is_same<half, T>::value) {
                            Interleave(dstReg1, dstReg2, tmpReg, tmpReg);
                            Cast<float, half, castTraitB162B32>(tmpRegB32, dstReg1, maskReg1);
                            AscendC::MicroAPI::Add(sumResReg, sumResReg, tmpRegB32, maskReg1);
                        } else if constexpr (is_same<bfloat16_t, T>::value) {
                            Interleave(dstReg1, dstReg2, tmpReg, tmpReg);
                            Cast<float, bfloat16_t, castTraitB162B32>(tmpRegB32, dstReg1, maskReg1);
                            AscendC::MicroAPI::Add(sumResReg, sumResReg, tmpRegB32, maskReg1);
                        } else {
                            AscendC::MicroAPI::Add(sumResReg, sumResReg, tmpReg, maskReg1);
                        }
                    }
                    ubAddrUpdate = i * dimAFloatAlign_ + j * vfFloatNum_;
                    auto resBufAddrUpDate = resBufAddr + ubAddrUpdate;
                    DataCopy(resBufAddrUpDate, sumResReg, maskReg1);
                }
                gradWeightGmindex = segCount + gradWeightGmindex;
            }
        }
    }

    __aicore__ inline void ScatterToWorkspace(uint32_t uniqueNum, const LocalTensor<U>& sortedIdTensor,
                                              const LocalTensor<float>& SumResTensor,
                                              const GlobalTensor<float>& workspaceSumMax)
    {
        uint32_t localSumResOffset = 0;
        Abs(SumResTensor, SumResTensor, uniqueNum * dimAFloatAlign_);
        SetFlag<HardEvent::V_MTE3>(eventIdVMTE3_);
        WaitFlag<HardEvent::V_MTE3>(eventIdVMTE3_);
        SetAtomicMax<float>();
        for (uint32_t i = 0; i < uniqueNum; ++i) {
            uint64_t currentId = sortedIdTensor(i);
            uint64_t workspaceOffset = currentId * dimANum_;
            DataCopyExtParams dataCopyParams{(uint16_t)1, (uint32_t)(dimANum_ * sizeof(float)), 0, 0, 0};
            DataCopyPad(workspaceSumMax[workspaceOffset], SumResTensor[localSumResOffset], dataCopyParams);
            localSumResOffset += dimAFloatAlign_;
        }
        SetAtomicNone();
    }

    __aicore__ inline void FirstProcessInUB(uint64_t currentIdx)
    {
        LocalTensor<uint8_t> srcLocal = inQueue_.AllocTensor<uint8_t>();
        uint32_t sortUtilSizeB8 = sortTmpSize_;
        uint32_t segmentIdSizeB8 = Ops::Base::CeilAlign(singleCoreSampleNum_ * sizeof(U), UB_AGLIN_VALUE);
        uint32_t sortedIdSizeB8 = Ops::Base::CeilAlign(singleCoreSampleNum_ * sizeof(U) + SORT_STAT_PADDING, UB_AGLIN_VALUE);
        uint32_t sortedIdxSizeB8 = Ops::Base::CeilAlign(singleCoreSampleNum_ * sizeof(uint32_t), UB_AGLIN_VALUE);
        LocalTensor<U> segmentIdTensor = srcLocal[sortUtilSizeB8].template ReinterpretCast<U>();
        LocalTensor<T> inputTensor =
            srcLocal[sortUtilSizeB8 + segmentIdSizeB8 + sortedIdSizeB8 + sortedIdxSizeB8].template ReinterpretCast<T>();
        uint64_t inputOffset = currentIdx * singleCoreSampleNum_ * dimANum_;
        uint32_t rowsInUB = currentIdx == usedLogicCore_ - 1 ? tailSampleNum_ : singleCoreSampleNum_;
        uint32_t uniqueNum = 0;
        uint32_t positiveStart = 0;
        CopyInData(rowsInUB, currentIdx, inputTensor, segmentIdTensor);
        inQueue_.EnQue(srcLocal);
        LocalTensor<uint8_t> ubBuffer = inQueue_.DeQue<uint8_t>();
        LocalTensor<uint8_t> sortUtilTensor = ubBuffer[0];
        segmentIdTensor = ubBuffer[sortUtilSizeB8].template ReinterpretCast<U>();
        LocalTensor<int32_t> idCounts = ubBuffer[sortUtilSizeB8].template ReinterpretCast<int32_t>();
        LocalTensor<U> sortedIdTensor = ubBuffer[sortUtilSizeB8 + segmentIdSizeB8].template ReinterpretCast<U>();
        LocalTensor<uint32_t> sortedIdxTensor =
            ubBuffer[sortUtilSizeB8 + segmentIdSizeB8 + sortedIdSizeB8].template ReinterpretCast<uint32_t>();
        inputTensor =
            ubBuffer[sortUtilSizeB8 + segmentIdSizeB8 + sortedIdSizeB8 + sortedIdxSizeB8].template ReinterpretCast<T>();
        uint32_t inputSizeB8 = Ops::Base::CeilAlign(singleCoreSampleNum_ * dimANum_ * sizeof(T), UB_AGLIN_VALUE);
        LocalTensor<float> inputTensorInOrder =
            ubBuffer[sortUtilSizeB8 + segmentIdSizeB8 + sortedIdSizeB8 + sortedIdxSizeB8 + inputSizeB8]
                .template ReinterpretCast<float>();
        PipeBarrier<PIPE_ALL>(); // necessary
        SortIdInCore(segmentIdTensor, sortUtilTensor, sortedIdTensor, sortedIdxTensor, rowsInUB);
        PipeBarrier<PIPE_ALL>(); // necessary
        UniqueID(sortedIdTensor, uniqueNum, idCounts, rowsInUB, positiveStart);
        PipeBarrier<PIPE_ALL>(); // necessary
        DupSegmentProcess(sortedIdxTensor, uniqueNum, idCounts, inputTensorInOrder, inputTensor, positiveStart,
                          rowsInUB);
        SetFlag<HardEvent::V_MTE3>(eventIdVMTE3_);
        WaitFlag<HardEvent::V_MTE3>(eventIdVMTE3_);
        DataCopyExtParams dataCopyParam{(uint16_t)1, (uint32_t)(uniqueNum * dimAFloatAlign_ * sizeof(float)), 0, 0, 0};
        DataCopyPad(workspaceLogicCoreSumResults_[currentIdx * singleCoreSampleNum_ * dimAFloatAlign_],
                    inputTensorInOrder, dataCopyParam);
        DataCopyExtParams idCopyParam{(uint16_t)1, (uint32_t)(uniqueNum * sizeof(U)), 0, 0, 0};
        DataCopyPad(workspaceLogicCoreIds_[currentIdx * singleCoreSampleNum_], sortedIdTensor, idCopyParam);
        PipeBarrier<PIPE_ALL>(); // necessary
        ScatterToWorkspace(uniqueNum, sortedIdTensor, inputTensorInOrder, workspaceRValue_);

        Simt::VF_CALL<SimtUpdateIdCount<U>>(Simt::Dim3(USED_THREAD_NUMS), (__gm__ uint64_t*)(workspaceValidIdCount_.GetPhyAddr()),
            (__local_mem__ U*)(sortedIdTensor.GetPhyAddr()), uniqueNum, (__gm__ uint64_t*)(workspaceCoreValidIdNum_.GetPhyAddr()),
            currentIdx, (__local_mem__ int32_t*)(idCounts.GetPhyAddr()), segmentNum_);
        inQueue_.FreeTensor(ubBuffer);
    }

    __aicore__ inline void Scale2Int(const LocalTensor<float>& inputDataTensor, const LocalTensor<float>& inputRTensor,
                                     uint32_t totalLen)
    {
        __local_mem__ float* dataAddr = (__ubuf__ float*)inputDataTensor.GetPhyAddr();
        __local_mem__ float* rValueAddr = (__ubuf__ float*)inputRTensor.GetPhyAddr();
        LocalTensor<int32_t> scaledData = inputDataTensor[0].template ReinterpretCast<int32_t>();
        __local_mem__ int32_t* resAddr = (__ubuf__ int32_t*)scaledData.GetPhyAddr();
        uint32_t loopCnt = (totalLen + vfFloatNum_ - 1) / vfFloatNum_;
        float scaling = static_cast<float>(1 << 30);
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> dataReg;
            AscendC::MicroAPI::RegTensor<float> rReg;
            AscendC::MicroAPI::RegTensor<float> resReg;
            AscendC::MicroAPI::RegTensor<int32_t> scaleReg;
            AscendC::MicroAPI::MaskReg pregLoop;
            for (uint16_t i = 0; i < static_cast<uint16_t>(loopCnt); ++i) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(totalLen);
                DataCopy(dataReg, dataAddr + i * vfFloatNum_);
                DataCopy(rReg, rValueAddr + i * vfFloatNum_);
                Div(resReg, dataReg, rReg, pregLoop);
                Muls(resReg, resReg, scaling, pregLoop);
                Cast<int32_t, float, castTraitFP322INT32>(scaleReg, resReg, pregLoop);
                DataCopy(resAddr + i * vfFloatNum_, scaleReg, pregLoop);
            }
        }
    }

    __aicore__ inline void SecondProcessInUB(uint64_t currentIdx)
    {
        LocalTensor<uint8_t> ubBuffer = inQueue_.AllocTensor<uint8_t>();
        LocalTensor<float> inputDataTensor = ubBuffer[0].template ReinterpretCast<float>();
        uint32_t inputDataTensorSizeB8 = secondSampleNum_ * dimAFloatAlign_ * sizeof(float);
        LocalTensor<int32_t> scaledDataTensor = ubBuffer[0].template ReinterpretCast<int32_t>();
        LocalTensor<float> inputRTensor = ubBuffer[inputDataTensorSizeB8].template ReinterpretCast<float>();
        uint32_t validNum = workspaceCoreValidIdNum_(currentIdx);
        uint32_t innerLoop = (validNum + secondSampleNum_ - 1) / secondSampleNum_;
        uint32_t tailSize = validNum % secondSampleNum_ == 0 ? secondSampleNum_ : validNum % secondSampleNum_;
        uint64_t outterOffset = currentIdx * singleCoreSampleNum_ * dimAFloatAlign_;
        for (uint32_t i = 0; i < innerLoop; ++i) {
            uint32_t currentLen = i == innerLoop - 1 ? tailSize : secondSampleNum_;
            uint64_t innerOffset = i * secondSampleNum_ * dimAFloatAlign_;
            uint64_t totalOffset = outterOffset + innerOffset;
            DataCopyExtParams dataCopyParam{(uint16_t)1, (uint32_t)(validNum * dimAFloatAlign_ * sizeof(float)), 0, 0,
                                            0};
            DataCopyPadExtParams<float> dataPadParam{false, 0, 0, 0};
            DataCopyPad(inputDataTensor, workspaceLogicCoreSumResults_[totalOffset], dataCopyParam, dataPadParam);
            for (uint32_t j = 0; j < currentLen; ++j) {
                uint64_t currenrId =
                    workspaceLogicCoreIds_(currentIdx * singleCoreSampleNum_ + i * secondSampleNum_ + j);
                DataCopyExtParams rValueCopyParam{(uint16_t)1, (uint32_t)(dimANum_ * sizeof(float)), 0, 0, 0};
                // pad 1 incase div by 0
                DataCopyPadExtParams<float> rValuePadParam{true, 0, static_cast<uint8_t>(dimAFloatAlign_ - dimANum_),
                                                           1.0};
                DataCopyPad(inputRTensor[j * dimAFloatAlign_], workspaceRValue_[currenrId * dimANum_], rValueCopyParam,
                            rValuePadParam);
            }
            inQueue_.EnQue(ubBuffer);
            ubBuffer = inQueue_.DeQue<uint8_t>();
            inputDataTensor = ubBuffer[0].template ReinterpretCast<float>();
            inputRTensor = ubBuffer[inputDataTensorSizeB8].template ReinterpretCast<float>();
            SetFlag<HardEvent::MTE2_V>(eventIdMTE2V_);
            WaitFlag<HardEvent::MTE2_V>(eventIdMTE2V_);
            Scale2Int(inputDataTensor, inputRTensor, currentLen * dimAFloatAlign_);
            SetFlag<HardEvent::V_MTE3>(eventIdVMTE3_);
            WaitFlag<HardEvent::V_MTE3>(eventIdVMTE3_);
            SetAtomicAdd<int32_t>();
            for (uint32_t j = 0; j < currentLen; ++j) {
                uint64_t currenrId =
                    workspaceLogicCoreIds_(currentIdx * singleCoreSampleNum_ + i * secondSampleNum_ + j);
                DataCopyExtParams dataCopyParams{(uint16_t)1, (uint32_t)(dimANum_ * sizeof(int32_t)), 0, 0, 0};
                if constexpr (is_same<float, T>::value) {
                    DataCopyPad(intOutGm_[currenrId * dimANum_], scaledDataTensor[j * dimAFloatAlign_], dataCopyParams);
                } else {
                    DataCopyPad(workspaceInt32Res_[currenrId * dimANum_], scaledDataTensor[j * dimAFloatAlign_],
                                dataCopyParams);
                }
            }
            SetAtomicNone();
            inQueue_.FreeTensor(ubBuffer);
        }
    }

    __aicore__ inline void ScaleBack2Float(const LocalTensor<float>& inputDataTensor,
                                           const LocalTensor<float>& inputRTensor, uint32_t totalLen)
    {
        __local_mem__ float* dataAddr = (__ubuf__ float*)inputDataTensor.GetPhyAddr();
        __local_mem__ float* rValueAddr = (__ubuf__ float*)inputRTensor.GetPhyAddr();
        LocalTensor<int32_t> scaledData = inputDataTensor[0].template ReinterpretCast<int32_t>();
        __local_mem__ int32_t* scaledDataAddr = (__ubuf__ int32_t*)scaledData.GetPhyAddr();
        uint32_t loopCnt = (totalLen + vfFloatNum_ - 1) / vfFloatNum_;
        float scaling = static_cast<float>(1 << 30);
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> dataReg;
            AscendC::MicroAPI::RegTensor<float> rReg;
            AscendC::MicroAPI::RegTensor<float> resReg;
            AscendC::MicroAPI::RegTensor<int32_t> scaleReg;
            AscendC::MicroAPI::MaskReg pregLoop;
            AscendC::MicroAPI::MaskReg maskReg =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::Duplicate(dataReg, scaling, maskReg);
            for (uint16_t i = 0; i < static_cast<uint16_t>(loopCnt); ++i) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(totalLen);
                DataCopy(scaleReg, scaledDataAddr + i * vfFloatNum_);
                DataCopy(rReg, rValueAddr + i * vfFloatNum_);
                Cast<float, int32_t, castTraitINT322FP32>(resReg, scaleReg, pregLoop);
                Mul(resReg, resReg, rReg, pregLoop);
                Div(resReg, resReg, dataReg, pregLoop);
                DataCopy(dataAddr + i * vfFloatNum_, resReg, pregLoop);
            }
        }
    }

    __aicore__ inline void ThirdProcessInUB(uint64_t currentIdx)
    {
        LocalTensor<uint8_t> ubBuffer = inQueue_.AllocTensor<uint8_t>();
        LocalTensor<float> inputDataTensor = ubBuffer[0].template ReinterpretCast<float>();
        uint32_t inputDataTensorSizeB8 = Ops::Base::CeilAlign(secondSampleNum_ * dimANum_ * sizeof(float), UB_AGLIN_VALUE);
        LocalTensor<int32_t> scaledBackTensor = ubBuffer[0].template ReinterpretCast<int32_t>();
        LocalTensor<float> inputRTensor = ubBuffer[inputDataTensorSizeB8].template ReinterpretCast<float>();
        uint64_t offset = currentIdx * secondSampleNum_ * dimANum_;
        uint32_t validNum = currentIdx == lastUsedCoreNum_ - 1 ? lastProcessTailSize_ : secondSampleNum_;
        uint32_t lengthInCore = validNum * dimANum_;
        DataCopyExtParams dataCopyParam{(uint16_t)1, (uint32_t)(lengthInCore * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> intPadParam{false, 0, 0, 0};
        if constexpr (is_same<float, T>::value) {
            DataCopyPad(scaledBackTensor, intOutGm_[offset], dataCopyParam, intPadParam);
        } else {
            DataCopyPad(scaledBackTensor, workspaceInt32Res_[offset], dataCopyParam, intPadParam);
        }
        DataCopyPadExtParams<float> rValuePadParam{false, 0, 0, 0};
        DataCopyPad(inputRTensor, workspaceRValue_[offset], dataCopyParam, rValuePadParam);
        inQueue_.EnQue(ubBuffer);
        ubBuffer = inQueue_.DeQue<uint8_t>();
        inputDataTensor = ubBuffer[0].template ReinterpretCast<float>();
        inputRTensor = ubBuffer[inputDataTensorSizeB8].template ReinterpretCast<float>();
        SetFlag<HardEvent::MTE2_V>(eventIdMTE2V_);
        WaitFlag<HardEvent::MTE2_V>(eventIdMTE2V_);
        ScaleBack2Float(inputDataTensor, inputRTensor, lengthInCore);
        SetFlag<HardEvent::V_MTE3>(eventIdVMTE3_);
        WaitFlag<HardEvent::V_MTE3>(eventIdVMTE3_);
        if constexpr (!is_same<float, T>::value) {
            LocalTensor<T> finalResTensor = ubBuffer[inputDataTensorSizeB8].template ReinterpretCast<T>();
            Cast(finalResTensor, inputDataTensor, RoundMode::CAST_RINT, lengthInCore);
            DataCopyExtParams dataTCopyParam{(uint16_t)1, (uint32_t)(lengthInCore * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm_[offset], finalResTensor, dataTCopyParam);
        } else {
            DataCopyPad(yGm_[offset], inputDataTensor, dataCopyParam);
        }
    }

    __aicore__ inline void Process()
    {
        Simt::VF_CALL<SetZero<float>>(Simt::Dim3(USED_THREAD_NUMS), blockId_, blockNums_, (__gm__ float*)(workspaceRValue_.GetPhyAddr()), segmentNum_ * dimANum_);
        Simt::VF_CALL<SetZero<uint64_t>>(Simt::Dim3(USED_THREAD_NUMS), blockId_, blockNums_, (__gm__ uint64_t*)(workspaceValidIdCount_.GetPhyAddr()), segmentNum_ + 1);
        Simt::VF_CALL<SetZero<T>>(Simt::Dim3(USED_THREAD_NUMS), blockId_, blockNums_, (__gm__ T*)(yGm_.GetPhyAddr()), segmentNum_ * dimANum_);
        if constexpr (!is_same<float, T>::value) {
            Simt::VF_CALL<SetZero<int32_t>>(Simt::Dim3(USED_THREAD_NUMS), blockId_, blockNums_, (__gm__ int32_t*)(workspaceInt32Res_.GetPhyAddr()), segmentNum_ * dimANum_);
        }
        PipeBarrier<PIPE_ALL>();
        SyncAll();
        for (uint32_t loop = 0; loop < maxLoopInCore_; ++loop) {
            uint64_t currentIdx = loop * blockNums_ + blockId_;
            if (currentIdx >= usedLogicCore_) {
                break;
            }
            FirstProcessInUB(currentIdx);
        }
        PipeBarrier<PIPE_ALL>();
        SyncAll();
        uint64_t lastValidIDNum = MIN(workspaceValidIdCount_(segmentNum_) + 1, static_cast<uint64_t>(segmentNum_));
        lastUsedCoreNum_ = Ops::Base::CeilDiv(lastValidIDNum, static_cast<uint64_t>(thirdSampleNum_));
        lastLoopInCore_ = Ops::Base::CeilDiv(lastUsedCoreNum_, blockNums_);
        lastProcessTailSize_ = lastValidIDNum - (lastUsedCoreNum_ - 1) * thirdSampleNum_;
        Simt::VF_CALL<SimtComputeR>(Simt::Dim3(USED_THREAD_NUMS), blockId_, segmentNum_, dimANum_, blockNums_,
            (__gm__ float*)(workspaceRValue_.GetPhyAddr()), (__gm__ uint64_t*)(workspaceValidIdCount_.GetPhyAddr()));
        PipeBarrier<PIPE_ALL>();
        SyncAll();
        for (uint32_t loop = 0; loop < maxLoopInCore_; ++loop) {
            uint64_t currentIdx = loop * blockNums_ + blockId_;
            if (currentIdx >= usedLogicCore_) {
                break;
            }
            SecondProcessInUB(currentIdx);
        }
        PipeBarrier<PIPE_ALL>();
        SyncAll();
        for (uint32_t loop = 0; loop < lastLoopInCore_; ++loop) {
            uint64_t currentIdx = loop * blockNums_ + blockId_;
            if (currentIdx >= lastUsedCoreNum_) {
                break;
            }
            ThirdProcessInUB(currentIdx);
        }
        PipeBarrier<PIPE_ALL>();
        SyncAll();
    }

private:
    TPipe *pipe_ = nullptr;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DOUBLE> inQueue_;

    AscendC::GlobalTensor<T> xGm_, yGm_;
    AscendC::GlobalTensor<int32_t> intOutGm_;
    AscendC::GlobalTensor<U> segmentIds_;
    AscendC::GlobalTensor<float> workspaceRValue_, workspaceLogicCoreSumResults_;
    AscendC::GlobalTensor<uint64_t> workspaceValidIdCount_, workspaceCoreValidIdNum_;
    AscendC::GlobalTensor<int32_t> workspaceInt32Res_;
    AscendC::GlobalTensor<U> workspaceLogicCoreIds_;

    uint32_t vfFloatNum_{0};
    uint32_t vfUNum_{0};
    uint32_t blockId_;
    uint32_t blockNums_;
    event_t eventIdVMTE3_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    event_t eventIdMTE2V_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));

    uint32_t singleCoreSampleNum_{1};
    uint32_t tailSampleNum_{1};
    uint32_t dimANum_{1};
    uint32_t dimAFloatAlign_{1};
    uint32_t maxLoopInCore_{1};
    uint32_t usedLogicCore_{1};
    uint64_t segmentNum_{1};
    uint64_t totalSampleNum_{1};
    uint32_t halfUBSize_{1};
    uint32_t secondSampleNum_{1};
    uint32_t thirdSampleNum_{1};
    uint32_t lastLoopInCore_{1};
    uint32_t lastUsedCoreNum_{1};
    uint32_t lastProcessTailSize_{1};
    uint32_t sortTmpSize_{0};
    uint32_t usedThread_{128};
};
#endif