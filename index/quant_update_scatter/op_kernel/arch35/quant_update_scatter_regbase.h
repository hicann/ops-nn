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
 * \file quant_update_scatter_regbase.h
 * \brief quant_update_scatter little axis little quant template
 */
#ifndef QUANT_UPDATE_SCATTER_REGBASE_H_
#define QUANT_UPDATE_SCATTER_REGBASE_H_

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace QuantUpdateScatter {
using namespace AscendC;
template <
    typename VarType, typename IndicesType, typename UpdatesType, typename ScalesType, typename OffsetsType,
    uint64_t DivMode, uint64_t CastRoundMode>
class QuantUpdateScatterRegbase
    : public QuantUpdateScatterBase<VarType, IndicesType, UpdatesType, ScalesType, OffsetsType, DivMode, CastRoundMode>
{
public:
    __aicore__ inline QuantUpdateScatterRegbase(){};
    using Base =
        QuantUpdateScatterBase<VarType, IndicesType, UpdatesType, ScalesType, OffsetsType, DivMode, CastRoundMode>;
    constexpr static int64_t BUFFER_NUM = 2;
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueVar_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueIndices_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueUpdates_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueScales_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueZeroPoints_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueVar_;

    GlobalTensor<VarType> varGm_;
    GlobalTensor<IndicesType> indicesGm_;
    GlobalTensor<UpdatesType> updatesGm_;
    GlobalTensor<ScalesType> scalesGm_;
    GlobalTensor<OffsetsType> offsetsGm_;
    QuantUpdateScatterTilingData tilingData_;
    int64_t blockIdx_ = 0;
    int64_t gmVarOffset_ = 0;
    int64_t gmIndicesOffset_ = 0;
    int64_t gmUpdatesOffset_ = 0;
    int64_t gmScalesOffset_ = 0;
    int64_t gmZeroPointsOffset_ = 0;
    int64_t coreBsNum_ = 0;
    int64_t copyBlockCount_ = 0;

    __aicore__ inline void Init(
        GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR quant_scales, GM_ADDR quant_zero_points, GM_ADDR out,
        const QuantUpdateScatterTilingData* tiling)
    {
        blockIdx_ = GetBlockIdx();
        varGm_.SetGlobalBuffer(reinterpret_cast<__gm__ VarType*>(var));
        indicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ IndicesType*>(indices));
        updatesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ UpdatesType*>(updates));
        scalesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ ScalesType*>(quant_scales));
        if constexpr (!IsSameType<OffsetsType, bool>::value) {
            offsetsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ OffsetsType*>(quant_zero_points));
        }
        this->ParseTilingData(tiling, tilingData_);

        copyBlockCount_ = tilingData_.updateDim2 * tilingData_.updateDim3 / tilingData_.updateOriLastDim;
        pipe_.InitBuffer(
            inQueueIndices_, BUFFER_NUM,
            this->CeilAlign(tilingData_.indexElements * sizeof(IndicesType), this->BLOCK_SIZE));
        pipe_.InitBuffer(
            inQueueUpdates_, BUFFER_NUM,
            tilingData_.eachCoreBsNum * copyBlockCount_ * tilingData_.updateOriLastDimAlign * sizeof(UpdatesType));
        pipe_.InitBuffer(inQueueScales_, BUFFER_NUM, tilingData_.updateOriLastDimAlign * sizeof(ScalesType));
        if constexpr (!IsSameType<OffsetsType, bool>::value) {
            pipe_.InitBuffer(inQueueZeroPoints_, BUFFER_NUM, tilingData_.updateOriLastDimAlign * sizeof(OffsetsType));
        }

        pipe_.InitBuffer(
            outQueueVar_, BUFFER_NUM, copyBlockCount_ * tilingData_.updateOriLastDimAlign * sizeof(VarType));
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= tilingData_.coreNum) {
            return;
        }
        if (blockIdx_ == tilingData_.coreNum - 1) {
            coreBsNum_ = tilingData_.lastCoreBsNum;
        } else {
            coreBsNum_ = tilingData_.eachCoreBsNum;
        }

        CopyInScale();
        CopyInIndices();
        CopyInUpdate();
        LocalTensor<OffsetsType> offsetLocal;
        if constexpr (!IsSameType<OffsetsType, bool>::value) {
            CopyInOffset();
            offsetLocal = inQueueZeroPoints_.DeQue<OffsetsType>();
        }
        LocalTensor<IndicesType> iLocal = inQueueIndices_.DeQue<IndicesType>();
        LocalTensor<UpdatesType> updateLocal = inQueueUpdates_.DeQue<UpdatesType>();
        LocalTensor<ScalesType> scaleLocal = inQueueScales_.DeQue<ScalesType>();
        for (int64_t bsIdx = 0; bsIdx < coreBsNum_; bsIdx++) {
            if constexpr (!IsSameType<OffsetsType, bool>::value) {
                QuantUpdate(bsIdx, updateLocal, scaleLocal, offsetLocal);
            } else {
                QuantUpdateWithoutOffset(bsIdx, updateLocal, scaleLocal);
            }
            CalcDstOffset(bsIdx, iLocal);
            CopyOutVar();
        }
        inQueueIndices_.FreeTensor(iLocal);
        inQueueUpdates_.FreeTensor(updateLocal);
        inQueueScales_.FreeTensor(scaleLocal);
        if constexpr (!IsSameType<OffsetsType, bool>::value) {
            inQueueZeroPoints_.FreeTensor(offsetLocal);
        }
    }

    __aicore__ inline void CopyInUpdate()
    {
        gmUpdatesOffset_ = (blockIdx_ * tilingData_.eachCoreBsNum) * tilingData_.srcBsStride;
        DataCopyExtParams copyParams;
        copyParams.blockCount = coreBsNum_ * copyBlockCount_;
        copyParams.blockLen = tilingData_.updateOriLastDim * sizeof(UpdatesType);
        copyParams.dstStride =
            (tilingData_.updateOriLastDimAlign - tilingData_.updateOriLastDim) * sizeof(UpdatesType) / this->BLOCK_SIZE;
        copyParams.srcStride = 0;
        copyParams.rsv = 0;
        LocalTensor<UpdatesType> updateLocal = inQueueUpdates_.AllocTensor<UpdatesType>();
        DataCopyPad<UpdatesType>(updateLocal, updatesGm_[gmUpdatesOffset_], copyParams, {false, 0, 0, 0});
        inQueueUpdates_.EnQue(updateLocal);
    }

    __aicore__ inline void CopyInScale()
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = tilingData_.quantScalesElements * sizeof(ScalesType);
        copyParams.dstStride = 0;
        copyParams.srcStride = 0;
        copyParams.rsv = 0;
        LocalTensor<ScalesType> sLocal = inQueueScales_.AllocTensor<ScalesType>();
        DataCopyPad<ScalesType>(sLocal, scalesGm_, copyParams, {false, 0, 0, 0});
        inQueueScales_.EnQue(sLocal);
    }

    __aicore__ inline void CopyInOffset()
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = tilingData_.quantZeroPointsElements * sizeof(OffsetsType);
        copyParams.dstStride = 0;
        copyParams.srcStride = 0;
        copyParams.rsv = 0;
        LocalTensor<OffsetsType> oLocal = inQueueZeroPoints_.AllocTensor<OffsetsType>();
        DataCopyPad<OffsetsType>(oLocal, offsetsGm_, copyParams, {false, 0, 0, 0});
        inQueueZeroPoints_.EnQue(oLocal);
    }

    __aicore__ inline void QuantUpdate(
        int64_t bsIdx, LocalTensor<UpdatesType> updateLocal, LocalTensor<ScalesType> scaleLocal,
        LocalTensor<OffsetsType> offsetLocal)
    {
        LocalTensor<VarType> outLocal = outQueueVar_.AllocTensor<VarType>();

        __local_mem__ UpdatesType* updateLocalAddr = (__local_mem__ UpdatesType*)updateLocal.GetPhyAddr() +
                                                     bsIdx * copyBlockCount_ * tilingData_.updateOriLastDimAlign;
        __local_mem__ ScalesType* scaleLocalAddr = (__local_mem__ ScalesType*)scaleLocal.GetPhyAddr();
        __local_mem__ OffsetsType* offsetLocalAddr = (__local_mem__ OffsetsType*)offsetLocal.GetPhyAddr();
        __local_mem__ VarType* outLocalAddr = (__local_mem__ VarType*)outLocal.GetPhyAddr();

        uint16_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
        uint32_t xLocalOffset = static_cast<uint32_t>(tilingData_.updateOriLastDimAlign);
        uint16_t row = static_cast<uint16_t>(copyBlockCount_);
        uint16_t vfLoopNum = (tilingData_.updateOriLastDim + VL - 1) / VL;

        // has offset
        __VEC_SCOPE__
        {
            // update: fp16, bf16
            AscendC::MicroAPI::RegTensor<UpdatesType> vregX;
            AscendC::MicroAPI::RegTensor<float> vregFloatX;
            // scales: fp32, bp16
            AscendC::MicroAPI::RegTensor<ScalesType> vregS;
            AscendC::MicroAPI::RegTensor<float> vregFloatS;
            // zero_points: int32, bp16
            AscendC::MicroAPI::RegTensor<OffsetsType> vregO;
            AscendC::MicroAPI::RegTensor<half> vregHalfO;
            AscendC::MicroAPI::RegTensor<float> vregFloatO;
            // y: int8
            AscendC::MicroAPI::RegTensor<float> vregFloatY;
            AscendC::MicroAPI::RegTensor<int16_t> vregInt16Y;
            AscendC::MicroAPI::RegTensor<half> vregHalfY;
            AscendC::MicroAPI::RegTensor<VarType> vregY;

            AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<float>();
            for (uint16_t j = 0; j < row; ++j) {
                uint32_t count = static_cast<uint32_t>(tilingData_.updateOriLastDim);
                for (uint16_t i = 0; i < vfLoopNum; i++) {
                    mask = AscendC::MicroAPI::UpdateMask<float>(count);
                    // ld and cast for update
                    if constexpr (IsSameType<UpdatesType, half>::value) {
                        // fp16
                        AscendC::MicroAPI::DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                            vregX, updateLocalAddr + i * VL + j * xLocalOffset);
                        AscendC::MicroAPI::Cast<float, half, Base::CAST_TRAIT_HALF_TO_FP32>(vregFloatX, vregX, mask);
                    } else if constexpr (IsSameType<UpdatesType, bfloat16_t>::value) {
                        // bf16
                        AscendC::MicroAPI::DataCopy<UpdatesType, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                            vregX, updateLocalAddr + i * VL + j * xLocalOffset);
                        AscendC::MicroAPI::Cast<float, UpdatesType, Base::CAST_TRAIT_BF16_TO_FP32>(
                            vregFloatX, vregX, mask);
                    }

                    // ld and cast for scale
                    if constexpr (IsSameType<ScalesType, float>::value) {
                        // fp32
                        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                            vregFloatS, scaleLocalAddr + i * VL);
                    } else if constexpr (IsSameType<ScalesType, bfloat16_t>::value) {
                        // bf16
                        AscendC::MicroAPI::DataCopy<ScalesType, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                            vregS, scaleLocalAddr + i * VL);
                        AscendC::MicroAPI::Cast<float, ScalesType, Base::CAST_TRAIT_BF16_TO_FP32>(
                            vregFloatS, vregS, mask);
                    }

                    // ld and cast for offset
                    if constexpr (IsSameType<OffsetsType, int32_t>::value) {
                        // int32
                        AscendC::MicroAPI::DataCopy<OffsetsType, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                            vregO, offsetLocalAddr + i * VL);
                        AscendC::MicroAPI::Cast<float, OffsetsType, Base::CAST_TRAIT_INT32_TO_FP32>(
                            vregFloatO, vregO, mask);
                    } else if constexpr (IsSameType<OffsetsType, bfloat16_t>::value) {
                        // bf16
                        AscendC::MicroAPI::DataCopy<OffsetsType, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                            vregO, offsetLocalAddr + i * VL);
                        AscendC::MicroAPI::Cast<float, OffsetsType, Base::CAST_TRAIT_BF16_TO_FP32>(
                            vregFloatO, vregO, mask);
                    }
                    if constexpr (DivMode == TPL_DIV_MODE_DIV) {
                        static constexpr AscendC::MicroAPI::DivSpecificMode divMode = {
                            AscendC::MicroAPI::MaskMergeMode::ZEROING, false};
                        AscendC::MicroAPI::Div<float, &divMode>(vregFloatY, vregFloatX, vregFloatS, mask);
                    } else {
                        AscendC::MicroAPI::Mul(vregFloatY, vregFloatX, vregFloatS, mask);
                    }
                    if constexpr (!IsSameType<OffsetsType, bool>::value) {
                        AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                            vregFloatY, vregFloatY, vregFloatO, mask);
                    }
                    // cast and sd for y
                    if constexpr (IsSameType<VarType, hifloat8_t>::value) {
                        // hifp8
                        AscendC::MicroAPI::Cast<VarType, float, Base::CAST_TRAIT_FP32_TO_HIFP8>(
                            vregY, vregFloatY, mask);
                        AscendC::MicroAPI::DataCopy<VarType, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            outLocalAddr + i * VL + j * xLocalOffset, vregY, mask);
                    } else if constexpr (IsSameType<VarType, fp8_e5m2_t>::value) {
                        // fp8_e5m2
                        AscendC::MicroAPI::Cast<VarType, float, Base::CAST_TRAIT_FP32_TO_FP8E5M2>(
                            vregY, vregFloatY, mask);
                        AscendC::MicroAPI::DataCopy<VarType, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            outLocalAddr + i * VL + j * xLocalOffset, vregY, mask);
                    } else if constexpr (IsSameType<VarType, fp8_e4m3fn_t>::value) {
                        // fp8_e4m3
                        AscendC::MicroAPI::Cast<VarType, float, Base::CAST_TRAIT_FP32_TO_FP8E4M3>(
                            vregY, vregFloatY, mask);
                        AscendC::MicroAPI::DataCopy<VarType, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            outLocalAddr + i * VL + j * xLocalOffset, vregY, mask);
                    } else if constexpr (IsSameType<VarType, int8_t>::value) {
                        // int8
                        AscendC::MicroAPI::Cast<int16_t, float, Base::CAST_TRAIT_FP32_TO_INT16>(
                            vregInt16Y, vregFloatY, mask);
                        AscendC::MicroAPI::Cast<half, int16_t, Base::CAST_TRAIT_INT16_TO_HALF>(
                            vregHalfY, vregInt16Y, mask);
                        AscendC::MicroAPI::Cast<int8_t, half, Base::CAST_TRAIT_HALF_TO_INT8>(vregY, vregHalfY, mask);
                        AscendC::MicroAPI::DataCopy<VarType, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            outLocalAddr + i * VL + j * xLocalOffset, vregY, mask);
                    }
                }
            }
        }

        outQueueVar_.EnQue(outLocal);
    }

    __aicore__ inline void QuantUpdateWithoutOffset(
        int64_t bsIdx, LocalTensor<UpdatesType> updateLocal, LocalTensor<ScalesType> scaleLocal)
    {
        LocalTensor<VarType> outLocal = outQueueVar_.AllocTensor<VarType>();

        __local_mem__ UpdatesType* updateLocalAddr = (__local_mem__ UpdatesType*)updateLocal.GetPhyAddr() +
                                                     bsIdx * copyBlockCount_ * tilingData_.updateOriLastDimAlign;
        __local_mem__ ScalesType* scaleLocalAddr = (__local_mem__ ScalesType*)scaleLocal.GetPhyAddr();
        __local_mem__ VarType* outLocalAddr = (__local_mem__ VarType*)outLocal.GetPhyAddr();

        uint16_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
        uint32_t xLocalOffset = static_cast<uint32_t>(tilingData_.updateOriLastDimAlign);
        uint16_t row = static_cast<uint16_t>(copyBlockCount_);
        uint16_t vfLoopNum = (tilingData_.updateOriLastDim + VL - 1) / VL;

        // without offset
        __VEC_SCOPE__
        {
            // update: fp16, bf16
            AscendC::MicroAPI::RegTensor<UpdatesType> vregX;
            AscendC::MicroAPI::RegTensor<float> vregFloatX;
            // scales: fp32, bp16
            AscendC::MicroAPI::RegTensor<ScalesType> vregS;
            AscendC::MicroAPI::RegTensor<float> vregFloatS;
            // y: int8
            AscendC::MicroAPI::RegTensor<float> vregFloatY;
            AscendC::MicroAPI::RegTensor<int16_t> vregInt16Y;
            AscendC::MicroAPI::RegTensor<half> vregHalfY;
            AscendC::MicroAPI::RegTensor<VarType> vregY;

            AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<float>();
            for (uint16_t j = 0; j < row; ++j) {
                uint32_t count = static_cast<uint32_t>(tilingData_.updateOriLastDim);
                for (uint16_t i = 0; i < vfLoopNum; i++) {
                    mask = AscendC::MicroAPI::UpdateMask<float>(count);
                    // ld and cast for update
                    if constexpr (IsSameType<UpdatesType, half>::value) {
                        // fp16
                        AscendC::MicroAPI::DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                            vregX, updateLocalAddr + i * VL + j * xLocalOffset);
                        AscendC::MicroAPI::Cast<float, half, Base::CAST_TRAIT_HALF_TO_FP32>(vregFloatX, vregX, mask);
                    } else if constexpr (IsSameType<UpdatesType, bfloat16_t>::value) {
                        // bf16
                        AscendC::MicroAPI::DataCopy<UpdatesType, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                            vregX, updateLocalAddr + i * VL + j * xLocalOffset);
                        AscendC::MicroAPI::Cast<float, UpdatesType, Base::CAST_TRAIT_BF16_TO_FP32>(
                            vregFloatX, vregX, mask);
                    }

                    // ld and cast for scale
                    if constexpr (IsSameType<ScalesType, float>::value) {
                        // fp32
                        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                            vregFloatS, scaleLocalAddr + i * VL);
                    } else if constexpr (IsSameType<ScalesType, bfloat16_t>::value) {
                        // bf16
                        AscendC::MicroAPI::DataCopy<ScalesType, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                            vregS, scaleLocalAddr + i * VL);
                        AscendC::MicroAPI::Cast<float, ScalesType, Base::CAST_TRAIT_BF16_TO_FP32>(
                            vregFloatS, vregS, mask);
                    }

                    if constexpr (DivMode == TPL_DIV_MODE_DIV) {
                        static constexpr AscendC::MicroAPI::DivSpecificMode divMode = {
                            AscendC::MicroAPI::MaskMergeMode::ZEROING, false};
                        AscendC::MicroAPI::Div<float, &divMode>(vregFloatY, vregFloatX, vregFloatS, mask);
                    } else {
                        AscendC::MicroAPI::Mul(vregFloatY, vregFloatX, vregFloatS, mask);
                    }
                    // cast and sd for y
                    if constexpr (IsSameType<VarType, hifloat8_t>::value) {
                        // hifp8
                        AscendC::MicroAPI::Cast<VarType, float, Base::CAST_TRAIT_FP32_TO_HIFP8>(
                            vregY, vregFloatY, mask);
                        AscendC::MicroAPI::DataCopy<VarType, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            outLocalAddr + i * VL + j * xLocalOffset, vregY, mask);
                    } else if constexpr (IsSameType<VarType, fp8_e5m2_t>::value) {
                        // fp8_e5m2
                        AscendC::MicroAPI::Cast<VarType, float, Base::CAST_TRAIT_FP32_TO_FP8E5M2>(
                            vregY, vregFloatY, mask);
                        AscendC::MicroAPI::DataCopy<VarType, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            outLocalAddr + i * VL + j * xLocalOffset, vregY, mask);
                    } else if constexpr (IsSameType<VarType, fp8_e4m3fn_t>::value) {
                        // fp8_e4m3
                        AscendC::MicroAPI::Cast<VarType, float, Base::CAST_TRAIT_FP32_TO_FP8E4M3>(
                            vregY, vregFloatY, mask);
                        AscendC::MicroAPI::DataCopy<VarType, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            outLocalAddr + i * VL + j * xLocalOffset, vregY, mask);
                    } else if constexpr (IsSameType<VarType, int8_t>::value) {
                        // int8
                        AscendC::MicroAPI::Cast<int16_t, float, Base::CAST_TRAIT_FP32_TO_INT16>(
                            vregInt16Y, vregFloatY, mask);
                        AscendC::MicroAPI::Cast<half, int16_t, Base::CAST_TRAIT_INT16_TO_HALF>(
                            vregHalfY, vregInt16Y, mask);
                        AscendC::MicroAPI::Cast<int8_t, half, Base::CAST_TRAIT_HALF_TO_INT8>(vregY, vregHalfY, mask);
                        AscendC::MicroAPI::DataCopy<VarType, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            outLocalAddr + i * VL + j * xLocalOffset, vregY, mask);
                    }
                }
            }
        }

        outQueueVar_.EnQue(outLocal);
    }

    __aicore__ inline void CopyInIndices()
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = tilingData_.indexElements * sizeof(IndicesType);
        copyParams.dstStride = 0;
        copyParams.srcStride = 0;
        copyParams.rsv = 0;
        LocalTensor<IndicesType> iLocal = inQueueIndices_.AllocTensor<IndicesType>();
        DataCopyPad<IndicesType>(iLocal, indicesGm_, copyParams, {false, 0, 0, 0});
        inQueueIndices_.EnQue(iLocal);
    }

    __aicore__ inline void CalcDstOffset(int64_t bsIdx, LocalTensor<IndicesType> iLocal)
    {
        event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
        int64_t updateDim0Idx = (blockIdx_ * tilingData_.eachCoreBsNum + bsIdx) / tilingData_.updateDim1;
        int64_t updateDim1Idx = (blockIdx_ * tilingData_.eachCoreBsNum + bsIdx) % tilingData_.updateDim1;
        if (tilingData_.indicesShapeRank == this->INDICES_SHAPE_RANK_2) {
            int64_t varDim0Idx = iLocal.GetValue(2 * updateDim0Idx);
            int64_t axisOffset = iLocal.GetValue(2 * updateDim0Idx + 1);
            int64_t actualBsIdx = varDim0Idx * tilingData_.varDim1 + updateDim1Idx;
            gmVarOffset_ = actualBsIdx * tilingData_.dstBsStride + axisOffset * tilingData_.varDim3;
        } else {
            int64_t axisOffset = iLocal.GetValue(updateDim0Idx);
            gmVarOffset_ = (blockIdx_ * tilingData_.eachCoreBsNum + bsIdx) * tilingData_.dstBsStride +
                           axisOffset * tilingData_.varDim3;
        }
        event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    }

    __aicore__ inline void CopyOutVar()
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = copyBlockCount_;
        copyParams.blockLen = tilingData_.updateOriLastDim * sizeof(VarType);
        copyParams.dstStride = 0;
        copyParams.srcStride =
            (tilingData_.updateOriLastDimAlign - tilingData_.updateOriLastDim) * sizeof(VarType) / this->BLOCK_SIZE;
        copyParams.rsv = 0;
        LocalTensor<VarType> outLocal = outQueueVar_.DeQue<VarType>();
        DataCopyPad<VarType>(varGm_[gmVarOffset_], outLocal, copyParams);
        outQueueVar_.FreeTensor(outLocal);
    }
};
} // namespace QuantUpdateScatter
#endif // QUANT_UPDATE_SCATTER_REGBASE_