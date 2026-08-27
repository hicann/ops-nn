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
 * \file quant_update_scatter_large_batch_regbase.h
 * \brief quant_update_scatter for large_batch template
 */
#ifndef QUANT_UPDATE_SCATTER_LARGE_BATCH_REGBASE_BASE_H_
#define QUANT_UPDATE_SCATTER_LARGE_BATCH_REGBASE_BASE_H_

#include "kernel_tiling/kernel_tiling.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_vec_intf.h"
#else
#include "kernel_operator.h"
#endif

namespace QuantUpdateScatter {
using namespace AscendC;

template <typename VarType, typename IndicesType, typename UpdatesType, typename ScalesType, typename OffsetsType,
          uint64_t DivMode, uint64_t CastRoundMode>
class QuantUpdateScatterLargeBatchRegbase : public QuantUpdateScatterBase<VarType, IndicesType, UpdatesType, ScalesType,
                                                                          OffsetsType, DivMode, CastRoundMode> {
public:
    __aicore__ inline QuantUpdateScatterLargeBatchRegbase(){};
    using Base = QuantUpdateScatterBase<VarType, IndicesType, UpdatesType, ScalesType, OffsetsType, DivMode,
                                        CastRoundMode>;
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

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR quant_scales,
                                GM_ADDR quant_zero_points, GM_ADDR out, const QuantUpdateScatterTilingData* tiling)
    {
        Base::SetFloatOverflowModeForRegbase();
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
        pipe_.InitBuffer(inQueueIndices_, BUFFER_NUM,
                         this->CeilAlign(tilingData_.indexElements * sizeof(IndicesType), this->BLOCK_SIZE));
        pipe_.InitBuffer(inQueueUpdates_, BUFFER_NUM,
                         this->CeilAlign(copyBlockCount_ * tilingData_.updateOriLastDimAlign * sizeof(UpdatesType),
                                         this->BLOCK_SIZE));
        pipe_.InitBuffer(inQueueScales_, BUFFER_NUM,
                         this->CeilAlign(tilingData_.quantScalesElements * sizeof(ScalesType), this->BLOCK_SIZE));
        if constexpr (!IsSameType<OffsetsType, bool>::value) {
            pipe_.InitBuffer(
                inQueueZeroPoints_, BUFFER_NUM,
                this->CeilAlign(tilingData_.quantZeroPointsElements * sizeof(OffsetsType), this->BLOCK_SIZE));
        }

        pipe_.InitBuffer(
            outQueueVar_, BUFFER_NUM,
            this->CeilAlign(copyBlockCount_ * tilingData_.updateOriLastDimAlign * sizeof(VarType), this->BLOCK_SIZE));
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

        CopyInIndices();
        CopyInScale();
        LocalTensor<OffsetsType> offsetLocal;
        if constexpr (!IsSameType<OffsetsType, bool>::value) {
            CopyInOffset();
            offsetLocal = inQueueZeroPoints_.DeQue<OffsetsType>();
        }
        LocalTensor<IndicesType> iLocal = inQueueIndices_.DeQue<IndicesType>();
        LocalTensor<ScalesType> scaleLocal = inQueueScales_.DeQue<ScalesType>();

        for (int64_t bsIdx = 0; bsIdx < coreBsNum_; bsIdx++) {
            CopyInUpdate(bsIdx);
            if constexpr (!IsSameType<OffsetsType, bool>::value) {
                QuantUpdate(scaleLocal, offsetLocal);
            } else {
                QuantUpdateWithoutOffset(scaleLocal);
            }
            CalcDstOffset(bsIdx, iLocal);
            CopyOutVar(bsIdx);
        }
        inQueueIndices_.FreeTensor(iLocal);
        inQueueScales_.FreeTensor(scaleLocal);
        if constexpr (!IsSameType<OffsetsType, bool>::value) {
            inQueueZeroPoints_.FreeTensor(offsetLocal);
        }
    }

    __aicore__ inline void CopyInUpdate(int64_t bsIdx)
    {
        gmUpdatesOffset_ = (blockIdx_ * tilingData_.eachCoreBsNum + bsIdx) * tilingData_.srcBsStride;
        DataCopyExtParams copyParams;
        copyParams.blockCount = copyBlockCount_;
        copyParams.blockLen = tilingData_.updateOriLastDim * sizeof(UpdatesType);
        copyParams.dstStride = (tilingData_.updateOriLastDimAlign - tilingData_.updateOriLastDim) *
                               sizeof(UpdatesType) / this->BLOCK_SIZE;
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

    __aicore__ inline void QuantUpdate(LocalTensor<ScalesType> scaleLocal, LocalTensor<OffsetsType> offsetLocal)
    {
        LocalTensor<UpdatesType> updateLocal = inQueueUpdates_.DeQue<UpdatesType>();
        LocalTensor<VarType> outLocal = outQueueVar_.AllocTensor<VarType>();

        __ubuf__ UpdatesType* updateLocalAddr = (__ubuf__ UpdatesType*)updateLocal.GetPhyAddr();
        __ubuf__ ScalesType* scaleLocalAddr = (__ubuf__ ScalesType*)scaleLocal.GetPhyAddr();
        __ubuf__ OffsetsType* offsetLocalAddr = (__ubuf__ OffsetsType*)offsetLocal.GetPhyAddr();
        __ubuf__ VarType* outLocalAddr = (__ubuf__ VarType*)outLocal.GetPhyAddr();

        uint16_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
        uint32_t xLocalOffset = tilingData_.updateOriLastDimAlign;
        uint16_t row = static_cast<uint16_t>(copyBlockCount_);
        uint16_t vfLoopNum = (tilingData_.updateOriLastDim + VL - 1) / VL;

        // has offset
        __VEC_SCOPE__
        {
            // update: fp16, bf16
            AscendC::Reg::RegTensor<UpdatesType> vregX;
            AscendC::Reg::RegTensor<float> vregFloatX;
            // scales: fp32, bp16
            AscendC::Reg::RegTensor<ScalesType> vregS;
            AscendC::Reg::RegTensor<float> vregFloatS;
            // zero_points: int32, bp16
            AscendC::Reg::RegTensor<OffsetsType> vregO;
            AscendC::Reg::RegTensor<half> vregHalfO;
            AscendC::Reg::RegTensor<float> vregFloatO;
            // y: int8
            AscendC::Reg::RegTensor<float> vregFloatY;
            AscendC::Reg::RegTensor<int16_t> vregInt16Y;
            AscendC::Reg::RegTensor<half> vregHalfY;
            AscendC::Reg::RegTensor<VarType> vregY;

            AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float>();
            for (uint16_t j = 0; j < row; ++j) {
                uint32_t count = static_cast<uint32_t>(tilingData_.updateOriLastDim);
                for (uint16_t i = 0; i < vfLoopNum; i++) {
                    mask = AscendC::Reg::UpdateMask<float>(count);
                    // ld and cast for update
                    if constexpr (IsSameType<UpdatesType, half>::value) {
                        // fp16
                        AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                            vregX, updateLocalAddr + i * VL + j * xLocalOffset);
                        AscendC::Reg::Cast<float, half, Base::CAST_TRAIT_HALF_TO_FP32>(vregFloatX, vregX, mask);
                    } else if constexpr (IsSameType<UpdatesType, bfloat16_t>::value) {
                        // bf16
                        AscendC::Reg::LoadAlign<UpdatesType, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                            vregX, updateLocalAddr + i * VL + j * xLocalOffset);
                        AscendC::Reg::Cast<float, UpdatesType, Base::CAST_TRAIT_BF16_TO_FP32>(vregFloatX, vregX, mask);
                    }

                    // ld and cast for scale
                    if constexpr (IsSameType<ScalesType, float>::value) {
                        // fp32
                        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(vregFloatS,
                                                                                          scaleLocalAddr + i * VL);
                    } else if constexpr (IsSameType<ScalesType, bfloat16_t>::value) {
                        // bf16
                        AscendC::Reg::LoadAlign<ScalesType, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                            vregS, scaleLocalAddr + i * VL);
                        AscendC::Reg::Cast<float, ScalesType, Base::CAST_TRAIT_BF16_TO_FP32>(vregFloatS, vregS, mask);
                    }
                    // ld and cast for offset
                    if constexpr (IsSameType<OffsetsType, int32_t>::value) {
                        // int32
                        AscendC::Reg::LoadAlign<OffsetsType, AscendC::Reg::LoadDist::DIST_NORM>(
                            vregO, offsetLocalAddr + i * VL);
                        AscendC::Reg::Cast<float, OffsetsType, Base::CAST_TRAIT_INT32_TO_FP32>(vregFloatO, vregO, mask);
                    } else if constexpr (IsSameType<OffsetsType, bfloat16_t>::value) {
                        // bf16
                        AscendC::Reg::LoadAlign<OffsetsType, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                            vregO, offsetLocalAddr + i * VL);
                        AscendC::Reg::Cast<float, OffsetsType, Base::CAST_TRAIT_BF16_TO_FP32>(vregFloatO, vregO, mask);
                    }

                    if constexpr (DivMode == TPL_DIV_MODE_DIV) {
                        static constexpr AscendC::Reg::DivSpecificMode divMode = {AscendC::Reg::MaskMergeMode::ZEROING,
                                                                                  false};
                        AscendC::Reg::Div<float, &divMode>(vregFloatY, vregFloatX, vregFloatS, mask);
                    } else {
                        AscendC::Reg::Mul(vregFloatY, vregFloatX, vregFloatS, mask);
                    }
                    if constexpr (!IsSameType<OffsetsType, bool>::value) {
                        AscendC::Reg::Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(vregFloatY, vregFloatY,
                                                                                       vregFloatO, mask);
                    }
                    // cast and sd for y
                    if constexpr (IsSameType<VarType, hifloat8_t>::value) {
                        // hifp8
                        AscendC::Reg::Cast<VarType, float, Base::CAST_TRAIT_FP32_TO_HIFP8>(vregY, vregFloatY, mask);
                        AscendC::Reg::StoreAlign<VarType, AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                            outLocalAddr + i * VL + j * xLocalOffset, vregY, mask);
                    } else if constexpr (IsSameType<VarType, fp8_e5m2_t>::value) {
                        // fp8_e5m2
                        AscendC::Reg::Cast<VarType, float, Base::CAST_TRAIT_FP32_TO_FP8E5M2>(vregY, vregFloatY, mask);
                        AscendC::Reg::StoreAlign<VarType, AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                            outLocalAddr + i * VL + j * xLocalOffset, vregY, mask);
                    } else if constexpr (IsSameType<VarType, fp8_e4m3fn_t>::value) {
                        // fp8_e4m3
                        AscendC::Reg::Cast<VarType, float, Base::CAST_TRAIT_FP32_TO_FP8E4M3>(vregY, vregFloatY, mask);
                        AscendC::Reg::StoreAlign<VarType, AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                            outLocalAddr + i * VL + j * xLocalOffset, vregY, mask);
                    } else if constexpr (IsSameType<VarType, int8_t>::value) {
                        // int8
                        AscendC::Reg::Cast<int16_t, float, Base::CAST_TRAIT_FP32_TO_INT16>(vregInt16Y, vregFloatY,
                                                                                           mask);
                        AscendC::Reg::Cast<half, int16_t, Base::CAST_TRAIT_INT16_TO_HALF>(vregHalfY, vregInt16Y, mask);
                        AscendC::Reg::Cast<int8_t, half, Base::CAST_TRAIT_HALF_TO_INT8>(vregY, vregHalfY, mask);
                        AscendC::Reg::StoreAlign<VarType, AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                            outLocalAddr + i * VL + j * xLocalOffset, vregY, mask);
                    }
                }
            }
        }
        inQueueUpdates_.FreeTensor(updateLocal);
        outQueueVar_.EnQue(outLocal);
    }

    __aicore__ inline void QuantUpdateWithoutOffset(LocalTensor<ScalesType> scaleLocal)
    {
        LocalTensor<UpdatesType> updateLocal = inQueueUpdates_.DeQue<UpdatesType>();
        LocalTensor<VarType> outLocal = outQueueVar_.AllocTensor<VarType>();

        __ubuf__ UpdatesType* updateLocalAddr = (__ubuf__ UpdatesType*)updateLocal.GetPhyAddr();
        __ubuf__ ScalesType* scaleLocalAddr = (__ubuf__ ScalesType*)scaleLocal.GetPhyAddr();
        __ubuf__ VarType* outLocalAddr = (__ubuf__ VarType*)outLocal.GetPhyAddr();

        uint16_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
        uint32_t xLocalOffset = tilingData_.updateOriLastDimAlign;
        uint16_t row = static_cast<uint16_t>(copyBlockCount_);
        uint16_t vfLoopNum = (tilingData_.updateOriLastDim + VL - 1) / VL;

        // without offset
        __VEC_SCOPE__
        {
            // update: fp16, bf16
            AscendC::Reg::RegTensor<UpdatesType> vregX;
            AscendC::Reg::RegTensor<float> vregFloatX;
            // scales: fp32, bp16
            AscendC::Reg::RegTensor<ScalesType> vregS;
            AscendC::Reg::RegTensor<float> vregFloatS;
            // y: int8
            AscendC::Reg::RegTensor<float> vregFloatY;
            AscendC::Reg::RegTensor<int16_t> vregInt16Y;
            AscendC::Reg::RegTensor<half> vregHalfY;
            AscendC::Reg::RegTensor<VarType> vregY;

            AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float>();
            for (uint16_t j = 0; j < row; ++j) {
                uint32_t count = static_cast<uint32_t>(tilingData_.updateOriLastDim);
                for (uint16_t i = 0; i < vfLoopNum; i++) {
                    mask = AscendC::Reg::UpdateMask<float>(count);
                    // ld and cast for update
                    if constexpr (IsSameType<UpdatesType, half>::value) {
                        // fp16
                        AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                            vregX, updateLocalAddr + i * VL + j * xLocalOffset);
                        AscendC::Reg::Cast<float, half, Base::CAST_TRAIT_HALF_TO_FP32>(vregFloatX, vregX, mask);
                    } else if constexpr (IsSameType<UpdatesType, bfloat16_t>::value) {
                        // bf16
                        AscendC::Reg::LoadAlign<UpdatesType, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                            vregX, updateLocalAddr + i * VL + j * xLocalOffset);
                        AscendC::Reg::Cast<float, UpdatesType, Base::CAST_TRAIT_BF16_TO_FP32>(vregFloatX, vregX, mask);
                    }

                    // ld and cast for scale
                    if constexpr (IsSameType<ScalesType, float>::value) {
                        // fp32
                        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(vregFloatS,
                                                                                          scaleLocalAddr + i * VL);
                    } else if constexpr (IsSameType<ScalesType, bfloat16_t>::value) {
                        // bf16
                        AscendC::Reg::LoadAlign<ScalesType, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                            vregS, scaleLocalAddr + i * VL);
                        AscendC::Reg::Cast<float, ScalesType, Base::CAST_TRAIT_BF16_TO_FP32>(vregFloatS, vregS, mask);
                    }

                    if constexpr (DivMode == TPL_DIV_MODE_DIV) {
                        static constexpr AscendC::Reg::DivSpecificMode divMode = {AscendC::Reg::MaskMergeMode::ZEROING,
                                                                                  false};
                        AscendC::Reg::Div<float, &divMode>(vregFloatY, vregFloatX, vregFloatS, mask);
                    } else {
                        AscendC::Reg::Mul(vregFloatY, vregFloatX, vregFloatS, mask);
                    }

                    // cast and sd for y
                    if constexpr (IsSameType<VarType, hifloat8_t>::value) {
                        // hifp8
                        AscendC::Reg::Cast<VarType, float, Base::CAST_TRAIT_FP32_TO_HIFP8>(vregY, vregFloatY, mask);
                        AscendC::Reg::StoreAlign<VarType, AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                            outLocalAddr + i * VL + j * xLocalOffset, vregY, mask);
                    } else if constexpr (IsSameType<VarType, fp8_e5m2_t>::value) {
                        // fp8_e5m2
                        AscendC::Reg::Cast<VarType, float, Base::CAST_TRAIT_FP32_TO_FP8E5M2>(vregY, vregFloatY, mask);
                        AscendC::Reg::StoreAlign<VarType, AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                            outLocalAddr + i * VL + j * xLocalOffset, vregY, mask);
                    } else if constexpr (IsSameType<VarType, fp8_e4m3fn_t>::value) {
                        // fp8_e4m3
                        AscendC::Reg::Cast<VarType, float, Base::CAST_TRAIT_FP32_TO_FP8E4M3>(vregY, vregFloatY, mask);
                        AscendC::Reg::StoreAlign<VarType, AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                            outLocalAddr + i * VL + j * xLocalOffset, vregY, mask);
                    } else if constexpr (IsSameType<VarType, int8_t>::value) {
                        // int8
                        AscendC::Reg::Cast<int16_t, float, Base::CAST_TRAIT_FP32_TO_INT16>(vregInt16Y, vregFloatY,
                                                                                           mask);
                        AscendC::Reg::Cast<half, int16_t, Base::CAST_TRAIT_INT16_TO_HALF>(vregHalfY, vregInt16Y, mask);
                        AscendC::Reg::Cast<int8_t, half, Base::CAST_TRAIT_HALF_TO_INT8>(vregY, vregHalfY, mask);
                        AscendC::Reg::StoreAlign<VarType, AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                            outLocalAddr + i * VL + j * xLocalOffset, vregY, mask);
                    }
                }
            }
        }
        inQueueUpdates_.FreeTensor(updateLocal);
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

    __aicore__ inline void CopyOutVar(int64_t bsIdx)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = copyBlockCount_;
        copyParams.blockLen = tilingData_.updateOriLastDim * sizeof(VarType);
        copyParams.dstStride = 0;
        copyParams.srcStride = (tilingData_.updateOriLastDimAlign - tilingData_.updateOriLastDim) * sizeof(VarType) /
                               this->BLOCK_SIZE;
        copyParams.rsv = 0;
        LocalTensor<VarType> outLocal = outQueueVar_.DeQue<VarType>();
        DataCopyPad<VarType>(varGm_[gmVarOffset_], outLocal, copyParams);
        outQueueVar_.FreeTensor(outLocal);
    }
};
} // namespace QuantUpdateScatter
#endif // QuantUpdateScatterLargeBatchRegbase
