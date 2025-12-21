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
 * \file modulate_grad.cpp
 * \brief
 */
#include "kernel_operator.h"
#include <acl/acl.h>

namespace AscendC{
template<typename T>
class ModulateGradKernel
{
private:
    static constexpr uint32_t UB_CAPACITY = 192 * 1024;//UB size
    static constexpr uint32_t MAX_TILE_LENGTH = 12 * 1024 / sizeof(T);

    //GlobalTensor
    //input
    GlobalTensor<T> inputGm_, gradOutputGm_, scaleGm_;
    //output 
    GlobalTensor<T> gradInputGm_, gradScaleGm_, gradShiftGm_;
    //LocalTensor
    LocalTensor<T> inputLocal, gradOutputLocal, scaleLocal;

    //QUEUE
    TPipe pipe;

    TQue<TPosition::VECIN, 1> inputQueue_;
    TQue<TPosition::VECIN, 1> gradOutputQueue_;
    TQue<TPosition::VECIN, 1> scaleQueue_;

    TQue<TPosition::VECIN, 1> tileGradShiftQueue_; 
    TQue<TPosition::VECIN, 1> tileGradScaleQueue_;
    TQue<TPosition::VECIN, 1> currenttileGradScaleQueue_;

    TQue<TPosition::VECOUT, 1> gradInputQueue_;
    TQue<TPosition::VECOUT, 1> gradShiftLocalQueue_;
    TQue<TPosition::VECOUT, 1> gradScaleLocalQueue_;

    // 分块参数
    uint32_t B_, L_, D_, has_scale_, has_shift_;
    uint32_t splitByD_, coresPerB_, formerNum_, formerLength_;
    uint32_t tailNum_, tailLength_, maxRowsPerTile_;

    //当前核心处理的相关参数
    uint32_t coreId_, b_index_, d_start_, totalD_, totalcutD_;
    uint32_t currentL_, currentD_, len_, num, cores_;
    //B较大时分块参数
    uint32_t  b_start, tileNumB_, rowBytes, alignedRowBytes;
    uint8_t rowPadElements;
    bool isDOverflow; 
    uint32_t tileLIndex = 0;
    uint32_t tileDIndex = 0;
    uint32_t tileBIndex = 0;
    //D分块参数
    uint32_t tileNumD_; 
    uint32_t lastTileLengthD_;
    uint32_t d_size;
    //L分行参数
    uint32_t tileNumL_;
    uint32_t lastTileLengthL_;

public:
    __aicore__ inline ModulateGradKernel(){}

    __aicore__ inline void Init(const ModulateGradTiling& tiling,
                                GM_ADDR inputGm,GM_ADDR gradOutputGm,GM_ADDR gradInputGm,
                                GM_ADDR scaleGm,GM_ADDR gradScaleGm,GM_ADDR gradShiftGm){ 
            InitGlobalTensors(inputGm, gradOutputGm, gradInputGm, scaleGm, gradScaleGm, gradShiftGm);
            InitTilingParams(tiling);
            InitCoreSpecificParams(tiling);
            InitTileParams();
            InitPipeBuffers();
    }

    __aicore__ inline void CopyIn(uint32_t tileLIndex){
        PipeBarrier<PIPE_ALL>();
        uint32_t allOffset, scaleOffset, LenD, lenD;
        CalculateOffsets(tileLIndex, allOffset, scaleOffset, LenD, lenD);
        PipeBarrier<PIPE_ALL>();
        CopyInInput(allOffset, LenD, lenD);
        PipeBarrier<PIPE_ALL>();
        CopyInGradOutput(allOffset, LenD, lenD);
        PipeBarrier<PIPE_ALL>();
        CopyInScale(scaleOffset, lenD);
    }
    
    __aicore__ inline void Compute(uint32_t tileLIndex){
        LocalTensor<T> computeinputLocal = inputQueue_.DeQue<T>();
        LocalTensor<T> computegradOutputLocal = gradOutputQueue_.DeQue<T>();
        LocalTensor<T> computescaleLocal = scaleQueue_.DeQue<T>();
        LocalTensor<T> computegradInputLocal = gradInputQueue_.AllocTensor<T>();
        computegradInputLocal.SetSize(currentL_ * alignedRowBytes);

        ComputeGradients(computeinputLocal, computegradOutputLocal, computescaleLocal, computegradInputLocal);
        PipeBarrier<PIPE_ALL>();
        inputQueue_.FreeTensor(computeinputLocal);
        gradOutputQueue_.FreeTensor(computegradOutputLocal);
        scaleQueue_.FreeTensor(computescaleLocal);
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void CopyOut(uint32_t tileIndex){
        uint32_t allOffset, scaleOffset, LenD, lenD;
        CalculateOffsets(tileLIndex, allOffset, scaleOffset, LenD, lenD);
        PipeBarrier<PIPE_ALL>();
        CopyOutGradInput(allOffset, LenD, lenD);
        PipeBarrier<PIPE_ALL>();
        if (tileLIndex == tileNumL_ - 1 && (has_scale_ || has_shift_)){
            CopyOutScaleShift(scaleOffset, lenD);
        }
    }

    __aicore__ inline void Process(){
        if (coreId_ >= cores_) return;
        for (tileBIndex = 0; tileBIndex < tileNumB_; ++tileBIndex){
            for (tileDIndex = 0; tileDIndex < tileNumD_; ++tileDIndex){
                UpdatetileParamsDsize();
                for (tileLIndex = 0; tileLIndex < tileNumL_; ++tileLIndex){
                    CopyIn(tileLIndex);
                    PipeBarrier<PIPE_ALL>();
                    Compute(tileLIndex);
                    PipeBarrier<PIPE_ALL>();
                    CopyOut(tileLIndex);
                    PipeBarrier<PIPE_ALL>();
                }
            }
        }
    }
    
private:
    __aicore__ inline void InitGlobalTensors(GM_ADDR inputGm,GM_ADDR gradOutputGm, GM_ADDR gradInputGm,
                                            GM_ADDR scaleGm, GM_ADDR gradScaleGm, GM_ADDR gradShiftGm){
        inputGm_.SetGlobalBuffer((__gm__ float*)inputGm);
        gradOutputGm_.SetGlobalBuffer((__gm__ float*)gradOutputGm);
        scaleGm_.SetGlobalBuffer((__gm__ float*)scaleGm);

        gradInputGm_.SetGlobalBuffer((__gm__ float*)gradInputGm);
        gradShiftGm_.SetGlobalBuffer((__gm__ float*)gradShiftGm);
        gradScaleGm_.SetGlobalBuffer((__gm__ float*)gradScaleGm);
    }

    __aicore__ inline void UpdatetileParamsDsize(){
        if (isDOverflow) {
            d_size = (tileDIndex == tileNumD_ - 1) ? lastTileLengthD_ : totalcutD_ ;
            rowBytes = d_size * sizeof(float);
            uint32_t rowPadBytes = (32 - (rowBytes % 32)) % 32;
            alignedRowBytes = rowBytes + rowPadBytes;
            currentD_ = alignedRowBytes / sizeof(T);
            rowPadElements = rowPadBytes / sizeof(T);
            len_ = currentL_ * alignedRowBytes;
        }
    }
    __aicore__ inline void InitTilingParams(const ModulateGradTiling& tiling){
        has_scale_ = tiling.has_scale;
        has_shift_ = tiling.has_shift;
        B_ = tiling.B;
        L_ = tiling.L;
        D_ = tiling.D;

        splitByD_ = tiling.splitB;
        coresPerB_ = tiling.coresPerB;
        formerNum_ = tiling.formerNum;
        formerLength_= tiling.formerLength;
        tailNum_ = tiling.tailNum;
        tailLength_ = tiling.tailLength;
    }
    __aicore__ inline void InitCoreSpecificParams(const ModulateGradTiling& tiling){
        coreId_ = GetBlockIdx();
        if (splitByD_ != 0){
            b_index_= coreId_ / tiling.coresPerB;
            uint32_t groupId = coreId_ % tiling.coresPerB;
            tileNumB_ = 1;

            if (b_index_ >= tiling.usedCores){
                totalD_ = 0;
                tileNumL_ = 0;
                return;
            }

            if (groupId < formerNum_){
                d_start_ = groupId * formerLength_;
                totalD_ = formerLength_;
            }else{
                d_start_ = formerNum_ * formerLength_ + (groupId - formerNum_) * tailLength_;
                totalD_ = tailLength_;
            }
            cores_ = tiling.usedCores;
        }else{
            d_start_ = 0;
            if (coreId_ < formerNum_){
                b_index_ = coreId_ * formerLength_;
                tileNumB_ = formerLength_;
            }else{
                b_index_ = formerNum_ * formerLength_ + (coreId_ - formerNum_) * tailLength_;
                tileNumB_ = tailLength_;
            }
            totalD_ = D_;
            cores_ = tiling.usedCores;
        }
    }

    __aicore__ inline void InitTileParams(){
        maxRowsPerTile_ = MAX_TILE_LENGTH / totalD_;
        isDOverflow = (maxRowsPerTile_ == 0);

        if (isDOverflow) {
            totalcutD_ = MAX_TILE_LENGTH;
            uint32_t standard_rowBytes = totalcutD_ * sizeof(float);
            uint32_t standard_rowPadBytes = (32 - (standard_rowBytes % 32)) % 32;
            uint32_t standard_alignedRowBytes = standard_rowBytes + standard_rowPadBytes;
            uint32_t standard_currentD = standard_alignedRowBytes / sizeof(T);

            tileNumD_ = (totalD_ + standard_currentD -1) / standard_currentD;
            lastTileLengthD_ = totalD_ - (tileNumD_ - 1) * standard_currentD;
        
            d_size = (tileDIndex == tileNumD_ - 1) ? lastTileLengthD_ : totalcutD_ ;
            rowBytes = d_size * sizeof(float);
            uint32_t rowPadBytes = (32 - (rowBytes % 32)) % 32;
            alignedRowBytes = rowBytes + rowPadBytes;
            currentD_ = alignedRowBytes / sizeof(T);
            rowPadElements = rowPadBytes / sizeof(T);
            maxRowsPerTile_ = 1;
            currentL_ = 1;
            tileNumL_ = L_;
            len_ = currentL_ * alignedRowBytes;
        }else{
            d_size = totalD_;
            rowBytes = d_size * sizeof(float);
            uint32_t rowPadBytes = (32 - (rowBytes % 32)) % 32;
            alignedRowBytes = rowBytes + rowPadBytes;
            currentD_ = alignedRowBytes / sizeof(T);
            rowPadElements = rowPadBytes / sizeof(T);
            maxRowsPerTile_ = MAX_TILE_LENGTH / currentD_;
            tileNumD_ = 1; 
            tileNumL_ = (L_ + maxRowsPerTile_ - 1) / maxRowsPerTile_;
            lastTileLengthL_ = L_ - (tileNumL_ - 1) * maxRowsPerTile_;
            currentL_ = (tileLIndex == tileNumL_ - 1) ? lastTileLengthL_ : maxRowsPerTile_;
            len_ = currentL_ * alignedRowBytes;
        }
    }

    __aicore__ inline void InitPipeBuffers(){
        pipe.InitBuffer(inputQueue_, 1, len_);
        pipe.InitBuffer(gradOutputQueue_, 1, len_);
        pipe.InitBuffer(scaleQueue_, 1, alignedRowBytes);
        pipe.InitBuffer(gradInputQueue_, 1, len_);
        pipe.InitBuffer(gradShiftLocalQueue_, 1, alignedRowBytes);
        pipe.InitBuffer(gradScaleLocalQueue_, 1, alignedRowBytes);
        pipe.InitBuffer(tileGradShiftQueue_, 1, alignedRowBytes);
        pipe.InitBuffer(tileGradScaleQueue_, 1, alignedRowBytes);
        pipe.InitBuffer(currenttileGradScaleQueue_, 1, alignedRowBytes);
    }

    __aicore__ inline void CalculateOffsets(uint32_t tileLIndex, uint32_t& allOffset,uint32_t& scaleOffset,
                                                uint32_t& LenD, uint32_t& lenD){
        LenD = D_;                   
        if (isDOverflow){
            lenD = (tileDIndex == tileNumD_ - 1) ? lastTileLengthD_ : MAX_TILE_LENGTH;
            uint32_t d_block_offset;
            if (tileDIndex == 0){
                d_block_offset = 0;
            }else {
                d_block_offset = tileDIndex * MAX_TILE_LENGTH;
                if (d_block_offset + lenD > totalD_){
                    d_block_offset = totalD_ - lenD;
                }
            }
            uint32_t offset = tileLIndex;
            allOffset = (b_index_ + tileBIndex) * L_ * D_ + offset * D_ + d_start_ + d_block_offset;
            scaleOffset = (b_index_ + tileBIndex) * D_ + d_start_ + d_block_offset;
        }else{
            lenD = totalD_;
            uint32_t offset;
            if(tileLIndex == tileNumL_ - 1)
            {
                currentL_ = lastTileLengthL_;
                offset = L_ - lastTileLengthL_;
            }else {
                currentL_ = maxRowsPerTile_;
                offset = tileLIndex * maxRowsPerTile_;
            }
            allOffset = (b_index_ + tileBIndex) * L_ * D_ + offset * D_ + d_start_;
            scaleOffset = (b_index_ + tileBIndex) * D_ + d_start_;
        }    
    }

    __aicore__ inline void CopyInInput(uint32_t allOffset, uint32_t LenD,uint32_t lenD){
        LocalTensor<T> inputLocal = inputQueue_.AllocTensor<T>();

        DataCopyExtParams inputCopyParams{
            static_cast <uint16_t>(currentL_),
            static_cast <uint32_t>(rowBytes),
            static_cast <uint32_t>((LenD - lenD) * sizeof(T)),
            0,
            0
        };
        DataCopyPadExtParams<T> inputpadParams{true, 0, rowPadElements, static_cast<T>(0)};

        DataCopyPad(inputLocal[0], inputGm_[allOffset], inputCopyParams, inputpadParams);
        inputQueue_.EnQue(inputLocal);
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void CopyInGradOutput(uint32_t allOffset, uint32_t LenD,uint32_t lenD){
        LocalTensor<T> gradOutputLocal = gradOutputQueue_.AllocTensor<T>();

        DataCopyExtParams gradOutputCopyParams{
            static_cast <uint16_t>(currentL_),
            static_cast <uint32_t>(rowBytes),
            static_cast <uint32_t>((LenD - lenD) * sizeof(T)),
            0,
            0
        };
        DataCopyPadExtParams<T> gradOutputCopypadParams{true, 0, rowPadElements, static_cast<T>(0)};

        DataCopyPad(gradOutputLocal[0], gradOutputGm_[allOffset], gradOutputCopyParams, gradOutputCopypadParams);
        gradOutputQueue_.EnQue(gradOutputLocal);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyInScale(uint32_t scaleOffset, uint32_t lenD){
        LocalTensor<T> scaleLocal = scaleQueue_.AllocTensor<T>();
        DataCopyExtParams scaleParams{
            static_cast <uint16_t>(1),
            static_cast <uint32_t>(rowBytes),
            0,
            0,
            0
        };

        DataCopyPadExtParams<T> scalePadParams{true, 0, rowPadElements, static_cast<T>(0)};

        DataCopyPad(scaleLocal, scaleGm_[scaleOffset], scaleParams, scalePadParams);
        scaleQueue_.EnQue(scaleLocal);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ComputeGradients(LocalTensor<T>& computeinputLocal,LocalTensor<T>& computegradOutputLocal, LocalTensor<T>& computescaleLocal,LocalTensor<T>& computegradInputLocal){
        if (!has_scale_ && !has_shift_){
            DataCopy(computegradInputLocal[0], computegradOutputLocal[0], currentL_ * currentD_);
            gradInputQueue_.EnQue(computegradInputLocal);
            return;}
        LocalTensor<float> tileGradShift = tileGradShiftQueue_.AllocTensor<float>();
        Duplicate<T>(tileGradShift, static_cast<T>(0), d_size);
        tileGradShift.SetSize(d_size);
        LocalTensor<float> tileGradScale = tileGradScaleQueue_.AllocTensor<float>();
        Duplicate<T>(tileGradScale,static_cast<T>(0), d_size);
        tileGradScale.SetSize(d_size);
        LocalTensor<float> currenttileGradScale = currenttileGradScaleQueue_.AllocTensor<float>();
        Duplicate<T>(currenttileGradScale, static_cast<T>(0), d_size);
        currenttileGradScale.SetSize(d_size);
        LocalTensor<float> computegradScaleLocal_, computegradShiftLocal_;
        if (tileLIndex == 0) {
            computegradShiftLocal_ = gradShiftLocalQueue_.AllocTensor<float>();
            computegradScaleLocal_ = gradScaleLocalQueue_.AllocTensor<float>();
            Duplicate<T>(computegradShiftLocal_, static_cast<T>(0), d_size);
            Duplicate<T>(computegradScaleLocal_, static_cast<T>(0), d_size);
            gradShiftLocalQueue_.EnQue(computegradShiftLocal_);
            gradScaleLocalQueue_.EnQue(computegradScaleLocal_);}
        computegradShiftLocal_ = gradShiftLocalQueue_.DeQue<float>();
        computegradScaleLocal_ = gradScaleLocalQueue_.DeQue<float>();
        PipeBarrier<PIPE_ALL>();
        if (has_scale_ || has_shift_){
            for (uint32_t l = 0; l<currentL_; ++l){
                uint32_t address = currentD_ * l;
                if (has_shift_){
                    Add(tileGradShift[0], tileGradShift[0], computegradOutputLocal[address], d_size);
                }
                if(has_scale_){
                    Mul(computegradInputLocal[address], computegradOutputLocal[address], computescaleLocal[0], d_size);
                    PipeBarrier<PIPE_V>();
                    Mul(currenttileGradScale[0], computeinputLocal[address], computegradOutputLocal[address], d_size);
                    PipeBarrier<PIPE_V>();
                    Add(tileGradScale[0], tileGradScale[0], currenttileGradScale[0], d_size);
                    PipeBarrier<PIPE_V>();
                }else{
                    DataCopy(computegradInputLocal[address], computegradOutputLocal[address], d_size);
                }
            }
        }
        if (has_scale_){
            Add(computegradScaleLocal_, computegradScaleLocal_, tileGradScale, d_size);
            PipeBarrier<PIPE_V>();
        }
        if (has_shift_){
            Add(computegradShiftLocal_, computegradShiftLocal_, tileGradShift, d_size);
            PipeBarrier<PIPE_V>();
        }
        gradInputQueue_.EnQue(computegradInputLocal);
        gradScaleLocalQueue_.EnQue(computegradScaleLocal_);
        gradShiftLocalQueue_.EnQue(computegradShiftLocal_);
        tileGradShiftQueue_.FreeTensor(tileGradShift);
        tileGradScaleQueue_.FreeTensor(tileGradScale);
        currenttileGradScaleQueue_.FreeTensor(currenttileGradScale);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyOutGradInput(uint32_t allOffset,uint32_t LenD,uint32_t lenD){
        LocalTensor<T> gradInputLocalout = gradInputQueue_.DeQue<float>();

        DataCopyExtParams inputParams{
            static_cast<uint16_t>(currentL_),
            static_cast<uint32_t>(rowBytes),
            0,
            static_cast<uint32_t>((LenD - lenD)*sizeof(T)),
            0
        };
        DataCopyPad(gradInputGm_[allOffset], gradInputLocalout[0], inputParams);
        gradInputQueue_.FreeTensor(gradInputLocalout);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyOutScaleShift(uint32_t ssOffset, uint32_t lenD){
        if (has_scale_) {
                LocalTensor<T> gradScaleLocalout = gradScaleLocalQueue_.DeQue<float>();
                DataCopyExtParams scaleParams{1, static_cast<uint32_t>(rowBytes), 0, 0, 0};
                DataCopyPad(gradScaleGm_[ssOffset], gradScaleLocalout[0], scaleParams);
                gradScaleLocalQueue_.FreeTensor(gradScaleLocalout);
                PipeBarrier<PIPE_ALL>();
            }else{
                LocalTensor<T> gradScaleLocalout = gradScaleLocalQueue_.DeQue<float>();
                gradScaleLocalQueue_.FreeTensor(gradScaleLocalout);
                PipeBarrier<PIPE_ALL>();
            }
        if (has_shift_){
                LocalTensor<T> gradShiftLocalout = gradShiftLocalQueue_.DeQue<float>();
                DataCopyExtParams shiftParams{1, static_cast<uint32_t>(rowBytes), 0, 0, 0};
                DataCopyPad(gradShiftGm_[ssOffset], gradShiftLocalout[0], shiftParams);
                gradShiftLocalQueue_.FreeTensor(gradShiftLocalout);
                PipeBarrier<PIPE_ALL>();
            }else{
                LocalTensor<T> gradShiftLocalout = gradShiftLocalQueue_.DeQue<float>();
                gradShiftLocalQueue_.FreeTensor(gradShiftLocalout);
                PipeBarrier<PIPE_ALL>();
            }
    }
};
}

extern "C" __global__ __aicore__ void modulate_grad(
    GM_ADDR grad_Output, GM_ADDR Input, GM_ADDR Scale,
    GM_ADDR Shift, GM_ADDR grad_input, GM_ADDR grad_scale,
    GM_ADDR grad_shift, GM_ADDR workspace, GM_ADDR tiling  
){

    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(0)){}

    using namespace AscendC;
    ModulateGradKernel<float> kernel;
    PipeBarrier<PIPE_ALL>();
    kernel.Init(
        tiling_data, Input, grad_Output,
        grad_input, Scale, grad_scale, grad_shift);
    PipeBarrier<PIPE_ALL>();
    kernel.Process();
}