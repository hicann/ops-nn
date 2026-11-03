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
 * \file modulate_grad.cpp
 * \brief modulate_grad kernel for arch35 (ascend950, regbase path)
 */

#include "kernel_operator.h"
#include <type_traits>
#include "modulate_grad_struct.h"
#include "modulate_grad_regbase_tiling_key.h"

namespace AscendC {
template <typename T, uint8_t tilingStrategy, bool isScale, bool isShift>
class ModulateGradKernel {
private:
    static constexpr uint32_t MAX_TILE_LENGTH = 12 * 1024 / sizeof(T);
    static constexpr uint32_t COMPUTE_CHUNK = 2048;
    // Fast tile path constants: one vector repeat covers 256B (64 fp32), and the
    // broadcast scale pattern is one full repeat (256B = M padded rows).
    static constexpr uint32_t FAST_REPEAT_ELEMS = 64;
    static constexpr uint32_t FAST_PATTERN_BYTES = 256;
    static constexpr uint32_t FAST_MAX_D = 16;

    // GlobalTensor
    GlobalTensor<T> inputGm_, gradOutputGm_, scaleGm_;
    GlobalTensor<T> gradInputGm_, gradScaleGm_, gradShiftGm_;

    // QUEUE
    TPipe pipe;

    TQue<TPosition::VECIN, 1> inputQueue_;
    TQue<TPosition::VECIN, 1> gradOutputQueue_;
    TQue<TPosition::VECIN, 1> scaleQueue_;

    TQue<TPosition::VECOUT, 1> gradInputQueue_;

    // fp32 accumulators and cast scratch (dtype-aware addressing with fp32 internal
    // compute for 2-byte dtypes, matching the documented upcast semantics)
    TBuf<TPosition::VECCALC> accScaleF_;
    TBuf<TPosition::VECCALC> accShiftF_;
    TBuf<TPosition::VECCALC> scaleF_;
    TBuf<TPosition::VECCALC> gradOutF_;
    TBuf<TPosition::VECCALC> inputF_;
    TBuf<TPosition::VECCALC> prodF_;
    TBuf<TPosition::VECCALC> castOutScaleT_;
    TBuf<TPosition::VECCALC> castOutShiftT_;

    // 分块参数
    uint64_t B_, L_, D_;
    // 有效策略与存在性以 tiling 运行时标志为准(模板参数用于变体选择/ABI 绑定;
    // 在线编译若误选变体,运行时标志仍保证正确路由与缺席参数不被读写)。
    bool splitByD_ = false;
    bool has_scale_ = false;
    bool has_shift_ = false;
    uint32_t coresPerB_, formerNum_, formerLength_;
    uint32_t tailNum_, tailLength_, maxRowsPerTile_;

    // 当前核心处理的相关参数
    uint32_t coreId_, b_index_, d_start_, totalD_, totalcutD_;
    uint32_t currentL_, currentD_, len_, num, cores_;
    uint32_t b_start, tileNumB_, rowBytes, alignedRowBytes;
    uint8_t rowPadElements;
    bool isDOverflow;
    // Small-d tall-thin tiles (rows*64 elems fit few vector repeats) process the
    // whole L-tile flat instead of one Kahan chain per row.
    bool fastTilePath_ = false;
    // D==1 tall-thin: the per-core L range is contiguous in GM, so copies go as
    // one flat block per tile and reductions use the hardware ReduceSum.
    bool flatOneD_ = false;
    float flatScale0_ = 0.0f;
    uint32_t tileLIndex = 0;
    uint32_t tileDIndex = 0;
    uint32_t tileBIndex = 0;
    // D分块参数
    uint32_t tileNumD_;
    uint32_t lastTileLengthD_;
    uint32_t d_size;
    // L分行参数
    uint32_t tileNumL_;
    uint32_t lastTileLengthL_;

public:
    __aicore__ inline ModulateGradKernel() {}

    __aicore__ inline void Init(const ModulateGradRegbaseTilingData& tiling, GM_ADDR inputGm, GM_ADDR gradOutputGm,
                                GM_ADDR gradInputGm, GM_ADDR scaleGm, GM_ADDR gradScaleGm, GM_ADDR gradShiftGm)
    {
        InitTilingParams(tiling);
        InitGlobalTensors(inputGm, gradOutputGm, gradInputGm, scaleGm, gradScaleGm, gradShiftGm);
        InitCoreSpecificParams(tiling);
        InitTileParams();
        InitPipeBuffers();
    }

    __aicore__ inline void CopyGradInputFlat()
    {
        uint64_t total = B_ * L_ * D_;
        if (total == 0) {
            return;
        }
        const uint32_t alignElems = 32u / static_cast<uint32_t>(sizeof(T));
        uint64_t slab = (total + cores_ - 1) / cores_;
        slab = ((slab + alignElems - 1) / alignElems) * alignElems;
        uint64_t start = static_cast<uint64_t>(coreId_) * slab;
        if (start >= total) {
            return;
        }
        uint64_t end = (start + slab < total) ? (start + slab) : total;
        uint64_t count = end - start;
        // The queue buffers are sized for the row-pitch layout, which shrinks to a
        // single row when L==1; chunk by the real buffer capacity, not MAX_TILE_LENGTH.
        const uint32_t chunk = len_ / static_cast<uint32_t>(sizeof(T));
        for (uint64_t off = 0; off < count; off += chunk) {
            uint64_t rem = count - off;
            uint32_t n = (rem > chunk) ? chunk : static_cast<uint32_t>(rem);
            uint32_t nBytes = n * static_cast<uint32_t>(sizeof(T));
            uint32_t padElems = (alignElems - (n % alignElems)) % alignElems;
            uint32_t nPad = n + padElems;
            LocalTensor<T> gradOutputLocal = gradOutputQueue_.AllocTensor<T>();
            DataCopyExtParams inParams{1, nBytes, 0, 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(padElems), static_cast<T>(0)};
            DataCopyPad(gradOutputLocal[0], gradOutputGm_[start + off], inParams, padParams);
            gradOutputQueue_.EnQue(gradOutputLocal);
            PipeBarrier<PIPE_ALL>();
            LocalTensor<T> srcLocal = gradOutputQueue_.DeQue<T>();
            LocalTensor<T> gradInputLocal = gradInputQueue_.AllocTensor<T>();
            DataCopy(gradInputLocal[0], srcLocal[0], nPad);
            gradInputQueue_.EnQue(gradInputLocal);
            PipeBarrier<PIPE_ALL>();
            gradOutputQueue_.FreeTensor(srcLocal);
            LocalTensor<T> outLocal = gradInputQueue_.DeQue<T>();
            DataCopyExtParams outParams{1, nBytes, 0, 0, 0};
            DataCopyPad(gradInputGm_[start + off], outLocal[0], outParams);
            gradInputQueue_.FreeTensor(outLocal);
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void CopyIn(uint32_t tileLIndex)
    {
        PipeBarrier<PIPE_ALL>();
        uint64_t allOffset, scaleOffset;
        uint32_t LenD, lenD;
        CalculateOffsets(tileLIndex, allOffset, scaleOffset, LenD, lenD);
        PipeBarrier<PIPE_ALL>();
        CopyInInput(allOffset, LenD, lenD);
        PipeBarrier<PIPE_ALL>();
        CopyInGradOutput(allOffset, LenD, lenD);
        PipeBarrier<PIPE_ALL>();
        if (has_scale_) {
            CopyInScale(scaleOffset, lenD);
        }
    }

    __aicore__ inline void Compute(uint32_t tileLIndex)
    {
        LocalTensor<T> computeinputLocal = inputQueue_.DeQue<T>();
        LocalTensor<T> computegradOutputLocal = gradOutputQueue_.DeQue<T>();
        LocalTensor<T> computescaleLocal;
        if (has_scale_) {
            computescaleLocal = scaleQueue_.DeQue<T>();
        }
        LocalTensor<T> computegradInputLocal = gradInputQueue_.AllocTensor<T>();
        computegradInputLocal.SetSize(currentL_ * currentD_);

        ComputeGradients(computeinputLocal, computegradOutputLocal, computescaleLocal, computegradInputLocal);
        PipeBarrier<PIPE_ALL>();
        inputQueue_.FreeTensor(computeinputLocal);
        gradOutputQueue_.FreeTensor(computegradOutputLocal);
        if (has_scale_) {
            scaleQueue_.FreeTensor(computescaleLocal);
        }
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void CopyOut(uint32_t tileIndex)
    {
        uint64_t allOffset, scaleOffset;
        uint32_t LenD, lenD;
        CalculateOffsets(tileLIndex, allOffset, scaleOffset, LenD, lenD);
        PipeBarrier<PIPE_ALL>();
        CopyOutGradInput(allOffset, LenD, lenD);
        PipeBarrier<PIPE_ALL>();
        if (tileLIndex == tileNumL_ - 1 && (has_scale_ || has_shift_)) {
            CopyOutScaleShift(scaleOffset);
        }
    }

    __aicore__ inline void Process()
    {
        if (coreId_ >= cores_)
            return;
        if (L_ == 0) {
            // L==0 leaves grad_scale/grad_shift ([B,D] sums over an empty L) defined as
            // zeros; grad_input ([B,0,D]) is empty and needs no write.
            if (B_ > 0 && D_ > 0 && coreId_ == 0) {
                ZeroFillScaleShift();
            }
            return;
        }
        if (!has_scale_ && !has_shift_) {
            // grad_input equals grad_output element-wise: reinterpret the launched
            // cores as flat contiguous slabs over B*L*D instead of the (b, d) row
            // split, so each core DMA-carries one aligned chunk (2-16B strided rows
            // would cost one DMA block per row). The input tensor needs no load.
            // Small copies stay on the row loops: the flat path's queue round-trips
            // cost ~3us fixed, which loses below ~512 elements per core.
            uint64_t totalElems = B_ * L_ * D_;
            uint64_t slab = (totalElems + cores_ - 1) / cores_;
            if (slab >= 512u) {
                CopyGradInputFlat();
                return;
            }
        }
        for (tileBIndex = 0; tileBIndex < tileNumB_; ++tileBIndex) {
            for (tileDIndex = 0; tileDIndex < tileNumD_; ++tileDIndex) {
                UpdatetileParamsDsize();
                for (tileLIndex = 0; tileLIndex < tileNumL_; ++tileLIndex) {
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
    __aicore__ inline void InitGlobalTensors(GM_ADDR inputGm, GM_ADDR gradOutputGm, GM_ADDR gradInputGm,
                                             GM_ADDR scaleGm, GM_ADDR gradScaleGm, GM_ADDR gradShiftGm)
    {
        inputGm_.SetGlobalBuffer((__gm__ T*)inputGm);
        gradOutputGm_.SetGlobalBuffer((__gm__ T*)gradOutputGm);
        if (has_scale_) {
            scaleGm_.SetGlobalBuffer((__gm__ T*)scaleGm);
        }
        gradInputGm_.SetGlobalBuffer((__gm__ T*)gradInputGm);
        if (has_scale_) {
            gradScaleGm_.SetGlobalBuffer((__gm__ T*)gradScaleGm);
        }
        if (has_shift_) {
            gradShiftGm_.SetGlobalBuffer((__gm__ T*)gradShiftGm);
        }
    }

    __aicore__ inline void UpdatetileParamsDsize()
    {
        if (isDOverflow) {
            d_size = (tileDIndex == tileNumD_ - 1) ? lastTileLengthD_ : totalcutD_;
            rowBytes = d_size * sizeof(T);
            uint32_t rowPadBytes = (32 - (rowBytes % 32)) % 32;
            alignedRowBytes = rowBytes + rowPadBytes;
            currentD_ = alignedRowBytes / sizeof(T);
            rowPadElements = rowPadBytes / sizeof(T);
            len_ = currentL_ * alignedRowBytes;
        }
    }
    __aicore__ inline void InitTilingParams(const ModulateGradRegbaseTilingData& tiling)
    {
        B_ = tiling.B;
        L_ = tiling.L;
        D_ = tiling.D;
        splitByD_ = tiling.split_by_d != 0;
        has_scale_ = tiling.has_scale != 0;
        has_shift_ = tiling.has_shift != 0;

        coresPerB_ = tiling.coresPerB;
        formerNum_ = tiling.formerNum;
        formerLength_ = tiling.formerLength;
        tailNum_ = tiling.tailNum;
        tailLength_ = tiling.tailLength;
    }
    __aicore__ inline void InitCoreSpecificParams(const ModulateGradRegbaseTilingData& tiling)
    {
        coreId_ = GetBlockIdx();
        if (splitByD_) {
            b_index_ = coreId_ / tiling.coresPerB;
            uint32_t groupId = coreId_ % tiling.coresPerB;
            tileNumB_ = 1;

            if (b_index_ >= tiling.usedCores) {
                totalD_ = 0;
                tileNumL_ = 0;
                return;
            }

            if (groupId < formerNum_) {
                d_start_ = groupId * formerLength_;
                totalD_ = formerLength_;
            } else {
                d_start_ = formerNum_ * formerLength_ + (groupId - formerNum_) * tailLength_;
                totalD_ = tailLength_;
            }
            cores_ = tiling.usedCores;
        } else {
            d_start_ = 0;
            if (coreId_ < formerNum_) {
                b_index_ = coreId_ * formerLength_;
                tileNumB_ = formerLength_;
            } else {
                b_index_ = formerNum_ * formerLength_ + (coreId_ - formerNum_) * tailLength_;
                tileNumB_ = tailLength_;
            }
            totalD_ = D_;
            cores_ = tiling.usedCores;
        }
    }

    __aicore__ inline void InitTileParams()
    {
        if (totalD_ == 0) {
            // No feature elements for this core (D==0 or an empty D-split share);
            // keep every derived parameter safe and skip all tile loops.
            isDOverflow = false;
            totalcutD_ = 0;
            tileNumD_ = 0;
            tileNumL_ = 0;
            d_size = 16;
            rowBytes = d_size * sizeof(T);
            uint32_t rowPadBytes = (32 - (rowBytes % 32)) % 32;
            alignedRowBytes = rowBytes + rowPadBytes;
            currentD_ = alignedRowBytes / sizeof(T);
            rowPadElements = rowPadBytes / sizeof(T);
            maxRowsPerTile_ = 1;
            currentL_ = 1;
            len_ = alignedRowBytes;
            return;
        }
        maxRowsPerTile_ = MAX_TILE_LENGTH / totalD_;
        isDOverflow = (maxRowsPerTile_ == 0);

        if (isDOverflow) {
            totalcutD_ = MAX_TILE_LENGTH;
            uint32_t standard_rowBytes = totalcutD_ * sizeof(T);
            uint32_t standard_rowPadBytes = (32 - (standard_rowBytes % 32)) % 32;
            uint32_t standard_alignedRowBytes = standard_rowBytes + standard_rowPadBytes;
            uint32_t standard_currentD = standard_alignedRowBytes / sizeof(T);

            tileNumD_ = (totalD_ + standard_currentD - 1) / standard_currentD;
            lastTileLengthD_ = totalD_ - (tileNumD_ - 1) * standard_currentD;

            d_size = (tileDIndex == tileNumD_ - 1) ? lastTileLengthD_ : totalcutD_;
            rowBytes = d_size * sizeof(T);
            uint32_t rowPadBytes = (32 - (rowBytes % 32)) % 32;
            alignedRowBytes = rowBytes + rowPadBytes;
            currentD_ = alignedRowBytes / sizeof(T);
            rowPadElements = rowPadBytes / sizeof(T);
            maxRowsPerTile_ = 1;
            currentL_ = 1;
            tileNumL_ = L_;
            len_ = currentL_ * alignedRowBytes;
        } else {
            d_size = totalD_;
            rowBytes = d_size * sizeof(T);
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
        // d_size is the per-core D share here; small rows keep the per-row Kahan
        // chain (15 vector ops + 8 barriers per row) from filling a single repeat,
        // so such tiles switch to flat whole-tile compute (see ComputeGradientsTileFlat).
        fastTilePath_ = !isDOverflow && d_size <= FAST_MAX_D;
        // The flat CopyIn/Out pair belongs to the compute path only: the pure-copy
        // fallback bulk-copies the row-pitched tile and would read the flat layout
        // past its valid elements.
        flatOneD_ = fastTilePath_ && d_size == 1 && D_ == 1 && (has_scale_ || has_shift_);
    }

    __aicore__ inline void InitPipeBuffers()
    {
        // flatOneD computes on ceil64(currentL_) elements of the fp32 queue tiles,
        // which exceeds the single-row len_ (L==1) layout; raise the floor.
        if (flatOneD_) {
            uint32_t flatBytes = ((maxRowsPerTile_ + 63u) & ~63u) * sizeof(float);
            if (len_ < flatBytes) {
                len_ = flatBytes;
            }
        }
        pipe.InitBuffer(inputQueue_, 1, len_);
        pipe.InitBuffer(gradOutputQueue_, 1, len_);
        pipe.InitBuffer(scaleQueue_, 1, alignedRowBytes);
        pipe.InitBuffer(gradInputQueue_, 1, len_);
        // TBuf allocations are kept at 32-byte granularity to avoid sub-block
        // placement hazards for tiny rows.
        uint32_t accBytes = (d_size * sizeof(float) + 31u) & ~31u;
        uint32_t chunkBytes = (COMPUTE_CHUNK * sizeof(float) + 31u) & ~31u;
        uint32_t castElems = (d_size > COMPUTE_CHUNK) ? d_size : COMPUTE_CHUNK;
        uint32_t castBytes = (castElems * sizeof(T) + 31u) & ~31u;
        uint32_t compensationBytes = (d_size * sizeof(float) + 255u) & ~255u;
        uint32_t reductionBytes = (compensationBytes > castBytes) ? compensationBytes : castBytes;
        pipe.InitBuffer(accScaleF_, accBytes);
        pipe.InitBuffer(accShiftF_, accBytes);
        // The output-cast buffers hold the fp32 Kahan compensation while reducing.
        // They are reused for the final dtype cast after the compensation is no longer needed.
        pipe.InitBuffer(castOutScaleT_, reductionBytes);
        pipe.InitBuffer(castOutShiftT_, reductionBytes);
        if (!std::is_same<T, float>::value) {
            // Fast tile path upcasts the whole L-tile (rows*currentD_ <= MAX_TILE_LENGTH
            // floats) at once, so the cast scratch grows from one COMPUTE_CHUNK.
            uint32_t fastBytes = fastTilePath_ ? (MAX_TILE_LENGTH + 64u) * sizeof(float) : 0u;
            uint32_t vecBytes = (chunkBytes > fastBytes) ? chunkBytes : fastBytes;
            pipe.InitBuffer(scaleF_, chunkBytes);
            pipe.InitBuffer(gradOutF_, vecBytes);
            pipe.InitBuffer(inputF_, vecBytes);
            pipe.InitBuffer(prodF_, vecBytes);
        } else {
            // 256-byte (64-float) granularity matches the compensation buffers, so the
            // ceil64(n) NaN-clear scratch access never crosses the allocation end.
            uint32_t scratchBytes = (d_size * sizeof(float) + 255u) & ~255u;
            // Fast tile path: prod holds one whole fp32 L-tile; scaleF holds the
            // 256B broadcast pattern; gradOutF/inputF stay Kahan-temp scratch.
            uint32_t fastBytes = fastTilePath_ ? (MAX_TILE_LENGTH + 64u) * sizeof(float) : 0u;
            uint32_t prodBytes = (scratchBytes > fastBytes) ? scratchBytes : fastBytes;
            pipe.InitBuffer(prodF_, prodBytes);
            pipe.InitBuffer(gradOutF_, scratchBytes);
            pipe.InitBuffer(inputF_, scratchBytes);
            if (fastTilePath_) {
                pipe.InitBuffer(scaleF_, chunkBytes);
            }
        }
    }

    __aicore__ inline void CalculateOffsets(uint32_t tileLIndex, uint64_t& allOffset, uint64_t& scaleOffset,
                                            uint32_t& LenD, uint32_t& lenD)
    {
        LenD = D_;
        if (isDOverflow) {
            lenD = (tileDIndex == tileNumD_ - 1) ? lastTileLengthD_ : MAX_TILE_LENGTH;
            uint32_t d_block_offset;
            if (tileDIndex == 0) {
                d_block_offset = 0;
            } else {
                d_block_offset = tileDIndex * MAX_TILE_LENGTH;
                if (d_block_offset + lenD > totalD_) {
                    d_block_offset = totalD_ - lenD;
                }
            }
            uint64_t offset = tileLIndex;
            allOffset = (b_index_ + tileBIndex) * L_ * D_ + offset * D_ + d_start_ + d_block_offset;
            scaleOffset = (b_index_ + tileBIndex) * D_ + d_start_ + d_block_offset;
        } else {
            lenD = totalD_;
            uint64_t offset;
            if (tileLIndex == tileNumL_ - 1) {
                currentL_ = lastTileLengthL_;
                offset = L_ - lastTileLengthL_;
            } else {
                currentL_ = maxRowsPerTile_;
                offset = tileLIndex * maxRowsPerTile_;
            }
            allOffset = (b_index_ + tileBIndex) * L_ * D_ + offset * D_ + d_start_;
            scaleOffset = (b_index_ + tileBIndex) * D_ + d_start_;
        }
    }

    __aicore__ inline void CopyInInput(uint64_t allOffset, uint32_t LenD, uint32_t lenD)
    {
        LocalTensor<T> inputLocal = inputQueue_.AllocTensor<T>();

        if (flatOneD_) {
            // D==1: the tile is one contiguous GM range; a single padded block
            // replaces one 2-byte DMA block per row.
            uint32_t nBytes = currentL_ * static_cast<uint32_t>(sizeof(T));
            uint32_t alignElems = 32u / static_cast<uint32_t>(sizeof(T));
            uint32_t padElems = (alignElems - (currentL_ % alignElems)) % alignElems;
            DataCopyExtParams flatParams{1, nBytes, 0, 0, 0};
            DataCopyPadExtParams<T> flatPad{true, 0, static_cast<uint8_t>(padElems), static_cast<T>(0)};
            DataCopyPad(inputLocal[0], inputGm_[allOffset], flatParams, flatPad);
            inputQueue_.EnQue(inputLocal);
            PipeBarrier<PIPE_ALL>();
            return;
        }
        DataCopyExtParams inputCopyParams{static_cast<uint16_t>(currentL_), static_cast<uint32_t>(rowBytes),
                                          static_cast<uint32_t>((LenD - lenD) * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> inputpadParams{true, 0, rowPadElements, static_cast<T>(0)};

        DataCopyPad(inputLocal[0], inputGm_[allOffset], inputCopyParams, inputpadParams);
        inputQueue_.EnQue(inputLocal);
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void CopyInGradOutput(uint64_t allOffset, uint32_t LenD, uint32_t lenD)
    {
        LocalTensor<T> gradOutputLocal = gradOutputQueue_.AllocTensor<T>();

        if (flatOneD_) {
            uint32_t nBytes = currentL_ * static_cast<uint32_t>(sizeof(T));
            uint32_t alignElems = 32u / static_cast<uint32_t>(sizeof(T));
            uint32_t padElems = (alignElems - (currentL_ % alignElems)) % alignElems;
            DataCopyExtParams flatParams{1, nBytes, 0, 0, 0};
            DataCopyPadExtParams<T> flatPad{true, 0, static_cast<uint8_t>(padElems), static_cast<T>(0)};
            DataCopyPad(gradOutputLocal[0], gradOutputGm_[allOffset], flatParams, flatPad);
            gradOutputQueue_.EnQue(gradOutputLocal);
            PipeBarrier<PIPE_ALL>();
            return;
        }
        DataCopyExtParams gradOutputCopyParams{static_cast<uint16_t>(currentL_), static_cast<uint32_t>(rowBytes),
                                               static_cast<uint32_t>((LenD - lenD) * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> gradOutputCopypadParams{true, 0, rowPadElements, static_cast<T>(0)};

        DataCopyPad(gradOutputLocal[0], gradOutputGm_[allOffset], gradOutputCopyParams, gradOutputCopypadParams);
        gradOutputQueue_.EnQue(gradOutputLocal);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyInScale(uint64_t scaleOffset, uint32_t lenD)
    {
        LocalTensor<T> scaleLocal = scaleQueue_.AllocTensor<T>();
        if (has_scale_) {
            DataCopyExtParams scaleParams{static_cast<uint16_t>(1), static_cast<uint32_t>(rowBytes), 0, 0, 0};
            DataCopyPadExtParams<T> scalePadParams{true, 0, rowPadElements, static_cast<T>(0)};
            DataCopyPad(scaleLocal, scaleGm_[scaleOffset], scaleParams, scalePadParams);
        }
        scaleQueue_.EnQue(scaleLocal);
        PipeBarrier<PIPE_ALL>();
    }

    // value and temp are scratch tensors. compensation persists across all L tiles.
    __aicore__ inline void CompensatedAdd(LocalTensor<float> acc, LocalTensor<float> compensation,
                                          LocalTensor<float> value, LocalTensor<float> temp, uint32_t n)
    {
        // Kahan: y = value - compensation; t = acc + y;
        // compensation = (t - acc) - y; acc = t.
        Sub(value, value, compensation, n);
        PipeBarrier<PIPE_V>();
        Add(temp, acc, value, n);
        PipeBarrier<PIPE_V>();
        Sub(compensation, temp, acc, n);
        PipeBarrier<PIPE_V>();
        Sub(compensation, compensation, value, n);
        PipeBarrier<PIPE_V>();
        Adds(acc, temp, 0.0f, n);
        PipeBarrier<PIPE_V>();
        // Kahan's correction becomes NaN for an infinite running sum. Clear only
        // that correction so Inf remains Inf while a NaN running sum still propagates.
        // Applies to every fp32 accumulation path (float and upcast 2-byte dtypes):
        // value/temp hold >= ceil64(n) floats (COMPUTE_CHUNK-sized for 2-byte dtypes,
        // d_size-sized for float), and compensation is ceil256(d_size*4) >= ceil64(n).
        uint32_t alignedN = (n + 63u) & ~63u;
        LocalTensor<uint8_t> nanMask = value.template ReinterpretCast<uint8_t>();
        Duplicate(temp, 0.0f, alignedN);
        Compare(nanMask, compensation, compensation, CMPMODE::NE, alignedN);
        PipeBarrier<PIPE_V>();
        Select(compensation, nanMask, temp, compensation, SELMODE::VSEL_TENSOR_TENSOR_MODE, alignedN);
        PipeBarrier<PIPE_V>();
    }

    // Fold `rows` rows (row pitch `pitch` floats) of tile `t` into row 0 with one
    // vector Add per level. In-place adds are only safe when dst spans src1
    // element-aligned (dst==src0): an offset overlap with src2 is not guaranteed
    // to read before write, so the odd trailing row gets a separate fold add
    // after the barrier instead of extending the pair add past row `half`.
    __aicore__ inline void FlatRowTreeReduce(LocalTensor<float> t, uint32_t rows, uint32_t pitch)
    {
        while (rows > 1) {
            uint32_t half = rows / 2;
            Add(t, t, t[half * pitch], static_cast<int32_t>(half * pitch));
            PipeBarrier<PIPE_V>();
            if ((rows & 1u) != 0) {
                Add(t[(half - 1u) * pitch], t[(half - 1u) * pitch], t[(rows - 1u) * pitch],
                    static_cast<int32_t>(pitch));
                PipeBarrier<PIPE_V>();
            }
            rows = half;
        }
    }

    // dst = src0 * scaleRow, scale broadcast per padded row: each 256B vector
    // repeat re-reads the same 256B pattern (an integer number of padded rows)
    // through src1RepStride = 0; a masked tail Mul handles the remainder.
    __aicore__ inline void BroadcastMulScale(LocalTensor<float> dst, LocalTensor<float> src0,
                                             LocalTensor<float> pattern, uint32_t total)
    {
        BinaryRepeatParams brcParams(1, 1, 1, 8, 8, 0);
        uint32_t full = total / FAST_REPEAT_ELEMS;
        uint32_t tail = total - full * FAST_REPEAT_ELEMS;
        if (full > 0) {
            Mul(dst, src0, pattern, static_cast<uint64_t>(FAST_REPEAT_ELEMS), static_cast<uint8_t>(full), brcParams);
        }
        if (tail > 0) {
            Mul(dst[full * FAST_REPEAT_ELEMS], src0[full * FAST_REPEAT_ELEMS], pattern, static_cast<int32_t>(tail));
        }
    }

    // Tall-thin fast path (d_size <= 16, no D-overflow): process the whole L-tile
    // flat instead of running one Kahan chain per row. The per-row chain costs
    // ~15 vector ops + 8 barriers per row; here the tile work collapses to a few
    // full-width casts/muls, the row sums come from a log2(rows) add tree, and
    // the Kahan accumulation runs once per tile on the reduced row.
    __aicore__ inline void ComputeGradientsTileFlat(LocalTensor<T>& computeinputLocal,
                                                    LocalTensor<T>& computegradOutputLocal,
                                                    LocalTensor<T>& computescaleLocal,
                                                    LocalTensor<T>& computegradInputLocal)
    {
        const uint32_t total = currentL_ * currentD_;
        LocalTensor<float> accScale = accScaleF_.Get<float>();
        LocalTensor<float> accShift = accShiftF_.Get<float>();
        LocalTensor<float> compScale = castOutScaleT_.Get<float>();
        LocalTensor<float> compShift = castOutShiftT_.Get<float>();
        if (tileLIndex == 0) {
            for (uint32_t off = 0; off < d_size; off += COMPUTE_CHUNK) {
                uint32_t n = (d_size - off < COMPUTE_CHUNK) ? (d_size - off) : COMPUTE_CHUNK;
                Duplicate<float>(accScale[off], 0.0f, n);
                Duplicate<float>(accShift[off], 0.0f, n);
                Duplicate<float>(compScale[off], 0.0f, n);
                Duplicate<float>(compShift[off], 0.0f, n);
                PipeBarrier<PIPE_V>();
            }
        }
        // 2-byte dtypes: gradOutF/inputF hold the whole upcast tile.
        // float: the queue tiles are the fp32 compute tiles; gradOutF/inputF stay
        // small Kahan-temp scratch (256B each).
        LocalTensor<float> gradOutF = gradOutF_.Get<float>();
        LocalTensor<float> inputF = inputF_.Get<float>();
        LocalTensor<float> prod = prodF_.Get<float>();
        LocalTensor<float> scaleF = scaleF_.Get<float>();
        if (flatOneD_) {
            // D==1: the tile is one contiguous vector of currentL_ valid elements.
            // Zero-fill to a 64-float boundary so the flat Mul and ReduceSum stay
            // aligned, and read the single scale value once per (b, d-tile).
            const uint32_t n = currentL_;
            const uint32_t nPad = (n + 63u) & ~63u;
            const uint32_t alignElems = 32u / static_cast<uint32_t>(sizeof(T));
            const uint32_t n32 = (n + alignElems - 1u) & ~(alignElems - 1u);
            if (!has_scale_) {
                DataCopy(computegradInputLocal[0], computegradOutputLocal[0], n32);
                if constexpr (std::is_same<T, float>::value) {
                    DataCopy(prod, computegradOutputLocal[0], n32);
                    if (nPad > n32) {
                        Duplicate(prod[n32], 0.0f, nPad - n32);
                    }
                    PipeBarrier<PIPE_ALL>();
                    ReduceSum(prod, prod, prod, nPad);
                    PipeBarrier<PIPE_V>();
                    CompensatedAdd(accShift, compShift, prod, gradOutF, 1);
                } else {
                    // Cast up to a 32B-aligned element boundary (the source tile's
                    // zero pads cover [n, n8)), then zero-fill from that aligned
                    // address: vector ops fault on unaligned element offsets.
                    const uint32_t n8 = (n + 7u) & ~7u;
                    Cast(gradOutF, computegradOutputLocal[0], RoundMode::CAST_NONE, static_cast<int32_t>(n8));
                    if (nPad > n8) {
                        Duplicate(gradOutF[n8], 0.0f, nPad - n8);
                    }
                    PipeBarrier<PIPE_V>();
                    ReduceSum(gradOutF, gradOutF, gradOutF, nPad);
                    PipeBarrier<PIPE_V>();
                    CompensatedAdd(accShift, compShift, gradOutF, inputF, 1);
                }
                gradInputQueue_.EnQue(computegradInputLocal);
                return;
            }
            if constexpr (std::is_same<T, float>::value) {
                if (nPad > n32) {
                    Duplicate(computegradOutputLocal[n32], 0.0f, nPad - n32);
                    Duplicate(computeinputLocal[n32], 0.0f, nPad - n32);
                }
                PipeBarrier<PIPE_V>();
                Mul(prod, computeinputLocal[0], computegradOutputLocal[0], static_cast<int32_t>(nPad));
                PipeBarrier<PIPE_V>();
                if (tileLIndex == 0) {
                    SetFlag<HardEvent::V_S>(EVENT_ID0);
                    WaitFlag<HardEvent::V_S>(EVENT_ID0);
                    flatScale0_ = computescaleLocal.GetValue(0);
                    SetFlag<HardEvent::S_V>(EVENT_ID0);
                    WaitFlag<HardEvent::S_V>(EVENT_ID0);
                }
                Muls(computegradInputLocal[0], computegradOutputLocal[0], flatScale0_, static_cast<int32_t>(nPad));
                PipeBarrier<PIPE_V>();
                ReduceSum(prod, prod, prod, nPad);
                PipeBarrier<PIPE_V>();
                CompensatedAdd(accScale, compScale, prod, gradOutF, 1);
                if (has_shift_) {
                    ReduceSum(computegradOutputLocal, computegradOutputLocal, computegradOutputLocal, nPad);
                    PipeBarrier<PIPE_V>();
                    CompensatedAdd(accShift, compShift, computegradOutputLocal, inputF, 1);
                }
            } else {
                const uint32_t n8 = (n + 7u) & ~7u;
                Cast(gradOutF, computegradOutputLocal[0], RoundMode::CAST_NONE, static_cast<int32_t>(n8));
                Cast(inputF, computeinputLocal[0], RoundMode::CAST_NONE, static_cast<int32_t>(n8));
                if (nPad > n8) {
                    Duplicate(gradOutF[n8], 0.0f, nPad - n8);
                    Duplicate(inputF[n8], 0.0f, nPad - n8);
                }
                PipeBarrier<PIPE_V>();
                Mul(prod, inputF, gradOutF, static_cast<int32_t>(nPad));
                PipeBarrier<PIPE_V>();
                if (tileLIndex == 0) {
                    Cast(scaleF, computescaleLocal[0], RoundMode::CAST_NONE, 1);
                    PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_S>(EVENT_ID0);
                    WaitFlag<HardEvent::V_S>(EVENT_ID0);
                    flatScale0_ = scaleF.GetValue(0);
                    SetFlag<HardEvent::S_V>(EVENT_ID0);
                    WaitFlag<HardEvent::S_V>(EVENT_ID0);
                }
                Muls(inputF, gradOutF, flatScale0_, static_cast<int32_t>(nPad));
                PipeBarrier<PIPE_V>();
                Cast(computegradInputLocal[0], inputF, RoundMode::CAST_RINT, static_cast<int32_t>(n));
                PipeBarrier<PIPE_V>();
                ReduceSum(prod, prod, prod, nPad);
                PipeBarrier<PIPE_V>();
                CompensatedAdd(accScale, compScale, prod, inputF, 1);
                if (has_shift_) {
                    ReduceSum(gradOutF, gradOutF, gradOutF, nPad);
                    PipeBarrier<PIPE_V>();
                    CompensatedAdd(accShift, compShift, gradOutF, inputF, 1);
                }
            }
            gradInputQueue_.EnQue(computegradInputLocal);
            return;
        }
        if (!has_scale_) {
            // grad_input equals grad_output; only grad_shift needs the reduction.
            DataCopy(computegradInputLocal[0], computegradOutputLocal[0], total);
            if constexpr (std::is_same<T, float>::value) {
                DataCopy(prod, computegradOutputLocal[0], total);
                PipeBarrier<PIPE_ALL>();
                FlatRowTreeReduce(prod, currentL_, currentD_);
                CompensatedAdd(accShift, compShift, prod, gradOutF, d_size);
            } else {
                Cast(gradOutF, computegradOutputLocal[0], RoundMode::CAST_NONE, static_cast<int32_t>(total));
                PipeBarrier<PIPE_V>();
                FlatRowTreeReduce(gradOutF, currentL_, currentD_);
                CompensatedAdd(accShift, compShift, gradOutF, inputF, d_size);
            }
            gradInputQueue_.EnQue(computegradInputLocal);
            return;
        }
        if (tileLIndex == 0) {
            // Build the 256B broadcast pattern once per (b, d-tile): the padded
            // scale row (zero row pad from CopyIn) repeated to one full repeat.
            if constexpr (std::is_same<T, float>::value) {
                DataCopy(scaleF, computescaleLocal[0], currentD_);
            } else {
                Cast(scaleF, computescaleLocal[0], RoundMode::CAST_NONE, currentD_);
            }
            PipeBarrier<PIPE_ALL>();
            uint32_t patRows = FAST_PATTERN_BYTES / (currentD_ * static_cast<uint32_t>(sizeof(float)));
            for (uint32_t r = 1; r < patRows; r *= 2) {
                DataCopy(scaleF[r * currentD_], scaleF, r * currentD_);
            }
            PipeBarrier<PIPE_ALL>();
        }
        if constexpr (std::is_same<T, float>::value) {
            Mul(prod, computeinputLocal[0], computegradOutputLocal[0], static_cast<int32_t>(total));
            PipeBarrier<PIPE_V>();
            BroadcastMulScale(computegradInputLocal[0], computegradOutputLocal[0], scaleF, total);
            PipeBarrier<PIPE_V>();
            FlatRowTreeReduce(prod, currentL_, currentD_);
            CompensatedAdd(accScale, compScale, prod, gradOutF, d_size);
            if (has_shift_) {
                FlatRowTreeReduce(computegradOutputLocal, currentL_, currentD_);
                CompensatedAdd(accShift, compShift, computegradOutputLocal, inputF, d_size);
            }
        } else {
            Cast(gradOutF, computegradOutputLocal[0], RoundMode::CAST_NONE, static_cast<int32_t>(total));
            Cast(inputF, computeinputLocal[0], RoundMode::CAST_NONE, static_cast<int32_t>(total));
            PipeBarrier<PIPE_V>();
            Mul(prod, inputF, gradOutF, static_cast<int32_t>(total));
            PipeBarrier<PIPE_V>();
            BroadcastMulScale(inputF, gradOutF, scaleF, total);
            PipeBarrier<PIPE_V>();
            Cast(computegradInputLocal[0], inputF, RoundMode::CAST_RINT, static_cast<int32_t>(total));
            PipeBarrier<PIPE_V>();
            FlatRowTreeReduce(prod, currentL_, currentD_);
            CompensatedAdd(accScale, compScale, prod, inputF, d_size);
            if (has_shift_) {
                FlatRowTreeReduce(gradOutF, currentL_, currentD_);
                CompensatedAdd(accShift, compShift, gradOutF, inputF, d_size);
            }
        }
        gradInputQueue_.EnQue(computegradInputLocal);
    }

    __aicore__ inline void ComputeGradients(LocalTensor<T>& computeinputLocal, LocalTensor<T>& computegradOutputLocal,
                                            LocalTensor<T>& computescaleLocal, LocalTensor<T>& computegradInputLocal)
    {
        if (!has_scale_ && !has_shift_) {
            DataCopy(computegradInputLocal[0], computegradOutputLocal[0], currentL_ * currentD_);
            gradInputQueue_.EnQue(computegradInputLocal);
            return;
        }
        if (fastTilePath_) {
            ComputeGradientsTileFlat(computeinputLocal, computegradOutputLocal, computescaleLocal,
                                     computegradInputLocal);
            return;
        }
        LocalTensor<float> accScale = accScaleF_.Get<float>();
        LocalTensor<float> accShift = accShiftF_.Get<float>();
        LocalTensor<float> compScale = castOutScaleT_.Get<float>();
        LocalTensor<float> compShift = castOutShiftT_.Get<float>();
        if (tileLIndex == 0) {
            for (uint32_t off = 0; off < d_size; off += COMPUTE_CHUNK) {
                uint32_t n = (d_size - off < COMPUTE_CHUNK) ? (d_size - off) : COMPUTE_CHUNK;
                Duplicate<float>(accScale[off], 0.0f, n);
                Duplicate<float>(accShift[off], 0.0f, n);
                Duplicate<float>(compScale[off], 0.0f, n);
                Duplicate<float>(compShift[off], 0.0f, n);
                PipeBarrier<PIPE_V>();
            }
        }
        if (!has_scale_) {
            // grad_input rows equal grad_output rows; copy the whole tile at once
            // (currentD_ is 32B-aligned in elements) and only accumulate grad_shift.
            DataCopy(computegradInputLocal[0], computegradOutputLocal[0], currentL_ * currentD_);
            for (uint32_t off = 0; off < d_size; off += COMPUTE_CHUNK) {
                uint32_t n = (d_size - off < COMPUTE_CHUNK) ? (d_size - off) : COMPUTE_CHUNK;
                LocalTensor<float> gradOutF = gradOutF_.Get<float>();
                LocalTensor<float> temp = inputF_.Get<float>();
                for (uint32_t l = 0; l < currentL_; ++l) {
                    uint32_t address = currentD_ * l + off;
                    if constexpr (std::is_same<T, float>::value) {
                        Adds(gradOutF, computegradOutputLocal[address], 0.0f, n);
                    } else {
                        Cast(gradOutF, computegradOutputLocal[address], RoundMode::CAST_NONE, n);
                    }
                    PipeBarrier<PIPE_V>();
                    CompensatedAdd(accShift[off], compShift[off], gradOutF, temp, n);
                }
            }
            gradInputQueue_.EnQue(computegradInputLocal);
            return;
        }
        LocalTensor<float> scaleF;
        if constexpr (!std::is_same<T, float>::value) {
            scaleF = scaleF_.Get<float>();
        }
        LocalTensor<float> gradOutF = gradOutF_.Get<float>();
        LocalTensor<float> inputF = inputF_.Get<float>();
        LocalTensor<float> prod = prodF_.Get<float>();
        for (uint32_t off = 0; off < d_size; off += COMPUTE_CHUNK) {
            uint32_t n = (d_size - off < COMPUTE_CHUNK) ? (d_size - off) : COMPUTE_CHUNK;
            if constexpr (!std::is_same<T, float>::value) {
                Cast(scaleF, computescaleLocal[off], RoundMode::CAST_NONE, n);
                PipeBarrier<PIPE_V>();
            }
            for (uint32_t l = 0; l < currentL_; ++l) {
                uint32_t address = currentD_ * l + off;
                if constexpr (std::is_same<T, float>::value) {
                    Mul(computegradInputLocal[address], computegradOutputLocal[address], computescaleLocal[off], n);
                    Mul(prod, computeinputLocal[address], computegradOutputLocal[address], n);
                } else {
                    Cast(gradOutF, computegradOutputLocal[address], RoundMode::CAST_NONE, n);
                    Cast(inputF, computeinputLocal[address], RoundMode::CAST_NONE, n);
                    PipeBarrier<PIPE_V>();
                    Mul(prod, gradOutF, scaleF, n);
                    PipeBarrier<PIPE_V>();
                    Cast(computegradInputLocal[address], prod, RoundMode::CAST_RINT, n);
                    PipeBarrier<PIPE_V>();
                    Mul(prod, inputF, gradOutF, n);
                }
                PipeBarrier<PIPE_V>();
                CompensatedAdd(accScale[off], compScale[off], prod, inputF, n);
                if (has_shift_) {
                    if constexpr (std::is_same<T, float>::value) {
                        Adds(gradOutF, computegradOutputLocal[address], 0.0f, n);
                        PipeBarrier<PIPE_V>();
                    }
                    CompensatedAdd(accShift[off], compShift[off], gradOutF, prod, n);
                }
            }
        }
        gradInputQueue_.EnQue(computegradInputLocal);
    }

    __aicore__ inline void ZeroFillScaleShift()
    {
        if (!has_scale_ && !has_shift_) {
            return;
        }
        uint64_t total = B_ * D_;
        uint32_t chunk = COMPUTE_CHUNK;
        LocalTensor<T> outScale = castOutScaleT_.Get<T>();
        LocalTensor<T> outShift = castOutShiftT_.Get<T>();
        for (uint64_t off = 0; off < total; off += chunk) {
            uint32_t n = (total - off < chunk) ? static_cast<uint32_t>(total - off) : chunk;
            DataCopyExtParams fillParams{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
            if (has_scale_) {
                Duplicate<T>(outScale, static_cast<T>(0), n);
                PipeBarrier<PIPE_V>();
                DataCopyPad(gradScaleGm_[off], outScale, fillParams);
            }
            if (has_shift_) {
                Duplicate<T>(outShift, static_cast<T>(0), n);
                PipeBarrier<PIPE_V>();
                DataCopyPad(gradShiftGm_[off], outShift, fillParams);
            }
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void CopyOutGradInput(uint64_t allOffset, uint32_t LenD, uint32_t lenD)
    {
        LocalTensor<T> gradInputLocalout = gradInputQueue_.DeQue<T>();

        if (flatOneD_) {
            DataCopyExtParams flatParams{1, currentL_ * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            DataCopyPad(gradInputGm_[allOffset], gradInputLocalout[0], flatParams);
            gradInputQueue_.FreeTensor(gradInputLocalout);
            PipeBarrier<PIPE_ALL>();
            return;
        }
        DataCopyExtParams inputParams{static_cast<uint16_t>(currentL_), static_cast<uint32_t>(rowBytes), 0,
                                      static_cast<uint32_t>((LenD - lenD) * sizeof(T)), 0};
        DataCopyPad(gradInputGm_[allOffset], gradInputLocalout[0], inputParams);
        gradInputQueue_.FreeTensor(gradInputLocalout);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyOutScaleShift(uint64_t ssOffset)
    {
        if (has_scale_) {
            DataCopyExtParams scaleParams{1, static_cast<uint32_t>(rowBytes), 0, 0, 0};
            if constexpr (std::is_same<T, float>::value) {
                DataCopyPad(gradScaleGm_[ssOffset], accScaleF_.Get<float>(), scaleParams);
            } else {
                LocalTensor<T> out = castOutScaleT_.Get<T>();
                LocalTensor<float> acc = accScaleF_.Get<float>();
                for (uint32_t off = 0; off < d_size; off += COMPUTE_CHUNK) {
                    uint32_t n = (d_size - off < COMPUTE_CHUNK) ? (d_size - off) : COMPUTE_CHUNK;
                    Cast(out[off], acc[off], RoundMode::CAST_RINT, n);
                    PipeBarrier<PIPE_V>();
                }
                PipeBarrier<PIPE_ALL>();
                DataCopyPad(gradScaleGm_[ssOffset], out, scaleParams);
            }
            PipeBarrier<PIPE_ALL>();
        }
        if (has_shift_) {
            DataCopyExtParams shiftParams{1, static_cast<uint32_t>(rowBytes), 0, 0, 0};
            if constexpr (std::is_same<T, float>::value) {
                DataCopyPad(gradShiftGm_[ssOffset], accShiftF_.Get<float>(), shiftParams);
            } else {
                LocalTensor<T> out = castOutShiftT_.Get<T>();
                LocalTensor<float> acc = accShiftF_.Get<float>();
                for (uint32_t off = 0; off < d_size; off += COMPUTE_CHUNK) {
                    uint32_t n = (d_size - off < COMPUTE_CHUNK) ? (d_size - off) : COMPUTE_CHUNK;
                    Cast(out[off], acc[off], RoundMode::CAST_RINT, n);
                    PipeBarrier<PIPE_V>();
                }
                PipeBarrier<PIPE_ALL>();
                DataCopyPad(gradShiftGm_[ssOffset], out, shiftParams);
            }
            PipeBarrier<PIPE_ALL>();
        }
    }
};
} // namespace AscendC

template <uint8_t tilingStrategy, bool isScale, bool isShift>
__global__ __aicore__ void modulate_grad(GM_ADDR grad_Output, GM_ADDR Input, GM_ADDR Scale, GM_ADDR Shift,
                                         GM_ADDR grad_input, GM_ADDR grad_scale, GM_ADDR grad_shift, GM_ADDR workspace,
                                         GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ModulateGradRegbaseTilingData);
    GET_TILING_DATA(tilingData, tiling);
    using namespace AscendC;
#if defined(DTYPE_GRAD_OUTPUT)
    AscendC::ModulateGradKernel<DTYPE_GRAD_OUTPUT, tilingStrategy, isScale, isShift> kernel;
    kernel.Init(tilingData, Input, grad_Output, grad_input, Scale, grad_scale, grad_shift);
    kernel.Process();
#endif
}
