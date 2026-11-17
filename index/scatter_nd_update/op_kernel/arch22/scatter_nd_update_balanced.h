/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SCATTER_ND_UPDATE_BALANCED_H
#define SCATTER_ND_UPDATE_BALANCED_H

#include "scatter_nd_update_common.h"

namespace ScatterNdUpdate {
// 一次搬入Host路径标志和参数，避免单独读取GM标志；小索引的临时区计入原UB预算。
template <typename T>
__aicore__ inline void LoadLocalTiling(GM_ADDR address, ScatterNdUpdateArch22TilingData& local, TPipe& pipe)
{
    constexpr uint32_t tilingBytes = 512;
    static_assert(sizeof(ScatterNdUpdateArch22TilingData) <= tilingBytes);
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, tilingBytes);
    auto bytes = buffer.Get<uint8_t>();
    GlobalTensor<uint8_t> sourceGm;
    sourceGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(address));
    DataCopyExtParams copy{1, sizeof(ScatterNdUpdateArch22TilingData), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
    DataCopyPad(bytes, sourceGm, copy, pad);
    PipeMte2ToS();
    auto source = reinterpret_cast<__ubuf__ ScatterNdUpdateArch22TilingData*>(bytes.GetPhyAddr());
    local.localIndexMode = source->localIndexMode;
    if (local.localIndexMode == 0) {
        // 通用分支也复用本次搬入的参数，释放临时UB后再建立排序缓冲。
        for (uint32_t i = 0; i < sizeof(local) / sizeof(uint64_t); ++i) {
            reinterpret_cast<uint64_t*>(&local)[i] = reinterpret_cast<__ubuf__ uint64_t*>(source)[i];
        }
        return;
    }
    local.linearIndexTiling.blockRemainLength = source->linearIndexTiling.blockRemainLength;
    local.linearIndexTiling.coreNum = source->linearIndexTiling.coreNum;
    local.scatterTiling.tailRow = source->scatterTiling.tailRow;
    local.scatterTiling.frontNum = source->scatterTiling.frontNum;
    local.linearIndexTiling.isInt64Indices = source->linearIndexTiling.isInt64Indices;
    local.scatterTiling.scatterLength = source->scatterTiling.scatterLength;
    local.scatterTiling.ubLengthForUpdates = source->scatterTiling.ubLengthForUpdates - tilingBytes / sizeof(T);
    local.viewTiling.isViewStride0 = source->viewTiling.isViewStride0;
    local.viewTiling.varStride0Elements = source->viewTiling.varStride0Elements;
    local.viewTiling.firstDimStrideRows = source->viewTiling.firstDimStrideRows;
}

// 小索引空间可在各核本地生成相同的最后位置表，避免共享索引阶段及核间同步。
__aicore__ inline bool UseLocalIndices(const ScatterNdUpdateArch22TilingData& tiling)
{
    uint64_t rows = tiling.scatterTiling.tailRow * tiling.linearIndexTiling.coreNum + tiling.scatterTiling.frontNum;
    return tiling.linearIndexTiling.indexDim == 1 && tiling.linearIndexTiling.blockNum == 0 &&
           tiling.linearIndexTiling.blockRemainLength < 32 && rows > 0 && rows <= 128;
}

template <typename T, bool isViewStride0 = false>
class ScatterNdUpdateBalancedKernel {
public:
    __aicore__ inline ScatterNdUpdateBalancedKernel(GM_ADDR updates, GM_ADDR output, GM_ADDR workspace,
                                                    const ScatterNdUpdateArch22TilingData& tiling, TPipe& pipe,
                                                    GM_ADDR indices = nullptr)
    {
        core_ = GetBlockIdx();
        cores_ = tiling.linearIndexTiling.coreNum;
        blockLength_ = tiling.linearIndexTiling.blockLength;
        fullBlocks_ = tiling.linearIndexTiling.blockNum;
        remaining_ = tiling.linearIndexTiling.blockRemainLength;
        localIndices_ = indices != nullptr;
        index64_ = tiling.linearIndexTiling.isInt64Indices != 0;
        targetRows_ = tiling.scatterTiling.tailRow * cores_ + tiling.scatterTiling.frontNum;
        indices32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int*>(indices));
        indices64Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(indices));
        rowLength_ = tiling.scatterTiling.scatterLength;
        stride0_ = tiling.viewTiling.varStride0Elements;
        rowsPerFirstDim_ = tiling.viewTiling.firstDimStrideRows;
        capacity_ = tiling.scatterTiling.ubLengthForUpdates;
        // 本地查询的向量临时区不超过4KB，仍在原索引缓冲预留范围内。
        uint64_t indexCapacity = localIndices_ && blockLength_ < 1024 ? 1024 : blockLength_;
        pipe.InitBuffer(indexQueue_, 1, indexCapacity * sizeof(int));
        pipe.InitBuffer(positionQueue_, 1, blockLength_ * sizeof(int));
        pipe.InitBuffer(updateQueue_, 1, capacity_ * sizeof(T));
        keysGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int*>(workspace));
        positionsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int*>(workspace) + tiling.linearIndexTiling.sortWorkspace);
        updatesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(updates));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
    }

    __aicore__ inline void Process()
    {
        uint64_t blocks = fullBlocks_ + (remaining_ != 0 ? 1 : 0);
        for (uint64_t block = 0; block < blocks; ++block) {
            uint64_t count = block < fullBlocks_ ? blockLength_ : remaining_;
            auto keys = indexQueue_.AllocTensor<int>();
            auto positions = positionQueue_.AllocTensor<int>();
            DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(int)), 0, 0, 0};
            DataCopyPadExtParams<int> padding{false, 0, 0, 0};
            if (localIndices_) {
                if (index64_) {
                    params.blockLen = count * sizeof(int64_t);
                    DataCopyPadExtParams<int64_t> pad64{false, 0, 0, 0};
                    DataCopyPad(keys.ReinterpretCast<int64_t>(), indices64Gm_, params, pad64);
                } else {
                    DataCopyPad(keys, indices32Gm_, params, padding);
                }
            } else {
                DataCopyPad(keys, keysGm_[block * blockLength_], params, padding);
                DataCopyPad(positions, positionsGm_[block * blockLength_], params, padding);
            }
            indexQueue_.EnQue(keys);
            positionQueue_.EnQue(positions);
            keys = indexQueue_.DeQue<int>();
            positions = positionQueue_.DeQue<int>();
            PipeMte2ToS();

            // 去重结果由排序核发布，负的末位置编码有效行数；满块唯一时无需标记。
            uint64_t valid = 0;
            if (localIndices_) {
                uint64_t seenLow = 0;
                uint64_t seenHigh = 0;
                // 倒序第一次命中就是最大pos，位图放在寄存器中，不初始化整张UB位置表。
                for (uint64_t reverse = count; reverse > 0; --reverse) {
                    uint64_t pos = reverse - 1;
                    int64_t key = index64_ ? keys.ReinterpretCast<int64_t>().GetValue(pos) : keys.GetValue(pos);
                    if (key >= 0 && static_cast<uint64_t>(key) < targetRows_) {
                        uint64_t bit = uint64_t{1} << (key % 64);
                        uint64_t seen = key < 64 ? seenLow : seenHigh;
                        if ((seen & bit) == 0) {
                            positions.SetValue(key, static_cast<int>(pos));
                            if (key < 64) {
                                seenLow |= bit;
                            } else {
                                seenHigh |= bit;
                            }
                        }
                    }
                }
                // 原序覆盖得到最大pos，再按目标行升序压缩；尚未访问的位置不会被覆盖。
                for (uint64_t key = 0; key < targetRows_ && (seenLow != 0 || seenHigh != 0); ++key) {
                    uint64_t seen = key < 64 ? seenLow : seenHigh;
                    uint64_t bit = uint64_t{1} << (key % 64);
                    if ((seen & bit) != 0) {
                        int pos = positions.GetValue(key);
                        keys.SetValue(valid, static_cast<int>(key));
                        positions.SetValue(valid++, pos);
                        if (key < 64) {
                            seenLow &= ~bit;
                        } else {
                            seenHigh &= ~bit;
                        }
                    }
                }
            } else if (count > 128) {
                // 大块仍按原记录区间分散去重，避免在排序核上串行扫描整个块。
                uint64_t quotient = count / cores_;
                uint64_t remainder = count % cores_;
                uint64_t begin = core_ * quotient + (core_ < remainder ? core_ : remainder);
                uint64_t end = begin + quotient + (core_ < remainder ? 1 : 0);
                // 半精度更新使用单次组扫描；32位类型保留已验证的原扫描方式。
                if constexpr (sizeof(T) == 2) {
                    // 首个组跨左边界时，只回溯其前缀并累计最大pos，后续记录顺序扫描一次。
                    // 整段属于跨右边界的同一重复组时，本核没有组末，直接跳过。
                    if (begin < end && end < count && keys.GetValue(begin) == keys.GetValue(end)) {
                        begin = end;
                    }
                    uint64_t read = begin;
                    int prefixPosition = -1;
                    if (read < end) {
                        int firstKey = keys.GetValue(read);
                        for (uint64_t previous = begin; previous > 0; --previous) {
                            if (keys.GetValue(previous - 1) != firstKey) {
                                break;
                            }
                            int candidate = positions.GetValue(previous - 1);
                            prefixPosition = candidate > prefixPosition ? candidate : prefixPosition;
                        }
                    }
                    while (read < end) {
                        int key = keys.GetValue(read);
                        int position = positions.GetValue(read++);
                        position = position > prefixPosition ? position : prefixPosition;
                        prefixPosition = -1;
                        while (read < end && keys.GetValue(read) == key) {
                            int candidate = positions.GetValue(read++);
                            position = candidate > position ? candidate : position;
                        }
                        // 跨右边界的组由包含组末的核处理，保持每个目标只有一个写入核。
                        if (read == end && end < count && keys.GetValue(end) == key) {
                            break;
                        }
                        keys.SetValue(valid, key);
                        positions.SetValue(valid++, position);
                    }
                } else {
                    for (uint64_t read = begin; read < end; ++read) {
                        int key = keys.GetValue(read);
                        if (read + 1 < count && keys.GetValue(read + 1) == key) {
                            continue;
                        }
                        int position = positions.GetValue(read);
                        for (uint64_t previous = read; previous > 0; --previous) {
                            if (keys.GetValue(previous - 1) != key) {
                                break;
                            }
                            int candidate = positions.GetValue(previous - 1);
                            position = candidate > position ? candidate : position;
                        }
                        keys.SetValue(valid, key);
                        positions.SetValue(valid++, position);
                    }
                }
            } else {
                int lastPosition = positions.GetValue(count - 1);
                valid = lastPosition < 0 ? static_cast<uint64_t>(-lastPosition - 1) : count;
            }
            if (valid == 0) {
                indexQueue_.FreeTensor(keys);
                positionQueue_.FreeTensor(positions);
                if (cores_ > 1 && block + 1 < blocks) {
                    SyncAll();
                }
                continue;
            }
            uint64_t slices = (cores_ + valid - 1) / valid;
            if (count > 128) {
                slices = 1;
            }
            // 初始按每片至少2KB限制拆分，避免为了用满核产生大量短搬运。
            uint64_t maxSlices = rowLength_ * sizeof(T) / 2048;
            maxSlices = maxSlices == 0 ? 1 : maxSlices;
            slices = slices < maxSlices ? slices : maxSlices;
            uint64_t alignElements = 32 / sizeof(T);
            uint64_t rowUnits = (rowLength_ + alignElements - 1) / alignElements;
            uint64_t quotient = rowUnits / slices;
            uint64_t remainder = rowUnits % slices;
            // 先按有效行再按尾片编号，每个输出元素只归一个任务，均匀输入整行分工。
            uint64_t taskBegin = count > 128 ? 0 : core_;
            uint64_t taskStep = count > 128 ? 1 : cores_;
            uint64_t batchRows = slices == 1 ? capacity_ / (rowUnits * alignElements) : 1;
            batchRows = batchRows == 0 ? 1 : batchRows;
            for (uint64_t task = taskBegin; task < valid * slices; task += taskStep * batchRows) {
                uint64_t rows = (valid * slices - task + taskStep - 1) / taskStep;
                rows = rows < batchRows ? rows : batchRows;
                uint64_t slice = task % slices;
                uint64_t start = (slice * quotient + (slice < remainder ? slice : remainder)) * alignElements;
                uint64_t end = start + (quotient + (slice < remainder ? 1 : 0)) * alignElements;
                end = end < rowLength_ ? end : rowLength_;
                for (uint64_t offset = start; offset < end; offset += capacity_) {
                    uint64_t length = end - offset;
                    length = length < capacity_ ? length : capacity_;
                    uint64_t tileStride = (length + alignElements - 1) / alignElements * alignElements;
                    auto data = updateQueue_.AllocTensor<T>();
                    DataCopyExtParams copy{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
                    DataCopyPadExtParams<T> pad{false, 0, 0, 0};
                    // 整行任务保留原批量搬运方式，避免大块均匀输入每行都单独等待。
                    for (uint64_t j = 0; j < rows; ++j) {
                        uint64_t row = (task + j * taskStep) / slices;
                        uint64_t pos = positions.GetValue(row);
                        DataCopyPad(data[j * tileStride], updatesGm_[pos * rowLength_ + offset], copy, pad);
                    }
                    updateQueue_.EnQue(data);
                    data = updateQueue_.DeQue<T>();
                    event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
                    SetFlag<HardEvent::MTE2_MTE3>(event);
                    WaitFlag<HardEvent::MTE2_MTE3>(event);
                    for (uint64_t j = 0; j < rows; ++j) {
                        uint64_t row = (task + j * taskStep) / slices;
                        uint64_t key = keys.GetValue(row);
                        uint64_t dst = ResolveOutOffset<isViewStride0>(key, rowLength_, rowsPerFirstDim_, stride0_,
                                                                       offset);
                        DataCopyPad(outputGm_[dst], data[j * tileStride], copy);
                    }
                    PipeMte3ToS();
                    updateQueue_.FreeTensor(data);
                }
            }
            indexQueue_.FreeTensor(keys);
            positionQueue_.FreeTensor(positions);
            if (cores_ > 1 && block + 1 < blocks) {
                // 同一目标在后块可能归另一个核；所有核完成前块写出后才进入后块。
                PipeMte3ToS();
                SyncAll();
            }
        }
    }

public:
    // 小索引只保留输入比较与更新所需的缓冲，不构造通用排序/分块状态。
    static __aicore__ inline void ProcessLocal(GM_ADDR indices, GM_ADDR updates, GM_ADDR output,
                                               const ScatterNdUpdateArch22TilingData& tiling, TPipe& pipe)
    {
        uint64_t core_ = GetBlockIdx();
        uint64_t cores_ = tiling.linearIndexTiling.coreNum;
        uint64_t remaining_ = tiling.linearIndexTiling.blockRemainLength;
        bool index64_ = tiling.linearIndexTiling.isInt64Indices != 0;
        uint64_t targetRows_ = tiling.scatterTiling.tailRow * cores_ + tiling.scatterTiling.frontNum;
        uint64_t rowLength_ = tiling.scatterTiling.scatterLength;
        uint64_t stride0_ = tiling.viewTiling.varStride0Elements;
        uint64_t rowsPerFirstDim_ = tiling.viewTiling.firstDimStrideRows;
        uint64_t capacity_ = tiling.scatterTiling.ubLengthForUpdates;
        uint64_t alignedLength = (rowLength_ * sizeof(T) + 31) / 32 * 32 / sizeof(T);
        capacity_ = capacity_ < alignedLength ? capacity_ : alignedLength;
        TBuf<TPosition::VECCALC> indexBuffer;
        TQue<TPosition::VECOUT, 1> updateQueue_;
        pipe.InitBuffer(indexBuffer, 4096);
        pipe.InitBuffer(updateQueue_, 1, capacity_ * sizeof(T));
        auto keys = indexBuffer.Get<int>();
        GlobalTensor<int> indices32Gm_;
        GlobalTensor<int64_t> indices64Gm_;
        GlobalTensor<T> updatesGm_, outputGm_;
        indices32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int*>(indices));
        indices64Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(indices));
        updatesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(updates));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
        DataCopyExtParams copyIndex{1, static_cast<uint32_t>(remaining_ * (index64_ ? 8 : 4)), 0, 0, 0};
        if (index64_) {
            DataCopyPadExtParams<int64_t> pad{false, 0, 0, 0};
            DataCopyPad(keys.ReinterpretCast<int64_t>(), indices64Gm_, copyIndex, pad);
        } else {
            DataCopyPadExtParams<int> pad{false, 0, 0, 0};
            DataCopyPad(keys, indices32Gm_, copyIndex, pad);
        }
        event_t inputEvent = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(inputEvent);
        WaitFlag<HardEvent::MTE2_V>(inputEvent);
        if (index64_) {
            Cast(keys, keys.ReinterpretCast<int64_t>(), RoundMode::CAST_NONE, remaining_);
            PipeBarrier<PIPE_V>();
        }
        auto keyFloat = keys[256].ReinterpretCast<float>();
        auto mask = keys[512].ReinterpretCast<uint8_t>();
        // 补齐向量尾部，避免非整重复长度的归约和比较读取未初始化元素。
        uint32_t vectorCount = (remaining_ + 63) / 64 * 64;
        if (remaining_ != vectorCount) {
            Duplicate(keyFloat, -1.0f, vectorCount);
            PipeBarrier<PIPE_V>();
        }
        Cast(keyFloat, keys, RoundMode::CAST_ROUND, remaining_);
        PipeBarrier<PIPE_V>();
        auto maxKey = keys[768].ReinterpretCast<float>();
        auto reduceTemporary = keys[896].ReinterpretCast<float>();
        ReduceMax(maxKey, keyFloat, reduceTemporary, vectorCount, false);
        PipeVToS();
        int lastKey = static_cast<int>(maxKey.GetValue(0));
        // 最大索引之后不存在更新，避免热点集中时继续查询空行。
        uint64_t activeRows = lastKey < 0 ? 0 : static_cast<uint64_t>(lastKey) + 1;
        activeRows = activeRows < targetRows_ ? activeRows : targetRows_;
        // 每核只查询自己负责的目标行，匹配位图的最高置位直接给出最后输入位置。
        // 长行拆为两片，连续热点目标也可分给更多核；短行避免额外的小片搬运。
        bool split = rowLength_ * sizeof(T) >= 4096;
        uint64_t slices = split ? 2 : 1;
        uint64_t alignElements = 32 / sizeof(T);
        uint64_t units = (rowLength_ + alignElements - 1) / alignElements;
        uint64_t quotient = split ? units / 2 : units;
        uint64_t remainder = split ? units % 2 : 0;
        for (uint64_t task = core_; task < activeRows * slices; task += cores_) {
            uint64_t key = split ? task / 2 : task;
            uint64_t slice = split ? task % 2 : 0;
            CompareScalar(mask, keyFloat, static_cast<float>(static_cast<int>(key)), CMPMODE::EQ, vectorCount);
            PipeVToS();
            auto words = mask.ReinterpretCast<uint64_t>();
            uint64_t low = 0;
            uint64_t high = 0;
            uint64_t positionBase = 0;
            if (remaining_ > 128) {
                // 现有4KB缓冲可容纳256条索引，优先检查后128条以选择最大原始位置。
                low = words.GetValue(2);
                high = remaining_ > 192 ? words.GetValue(3) : 0;
                if (remaining_ < 192) {
                    low &= (uint64_t{1} << (remaining_ - 128)) - 1;
                } else if (remaining_ > 192 && remaining_ < 256) {
                    high &= (uint64_t{1} << (remaining_ - 192)) - 1;
                }
                if (low != 0 || high != 0) {
                    positionBase = 128;
                }
            }
            if (positionBase == 0) {
                low = words.GetValue(0);
                high = remaining_ > 64 ? words.GetValue(1) : 0;
            }
            if (remaining_ < 64) {
                low &= (uint64_t{1} << remaining_) - 1;
            } else if (remaining_ > 64 && remaining_ < 128) {
                high &= (uint64_t{1} << (remaining_ - 64)) - 1;
            }
            uint64_t bits = high != 0 ? high : low;
            if (bits == 0) {
                continue;
            }
            uint64_t pos = positionBase + (high != 0 ? 64 : 0) + 63 -
                           static_cast<uint64_t>(ScalarCountLeadingZero(bits));
            uint64_t start = (slice * quotient + (slice < remainder ? slice : remainder)) * alignElements;
            uint64_t end = start + (quotient + (slice < remainder ? 1 : 0)) * alignElements;
            end = end < rowLength_ ? end : rowLength_;
            for (uint64_t offset = start; offset < end; offset += capacity_) {
                uint64_t length = end - offset;
                length = length < capacity_ ? length : capacity_;
                auto data = updateQueue_.AllocTensor<T>();
                DataCopyExtParams copy{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
                DataCopyPadExtParams<T> pad{false, 0, 0, 0};
                DataCopyPad(data, updatesGm_[pos * rowLength_ + offset], copy, pad);
                updateQueue_.EnQue(data);
                data = updateQueue_.DeQue<T>();
                event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
                SetFlag<HardEvent::MTE2_MTE3>(event);
                WaitFlag<HardEvent::MTE2_MTE3>(event);
                uint64_t dst = ResolveOutOffset<isViewStride0>(key, rowLength_, rowsPerFirstDim_, stride0_, offset);
                DataCopyPad(outputGm_[dst], data, copy);
                PipeMte3ToS();
                updateQueue_.FreeTensor(data);
            }
        }
    }

private:
    GlobalTensor<int> keysGm_, positionsGm_;
    GlobalTensor<int> indices32Gm_;
    GlobalTensor<int64_t> indices64Gm_;
    GlobalTensor<T> updatesGm_, outputGm_;
    TQue<TPosition::VECIN, 1> indexQueue_, positionQueue_;
    TQue<TPosition::VECOUT, 1> updateQueue_;
    uint64_t core_, cores_, blockLength_, fullBlocks_, remaining_;
    uint64_t rowLength_, capacity_;
    uint64_t stride0_, rowsPerFirstDim_;
    uint64_t targetRows_;
    bool localIndices_, index64_;
};
} // namespace ScatterNdUpdate
#endif
