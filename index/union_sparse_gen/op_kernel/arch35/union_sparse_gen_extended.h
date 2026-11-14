/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Included in UnionSparseGenNs after BuildCausalQueryMask.
__simt_callee__ __aicore__ inline void LoadExtendedGroup(__ubuf__ UnionSparseGenA5GroupState* s, __gm__ int32_t* counts,
                                                         __gm__ int64_t* cuQ, __gm__ int64_t* cuKv,
                                                         __gm__ int32_t* usedQ, __gm__ int32_t* usedKv, int64_t work,
                                                         int64_t groups, int64_t batch, int64_t totalQ, int64_t topK,
                                                         int64_t capacity, int64_t width, int64_t y, bool residual)
{
    const int32_t tid = threadIdx.x;
    auto* stateWords = reinterpret_cast<__ubuf__ uint32_t*>(s);
    for (int32_t i = tid; i < sizeof(*s) / Plan::kWordBytes; i += kThreadNum) {
        stateWords[i] = 0U;
    }
    asc_syncthreads();
    if (tid == 0) {
        s->head = work / groups;
        s->outputBase = work * capacity;
        int64_t group = work % groups;
        for (int64_t b = 0; b < batch; ++b) {
            const int64_t start = cuQ != nullptr ? cuQ[b] : b * (totalQ / batch);
            const int64_t length = usedQ != nullptr ? usedQ[b] : int64_t(cuQ[b + 1]) - start;
            const int64_t n = (length + width - 1) / width;
            if (group >= n) {
                group -= n;
                continue;
            }
            s->local = group * width;
            s->qStart = start + s->local;
            s->qLen = length;
            s->kvLen = usedKv != nullptr ? usedKv[b] : int64_t(cuKv[b + 1]) - cuKv[b];
            s->logical = (s->kvLen + y - 1) / y;
            s->rows = length - s->local < width ? length - s->local : width;
            for (int32_t r = 0; r < s->rows; ++r) {
                const int64_t count = counts[s->head * totalQ + s->qStart + r];
                s->counts[r] = int32_t(count < 0 ? 0 : (count > topK ? topK : count));
                s->inputCount += s->counts[r];
                s->starts[r + 1] = s->starts[r] + s->counts[r] + (residual ? 1 : 0);
            }
            s->recordCount = s->starts[s->rows];
            break;
        }
    }
    asc_syncthreads();
}

template <typename OffsetT>
__simt_callee__ __aicore__ inline void ClearExtendedOutput(__ubuf__ UnionSparseGenA5GroupState* s, __gm__ int32_t* idx,
                                                           __gm__ int32_t* mask, OffsetT capacity, OffsetT y)
{
    const OffsetT tid = threadIdx.x, base = OffsetT(s->outputBase);
    for (OffsetT i = tid; i < capacity; i += kThreadNum) {
        idx[base + i] = -1;
    }
    auto* vectors = reinterpret_cast<__gm__ uint4*>(mask + base * y);
    const uint4 zero = make_uint4(0U, 0U, 0U, 0U);
    for (OffsetT i = tid; i < capacity * y / Plan::kVectorWords; i += kThreadNum) {
        vectors[i] = zero;
    }
    asc_syncthreads();
}

template <typename OffsetT, bool Causal>
__simt_callee__ __aicore__ inline void WriteExtendedPairs(__ubuf__ UnionSparseGenA5GroupState* s,
                                                          __ubuf__ uint32_t* ids, __ubuf__ uint32_t* bits,
                                                          __gm__ int32_t* idx, __gm__ int32_t* mask, OffsetT count,
                                                          OffsetT rankOffset, OffsetT y)
{
    // Count is block-uniform. Keep a single completion barrier at the
    // common exit; empty groups do not read state or compact buffers.
    if (count != 0) {
        const OffsetT base = OffsetT(s->outputBase) + rankOffset;
        for (OffsetT i = threadIdx.x; i < count; i += kThreadNum) {
            idx[base + i] = int32_t(ids[i]);
        }
        // Consecutive threads write consecutive four-token vectors. Membership
        // remains packed until writeback; only diagonal vectors need per-token masks.
        auto* output = reinterpret_cast<__gm__ uint4*>(mask + base * y);
        const int32_t vectorShift = __builtin_ctz(uint32_t(y)) - Plan::kVectorShift;
        const OffsetT vectorMask = (OffsetT(1) << vectorShift) - 1;
        const OffsetT causalBase = OffsetT(s->kvLen) - OffsetT(s->qLen) + OffsetT(s->local);
        for (OffsetT i = threadIdx.x; i < (count << vectorShift); i += kThreadNum) {
            const OffsetT rank = i >> vectorShift;
            const int32_t token = int32_t(i & vectorMask) * Plan::kVectorWords;
            const int32_t block = int32_t(ids[rank]);
            const uint32_t member = bits[rank];
            const OffsetT first = OffsetT(block) * y + token;
            if constexpr (!Causal) {
                const OffsetT end = OffsetT(s->kvLen);
                output[i] = make_uint4(first < end ? member : 0U, first + 1 < end ? member : 0U,
                                       first + 2 < end ? member : 0U, first + 3 < end ? member : 0U);
            } else if (first + (Plan::kVectorWords - 1) <= causalBase &&
                       first + (Plan::kVectorWords - 1) < OffsetT(s->kvLen)) {
                output[i] = make_uint4(member, member, member, member);
            } else if (first >= OffsetT(s->kvLen) || first >= causalBase + OffsetT(s->rows)) {
                output[i] = make_uint4(0U, 0U, 0U, 0U);
            } else {
                const uint32_t a = member & BuildCausalQueryMask(block, token, OffsetT(s->kvLen), OffsetT(s->qLen),
                                                                 OffsetT(s->local), int32_t(s->rows), int32_t(y));
                const uint32_t b = member & BuildCausalQueryMask(block, token + 1, OffsetT(s->kvLen), OffsetT(s->qLen),
                                                                 OffsetT(s->local), int32_t(s->rows), int32_t(y));
                const uint32_t c = member & BuildCausalQueryMask(block, token + 2, OffsetT(s->kvLen), OffsetT(s->qLen),
                                                                 OffsetT(s->local), int32_t(s->rows), int32_t(y));
                const uint32_t d = member & BuildCausalQueryMask(block, token + 3, OffsetT(s->kvLen), OffsetT(s->qLen),
                                                                 OffsetT(s->local), int32_t(s->rows), int32_t(y));
                output[i] = make_uint4(a, b, c, d);
            }
        }
    }
    asc_syncthreads();
}

template <typename OffsetT, bool Causal>
__simt_callee__ __aicore__ inline bool WriteOnlyVirtualTails(__ubuf__ UnionSparseGenA5GroupState* s,
                                                             __ubuf__ uint32_t* ids, __ubuf__ uint32_t* bits,
                                                             __gm__ int32_t* idx, __gm__ int32_t* count,
                                                             __gm__ int32_t* mask, OffsetT work, OffsetT y,
                                                             bool residual)
{
    if (s->inputCount != 0) {
        return false; // uniform CTA decision
    }
    if (threadIdx.x == 0 && residual) {
        for (int32_t r = 0; r < s->rows; ++r) {
            const int64_t visible = s->kvLen - s->qLen + s->local + r + 1;
            if (visible <= 0 || visible % y == 0) {
                continue;
            }
            const uint32_t block = uint32_t(visible / y), bit = 1U << uint32_t(r);
            if (s->rank && ids[s->rank - 1] == block) {
                bits[s->rank - 1] |= bit;
            } else {
                ids[s->rank] = block;
                bits[s->rank] = bit;
                ++s->rank;
            }
        }
    }
    asc_syncthreads();
    WriteExtendedPairs<OffsetT, Causal>(s, ids, bits, idx, mask, OffsetT(s->rank), OffsetT(0), y);
    if (threadIdx.x == 0) {
        count[work] = int32_t(s->rank);
    }
    asc_syncthreads();
    return true;
}

__simt_callee__ __aicore__ inline bool ExtendedBlockValid(__ubuf__ UnionSparseGenA5GroupState* s, int32_t row,
                                                          int32_t block, int32_t yShift, bool residual)
{
    if (block < 0 || int64_t(block) >= s->logical) {
        return false;
    }
    const int64_t visible = s->kvLen - s->qLen + s->local + row + 1;
    return !residual || (visible > 0 && int64_t(block) < (visible >> yShift));
}

// Block-wide exclusive prefix for small nonnegative integer counts.
__simt_callee__ __aicore__ inline int32_t ExtendedPrefix(int32_t value, __ubuf__ UnionSparseGenA5GroupState* s)
{
    const int32_t tid = threadIdx.x, lane = tid & Plan::kWarpLaneMask, warp = tid >> Plan::kWarpShift;
    int32_t inclusive = value;
    for (int32_t delta = 1; delta < Plan::kWarpLanes; delta *= Plan::kBinaryScanRadix) {
        const int32_t other = asc_shfl_up(inclusive, delta);
        if (lane >= delta) {
            inclusive += other;
        }
    }
    if (lane == Plan::kWarpLanes - 1) {
        s->prefix[warp] = inclusive;
    }
    asc_syncthreads();
    if (tid == 0) {
        int32_t sum = 0;
        for (int32_t w = 0; w < kThreadNum / Plan::kWarpLanes; ++w) {
            const int32_t n = s->prefix[w];
            s->prefix[w] = sum;
            sum += n;
        }
        s->prefix[kThreadNum / Plan::kWarpLanes] = sum;
    }
    asc_syncthreads();
    return s->prefix[warp] + inclusive - value;
}

template <typename OffsetT, bool Causal>
__simt_vf__ __aicore__ __launch_bounds__(kThreadNum) inline void PrepareRadixMembership(
    __gm__ int32_t* input, __gm__ int32_t* counts, __gm__ int64_t* cuQ, __gm__ int64_t* cuKv, __gm__ int32_t* usedQ,
    __gm__ int32_t* usedKv, __gm__ int32_t* idx, __gm__ int32_t* count, __gm__ int32_t* mask, __ubuf__ uint32_t* keys,
    __ubuf__ uint32_t* bits, __ubuf__ UnionSparseGenA5GroupState* s, OffsetT work, OffsetT groups, OffsetT batch,
    OffsetT totalQ, OffsetT topK, OffsetT capacity, OffsetT width, OffsetT y, OffsetT allocated, bool residual)
{
    LoadExtendedGroup(s, counts, cuQ, cuKv, usedQ, usedKv, work, groups, batch, totalQ, topK, capacity, width, y,
                      residual);
    ClearExtendedOutput(s, idx, mask, capacity, y);
    if (WriteOnlyVirtualTails<OffsetT, Causal>(s, keys, bits, idx, count, mask, work, y, residual)) {
        return;
    }
    const int32_t tid = threadIdx.x;
    const int32_t yShift = __builtin_ctz(uint32_t(y));
    // Each row contributes only its count prefix and its optional virtual slot.
    for (int32_t row = 0; row < s->rows; ++row) {
        const int64_t start = s->starts[row];
        for (int64_t col = tid; col < s->counts[row]; col += kThreadNum) {
            const int32_t block = input[(s->head * totalQ + s->qStart + row) * topK + col];
            const bool valid = ExtendedBlockValid(s, row, block, yShift, residual);
            keys[start + col] = valid ? uint32_t(block) : Plan::kInvalidSortKey;
            bits[start + col] = valid ? 1U << uint32_t(row) : 0U;
        }
        if (residual && tid == 0) {
            const int64_t visible = s->kvLen - s->qLen + s->local + row + 1;
            const bool tail = visible > 0 && (visible & (y - 1)) != 0;
            keys[start + s->counts[row]] = tail ? uint32_t(visible >> yShift) : Plan::kInvalidSortKey;
            bits[start + s->counts[row]] = tail ? 1U << uint32_t(row) : 0U;
        }
    }
    for (int64_t i = s->recordCount + tid; i < allocated; i += kThreadNum) {
        keys[i] = Plan::kInvalidSortKey;
        bits[i] = 0U;
    }
    asc_syncthreads();
}

template <typename OffsetT, bool Causal>
__simt_vf__ __aicore__ __launch_bounds__(kThreadNum) inline void CompactRadixMembership(
    __ubuf__ uint32_t* sortedKeys, __ubuf__ uint32_t* sortedBits, __ubuf__ uint32_t* compactKeys,
    __ubuf__ uint32_t* compactBits, __ubuf__ UnionSparseGenA5GroupState* s, __gm__ int32_t* idx, __gm__ int32_t* count,
    __gm__ int32_t* mask, OffsetT work, OffsetT capacity, OffsetT y)
{
    const int32_t tid = threadIdx.x;
    for (OffsetT i = tid; i < capacity; i += kThreadNum) {
        compactBits[i] = 0U;
    }
    asc_syncthreads();
    OffsetT rankBase = 0;
    for (OffsetT slab = 0; slab < s->recordCount; slab += kThreadNum) {
        const OffsetT i = slab + tid;
        const uint32_t key = i < s->recordCount ? sortedKeys[i] : Plan::kInvalidSortKey;
        const bool valid = key != Plan::kInvalidSortKey;
        const bool head = valid && (i == 0 || key != sortedKeys[i - 1]);
        const int32_t before = ExtendedPrefix(head ? 1 : 0, s);
        if (valid) {
            const OffsetT rank = rankBase + before + (head ? 0 : -1);
            if (head) {
                compactKeys[rank] = key;
            }
            asc_atomic_or(compactBits + rank, sortedBits[i]);
        }
        rankBase += s->prefix[kThreadNum / Plan::kWarpLanes];
        asc_syncthreads();
    }
    if (tid == 0) {
        s->rank = rankBase;
        count[work] = int32_t(rankBase);
    }
    asc_syncthreads();
    WriteExtendedPairs<OffsetT, Causal>(s, compactKeys, compactBits, idx, mask, rankBase, OffsetT(0), y);
}

template <typename OffsetT>
__simt_callee__ __aicore__ inline void ClearPackedChunk(__ubuf__ uint32_t* table, __ubuf__ uint32_t* presence,
                                                        OffsetT tableWords, OffsetT bitmapWords)
{
    const int32_t tid = threadIdx.x;
    for (OffsetT i = tid; i < tableWords; i += kThreadNum) {
        table[i] = 0U;
    }
    for (OffsetT i = tid; i < bitmapWords; i += kThreadNum) {
        presence[i] = 0U;
    }
}

template <typename OffsetT>
__simt_callee__ __aicore__ inline void AccumulatePackedChunk(__ubuf__ UnionSparseGenA5GroupState* s,
                                                             __gm__ int32_t* input, __ubuf__ uint32_t* table,
                                                             __ubuf__ uint32_t* presence, OffsetT base, OffsetT length,
                                                             OffsetT totalQ, OffsetT topK, OffsetT y,
                                                             int32_t groupShift, int32_t packShift, int32_t pack,
                                                             int32_t yShift, bool residual)
{
    const int32_t tid = threadIdx.x;
    for (int32_t row = 0; row < s->rows; ++row) {
        for (int64_t col = tid; col < s->counts[row]; col += kThreadNum) {
            const int32_t block = input[(s->head * totalQ + s->qStart + row) * topK + col];
            if (!ExtendedBlockValid(s, row, block, yShift, residual) || block < base || block >= base + length) {
                continue;
            }
            const OffsetT local = OffsetT(block) - base;
            asc_atomic_or(table + (local >> packShift), 1U << uint32_t(row + ((local & (pack - 1)) << groupShift)));
            asc_atomic_or(presence + local / Plan::kBitsPerWord, 1U << uint32_t(local % Plan::kBitsPerWord));
        }
        if (residual && tid == 0) {
            const int64_t visible = s->kvLen - s->qLen + s->local + row + 1;
            const int64_t block = visible >> yShift;
            if (visible > 0 && (visible & (y - 1)) != 0 && block >= base && block < base + length) {
                const OffsetT local = OffsetT(block) - base;
                asc_atomic_or(table + (local >> packShift), 1U << uint32_t(row + ((local & (pack - 1)) << groupShift)));
                asc_atomic_or(presence + local / Plan::kBitsPerWord, 1U << uint32_t(local % Plan::kBitsPerWord));
            }
        }
    }
}

template <typename OffsetT>
__simt_callee__ __aicore__ inline OffsetT CompactPackedSlab(__ubuf__ UnionSparseGenA5GroupState* s,
                                                            __ubuf__ uint32_t* table, __ubuf__ uint32_t* presence,
                                                            __ubuf__ uint32_t* compactKeys,
                                                            __ubuf__ uint32_t* compactBits, OffsetT base, OffsetT slab,
                                                            OffsetT slabWords, OffsetT bitmapWords, OffsetT rankBase,
                                                            int32_t groupShift, int32_t packShift, int32_t pack,
                                                            uint32_t memberMask, bool retainAll)
{
    const int32_t tid = threadIdx.x;
    const OffsetT wordIndex = slab + tid;
    const uint32_t word = tid < slabWords && wordIndex < bitmapWords ? presence[wordIndex] : 0U;
    const int32_t before = ExtendedPrefix(__builtin_popcount(word), s);
    uint32_t remaining = word;
    OffsetT rank = retainAll ? rankBase + before : before;
    while (remaining) {
        const uint32_t bit = __builtin_ctz(remaining);
        const OffsetT local = wordIndex * Plan::kBitsPerWord + bit;
        compactKeys[rank] = uint32_t(base + local);
        compactBits[rank++] = (table[local >> packShift] >> uint32_t(((local & (pack - 1)) << groupShift))) &
                              memberMask;
        remaining &= remaining - 1;
    }
    const OffsetT kept = s->prefix[kThreadNum / Plan::kWarpLanes];

    return kept;
}

template <typename OffsetT, bool Causal>
__simt_vf__ __aicore__ __launch_bounds__(kThreadNum) inline void PackedMembership(
    __gm__ int32_t* input, __gm__ int32_t* counts, __gm__ int64_t* cuQ, __gm__ int64_t* cuKv, __gm__ int32_t* usedQ,
    __gm__ int32_t* usedKv, __gm__ int32_t* idx, __gm__ int32_t* count, __gm__ int32_t* mask, __ubuf__ uint32_t* table,
    __ubuf__ uint32_t* presence, __ubuf__ uint32_t* compactKeys, __ubuf__ uint32_t* compactBits,
    __ubuf__ UnionSparseGenA5GroupState* s, OffsetT startWork, OffsetT workCount, OffsetT groups, OffsetT batch,
    OffsetT totalQ, OffsetT topK, OffsetT capacity, OffsetT width, OffsetT y, OffsetT chunkCapacity,
    OffsetT compactCapacity, bool residual)
{
    // All supported block/group widths are powers of two. Keep per-entry
    // filtering and packed-table addressing free of runtime integer division.
    const int32_t tid = threadIdx.x, groupShift = __builtin_ctz(uint32_t(width));
    const int32_t packShift = Plan::kWordShift - groupShift, pack = 1 << packShift;
    const int32_t yShift = __builtin_ctz(uint32_t(y));
    const uint32_t memberMask = width == Plan::kBitsPerWord ? Plan::kAllQueryBits : (1U << uint32_t(width)) - 1U;
    const bool retainAll = compactCapacity >= capacity;
    const OffsetT slabWords = retainAll ? kThreadNum :
                                          (compactCapacity / Plan::kBitsPerWord < kThreadNum ?
                                               compactCapacity / Plan::kBitsPerWord :
                                               kThreadNum);
    for (OffsetT work = startWork; work < startWork + workCount; ++work) {
        LoadExtendedGroup(s, counts, cuQ, cuKv, usedQ, usedKv, work, groups, batch, totalQ, topK, capacity, width, y,
                          residual);
        ClearExtendedOutput(s, idx, mask, capacity, y);
        if (WriteOnlyVirtualTails<OffsetT, Causal>(s, compactKeys, compactBits, idx, count, mask, work, y, residual)) {
            continue;
        }
        OffsetT rankBase = 0;
        for (OffsetT base = 0; base < s->logical; base += chunkCapacity) {
            const OffsetT length = s->logical - base < chunkCapacity ? s->logical - base : chunkCapacity;
            const OffsetT tableWords = ((length + pack - 1) >> packShift),
                          bitmapWords = (length + Plan::kBitsPerWord - 1) / Plan::kBitsPerWord;
            ClearPackedChunk(table, presence, tableWords, bitmapWords);
            asc_syncthreads();
            AccumulatePackedChunk(s, input, table, presence, base, length, totalQ, topK, y, groupShift, packShift, pack,
                                  yShift, residual);
            asc_syncthreads();
            for (OffsetT slab = 0; slab < bitmapWords; slab += slabWords) {
                const OffsetT kept = CompactPackedSlab(s, table, presence, compactKeys, compactBits, base, slab,
                                                       slabWords, bitmapWords, rankBase, groupShift, packShift, pack,
                                                       memberMask, retainAll);
                asc_syncthreads();
                if (!retainAll) {
                    WriteExtendedPairs<OffsetT, Causal>(s, compactKeys, compactBits, idx, mask, kept, rankBase, y);
                }
                // The barrier above protects prefix reads and compaction.
                // Streaming writes have their own completion barrier in the writer.
                // Only thread-private rankBase changes before the next iteration.
                rankBase += kept;
            }
        }
        if (tid == 0) {
            s->rank = rankBase;
            count[work] = int32_t(rankBase);
        }
        asc_syncthreads();
        if (retainAll) {
            WriteExtendedPairs<OffsetT, Causal>(s, compactKeys, compactBits, idx, mask, rankBase, OffsetT(0), y);
        }
    }
}
