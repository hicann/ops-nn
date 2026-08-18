/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bn_training_reduce_tiling_public.h"

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include "op_common/op_host/util/math_util.h"

namespace optiling {
namespace {

constexpr int64_t kCacheBytes = 16 * 1024;
constexpr int64_t kFp32Bytes = 4;
constexpr int64_t kVectorBytes = 256;
using Ops::Base::CeilDiv;

bool TryAddNonNegative(int64_t lhs, int64_t rhs, int64_t& result)
{
    if (lhs < 0 || rhs < 0 || lhs > std::numeric_limits<int64_t>::max() - rhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

bool TryMulNonNegative(int64_t lhs, int64_t rhs, int64_t& result)
{
    if (lhs < 0 || rhs < 0 || (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool TryCeilAlignNonNegative(int64_t value, int64_t alignment, int64_t& result)
{
    if (value < 0 || alignment <= 0) {
        return false;
    }
    return TryMulNonNegative(CeilDiv(value, alignment), alignment, result);
}

int64_t FloorAlign(int64_t value, int64_t alignment) { return value - value % alignment; }

int64_t DTypeSize(BNTrainingReducePublicDType dtype)
{
    switch (dtype) {
        case BNTrainingReducePublicDType::FLOAT16:
        case BNTrainingReducePublicDType::BFLOAT16:
            return 2;
        case BNTrainingReducePublicDType::FLOAT32:
            return 4;
        default:
            return 0;
    }
}

size_t ChannelIndex(BNTrainingReducePublicFormat format)
{
    return format == BNTrainingReducePublicFormat::NCHW ? 1U : 3U;
}

int64_t ChannelSize(const BNTrainingReducePublicInputs& inputs) { return inputs.shape[ChannelIndex(inputs.format)]; }

bool TryComputeGroupWorkspaceSize(int64_t rGroupCnt, int64_t channels, size_t systemWorkspaceSize,
                                  size_t& workspaceSize)
{
    if (rGroupCnt <= 0 || channels <= 0) {
        return false;
    }
    constexpr size_t kMaxSize = std::numeric_limits<size_t>::max();
    const size_t groups = static_cast<size_t>(rGroupCnt);
    const size_t channelCount = static_cast<size_t>(channels);
    if (groups > kMaxSize / channelCount) {
        return false;
    }
    const size_t elements = groups * channelCount;
    if (elements > kMaxSize / static_cast<size_t>(kFp32Bytes)) {
        return false;
    }
    const size_t userBytes = elements * static_cast<size_t>(kFp32Bytes);
    if (systemWorkspaceSize > kMaxSize - userBytes) {
        return false;
    }
    workspaceSize = systemWorkspaceSize + userBytes;
    return true;
}

struct TilingContext {
    std::vector<int64_t> axisShape;
    std::vector<bool> isReduce;
    std::array<int64_t, MAX_PATTERN_RANK> axisStride = {};
    int32_t axisNum = 0;
    bool isTailR = false;
    bool isGroup = false;
    int64_t coreNum = 0;
    int64_t ubSize = 0;
    int64_t blockSize = 0;
    int64_t cacheLineSize = 0;
    int64_t dtypeSize = 0;
    int32_t aSplitAxisIdx = 0;
    int32_t rSplitAxisIdx = 0;
    int64_t aUbFactor = 0;
    int64_t aUbFactorAlign = 0;
    int64_t rUbFactor = 0;
    int64_t rUbFactorAlign = 0;
    int64_t innerAProd = 0;
    int64_t innerAProdAlign = 0;
    int64_t innerRProd = 0;
    int64_t innerRProdAlign = 0;
    int64_t aSplitChunkCnt = 0;
    int64_t aLoopCntTotal = 0;
    int64_t rLoopCntTotal = 0;
    int64_t aSmallCoreLoopCnt = 0;
    int64_t aBigCoreLoopCnt = 0;
    int32_t aBigCoreCnt = 0;
    int32_t usedCoreNum = 0;
    int64_t rGroupCnt = 0;
    int64_t preReduceUbSize = 0;
    int64_t postReduceUbSize = 0;
    int64_t tmpBufUbSize = 0;
    bool ubGatePassed = false;
    bool cacheGatePassed = false;
    bool arithmeticOverflow = false;
};

BNTrainingReducePublicStatus ValidateInputs(const BNTrainingReducePublicInputs& inputs)
{
    if (!inputs.inputPresent) {
        return BNTrainingReducePublicStatus::NULL_INPUT;
    }
    if (inputs.rank != 4 ||
        (inputs.format != BNTrainingReducePublicFormat::NCHW && inputs.format != BNTrainingReducePublicFormat::NHWC)) {
        return BNTrainingReducePublicStatus::SHAPE_MISMATCH;
    }
    if (std::any_of(inputs.shape.begin(), inputs.shape.end(), [](int64_t dim) { return dim < 0; })) {
        return BNTrainingReducePublicStatus::SHAPE_MISMATCH;
    }
    if (DTypeSize(inputs.inputDtype) == 0) {
        return BNTrainingReducePublicStatus::DTYPE_NOT_SUPPORTED;
    }
    const int64_t channels = ChannelSize(inputs);
    if (inputs.sumRank != 1 || inputs.sumDim0 != channels || inputs.squareSumRank != 1 ||
        inputs.squareSumDim0 != channels) {
        return BNTrainingReducePublicStatus::SHAPE_MISMATCH;
    }
    if (inputs.sumDtype != BNTrainingReducePublicDType::FLOAT32 ||
        inputs.squareSumDtype != BNTrainingReducePublicDType::FLOAT32) {
        return BNTrainingReducePublicStatus::DTYPE_NOT_SUPPORTED;
    }
    if (inputs.coreNum <= 0 || inputs.coreNum > std::numeric_limits<int32_t>::max() || inputs.ubSize <= 0 ||
        inputs.blockSize <= 0 || inputs.cacheLineSize <= 0 || inputs.vectorSize != kVectorBytes ||
        inputs.blockSize < DTypeSize(inputs.inputDtype)) {
        return BNTrainingReducePublicStatus::TILING_FAILED;
    }
    return BNTrainingReducePublicStatus::SUCCESS;
}

BNTrainingReduceEmptyKind ClassifyEmpty(const BNTrainingReducePublicInputs& inputs)
{
    const size_t channelIndex = ChannelIndex(inputs.format);
    if (inputs.shape[channelIndex] == 0) {
        return BNTrainingReduceEmptyKind::EMPTY_A;
    }
    for (size_t i = 0; i < inputs.shape.size(); ++i) {
        if (i != channelIndex && inputs.shape[i] == 0) {
            return BNTrainingReduceEmptyKind::EMPTY_R;
        }
    }
    return BNTrainingReduceEmptyKind::NORMAL;
}

bool NormalizePattern(const BNTrainingReducePublicInputs& inputs, TilingContext& ctx)
{
    ctx.coreNum = inputs.coreNum;
    ctx.ubSize = inputs.ubSize;
    ctx.blockSize = inputs.blockSize;
    ctx.cacheLineSize = inputs.cacheLineSize;
    ctx.dtypeSize = DTypeSize(inputs.inputDtype);

    std::array<bool, 4> initialTypes = {true, true, true, true};
    initialTypes[ChannelIndex(inputs.format)] = false;
    for (size_t i = 0; i < inputs.shape.size(); ++i) {
        if (inputs.shape[i] != 1) {
            ctx.axisShape.push_back(inputs.shape[i]);
            ctx.isReduce.push_back(initialTypes[i]);
        }
    }
    if (ctx.axisShape.empty()) {
        ctx.axisShape.push_back(1);
        ctx.isReduce.push_back(false);
    }

    std::vector<int64_t> fusedShape;
    std::vector<bool> fusedTypes;
    for (size_t i = 0; i < ctx.axisShape.size(); ++i) {
        if (!fusedTypes.empty() && fusedTypes.back() == ctx.isReduce[i]) {
            int64_t fusedSize = 0;
            if (!TryMulNonNegative(fusedShape.back(), ctx.axisShape[i], fusedSize)) {
                ctx.arithmeticOverflow = true;
                return false;
            }
            fusedShape.back() = fusedSize;
        } else {
            fusedShape.push_back(ctx.axisShape[i]);
            fusedTypes.push_back(ctx.isReduce[i]);
        }
    }
    ctx.axisShape = std::move(fusedShape);
    ctx.isReduce = std::move(fusedTypes);

    if (ctx.isReduce.front()) {
        ctx.axisShape.insert(ctx.axisShape.begin(), 1);
        ctx.isReduce.insert(ctx.isReduce.begin(), false);
    }
    const bool hasReduce = std::any_of(ctx.isReduce.begin(), ctx.isReduce.end(), [](bool value) { return value; });
    if (!hasReduce) {
        if (ctx.axisShape.size() == 1 && ctx.axisShape[0] == 1) {
            ctx.axisShape.push_back(1);
            ctx.isReduce.push_back(true);
        } else {
            ctx.axisShape.insert(ctx.axisShape.begin(), {1, 1});
            ctx.isReduce.insert(ctx.isReduce.begin(), {false, true});
        }
    }

    ctx.axisNum = static_cast<int32_t>(ctx.axisShape.size());
    int64_t stride = 1;
    for (int32_t i = ctx.axisNum - 1; i >= 0; --i) {
        ctx.axisStride[static_cast<size_t>(i)] = stride;
        int64_t nextStride = 0;
        if (!TryMulNonNegative(stride, ctx.axisShape[static_cast<size_t>(i)], nextStride)) {
            ctx.arithmeticOverflow = true;
            return false;
        }
        stride = nextStride;
    }
    int64_t tensorBytes = 0;
    if (!TryMulNonNegative(stride, ctx.dtypeSize, tensorBytes)) {
        ctx.arithmeticOverflow = true;
        return false;
    }
    ctx.isTailR = (ctx.axisNum % 2 == 0);
    return true;
}

int32_t LastAxis(const TilingContext& ctx, bool wantReduce)
{
    for (int32_t i = ctx.axisNum - 1; i >= 0; --i) {
        if (ctx.isReduce[static_cast<size_t>(i)] == wantReduce) {
            return i;
        }
    }
    return -1;
}

int32_t PrevAxis(const TilingContext& ctx, int32_t from, bool wantReduce)
{
    for (int32_t i = from - 1; i >= 0; --i) {
        if (ctx.isReduce[static_cast<size_t>(i)] == wantReduce) {
            return i;
        }
    }
    return -1;
}

bool ComputeAUbFactor(TilingContext& ctx, int64_t maxInnerAElems)
{
    const int32_t lastA = LastAxis(ctx, false);
    const int64_t blockElems = ctx.blockSize / ctx.dtypeSize;
    int64_t target = maxInnerAElems;
    if (ctx.isTailR) {
        int64_t totalA = 1;
        for (int32_t i = 0; i < ctx.axisNum; i += 2) {
            if (!TryMulNonNegative(totalA, ctx.axisShape[static_cast<size_t>(i)], totalA)) {
                ctx.arithmeticOverflow = true;
                return false;
            }
        }
        target = std::min(target, CeilDiv(totalA, ctx.coreNum));
    }
    target = std::max<int64_t>(target, 1);

    ctx.innerAProd = 1;
    ctx.innerAProdAlign = 1;
    ctx.aSplitAxisIdx = lastA;
    while (true) {
        int64_t currentSpan = 0;
        if (!TryMulNonNegative(ctx.axisShape[static_cast<size_t>(ctx.aSplitAxisIdx)], ctx.innerAProdAlign,
                               currentSpan)) {
            ctx.arithmeticOverflow = true;
            return false;
        }
        if (currentSpan >= target) {
            break;
        }
        const int32_t prev = PrevAxis(ctx, ctx.aSplitAxisIdx, false);
        if (prev < 0) {
            break;
        }
        const int64_t size = ctx.axisShape[static_cast<size_t>(ctx.aSplitAxisIdx)];
        int64_t alignedSize = size;
        if (ctx.aSplitAxisIdx == lastA && !ctx.isTailR && !TryCeilAlignNonNegative(size, blockElems, alignedSize)) {
            ctx.arithmeticOverflow = true;
            return false;
        }
        if (!TryMulNonNegative(ctx.innerAProd, size, ctx.innerAProd) ||
            !TryMulNonNegative(ctx.innerAProdAlign, alignedSize, ctx.innerAProdAlign)) {
            ctx.arithmeticOverflow = true;
            return false;
        }
        ctx.aSplitAxisIdx = prev;
    }
    const int64_t splitSize = ctx.axisShape[static_cast<size_t>(ctx.aSplitAxisIdx)];
    ctx.aUbFactor = std::min(CeilDiv(target, ctx.innerAProdAlign), splitSize);
    ctx.aUbFactor = std::max<int64_t>(ctx.aUbFactor, 1);
    ctx.aUbFactorAlign = ctx.aUbFactor;
    if (ctx.aSplitAxisIdx == lastA && !ctx.isTailR &&
        !TryCeilAlignNonNegative(ctx.aUbFactor, blockElems, ctx.aUbFactorAlign)) {
        ctx.arithmeticOverflow = true;
        return false;
    }
    return true;
}

bool ComputeRiMax(TilingContext& ctx, int64_t& rMax)
{
    rMax = 0;
    if (ctx.ubSize <= kCacheBytes) {
        return true;
    }
    const int64_t ubAvailable = ctx.ubSize - kCacheBytes;
    int64_t aUnit = 0;
    if (!TryMulNonNegative(ctx.aUbFactorAlign, ctx.innerAProdAlign, aUnit)) {
        ctx.arithmeticOverflow = true;
        return false;
    }
    const int64_t bytesPerRElem = 2 * ctx.dtypeSize + 2 * kFp32Bytes;
    int64_t outputBytes = 0;
    int64_t denominator = 0;
    if (!TryMulNonNegative(aUnit, kFp32Bytes, outputBytes) || !TryMulNonNegative(aUnit, bytesPerRElem, denominator)) {
        ctx.arithmeticOverflow = true;
        return false;
    }
    if (aUnit <= 0 || outputBytes >= ubAvailable) {
        return true;
    }
    const int64_t numerator = ubAvailable - outputBytes;
    rMax = numerator / denominator;
    return true;
}

bool ComputeRUbFactor(TilingContext& ctx)
{
    int64_t rMax = 0;
    if (!ComputeRiMax(ctx, rMax)) {
        return false;
    }
    if (rMax < 1) {
        return false;
    }
    const int32_t lastR = LastAxis(ctx, true);
    const int64_t blockElems = ctx.blockSize / ctx.dtypeSize;
    ctx.innerRProd = 1;
    ctx.innerRProdAlign = 1;
    ctx.rSplitAxisIdx = lastR;
    while (true) {
        int64_t currentSpan = 0;
        if (!TryMulNonNegative(ctx.axisShape[static_cast<size_t>(ctx.rSplitAxisIdx)], ctx.innerRProdAlign,
                               currentSpan)) {
            ctx.arithmeticOverflow = true;
            return false;
        }
        if (currentSpan > rMax) {
            break;
        }
        const int32_t prev = PrevAxis(ctx, ctx.rSplitAxisIdx, true);
        if (prev < 0) {
            break;
        }
        const int64_t size = ctx.axisShape[static_cast<size_t>(ctx.rSplitAxisIdx)];
        int64_t alignedSize = size;
        if (ctx.rSplitAxisIdx == lastR && ctx.isTailR && !TryCeilAlignNonNegative(size, blockElems, alignedSize)) {
            ctx.arithmeticOverflow = true;
            return false;
        }
        if (!TryMulNonNegative(ctx.innerRProd, size, ctx.innerRProd) ||
            !TryMulNonNegative(ctx.innerRProdAlign, alignedSize, ctx.innerRProdAlign)) {
            ctx.arithmeticOverflow = true;
            return false;
        }
        ctx.rSplitAxisIdx = prev;
    }
    const int64_t splitSize = ctx.axisShape[static_cast<size_t>(ctx.rSplitAxisIdx)];
    ctx.rUbFactor = std::min(rMax / ctx.innerRProdAlign, splitSize);
    if (ctx.rUbFactor < 1) {
        return false;
    }
    if (ctx.isTailR && ctx.rSplitAxisIdx == lastR && ctx.rUbFactor < splitSize) {
        ctx.rUbFactor = FloorAlign(ctx.rUbFactor, blockElems);
        if (ctx.rUbFactor == 0) {
            return false;
        }
    }
    ctx.rUbFactorAlign = ctx.rUbFactor;
    if (ctx.isTailR && ctx.rSplitAxisIdx == lastR &&
        !TryCeilAlignNonNegative(ctx.rUbFactor, blockElems, ctx.rUbFactorAlign)) {
        ctx.arithmeticOverflow = true;
        return false;
    }
    return true;
}

bool RIsFullyLoaded(const TilingContext& ctx)
{
    return ctx.rUbFactor == ctx.axisShape[static_cast<size_t>(ctx.rSplitAxisIdx)] &&
           PrevAxis(ctx, ctx.rSplitAxisIdx, true) < 0;
}

bool ExpandAIfRFullyLoaded(TilingContext& ctx)
{
    if (!RIsFullyLoaded(ctx)) {
        return true;
    }
    int64_t rPadded = 0;
    if (!TryMulNonNegative(ctx.rUbFactorAlign, ctx.innerRProdAlign, rPadded)) {
        ctx.arithmeticOverflow = true;
        return false;
    }
    const int64_t bytesPerRElem = 2 * ctx.dtypeSize + 2 * kFp32Bytes;
    int64_t inputAndTmpBytes = 0;
    int64_t bytesPerA = 0;
    if (!TryMulNonNegative(rPadded, bytesPerRElem, inputAndTmpBytes) ||
        !TryAddNonNegative(inputAndTmpBytes, kFp32Bytes, bytesPerA)) {
        ctx.arithmeticOverflow = true;
        return false;
    }
    if (ctx.ubSize <= kCacheBytes || bytesPerA <= 0) {
        return false;
    }
    const int64_t solvedA = (ctx.ubSize - kCacheBytes) / bytesPerA;
    int64_t totalA = 1;
    for (int32_t i = 0; i < ctx.axisNum; i += 2) {
        if (!TryMulNonNegative(totalA, ctx.axisShape[static_cast<size_t>(i)], totalA)) {
            ctx.arithmeticOverflow = true;
            return false;
        }
    }
    int64_t target = std::min(solvedA, totalA);
    target = std::min(target, kCacheBytes / kFp32Bytes);
    int64_t currentAUnit = 0;
    if (!TryMulNonNegative(ctx.aUbFactorAlign, ctx.innerAProdAlign, currentAUnit)) {
        ctx.arithmeticOverflow = true;
        return false;
    }
    if (target <= currentAUnit) {
        return true;
    }

    const int32_t lastA = LastAxis(ctx, false);
    const int64_t blockElems = ctx.blockSize / ctx.dtypeSize;
    ctx.innerAProd = 1;
    ctx.innerAProdAlign = 1;
    ctx.aSplitAxisIdx = lastA;
    while (true) {
        int64_t currentSpan = 0;
        if (!TryMulNonNegative(ctx.axisShape[static_cast<size_t>(ctx.aSplitAxisIdx)], ctx.innerAProdAlign,
                               currentSpan)) {
            ctx.arithmeticOverflow = true;
            return false;
        }
        if (currentSpan >= target) {
            break;
        }
        const int32_t prev = PrevAxis(ctx, ctx.aSplitAxisIdx, false);
        if (prev < 0) {
            break;
        }
        const int64_t size = ctx.axisShape[static_cast<size_t>(ctx.aSplitAxisIdx)];
        int64_t alignedSize = size;
        if (ctx.aSplitAxisIdx == lastA && !ctx.isTailR && !TryCeilAlignNonNegative(size, blockElems, alignedSize)) {
            ctx.arithmeticOverflow = true;
            return false;
        }
        if (!TryMulNonNegative(ctx.innerAProd, size, ctx.innerAProd) ||
            !TryMulNonNegative(ctx.innerAProdAlign, alignedSize, ctx.innerAProdAlign)) {
            ctx.arithmeticOverflow = true;
            return false;
        }
        ctx.aSplitAxisIdx = prev;
    }
    const int64_t splitSize = ctx.axisShape[static_cast<size_t>(ctx.aSplitAxisIdx)];
    const int64_t aTotalUb = std::min(target / ctx.innerAProdAlign, splitSize);
    if (!ctx.isTailR && ctx.aSplitAxisIdx == lastA) {
        const int64_t aligned = FloorAlign(aTotalUb, blockElems);
        if (aligned == 0) {
            return false;
        }
        ctx.aUbFactor = aligned;
        if (!TryCeilAlignNonNegative(aligned, blockElems, ctx.aUbFactorAlign)) {
            ctx.arithmeticOverflow = true;
            return false;
        }
    } else {
        ctx.aUbFactor = aTotalUb;
        ctx.aUbFactorAlign = aTotalUb;
    }
    return true;
}

bool ComputeRLoopCnt(TilingContext& ctx)
{
    int64_t outerRProd = 1;
    for (int32_t i = 1; i < ctx.rSplitAxisIdx; i += 2) {
        if (!TryMulNonNegative(outerRProd, ctx.axisShape[static_cast<size_t>(i)], outerRProd)) {
            ctx.arithmeticOverflow = true;
            return false;
        }
    }
    const int64_t splitSize = ctx.axisShape[static_cast<size_t>(ctx.rSplitAxisIdx)];
    if (!TryMulNonNegative(outerRProd, CeilDiv(splitSize, ctx.rUbFactor), ctx.rLoopCntTotal)) {
        ctx.arithmeticOverflow = true;
        return false;
    }
    return true;
}

int64_t ComputeCacheCount(int64_t rLoopCntTotal)
{
    int64_t cacheCount = 0;
    for (int64_t loops = rLoopCntTotal; loops > 0; loops >>= 1) {
        ++cacheCount;
    }
    return cacheCount;
}

bool FitsCache(TilingContext& ctx)
{
    const int64_t cacheCount = ComputeCacheCount(ctx.rLoopCntTotal);
    int64_t aUnit = 0;
    int64_t cachedElements = 0;
    int64_t cacheBytes = 0;
    if (!TryMulNonNegative(ctx.aUbFactorAlign, ctx.innerAProdAlign, aUnit) ||
        !TryMulNonNegative(cacheCount, aUnit, cachedElements) ||
        !TryMulNonNegative(cachedElements, kFp32Bytes, cacheBytes)) {
        ctx.arithmeticOverflow = true;
        return false;
    }
    return cacheCount > 0 && cacheBytes <= kCacheBytes;
}

bool FitsUb(TilingContext& ctx)
{
    int64_t aUnit = 0;
    int64_t rUnit = 0;
    int64_t unit = 0;
    int64_t rawPreBytes = 0;
    int64_t rawTmpBytes = 0;
    int64_t rawOutBytes = 0;
    int64_t preBytes = 0;
    int64_t tmpBytes = 0;
    int64_t outBytes = 0;
    if (!TryMulNonNegative(ctx.aUbFactorAlign, ctx.innerAProdAlign, aUnit) ||
        !TryMulNonNegative(ctx.rUbFactorAlign, ctx.innerRProdAlign, rUnit) || !TryMulNonNegative(aUnit, rUnit, unit) ||
        !TryMulNonNegative(unit, ctx.dtypeSize, rawPreBytes) || !TryMulNonNegative(unit, kFp32Bytes, rawTmpBytes) ||
        !TryMulNonNegative(aUnit, kFp32Bytes, rawOutBytes) ||
        !TryCeilAlignNonNegative(rawPreBytes, ctx.blockSize, preBytes) ||
        !TryCeilAlignNonNegative(rawTmpBytes, ctx.blockSize, tmpBytes) ||
        !TryCeilAlignNonNegative(rawOutBytes, ctx.blockSize, outBytes)) {
        ctx.arithmeticOverflow = true;
        return false;
    }
    int64_t ubBytes = kCacheBytes;
    for (const int64_t bytes : {preBytes, preBytes, tmpBytes, tmpBytes, outBytes}) {
        if (!TryAddNonNegative(ubBytes, bytes, ubBytes)) {
            ctx.arithmeticOverflow = true;
            return false;
        }
    }
    return ubBytes <= ctx.ubSize;
}

bool ComputeUbSplit(TilingContext& ctx)
{
    int64_t aTarget = ctx.cacheLineSize / ctx.dtypeSize;
    while (aTarget >= 1) {
        if (!ComputeAUbFactor(ctx, aTarget)) {
            return false;
        }
        const bool hasRFactor = ComputeRUbFactor(ctx);
        if (ctx.arithmeticOverflow) {
            return false;
        }
        const bool expandedA = hasRFactor && ExpandAIfRFullyLoaded(ctx);
        if (ctx.arithmeticOverflow) {
            return false;
        }
        if (hasRFactor && expandedA) {
            if (!ComputeRLoopCnt(ctx)) {
                return false;
            }
            ctx.ubGatePassed = FitsUb(ctx);
            if (ctx.arithmeticOverflow) {
                return false;
            }
            ctx.cacheGatePassed = FitsCache(ctx);
            if (ctx.arithmeticOverflow) {
                return false;
            }
            if (ctx.ubGatePassed && ctx.cacheGatePassed) {
                return true;
            }
        }
        if (aTarget == 1) {
            break;
        }
        aTarget = std::max<int64_t>(1, aTarget / 2);
    }
    return false;
}

bool ComputeFusedALoopSplit(TilingContext& ctx)
{
    int64_t outerAProd = 1;
    for (int32_t i = 0; i < ctx.aSplitAxisIdx; i += 2) {
        if (!TryMulNonNegative(outerAProd, ctx.axisShape[static_cast<size_t>(i)], outerAProd)) {
            ctx.arithmeticOverflow = true;
            return false;
        }
    }
    const int64_t splitSize = ctx.axisShape[static_cast<size_t>(ctx.aSplitAxisIdx)];
    ctx.aSplitChunkCnt = CeilDiv(splitSize, ctx.aUbFactor);
    if (!TryMulNonNegative(outerAProd, ctx.aSplitChunkCnt, ctx.aLoopCntTotal)) {
        ctx.arithmeticOverflow = true;
        return false;
    }
    ctx.aSmallCoreLoopCnt = ctx.aLoopCntTotal / ctx.coreNum;
    ctx.aBigCoreCnt = static_cast<int32_t>(ctx.aLoopCntTotal % ctx.coreNum);
    ctx.aBigCoreLoopCnt = ctx.aSmallCoreLoopCnt + (ctx.aBigCoreCnt > 0 ? 1 : 0);
    ctx.usedCoreNum = ctx.aSmallCoreLoopCnt > 0 ? static_cast<int32_t>(ctx.coreNum) : ctx.aBigCoreCnt;
    return true;
}

bool ComputeGroupSplit(TilingContext& ctx)
{
    int64_t totalOuter = 0;
    if (!TryMulNonNegative(ctx.aLoopCntTotal, ctx.rLoopCntTotal, totalOuter)) {
        ctx.arithmeticOverflow = true;
        return false;
    }
    const int64_t perCore = CeilDiv(totalOuter, ctx.coreNum);
    int64_t numBlocks = CeilDiv(totalOuter, perCore);
    int64_t ceilBlocks = 0;
    if (!TryCeilAlignNonNegative(numBlocks, ctx.aLoopCntTotal, ceilBlocks)) {
        ctx.arithmeticOverflow = true;
        return false;
    }
    numBlocks = ceilBlocks <= ctx.coreNum ? ceilBlocks : FloorAlign(numBlocks, ctx.aLoopCntTotal);
    ctx.usedCoreNum = static_cast<int32_t>(numBlocks);
    ctx.rGroupCnt = numBlocks / ctx.aLoopCntTotal;
    ctx.isGroup = true;
    return true;
}

bool ComputeUbSizes(TilingContext& ctx)
{
    int64_t aUnit = 0;
    int64_t rUnit = 0;
    int64_t unit = 0;
    int64_t rawPreBytes = 0;
    int64_t rawPostBytes = 0;
    int64_t rawTmpBytes = 0;
    if (!TryMulNonNegative(ctx.aUbFactorAlign, ctx.innerAProdAlign, aUnit) ||
        !TryMulNonNegative(ctx.rUbFactorAlign, ctx.innerRProdAlign, rUnit) || !TryMulNonNegative(aUnit, rUnit, unit) ||
        !TryMulNonNegative(unit, ctx.dtypeSize, rawPreBytes) || !TryMulNonNegative(aUnit, kFp32Bytes, rawPostBytes) ||
        !TryMulNonNegative(unit, kFp32Bytes, rawTmpBytes) ||
        !TryCeilAlignNonNegative(rawPreBytes, ctx.blockSize, ctx.preReduceUbSize) ||
        !TryCeilAlignNonNegative(rawPostBytes, ctx.blockSize, ctx.postReduceUbSize) ||
        !TryCeilAlignNonNegative(rawTmpBytes, ctx.blockSize, ctx.tmpBufUbSize)) {
        ctx.arithmeticOverflow = true;
        return false;
    }
    return true;
}

void FillTilingData(const TilingContext& ctx, BNTrainingReduceTilingData& tilingData)
{
    tilingData.axisNum = ctx.axisNum;
    for (int32_t i = 0; i < MAX_PATTERN_RANK; ++i) {
        tilingData.axisShape[i] = i < ctx.axisNum ? ctx.axisShape[static_cast<size_t>(i)] : 1;
        tilingData.axisStride[i] = i < ctx.axisNum ? ctx.axisStride[static_cast<size_t>(i)] : 0;
    }
    tilingData.aLoopCntTotal = ctx.aLoopCntTotal;
    tilingData.aSplitChunkCnt = ctx.aSplitChunkCnt;
    tilingData.aBigCoreLoopCnt = ctx.aBigCoreLoopCnt;
    tilingData.aSmallCoreLoopCnt = ctx.aSmallCoreLoopCnt;
    tilingData.aBigCoreCnt = ctx.aBigCoreCnt;
    tilingData.usedCoreNum = ctx.usedCoreNum;
    tilingData.aSplitAxisIdx = ctx.aSplitAxisIdx;
    tilingData.rSplitAxisIdx = ctx.rSplitAxisIdx;
    tilingData.aUbFactor = ctx.aUbFactor;
    tilingData.aUbFactorAlign = ctx.aUbFactorAlign;
    tilingData.rUbFactor = ctx.rUbFactor;
    tilingData.rUbFactorAlign = ctx.rUbFactorAlign;
    tilingData.innerAProd = ctx.innerAProd;
    tilingData.innerAProdAlign = ctx.innerAProdAlign;
    tilingData.innerRProd = ctx.innerRProd;
    tilingData.innerRProdAlign = ctx.innerRProdAlign;
    tilingData.rLoopCntTotal = ctx.rLoopCntTotal;
    tilingData.preReduceUbSize = ctx.preReduceUbSize;
    tilingData.postReduceUbSize = ctx.postReduceUbSize;
    tilingData.tmpBufUbSize = ctx.tmpBufUbSize;
    tilingData.cacheBufUbSize = kCacheBytes;
    tilingData.rGroupCnt = ctx.rGroupCnt;
}

BNTrainingReducePublicResult ComputeEmptyTiling(const BNTrainingReducePublicInputs& inputs,
                                                BNTrainingReduceEmptyKind emptyKind)
{
    BNTrainingReducePublicResult result;
    result.status = BNTrainingReducePublicStatus::SUCCESS;
    result.tilingKey = static_cast<int64_t>(BNTrainingReduceTilingKey::EMPTY);
    result.workspaceSize = inputs.systemWorkspaceSize;
    result.scheduleMode = 0;
    if (emptyKind == BNTrainingReduceEmptyKind::EMPTY_A) {
        result.blockDim = 1;
        return result;
    }

    constexpr int64_t kMaxSingleBufBytes = 64 * 1024;
    constexpr int64_t kMinBytesPerCore = 4 * 1024;
    const int64_t channels = ChannelSize(inputs);
    const int64_t maxFactor = std::min(inputs.ubSize / kFp32Bytes, kMaxSingleBufBytes / kFp32Bytes);
    if (maxFactor <= 0) {
        result.status = BNTrainingReducePublicStatus::TILING_FAILED;
        return result;
    }
    const int64_t minFactor = CeilDiv(kMinBytesPerCore, kFp32Bytes);
    const int64_t factor = std::min(std::min(std::max(minFactor, CeilDiv(channels, inputs.coreNum)), maxFactor),
                                    channels);
    const int64_t loops = CeilDiv(channels, factor);
    const int64_t smallLoops = loops / inputs.coreNum;
    const int32_t bigCores = static_cast<int32_t>(loops % inputs.coreNum);
    const int32_t usedCores = smallLoops > 0 ? static_cast<int32_t>(inputs.coreNum) : bigCores;
    auto& td = result.tilingData;
    td.axisShape[0] = channels;
    td.usedCoreNum = usedCores;
    td.aLoopCntTotal = loops;
    td.aSplitChunkCnt = loops;
    td.aBigCoreLoopCnt = smallLoops + (bigCores > 0 ? 1 : 0);
    td.aSmallCoreLoopCnt = smallLoops;
    td.aBigCoreCnt = bigCores;
    td.aUbFactor = factor;
    int64_t rawPostBytes = 0;
    if (!TryMulNonNegative(factor, kFp32Bytes, rawPostBytes) ||
        !TryCeilAlignNonNegative(rawPostBytes, inputs.blockSize, td.postReduceUbSize)) {
        result.status = BNTrainingReducePublicStatus::TILING_FAILED;
        return result;
    }
    result.blockDim = static_cast<uint32_t>(std::max(usedCores, 1));
    return result;
}

BNTrainingReducePublicResult ComputeAllRoutes(const BNTrainingReducePublicInputs& inputs)
{
    BNTrainingReducePublicResult result;
    result.status = ValidateInputs(inputs);
    if (result.status != BNTrainingReducePublicStatus::SUCCESS) {
        return result;
    }

    const BNTrainingReduceEmptyKind emptyKind = ClassifyEmpty(inputs);
    if (emptyKind != BNTrainingReduceEmptyKind::NORMAL) {
        return ComputeEmptyTiling(inputs, emptyKind);
    }

    TilingContext ctx;
    if (!NormalizePattern(inputs, ctx) || ctx.axisNum < 2 || ctx.axisNum > MAX_PATTERN_RANK || !ComputeUbSplit(ctx)) {
        result.status = BNTrainingReducePublicStatus::TILING_FAILED;
        return result;
    }
    if (!ComputeFusedALoopSplit(ctx)) {
        result.status = BNTrainingReducePublicStatus::TILING_FAILED;
        return result;
    }
    if (ctx.aLoopCntTotal <= ctx.coreNum / 2 && ctx.rLoopCntTotal >= 2) {
        if (!ComputeGroupSplit(ctx)) {
            result.status = BNTrainingReducePublicStatus::TILING_FAILED;
            return result;
        }
    }
    if (!ComputeUbSizes(ctx)) {
        result.status = BNTrainingReducePublicStatus::TILING_FAILED;
        return result;
    }

    result.status = BNTrainingReducePublicStatus::SUCCESS;
    if (ctx.isGroup) {
        if (inputs.deterministic) {
            result.tilingKey = static_cast<int64_t>(ctx.isTailR ?
                                                        BNTrainingReduceTilingKey::DETERMINISTIC_GROUP_TAIL_R :
                                                        BNTrainingReduceTilingKey::DETERMINISTIC_GROUP_TAIL_A);
        } else {
            result.tilingKey = static_cast<int64_t>(ctx.isTailR ? BNTrainingReduceTilingKey::GROUP_TAIL_R :
                                                                  BNTrainingReduceTilingKey::GROUP_TAIL_A);
        }
    } else {
        result.tilingKey = static_cast<int64_t>(ctx.isTailR ? BNTrainingReduceTilingKey::NORMAL_TAIL_R :
                                                              BNTrainingReduceTilingKey::NORMAL_TAIL_A);
    }
    result.blockDim = static_cast<uint32_t>(ctx.usedCoreNum);
    result.workspaceSize = inputs.systemWorkspaceSize;
    if (ctx.isGroup && inputs.deterministic &&
        !TryComputeGroupWorkspaceSize(ctx.rGroupCnt, ChannelSize(inputs), inputs.systemWorkspaceSize,
                                      result.workspaceSize)) {
        result.status = BNTrainingReducePublicStatus::TILING_FAILED;
        return result;
    }
    result.scheduleMode = ctx.isGroup ? 1 : 0;
    FillTilingData(ctx, result.tilingData);
    return result;
}

} // namespace

BNTrainingReducePublicResult ComputeBNTrainingReducePublicTiling(const BNTrainingReducePublicInputs& inputs)
{
    return ComputeAllRoutes(inputs);
}

} // namespace optiling
