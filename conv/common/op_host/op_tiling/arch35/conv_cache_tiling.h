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
 * \file conv_cache_tiling .h
 * \brief
 */
#ifndef CONV_OP_TILING_CONV_CACHE_TILING_H
#define CONV_OP_TILING_CONV_CACHE_TILING_H
#include <unordered_map>
#include <string>
#include <mutex>
#include "conv_base_utils.h"
#include "op_host/tiling_base.h"

namespace optiling {
namespace conv_ops_tiling {
struct ConvInputArgs {
    ge::DataType inputDtype = ge::DataType::DT_UNDEFINED;
    ge::DataType weightDtype = ge::DataType::DT_UNDEFINED;
    ge::DataType outputDtype = ge::DataType::DT_UNDEFINED;
    ge::DataType biasDtype = ge::DataType::DT_UNDEFINED;
    int64_t inputShapeN = 1;
    int64_t inputShapeH = 1;
    int64_t inputShapeD = 1;
    int64_t inputShapeW = 1;
    int64_t weightShapeN = 1;
    int64_t weightShapeC = 1;
    int64_t weightShapeD = 1;
    int64_t weightShapeH = 1;
    int64_t weightShapeW = 1;
    int64_t outputShapeD = 1;
    int64_t outputShapeH = 1;
    int64_t outputShapeW = 1;
    ge::Format inputFormat = ge::FORMAT_NCHW;
    ge::Format weightFormat = ge::FORMAT_NCHW;
    ge::Format outputFormat = ge::FORMAT_NCHW;
    int64_t groups = 1;
    int64_t strideD = 1;
    int64_t strideH = 1;
    int64_t strideW = 1;
    int64_t dilationD = 1;
    int64_t dilationH = 1;
    int64_t dilationW = 1;
    int64_t padHead = 0;
    int64_t padTail = 0;
    int64_t padTop = 0;
    int64_t padBottom = 0;
    int64_t padLeft = 0;
    int64_t padRight = 0;
    bool biasFlag = false;
    bool hf32Flag = false;
    uint8_t dual_output = 0; // 是否双输出
    uint8_t quantMode0 = 0; // 通道1的量化模式
    uint8_t reluMode0 = 0; // 通道1的relu模式
    uint8_t clipMode0 = 0; //  通道1的clip模式
    uint8_t scaleFlag0 = 0; // 通道1是否有scale的标记
    uint8_t quantMode1 = 0; // 通道2的量化模式
    uint8_t reluMode1 = 0; // 通道2的relu模式
    uint8_t clipMode1 = 0; // 通道2的clip模式
    uint8_t scaleFlag1 = 0; // 通道2是否有scale的标记
    ge::Format output1Format = ge::FORMAT_NCHW;
    ge::DataType output1Dtype = ge::DataType::DT_UNDEFINED;
    int64_t output1ShapeN = 1;
    int64_t output1ShapeC = 1;
    int64_t output1ShapeH = 1;
    int64_t output1ShapeW = 1;

    // 重载 == 运算符，用于比较
    bool operator==(const ConvInputArgs& other) const {
        return inputDtype == other.inputDtype &&
               weightDtype == other.weightDtype &&
               outputDtype == other.outputDtype &&
               biasDtype == other.biasDtype &&
               inputShapeN == other.inputShapeN &&
               inputShapeH == other.inputShapeH &&
               inputShapeD == other.inputShapeD &&
               inputShapeW == other.inputShapeW &&
               weightShapeN == other.weightShapeN &&
               weightShapeC == other.weightShapeC &&
               weightShapeD == other.weightShapeD &&
               weightShapeH == other.weightShapeH &&
               weightShapeW == other.weightShapeW &&
               outputShapeD == other.outputShapeD &&
               outputShapeH == other.outputShapeH &&
               outputShapeW == other.outputShapeW &&
               inputFormat == other.inputFormat &&
               weightFormat == other.weightFormat &&
               outputFormat == other.outputFormat &&
               groups == other.groups &&
               strideD == other.strideD &&
               strideH == other.strideH &&
               strideW == other.strideW &&
               dilationD == other.dilationD &&
               dilationH == other.dilationH &&
               dilationW == other.dilationW &&
               padHead == other.padHead &&
               padTail == other.padTail &&
               padTop == other.padTop &&
               padBottom == other.padBottom &&
               padLeft == other.padLeft &&
               padRight == other.padRight &&
               biasFlag == other.biasFlag &&
               hf32Flag == other.hf32Flag &&
               dual_output == other.dual_output &&
               quantMode0 == other.quantMode0 &&
               reluMode0 == other.reluMode0 &&
               clipMode0 == other.clipMode0 &&
               scaleFlag0 == other.scaleFlag0 &&
               quantMode1 == other.quantMode1 &&
               reluMode1 == other.reluMode1 &&
               clipMode1 == other.clipMode1 &&
               scaleFlag1 == other.scaleFlag1 &&
               output1Format == other.output1Format &&
               output1Dtype == other.output1Dtype &&
               output1ShapeN == other.output1ShapeN &&
               output1ShapeC == other.output1ShapeC &&
               output1ShapeH == other.output1ShapeH &&
               output1ShapeW == other.output1ShapeW;
    }
};

struct Conv2dCacheTilingData {
    uint64_t singleCoreBatch = 0;
    uint64_t singleCoreHo = 0;
    uint64_t singleCoreWo = 0;
    uint32_t singleCoreCi = 0;
    uint32_t singleCoreCo = 0;
    uint64_t orgHixWi = 0;
    uint32_t hoL1 = 0;
    uint32_t woL1 = 0;
    uint32_t kAL1 = 0;
    uint32_t kBL1 = 0;
    uint32_t nBL1 = 0;
    uint32_t khL1 = 0;
    uint32_t kwL1 = 0;
    uint32_t hoL0 = 0;
    uint32_t woL0 = 0;
    uint32_t kL0 = 0;
    uint32_t nL0 = 0;
    uint32_t pBufferFlag = 0;
    uint32_t enlarge = 0;
    uint32_t singleCoreGroups = 0;
    uint32_t singleCoreGroupOpt = 0;
    uint32_t bUbNStep = 0;
    uint32_t bUbKStep = 0;
    uint32_t khUb = 0;
    uint32_t kwUb = 0;
    uint32_t kernelHxkernelW = 0;
    uint32_t kernelHxkernelWxkernelD = 0;
    uint32_t aL1SpaceSize = 0;
    uint32_t multiNBL1 = 0;
    uint32_t cinAInCore = 0;
    uint32_t cinATailInCore = 0;
    uint32_t cinBInCore = 0;
    uint32_t cinBTailInCore = 0;
    uint32_t mStep = 0;
    uint32_t kStep = 0;
    uint32_t nStep = 0;
    uint32_t fmapKStride = 0;
    uint32_t weightKStride = 0;
    uint32_t cinOffsetBlockInGM = 0;
    uint32_t coutOffsetBlock = 0;
    uint32_t nL1DivBlockSize = 0;
    uint8_t iterateMNOrder = 0;
    uint8_t biasFullLoadFlag = 0;
    uint8_t fixpParamsFullLoadFlag = 0;
    uint8_t hf32Enable = 0;
    uint8_t hf32TransMode = 0;
    uint32_t innerBatch = 0;
    uint32_t batchDim = 0;
    uint32_t groupDim = 0;
    uint32_t nDim = 0;
    uint32_t hoDim = 0;
    uint32_t woDim = 0;
    uint32_t cinOpt = 0;
    uint32_t coutOpt = 0;
    uint32_t groupOpt = 0;
    bool enableC04Flag = false;
    bool mSplitModeFlag = false;
};

struct Conv3dCacheTilingData {
    // uint64_t fields
    uint64_t orgDo = 0;
    uint64_t orgHo = 0;
    uint64_t orgWo = 0;
    uint64_t orgDi = 0;
    uint64_t orgHi = 0;
    uint64_t orgWi = 0;
    uint64_t singleCoreBatch = 0;
    uint64_t singleCoreDo = 0;
    uint64_t singleCoreM = 0;
    uint64_t singleCoreWo = 0;
    uint64_t singleCoreHo = 0;
    uint64_t kL0xorgCoAlignN0 = 0;
    uint64_t kernelHxkernelW = 0;
    uint64_t cin1xOriHixOriWixk0 = 0;
    uint64_t oriHixOriWixk0 = 0;
    uint64_t oriWixk0 = 0;
    uint64_t orgHixWi = 0;
    uint64_t orgHoxWo = 0;

    // uint32_t fields
    uint32_t orgCi = 0;
    uint32_t kernelD = 0;
    uint32_t kernelH = 0;
    uint32_t kernelW = 0;
    uint32_t singleCoreCo = 0;
    uint32_t orgCo = 0;
    uint32_t singleCoreCi = 0;
    uint32_t singleCoreGroups = 0;
    uint32_t singleCoreGroupOpt = 0;
    uint32_t groups = 0;
    uint32_t enlarge = 0;
    uint32_t strideH = 0;
    uint32_t strideW = 0;
    uint32_t strideD = 0;
    uint32_t dilationH = 0;
    uint32_t dilationW = 0;
    uint32_t dilationD = 0;
    uint32_t padHead = 0;
    uint32_t padTail = 0;
    uint32_t padTop = 0;
    uint32_t padBottom = 0;
    uint32_t padLeft = 0;
    uint32_t padRight = 0;
    uint32_t mL0 = 0;
    uint32_t woL0 = 0;
    uint32_t kL0 = 0;
    uint32_t nL0 = 0;
    uint32_t kAL1 = 0;
    uint32_t kAL1Tail = 0;
    uint32_t kBL1 = 0;
    uint32_t kBL1Tail = 0;
    uint32_t nBL1 = 0;
    uint32_t mAL1 = 0;
    uint32_t woL1 = 0;
    uint32_t hoL0 = 0;
    uint32_t hoL1 = 0;
    uint32_t KBL1Divk0 = 0;
    uint32_t KBL1TailDivk0 = 0;
    uint32_t nBL1DivnL0 = 0;
    uint32_t mAL1DivmL0 = 0;
    uint32_t fmapKStride = 0;
    uint32_t weightKStride = 0;
    uint32_t cinOffsetBlockInGM = 0;
    uint32_t coutOffsetBlock = 0;
    uint32_t nL1DivBlockSize = 0;
    uint32_t cin1InAL1 = 0;
    uint32_t cin1InAL1Tail = 0;
    uint32_t cinBInCore = 0;
    uint32_t cinBTailInCore = 0;
    uint32_t cinAInCore = 0;
    uint32_t cinATailInCore = 0;
    uint32_t nL0xk0 = 0;
    uint32_t mStep = 0;
    uint32_t kStep = 0;
    uint32_t nStep = 0;
    uint32_t aL1SpaceSize = 0;
    uint32_t multiNBL1 = 0;
    uint32_t pBufferFlag = 0;
    uint32_t groupOpt = 0;
    uint32_t cinOpt = 0;
    uint32_t coutOpt = 0;
    uint32_t mUB = 0;
    uint32_t nUB = 0;
    uint32_t scaleAndBiasLoadType = 0;
    uint32_t workspaceSize = 0;
    uint32_t kernelHxkernelWxkernelD = 0;

    // int8_t fields
    int8_t offsetx = 0;
    int8_t roundMode = 0;

    // uint8_t fields
    uint8_t hasBias = 0;
    uint8_t hasScale = 0;
    uint8_t bl1FullLoad = 0;
    uint8_t al1FullLoad = 0;
    uint8_t bl1BypassFlag = 0;
    uint8_t iterateMNOrder = 0;
    uint8_t biasFullLoadFlag = 0;
    uint8_t fixpParamsFullLoadFlag = 0;
    uint8_t hf32Enable = 0;
    uint8_t hf32TransMode = 0;
    uint8_t quantType = 0;
    uint8_t resvered1 = 0;
    uint8_t resvered2 = 0;
    uint8_t resvered3 = 0;

    uint32_t batch = 0;
    uint32_t batchDim = 0;
    uint32_t doDim = 0;
    uint32_t mDim = 0;
    uint32_t wDim = 0;
    uint32_t nDim = 0;
    uint32_t groupDim = 0;
    uint32_t hoDim = 0;
    bool mSplitModeFlag = false;
};
/*
Use a constant (0x9e3779b9, also known as the golden ratio) and bit shifts to combine
the current seed with the hash of v.
The bit shifts (6 to the left and 2 to the right) help to spread the bits of the seed,
so that each bit of the seed can affect each bit of the result.
This helps to produce a more uniform distribution of hash values and reduce hash collisions.
*/
constexpr size_t LEFT_SHIFT_BITS = 6;
constexpr size_t RIGHT_SHIFT_BITS = 2;
template <typename T>
inline void HashCombine(size_t& seed, const T& v)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << LEFT_SHIFT_BITS) + (seed >> RIGHT_SHIFT_BITS);
}

struct ConvInputArgsHash {
    size_t operator()(const ConvInputArgs& args) const
    {
        size_t hash_value = 0;
        HashCombine(hash_value, args.inputDtype);
        HashCombine(hash_value, args.weightDtype);
        HashCombine(hash_value, args.outputDtype);
        HashCombine(hash_value, args.biasDtype);
        HashCombine(hash_value, args.inputShapeN);
        HashCombine(hash_value, args.inputShapeH);
        HashCombine(hash_value, args.inputShapeD);
        HashCombine(hash_value, args.inputShapeW);
        HashCombine(hash_value, args.weightShapeN);
        HashCombine(hash_value, args.weightShapeC);
        HashCombine(hash_value, args.weightShapeD);
        HashCombine(hash_value, args.weightShapeH);
        HashCombine(hash_value, args.weightShapeW);
        HashCombine(hash_value, args.outputShapeD);
        HashCombine(hash_value, args.outputShapeH);
        HashCombine(hash_value, args.outputShapeW);
        HashCombine(hash_value, args.inputFormat);
        HashCombine(hash_value, args.weightFormat);
        HashCombine(hash_value, args.outputFormat);
        HashCombine(hash_value, args.groups);
        HashCombine(hash_value, args.strideD);
        HashCombine(hash_value, args.strideH);
        HashCombine(hash_value, args.strideW);
        HashCombine(hash_value, args.dilationD);
        HashCombine(hash_value, args.dilationH);
        HashCombine(hash_value, args.dilationW);
        HashCombine(hash_value, args.padHead);
        HashCombine(hash_value, args.padTail);
        HashCombine(hash_value, args.padTop);
        HashCombine(hash_value, args.padBottom);
        HashCombine(hash_value, args.padLeft);
        HashCombine(hash_value, args.padRight);
        HashCombine(hash_value, args.biasFlag);
        HashCombine(hash_value, args.hf32Flag);
        HashCombine(hash_value, args.dual_output);
        HashCombine(hash_value, args.quantMode0);
        HashCombine(hash_value, args.reluMode0);
        HashCombine(hash_value, args.clipMode0);
        HashCombine(hash_value, args.scaleFlag0);
        HashCombine(hash_value, args.quantMode1);
        HashCombine(hash_value, args.reluMode1);
        HashCombine(hash_value, args.clipMode1);
        HashCombine(hash_value, args.scaleFlag1);
        return hash_value;
    }
};

constexpr size_t MAX_CACHE_SIZE = 4096;

template <typename Tiling>
class ConvTilingCache
{
public:
    ConvTilingCache() = default;
    virtual ~ConvTilingCache() = default;
    ConvTilingCache(const ConvTilingCache &) = delete;
    ConvTilingCache& operator=(const ConvTilingCache&) = delete;
    ConvTilingCache(const ConvTilingCache &&) = delete;
    ConvTilingCache& operator=(const ConvTilingCache &&) = delete;

    virtual bool GetCachedTiling(const ConvInputArgs& ConvInputArgs, Tiling& tiling);
    virtual bool AddCachedTiling(const ConvInputArgs& ConvInputArgs, const Tiling& tiling);
    size_t GetCacheSize() {
        std::lock_guard<std::mutex> lock(mutex_);
        return cachedTiling.size();
    }

    NpuArch GetSocVersion()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return npuArch_;
    }

    void SetSocVersion(NpuArch npuArch, ConvTilingParseInfo convTilingParseInfo)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        npuArch_ = std::move(npuArch);
        convTilingParseInfo_ = std::move(convTilingParseInfo);
    }

    ConvTilingParseInfo* GetPlatFormInfo()
    {
        return &convTilingParseInfo_;
    }

protected:
    std::unordered_map<ConvInputArgs, Tiling, ConvInputArgsHash> cachedTiling;
    const size_t cacheSize = MAX_CACHE_SIZE;
    std::mutex mutex_;
    NpuArch npuArch_ = NpuArch::DAV_RESV;
    ConvTilingParseInfo convTilingParseInfo_;
};

template <typename Tiling>
bool ConvTilingCache<Tiling>::GetCachedTiling(const ConvInputArgs& ConvInputArgs, Tiling& tiling)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cachedTiling.find(ConvInputArgs);
    if (it != cachedTiling.end()) {
        tiling = it->second;
        return true;
    }
    return false;
}

template <typename Tiling>
bool ConvTilingCache<Tiling>::AddCachedTiling(const ConvInputArgs& ConvInputArgs, const Tiling& tiling)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (cachedTiling.size() < cacheSize && cachedTiling.find(ConvInputArgs) == cachedTiling.end()) {
        cachedTiling[ConvInputArgs] = tiling;
        return true;
    }
    return false;
}

}
}
#endif