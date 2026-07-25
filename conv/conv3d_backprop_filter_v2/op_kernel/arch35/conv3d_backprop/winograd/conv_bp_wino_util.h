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
 * \file conv_bp_wino_util.h
 * \brief
 */

#ifndef CONV_BP_WINO_UTIL_H
#define CONV_BP_WINO_UTIL_H

#include "basic_api/kernel_basic_intf.h"
#include "utils/std/algorithm.h"
#include "op_kernel/math_util.h"

template <typename T>
static constexpr __aicore__ inline uint32_t C0()
{
    return AscendC::DEFAULT_C0_SIZE / sizeof(T);
}

template <typename T>
static constexpr __aicore__ inline uint32_t VL()
{
    return AscendC::VECTOR_REG_WIDTH / sizeof(T);
}

struct HWBox {
    uint32_t hIdx = 0;
    uint32_t wIdx = 0;
    uint32_t hLength = 0;
    uint32_t wLength = 0;
    uint32_t elements = 0;
};

struct HWPad {
    uint16_t hTop = 0;
    uint16_t hBottom = 0;
    uint16_t wLeft = 0;
    uint16_t wRight = 0;
};

template <uint32_t STRIDE, uint32_t WINDOW_SIZE>
class SlideWindows {
public:
    static __aicore__ inline void CalculateSrcBox(const HWBox& tile, uint32_t srcH, uint32_t srcW, uint32_t padH,
                                                  uint32_t padW, HWBox& outSrc, HWPad& outPad)
    {
        // 将tile转换成(src+pad)中坐标[start,end)
        uint32_t startH = tile.hIdx * STRIDE;
        uint32_t startW = tile.wIdx * STRIDE;
        uint32_t endH = (tile.hIdx + tile.hLength - 1) * STRIDE + WINDOW_SIZE;
        uint32_t endW = (tile.wIdx + tile.wLength - 1) * STRIDE + WINDOW_SIZE;

        //(src+pad)中实际非pad区域的坐标[start,end)
        uint32_t startValidH = padH;
        uint32_t startValidW = padW;
        uint32_t endValidH = padH + srcH;
        uint32_t endValidW = padW + srcW;

        // 计算两个区域的相交矩形[start,end)
        uint32_t startSrcH = AscendC::Std::max(startValidH, startH);
        uint32_t startSrcW = AscendC::Std::max(startValidW, startW);
        uint32_t endSrcH = AscendC::Std::min(endH, endValidH);
        uint32_t endSrcW = AscendC::Std::min(endW, endValidW);

        // tile区域和非padding区域不相交,整个tile都是在padding区域内
        if (startSrcH >= endSrcH || startSrcW >= endSrcW) {
            outSrc.hIdx = 0;
            outSrc.wIdx = 0;
            outSrc.hLength = 0;
            outSrc.wLength = 0;
            outSrc.elements = 0;
            outPad.hTop = 0;
            outPad.hBottom = endH - startH;
            outPad.wLeft = 0;
            outPad.wRight = endW - startW;
        } else {
            outSrc.hIdx = startSrcH - padH;
            outSrc.wIdx = startSrcW - padW;
            outSrc.hLength = endSrcH - startSrcH;
            outSrc.wLength = endSrcW - startSrcW;
            outSrc.elements = outSrc.hLength * outSrc.wLength;
            outPad.hTop = startSrcH - startH;
            outPad.hBottom = endH - endSrcH;
            outPad.wLeft = startSrcW - startW;
            outPad.wRight = endW - endSrcW;
        }
    }

    static __aicore__ inline uint32_t SrcLength2Tiles(const uint32_t srcLength)
    {
        if constexpr (WINDOW_SIZE == STRIDE) {
            // 滑窗大小和stride相同时可以简化下处理逻辑
            return Ops::Base::CeilDiv(srcLength, WINDOW_SIZE);
        }
        return Ops::Base::CeilDiv(srcLength > WINDOW_SIZE ? srcLength - WINDOW_SIZE : 0, STRIDE) + 1;
    }

    static constexpr __aicore__ inline uint32_t Tiles2SrcLength(const uint32_t tiles)
    {
        if constexpr (WINDOW_SIZE == STRIDE) {
            // 滑窗大小和stride相同时可以简化下处理逻辑
            return tiles * WINDOW_SIZE;
        }
        // tile长度转换成fmap长度
        return tiles == 0 ? 0 : (tiles - 1) * STRIDE + WINDOW_SIZE;
    }

    static constexpr __aicore__ inline uint32_t Tiles2Elements(const uint32_t tiles)
    {
        // tile里的元素个数
        return WINDOW_SIZE * WINDOW_SIZE * tiles;
    }

    static __aicore__ inline uint32_t Tiles2Size(const uint32_t tiles)
    {
        // tile里单边元素个数
        return WINDOW_SIZE * tiles;
    }
};

static constexpr uint32_t F23_TRANSFORM_TILE_SIZE_4 = 4;
static constexpr uint32_t F23_TRANSFORM_TILE_ELEMENTS_16 = 16;

static constexpr __aicore__ inline uint32_t TileUnfoldElements(uint32_t tiles)
{
    return tiles * F23_TRANSFORM_TILE_ELEMENTS_16;
}

static constexpr __aicore__ inline uint32_t TileUnfoldSize(uint32_t tiles) { return tiles * F23_TRANSFORM_TILE_SIZE_4; }

namespace ConstexprMaths {
// 用于编译期计算的特殊函数，很多库函数没有constexpr标记，没办法赋值给constexpr对象

template <typename T>
static constexpr __aicore__ inline T Max(const T src0, const T src1)
{
    return (src0 > src1) ? src0 : src1;
}

template <typename T>
static constexpr __aicore__ inline T Min(const T src0, const T src1)
{
    return (src0 < src1) ? src0 : src1;
}

static constexpr __aicore__ inline uint32_t AlignUp(const uint32_t a, const uint32_t b)
{
    return AscendC::AlignUp(a, b);
}

static constexpr __aicore__ inline uint32_t AlignDown(const uint32_t a, const uint32_t b) { return a / b * b; }

static constexpr __aicore__ inline uint32_t CeilDiv(const uint32_t a, const uint32_t b)
{
    return AscendC::ConstCeil(a, b);
}
} // namespace ConstexprMaths

namespace BlockConfig {
enum InputTensor {
    FMAP,
    DY,
};

template <uint16_t SingleShapeCoutVal, uint16_t SingleShapeCinVal, uint16_t SingleTransformC1Val,
          uint16_t SingleShapeTileHVal, uint16_t SingleShapeTileWVal, uint8_t SingleTransformBufCntVal,
          uint16_t SingleShapeResidentCValue, InputTensor ResidentTargetValue, uint8_t InvTransformBufCntValue,
          uint16_t SingleShapeInvTransformCoutVal>
struct Tiling {
    static constexpr uint16_t SingleShapeCout = SingleShapeCoutVal;
    static constexpr uint16_t SingleShapeCin = SingleShapeCinVal;
    static constexpr uint16_t SingleTransformC1 = SingleTransformC1Val;
    static constexpr uint16_t SingleShapeTileH = SingleShapeTileHVal;
    static constexpr uint16_t SingleShapeTileW = SingleShapeTileWVal;
    static constexpr uint8_t SingleTransformBufCnt = SingleTransformBufCntVal;
    static constexpr uint16_t SingleShapeResidentC = SingleShapeResidentCValue;
    static constexpr InputTensor ResidentTarget = ResidentTargetValue;
    static constexpr uint8_t InvTransformBufCnt = InvTransformBufCntValue;
    static constexpr uint16_t SingleShapeInvTransformCout = SingleShapeInvTransformCoutVal;
};

template <typename TilingT>
static constexpr __aicore__ inline uint16_t SingleShapeCout()
{
    return TilingT::SingleShapeCout;
}

template <typename TilingT>
static constexpr __aicore__ inline uint16_t SingleShapeCin()
{
    return TilingT::SingleShapeCin;
}

template <typename TilingT, InputTensor TensorType>
static constexpr __aicore__ inline uint16_t SingleShapeC()
{
    if constexpr (TensorType == FMAP) {
        return SingleShapeCin<TilingT>();
    } else if constexpr (TensorType == DY) {
        return SingleShapeCout<TilingT>();
    }
}

template <typename TilingT>
static constexpr __aicore__ inline uint16_t SingleTransformC1()
{
    return TilingT::SingleTransformC1;
}

template <typename TilingT>
static constexpr __aicore__ inline uint16_t SingleShapeTileH()
{
    return TilingT::SingleShapeTileH;
}

template <typename TilingT>
static constexpr __aicore__ inline uint16_t SingleShapeTileW()
{
    return TilingT::SingleShapeTileW;
}

template <typename TilingT>
static constexpr __aicore__ inline uint16_t SingleShapeTileHW()
{
    return SingleShapeTileH<TilingT>() * SingleShapeTileW<TilingT>();
}

template <typename TilingT>
static constexpr __aicore__ inline uint8_t SingleTransformBufCnt()
{
    return TilingT::SingleTransformBufCnt;
}

template <typename TilingT>
static constexpr __aicore__ inline uint16_t SingleShapeResidentC()
{
    return TilingT::SingleShapeResidentC;
}

template <typename TilingT>
static constexpr __aicore__ inline InputTensor ResidentTarget()
{
    return TilingT::ResidentTarget;
}

template <typename TilingT>
static constexpr __aicore__ inline uint8_t InvTransformBufCnt()
{
    return TilingT::InvTransformBufCnt;
}

template <typename TilingT>
static constexpr __aicore__ inline uint16_t SingleShapeInvTransformCout()
{
    return TilingT::SingleShapeInvTransformCout;
}
} // namespace BlockConfig

struct CoutCinRange {
    uint32_t coutIdx = 0;
    uint32_t cinIdx = 0;
    uint32_t coutLength = 0;
    uint32_t cinLength = 0;

    template <BlockConfig::InputTensor t>
    __aicore__ inline uint32_t GetIdx() const
    {
        if constexpr (t == BlockConfig::InputTensor::FMAP) {
            return cinIdx;
        } else if constexpr (t == BlockConfig::InputTensor::DY) {
            return coutIdx;
        }
    }

    template <BlockConfig::InputTensor t>
    __aicore__ inline uint32_t GetLen() const
    {
        if constexpr (t == BlockConfig::InputTensor::FMAP) {
            return cinLength;
        } else if constexpr (t == BlockConfig::InputTensor::DY) {
            return coutLength;
        }
    }

    __aicore__ inline bool NotEmpty() const { return coutLength != 0 && cinLength != 0; }
};

static inline constexpr uint32_t __aicore__ AivNumInBlock()
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    return 2;
#else
    return 1;
#endif
}

static inline uint32_t __aicore__ AivCoreId()
{
    // use it in aiv only
    return GetBlockIdx();
}

static inline uint32_t __aicore__ AicCoreId()
{
    if ASCEND_IS_AIC {
        return GetBlockIdx();
    }
    if ASCEND_IS_AIV {
        return GetBlockIdx() / AivNumInBlock();
    }
}

static inline uint32_t __aicore__ AivNums() { return GetBlockNum() * AivNumInBlock(); }

// 余数均摊切分实现
//
//  比如将5个任务切出3份
//  会先算出base=5/3=1 , remainder=5%3=2
//  那意味着每个切分的基础数量为5，然后将余数均摊到每个切分上，也就是将前2个切分额外加1，切出2,2,1，
//
//  余数均摊可以保证切分均衡，两个切分间最多差1
//
//  相比直接CeilDiv的话会更均衡一些
//  比如5个任务4个cut余数均摊时2,1,1,1，而CeilDiv切的话就是2,2,1,0
//
class RemainderDistributionSpliter {
public:
    __aicore__ inline RemainderDistributionSpliter(uint32_t totalTask, uint32_t totalSplit)
        // 为了codecheck加个除0保护，但是上层不应该传0到这里
        : base_(totalSplit == 0 ? 0 : totalTask / totalSplit),
          remainer_(totalSplit == 0 ? 0 : totalTask - base_ * totalSplit)
    {}

    template <typename T>
    __aicore__ inline void GetSplit(uint32_t splitIdx, T& outOffset, T& outLength) const
    {
        static_assert(Std::is_same_v<T, uint16_t> || Std::is_same_v<T, uint32_t>);
        // 前remainer_个切分额外补1，均摊余数
        outLength = base_ + (splitIdx < remainer_ ? 1 : 0);
        outOffset = splitIdx * base_ + Std::min(splitIdx, remainer_);
    }

    __aicore__ inline uint32_t GetMaxLength() const { return base_ + (remainer_ > 0 ? 1 : 0); }

private:
    const uint32_t base_;
    const uint32_t remainer_;
};

#endif // CONV_BP_WINO_UTIL_H
