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
 * \file pp_matmul_common_tiling.h
 * \brief
 */

#ifndef __OP_HOST_PPMAT_MUL_COMMMON_TILING_H__
#define __OP_HOST_PPMAT_MUL_COMMMON_TILING_H__


#include <cmath>
#include "pp_matmul_info.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace pp_matmul {

constexpr uint64_t FP16_SIZE = 2;
constexpr uint64_t FP32_SIZE = 4;
constexpr uint64_t BLOCK_SIZE = 16;
constexpr uint64_t BLOCK_SIZE_INT8_K = 32;
constexpr uint64_t BASE_BLOCK_STEP = 2;
constexpr uint64_t AXES_ALIGN_SIZE = 512;
constexpr uint64_t AXES_ALIGN_SIZE_INT8 = 256;
constexpr uint64_t ND_SHAPE_SIZE = 2;
constexpr uint64_t NZ_SHAPE_SIZE = 4;
constexpr uint64_t CUBE_BLOCK_SIZE = 256;
constexpr uint64_t CUBE_BLOCK_SIZE_INT8 = 512;
constexpr uint64_t L1AB_PINGPONG_BUFFER_SIZE = 262144;
constexpr uint64_t L0AB_PINGPONG_BUFFER_SIZE_INT8 = 262144; // 131072 * 2 = 256 KB
constexpr uint64_t L0AB_PINGPONG_BUFFER_SIZE_FP16 = 131072; // 128 KB
constexpr uint64_t L1AB_PINGPONG_BUFFER_SIZE_INT8_SPARSE = 163840; // 160 * 1024
constexpr uint64_t UB_LIMIT_SIZE_910A = 131072; // 128 * 1024

template <uint64_t DIV>
inline uint64_t CeilDiv(uint64_t num)
{
    if (DIV == 0UL || num + DIV - 1UL < num) {
        return num;
    }
    return (num + DIV - 1UL) / DIV;
}

inline uint64_t CeilDiv(uint64_t dividend, uint64_t divisor)
{
    if (divisor == 0UL || dividend + divisor - 1UL < dividend) {
        return dividend;
    }
    return (dividend + divisor - 1UL) / divisor;
}

template <uint64_t RND>
inline uint64_t Round(uint64_t num)
{
    if (RND == 0UL || num + RND - 1UL < num) {
        return num;
    }
    return (num + RND - 1UL) / RND * RND;
}

inline uint64_t RoundUp(uint64_t num, uint64_t rnd)
{
    if (rnd == 0UL || num + rnd - 1UL < num) {
        return num;
    }
    return (num + rnd - 1UL) / rnd * rnd;
}

inline int64_t RoundUp(int64_t num, int64_t rnd)
{
    if (rnd == 0UL || static_cast<int64_t>(num + rnd - 1) < num) {
        return num;
    }
    return static_cast<int64_t>((num + rnd - 1) / rnd * rnd);
}

inline uint64_t RoundDown(uint64_t num, uint64_t rnd)
{
    if (rnd == 0UL) {
        return 0UL;
    }
    return num / rnd * rnd;
}

inline uint64_t GetN0TilingLimit(bool compressFlag, uint64_t tilingN, const platform_ascendc::SocVersion &platformType)
{
    if (compressFlag) {
        return std::min(tilingN * BLOCK_SIZE, AXES_ALIGN_SIZE_INT8);
    } else {
        return (platformType == platform_ascendc::SocVersion::ASCEND310P || platformType == platform_ascendc::SocVersion::ASCEND910)
                   ? AXES_ALIGN_SIZE
                   : AXES_ALIGN_SIZE_INT8;
    }
}

template <typename OpShareType>
inline uint64_t GetN0TilingInit(const OpShareType &opShape, bool compressFlag,
                                                               uint64_t tilingN)
{
    const uint64_t RND = 16UL;
    if (compressFlag) {
        if (tilingN * BLOCK_SIZE > opShape.n) {
            return Round<RND>(opShape.n);
        } else {
            return tilingN * BLOCK_SIZE;
        }
    } else {
        return BLOCK_SIZE;
    }
}

template <bool PRI_FLAG>
inline bool IsExceedTilingLimit(uint64_t axes0, uint64_t priAxes0,
                                uint64_t n0TilingLimit, platform_ascendc::SocVersion platformType,
                                uint64_t basicBlockSize)
{
    return (PRI_FLAG && axes0 > n0TilingLimit) || (!PRI_FLAG && priAxes0 > n0TilingLimit) ||
           (platformType == platform_ascendc::SocVersion::ASCEND910 && basicBlockSize > UB_LIMIT_SIZE_910A);
}

template <bool PRI_FLAG, typename OpShareType>
inline void SetOpShapeAxesInfo(OpShareType &opShape, uint64_t priAxes0, uint64_t axes0)
{
    opShape.m0 = PRI_FLAG ? priAxes0 : axes0;
    opShape.n0 = PRI_FLAG ? axes0 : priAxes0;
}

template <typename HardwareType, typename OpShapeType, typename MatMulInfoType>
inline float CostFunc(const HardwareType &hwInfor, OpShapeType &shape, const MatMulInfoType &mmInfo)
{
    float aCoef = 1;
    float bCoef = 1;
    float bwCoef = static_cast<float>(hwInfor.l2BandWidth) / static_cast<float>(hwInfor.hbmBandWidth);
    uint64_t mLoop = CeilDiv(shape.m, shape.m0);
    uint64_t nLoop = CeilDiv(shape.n, shape.n0);
    if (mLoop == 0UL || nLoop == 0UL) {
        return 1;
    }
    uint64_t coreNeed = shape.batchSize * mLoop * nLoop;
    uint64_t blockDim = std::min(coreNeed, hwInfor.coreNum);
    uint64_t mOnce = blockDim < nLoop ? shape.m0 : blockDim / nLoop * shape.m0;
    uint64_t nOnce = blockDim < nLoop ? hwInfor.coreNum * shape.n0 : shape.n;
    if (mOnce * shape.k * mmInfo.inDtype > hwInfor.l2Size) {
        aCoef = bwCoef;
    }
    if (nOnce * shape.k * mmInfo.inDtype > hwInfor.l2Size) {
        bCoef = bwCoef;
    }
    return 1 / (aCoef * static_cast<float>(shape.n0)) + 1 / (bCoef * static_cast<float>(shape.m0));
}

// OpShareType is OpShape, TilingType is PpTilingData, HardwareType is  HardwareType, MatMulInfoType is MatMulInfo
template <bool PRI_FLAG, typename OpShareType, typename TilingType, typename HardwareType, typename MatMulInfoType>
void TilingFunc(OpShareType &opShape, TilingType &tilingParam, const HardwareType &hwInfor,
                const MatMulInfoType &mmInfo, bool compressFlag = false, const uint64_t tilingN = 1)
{
    float costMin = 1;
    const float CONST_2 = 2.0;
    const uint64_t CONST_16 = 16UL;
    uint64_t roundBase =
        static_cast<uint64_t>(pow(2, ceil(log(CeilDiv(PRI_FLAG ? opShape.n : opShape.m, CONST_16)))) * CONST_16);
    uint64_t priAxes = RoundUp(PRI_FLAG ? opShape.m : opShape.n, CONST_16);
    uint64_t axes = RoundUp(PRI_FLAG ? opShape.n : opShape.m, roundBase);
    float axes0Max = static_cast<float>(AXES_ALIGN_SIZE) / mmInfo.inDtype;
    auto platformType = hwInfor.socVersion;
    if (mmInfo.isInt8 && (platformType == platform_ascendc::SocVersion::ASCEND310P || platformType == platform_ascendc::SocVersion::ASCEND910)) {
        axes0Max /= CONST_2;
    }
    uint64_t n0TilingInit = GetN0TilingInit(opShape, compressFlag, tilingN);
    uint64_t n0TilingLimit = GetN0TilingLimit(compressFlag, tilingN, platformType);
    uint64_t priAxes0Init = PRI_FLAG ? BLOCK_SIZE : n0TilingInit;
    uint64_t axes0Init = PRI_FLAG ? n0TilingInit : BLOCK_SIZE;
    bool isAscend310P = platformType == platform_ascendc::SocVersion::ASCEND310P;
    for (uint64_t priAxes0 = priAxes0Init; priAxes0 <= priAxes && priAxes0 <= axes0Max; priAxes0 *= BASE_BLOCK_STEP) {
        for (uint64_t axes0 = axes0Init; axes0 <= axes && axes0 <= axes0Max; axes0 *= BASE_BLOCK_STEP) {
            uint64_t basicBlockSize = priAxes0 * axes0 * FP32_SIZE;
            if (basicBlockSize > hwInfor.l0cSize) {
                continue;
            }
            if (mmInfo.isInt8 &&
                IsExceedTilingLimit<PRI_FLAG>(axes0, priAxes0, n0TilingLimit, platformType, basicBlockSize)) {
                continue;
            }
            SetOpShapeAxesInfo<PRI_FLAG>(opShape, priAxes0, axes0);
            float cost = CostFunc<HardwareType, OpShareType, MatMulInfoType>(hwInfor, opShape, mmInfo);
            if (cost >= costMin) {
                continue;
            }
            costMin = cost;
            tilingParam.SetBaseOp(hwInfor.coreNum, hwInfor.l0cSize, opShape.m0, opShape.n0, mmInfo, isAscend310P);
        }
    }
}

template <typename PpTilingDataType>
uint64_t Swizzl(PpTilingDataType &tilingData)
{
    uint64_t swizzlDirect = 0UL;
    uint64_t swizzlCount = 1UL;
    float m0 = tilingData.opShape.m0;
    float n0 = tilingData.opShape.n0;
    float m = tilingData.opShape.m;
    float k = tilingData.opShape.k;
    float n = tilingData.opShape.n;
    float mincost = m * k + k * n;

    for (uint32_t i = 1; i <= tilingData.blockDim; ++i) {
        int c = static_cast<int32_t>((tilingData.blockDim + i - 1) / i);
        float cost;
        // B0 + A < A0 + B
        if (i * n0 + m < m0 * c + n) {
            swizzlDirect = 1UL; // Nz
            cost = n0 * i + m0 * c;
            if (cost <= mincost) {
                mincost = cost;
                swizzlCount = i;
            }
        } else {
            swizzlDirect = 0UL; // Zn
            cost = m0 * i + n0 * c;
            if (cost < mincost) {
                mincost = cost;
                swizzlCount = i;
            }
        }
    }
    tilingData.swizzlDirect = swizzlDirect;
    tilingData.swizzlCount = swizzlCount;
    return swizzlDirect;
}
} // namespace pp_matmul
} // namespace optiling
#endif