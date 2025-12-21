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
 * \file conv3d_backprop_filter_v2_tiling_data.h
 * \brief
 */
#ifndef CONV3D_BACKPROP_FILTER_V2_TILING_DATA_H
#define CONV3D_BACKPROP_FILTER_V2_TILING_DATA_H

#include <cstdint>
#include <string>
#include <sstream>

namespace AscendC {
namespace conv_bp_v2_kernel {
struct TConv3DDwTiling {
    uint32_t batch = 1;
    uint32_t cin = 1;
    uint32_t cout = 1;
    uint32_t cin1G = 1;
    uint32_t cout1G = 1;
    uint32_t dout = 1;
    uint32_t ho = 1;
    uint32_t wo = 1;
    uint32_t di = 1;
    uint32_t hi = 1;
    uint32_t wi = 1;
    uint32_t dk = 1;
    uint32_t hk = 1;
    uint32_t wk = 1;
    uint32_t group = 1;
    uint32_t realGroup = 1;
    uint32_t strideD = 1;
    uint32_t strideH = 1;
    uint32_t strideW = 1;
    uint32_t padFront = 1;
    uint32_t padBack = 1;
    uint32_t padUp = 1;
    uint32_t padDown = 1;
    uint32_t padLeft = 1;
    uint32_t padRight = 1;
    uint32_t dilationD = 1;
    uint32_t dilationH = 1;
    uint32_t dilationW = 1;
    uint32_t channelSize = 1;
    uint32_t al0Pbuffer = 1;
    uint32_t bl0Pbuffer = 1;
    uint32_t cl0Pbuffer = 1;
    uint32_t al1Pbuffer = 1;
    uint32_t bl1Pbuffer = 1;
    uint32_t baseM = 1;
    uint32_t baseK = 1;
    uint32_t baseN = 1;
    uint32_t m0 = 1;
    uint32_t k0 = 1;
    uint32_t n0 = 1;
    uint32_t stepM = 1;
    uint32_t stepN = 1;
    uint32_t stepKa = 1;
    uint32_t stepKb = 1;
    uint32_t iterateOrder = 1;
    uint32_t bl1Bound = 1;
    uint32_t al1Bound = 1;
    uint32_t hf32Flag = 1;
    uint32_t singleCoreDk = 1;
    uint32_t singleCoreGroup = 1;
    uint32_t singleCoreCout = 1;
    uint32_t singleCoreHo = 1;
    uint32_t splitWo = 128;
    uint32_t isSplitKernelHW = 0;
    uint64_t singleCoreBatch = 1;
    uint64_t singleCoreCin = 1;

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "batch: " << batch << " cin: " << cin << " cout: " << cout << " cin1G: " << cin1G << " cout1G: " << cout1G
           << " dout: " << dout << " ho: " << ho << " wo: " << wo << " di: " << di << " hi: " << hi << " wi: " << wi
           << " dk: " << dk << " hk: " << hk << " wk: " << wk << " group: " << group << " realGroup: " << realGroup
           << " strideD: " << strideD << " strideH: " << strideH << " strideW: " << strideW << " padFront: " << padFront
           << " padBack: " << padBack << " padUp: " << padUp << " padDown: " << padDown << " padLeft: " << padLeft
           << " padRight: " << padRight << " dilationD: " << dilationD << " dilationH: " << dilationH
           << " dilationW: " << dilationW << " channelSize: " << channelSize << " al0Pbuffer: " << al0Pbuffer
           << " bl0Pbuffer: " << bl0Pbuffer << " cl0Pbuffer: " << cl0Pbuffer << " al1Pbuffer: " << al1Pbuffer
           << " bl1Pbuffer: " << bl1Pbuffer << " baseM: " << baseM << " baseK: " << baseK << " baseN: " << baseN
           << " m0: " << m0 << " k0: " << k0 << " n0: " << n0 << " stepM: " << stepM << " stepN: " << stepN
           << " stepKa: " << stepKa << " stepKb: " << stepKb << " iterateOrder: " << iterateOrder
           << " al1Bound: " << al1Bound << " bl1Bound: " << bl1Bound << " hf32Flag: " << hf32Flag
           << " singleCoreDk: " << singleCoreDk << " singleCoreGroup: " << singleCoreGroup
           << " singleCoreCout: " << singleCoreCout << " singleCoreHo: " << singleCoreHo << " splitWoSize: " << splitWo
           << " isSplitKernelHW: " << isSplitKernelHW << " singleCoreBatch: " << singleCoreBatch
           << " singleCoreCin: " << singleCoreCin;
        return ss.str();
    }
};

struct Conv3DBackpropFilterV2Params {
    uint64_t batchDim = 1;
    uint32_t groupDim = 1;
    uint32_t mDim = 1;
    uint32_t kDim = 1;
    uint32_t nDim = 1;
    uint32_t dkDim = 1;
    uint32_t totalL1Size = 1;
};

struct TConv3DDwBasicBlockTiling {
    uint64_t singleCoreBatchDout = 1;
    uint32_t streamkType = 1;
    uint32_t usedCoreNum = 1;
    uint32_t singleCoreM = 1;
    uint32_t singleCoreN = 1;
    uint32_t singleCoreK = 1;
    uint32_t reserve0 = 0; // 占位字段，为了8位字节对齐，否则 memcpy_s tilingData时候会出现补位导致数据异常

    std::string ToString() const
    {
        std::stringstream ss;
        ss << " singleCoreBatchDout: " << singleCoreBatchDout << " streamkType: " << streamkType
           << " usedCoreNum: " << usedCoreNum << " singleCoreM: " << singleCoreM << " singleCoreN: " << singleCoreN
           << " singleCoreK: " << singleCoreK;
        return ss.str();
    }
};

struct Conv3DBackpropFilterV2TilingData {
    Conv3DBackpropFilterV2Params params;
    TConv3DDwTiling dwTiling;
    TConv3DDwBasicBlockTiling basicBlockTiling;
};
}  // namespace conv_bp_v2_kernel
}
#endif  // CONV3D_BACKPROP_FILTER_V2_TILING_DATA_H