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
 * \file pp_matmul_info.h
 * \brief
 */
#ifndef __OP_HOST_PP_MATMUL_INFO_H__
#define __OP_HOST_PP_MATMUL_INFO_H__

#include <array>
#include <iostream>
#include <map>
#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_base.h"

namespace optiling {
namespace pp_matmul {
struct MatMulInfo {
    uint64_t batchSize{0};
    uint64_t m{0}; // 实际输入的 m
    uint64_t n{0}; // 实际输入的 n
    uint64_t k{0}; // 实际输入的 k
    ge::DataType dtypeA = ge::DT_FLOAT16;
    ge::DataType dtypeB = ge::DT_FLOAT16;
    ge::DataType dtypeC = ge::DT_FLOAT16;
    ge::Format formatA = ge::FORMAT_ND;
    ge::Format formatB = ge::FORMAT_ND;
    ge::Format formatC = ge::FORMAT_ND;
    uint64_t transA{0}; 
    uint64_t transB{0}; 
    bool biasFlag{0}; // false: 0, true: 1
    bool isInt8{0}; // 是否shi int8融合
    float inDtype{0};
    float outDtype{0};
};

struct HardwareInfo {
    uint64_t coreNum{0};
    uint64_t l2Size{0};
    uint64_t l1Size{0};
    uint64_t l0aSize{0};
    uint64_t l0bSize{0};
    uint64_t l0cSize{0};
    uint64_t hbmBandWidth{1};
    uint64_t l2BandWidth{5};// 5x faster than hbm.
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
    std::string socVersionStr = "";
};

struct OpShape {
    uint64_t batchSize{0};
    uint64_t m{0};
    uint64_t k{0};
    uint64_t n{0};
    uint64_t m0{0};
    uint64_t k0{0};
    uint64_t n0{0};
};

struct PpMatmulDefaultTilingData {
    OpShape opShape{};
    uint64_t mLoop{1};
    uint64_t kLoop{1};
    uint64_t nLoop{1};
    uint64_t coreLoop{1};
    uint64_t swizzlCount{1};
    uint32_t tilingKey{0};
    uint64_t blockDim{1};
    uint64_t swizzlDirect{0};
    uint64_t splitk{0};
    uint64_t enShuffleK{0};

    void SetBaseShape(uint64_t batchSize, uint64_t m, uint64_t k, uint64_t n);
    void SetBaseOp(uint64_t coreNum, uint64_t l0cSize, uint64_t mBase, uint64_t nBase, const MatMulInfo &mmInfo, bool isAscend310P);
    void End(const MatMulInfo &mmInfo, bool isAscend310P);
};

} // namespace pp_matmul
} // namespace optiling
#endif