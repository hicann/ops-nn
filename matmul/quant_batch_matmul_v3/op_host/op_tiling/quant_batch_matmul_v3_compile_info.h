/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_batch_matmul_v4_compile_info.h
 * \brief
 */
#ifndef __OP_HOST_QUANT_BATCH_MATMUL_V3_COMPILE_INFO_H__
#define __OP_HOST_QUANT_BATCH_MATMUL_V3_COMPILE_INFO_H__

#include <cstdint>
#include <string>

#include "tiling/platform/platform_ascendc.h"
#include "platform/soc_spec.h"
#include "graph/types.h"

namespace optiling {

// SoC 规格参数：TilingParse/InitCompileInfo 阶段从平台信息解析并缓存在 compileInfo 中，
// 缺失项由 FillSocSpecifyParamsFromPlatform 回退默认值；dtype 相关修正（如 9202 hif4 mmadFracN 翻倍）
// 依赖节点信息，保留在 tiling 期 GetSocSpecAdjustBalanceInfo 中施加。
struct SocSpecifyParams {
    double hbmBw = 0;                  // HBM带宽 TB/s
    double l2Bw = 0;                   // L2带宽 TB/s
    double freq = 0;                   // GHz
    double singleCoreComputePower = 0; // 单核算力
    double l0Bw = 256;                 // B/c
    double l0BwB4KoutterAxis = 128;    // B/c
    double l0mxBw = 64;                // B/c
    uint64_t mmadFracM = 16;           // 元素数
    uint64_t mmadFracN = 16;           // 元素数
    uint64_t mmadFracK = 32;           // 32B
    void CalSingleCoreComputePower(ge::DataType aDtype)
    {
        // Keep mmadFracK in bytes for cost estimation; compute power uses logical K elements.
        // 4bit dtype 下 mmadFracK(32B) 对应 64 个逻辑 K 元素，算力按逻辑元素数折算。
        const bool isFourBitDtype = aDtype == ge::DT_FLOAT4_E2M1 || aDtype == ge::DT_FLOAT4_E1M2 ||
                                    aDtype == ge::DT_INT4 || aDtype == ge::DT_HIFLOAT4;
        const uint64_t logicalMmadK = isFourBitDtype ? mmadFracK + mmadFracK : mmadFracK;
        singleCoreComputePower = mmadFracM * mmadFracN * logicalMmadK * freq / 1024; // 1024: KB to B
    }
};

struct QuantBatchMatmulV3CompileInfo {
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l2Size;
    uint64_t l0cSize;
    uint64_t l0aSize;
    uint64_t l0bSize;
    uint32_t workspaceNum;
    uint32_t aivNum;
    uint32_t aicNum;
    bool supportL0c2Out;
    bool supportL12BtBf16;
    bool supportMmadS8S4;
    platform_ascendc::SocVersion socVersion;
    NpuArch npuArch;
    std::string socVersionStr = "";
    SocSpecifyParams socSpec;
};

} // namespace optiling
#endif // __OP_HOST_QUANT_BATCH_MATMUL_V3_COMPILE_INFO_H__
