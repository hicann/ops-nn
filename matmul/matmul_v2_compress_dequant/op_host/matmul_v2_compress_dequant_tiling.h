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
 * \file matmul_v2_compress_dequant_tiling.h
 * \brief Tiling data and registration for matmul_v2_compress_dequant
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_MATMUL_V2_COMPRESS_DEQUANT_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_MATMUL_V2_COMPRESS_DEQUANT_H

#include <cstdint>
#include "register/op_impl_registry.h"
#include "platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MatmulV2CompressDequantTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, m);
    TILING_DATA_FIELD_DEF(uint32_t, k);
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, m0);
    TILING_DATA_FIELD_DEF(uint32_t, k0);
    TILING_DATA_FIELD_DEF(uint32_t, n0);
    TILING_DATA_FIELD_DEF(uint32_t, mLoop);
    TILING_DATA_FIELD_DEF(uint32_t, kLoop);
    TILING_DATA_FIELD_DEF(uint32_t, nLoop);
    TILING_DATA_FIELD_DEF(uint32_t, coreLoop);
    TILING_DATA_FIELD_DEF(uint32_t, swizzlCount);
    TILING_DATA_FIELD_DEF(uint32_t, tilingK);
    TILING_DATA_FIELD_DEF(uint32_t, tilingN);
    TILING_DATA_FIELD_DEF(uint32_t, compressOverlapN);
    TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
    TILING_DATA_FIELD_DEF(uint32_t, blockDimVal);
    TILING_DATA_FIELD_DEF(uint32_t, splitK);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatMulV2CompressDequant, MatmulV2CompressDequantTilingData)

struct MatmulV2CompressDequantCompileInfo {
    uint64_t aicNum{0UL};
    uint64_t aivNum{0UL};
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
    uint64_t l2Size = 0;
    uint64_t l0CSize = 0;
    uint64_t l0ASize{0UL};
    uint64_t l0BSize{0UL};
    uint64_t btSize{0UL};
    float cubeFreq{0};
    NpuArch npuArch;
    platform_ascendc::SocVersion socVersion;
    std::string socVersionStr = "";
    bool supportL0c2out = false;
    bool supportL12BtBf16 = false;
    uint64_t workSpaceSize = 0;
    bool isRegbase = false;
};

ge::graphStatus TilingForMatmulV2CompressDequant(gert::TilingContext *context);

}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_MATMUL_V2_COMPRESS_DEQUANT_H
