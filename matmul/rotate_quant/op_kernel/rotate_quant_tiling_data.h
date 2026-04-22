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
 * \file rotate_quant_tiling_data.h
 * \brief
 */

#ifndef ROTATE_QUANT_TILING_DATA_H
#define ROTATE_QUANT_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

namespace RotateQuantOpt {
    constexpr uint64_t STRUCT_ALIGNAS = 8;
    #pragma pack(push, 8)
    struct alignas(STRUCT_ALIGNAS) RotateQuantTilingData {
        int32_t M;
        int32_t N;
        int32_t K;
        int32_t loopM;
        int64_t dstType;
        int64_t stepLoop;
        int32_t numBlocks;
        int32_t aicCoreNum;
        int32_t aivCoreNum;
        uint32_t multiRowNumHeadCore;
        uint32_t headCoreNum;
        uint32_t rowPerHeadCore;
        uint32_t rowPerCubeHeadCore;
        uint32_t rowPerCubeTailCore;
        uint32_t rowPerVectorTailCore;
        uint32_t rowPerVectorLastCore;
        uint32_t lastUbRows;
        uint32_t ubSize;
        AscendC::tiling::TCubeTiling matmulTiling;
    };
    #pragma pack(pop)
}  // RotateQuantOpt

#endif  // ROTATE_QUANT_TILING_DATA_H
