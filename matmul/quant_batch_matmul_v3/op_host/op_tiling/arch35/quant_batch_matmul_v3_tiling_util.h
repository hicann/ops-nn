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
 * \file quant_batch_matmul_v3_tiling_util.h
 * \brief
 */
#ifndef QUANT_BATCH_MATMUL_V3_TILING_UTIL_H
#define QUANT_BATCH_MATMUL_V3_TILING_UTIL_H

#include "../quant_batch_matmul_v3_tiling_base.h"
#include "../../../op_kernel/arch35/quant_batch_matmul_v3_tiling_data.h"

namespace optiling {

struct BasicRunInfoTiling {
    uint32_t usedCoreNum = 1;
    uint32_t singleCoreM = 1;
    uint32_t singleCoreN = 1;
    uint32_t singleCoreK = 1;
    uint32_t baseM = 1;
    uint32_t baseN = 1;
    uint32_t baseK = 1;
    uint32_t stepKa = 1;
    uint32_t stepKb = 1;
    uint32_t depthA1 = 1;
    uint32_t depthB1 = 1;
    uint32_t stepM = 1;
    uint32_t stepN = 1;
    uint32_t iterateOrder = 0;
    uint32_t dbL0c = 1;
    uint32_t ubCalcN = 0;
    uint32_t ubCalcM = 0;
    uint32_t scaleFactorA = 0;
    uint32_t scaleFactorB = 0;
    uint32_t iterBatch = 0;
};

enum class BiasMode : uint32_t {
    EXCLUEDE_FROM_TEMPLATE = 0,
    CUBE_BIAS_BF16_TEMPLATE = 1,
    CUBE_BIAS_FP16_TEMPLATE = 2
};

enum class QMMKernelType : uint32_t {
    NO_VEC_EPILOGUE_WITH_MMAPI = 0,
    NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI = 1,
    VEC_EPILOGUE_WITH_MMAPI = 2,
    VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI = 3,
    VEC_EPILOGUE_WITH_CUSTOM_MM = 4,
    NO_VEC_EPILOGUE_CUSTOM_GMTOABL1_WITH_MMAPI = 5,
    NO_VEC_EPILOGUE_WITH_BMMAPI = 6,
    NO_VEC_EPILOGUE_WITH_BMMAPI_NO_BATCH_OUT = 7,
    NO_VEC_EPILOGUE_CUSTOM_GMTOBL1_WITH_MMAPI = 8
};

class QuantBatchMatMulV3TilingUtil {
public:
    static void SetBasicTilingData(const QuantBatchMatmulInfo &inputParams, const BasicRunInfoTiling &basicTiling,
                                   DequantBmm::QuantBatchMatmulV3TilingDataParams &tilingData);
    static void SetBasicLibApiTiling(const QuantBatchMatmulInfo &inputParams, const BasicRunInfoTiling &basicTiling,
                                     DequantBmm::QuantBatchMatmulV3TilingDataParams &tilingData);
    static uint64_t GetKernelType(const QuantBatchMatmulInfo &inputParams, const BasicRunInfoTiling &basicTiling,
                                  bool isBf16Mix, bool isAFullLoad, bool isBFullLoad, bool isABFullLoad);
    static uint64_t GetBiasMode(const QuantBatchMatmulInfo &inputParams);
};
} // namespace optiling
#endif