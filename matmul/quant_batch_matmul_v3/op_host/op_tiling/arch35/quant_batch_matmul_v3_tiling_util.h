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
#pragma once

#include <cstdint>
#include <vector>

#include "acl/acl_rt.h"
#include "tiling/platform/platform_ascendc.h"

#include "../quant_batch_matmul_v3_tiling_base.h"
#include "matmul/quant_batch_matmul_v3/op_kernel/arch35/quant_batch_matmul_v3_tiling_data.h"

namespace optiling {

namespace qmmv3_tiling_const {
constexpr uint64_t CUBE_BLOCK = 16UL;
constexpr uint64_t CUBE_REDUCE_BLOCK = 32UL;
constexpr uint64_t L1_ALIGN_SIZE = 32UL;
constexpr uint64_t L2_ALIGN_SIZE = 128UL;
// MTE2 address alignment size in bytes.
constexpr uint64_t MTE2_ADDRESS_ALIGN_SIZE = 128UL;
constexpr uint64_t MX_GROUP_SIZE = 32UL;
constexpr uint64_t MXFP_DIVISOR_SIZE = 64UL;
constexpr uint64_t MXFP_MULTI_BASE_SIZE = 2UL;
constexpr uint64_t BASIC_BLOCK_SIZE_32 = 32UL;
constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128UL;
constexpr uint64_t BASIC_BLOCK_SIZE_256 = 256UL;
constexpr uint64_t PER_BLOCK_SIZE = 128UL;
constexpr uint32_t DOUBLE_BUFFER_NUM = 2U;
constexpr uint32_t DATA_SIZE_L0C = 4U;
constexpr uint8_t L1_TWO_BUFFER = 2U;
constexpr uint8_t L1_FOUR_BUFFER = 4U;
constexpr uint32_t NUM_HALF = 2U;
// Mix/per-block templates require the AIC:AIV core ratio to be 1:2.
constexpr uint32_t CORE_RATIO = 2U;
constexpr uint64_t WINDOW_LEN = 4UL;
constexpr uint64_t AFULLLOAD_SINGLE_CORE_A_SCALER = 2UL;
constexpr uint64_t AFULLLOAD_SINGLE_CORE_B_SCALER = 2UL;
// Heuristic cap for reserved MX scaleKL1 size.
constexpr uint64_t ESTIMATED_SCALE_K = 4096UL;
constexpr uint32_t SCALER_FACTOR_MAX = 127U;
constexpr uint32_t SCALER_FACTOR_MIN = 1U;
} // namespace qmmv3_tiling_const

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

enum class BiasMode : uint32_t { EXCLUEDE_FROM_TEMPLATE = 0, CUBE_BIAS_BF16_TEMPLATE = 1, CUBE_BIAS_FP16_TEMPLATE = 2 };

enum class QMMApiLevel : uint32_t { HIGH_LEVEL = 0, BASIC_LEVEL = 1, BLAZE_LEVEL = 2 };

inline bool IsTensorapiCapable()
{
    char pkgName[] = "asc-devkit";
    int32_t versionNum = 0;
    // asc-devkit 9.1.0版本引入tensor api
    constexpr int32_t MinTensorApiRuntimeVersion = 90100000;
    return aclsysGetVersionNum(pkgName, &versionNum) == ACL_SUCCESS && versionNum >= MinTensorApiRuntimeVersion;
}

inline bool IsCubeBasicApiCapable(const QuantBatchMatmulInfo& inputParams)
{
    const auto aDtype = (inputParams.aDtype == ge::DT_INT4 && inputParams.bDtype == ge::DT_INT4) ? ge::DT_INT8 :
                                                                                                   inputParams.aDtype;
    bool isCubePerTensor = inputParams.isPerTensor &&
                           ((aDtype == ge::DT_INT8 && inputParams.biasDtype == ge::DT_INT32 &&
                             !inputParams.isPertoken) ||
                            ((aDtype == ge::DT_FLOAT8_E4M3FN || aDtype == ge::DT_FLOAT8_E5M2 ||
                              aDtype == ge::DT_HIFLOAT8) &&
                             (inputParams.scaleDtype == ge::DT_UINT64 || inputParams.scaleDtype == ge::DT_INT64)));
    bool isCubePerChannel = inputParams.isPerChannel &&
                            (inputParams.scaleDtype == ge::DT_UINT64 || inputParams.scaleDtype == ge::DT_INT64 ||
                             inputParams.cDtype == ge::DT_INT32);
    return (inputParams.isDoubleScale && !inputParams.isPerChannel) || isCubePerTensor || isCubePerChannel;
}

inline bool IsMxBasicApiCapable(const QuantBatchMatmulInfo& inputParams)
{
    bool isMxfp8 = (inputParams.aDtype == ge::DT_FLOAT8_E4M3FN || inputParams.aDtype == ge::DT_FLOAT8_E5M2) &&
                   (inputParams.bDtype == ge::DT_FLOAT8_E4M3FN || inputParams.bDtype == ge::DT_FLOAT8_E5M2) &&
                   inputParams.isMxPerGroup;
    bool isMxfp4 = inputParams.isMxPerGroup && inputParams.aDtype == ge::DT_FLOAT4_E2M1 &&
                   inputParams.bDtype == ge::DT_FLOAT4_E2M1;
    return isMxfp8 || isMxfp4;
}

bool IsMxL0CPingpong(const QuantBatchMatmulInfo& inputParams);

inline bool IsPerblockBasicApiCapable(const QuantBatchMatmulInfo& inputParams)
{
    return inputParams.isPerBlock && (inputParams.bFormat == ge::FORMAT_ND || inputParams.bFormat == ge::FORMAT_FRACTAL_NZ);
}

inline bool IsFp8OrHif8TTFloatBiasMix(const QuantBatchMatmulInfo& inputParams)
{
    bool isFp8OrHif8Input = inputParams.aDtype == ge::DT_FLOAT8_E4M3FN || inputParams.aDtype == ge::DT_FLOAT8_E5M2 ||
                            inputParams.aDtype == ge::DT_HIFLOAT8;
    bool isPertensorDoubleScale = inputParams.isDoubleScale && inputParams.isPerTensor && !inputParams.isPerChannel;
    bool isFp32Scale = inputParams.scaleDtype == ge::DT_FLOAT && inputParams.perTokenScaleDtype == ge::DT_FLOAT;
    bool hasFp32Bias = inputParams.hasBias && inputParams.biasDtype == ge::DT_FLOAT;
    return isFp8OrHif8Input && isPertensorDoubleScale && isFp32Scale && hasFp32Bias;
}

inline bool IsNonMxCubeBasicApiCapable(const QuantBatchMatmulInfo& inputParams)
{
    return IsCubeBasicApiCapable(inputParams) && !IsMxBasicApiCapable(inputParams) &&
           !IsFp8OrHif8TTFloatBiasMix(inputParams);
}

// WeightNz + Blaze: use low-order BasicAPITilingData with Blaze API-level tiling key.
inline bool IsWeightNzNonMxCubeBasicApiCapable(const QuantBatchMatmulInfo& inputParams)
{
    return inputParams.bFormat == ge::FORMAT_FRACTAL_NZ && IsTensorapiCapable() &&
           IsNonMxCubeBasicApiCapable(inputParams);
}

enum class QMMKernelType : uint32_t {
    NO_VEC_EPILOGUE_WITH_MMAPI = 0,
    NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI = 1,
    VEC_EPILOGUE_WITH_MMAPI = 2,
    VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI = 3,
    VEC_EPILOGUE_WITH_CUSTOM_MM = 4,
    NO_VEC_EPILOGUE_CUSTOM_GMTOABL1_WITH_MMAPI = 5,
    NO_VEC_EPILOGUE_WITH_BMMAPI = 6,
    NO_VEC_EPILOGUE_WITH_BMMAPI_NO_BATCH_OUT = 7,
    NO_VEC_EPILOGUE_CUSTOM_GMTOBL1_WITH_MMAPI = 8,
    NO_VEC_EPILOGUE_WITH_MMAPI_WITHOUT_BATCH = 9,
    NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI_WITHOUT_BATCH = 10,
    NO_VEC_EPILOGUE_WITH_MMAPI_MX_L0C_PINGPONG = 11,
    NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI_MX_L0C_PINGPONG = 12,
    NO_VEC_EPILOGUE_WITH_MMAPI_MX_L0C_PINGPONG_WITHOUT_BATCH = 13,
    NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI_MX_L0C_PINGPONG_WITHOUT_BATCH = 14
};

class QuantBatchMatMulV3TilingUtil {
public:
    template <typename TilingData>
    static void SetCommonTilingData(const QuantBatchMatmulInfo& inputParams, TilingData& tilingData)
    {
        tilingData.params.batchA = inputParams.batchA;
        tilingData.params.batchB = inputParams.batchB;
        tilingData.params.batchC = inputParams.batchC;
        tilingData.params.batchA1 = inputParams.batchA1;
        tilingData.params.batchA2 = inputParams.batchA2;
        tilingData.params.batchA3 = inputParams.batchA3;
        tilingData.params.batchA4 = inputParams.batchA4;
        tilingData.params.batchB1 = inputParams.batchB1;
        tilingData.params.batchB2 = inputParams.batchB2;
        tilingData.params.batchB3 = inputParams.batchB3;
        tilingData.params.batchB4 = inputParams.batchB4;
        tilingData.params.batchC1 = inputParams.batchC1;
        tilingData.params.batchC2 = inputParams.batchC2;
        tilingData.params.batchC3 = inputParams.batchC3;
        tilingData.params.batchC4 = inputParams.batchC4;
        tilingData.params.biasDtype = static_cast<uint32_t>(inputParams.biasDtype);
        tilingData.params.biasThreeDim = static_cast<uint32_t>(inputParams.batchBias > 1UL);
        tilingData.params.groupSizeM = static_cast<uint32_t>(inputParams.groupSizeM);
        tilingData.params.groupSizeN = static_cast<uint32_t>(inputParams.groupSizeN);
        tilingData.params.groupSizeK = static_cast<uint32_t>(inputParams.groupSizeK);
    }
    static void SetBasicTilingData(const QuantBatchMatmulInfo& inputParams, const BasicRunInfoTiling& basicTiling,
                                   DequantBmm::QuantBatchMatmulV3TilingDataParams& tilingData);
    static void SetBasicLibApiTiling(const QuantBatchMatmulInfo& inputParams, const BasicRunInfoTiling& basicTiling,
                                     DequantBmm::QuantBatchMatmulV3TilingDataParams& tilingData);
    static bool CheckDtype(gert::TilingContext* context, const QuantBatchMatmulInfo& inputParams,
                           const QuantBatchMatmulV3CompileInfo& compileInfo);
    static bool CheckShape(gert::TilingContext* context, const QuantBatchMatmulInfo& inputParams,
                           const QuantBatchMatmulV3CompileInfo& compileInfo,
                           const std::vector<gert::Shape*>& mandatoryShape, const gert::StorageShape* biasShape,
                           const gert::StorageShape* pertokenShape, const std::vector<int64_t>& dimValueOfMKN);
    static uint64_t GetKernelType(const QuantBatchMatmulInfo& inputParams, const BasicRunInfoTiling& basicTiling,
                                  bool isBf16Mix, bool isAFullLoad, bool isBFullLoad, bool isABFullLoad);
    static uint64_t GetBiasMode(const QuantBatchMatmulInfo& inputParams);
};
} // namespace optiling
