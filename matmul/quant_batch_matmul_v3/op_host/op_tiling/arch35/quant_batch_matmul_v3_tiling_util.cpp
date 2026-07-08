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
 * \file quant_batch_matmul_v3_tiling_util.cpp
 * \brief
 */
#include "quant_batch_matmul_v3_tiling_util.h"

#include <new>
#include "error_util.h"
#include "quant_batch_matmul_v3_checker.h"
#include "quant_batch_matmul_v3_checker_for_mmads8s4.h"

namespace optiling {
namespace {
// Minimum HBM bandwidth of Ascend 950 series, in TB/s.
constexpr double HBM_BW = 1.4;
// Minimum L2 bandwidth of Ascend 950 series, in TB/s.
constexpr double L2_BW = 5.2;
// MXFP4 Cube throughput of 32-core Ascend 950 series, in TFLOPS.
constexpr double MXFP4_CUBE_THROUGHPUT = 1728.0;
// MXFP8 Cube throughput of 32-core Ascend 950 series, in TFLOPS.
constexpr double MXFP8_CUBE_THROUGHPUT = 864.0;
constexpr double MMAD_OPS_PER_DOT = 2.0;
constexpr uint64_t MMAD_BLOCK_SIZE = 256UL;
constexpr uint64_t MX_SCALE_GROUP_SIZE = 32UL;
constexpr uint64_t FP4_ELEMENTS_PER_BYTE = 2UL;
constexpr uint64_t L2_CACHE_SIZE = 128UL * 1024UL * 1024UL;
constexpr uint64_t L2_CACHE_THRESHOLD_PERCENT = 80UL;
constexpr uint64_t PERCENT_BASE = 100UL;
constexpr uint64_t L2_CACHE_THRESHOLD_SIZE = L2_CACHE_SIZE * L2_CACHE_THRESHOLD_PERCENT / PERCENT_BASE;
constexpr uint64_t MXFP8_L0C_PINGPONG_INNER_AXIS_ALIGN_SIZE = 128UL;
constexpr uint64_t MXFP4_L0C_PINGPONG_INNER_AXIS_ALIGN_SIZE = 256UL;

uint64_t GetDataSizeByDtype(uint64_t shape, ge::DataType dtype)
{
    if (dtype == ge::DT_FLOAT4_E2M1) {
        return ops::CeilDiv(shape, FP4_ELEMENTS_PER_BYTE);
    }
    return shape * static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
}

template <typename Checker>
bool CheckDtypeWithChecker(gert::TilingContext* context, const QuantBatchMatmulInfo& inputParams)
{
    auto* qmmV3Checker = new (std::nothrow) Checker(context, inputParams);
    OP_TILING_CHECK(qmmV3Checker == nullptr,
                    CUBE_INNER_ERR_REPORT(inputParams.opName, "Failed to instantiate qmmV3Checker."), return false);
    bool res = qmmV3Checker->CheckDtype();
    delete qmmV3Checker;
    return res;
}

template <typename Checker>
bool CheckShapeWithChecker(gert::TilingContext* context, const QuantBatchMatmulInfo& inputParams,
                           const std::vector<gert::Shape*>& mandatoryShape, const gert::StorageShape* biasShape,
                           const gert::StorageShape* pertokenShape, const std::vector<int64_t>& dimValueOfMKN)
{
    auto* qmmV3Checker = new (std::nothrow) Checker(context, inputParams);
    OP_TILING_CHECK(qmmV3Checker == nullptr,
                    CUBE_INNER_ERR_REPORT(inputParams.opName, "Failed to instantiate qmmV3Checker."), return false);
    bool res = qmmV3Checker->CheckShape(mandatoryShape, biasShape, pertokenShape, dimValueOfMKN);
    delete qmmV3Checker;
    return res;
}
bool IsMxCubeBound(const QuantBatchMatmulInfo& inputParams)
{
    const uint64_t m = inputParams.mSize;
    const uint64_t n = inputParams.nSize;
    const uint64_t k = inputParams.kSize;
    bool isMxfp4Input = inputParams.isMxPerGroup && inputParams.aDtype == ge::DT_FLOAT4_E2M1 &&
                        inputParams.bDtype == ge::DT_FLOAT4_E2M1;
    bool isMxfp8Input =
        inputParams.isMxPerGroup &&
        (inputParams.aDtype == ge::DT_FLOAT8_E4M3FN || inputParams.aDtype == ge::DT_FLOAT8_E5M2) &&
        (inputParams.bDtype == ge::DT_FLOAT8_E4M3FN || inputParams.bDtype == ge::DT_FLOAT8_E5M2);
    if (!isMxfp4Input && !isMxfp8Input) {
        return false;
    }
    const double cubeThroughput = isMxfp4Input ? MXFP4_CUBE_THROUGHPUT : MXFP8_CUBE_THROUGHPUT;
    const uint64_t aInputCost = m * k;
    const uint64_t bInputCost = n * k;
    const uint64_t aScaleCost = ops::CeilDiv(aInputCost, MX_SCALE_GROUP_SIZE);
    const uint64_t bScaleCost = ops::CeilDiv(bInputCost, MX_SCALE_GROUP_SIZE);
    const uint64_t copyOutBytes = GetDataSizeByDtype(m * n, inputParams.cDtype);
    const uint64_t hbmBytes =
        GetDataSizeByDtype(aInputCost, inputParams.aDtype) +
        GetDataSizeByDtype(aScaleCost, inputParams.perTokenScaleDtype) +
        GetDataSizeByDtype(bInputCost, inputParams.bDtype) +
        GetDataSizeByDtype(bScaleCost, inputParams.scaleDtype);
    const uint64_t aL2Cost = m * k * (ops::CeilDiv(n, MMAD_BLOCK_SIZE) - 1UL);
    const uint64_t bL2Cost = n * k * (ops::CeilDiv(m, MMAD_BLOCK_SIZE) - 1UL);
    const uint64_t l2Bytes =
        GetDataSizeByDtype(aL2Cost, inputParams.aDtype) +
        GetDataSizeByDtype(ops::CeilDiv(aL2Cost, MX_SCALE_GROUP_SIZE), inputParams.perTokenScaleDtype) +
        GetDataSizeByDtype(bL2Cost, inputParams.bDtype) +
        GetDataSizeByDtype(ops::CeilDiv(bL2Cost, MX_SCALE_GROUP_SIZE), inputParams.scaleDtype);
    const uint64_t l2Footprint = hbmBytes + copyOutBytes;
    const double copyOutBandwidth = l2Footprint < L2_CACHE_THRESHOLD_SIZE ? L2_BW : HBM_BW;
    const double transferCost =
        static_cast<double>(hbmBytes) / HBM_BW + static_cast<double>(l2Bytes) / L2_BW +
        static_cast<double>(copyOutBytes) / copyOutBandwidth;
    const double computeCost = MMAD_OPS_PER_DOT * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k) /
                            cubeThroughput;
    return computeCost > transferCost;
}

} // namespace

bool IsMxL0CPingpong(const QuantBatchMatmulInfo& inputParams)
{
    bool isMxfp4Input = inputParams.aDtype == ge::DT_FLOAT4_E2M1 && inputParams.bDtype == ge::DT_FLOAT4_E2M1;
    bool isMxfp8Input =
        (inputParams.aDtype == ge::DT_FLOAT8_E4M3FN || inputParams.aDtype == ge::DT_FLOAT8_E5M2) &&
        (inputParams.bDtype == ge::DT_FLOAT8_E4M3FN || inputParams.bDtype == ge::DT_FLOAT8_E5M2);
    bool isMxDataInput = isMxfp4Input || isMxfp8Input;
    bool isSupportedOutputDtype = inputParams.cDtype == ge::DT_FLOAT16 || inputParams.cDtype == ge::DT_BF16;
    uint64_t aInnerAxis = inputParams.transA ? inputParams.mSize : inputParams.kSize;
    uint64_t bInnerAxis = inputParams.transB ? inputParams.kSize : inputParams.nSize;
    uint64_t innerAxisAlignSize =
        isMxfp4Input ? MXFP4_L0C_PINGPONG_INNER_AXIS_ALIGN_SIZE : MXFP8_L0C_PINGPONG_INNER_AXIS_ALIGN_SIZE;
    bool isInnerAxisAligned =
        GetDataSizeByDtype(aInnerAxis, inputParams.aDtype) % innerAxisAlignSize == 0UL &&
        GetDataSizeByDtype(bInnerAxis, inputParams.bDtype) % innerAxisAlignSize == 0UL;
    return inputParams.isMxPerGroup && inputParams.scaleDtype == ge::DT_FLOAT8_E8M0 && IsTensorapiCapable() &&
           isMxDataInput && isSupportedOutputDtype && isInnerAxisAligned && IsMxCubeBound(inputParams);
}

void QuantBatchMatMulV3TilingUtil::SetBasicTilingData(const QuantBatchMatmulInfo& inputParams,
                                                      const BasicRunInfoTiling& basicTiling,
                                                      DequantBmm::QuantBatchMatmulV3TilingDataParams& tilingData)
{
    SetCommonTilingData(inputParams, tilingData);
    tilingData.params.ubCalcN = basicTiling.ubCalcN;
    tilingData.params.ubCalcM = basicTiling.ubCalcM;
    tilingData.params.isPerTensor = static_cast<uint32_t>(inputParams.isPerTensor);
    tilingData.params.isPertoken = static_cast<uint32_t>(inputParams.isPertoken);
    tilingData.params.isDoubleScale = static_cast<uint32_t>(inputParams.isDoubleScale);
}

void QuantBatchMatMulV3TilingUtil::SetBasicLibApiTiling(const QuantBatchMatmulInfo& inputParams,
                                                        const BasicRunInfoTiling& basicTiling,
                                                        DequantBmm::QuantBatchMatmulV3TilingDataParams& tilingData)
{
    tilingData.matmulTiling.M = inputParams.mSize;
    tilingData.matmulTiling.N = inputParams.nSize;
    tilingData.matmulTiling.Ka = inputParams.kSize;
    tilingData.matmulTiling.Kb = inputParams.kSize;
    tilingData.matmulTiling.usedCoreNum = basicTiling.usedCoreNum;
    tilingData.matmulTiling.singleCoreM = basicTiling.singleCoreM;
    tilingData.matmulTiling.singleCoreN = basicTiling.singleCoreN;
    tilingData.matmulTiling.singleCoreK = basicTiling.singleCoreK;
    tilingData.matmulTiling.baseM = basicTiling.baseM;
    tilingData.matmulTiling.baseN = basicTiling.baseN;
    tilingData.matmulTiling.baseK = basicTiling.baseK;
    tilingData.matmulTiling.depthA1 = basicTiling.depthA1;
    tilingData.matmulTiling.depthB1 = basicTiling.depthB1;
    tilingData.matmulTiling.stepM = basicTiling.stepM;
    tilingData.matmulTiling.stepN = basicTiling.stepN;
    tilingData.matmulTiling.stepKa = basicTiling.stepKa;
    tilingData.matmulTiling.stepKb = basicTiling.stepKb;
    tilingData.matmulTiling.isBias = inputParams.hasBias ? 1 : 0;
    tilingData.matmulTiling.iterateOrder = basicTiling.iterateOrder;
    tilingData.matmulTiling.dbL0A = 2; // Double buffer flag: 1 disables DB, 2 enables DB.
    tilingData.matmulTiling.dbL0B = 2; // Double buffer flag: 1 disables DB, 2 enables DB.
    tilingData.matmulTiling.dbL0C = basicTiling.dbL0c;
}

bool QuantBatchMatMulV3TilingUtil::CheckDtype(gert::TilingContext* context, const QuantBatchMatmulInfo& inputParams,
                                              const QuantBatchMatmulV3CompileInfo& compileInfo)
{
    switch (compileInfo.npuArch) {
        case NpuArch::DAV_3510:
            return CheckDtypeWithChecker<QuantBatchMatmulV3Checker>(context, inputParams);
        case NpuArch::DAV_RESV:
            return CheckDtypeWithChecker<QuantBatchMatmulV3Checker4MmadS8S4>(context, inputParams);
        default:
            CUBE_INNER_ERR_REPORT(inputParams.opName,
                                  "Failed to find CheckDtype function for current NPU architecture.");
            return false;
    }
}

bool QuantBatchMatMulV3TilingUtil::CheckShape(gert::TilingContext* context, const QuantBatchMatmulInfo& inputParams,
                                              const QuantBatchMatmulV3CompileInfo& compileInfo,
                                              const std::vector<gert::Shape*>& mandatoryShape,
                                              const gert::StorageShape* biasShape,
                                              const gert::StorageShape* pertokenShape,
                                              const std::vector<int64_t>& dimValueOfMKN)
{
    switch (compileInfo.npuArch) {
        case NpuArch::DAV_3510:
            return CheckShapeWithChecker<QuantBatchMatmulV3Checker>(context, inputParams, mandatoryShape, biasShape,
                                                                    pertokenShape, dimValueOfMKN);
        case NpuArch::DAV_RESV:
            return CheckShapeWithChecker<QuantBatchMatmulV3Checker4MmadS8S4>(context, inputParams, mandatoryShape,
                                                                             biasShape, pertokenShape, dimValueOfMKN);
        default:
            CUBE_INNER_ERR_REPORT(inputParams.opName,
                                  "Failed to find CheckShape function for current NPU architecture.");
            return false;
    }
}

uint64_t QuantBatchMatMulV3TilingUtil::GetBiasMode(const QuantBatchMatmulInfo& inputParams)
{
    uint64_t biasMode = 0UL;
    if (!inputParams.hasBias) {
        biasMode = static_cast<uint64_t>(BiasMode::EXCLUEDE_FROM_TEMPLATE);
    } else if (inputParams.aDtype == ge::DT_INT8) {
        biasMode = static_cast<uint64_t>(BiasMode::EXCLUEDE_FROM_TEMPLATE);
    } else {
        if (inputParams.biasDtype == ge::DT_FLOAT) {
            biasMode = static_cast<uint64_t>(BiasMode::EXCLUEDE_FROM_TEMPLATE);
        } else if (inputParams.biasDtype == ge::DT_BF16) {
            biasMode = static_cast<uint64_t>(BiasMode::CUBE_BIAS_BF16_TEMPLATE);
        } else if (inputParams.biasDtype == ge::DT_FLOAT16) {
            biasMode = static_cast<uint64_t>(BiasMode::CUBE_BIAS_FP16_TEMPLATE);
        }
    }
    return biasMode;
}

uint64_t QuantBatchMatMulV3TilingUtil::GetKernelType(const QuantBatchMatmulInfo& inputParams,
                                                     const BasicRunInfoTiling& basicTiling, bool isBf16Mix,
                                                     bool isAFullLoad, bool isBFullLoad, bool isABFullLoad)
{
    uint64_t kernelType = 0UL;
    bool isScaleVecPostProcess = inputParams.isPerChannel &&
                                 !(inputParams.scaleDtype == ge::DT_UINT64 || inputParams.scaleDtype == ge::DT_INT64);
    bool isVecPostProcess = (isScaleVecPostProcess || inputParams.isPertoken || inputParams.isPerBlock || isBf16Mix) &&
                            (inputParams.cDtype != ge::DT_INT32);
    bool isMxWithoutBatch = IsTensorapiCapable() && inputParams.isMxPerGroup && inputParams.batchC == 1UL;
    bool useMxL0CPingpong = IsMxL0CPingpong(inputParams);
    bool isMixWithoutBatch = !inputParams.transA && inputParams.bFormat == ge::FORMAT_FRACTAL_NZ && 
                             ((inputParams.aDtype == ge::DT_INT8 && inputParams.cDtype == ge::DT_BF16) || 
                             inputParams.aDtype == ge::DT_FLOAT8_E4M3FN || inputParams.aDtype == ge::DT_HIFLOAT8) && 
                             inputParams.aDtype == inputParams.bDtype && inputParams.batchC == 1UL && 
                             inputParams.cDtype != ge::DT_INT32;
    if (basicTiling.iterBatch >= 1U) {
        if (basicTiling.baseM == basicTiling.singleCoreM && basicTiling.baseN == basicTiling.singleCoreN) {
            kernelType = static_cast<uint64_t>(QMMKernelType::NO_VEC_EPILOGUE_WITH_BMMAPI);
        } else {
            kernelType = static_cast<uint64_t>(QMMKernelType::NO_VEC_EPILOGUE_WITH_BMMAPI_NO_BATCH_OUT);
        }
    } else if (!isVecPostProcess && isABFullLoad) {
        kernelType = static_cast<uint64_t>(QMMKernelType::NO_VEC_EPILOGUE_CUSTOM_GMTOABL1_WITH_MMAPI);
    } else if (!isVecPostProcess && isAFullLoad) {
        if (useMxL0CPingpong && isMxWithoutBatch) {
            kernelType = static_cast<uint64_t>(
                QMMKernelType::NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI_MX_L0C_PINGPONG_WITHOUT_BATCH);
        } else if (useMxL0CPingpong) {
            kernelType = static_cast<uint64_t>(
                QMMKernelType::NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI_MX_L0C_PINGPONG);
        } else if (isMxWithoutBatch) {
            kernelType = static_cast<uint64_t>(
                QMMKernelType::NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI_WITHOUT_BATCH);
        } else {
            kernelType = static_cast<uint64_t>(QMMKernelType::NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI);
        }
    } else if (!isVecPostProcess && isMxWithoutBatch) {
        if (useMxL0CPingpong) {
            kernelType = static_cast<uint64_t>(QMMKernelType::NO_VEC_EPILOGUE_WITH_MMAPI_MX_L0C_PINGPONG_WITHOUT_BATCH);
        } else {
            kernelType = static_cast<uint64_t>(QMMKernelType::NO_VEC_EPILOGUE_WITH_MMAPI_WITHOUT_BATCH);
        }
    } else if (!isVecPostProcess && !isAFullLoad && !isBFullLoad) {
        if (useMxL0CPingpong) {
            kernelType = static_cast<uint64_t>(QMMKernelType::NO_VEC_EPILOGUE_WITH_MMAPI_MX_L0C_PINGPONG);
        } else {
            kernelType = static_cast<uint64_t>(QMMKernelType::NO_VEC_EPILOGUE_WITH_MMAPI);
        }
    } else if (!isVecPostProcess && isBFullLoad) {
        kernelType = static_cast<uint64_t>(QMMKernelType::NO_VEC_EPILOGUE_CUSTOM_GMTOBL1_WITH_MMAPI);
    } else if (IsTensorapiCapable() && (isScaleVecPostProcess || inputParams.isPertoken || isBf16Mix) && !isAFullLoad &&
               isMixWithoutBatch) {
        kernelType = static_cast<uint64_t>(QMMKernelType::VEC_EPILOGUE_WITH_MMAPI_WITHOUT_BATCH);
    } else if (IsTensorapiCapable() && (isScaleVecPostProcess || inputParams.isPertoken || isBf16Mix) && isAFullLoad &&
               isMixWithoutBatch) {
        kernelType = static_cast<uint64_t>(QMMKernelType::VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI_WITHOUT_BATCH);
    } else if ((isScaleVecPostProcess || inputParams.isPertoken || isBf16Mix) && !isAFullLoad) {
        kernelType = static_cast<uint64_t>(QMMKernelType::VEC_EPILOGUE_WITH_MMAPI);
    } else if ((isScaleVecPostProcess || inputParams.isPertoken || isBf16Mix) && isAFullLoad) {
        kernelType = static_cast<uint64_t>(QMMKernelType::VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI);
    } else if (inputParams.isPerBlock) {
        kernelType = static_cast<uint64_t>(QMMKernelType::VEC_EPILOGUE_WITH_CUSTOM_MM);
    }
    return kernelType;
}

} // namespace optiling
