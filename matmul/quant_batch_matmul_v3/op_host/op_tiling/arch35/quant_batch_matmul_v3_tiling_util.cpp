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
template <typename Checker>
bool CheckDtypeWithChecker(gert::TilingContext *context, const QuantBatchMatmulInfo& inputParams)
{
    auto *qmmV3Checker = new (std::nothrow) Checker(context, inputParams);
    OP_TILING_CHECK(
        qmmV3Checker == nullptr, CUBE_INNER_ERR_REPORT(inputParams.opName, "Failed to instantiate qmmV3Checker."),
        return false);
    bool res = qmmV3Checker->CheckDtype();
    delete qmmV3Checker;
    return res;
}

template <typename Checker>
bool CheckShapeWithChecker(
    gert::TilingContext *context, const QuantBatchMatmulInfo& inputParams,
    const std::vector<gert::Shape *>& mandatoryShape, const gert::StorageShape *biasShape,
    const gert::StorageShape *pertokenShape, const std::vector<int64_t>& dimValueOfMKN)
{
    auto *qmmV3Checker = new (std::nothrow) Checker(context, inputParams);
    OP_TILING_CHECK(
        qmmV3Checker == nullptr, CUBE_INNER_ERR_REPORT(inputParams.opName, "Failed to instantiate qmmV3Checker."),
        return false);
    bool res = qmmV3Checker->CheckShape(mandatoryShape, biasShape, pertokenShape, dimValueOfMKN);
    delete qmmV3Checker;
    return res;
}
} // namespace

void QuantBatchMatMulV3TilingUtil::SetBasicTilingData(
    const QuantBatchMatmulInfo& inputParams, const BasicRunInfoTiling& basicTiling,
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
    tilingData.matmulTiling.dbL0A = 2;  // Double buffer flag: 1 disables DB, 2 enables DB.
    tilingData.matmulTiling.dbL0B = 2;  // Double buffer flag: 1 disables DB, 2 enables DB.
    tilingData.matmulTiling.dbL0C = basicTiling.dbL0c;
}

bool QuantBatchMatMulV3TilingUtil::CheckDtype(
    gert::TilingContext *context, const QuantBatchMatmulInfo& inputParams,
    const QuantBatchMatmulV3CompileInfo& compileInfo)
{
    switch (compileInfo.npuArch) {
        case NpuArch::DAV_3510:
            return CheckDtypeWithChecker<QuantBatchMatmulV3Checker>(context, inputParams);
        case NpuArch::DAV_RESV:
            return CheckDtypeWithChecker<QuantBatchMatmulV3Checker4MmadS8S4>(context, inputParams);
        default:
            CUBE_INNER_ERR_REPORT(inputParams.opName, "Failed to find CheckDtype function for current NPU architecture.");
            return false;
    }
}

bool QuantBatchMatMulV3TilingUtil::CheckShape(
    gert::TilingContext *context, const QuantBatchMatmulInfo& inputParams,
    const QuantBatchMatmulV3CompileInfo& compileInfo, const std::vector<gert::Shape *>& mandatoryShape,
    const gert::StorageShape *biasShape, const gert::StorageShape *pertokenShape,
    const std::vector<int64_t>& dimValueOfMKN)
{
    switch (compileInfo.npuArch) {
        case NpuArch::DAV_3510:
            return CheckShapeWithChecker<QuantBatchMatmulV3Checker>(
                context, inputParams, mandatoryShape, biasShape, pertokenShape, dimValueOfMKN);
        case NpuArch::DAV_RESV:
            return CheckShapeWithChecker<QuantBatchMatmulV3Checker4MmadS8S4>(
                context, inputParams, mandatoryShape, biasShape, pertokenShape, dimValueOfMKN);
        default:
            CUBE_INNER_ERR_REPORT(inputParams.opName, "Failed to find CheckShape function for current NPU architecture.");
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
                                                    const BasicRunInfoTiling& basicTiling,  bool isBf16Mix,
                                                    bool isAFullLoad, bool isBFullLoad, bool isABFullLoad)
{
    uint64_t kernelType = 0UL;
    bool isScaleVecPostProcess = inputParams.isPerChannel &&
                                 !(inputParams.scaleDtype == ge::DT_UINT64 || inputParams.scaleDtype == ge::DT_INT64);
    bool isVecPostProcess = (isScaleVecPostProcess || inputParams.isPertoken || inputParams.isPerBlock || isBf16Mix) &&
                            (inputParams.cDtype != ge::DT_INT32);
    bool isMxWithoutBatch = IsTensorapiCapable() && inputParams.isMxPerGroup && inputParams.batchC == 1UL;
    if (basicTiling.iterBatch >= 1U) {
        if (basicTiling.baseM == basicTiling.singleCoreM && basicTiling.baseN == basicTiling.singleCoreN) {
            kernelType = static_cast<uint64_t>(QMMKernelType::NO_VEC_EPILOGUE_WITH_BMMAPI);
        } else {
            kernelType = static_cast<uint64_t>(QMMKernelType::NO_VEC_EPILOGUE_WITH_BMMAPI_NO_BATCH_OUT);
        }
    } else if (!isVecPostProcess && isABFullLoad) {
        kernelType = static_cast<uint64_t>(QMMKernelType::NO_VEC_EPILOGUE_CUSTOM_GMTOABL1_WITH_MMAPI);
    } else if (!isVecPostProcess && isMxWithoutBatch && isAFullLoad) {
        kernelType = static_cast<uint64_t>(QMMKernelType::NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI_WITHOUT_BATCH);
    } else if (!isVecPostProcess && isMxWithoutBatch) {
        kernelType = static_cast<uint64_t>(QMMKernelType::NO_VEC_EPILOGUE_WITH_MMAPI_WITHOUT_BATCH);
    } else if (!isVecPostProcess && !isAFullLoad && !isBFullLoad) {
        kernelType = static_cast<uint64_t>(QMMKernelType::NO_VEC_EPILOGUE_WITH_MMAPI);
    } else if (!isVecPostProcess && isAFullLoad) {
        kernelType = static_cast<uint64_t>(QMMKernelType::NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI);
    } else if (!isVecPostProcess && isBFullLoad) {
        kernelType = static_cast<uint64_t>(QMMKernelType::NO_VEC_EPILOGUE_CUSTOM_GMTOBL1_WITH_MMAPI);
    } else if ((isScaleVecPostProcess || inputParams.isPertoken || isBf16Mix) && !isAFullLoad) {
        kernelType = static_cast<uint64_t>(QMMKernelType::VEC_EPILOGUE_WITH_MMAPI);
    } else if ((isScaleVecPostProcess || inputParams.isPertoken || isBf16Mix) && isAFullLoad) {
        kernelType = static_cast<uint64_t>(QMMKernelType::VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI);
    } else if (inputParams.isPerBlock) {
        kernelType = static_cast<uint64_t>(QMMKernelType::VEC_EPILOGUE_WITH_CUSTOM_MM);
    }
    return kernelType;
}

}
