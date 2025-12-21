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

namespace optiling {

void QuantBatchMatMulV3TilingUtil::SetBasicTilingData(const QuantBatchMatmulInfo &inputParams,
                                                             const BasicRunInfoTiling &basicTiling,
                                                             DequantBmm::QuantBatchMatmulV3TilingDataParams &tilingData)
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
    tilingData.params.ubCalcN = basicTiling.ubCalcN;
    tilingData.params.ubCalcM = basicTiling.ubCalcM;
    tilingData.params.biasDtype = static_cast<uint32_t>(inputParams.biasDtype);
    tilingData.params.isPerTensor = static_cast<uint32_t>(inputParams.isPerTensor);
    tilingData.params.isPertoken = static_cast<uint32_t>(inputParams.isPertoken);
    tilingData.params.isDoubleScale = static_cast<uint32_t>(inputParams.isDoubleScale);
    tilingData.params.biasThreeDim = static_cast<uint32_t>(inputParams.batchBias > 1UL);
    tilingData.params.groupSizeM = static_cast<uint32_t>(inputParams.groupSizeM);
    tilingData.params.groupSizeN = static_cast<uint32_t>(inputParams.groupSizeN);
    tilingData.params.groupSizeK = static_cast<uint32_t>(inputParams.groupSizeK);
}

void QuantBatchMatMulV3TilingUtil::SetBasicLibApiTiling(const QuantBatchMatmulInfo &inputParams,
                                                        const BasicRunInfoTiling &basicTiling,
                                                        DequantBmm::QuantBatchMatmulV3TilingDataParams &tilingData)
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
    tilingData.matmulTiling.dbL0A = 2;  // db switch, 1: off, 2: on
    tilingData.matmulTiling.dbL0B = 2;  // db switch, 1: off, 2: on
    tilingData.matmulTiling.dbL0C = basicTiling.dbL0c;
}

uint64_t QuantBatchMatMulV3TilingUtil::GetBiasMode(const QuantBatchMatmulInfo &inputParams)
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

uint64_t QuantBatchMatMulV3TilingUtil::GetKernelType(const QuantBatchMatmulInfo &inputParams,
                                                    const BasicRunInfoTiling &basicTiling,  bool isBf16Mix,
                                                    bool isAFullLoad, bool isBFullLoad, bool isABFullLoad)
{
    uint64_t kernelType = 0UL;
    bool isScaleVecPostProcess = inputParams.isPerChannel &&
                                 !(inputParams.scaleDtype == ge::DT_UINT64 || inputParams.scaleDtype == ge::DT_INT64);
    bool isVecPostProcess = (isScaleVecPostProcess || inputParams.isPertoken || inputParams.isPerBlock || isBf16Mix);
    if (basicTiling.iterBatch >= 1U) {
        if (basicTiling.baseM == basicTiling.singleCoreM && basicTiling.baseN == basicTiling.singleCoreN) {
            kernelType = static_cast<uint64_t>(QMMKernelType::NO_VEC_EPILOGUE_WITH_BMMAPI);
        } else {
            kernelType = static_cast<uint64_t>(QMMKernelType::NO_VEC_EPILOGUE_WITH_BMMAPI_NO_BATCH_OUT);
        }
    } else if (!isVecPostProcess && isABFullLoad) {
        kernelType = static_cast<uint64_t>(QMMKernelType::NO_VEC_EPILOGUE_CUSTOM_GMTOABL1_WITH_MMAPI);
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