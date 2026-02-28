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
 * \file flat_quant_tiling.cpp
 * \brief
 */
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "flat_quant_tiling.h"

namespace optiling {
using namespace Ops::NN::OpTiling;

constexpr uint8_t INPUT_TENSOR_NUM = 3;
constexpr uint8_t INDEX_TWO = 2;
constexpr uint8_t INDEX_ONE = 1;
constexpr uint8_t INDEX_ZERO = 0;
constexpr uint8_t DIM_THREE = 3;
constexpr uint8_t DIM_TWO = 2;

constexpr uint8_t BYTE_LEN_2 = 2;
constexpr uint8_t BYTE_LEN_4 = 4;
constexpr uint8_t CEIL_SIZE = 16;
constexpr int32_t MAX_K_SIZE = 262144;
constexpr int32_t NUM_EIGHT = 8;
constexpr int32_t MAX_MN_SIZE = 256;
constexpr int32_t FACTOR_TWO = 2;
constexpr int32_t K_PER_VEC = 4;
constexpr int32_t BASE_SIZE = 128;
constexpr int32_t L1_SIZE = 512 * 1024;

constexpr float ZERO_FLOAT = 0.0f;
constexpr float ONE_FLOAT = 1.0f;

constexpr uint8_t MM_BASE_MODE = 1;
constexpr uint8_t MM_DOUBLE_MODE = 2;
constexpr uint8_t MM_SPLIT_MODE = 3;
constexpr uint8_t MM_HIGH_MODE = 4;
constexpr uint8_t MM_ONE_MODE = 5;
constexpr uint64_t WORK_SPACE_SIZE = 16 * 1024 * 1024;
constexpr uint64_t WORK_SPACE_SIZE_APT = 48 * 1024 * 1024;

class FlatQuantTiling {
public:
    explicit FlatQuantTiling(gert::TilingContext *context) : tilingContext(context){};
    ge::graphStatus RunBigKernelTiling();

private:
    bool CheckShapes() const;
    bool CheckClipRatio() const;
    void GetKernelMode(int64_t aivNum);
    ge::graphStatus GetTCubeTiling();

    template <typename T1, typename T2>
    inline auto CeilA2B(T1 a, T2 b) const -> T1;

private:
    ge::DataType dataType = ge::DT_UNDEFINED;

    gert::TilingContext *tilingContext = nullptr;
    gert::Shape xShape;
    gert::Shape p1Shape;
    gert::Shape p2Shape;

    uint8_t mmMode = MM_BASE_MODE;

    const float *clipRatio = nullptr;
    FlatQuantTilingData tilingData;
};

ge::graphStatus FlatQuantTiling::RunBigKernelTiling()
{
    // 获取输入矩阵
    auto xTensor = tilingContext->GetInputTensor(INDEX_ZERO);
    auto p1Tensor = tilingContext->GetInputTensor(INDEX_ONE);
    auto p2Tensor = tilingContext->GetInputTensor(INDEX_TWO);
    const gert::RuntimeAttrs *attrs = tilingContext->GetAttrs();
    if (xTensor == nullptr || p1Tensor == nullptr || p2Tensor == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 获取输入的数据类型，输入Tensor的数据类型保持一致
    for (int i = 0; i < INPUT_TENSOR_NUM; i++) {
        auto temp = tilingContext->GetInputDesc(i);
        if (dataType == ge::DT_UNDEFINED) {
            dataType = temp->GetDataType();
        } else if (dataType != temp->GetDataType()) {
            return ge::GRAPH_FAILED;
        }
    }

    // 获取输入的shape
    xShape = tilingContext->GetInputShape(INDEX_ZERO)->GetOriginShape();
    p1Shape = tilingContext->GetInputShape(INDEX_ONE)->GetOriginShape();
    p2Shape = tilingContext->GetInputShape(INDEX_TWO)->GetOriginShape();
    clipRatio = attrs->GetAttrPointer<float>(0);
    if (!CheckShapes() || !CheckClipRatio()) {
        return ge::GRAPH_FAILED;
    }
    tilingData.set_K(xShape.GetDim(INDEX_ZERO));
    tilingData.set_M(xShape.GetDim(INDEX_ONE));
    tilingData.set_N(xShape.GetDim(INDEX_TWO));
    tilingData.set_clipRatio(*clipRatio);
    auto compileInfo = tilingContext->GetCompileInfo<FlatQuantCompileInfo>();
    
    int64_t aivNum = FACTOR_TWO * compileInfo->coreNum;
    GetKernelMode(aivNum);
    if (mmMode == MM_HIGH_MODE && GetTCubeTiling() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    tilingContext->SetBlockDim(compileInfo->coreNum);
    tilingContext->SetTilingKey(mmMode);

    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void FlatQuantTiling::GetKernelMode(int64_t aivNum)
{
    int64_t N = xShape.GetDim(INDEX_TWO);
    int64_t M = xShape.GetDim(INDEX_ONE);
    int64_t K = xShape.GetDim(INDEX_ZERO);
    int64_t Nceil = CeilA2B(N, CEIL_SIZE) * CEIL_SIZE;
    int64_t Mceil = CeilA2B(M, CEIL_SIZE) * CEIL_SIZE;
    auto outDtype = tilingContext->GetOutputDesc(0)->GetDataType();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    // 使用soc和out_dtype区分 mxfp4和int的tiling
    if (outDtype == ge::DT_FLOAT4_E2M1) {
        if (IsRegbaseSocVersion(tilingContext)) {
            mmMode = MM_HIGH_MODE;
            size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
            int64_t useAivNum = CeilA2B(K, K_PER_VEC) <= aivNum ? CeilA2B(K, K_PER_VEC) : aivNum;
            workspaces[0] =
                useAivNum * (K_PER_VEC * M * N * BYTE_LEN_2 + FACTOR_TWO * K_PER_VEC * Mceil * N * BYTE_LEN_4) +
                WORK_SPACE_SIZE_APT;
        }
    } else {
        if (M == 1 && N % NUM_EIGHT == 0) {
            mmMode = MM_ONE_MODE;
        } else if (Mceil <= BASE_SIZE / FACTOR_TWO && Nceil <= BASE_SIZE && K > 1) {
            mmMode = MM_DOUBLE_MODE;
        } else if (Mceil * Mceil + Nceil * Nceil + FACTOR_TWO * FACTOR_TWO * Mceil * Nceil > L1_SIZE / BYTE_LEN_2) {
            mmMode = MM_HIGH_MODE;
        } else if (Mceil > BASE_SIZE || Nceil > BASE_SIZE) {
            mmMode = MM_SPLIT_MODE;
        } else {
            mmMode = MM_BASE_MODE;
        }

        size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
        if (mmMode == MM_HIGH_MODE) {
            int64_t useAivNum = CeilA2B(K, K_PER_VEC) <= aivNum ? CeilA2B(K, K_PER_VEC) : aivNum;
            workspaces[0] =
                useAivNum * (K_PER_VEC * M * N * BYTE_LEN_2 + FACTOR_TWO * K_PER_VEC * Mceil * N * BYTE_LEN_4) +
                WORK_SPACE_SIZE;
        } else if (mmMode == MM_DOUBLE_MODE) {
            K += (K % FACTOR_TWO);
            workspaces[0] = (K * Mceil * N + Mceil * Mceil) * BYTE_LEN_2 + WORK_SPACE_SIZE;
        } else if (mmMode == MM_ONE_MODE) {
            workspaces[0] = K * Mceil * Nceil * BYTE_LEN_2 + WORK_SPACE_SIZE;
        } else {
            workspaces[0] = (K * Mceil * N) * BYTE_LEN_2 + WORK_SPACE_SIZE;
        }
    }
}

ge::graphStatus FlatQuantTiling::GetTCubeTiling()
{
    int64_t N = xShape.GetDim(INDEX_TWO);
    int64_t M = xShape.GetDim(INDEX_ONE);
    int64_t K = xShape.GetDim(INDEX_ZERO);
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);

    matmul_tiling::MatmulApiTiling mmTilingR;
    mmTilingR.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTilingR.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTilingR.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTilingR.SetOrgShape(K * M, N, N);
    mmTilingR.SetShape(K_PER_VEC * M, N, N);
    if (mmTilingR.GetTiling(tilingData.matmulTilingR) == -1) {
        return ge::GRAPH_FAILED;
    }

    matmul_tiling::MatmulApiTiling mmTilingL;
    mmTilingL.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTilingL.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTilingL.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mmTilingL.SetOrgShape(M, N, M);
    mmTilingL.SetShape(M, N, M);
    if (mmTilingL.GetTiling(tilingData.matmulTilingL) == -1) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool FlatQuantTiling::CheckShapes() const
{
    if (xShape.GetDimNum() != DIM_THREE || p1Shape.GetDimNum() != DIM_TWO || p2Shape.GetDimNum() != DIM_TWO) {
        return false;
    }
    int64_t K = xShape.GetDim(INDEX_ZERO);
    int64_t M = xShape.GetDim(INDEX_ONE);
    int64_t N = xShape.GetDim(INDEX_TWO);
    if (K > MAX_K_SIZE || M > MAX_MN_SIZE || N > MAX_MN_SIZE) {
        return false;
    }
    if(p1Shape.GetDim(INDEX_ZERO) != M || p1Shape.GetDim(INDEX_ONE) != M || p2Shape.GetDim(INDEX_ZERO) != N || p2Shape.GetDim(INDEX_ONE) != N) {
        return false;
    }
    return true;
}

bool FlatQuantTiling::CheckClipRatio() const
{
    return clipRatio != nullptr && *clipRatio > ZERO_FLOAT && *clipRatio <= ONE_FLOAT;
}

template <typename T1, typename T2>
inline auto FlatQuantTiling::CeilA2B(T1 a, T2 b) const -> T1
{
    if (b != 0) {
        return (a + b - 1) / b;
    } else {
        return a;
    }
}

static ge::graphStatus Tiling4FlatQuantTiling(gert::TilingContext *context)
{
    FlatQuantTiling tilingObject(context);
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus TilingPrepareTiling(gert::TilingParseContext *context)
{
    auto compileInfo =context->GetCompiledInfo<FlatQuantCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAic();

    OP_CHECK_IF(compileInfo->coreNum <= 0,
        OP_LOGE(context->GetNodeName(), "FlatQuanrtTiling GetHardwareInfo Failed"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FlatQuant).Tiling(Tiling4FlatQuantTiling).TilingParse<FlatQuantCompileInfo>(TilingPrepareTiling);
}  // namespace optiling
