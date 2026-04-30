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
#include "error_util.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "flat_quant_tiling.h"
#include "platform/soc_spec.h"

namespace optiling {
using namespace Ops::NN::OpTiling;

constexpr uint8_t INPUT_TENSOR_NUM = 3;
constexpr uint8_t INDEX_TWO = 2;
constexpr uint8_t INDEX_ONE = 1;
constexpr uint8_t INDEX_ZERO = 0;
constexpr uint8_t INDEX_THREE = 3;
constexpr uint8_t DIM_THREE = 3;
constexpr uint8_t DIM_TWO = 2;
constexpr uint8_t WORKSPACE_NUM = 1;

constexpr uint8_t BYTE_LEN_2 = 2;
constexpr uint8_t BYTE_LEN_4 = 4;
constexpr uint8_t CEIL_SIZE = 16;
constexpr int32_t MAX_K_SIZE = 262144;
constexpr int32_t NUM_EIGHT = 8;
constexpr int32_t MAX_MN_SIZE = 256;
constexpr int32_t FACTOR_TWO = 2;
constexpr int32_t DATA_TYPE_SIZE = 2;
constexpr int32_t K_PER_VEC = 4;
constexpr int32_t BASE_K = 64;
constexpr int32_t BASE_SIZE = 128;
constexpr int32_t L1_SIZE = 512 * 1024;
constexpr int32_t L0_SIZE = 64 * 1024;
constexpr int32_t L0C_SIZE = 256 * 1024;

constexpr float ZERO_FLOAT = 0.0f;
constexpr float ONE_FLOAT = 1.0f;
constexpr float SIX_FLOAT = 6.0f;
constexpr float TWELVE_FLOAT = 12.0f;

constexpr uint8_t MM_BASE_MODE = 1;
constexpr uint8_t MM_DOUBLE_MODE = 2;
constexpr uint8_t MM_SPLIT_MODE = 3;
constexpr uint8_t MM_HIGH_MODE = 4;
constexpr uint8_t MM_ONE_MODE = 5;
constexpr uint8_t MM_HIGH_MODE_ALIGN = 6;
constexpr uint64_t WORK_SPACE_SIZE = 16 * 1024 * 1024;
constexpr uint64_t WORK_SPACE_SIZE_APT = 48 * 1024 * 1024;

class FlatQuantTiling {
public:
    explicit FlatQuantTiling(gert::TilingContext* context) : tilingContext_(context) {};
    ge::graphStatus RunBigKernelTiling();

private:
    bool CheckShapes();
    bool CheckClipRatio() const;
    bool CheckDstTypeMax() const;
    bool CheckDstDtype() const;
    void GetKernelMode(int64_t aivNum);
    ge::graphStatus GetTCubeTiling();
    ge::graphStatus InitializeInputsAndAttributes();
    ge::graphStatus SetBasicTilingData();
    ge::graphStatus CalculateIterBatch();
    ge::graphStatus SetTilingContextAndSaveData();

    template <typename T1, typename T2>
    inline auto CeilA2B(T1 a, T2 b) const -> T1;

private:
    ge::DataType dataType_ = ge::DT_UNDEFINED;

    gert::TilingContext* tilingContext_ = nullptr;
    gert::Shape xShape_;
    gert::Shape p1Shape_;
    gert::Shape p2Shape_;

    uint64_t mAlign_ = 0;
    uint64_t nAlign_ = 0;
    uint64_t iterBatch_ = 0;

    uint8_t mmMode_ = MM_BASE_MODE;
    bool hasP2_ = true;

    const float* clipRatio_ = nullptr;
    const int64_t* dstDtype_ = nullptr;
    const float* dstTypeMax_ = nullptr;
    FlatQuantTilingData tilingData_;
};

ge::graphStatus FlatQuantTiling::InitializeInputsAndAttributes()
{
    // 获取输入矩阵
    auto xTensor = tilingContext_->GetInputTensor(INDEX_ZERO);
    auto p1Tensor = tilingContext_->GetInputTensor(INDEX_ONE);
    auto p2Tensor = tilingContext_->GetInputTensor(INDEX_TWO);
    const gert::RuntimeAttrs* attrs = tilingContext_->GetAttrs();
    if (xTensor == nullptr || p1Tensor == nullptr || p2Tensor == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // 获取输入的数据类型，输入Tensor的数据类型保持一致
    for (uint64_t i = 0; i < INPUT_TENSOR_NUM; i++) {
        auto temp = tilingContext_->GetInputDesc(i);
        if (dataType_ == ge::DT_UNDEFINED) {
            dataType_ = temp->GetDataType();
        } else if (dataType_ != temp->GetDataType()) {
            return ge::GRAPH_FAILED;
        }
    }
    // 获取输入的shape和属性
    xShape_ = tilingContext_->GetInputShape(INDEX_ZERO)->GetOriginShape();
    p1Shape_ = tilingContext_->GetInputShape(INDEX_ONE)->GetOriginShape();
    p2Shape_ = tilingContext_->GetInputShape(INDEX_TWO)->GetOriginShape();
    hasP2_ = (p2Shape_.GetDim(INDEX_ZERO) > 0 && p2Shape_.GetDim(INDEX_ONE) > 0);
    clipRatio_ = attrs->GetAttrPointer<float>(INDEX_ZERO);
    dstDtype_ = attrs->GetAttrPointer<int64_t>(INDEX_ONE);
    dstTypeMax_ = attrs->GetAttrPointer<float>(INDEX_TWO);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlatQuantTiling::SetBasicTilingData()
{
    int64_t M = xShape_.GetDim(INDEX_ONE);
    int64_t N = xShape_.GetDim(INDEX_TWO);
    nAlign_ = CeilA2B(N, CEIL_SIZE) * CEIL_SIZE;
    mAlign_ = CeilA2B(M, CEIL_SIZE) * CEIL_SIZE;
    if (!CheckShapes() || !CheckClipRatio() || !CheckDstDtype() || (dstTypeMax_ != nullptr && !CheckDstTypeMax())) {
        return ge::GRAPH_FAILED;
    }
    // 设置基本的tiling数据
    tilingData_.set_hasP2(hasP2_ ? 1 : 0);
    tilingData_.set_K(xShape_.GetDim(INDEX_ZERO));
    tilingData_.set_M(xShape_.GetDim(INDEX_ONE));
    tilingData_.set_N(xShape_.GetDim(INDEX_TWO));
    tilingData_.set_clipRatio(*clipRatio_);
    // 仅当dstTypeMax不为空时才设置相关字段，否则传默认值0
    if (dstTypeMax_ != nullptr) {
        tilingData_.set_dstTypeMax(*dstTypeMax_);
        tilingData_.set_invDstTypeMax(*dstTypeMax_ != ZERO_FLOAT ? ONE_FLOAT / *dstTypeMax_ : ZERO_FLOAT);
    } else {
        tilingData_.set_dstTypeMax(ZERO_FLOAT);
        tilingData_.set_invDstTypeMax(ZERO_FLOAT);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlatQuantTiling::CalculateIterBatch()
{
    auto compileInfo = tilingContext_->GetCompileInfo<FlatQuantCompileInfo>();
    int64_t aicNum = compileInfo->coreNum;
    int64_t aivNum = compileInfo->aivNum;
    GetKernelMode(aivNum);
    if (mmMode_ == MM_HIGH_MODE && GetTCubeTiling() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // 计算迭代批次
    uint64_t p1Size = nAlign_ * nAlign_ * DATA_TYPE_SIZE;
    uint64_t p2Size = mAlign_ * mAlign_ * DATA_TYPE_SIZE;
    uint64_t xInputSize = mAlign_ * nAlign_ * DATA_TYPE_SIZE;
    uint64_t xL0CSize = mAlign_ * nAlign_ * sizeof(float);
    uint64_t iterBatchL1 = (L1_SIZE - p1Size - p2Size) / xInputSize / FACTOR_TWO;
    uint64_t iterBatchL0 = L0_SIZE / std::max(mAlign_, nAlign_) / BASE_K / DATA_TYPE_SIZE / FACTOR_TWO;
    uint64_t iterBatchL0C = L0C_SIZE / xL0CSize / FACTOR_TWO;
    uint64_t iterBatchK = CeilA2B(xShape_.GetDim(INDEX_ZERO), aicNum); // 根据K计算的batch
    iterBatch_ = std::max(std::min({iterBatchL1, iterBatchL0, iterBatchL0C, iterBatchK}), 1UL);
    tilingData_.set_iterBatch(iterBatch_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlatQuantTiling::SetTilingContextAndSaveData()
{
    auto compileInfo = tilingContext_->GetCompileInfo<FlatQuantCompileInfo>();
    tilingContext_->SetBlockDim(compileInfo->coreNum);
    tilingContext_->SetTilingKey(mmMode_);
    // 保存tiling数据
    tilingData_.SaveToBuffer(
        tilingContext_->GetRawTilingData()->GetData(), tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlatQuantTiling::RunBigKernelTiling()
{
    // 初始化输入和属性
    if (InitializeInputsAndAttributes() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // 设置基本的tiling数据
    if (SetBasicTilingData() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // 计算迭代批次
    if (CalculateIterBatch() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // 设置tiling上下文并保存数据
    if (SetTilingContextAndSaveData() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void FlatQuantTiling::GetKernelMode(int64_t aivNum)
{
    int64_t N = xShape_.GetDim(INDEX_TWO);
    int64_t M = xShape_.GetDim(INDEX_ONE);
    int64_t K = xShape_.GetDim(INDEX_ZERO);

    auto outDtype = tilingContext_->GetOutputDesc(INDEX_ZERO)->GetDataType();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext_->GetPlatformInfo());
    // 使用soc和out_dtype区分 mxfp4和int的tiling
    if (outDtype == ge::DT_FLOAT4_E2M1) {
        if (IsRegbaseSocVersion(tilingContext_)) {
            mmMode_ = MM_HIGH_MODE;
            if ((N % CEIL_SIZE == 0) && (M % CEIL_SIZE == 0)) {
                mmMode_ = MM_HIGH_MODE_ALIGN;
            }
            size_t* workspaces = tilingContext_->GetWorkspaceSizes(WORKSPACE_NUM);
            int64_t useAivNum = CeilA2B(K, K_PER_VEC) <= aivNum ? CeilA2B(K, K_PER_VEC) : aivNum;
            workspaces[0] =
                useAivNum * (K_PER_VEC * M * N * BYTE_LEN_2 + FACTOR_TWO * K_PER_VEC * mAlign_ * N * BYTE_LEN_4) +
                WORK_SPACE_SIZE_APT;
        }
    } else {
        if (M == 1 && N % NUM_EIGHT == 0) {
            mmMode_ = MM_ONE_MODE;
        } else if (mAlign_ <= BASE_SIZE / FACTOR_TWO && nAlign_ <= BASE_SIZE && K > 1) {
            mmMode_ = MM_DOUBLE_MODE;
        } else if (
            mAlign_ * mAlign_ + nAlign_ * nAlign_ + FACTOR_TWO * FACTOR_TWO * mAlign_ * nAlign_ >
            L1_SIZE / BYTE_LEN_2) {
            mmMode_ = MM_HIGH_MODE;
        } else if (mAlign_ > BASE_SIZE || nAlign_ > BASE_SIZE) {
            mmMode_ = MM_SPLIT_MODE;
        } else {
            mmMode_ = MM_BASE_MODE;
        }

        size_t* workspaces = tilingContext_->GetWorkspaceSizes(WORKSPACE_NUM);
        if (mmMode_ == MM_HIGH_MODE) {
            int64_t useAivNum = CeilA2B(K, K_PER_VEC) <= aivNum ? CeilA2B(K, K_PER_VEC) : aivNum;
            workspaces[0] =
                useAivNum * (K_PER_VEC * M * N * BYTE_LEN_2 + FACTOR_TWO * K_PER_VEC * mAlign_ * N * BYTE_LEN_4) +
                WORK_SPACE_SIZE;
        } else if (mmMode_ == MM_DOUBLE_MODE) {
            K += (K % FACTOR_TWO);
            workspaces[0] = (K * mAlign_ * N + mAlign_ * mAlign_) * BYTE_LEN_2 + WORK_SPACE_SIZE;
        } else if (mmMode_ == MM_ONE_MODE) {
            workspaces[0] = K * mAlign_ * nAlign_ * BYTE_LEN_2 + WORK_SPACE_SIZE;
        } else {
            workspaces[0] = (K * mAlign_ * N) * BYTE_LEN_2 + WORK_SPACE_SIZE;
        }
    }
}

ge::graphStatus FlatQuantTiling::GetTCubeTiling()
{
    int64_t N = xShape_.GetDim(INDEX_TWO);
    int64_t M = xShape_.GetDim(INDEX_ONE);
    int64_t K = xShape_.GetDim(INDEX_ZERO);
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType_);

    matmul_tiling::MatmulApiTiling mmTilingR;
    mmTilingR.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTilingR.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTilingR.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTilingR.SetOrgShape(K * M, N, N);
    mmTilingR.SetShape(K_PER_VEC * M, N, N);
    if (hasP2_ && mmTilingR.GetTiling(tilingData_.matmulTilingR) == -1) {
        return ge::GRAPH_FAILED;
    }

    matmul_tiling::MatmulApiTiling mmTilingL;
    mmTilingL.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTilingL.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTilingL.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mmTilingL.SetOrgShape(M, N, M);
    mmTilingL.SetShape(M, N, M);
    if (mmTilingL.GetTiling(tilingData_.matmulTilingL) == -1) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool FlatQuantTiling::CheckShapes()
{
    if (xShape_.GetDimNum() != DIM_THREE || p1Shape_.GetDimNum() != DIM_TWO ||
        (hasP2_ && p2Shape_.GetDimNum() != DIM_TWO)) {
        return false;
    }
    int64_t K = xShape_.GetDim(INDEX_ZERO);
    int64_t M = xShape_.GetDim(INDEX_ONE);
    int64_t N = xShape_.GetDim(INDEX_TWO);
    if (K > MAX_K_SIZE || M > MAX_MN_SIZE || N > MAX_MN_SIZE) {
        return false;
    }
    if (p1Shape_.GetDim(INDEX_ZERO) != M || p1Shape_.GetDim(INDEX_ONE) != M ||
        (hasP2_ && (p2Shape_.GetDim(INDEX_ZERO) != N || p2Shape_.GetDim(INDEX_ONE) != N))) {
        return false;
    }
    return true;
}

bool FlatQuantTiling::CheckClipRatio() const
{
    return clipRatio_ != nullptr && *clipRatio_ > ZERO_FLOAT && *clipRatio_ <= ONE_FLOAT;
}

bool FlatQuantTiling::CheckDstDtype() const
{
    auto outDtype = tilingContext_->GetOutputDesc(INDEX_ZERO)->GetDataType();
    if (dstDtype_ != nullptr) {
        if (outDtype == ge::DT_FLOAT4_E2M1) {
            return *dstDtype_ == ge::DT_FLOAT4_E2M1;
        }
    }
    return true;
}

bool FlatQuantTiling::CheckDstTypeMax() const
{
    auto outDtype = tilingContext_->GetOutputDesc(0)->GetDataType();
    if (outDtype == ge::DT_FLOAT4_E2M1) {
        float localDstTypeMax = *dstTypeMax_;
        if ((localDstTypeMax != ZERO_FLOAT) && (localDstTypeMax < SIX_FLOAT || localDstTypeMax > TWELVE_FLOAT)) {
            OP_LOGE(
                tilingContext_->GetNodeName(), "The dst_type_max[%f] must be 0 or in range [6, 12], please check.",
                localDstTypeMax);
            return false;
        }
    }
    return true;
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

static ge::graphStatus Tiling4FlatQuantTiling(gert::TilingContext* context)
{
    FlatQuantTiling tilingObject(context);
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus TilingPrepareTiling(gert::TilingParseContext* context)
{
    OP_TILING_CHECK(context == nullptr, "FlatQuant context is null", return ge::GRAPH_FAILED);
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, "FlatQuant platformInfoPtr is null", return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    auto compileInfo = context->GetCompiledInfo<FlatQuantCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    compileInfo->coreNum = ascendcPlatform.GetCoreNumAic();
    compileInfo->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo->socVersion = ascendcPlatform.GetSocVersion();
    NpuArch npuArch = ascendcPlatform.GetCurNpuArch();
    bool IsArch3510 = npuArch == NpuArch::DAV_3510;

    OP_LOGI(
        "FlatQuant", "parse compile info success soc:%d, aicNum:%lu, aivNum:%lu",
        static_cast<int>(compileInfo->socVersion), compileInfo->coreNum, compileInfo->aivNum);
    OP_CHECK_IF(
        compileInfo->coreNum <= 0, OP_LOGE(context->GetNodeName(), "FlatQuant is not supported for aicNum <=0 "),
        return ge::GRAPH_FAILED);
    if (IsArch3510) {
        OP_CHECK_IF(
            compileInfo->aivNum != (compileInfo->coreNum * 2), // aivNum must == aicNum*2
            OP_LOGE(context->GetNodeName(), "FlatQuantTiling is only supported for aivNum == aicNum*2 "),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FlatQuant).Tiling(Tiling4FlatQuantTiling).TilingParse<FlatQuantCompileInfo>(TilingPrepareTiling);
} // namespace optiling
