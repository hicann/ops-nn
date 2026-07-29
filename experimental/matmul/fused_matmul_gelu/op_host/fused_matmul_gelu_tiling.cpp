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
 * \file fused_matmul_gelu_tiling.cpp
 * \brief Tiling for FusedMatmulGelu.
 */

#include <algorithm>
#include <map>

#include "error_util.h"
#include "fused_matmul_gelu_tiling.h"
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "util/math_util.h"

using namespace ge;

namespace {
constexpr size_t INPUT_X_IDX = 0;
constexpr size_t INPUT_WEIGHT_IDX = 1;
constexpr size_t INPUT_BIAS_IDX = 2;
constexpr size_t OUTPUT_Y_IDX = 0;
constexpr size_t ATTR_APPROXIMATE_IDX = 0;

constexpr size_t DIM_INDEX0 = 0;
constexpr size_t DIM_INDEX1 = 1;
constexpr size_t LAST_DIM_OFFSET = 1;
constexpr size_t WEIGHT_DIM_NUM = 2;
constexpr size_t BIAS_DIM_NUM = 1;

constexpr uint64_t BASE_M = 128;
constexpr uint64_t BASE_N = 256;
constexpr uint64_t BASE_K = 64;
constexpr uint64_t DOUBLE_COEF = 2;
constexpr uint64_t BLOCK_DATA_B32 = 8;
constexpr uint64_t WORKSPACE_ALIGN_BYTES = 512;
constexpr uint64_t RESERVED_BUFF_BYTES = 8192;
constexpr uint64_t SYS_WORKSPACE_BYTES = static_cast<uint64_t>(16 * 1024 * 1024);
constexpr uint64_t MAX_K_SIZE = 65534;
constexpr uint64_t MAX_VEC_LOOP_ELEMS = 8192;
constexpr uint32_t BATCH_MODE = 1;

constexpr uint64_t APPROXIMATE_TANH = 1;

const static std::map<ge::DataType, matmul_tiling::DataType> DTYPE_MAP = {
    {ge::DT_FLOAT16, matmul_tiling::DataType::DT_FLOAT16}, {ge::DT_BF16, matmul_tiling::DataType::DT_BF16}};

const static std::map<ge::DataType, uint64_t> BYTES_MAP = {{ge::DT_FLOAT16, 2}, {ge::DT_BF16, 2}};

bool IsSupportDtype(ge::DataType dtype) { return DTYPE_MAP.find(dtype) != DTYPE_MAP.end(); }
} // namespace

namespace optiling {

class FusedMatmulGeluTiling {
public:
    explicit FusedMatmulGeluTiling(gert::TilingContext* context) : tilingContext_(context) {}
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();

private:
    bool CheckAndParseShape();
    bool CheckAndParseInputShape();
    bool CheckOutputShape();
    bool CheckAndParseBiasShape();
    bool CheckAndParseDtype();
    bool CheckAndParseAttr();
    bool GetMatmulTiling();
    void SetTilingKey();
    void SetVectorTiling();
    void FillTilingData();
    void PrintTilingData();

private:
    FusedMatmulGeluTilingData tilingData_;
    gert::TilingContext* tilingContext_ = nullptr;
    const char* opName_ = nullptr;

    uint64_t tilingKey_ = 0;
    uint64_t mSize_ = 0;
    uint64_t kSize_ = 0;
    uint64_t nSize_ = 0;
    uint64_t totalElement_ = 0;

    uint64_t ubSize_ = 0;
    uint64_t l1Size_ = 0;
    uint64_t l0ASize_ = 0;
    uint64_t l0BSize_ = 0;
    uint64_t l0CSize_ = 0;
    uint64_t l2Size_ = 0;

    uint64_t aiVecNum_ = 1;
    uint64_t aiCubeNum_ = 1;
    uint64_t cubeCoreNumAligned_ = 0;

    uint64_t baseM_ = BASE_M;
    uint64_t baseK_ = BASE_K;
    uint64_t baseN_ = BASE_N;

    uint64_t bufSize_ = 0;
    uint64_t vecTasksPerCore_ = 0;
    uint64_t vecTasksTailCore_ = 0;
    uint64_t elemsPerVecLoop_ = 0;

    uint64_t hasBias_ = 0;
    uint64_t approximate_ = APPROXIMATE_TANH;
    uint64_t matmulWorkspaceSize_ = 0;

    ge::DataType inputDtype_ = ge::DT_UNDEFINED;
};

ge::graphStatus FusedMatmulGeluTiling::Init()
{
    opName_ = tilingContext_->GetNodeName();
    OP_LOGD(opName_, "FusedMatmulGeluTiling init.");

    auto platformInfo = platform_ascendc::PlatformAscendC(tilingContext_->GetPlatformInfo());
    aiVecNum_ = platformInfo.GetCoreNumAiv();
    aiCubeNum_ = platformInfo.GetCoreNumAic();

    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0ASize_);
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0BSize_);
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSize_);
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2Size_);

    OP_TILING_CHECK(aiCubeNum_ == 0 || aiVecNum_ == 0,
                    OP_LOGE(opName_, "Invalid core number, aic %lu, aiv %lu.", aiCubeNum_, aiVecNum_),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool FusedMatmulGeluTiling::CheckAndParseShape()
{
    OP_TILING_CHECK(!CheckAndParseInputShape(), OP_LOGE(opName_, "Check and parse input shape failed."), return false);

    OP_TILING_CHECK(!CheckOutputShape(), OP_LOGE(opName_, "Check output shape failed."), return false);

    OP_TILING_CHECK(!CheckAndParseBiasShape(), OP_LOGE(opName_, "Check and parse bias shape failed."), return false);

    totalElement_ = mSize_ * nSize_;
    return true;
}

bool FusedMatmulGeluTiling::CheckAndParseInputShape()
{
    auto xShapePtr = tilingContext_->GetInputShape(INPUT_X_IDX);
    auto weightShapePtr = tilingContext_->GetInputShape(INPUT_WEIGHT_IDX);

    OP_TILING_CHECK(xShapePtr == nullptr, OP_LOGE(opName_, "Get x shape failed."), return false);
    OP_TILING_CHECK(weightShapePtr == nullptr, OP_LOGE(opName_, "Get weight shape failed."), return false);

    auto xShape = xShapePtr->GetStorageShape();
    auto weightShape = weightShapePtr->GetStorageShape();

    OP_TILING_CHECK(xShape.GetDimNum() < 2,
                    OP_LOGE(opName_, "x should be at least 2-D [..., K], but dim num is %zu.", xShape.GetDimNum()),
                    return false);
    OP_TILING_CHECK(weightShape.GetDimNum() != WEIGHT_DIM_NUM,
                    OP_LOGE(opName_, "weight should be 2-D [N, K], but dim num is %zu.", weightShape.GetDimNum()),
                    return false);

    kSize_ = static_cast<uint64_t>(xShape.GetDim(xShape.GetDimNum() - LAST_DIM_OFFSET));
    nSize_ = static_cast<uint64_t>(weightShape.GetDim(DIM_INDEX0));

    OP_TILING_CHECK(kSize_ == 0 || nSize_ == 0,
                    OP_LOGE(opName_, "N/K should be greater than 0, N %lu, K %lu.", nSize_, kSize_), return false);
    OP_TILING_CHECK(kSize_ > MAX_K_SIZE, OP_LOGE(opName_, "K should be <= %lu, but got %lu.", MAX_K_SIZE, kSize_),
                    return false);
    OP_TILING_CHECK(static_cast<uint64_t>(weightShape.GetDim(DIM_INDEX1)) != kSize_,
                    OP_LOGE(opName_, "weight dim1 K should equal x last dim K."), return false);

    mSize_ = 1;
    for (size_t i = 0; i + LAST_DIM_OFFSET < xShape.GetDimNum(); ++i) {
        auto dim = xShape.GetDim(i);
        OP_TILING_CHECK(dim <= 0,
                        OP_LOGE(opName_, "x non-last dim should be greater than 0, but dim[%zu] is %ld.", i, dim),
                        return false);
        mSize_ *= static_cast<uint64_t>(dim);
    }

    return true;
}

bool FusedMatmulGeluTiling::CheckOutputShape()
{
    auto xShapePtr = tilingContext_->GetInputShape(INPUT_X_IDX);
    auto yShapePtr = tilingContext_->GetOutputShape(OUTPUT_Y_IDX);

    OP_TILING_CHECK(xShapePtr == nullptr, OP_LOGE(opName_, "Get x shape failed."), return false);
    OP_TILING_CHECK(yShapePtr == nullptr, OP_LOGE(opName_, "Get y shape failed."), return false);

    auto xShape = xShapePtr->GetStorageShape();
    auto yShape = yShapePtr->GetStorageShape();

    OP_TILING_CHECK(yShape.GetDimNum() != xShape.GetDimNum(), OP_LOGE(opName_, "y dim num should equal x dim num."),
                    return false);

    for (size_t i = 0; i + LAST_DIM_OFFSET < xShape.GetDimNum(); ++i) {
        OP_TILING_CHECK(yShape.GetDim(i) != xShape.GetDim(i),
                        OP_LOGE(opName_, "y dim[%zu] should equal x dim[%zu].", i, i), return false);
    }

    OP_TILING_CHECK(static_cast<uint64_t>(yShape.GetDim(yShape.GetDimNum() - LAST_DIM_OFFSET)) != nSize_,
                    OP_LOGE(opName_, "y last dim should equal weight dim0 N."), return false);

    return true;
}

bool FusedMatmulGeluTiling::CheckAndParseBiasShape()
{
    auto biasShapePtr = tilingContext_->GetOptionalInputShape(INPUT_BIAS_IDX);
    auto biasDescPtr = tilingContext_->GetOptionalInputDesc(INPUT_BIAS_IDX);
    hasBias_ = (biasShapePtr != nullptr && biasDescPtr != nullptr) ? 1UL : 0UL;

    if (hasBias_ == 0) {
        return true;
    }

    auto biasShape = biasShapePtr->GetStorageShape();
    OP_TILING_CHECK(biasShape.GetDimNum() != BIAS_DIM_NUM,
                    OP_LOGE(opName_, "bias should be 1-D [N], but dim num is %zu.", biasShape.GetDimNum()),
                    return false);
    OP_TILING_CHECK(static_cast<uint64_t>(biasShape.GetDim(DIM_INDEX0)) != nSize_,
                    OP_LOGE(opName_, "bias dim0 should equal N."), return false);

    return true;
}

bool FusedMatmulGeluTiling::CheckAndParseDtype()
{
    auto xDesc = tilingContext_->GetInputDesc(INPUT_X_IDX);
    auto weightDesc = tilingContext_->GetInputDesc(INPUT_WEIGHT_IDX);
    auto yDesc = tilingContext_->GetOutputDesc(OUTPUT_Y_IDX);

    OP_TILING_CHECK(xDesc == nullptr || weightDesc == nullptr || yDesc == nullptr,
                    OP_LOGE(opName_, "Get input/output desc failed."), return false);

    inputDtype_ = xDesc->GetDataType();
    auto weightDtype = weightDesc->GetDataType();
    auto yDtype = yDesc->GetDataType();

    OP_TILING_CHECK(!IsSupportDtype(inputDtype_), OP_LOGE(opName_, "x dtype is not supported."), return false);
    OP_TILING_CHECK(weightDtype != inputDtype_, OP_LOGE(opName_, "weight dtype should equal x dtype."), return false);
    OP_TILING_CHECK(yDtype != inputDtype_, OP_LOGE(opName_, "y dtype should equal x dtype."), return false);

    auto biasDescPtr = tilingContext_->GetOptionalInputDesc(INPUT_BIAS_IDX);
    if (hasBias_ != 0 && biasDescPtr != nullptr) {
        OP_TILING_CHECK(biasDescPtr->GetDataType() != inputDtype_, OP_LOGE(opName_, "bias dtype should equal x dtype."),
                        return false);
    }

    return true;
}

bool FusedMatmulGeluTiling::CheckAndParseAttr()
{
    auto attrs = tilingContext_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(opName_, "Get attrs failed."), return false);

    auto approximatePtr = attrs->GetAttrPointer<int64_t>(ATTR_APPROXIMATE_IDX);
    OP_TILING_CHECK(approximatePtr == nullptr, OP_LOGE(opName_, "Get attr approximate failed."), return false);

    approximate_ = static_cast<uint64_t>(*approximatePtr);
    OP_TILING_CHECK(approximate_ != APPROXIMATE_TANH,
                    OP_LOGE(opName_, "approximate should be 1(tanh), but got %lu.", approximate_), return false);

    return true;
}

bool FusedMatmulGeluTiling::GetMatmulTiling()
{
    matmul_tiling::MatmulApiTiling mmTiling;
    mmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, DTYPE_MAP.at(inputDtype_), false);

    // weight is stored as [N, K], and matmul computes x @ weight^T.
    mmTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, DTYPE_MAP.at(inputDtype_), true);

    // Matmul writes intermediate result to user workspace, then vector core applies bias + GELU.
    mmTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, DTYPE_MAP.at(inputDtype_));

    // FP16 uses MatmulImpl bias. BF16 adds bias in AIV epilogue.
    mmTiling.SetBias(hasBias_ != 0 && inputDtype_ == ge::DT_FLOAT16);

    mmTiling.SetOrgShape(mSize_, nSize_, kSize_);
    mmTiling.SetShape(baseM_, baseN_, kSize_);
    mmTiling.SetFixSplit(baseM_, baseN_, baseK_);
    mmTiling.SetBufferSpace(-1, -1, -1);

    OP_TILING_CHECK(mmTiling.GetTiling(tilingData_.mmTiling) == -1, OP_LOGE(opName_, "Get matmul tiling failed."),
                    return false);

    return true;
}

void FusedMatmulGeluTiling::SetTilingKey()
{
    // 1: tanh mode.
    // Dtype is selected by binary json compile macros, not by tiling key.
    // hasBias is a runtime tiling field.
    tilingKey_ = approximate_;
    tilingContext_->SetTilingKey(tilingKey_);
}

void FusedMatmulGeluTiling::SetVectorTiling()
{
    bufSize_ = ubSize_ > RESERVED_BUFF_BYTES ? ubSize_ - RESERVED_BUFF_BYTES : ubSize_;

    uint64_t maxElemsPerLoop = Ops::Base::FloorAlign(bufSize_ / 8 / sizeof(float), BLOCK_DATA_B32);
    elemsPerVecLoop_ = std::min(maxElemsPerLoop, MAX_VEC_LOOP_ELEMS);
    elemsPerVecLoop_ = std::max(elemsPerVecLoop_, BLOCK_DATA_B32);

    uint64_t vecTaskNum = Ops::Base::CeilDiv(totalElement_, elemsPerVecLoop_);
    vecTasksPerCore_ = vecTaskNum / aiVecNum_;
    vecTasksTailCore_ = vecTaskNum % aiVecNum_;
}

void FusedMatmulGeluTiling::FillTilingData()
{
    tilingData_.set_m(mSize_);
    tilingData_.set_k(kSize_);
    tilingData_.set_n(nSize_);
    tilingData_.set_totalElement(totalElement_);
    tilingData_.set_bufSize(bufSize_);
    tilingData_.set_cubeCoreNum(aiCubeNum_);
    tilingData_.set_vecCoreNum(aiVecNum_);
    tilingData_.set_vecTasksPerCore(vecTasksPerCore_);
    tilingData_.set_vecTasksTailCore(vecTasksTailCore_);
    tilingData_.set_elemsPerVecLoop(elemsPerVecLoop_);
    tilingData_.set_hasBias(hasBias_);
    tilingData_.set_approximate(approximate_);
    tilingData_.set_matmulWorkspaceSize(matmulWorkspaceSize_);
    tilingData_.set_cubeCoreNumAligned(cubeCoreNumAligned_);

    // Original CMCT bridge plan.
    // Keep this independent from official MatMulV3BasicTilingData to avoid
    // code-structure duplication while preserving the scheduling information
    // needed by a later CMCT-style kernel bridge.
    tilingData_.set_fmgUsedCoreNum(static_cast<uint32_t>(aiCubeNum_));
    tilingData_.set_fmgML1(static_cast<uint32_t>(baseM_));
    tilingData_.set_fmgNL1(static_cast<uint32_t>(baseN_));
    tilingData_.set_fmgKL1(static_cast<uint32_t>(baseK_));
    tilingData_.set_fmgBaseM(static_cast<uint32_t>(baseM_));
    tilingData_.set_fmgBaseN(static_cast<uint32_t>(baseN_));
    tilingData_.set_fmgBaseK(static_cast<uint32_t>(baseK_));
    tilingData_.set_fmgMTileCnt(static_cast<uint32_t>(Ops::Base::CeilDiv(mSize_, baseM_)));
    tilingData_.set_fmgNTileCnt(static_cast<uint32_t>(Ops::Base::CeilDiv(nSize_, baseN_)));
    tilingData_.set_fmgUseWorkspace(0U);
}

void FusedMatmulGeluTiling::PrintTilingData()
{
    OP_LOGD(opName_, "m: %lu.", tilingData_.get_m());
    OP_LOGD(opName_, "k: %lu.", tilingData_.get_k());
    OP_LOGD(opName_, "n: %lu.", tilingData_.get_n());
    OP_LOGD(opName_, "totalElement: %lu.", tilingData_.get_totalElement());
    OP_LOGD(opName_, "bufSize: %lu.", tilingData_.get_bufSize());
    OP_LOGD(opName_, "cubeCoreNum: %lu.", tilingData_.get_cubeCoreNum());
    OP_LOGD(opName_, "vecCoreNum: %lu.", tilingData_.get_vecCoreNum());
    OP_LOGD(opName_, "vecTasksPerCore: %lu.", tilingData_.get_vecTasksPerCore());
    OP_LOGD(opName_, "vecTasksTailCore: %lu.", tilingData_.get_vecTasksTailCore());
    OP_LOGD(opName_, "elemsPerVecLoop: %lu.", tilingData_.get_elemsPerVecLoop());
    OP_LOGD(opName_, "hasBias: %lu.", tilingData_.get_hasBias());
    OP_LOGD(opName_, "approximate: %lu.", tilingData_.get_approximate());
    OP_LOGD(opName_, "matmulWorkspaceSize: %lu.", tilingData_.get_matmulWorkspaceSize());
    OP_LOGD(opName_, "cubeCoreNumAligned: %lu.", tilingData_.get_cubeCoreNumAligned());
}

ge::graphStatus FusedMatmulGeluTiling::RunKernelTiling()
{
    OP_LOGD(opName_, "TilingForFusedMatmulGelu RunKernelTiling start.");

    if (!CheckAndParseShape()) {
        return ge::GRAPH_FAILED;
    }

    if (!CheckAndParseDtype()) {
        return ge::GRAPH_FAILED;
    }

    if (!CheckAndParseAttr()) {
        return ge::GRAPH_FAILED;
    }

    // Shape-aware tuning for FusedMatmulGelu.
    // Small-N shapes need more N-direction parallelism, but too small baseN
    // fragments Matmul tiles and hurts performance.
    // Measured policy on ascend910b BF16:
    //   M <= 1 and N <= 1024  -> baseN = 128
    //   M > 1  and N <= 1024  -> baseN = 64
    if (nSize_ <= 1024) {
        if (mSize_ <= 1) {
            baseN_ = 128;
        } else {
            baseN_ = 64;
        }
    }

    // baseK=64 is the best overall choice in current BF16 shape sweep.
    baseK_ = 64;

    uint64_t mBlockNum = Ops::Base::CeilDiv(mSize_, baseM_);
    uint64_t nBlockNum = Ops::Base::CeilDiv(nSize_, baseN_);
    uint64_t totalCubeTasks = mBlockNum * nBlockNum;

    if (totalCubeTasks < aiCubeNum_) {
        aiCubeNum_ = std::max(totalCubeTasks, 1UL);
        aiVecNum_ = std::max(aiCubeNum_ * DOUBLE_COEF, 1UL);
    }

    cubeCoreNumAligned_ = Ops::Base::CeilAlign(aiCubeNum_, BLOCK_DATA_B32);

    SetVectorTiling();

    uint64_t dtypeBytes = BYTES_MAP.at(inputDtype_);
    matmulWorkspaceSize_ = Ops::Base::CeilAlign(totalElement_ * dtypeBytes, WORKSPACE_ALIGN_BYTES);

    if (!GetMatmulTiling()) {
        return ge::GRAPH_FAILED;
    }

    SetTilingKey();
    FillTilingData();
    PrintTilingData();

    size_t* workspaces = tilingContext_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(opName_, "Get workspace size failed."), return ge::GRAPH_FAILED);

    // workspace returned to kernel includes system workspace.
    // GetUserWorkspace(workspace) points after SYS_WORKSPACE_BYTES.
    workspaces[0] = SYS_WORKSPACE_BYTES + matmulWorkspaceSize_;

    tilingData_.SaveToBuffer(tilingContext_->GetRawTilingData()->GetData(),
                             tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    tilingContext_->SetBlockDim(aiCubeNum_);
    tilingContext_->SetScheduleMode(BATCH_MODE);

    OP_LOGD(opName_, "TilingForFusedMatmulGelu RunKernelTiling end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForFusedMatmulGelu(gert::TilingContext* context)
{
    FusedMatmulGeluTiling tilingObject(context);
    auto ret = tilingObject.Init();
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "TilingForFusedMatmulGelu Init failed.");
        return ge::GRAPH_FAILED;
    }

    ret = tilingObject.RunKernelTiling();
    OP_LOGD(context->GetNodeName(), "TilingForFusedMatmulGelu end.");
    return ret;
}

ge::graphStatus TilingPrepareForFusedMatmulGelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FusedMatmulGelu)
    .Tiling(TilingForFusedMatmulGelu)
    .TilingParse<FusedMatmulGeluCompileInfo>(TilingPrepareForFusedMatmulGelu);

} // namespace optiling
