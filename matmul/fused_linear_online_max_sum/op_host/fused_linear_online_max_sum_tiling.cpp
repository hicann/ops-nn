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
 * \file fused_linear_online_max_sum_tiling.cpp
 * \brief
 */

#include <map>
#include <vector>
#include "error_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "fused_linear_online_max_sum_tiling.h"

using namespace ge;
using namespace std;

namespace {
constexpr size_t INPUT_INPUT_IDX = 0;
constexpr size_t INPUT_WEIGHT_IDX = 1;
constexpr size_t INPUT_TARGET_IDX = 2;

constexpr size_t ATTR_VOCAB_START_IDX = 0;
constexpr size_t ATTR_VOCAB_END_IDX = 1;
constexpr size_t ATTR_IS_VOCAB_PARALLEL_LOGITS_OUT_IDX = 2;

constexpr size_t DIM_INDEX0 = 0;
constexpr size_t DIM_INDEX1 = 1;
constexpr size_t EXPECTED_DIM_NUM_ONE = 1;
constexpr size_t EXPECTED_DIM_NUM_TWO = 2;

constexpr uint64_t BASE_M = 128;
constexpr uint64_t BASE_N = 256;
constexpr uint64_t MAX_INPUT_DIM_ONE_SIZE = 65534;
constexpr uint64_t DOUBLE_COF = 2;
constexpr uint64_t NUM_THREE = 3;
constexpr uint64_t BYTES_B32 = 4;
constexpr uint64_t BYTES_B16 = 2;
constexpr uint64_t INIT_TYPE_NUM = 2;
constexpr uint64_t BLOCK_DATA_B32 = 8;
constexpr uint64_t DATA_PER_UINT8 = 8;
constexpr uint64_t REPEAT_BYTES = 256;
constexpr uint64_t MAX_REPEAT_TIMES = 255;
constexpr uint64_t RESVERD_BUFF_BYTES = 8192;
constexpr uint64_t SYS_WORKSPACE_BYTES = static_cast<uint64_t>(16 * 1024 * 1024);

const static std::map<ge::DataType, matmul_tiling::DataType> DTYPE_MAP =
{
    {ge::DT_FLOAT16, matmul_tiling::DataType::DT_FLOAT16},
    {ge::DT_FLOAT, matmul_tiling::DataType::DT_FLOAT},
    {ge::DT_BF16, matmul_tiling::DataType::DT_BF16}
};
const static std::map<ge::DataType, uint64_t> BYTES_MAP =
{
    {ge::DT_FLOAT16, 2},
    {ge::DT_FLOAT, 4},
    {ge::DT_INT32, 4},
    {ge::DT_INT64, 8},
    {ge::DT_BF16, 2}
};
} // namespace

namespace optiling {
class FusedLinearOnlineMaxSumTiling {
public:
    explicit FusedLinearOnlineMaxSumTiling(gert::TilingContext* context) : tilingContext_(context){};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();
private:
    FusedLinearOnlineMaxSumTilingData tilingData_;
    gert::TilingContext* tilingContext_ = nullptr;
    bool GetAttrAndCheck();
    bool CheckShapeInfoValid();
    void SetTilingKey();
    void InitWorkspaceTiling();
    void TargetTiling();
    bool GetMatmulTiling();
    void FillTilingData();
    void PrintTilingData();
    const char *opName_ = nullptr;
    uint64_t tilingKey_ = 0;
    uint64_t btSize_ = 0;
    uint64_t hSize_ = 0;
    uint64_t vSize_ = 0;

    uint64_t ubSize_ = 0;
    uint64_t l1Size_ = 0;
    uint64_t l0ASize_ = 0;
    uint64_t l0BSize_ = 0;
    uint64_t l0CSize_ = 0;
    uint64_t l2Size_ = 0;
    uint64_t aiVecNum_ = 2;
    uint64_t aiCubeNum_ = 1;

    uint64_t baseM_ = 128;
    uint64_t baseK_ = 64;
    uint64_t baseN_ = 256;

    uint64_t bufSize_ = 0;
    uint64_t batchTaksPerVecCore_ = 0;
    uint64_t batchTaksTailVecCore_ = 0;
    uint64_t initWorkspaceLength_ = 0;
    uint64_t targetTasksPerLoop_ = 0;
    uint64_t cubeCoreNumAligned_ = 0;
    uint64_t matmulInputEmptyFlag_ = 1;

    float vocabStartIndex_ = 0;
    float vocabEndIndex_ = 0;
    uint64_t vocabParallelLogitsOutFlag_ = 1;
};

ge::graphStatus FusedLinearOnlineMaxSumTiling::Init() {
    opName_ = tilingContext_->GetNodeName();
    OP_LOGD(opName_, "FusedLinearOnlineMaxSumTiling init.");
    auto platformInfo = platform_ascendc::PlatformAscendC(tilingContext_->GetPlatformInfo());

    aiVecNum_ = platformInfo.GetCoreNumAiv();
    aiCubeNum_ = platformInfo.GetCoreNumAic();
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0ASize_);
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0BSize_);
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSize_);
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2Size_);
    return ge::GRAPH_SUCCESS;
}

bool FusedLinearOnlineMaxSumTiling::CheckShapeInfoValid() {
    auto inputShape = tilingContext_->GetInputShape(INPUT_INPUT_IDX)->GetStorageShape();
    auto weightShape = tilingContext_->GetInputShape(INPUT_WEIGHT_IDX)->GetStorageShape();
    auto targetShape = tilingContext_->GetInputShape(INPUT_TARGET_IDX)->GetStorageShape();
    OP_TILING_CHECK(
        inputShape.GetDimNum() != EXPECTED_DIM_NUM_TWO,
        OP_LOGE(opName_, "the dim of input should be 2."), return false);
    
    OP_TILING_CHECK(
        weightShape.GetDimNum() != EXPECTED_DIM_NUM_TWO,
        OP_LOGE(opName_, "the dim of weight should be 2."), return false);
    
    OP_TILING_CHECK(
        targetShape.GetDimNum() != EXPECTED_DIM_NUM_ONE,
        OP_LOGE(opName_, "the dim of target should be 1."), return false);
    btSize_ = inputShape.GetDim(DIM_INDEX0);
    hSize_ = inputShape.GetDim(DIM_INDEX1);
    vSize_ = weightShape.GetDim(DIM_INDEX0);

    OP_TILING_CHECK(
        hSize_ > MAX_INPUT_DIM_ONE_SIZE,
        OP_LOGE(opName_, "input.size(1) should be less than or equal to 65534."),
        return false);

    OP_TILING_CHECK(
        static_cast<uint64_t>(weightShape.GetDim(DIM_INDEX1)) != hSize_,
        OP_LOGE(opName_, "weight.size(1) should be equal to input.size(1)."),
        return false);

    OP_TILING_CHECK(
        static_cast<uint64_t>(targetShape.GetDim(DIM_INDEX0)) != btSize_,
        OP_LOGE(opName_, "the size of target should be equal to input.size(0)."),
        return false);

    return true;
}

void FusedLinearOnlineMaxSumTiling::InitWorkspaceTiling() {
    initWorkspaceLength_ = Ops::Base::FloorAlign(bufSize_ / BYTES_B32 / INIT_TYPE_NUM, BLOCK_DATA_B32);
    uint64_t btSizeAligned = Ops::Base::CeilAlign((btSize_ + 1) / DOUBLE_COF, BLOCK_DATA_B32);
    initWorkspaceLength_ = btSizeAligned < initWorkspaceLength_ ? btSizeAligned : initWorkspaceLength_;
}

void FusedLinearOnlineMaxSumTiling::SetTilingKey() {
    tilingKey_ = vocabParallelLogitsOutFlag_;
    tilingContext_->SetTilingKey(tilingKey_);
}

void FusedLinearOnlineMaxSumTiling::TargetTiling() {
    uint64_t batchTasks = Ops::Base::CeilDiv(btSize_, BLOCK_DATA_B32);
    batchTaksPerVecCore_ = batchTasks / aiVecNum_;
    batchTaksTailVecCore_ = batchTasks % aiVecNum_;

    uint64_t targetBytes = BYTES_MAP.at(tilingContext_->GetInputDesc(INPUT_TARGET_IDX)->GetDataType());
    uint64_t bytesAligned = Ops::Base::FloorAlign(bufSize_ / NUM_THREE, REPEAT_BYTES);
    bytesAligned = bytesAligned < MAX_REPEAT_TIMES * REPEAT_BYTES ? bytesAligned : MAX_REPEAT_TIMES * REPEAT_BYTES;
    uint64_t maxTasksPerLoop = bytesAligned / std::max(targetBytes, static_cast<uint64_t>(1)) / DATA_PER_UINT8;
    targetTasksPerLoop_ = maxTasksPerLoop;
}

bool FusedLinearOnlineMaxSumTiling::GetMatmulTiling() {
    auto inputDataType = tilingContext_->GetInputDesc(INPUT_INPUT_IDX)->GetDataType();
    matmul_tiling::MatmulApiTiling mmTiling;
    mmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, DTYPE_MAP.at(inputDataType), false);
    mmTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, DTYPE_MAP.at(inputDataType), true);
    mmTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, DTYPE_MAP.at(inputDataType));
    mmTiling.SetBias(false);
    mmTiling.SetOrgShape(btSize_, vSize_, hSize_); // 设置Matmul计算时的原始完整的形状M、N、K或Ka/Kb，单位均为元素个数。
    mmTiling.SetShape(baseM_, baseN_, hSize_); // 设置Matmul计算的形状m、n、k，该形状可以为原始完整矩阵或其局部矩阵，单位为元素个数。该形状的矩阵乘可以由单核或多核计算完成。
    mmTiling.SetFixSplit(baseM_, baseN_, baseK_); // 设置固定的baseM、baseN、baseK，单位为元素个数。
    mmTiling.SetBufferSpace(-1, -1, -1); // 设置Matmul计算时可用的L1 Buffer/L0C Buffer/Unified Buffer/BiasTable Buffer空间大小，单位为字节。

    if (mmTiling.GetTiling(tilingData_.mmTiling) == -1) {
        return false;
    }
    return true;
}

bool FusedLinearOnlineMaxSumTiling::GetAttrAndCheck() {
    auto attrs = tilingContext_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(opName_, "Get attrs Failed."),
                    return false);
    vocabStartIndex_ = static_cast<float>(*(attrs->GetAttrPointer<int64_t>(ATTR_VOCAB_START_IDX)));
    OP_TILING_CHECK(vocabStartIndex_ < 0,
                    OP_LOGE(opName_, "vocabStartIndex %f should not be smaller than 0.", vocabStartIndex_),
                    return false);
    vocabEndIndex_ = static_cast<float>(*(attrs->GetAttrPointer<int64_t>(ATTR_VOCAB_END_IDX)));
    OP_TILING_CHECK(vocabEndIndex_ < vocabStartIndex_,
                    OP_LOGE(opName_, "vocabEndIndex %f should be greater than or equal to vocabStartIndex %f.",
                            vocabEndIndex_, vocabStartIndex_),
                    return false);
    vocabParallelLogitsOutFlag_ =
        static_cast<uint64_t>(*(attrs->GetAttrPointer<bool>(ATTR_IS_VOCAB_PARALLEL_LOGITS_OUT_IDX)));
    return true;
}

ge::graphStatus FusedLinearOnlineMaxSumTiling::RunKernelTiling() {
    OP_LOGD(opName_, "TilingForFusedLinearOnlineMaxSum RunKernelTiling start.");
    bufSize_ = ubSize_ - RESVERD_BUFF_BYTES;

    if (!CheckShapeInfoValid()) {
        return ge::GRAPH_FAILED;
    }

    if (!GetAttrAndCheck()) {
        return ge::GRAPH_FAILED;
    }

    uint64_t mBlockNum = Ops::Base::CeilDiv(btSize_, baseM_);
    uint64_t nBlockNum = Ops::Base::CeilDiv(vSize_, baseN_);
    uint64_t totalTasks = mBlockNum * nBlockNum;
    if (totalTasks < aiCubeNum_) {
        aiCubeNum_ = totalTasks;
        aiVecNum_ = aiCubeNum_ * DOUBLE_COF;
    }
    cubeCoreNumAligned_ = Ops::Base::CeilAlign(aiCubeNum_, BLOCK_DATA_B32);
    InitWorkspaceTiling();
    TargetTiling();
    
    if (hSize_ > static_cast<uint64_t>(0)) {
        matmulInputEmptyFlag_ = static_cast<uint64_t>(0);
        if (!GetMatmulTiling()) {
            return ge::GRAPH_FAILED;
        }
    }

    SetTilingKey();
    FillTilingData();
    PrintTilingData();

    size_t* workspaces = tilingContext_->GetWorkspaceSizes(1);
    workspaces[0] = SYS_WORKSPACE_BYTES;
    workspaces[0] += btSize_ * BYTES_B32 * aiCubeNum_ * DOUBLE_COF; // high_performance
    if (!static_cast<bool>(vocabParallelLogitsOutFlag_)) {
        workspaces[0] += BASE_M * BASE_N * BYTES_B16 * aiCubeNum_ * DOUBLE_COF; // low memory
    }

    tilingData_.SaveToBuffer(tilingContext_->GetRawTilingData()->GetData(),
                             tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    tilingContext_->SetBlockDim(aiCubeNum_);
    
    OP_LOGD(opName_, "TilingForFusedLinearOnlineMaxSum RunKernelTiling end.");
    return ge::GRAPH_SUCCESS;
}

void FusedLinearOnlineMaxSumTiling::FillTilingData() {
    tilingData_.set_m(btSize_);
    tilingData_.set_k(hSize_);
    tilingData_.set_n(vSize_);
    tilingData_.set_bufSize(bufSize_);
    tilingData_.set_cubeCoreNum(aiCubeNum_);
    tilingData_.set_vecCoreNum(aiVecNum_);
    tilingData_.set_batchTaksPerVecCore(batchTaksPerVecCore_);
    tilingData_.set_batchTaksTailVecCore(batchTaksTailVecCore_);
    tilingData_.set_targetTasksPerLoop(targetTasksPerLoop_);
    tilingData_.set_vocabStartIndex(vocabStartIndex_);
    tilingData_.set_vocabEndIndex(vocabEndIndex_);
    tilingData_.set_initWorkspaceLength(initWorkspaceLength_);
    tilingData_.set_matmulInputEmptyFlag(matmulInputEmptyFlag_);
}

void FusedLinearOnlineMaxSumTiling::PrintTilingData() {
    OP_LOGD(opName_, "m:  %lu.", tilingData_.get_m());
    OP_LOGD(opName_, "k:  %lu.", tilingData_.get_k());
    OP_LOGD(opName_, "n:  %lu.", tilingData_.get_n());
    OP_LOGD(opName_, "bufSize:  %lu.", tilingData_.get_bufSize());
    OP_LOGD(opName_, "cubeCoreNum:  %lu.", tilingData_.get_cubeCoreNum());
    OP_LOGD(opName_, "vecCoreNum:  %lu.", tilingData_.get_vecCoreNum());
    OP_LOGD(opName_, "batchTaksPerVecCore:  %lu.", tilingData_.get_batchTaksPerVecCore());
    OP_LOGD(opName_, "batchTaksTailVecCore:  %lu.", tilingData_.get_batchTaksTailVecCore());
    OP_LOGD(opName_, "targetTasksPerLoop:  %lu.", tilingData_.get_targetTasksPerLoop());
    OP_LOGD(opName_, "vocabStartIndex:  %f.", tilingData_.get_vocabStartIndex());
    OP_LOGD(opName_, "vocabEndIndex:  %f.", tilingData_.get_vocabEndIndex());
    OP_LOGD(opName_, "initWorkspaceLength:  %lu.", tilingData_.get_initWorkspaceLength());
    OP_LOGD(opName_, "matmulInputEmptyFlag:  %lu.", tilingData_.get_matmulInputEmptyFlag());
}

static ge::graphStatus TilingForFusedLinearOnlineMaxSum(gert::TilingContext* context) {
    FusedLinearOnlineMaxSumTiling tilingObject(context);
    auto ret = tilingObject.Init();
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "TilingForFusedLinearOnlineMaxSum Init failed.");
        return ge::GRAPH_FAILED;
    }
    ret = tilingObject.RunKernelTiling();
    OP_LOGD(context->GetNodeName(), "TilingForFusedLinearOnlineMaxSum end.");
    return ret;
}

ge::graphStatus TilingPrepareForFusedLinearOnlineMaxSum([[maybe_unused]] gert::TilingParseContext* context) {
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FusedLinearOnlineMaxSum)
    .Tiling(TilingForFusedLinearOnlineMaxSum)
    .TilingParse<FusedLinearOnlineMaxSumCompileInfo>(TilingPrepareForFusedLinearOnlineMaxSum);
}
