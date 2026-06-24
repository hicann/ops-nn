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
 * \file gru_tiling.cpp
 * \brief
 */
#include "gru_tiling.h"

using namespace AscendC;
namespace optiling {

GruTilingOp::GruTilingOp(gert::TilingContext* context) : context_(context) {}

ge::graphStatus GruTilingOp::GetOpInfo() {
    OP_CHECK_IF(context_->GetComputeNodeInputNum() < DEFAULT_PARAMS_INPUT_SIZE,
                    OP_LOGE(context_->GetNodeName(), "input shape error."),
                    return ge::GRAPH_FAILED);

    // get x shape
    auto xTensor = context_->GetInputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, xTensor);
    auto xShape = xTensor->GetStorageShape();

    int32_t xDimNum = xShape.GetDimNum();

    // get bi shape
    auto biasTensor = context_->GetOptionalInputShape(3);
    auto biasHiddenTensor = context_->GetOptionalInputShape(4);
    gruParams_.isBias =  (biasTensor != nullptr && biasHiddenTensor != nullptr) ? 1 : 0;

    auto whTensor = context_->GetInputShape(2);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, whTensor);
    auto whShape = whTensor->GetStorageShape();
    gruParams_.hiddenSize = static_cast<int64_t>(whShape.GetDim(1)) / 3;

    // get seq_length, init_h
    auto inithShape = context_->GetOptionalInputShape(6);
    gruParams_.isInith = (inithShape != nullptr) ? 1 : 0;

    if (xDimNum == 3) {
        //  定长: (T, B, I)
        gruParams_.isSeqLength = 0;
        gruParams_.timeStep = static_cast<int64_t>(xShape.GetDim(0));
        gruParams_.batch = static_cast<int64_t>(xShape.GetDim(1));
        gruParams_.inputSize = static_cast<int64_t>(xShape.GetDim(2));
        gruParams_.totalSteps = gruParams_.timeStep * gruParams_.batch;    // T*B
    } else if (xDimNum == 2) {
        //  不定长：(total_step, I)
        gruParams_.isSeqLength = 1;
        gruParams_.inputSize = static_cast<int64_t>(xShape.GetDim(1));
        //  读取batch_size
        gruParams_.totalSteps = static_cast<int64_t>(xShape.GetDim(0));  // sum(batch_size)
        auto batchSizeShape = context_->GetOptionalInputShape(5)->GetStorageShape();
        gruParams_.timeStep = batchSizeShape.GetDim(0);
        
        auto outputHShape = context_->GetOutputShape(1)->GetStorageShape();   //  [B,H]
        gruParams_.batch = outputHShape.GetDim(0);
        
    } else {
        OP_LOGE(context_->GetNodeName(), "input_x only support 2D or 3D.");
        return ge::GRAPH_FAILED;
    }


    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GruTilingOp::GetAttr() {
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    //  get direction
    const char* direction = attrs->GetAttrPointer<char>(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, direction);
    if (strcmp(direction, "UNIDIRECTIONAL") == 0) {
        gruParams_.direction = 0;
    } else {
        gruParams_.direction = 1;
    }

    //  get is_training
    const bool* isTraining = attrs->GetAttrPointer<bool>(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, isTraining);
    gruParams_.isTraining = *isTraining;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GruTilingOp::GetVectorTiling() {
    auto usedCore = (gruParams_.usedCoreNum + DEFAULT_INDEX_TWO - 1) / DEFAULT_INDEX_TWO;

    //  不定长 
    int64_t realBatch = gruParams_.batch;

    // cut M
    gruParams_.singleCoreM = Ops::Base::CeilDiv(realBatch, usedCore);
    gruParams_.mCnt = Ops::Base::CeilDiv(realBatch, gruParams_.singleCoreM);
    gruParams_.singleCoreMTail = realBatch - (gruParams_.mCnt - 1) * gruParams_.singleCoreM;
    gruParams_.nCnt = realBatch < usedCore ? Ops::Base::CeilDiv(usedCore, gruParams_.mCnt) : 1;
    // cut N
    gruParams_.singleCoreN = Ops::Base::CeilDiv(gruParams_.hiddenSize, gruParams_.nCnt);
    gruParams_.nCnt = Ops::Base::CeilDiv(gruParams_.hiddenSize, gruParams_.singleCoreN);
    gruParams_.singleCoreNTail = gruParams_.hiddenSize - (gruParams_.nCnt - 1) * gruParams_.singleCoreN;

    // 激活函数UB预留
    uint32_t sigmoidMinSize = 0;
    uint32_t tanhMinSize = 0;
    std::vector<int64_t> dims = {MIN_BASE_SHAPE};
    GetSigmoidMaxMinTmpSize(ge::Shape(dims), 2, false, sigmoidMinSize, sigmoidMinSize);
    GetTanhMaxMinTmpSize(ge::Shape(dims), 2, false, tanhMinSize, tanhMinSize);

    uint32_t act_need = std::max(sigmoidMinSize, tanhMinSize);
    uint32_t multiple = (act_need + MIN_BASE_BUFFER - 1) / MIN_BASE_BUFFER;

    auto ubSplit = (gruParams_.dataType == 2 ? 6 : 5) + multiple;
    auto BLOCK_SIZE = 32;
    auto partUb = ((gruParams_.ubSize / ubSplit) + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE / 4;
    gruParams_.baseN = gruParams_.singleCoreN < partUb ? gruParams_.singleCoreN : partUb;
    gruParams_.baseM = partUb / gruParams_.baseN < gruParams_.singleCoreM ? partUb / gruParams_.baseN : gruParams_.singleCoreM;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GruTilingOp::GetMMTiling(GruTilingData& tilingData, matmul_tiling::DataType dataType) {
    int32_t hiddenBlock = 3;
    int64_t aivDouble = 2;
    matmul_tiling::MultiCoreMatmulTiling gruMatmul1;
    gruParams_.usedCoreNum = context_->GetPlatformInfo()->GetCoreNumByType("AiCore") * aivDouble;
    auto ret = gruMatmul1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "mm1 SetAType fail."),
                    return ge::GRAPH_FAILED);
    ret = gruMatmul1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "mm1 SetBType fail."),
                    return ge::GRAPH_FAILED);         
    ret = gruMatmul1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, 
                                matmul_tiling::DataType::DT_FLOAT);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "mm1 SetCType fail."),
                    return ge::GRAPH_FAILED);  

    if (gruParams_.isBias) {
        gruMatmul1.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
        ret = gruMatmul1.SetBias(true);
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "mm1 SetBias fail."),
                    return ge::GRAPH_FAILED);    
    }

    ret = gruMatmul1.SetDim(gruParams_.sysAivCoreNum);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "mm1 SetDim fail."),
                    return ge::GRAPH_FAILED);   
    ret = gruMatmul1.SetOrgShape(gruParams_.totalSteps,
                                    gruParams_.hiddenSize * hiddenBlock, gruParams_.inputSize);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "mm1 SetOrgShape fail."),
                    return ge::GRAPH_FAILED);
    ret = gruMatmul1.SetShape(gruParams_.totalSteps,
                                    gruParams_.hiddenSize * hiddenBlock, gruParams_.inputSize);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "mm1 Set single shape fail."),
                    return ge::GRAPH_FAILED); 
    ret = gruMatmul1.SetBufferSpace(-1, -1, -1);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "mm1 SetBufferShape fail."),
                    return ge::GRAPH_FAILED);  

    ret = gruMatmul1.GetTiling(tilingData.inputMMParam);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "mm1 GetTiling fail."),
                    return ge::GRAPH_FAILED);    

    matmul_tiling::MultiCoreMatmulTiling gruMatmul2;
    ret = gruMatmul2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "mm2 SetAType fail."),
                    return ge::GRAPH_FAILED);
    ret = gruMatmul2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "mm2 SetBType fail."),
                    return ge::GRAPH_FAILED); 
    ret = gruMatmul2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                                matmul_tiling::DataType::DT_FLOAT);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "mm2 SetCType fail."),
                    return ge::GRAPH_FAILED); 

    if (gruParams_.isBias) {
        gruMatmul2.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
        ret = gruMatmul2.SetBias(true);
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "mm2 SetBias fail."),
                    return ge::GRAPH_FAILED);    
    }

    ret = gruMatmul2.SetDim(gruParams_.sysAivCoreNum);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "mm2 SetDim fail."),
                    return ge::GRAPH_FAILED);
    ret = gruMatmul2.SetOrgShape(gruParams_.batch, gruParams_.hiddenSize * hiddenBlock, gruParams_.hiddenSize);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "mm2 SetOrgShape fail."),
                    return ge::GRAPH_FAILED);
    ret = gruMatmul2.SetShape(gruParams_.batch, gruParams_.hiddenSize * hiddenBlock, gruParams_.hiddenSize);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "mm2 Set single shape fail."),
                    return ge::GRAPH_FAILED);
    ret = gruMatmul2.SetBufferSpace(-1, -1, -1);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "mm2 SetBufferSpace fail."),
                    return ge::GRAPH_FAILED);  

    ret = gruMatmul2.GetTiling(tilingData.hiddenMMParam);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "mm2 GetTiling fail."),
                    return ge::GRAPH_FAILED); 

    return ge::GRAPH_SUCCESS;   
}

ge::graphStatus GruTilingOp::CalcTilingKey() {
    //  判断是否需要切分L0c输出，分次搬入UB
    int64_t tilingKey = 0;
    if (gruParams_.dataType == 2) {
        tilingKey = static_cast<int64_t>(GruTilingKey::MM_FP16_SPLIT);
    } else if (gruParams_.dataType == 4) {
        tilingKey = static_cast<int64_t>(GruTilingKey::MM_FP32_SPLIT);
    }
    gruParams_.tilingKey = tilingKey;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GruTilingOp::SetTilingData(GruTilingData& tilingData) {
    tilingData.set_tilingKey(gruParams_.tilingKey);
    tilingData.set_usedCoreNum(gruParams_.usedCoreNum);
    tilingData.set_timeStep(gruParams_.timeStep);
    tilingData.set_batch(gruParams_.batch);
    tilingData.set_inputSize(gruParams_.inputSize);
    tilingData.set_hiddenSize(gruParams_.hiddenSize);
    tilingData.set_isBias(gruParams_.isBias);
    tilingData.set_isInith(gruParams_.isInith);
    tilingData.set_isSeqLength(gruParams_.isSeqLength);
    tilingData.set_direction(gruParams_.direction);
    tilingData.set_isTraining(gruParams_.isTraining);
    tilingData.set_totalSteps(gruParams_.totalSteps);

    //  vector data
    tilingData.set_singleCoreM(gruParams_.singleCoreM);
    tilingData.set_singleCoreMTail(gruParams_.singleCoreMTail);
    tilingData.set_singleCoreN(gruParams_.singleCoreN);
    tilingData.set_singleCoreNTail(gruParams_.singleCoreNTail);
    tilingData.set_mCnt(gruParams_.mCnt);
    tilingData.set_nCnt(gruParams_.nCnt);
    tilingData.set_baseM(gruParams_.baseM);
    tilingData.set_baseN(gruParams_.baseN);

    gert::TilingData* gruRawTilingData = context_->GetRawTilingData();
    OP_LOGE_IF(gruRawTilingData == nullptr, ge::GRAPH_FAILED, context_->GetNodeType(), "GetRawTilingData failed.");
    OP_TILING_CHECK(tilingData.GetDataSize() > gruRawTilingData->GetCapacity(),
                    VECTOR_INNER_ERR_REPORT_TILIING(context_, "actual tiling data size %zu > context tiling data size %zu",
                                                    tilingData.GetDataSize(), gruRawTilingData->GetCapacity()),
                                                return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(gruRawTilingData->GetData(), gruRawTilingData->GetCapacity());
    gruRawTilingData->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

void GruTilingOp::PrintTilingData() {
    OP_LOGD(context_->GetNodeName(), "Start PrintTilingData");
    OP_LOGD(context_->GetNodeName(), "tilingKey is %ld.", gruParams_.tilingKey);
    OP_LOGD(context_->GetNodeName(), "ubSize is %ld.", gruParams_.ubSize);
    OP_LOGD(context_->GetNodeName(), "timeStep is %ld.", gruParams_.timeStep);
    OP_LOGD(context_->GetNodeName(), "batch is %ld.", gruParams_.batch);
    OP_LOGD(context_->GetNodeName(), "usedCoreNum is %ld.", gruParams_.usedCoreNum);
    OP_LOGD(context_->GetNodeName(), "inputSize is %ld.", gruParams_.inputSize);
    OP_LOGD(context_->GetNodeName(), "hiddenSize is %ld.", gruParams_.hiddenSize);
    OP_LOGD(context_->GetNodeName(), "isBias is %ld.", gruParams_.isBias);
    OP_LOGD(context_->GetNodeName(), "isInith is %ld.", gruParams_.isInith);
    OP_LOGD(context_->GetNodeName(), "direction is %ld.", gruParams_.direction);
    OP_LOGD(context_->GetNodeName(), "isTraining is %ld.", gruParams_.isTraining ? 1 : 0);
    OP_LOGD(context_->GetNodeName(), "isSeqLength is %ld.", gruParams_.isSeqLength ? 1 : 0);
    OP_LOGD(context_->GetNodeName(), "singleCoreM is %ld.", gruParams_.singleCoreM);
    OP_LOGD(context_->GetNodeName(), "singleCoreMTail is %ld.", gruParams_.singleCoreMTail);
    OP_LOGD(context_->GetNodeName(), "singleCoreN is %ld.", gruParams_.singleCoreN);
    OP_LOGD(context_->GetNodeName(), "singleCoreNTail is %ld.", gruParams_.singleCoreNTail);
    OP_LOGD(context_->GetNodeName(), "baseN is %ld.", gruParams_.baseN);
    OP_LOGD(context_->GetNodeName(), "baseM is %ld.", gruParams_.baseM);
    OP_LOGD(context_->GetNodeName(), "mCnt is %ld.", gruParams_.mCnt);
    OP_LOGD(context_->GetNodeName(), "nCnt is %ld.", gruParams_.nCnt);
    OP_LOGD(context_->GetNodeName(), "totalSteps is %ld.", gruParams_.totalSteps);
    OP_LOGD(context_->GetNodeName(), "End PrintTilingData.");
}

ge::graphStatus GruTilingOp::Init() {
    // set sync op batchmode
    context_->SetScheduleMode(SCHEDULE_MODE);

    auto dataType = context_->GetInputDesc(0)->GetDataType();
    gruParams_.dataType = dataType == ge::DT_FLOAT ? 4 : 2;

    // get op info
    GetOpInfo();

    // get attr
    GetAttr();

    // get UB L1 size
    auto platformInfo = context_->GetPlatformInfo();

    int32_t doubleNum = 2;
    gruParams_.sysAicCoreNum = context_->GetPlatformInfo()->GetCoreNumByType("AiCore");
    gruParams_.sysAivCoreNum = gruParams_.sysAicCoreNum * doubleNum;
    platformInfo->GetLocalMemSize(fe::LocalMemType::UB, gruParams_.ubSize);
    platformInfo->GetLocalMemSize(fe::LocalMemType::L1, gruParams_.l1Size);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GruTilingOp::DoTiling() {
    GruTilingData tilingData;
    // 910B/C AscendC

    auto dataType = context_->GetInputDesc(0)->GetDataType();

    GetMMTiling(tilingData, static_cast<matmul_tiling::DataType>(dataType));

    GetVectorTiling();

    CalcTilingKey();

    SetTilingData(tilingData);

    auto launchCore = (gruParams_.usedCoreNum + DEFAULT_INDEX_TWO - 1) / DEFAULT_INDEX_TWO;
    context_->SetBlockDim(launchCore);
    context_->SetTilingKey(gruParams_.tilingKey);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    size_t sysWorkspaceByteSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    // 两块MM的GM空间
    int64_t inputMMWorkspaceSize = gruParams_.totalSteps * 3 * gruParams_.hiddenSize * 4;
    int64_t hiddenMMWorkspaceSize = gruParams_.timeStep * gruParams_.batch * 3 * gruParams_.hiddenSize * 4;   //  3是门数量 4是float字节
    int64_t workspaceSize = inputMMWorkspaceSize + hiddenMMWorkspaceSize;

    size_t* workspace_size = context_->GetWorkspaceSizes(1);
    *workspace_size = workspaceSize + sysWorkspaceByteSize;

    PrintTilingData();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForGru(gert::TilingContext* context) {
    GruTilingOp tilingOp(context);
    tilingOp.Init();
    return tilingOp.DoTiling();
}

ge::graphStatus TilingPrepareForGru(gert::TilingParseContext* context) {
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Gru)
    // .TilingInputsDataDependency({5})
    .Tiling(TilingForGru)
    .TilingParse<GruCompileInfo>(TilingPrepareForGru);
}   // namespace optiling