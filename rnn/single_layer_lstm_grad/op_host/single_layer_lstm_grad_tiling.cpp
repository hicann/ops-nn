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
 * \file single_layer_lstm_grad_tiling.cpp
 * \brief
 */

#include "single_layer_lstm_grad_tiling.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"
#include "tiling/tiling_api.h"
#include "error_util.h"
#include "platform/platform_infos_def.h"

namespace optiling {
const std::string OP_NAME = "SingleLayerLstmGrad";
const int32_t GATES_NUM = 4;
const int64_t AIV_DOUBLE = 2;
const int64_t FP32_BYTES = 4;
const int64_t GM2L1_CHECK = 65535;
const int64_t INPUT_NUM = 17;
const int64_t OUTPUT_NUM = 5;
const int64_t OPT_INPUT_B = 2;
const int64_t OPT_INPUT_Y = 3;
const int64_t OPT_INPUT_SEQ = 16;
const int64_t INPUT_DIM_NUM = 3;
const int64_t DEFAULT_UB_RESERVE_SIZE = 1024;
const int64_t BLOCK_BYTES = 32;
const int64_t DEFAULT_ALIGNED_FP16 = 16;
const int64_t DEFAULT_ALIGNED_FP32 = 8;
const int64_t DEFAULT_BUFFER_SPACE = -1;
const int64_t MATRIX_DIM_M = 0;
const int64_t MATRIX_DIM_N = 1;
const int64_t MATRIX_DIM_K = 2;
const int64_t DEFAULT_SPLIT_FACTOR = 4;
const int64_t DEFAULT_REDUCE_N_LIMIT = 128;
const int64_t DEFAULT_COPY_FACTOR_FP32 = 4;
const int64_t DEFAULT_COPY_FACTOR_FP16 = 6;
const int64_t HIDDEN_UB_NUM_WITH_SEQ = 20;
const int64_t HIDDEN_UB_NUM_WITHOUT_SEQ = 19;
const int64_t MM_GATE_KEY_FACTOR = 1000;
const int64_t MM_WEIGHT_KEY_FACTOR = 100;
const int64_t SMALL_FLAG_MULTIPLIER = 1000;
const int64_t SMALL_FLAG_MULTIPLIER_CONCAT = 100;
const int64_t INPUT_X_INDEX = 0;
const int64_t INPUT_WEIGHT_INDEX = 1;
const int64_t INPUT_BIAS_INDEX = 2;
const int64_t INPUT_Y_INDEX = 3;
const int64_t INPUT_INIT_H_INDEX = 4;
const int64_t INPUT_INIT_C_INDEX = 5;
const int64_t INPUT_H_INDEX = 6;
const int64_t INPUT_C_INDEX = 7;
const int64_t INPUT_DY_INDEX = 8;
const int64_t INPUT_DH_INDEX = 9;
const int64_t INPUT_DC_INDEX = 10;
const int64_t INPUT_I_INDEX = 11;
const int64_t INPUT_J_INDEX = 12;
const int64_t INPUT_F_INDEX = 13;
const int64_t INPUT_O_INDEX = 14;
const int64_t INPUT_TANHC_INDEX = 15;
const int64_t INPUT_SEQ_LENGTH_INDEX = 16;
const int64_t OUTPUT_DW_INDEX = 0;
const int64_t OUTPUT_DB_INDEX = 1;
const int64_t OUTPUT_DX_INDEX = 2;
const int64_t OUTPUT_DINIT_H_INDEX = 3;
const int64_t OUTPUT_DINIT_C_INDEX = 4;
const int64_t ATTR_DIRECTION_INDEX = 0;
const int64_t ATTR_GATE_ORDER_INDEX = 1;
const int64_t ZERO_DIM_TENSOR_FLAG = 0;
const int64_t MAX_REDUCE_ROWS_FACTOR = 4;
const int64_t WORKSPACE_FP32_MULTIPLIER = 2;
const int64_t B_INDEX = 1;
const int64_t T_INDEX = 0;
const int64_t SIZE_INDEX = 2;
const int64_t SEQ_UB_NUM = 3;
const std::vector<std::string> SUPPORT_DIRECTION = {"UNIDIRECTIONAL", "REDIRECTIONAL"};

class SingleLayerLstmGradTiling {
public:
    explicit SingleLayerLstmGradTiling(gert::TilingContext* context) : context_(context){};
    ge::graphStatus Init();
    ge::graphStatus GetMMTilingDataSplit();
    ge::graphStatus GetMMTilingData();
    void SetTilingData();

    bool CheckParamsDtype();
    bool CheckParamsShape();
    bool CheckAttr();
    ge::graphStatus CheckAttrTiling();
    ge::graphStatus CheckAttrOps();
    void PrintTilingData();
    void LogCutBatchTilingParam(const std::string& paramName, const CutBatchTilingParam& param);
    CutBatchTilingParam CalculateCutBatchTilingParam(int64_t ubParaNum, int64_t alignedSize, int64_t actualSize,
                              int64_t copyMLinesMax, int64_t batch);
    void VectorBlockCalculate();
    void ReduceBlockCalculate();
    void SplitDxhBlockCalculate();
    void ConcatXhBlockCalculate();
    bool ValidateInputShape (int index, const std::vector<int64_t>& expected_dims);
    bool ValidateOutputShape (int index, const std::vector<int64_t>& expected_dims);
private:
    SingleLayerLstmGradTilingData tilingData_;
    gert::TilingContext* context_ = nullptr;
    SingleLayerLstmGradTilingParams rnnParams_;
    const char* nodeName_ = nullptr;
    int64_t alignedPara_ = DEFAULT_ALIGNED_FP32;
    int64_t inputDSize_ = FP32_BYTES;
    int64_t tilingKey_ = 0;
    CutBatchTilingParam dxhInputParam_;
    CutBatchTilingParam dxhHiddenParam_;
    CutBatchTilingParam xhInputParam_;
    CutBatchTilingParam xhHiddenParam_;
};

ge::graphStatus SingleLayerLstmGradTiling::GetMMTilingDataSplit()
{
    matmul_tiling::MultiCoreMatmulTiling rnnMatmul1;
    rnnParams_.usedCoreNum = context_->GetPlatformInfo()->GetCoreNum() * AIV_DOUBLE;
    auto ret = rnnMatmul1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                                   matmul_tiling::DataType::DT_FLOAT);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "mm1 SetAType fail."),
                    return ge::GRAPH_FAILED);
    ret = rnnMatmul1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, 
                              matmul_tiling::DataType::DT_FLOAT);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "mm1 SetBType fail."),
                    return ge::GRAPH_FAILED);
    ret = rnnMatmul1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                              matmul_tiling::DataType::DT_FLOAT);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "mm1 SetCType fail."),
                    return ge::GRAPH_FAILED);

    ret = rnnMatmul1.SetDim(rnnParams_.sysAivCoreNum);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "mm1 SetDim fail."),
                    return ge::GRAPH_FAILED);
    ret = rnnMatmul1.SetOrgShape(rnnParams_.batch, rnnParams_.inputSize + rnnParams_.hiddenSize,
                                 rnnParams_.hiddenSize * GATES_NUM);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "mm1 SetOrgShape fail."),
                    return ge::GRAPH_FAILED);
    ret = rnnMatmul1.SetShape(rnnParams_.batch, rnnParams_.inputSize + rnnParams_.hiddenSize,
                              rnnParams_.hiddenSize * GATES_NUM);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "mm1 Set single shape fail."),
                    return ge::GRAPH_FAILED);
    ret = rnnMatmul1.SetBufferSpace(DEFAULT_BUFFER_SPACE, DEFAULT_BUFFER_SPACE, DEFAULT_BUFFER_SPACE);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "mm1 SetBufferSpace fail."),
                    return ge::GRAPH_FAILED);

    ret = rnnMatmul1.GetTiling(tilingData_.dgateMMParam);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "mm1 GetTiling fail."),
                    return ge::GRAPH_FAILED);

    matmul_tiling::MultiCoreMatmulTiling rnnMatmul2;
    ret = rnnMatmul2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                              matmul_tiling::DataType::DT_FLOAT, true);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "mm2 SetAType fail."),
                    return ge::GRAPH_FAILED);
    ret = rnnMatmul2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                              matmul_tiling::DataType::DT_FLOAT);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "mm2 SetBType fail."),
                    return ge::GRAPH_FAILED);
    ret = rnnMatmul2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                                matmul_tiling::DataType::DT_FLOAT);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "mm2 SetCType fail."),
                    return ge::GRAPH_FAILED);
    ret = rnnMatmul2.SetDim(rnnParams_.sysAivCoreNum);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "mm2 SetDim fail."),
                    return ge::GRAPH_FAILED);
    ret = rnnMatmul2.SetOrgShape(rnnParams_.hiddenSize * GATES_NUM, rnnParams_.hiddenSize + rnnParams_.inputSize,
                                 rnnParams_.batch * rnnParams_.timeStep);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "mm2 SetOrgShape fail."),
                    return ge::GRAPH_FAILED);
    ret = rnnMatmul2.SetShape(rnnParams_.hiddenSize * GATES_NUM, rnnParams_.hiddenSize + rnnParams_.inputSize,
                              rnnParams_.batch * rnnParams_.timeStep);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "mm2 Set single shape fail."),
                    return ge::GRAPH_FAILED);
    ret = rnnMatmul2.SetBufferSpace(DEFAULT_BUFFER_SPACE, DEFAULT_BUFFER_SPACE, DEFAULT_BUFFER_SPACE);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "mm2 SetBufferSpace fail."),
                    return ge::GRAPH_FAILED);

    ret = rnnMatmul2.GetTiling(tilingData_.dwMMParam);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "mm2 GetTiling fail."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool SingleLayerLstmGradTiling::CheckParamsDtype()
{
    // dtype support list
    std::vector<ge::DataType> supportDtype = {ge::DT_FLOAT, ge::DT_FLOAT16};
    ge::DataType baseDtype = context_->GetInputDesc(INPUT_X_INDEX)->GetDataType();

    // input check
    for (int64_t inputIdx = 0; inputIdx < INPUT_NUM; inputIdx++) {
        bool optionalInputNull = (inputIdx == OPT_INPUT_B && !rnnParams_.isBias) || inputIdx == OPT_INPUT_Y ||
                                 (inputIdx == OPT_INPUT_SEQ && !rnnParams_.isSeqLength);
        if (optionalInputNull) {
            continue;
        }
        auto dtype = context_->GetInputDesc(inputIdx)->GetDataType();
        if (std::find(supportDtype.begin(), supportDtype.end(), dtype) == supportDtype.end()) {
            VECTOR_INNER_ERR_REPORT_TILIING(nodeName_,
                "Input dtype not supported at index %ld.", inputIdx);
            return false;
        }
        if (dtype != baseDtype) {
            VECTOR_INNER_ERR_REPORT_TILIING(nodeName_,
                "Input dtype inconsistent at index %ld.", inputIdx);
            return false;
        }
    }
    
    // output check
    for (int64_t outputIdx = 0; outputIdx < OUTPUT_NUM; outputIdx++) {
        bool optionalOutputNull = (outputIdx == OUTPUT_DB_INDEX && !rnnParams_.isBias);
        if (optionalOutputNull) {
            continue;
        }
        auto dtype = context_->GetOutputDesc(outputIdx)->GetDataType();
        if (std::find(supportDtype.begin(), supportDtype.end(), dtype) == supportDtype.end()) {
            VECTOR_INNER_ERR_REPORT_TILIING(nodeName_,
                "Output dtype not supported at index %ld.", outputIdx);
            return false;
        }
        if (dtype != baseDtype) {
            VECTOR_INNER_ERR_REPORT_TILIING(nodeName_,
                "Output dtype inconsistent at index %ld \n", outputIdx);
            return false;
        }
    }
    
    return true;
}

bool SingleLayerLstmGradTiling::ValidateInputShape(int index, const std::vector<int64_t>& expected_dims)
{
  auto input = context_->GetInputShape(index);
  OP_CHECK_IF(input == nullptr,
          OP_LOGE(nodeName_, "input %d is nullptr", index),
          return false);
  auto shape = input->GetStorageShape();
  if (shape.GetDimNum() != expected_dims.size()) {
      OP_LOGE(nodeName_, "Input %d has wrong dimension count", index);
      return false;
  }

  for (size_t i = 0; i < expected_dims.size(); ++i) {
      if (expected_dims[i] != shape.GetDim(i)) {
        OP_LOGE(nodeName_, "Input %d dim %zu mismatch", index, i);
        return false;
      }
  }
  return true;
};

bool SingleLayerLstmGradTiling::ValidateOutputShape (int index, const std::vector<int64_t>& expected_dims)
{
  auto output = context_->GetOutputShape(index);
  OP_CHECK_IF(output == nullptr,
              OP_LOGE(nodeName_, "output %d is nullptr", index),
              return false);
  auto shape = output->GetStorageShape();
  if (shape.GetDimNum() != expected_dims.size()) {
      OP_LOGE(nodeName_, "Output %d has wrong dimension count", index);
      return false;
  }

  for (size_t i = 0; i < expected_dims.size(); ++i) {
      if (expected_dims[i] != shape.GetDim(i)) {
          OP_LOGE(nodeName_, "Output %d dim %zu mismatch", index, i);
          return false;
      }
  }
  return true;
};

bool SingleLayerLstmGradTiling::CheckParamsShape()
{
    // get input shape
    auto xInput = context_->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_IF(xInput == nullptr,
              OP_LOGE(nodeName_, "input 0 is nullptr"),
              return false);
    auto xShape = xInput->GetStorageShape();
    if (xShape.GetDimNum() != INPUT_DIM_NUM) {
        OP_LOGE(nodeName_, "Input x must be 3D tensor");
        return false;
    }
    // get wight shape
    auto initHInput = context_->GetInputShape(INPUT_INIT_H_INDEX);
    OP_CHECK_IF(initHInput == nullptr,
              OP_LOGE(nodeName_, "input 4 is nullptr"),
              return false);
    auto initHShape = initHInput->GetStorageShape();
    if (initHShape.GetDimNum() != INPUT_DIM_NUM) {
        OP_LOGE(nodeName_, "Input initH must be 3D tensor");
        return false;
    }

    rnnParams_.timeStep = xShape.GetDim(T_INDEX);
    rnnParams_.batch = xShape.GetDim(B_INDEX);
    rnnParams_.inputSize = xShape.GetDim(SIZE_INDEX);
    rnnParams_.hiddenSize = initHShape.GetDim(SIZE_INDEX);
    OP_TILING_CHECK(
        rnnParams_.inputSize <= 0 || rnnParams_.hiddenSize <= 0 || rnnParams_.timeStep <= 0 || rnnParams_.batch <= 0,
        VECTOR_INNER_ERR_REPORT_TILIING(nodeName_,
        "SingleLayerLstmGrad tensor shape not support 0 or negtive, please check."), return false);

    // get optional input
    auto biasShape = context_->GetOptionalInputShape(INPUT_BIAS_INDEX);
    auto biasDesc = context_->GetOptionalInputDesc(INPUT_BIAS_INDEX);
    auto seqShape = context_->GetOptionalInputShape(INPUT_SEQ_LENGTH_INDEX);
    auto seqDesc = context_->GetOptionalInputDesc(INPUT_SEQ_LENGTH_INDEX);
    auto yShape = context_->GetOptionalInputShape(INPUT_Y_INDEX);
    auto yDesc = context_->GetOptionalInputDesc(INPUT_Y_INDEX);
    rnnParams_.isBias =
        (biasDesc != nullptr && biasShape != nullptr &&
        biasShape->GetStorageShape().GetDimNum() != ZERO_DIM_TENSOR_FLAG) ? 1 : 0;
    rnnParams_.isSeqLength = (seqDesc != nullptr && seqShape != nullptr &&
        seqShape->GetStorageShape().GetDimNum() != ZERO_DIM_TENSOR_FLAG) ? 1 : 0;
    bool isY = (yDesc != nullptr && yShape != nullptr &&
        yShape->GetStorageShape().GetDimNum() != ZERO_DIM_TENSOR_FLAG) ? true : false;

    std::vector<int64_t> biasDim = {GATES_NUM * rnnParams_.hiddenSize};
    std::vector<int64_t> initDim = {1, rnnParams_.batch, rnnParams_.hiddenSize};
    std::vector<int64_t> hiddenDim = {rnnParams_.timeStep, rnnParams_.batch, rnnParams_.hiddenSize};
    std::vector<int64_t> weightDim = {GATES_NUM * rnnParams_.hiddenSize, rnnParams_.inputSize + rnnParams_.hiddenSize};
    std::vector<int64_t> inputDim = {rnnParams_.timeStep, rnnParams_.batch, rnnParams_.inputSize};
    std::vector<int64_t> seqDim = {rnnParams_.timeStep, rnnParams_.batch, rnnParams_.hiddenSize};
    bool ret =
      ValidateInputShape(INPUT_WEIGHT_INDEX, weightDim) &&
      ValidateInputShape(INPUT_INIT_H_INDEX, initDim) &&
      ValidateInputShape(INPUT_INIT_C_INDEX, initDim) &&
      ValidateInputShape(INPUT_H_INDEX, hiddenDim) &&
      ValidateInputShape(INPUT_C_INDEX, hiddenDim) &&
      ValidateInputShape(INPUT_DY_INDEX, hiddenDim) &&
      ValidateInputShape(INPUT_DH_INDEX, initDim) &&
      ValidateInputShape(INPUT_DC_INDEX, initDim) &&
      ValidateInputShape(INPUT_I_INDEX, hiddenDim) &&
      ValidateInputShape(INPUT_J_INDEX, hiddenDim) &&
      ValidateInputShape(INPUT_F_INDEX, hiddenDim) &&
      ValidateInputShape(INPUT_O_INDEX, hiddenDim) &&
      ValidateInputShape(INPUT_TANHC_INDEX, hiddenDim) &&
      ValidateOutputShape(OUTPUT_DW_INDEX, weightDim) &&
      ValidateOutputShape(OUTPUT_DX_INDEX, inputDim) &&
      ValidateOutputShape(OUTPUT_DINIT_H_INDEX, initDim) &&
      ValidateOutputShape(OUTPUT_DINIT_C_INDEX, initDim);
    ret = rnnParams_.isBias ? ret && ValidateInputShape(INPUT_BIAS_INDEX, biasDim) &&
        ValidateOutputShape(OUTPUT_DB_INDEX, biasDim) : ret;
    ret = rnnParams_.isSeqLength ? ret && ValidateInputShape(INPUT_SEQ_LENGTH_INDEX, seqDim) : ret;
    ret = isY ? ret && ValidateInputShape(INPUT_Y_INDEX, hiddenDim) : ret;
    return ret;
}

bool SingleLayerLstmGradTiling::CheckAttr()
{
  return CheckAttrOps() == ge::GRAPH_SUCCESS && CheckAttrTiling() == ge::GRAPH_SUCCESS;
}

ge::graphStatus SingleLayerLstmGradTiling::CheckAttrTiling()
{
    // get attr
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const char* gateOrder = attrs->GetAttrPointer<char>(ATTR_GATE_ORDER_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gateOrder);
    if (strcmp(gateOrder, "ifjo") != 0 && strcmp(gateOrder, "ijfo") != 0) {
        OP_LOGE(nodeName_,
            "SingleLayerLstmGrad attr gate_order [%s] is not support, please check.", gateOrder);
        return ge::GRAPH_FAILED;
    }
    rnnParams_.gateOrder = strcmp(gateOrder, "ijfo") == 0 ? static_cast<int64_t>(GateOrder::IJFO) :
                                                            static_cast<int64_t>(GateOrder::IFJO);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SingleLayerLstmGradTiling::CheckAttrOps()
{
    // get attr
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const char* direction = attrs->GetAttrPointer<char>(ATTR_DIRECTION_INDEX);
    OP_CHECK_IF(std::find(SUPPORT_DIRECTION.begin(), SUPPORT_DIRECTION.end(), direction) == SUPPORT_DIRECTION.end(),
                    OP_LOGE(nodeName_,
                    "SingleLayerLstmGrad attr direction is not support, please check."), return ge::GRAPH_FAILED);
    rnnParams_.direction = strcmp(direction, "UNIDIRECTIONAL") == 0 ? 0 : 1;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SingleLayerLstmGradTiling::GetMMTilingData()
{
  auto dataType = context_->GetInputDesc(INPUT_X_INDEX)->GetDataType();
  inputDSize_ = dataType == ge::DT_FLOAT ? FP32_BYTES : FP32_BYTES / AIV_DOUBLE;
  alignedPara_ = dataType == ge::DT_FLOAT ? DEFAULT_ALIGNED_FP32 : DEFAULT_ALIGNED_FP16;
  auto ret = GetMMTilingDataSplit();

  return ret;
}

void SingleLayerLstmGradTiling::VectorBlockCalculate()
{
    int64_t hiddenUbNum = rnnParams_.isSeqLength ? HIDDEN_UB_NUM_WITH_SEQ : HIDDEN_UB_NUM_WITHOUT_SEQ;
    int64_t sizeLimit = (rnnParams_.ubSize - DEFAULT_UB_RESERVE_SIZE) / hiddenUbNum / FP32_BYTES / alignedPara_ *
        alignedPara_;
    //cut M
    rnnParams_.singleCoreM = Ops::Base::CeilDiv(rnnParams_.batch, rnnParams_.sysAivCoreNum);
    rnnParams_.mCnt = Ops::Base::CeilDiv(rnnParams_.batch, rnnParams_.singleCoreM);
    rnnParams_.singleCoreMTail = rnnParams_.batch - (rnnParams_.mCnt - 1)  * rnnParams_.singleCoreM;
    rnnParams_.nCnt = rnnParams_.batch < rnnParams_.sysAivCoreNum ?
        Ops::Base::CeilDiv(rnnParams_.sysAivCoreNum, rnnParams_.mCnt) : 1;
    //cut N
    rnnParams_.singleCoreN = Ops::Base::CeilDiv(rnnParams_.hiddenSize, rnnParams_.nCnt);
    rnnParams_.singleCoreN = rnnParams_.singleCoreN < (BLOCK_BYTES / inputDSize_) ? BLOCK_BYTES / inputDSize_ :
                             rnnParams_.singleCoreN;
    rnnParams_.nCnt = Ops::Base::CeilDiv(rnnParams_.hiddenSize, rnnParams_.singleCoreN);
    rnnParams_.singleCoreNTail = rnnParams_.hiddenSize - (rnnParams_.nCnt - 1) * rnnParams_.singleCoreN;
    rnnParams_.baseN = rnnParams_.singleCoreN < sizeLimit ? Ops::Base::CeilDiv(rnnParams_.singleCoreN, alignedPara_) *
                       alignedPara_ : sizeLimit;
    rnnParams_.baseM = sizeLimit / rnnParams_.baseN;
}

CutBatchTilingParam SingleLayerLstmGradTiling::CalculateCutBatchTilingParam(int64_t ubParaNum, int64_t alignedSize,
    int64_t actualSize, int64_t copyMLinesMax, int64_t batch)
{
    CutBatchTilingParam resParams;
    // calculate copy lines at M
    resParams.copyMLines = ubParaNum > alignedSize ? ubParaNum / alignedSize : 1;
    resParams.copyMLines = resParams.copyMLines < copyMLinesMax ? resParams.copyMLines : copyMLinesMax;
    resParams.taskNum = Ops::Base::CeilDiv(batch, resParams.copyMLines);
    resParams.copyMLinesTail = batch - (resParams.taskNum - 1) * resParams.copyMLines;
    
    // calculate loop params at N
    resParams.copyNLength = Ops::Base::CeilAlign(Ops::Base::CeilDiv(ubParaNum,  resParams.copyMLines), alignedSize);
    resParams.nLoop = Ops::Base::CeilDiv(actualSize, resParams.copyNLength);
    resParams.copyNLengthTail = actualSize - (resParams.nLoop - 1) * resParams.copyNLength ;
    
    // calculate block params
    resParams.splitTaskPerCore = resParams.taskNum / rnnParams_.sysAivCoreNum;
    resParams.splitPreCore = resParams.taskNum % rnnParams_.sysAivCoreNum;
    
    return resParams;
}

void SingleLayerLstmGradTiling::SplitDxhBlockCalculate()
{
    int64_t factor = DEFAULT_SPLIT_FACTOR;
    int64_t ubParaNum = Ops::Base::CeilAlign((rnnParams_.ubSize - DEFAULT_UB_RESERVE_SIZE) / factor, alignedPara_);
    rnnParams_.inputSizeAligned = Ops::Base::CeilAlign(rnnParams_.inputSize, alignedPara_);
    rnnParams_.hiddenSizeAligned = Ops::Base::CeilAlign(rnnParams_.hiddenSize, alignedPara_);
    rnnParams_.oneLineAligned = rnnParams_.inputSizeAligned + rnnParams_.hiddenSizeAligned;
    int64_t dxhSmallFlag = ubParaNum <= rnnParams_.oneLineAligned ? 1 : 0;
    tilingKey_ += dxhSmallFlag * SMALL_FLAG_MULTIPLIER;
    int64_t copyMLinesMax = Ops::Base::CeilDiv(rnnParams_.batch, rnnParams_.sysAivCoreNum);
    if (ubParaNum > rnnParams_.oneLineAligned && rnnParams_.isSeqLength == 0) {
        dxhInputParam_ = dxhHiddenParam_ = CalculateCutBatchTilingParam(ubParaNum, rnnParams_.oneLineAligned,
            rnnParams_.inputSize + rnnParams_.hiddenSize, copyMLinesMax, rnnParams_.batch);
    } else if (rnnParams_.isSeqLength == 0) {
        factor = inputDSize_ == FP32_BYTES ? DEFAULT_COPY_FACTOR_FP32 : DEFAULT_COPY_FACTOR_FP16;
        ubParaNum = Ops::Base::CeilAlign((rnnParams_.ubSize - DEFAULT_UB_RESERVE_SIZE) / factor, alignedPara_);
        dxhInputParam_ = CalculateCutBatchTilingParam(ubParaNum, rnnParams_.inputSizeAligned,
            rnnParams_.inputSize, copyMLinesMax, rnnParams_.batch);
        dxhHiddenParam_ = CalculateCutBatchTilingParam(ubParaNum, rnnParams_.hiddenSizeAligned,
            rnnParams_.hiddenSize, copyMLinesMax, rnnParams_.batch);
    } else if (rnnParams_.isSeqLength == 1) {
        factor = inputDSize_ == FP32_BYTES ? DEFAULT_COPY_FACTOR_FP32 : DEFAULT_COPY_FACTOR_FP16;
        ubParaNum = Ops::Base::CeilAlign((rnnParams_.ubSize - DEFAULT_UB_RESERVE_SIZE) / factor, alignedPara_);
        dxhInputParam_ = CalculateCutBatchTilingParam(ubParaNum, rnnParams_.inputSizeAligned,
            rnnParams_.inputSize, copyMLinesMax, rnnParams_.batch);
        int64_t factorHidden = inputDSize_ == FP32_BYTES ?
                                              DEFAULT_COPY_FACTOR_FP32 * SEQ_UB_NUM :
                                              DEFAULT_COPY_FACTOR_FP16 * SEQ_UB_NUM;
        int64_t ubParaNumHidden = Ops::Base::CeilAlign(
            (rnnParams_.ubSize - DEFAULT_UB_RESERVE_SIZE) / factorHidden, alignedPara_);
        dxhHiddenParam_ = CalculateCutBatchTilingParam(ubParaNumHidden, rnnParams_.hiddenSizeAligned,
            rnnParams_.hiddenSize, copyMLinesMax, rnnParams_.batch);
    }
}

void SingleLayerLstmGradTiling::ConcatXhBlockCalculate()
{
    int64_t factor = inputDSize_ == FP32_BYTES ? DEFAULT_COPY_FACTOR_FP32 : DEFAULT_COPY_FACTOR_FP16;
    int64_t ubParaNum = Ops::Base::CeilDiv((rnnParams_.ubSize - DEFAULT_UB_RESERVE_SIZE) / factor, alignedPara_) *
        alignedPara_;
    int64_t dxhSmallFlag = ubParaNum <= rnnParams_.oneLineAligned ? 1 : 0;
    tilingKey_ += dxhSmallFlag * SMALL_FLAG_MULTIPLIER_CONCAT;
    int64_t copyMLinesMax = Ops::Base::CeilDiv(rnnParams_.batch, rnnParams_.sysAivCoreNum);
    if (ubParaNum > rnnParams_.oneLineAligned) {
        xhInputParam_ = xhHiddenParam_ = CalculateCutBatchTilingParam(ubParaNum, rnnParams_.oneLineAligned,
            rnnParams_.inputSize + rnnParams_.hiddenSize, copyMLinesMax, rnnParams_.batch);
    } else {
        xhHiddenParam_ = CalculateCutBatchTilingParam(ubParaNum, rnnParams_.hiddenSizeAligned,
            rnnParams_.hiddenSize, copyMLinesMax, rnnParams_.batch);
        int64_t copyMLinesMaxInput= Ops::Base::CeilDiv(rnnParams_.batch * rnnParams_.timeStep, rnnParams_.sysAivCoreNum);
        xhInputParam_ = CalculateCutBatchTilingParam(ubParaNum, rnnParams_.inputSizeAligned,
            rnnParams_.inputSize, copyMLinesMaxInput, rnnParams_.batch * rnnParams_.timeStep);
    }
}

void SingleLayerLstmGradTiling::ReduceBlockCalculate()
{
    rnnParams_.baseReduceN = Ops::Base::CeilDiv(
        Ops::Base::CeilDiv(rnnParams_.hiddenSize * GATES_NUM, rnnParams_.sysAivCoreNum), alignedPara_) * alignedPara_;
    rnnParams_.baseReduceN =  rnnParams_.baseReduceN > DEFAULT_REDUCE_N_LIMIT ? DEFAULT_REDUCE_N_LIMIT :
        rnnParams_.baseReduceN;
    rnnParams_.maxReduceNumOnce = (rnnParams_.ubSize - DEFAULT_UB_RESERVE_SIZE) / FP32_BYTES / rnnParams_.baseReduceN;
    rnnParams_.reduceBlockSize = rnnParams_.timeStep * rnnParams_.batch;
    int64_t baseReduceNNums = Ops::Base::CeilDiv(rnnParams_.hiddenSize * GATES_NUM, rnnParams_.baseReduceN);
    rnnParams_.nReduceCnt = rnnParams_.sysAivCoreNum < baseReduceNNums ? rnnParams_.sysAivCoreNum : baseReduceNNums;
    int64_t reduceNNumsPerCore = Ops::Base::CeilDiv(baseReduceNNums, rnnParams_.nReduceCnt);
    rnnParams_.nReduceCnt = Ops::Base::CeilDiv(baseReduceNNums, reduceNNumsPerCore);
    rnnParams_.singleCoreReduceN = reduceNNumsPerCore * rnnParams_.baseReduceN;
    rnnParams_.singleCoreReduceNTail = rnnParams_.hiddenSize * GATES_NUM - (rnnParams_.nReduceCnt - 1) *
        rnnParams_.singleCoreReduceN;
    /*Since the data volume of the reduce operation will vary, the number of rows (for each core to move multiple rows
      at a time) has to be handled within the kernel.*/
}

ge::graphStatus SingleLayerLstmGradTiling::Init()
{
    nodeName_ = context_->GetNodeName();
    OP_LOGD(nodeName_, "SingleLayerLstmGrad tiling starts running");
    context_->SetScheduleMode(1);
    auto platformInfo = context_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t aicCoreNum = ascendcPlatform.GetCoreNumAic();
    OP_TILING_CHECK((aicCoreNum <= 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "Failed to get core num."),
                    return ge::GRAPH_FAILED);
    uint32_t aivCoreNum = aicCoreNum * AIV_DOUBLE;
    OP_TILING_CHECK((aivCoreNum <= 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "Failed to get core num."),
                    return ge::GRAPH_FAILED);
    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    OP_TILING_CHECK(!CheckParamsShape(),
                    VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "check shape fail."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckParamsDtype(),
                    VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "check dtype fail."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckAttr(),
                    VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "check attr fail."),
                    return ge::GRAPH_FAILED);
    
    // get UB L1 size
    rnnParams_.sysAicCoreNum = static_cast<int64_t>(aicCoreNum);
    rnnParams_.sysAivCoreNum = static_cast<int64_t>(aivCoreNum);
    rnnParams_.ubSize = ubSizePlatForm;
    // get matmul tiling data
    OP_TILING_CHECK(
        GetMMTilingData() != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "get matmul tiling data fail."),
        return ge::GRAPH_FAILED);

    VectorBlockCalculate();
    ReduceBlockCalculate();
    SplitDxhBlockCalculate();
    ConcatXhBlockCalculate();

    SetTilingData();
    context_->SetBlockDim(rnnParams_.sysAicCoreNum);
    int64_t mmGateKeyFlag = rnnParams_.hiddenSize * GATES_NUM > GM2L1_CHECK ? 1 : 0;
    int64_t mmWeightKeyFlag = rnnParams_.timeStep * rnnParams_.batch > GM2L1_CHECK ? MM_WEIGHT_KEY_FACTOR : 0;
    tilingKey_ = tilingKey_ + mmGateKeyFlag + mmWeightKeyFlag;
    context_->SetTilingKey(tilingKey_);
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    // wFp32 + dwFp32 + dhPrevFp32 + dcPrevFp32
    int64_t workspaceFp32 = GATES_NUM * rnnParams_.hiddenSize * (rnnParams_.hiddenSize + rnnParams_.inputSize) *
        FP32_BYTES * WORKSPACE_FP32_MULTIPLIER + rnnParams_.batch * rnnParams_.hiddenSize * FP32_BYTES *
        WORKSPACE_FP32_MULTIPLIER;

    // dgate + xh + dxh + sys
    currentWorkspace[0] = rnnParams_.timeStep * rnnParams_.batch * GATES_NUM * rnnParams_.hiddenSize * FP32_BYTES +
        rnnParams_.timeStep * rnnParams_.batch * (rnnParams_.hiddenSize + rnnParams_.inputSize) * FP32_BYTES +
        rnnParams_.batch * (rnnParams_.hiddenSize + rnnParams_.inputSize) * FP32_BYTES + sysWorkspaceSize;
    currentWorkspace[0] = inputDSize_ == FP32_BYTES ? currentWorkspace[0] : currentWorkspace[0] + workspaceFp32;
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

void SingleLayerLstmGradTiling::SetTilingData()
{   
    tilingData_.set_ubSize(rnnParams_.ubSize);
    tilingData_.set_timeStep(rnnParams_.timeStep);
    tilingData_.set_batch(rnnParams_.batch);
    tilingData_.set_inputSize(rnnParams_.inputSize);
    tilingData_.set_hiddenSize(rnnParams_.hiddenSize);
    tilingData_.set_isBias(rnnParams_.isBias);
    tilingData_.set_isSeqLength(rnnParams_.isSeqLength);
    // vector data
    tilingData_.set_singleCoreM(rnnParams_.singleCoreM);
    tilingData_.set_singleCoreMTail(rnnParams_.singleCoreMTail);
    tilingData_.set_singleCoreN(rnnParams_.singleCoreN);
    tilingData_.set_singleCoreNTail(rnnParams_.singleCoreNTail);
    tilingData_.set_mCnt(rnnParams_.mCnt);
    tilingData_.set_nCnt(rnnParams_.nCnt);
    tilingData_.set_baseM(rnnParams_.baseM);
    tilingData_.set_baseN(rnnParams_.baseN);
    // reduce data
    tilingData_.set_singleCoreReduceN(rnnParams_.singleCoreReduceN);
    tilingData_.set_singleCoreReduceNTail(rnnParams_.singleCoreReduceNTail);
    tilingData_.set_baseReduceN(rnnParams_.baseReduceN);
    tilingData_.set_nReduceCnt(rnnParams_.nReduceCnt);
    tilingData_.set_maxReduceNumOnce(rnnParams_.maxReduceNumOnce);
    tilingData_.set_reduceBlockSize(rnnParams_.reduceBlockSize);
    tilingData_.set_gateOrder(rnnParams_.gateOrder);
    tilingData_.set_direction(rnnParams_.direction);
    tilingData_.set_cellClip(rnnParams_.cellClip);
    tilingData_.set_forgetBias(rnnParams_.forgetBias);

    tilingData_.set_inputSizeAligned(rnnParams_.inputSizeAligned);
    tilingData_.set_hiddenSizeAligned(rnnParams_.hiddenSizeAligned);
    tilingData_.set_oneLineAligned(rnnParams_.oneLineAligned);
    tilingData_.dxhInputTiling.set_taskNum(dxhInputParam_.taskNum);
    tilingData_.dxhInputTiling.set_copyMLines(dxhInputParam_.copyMLines);
    tilingData_.dxhInputTiling.set_copyMLinesTail(dxhInputParam_.copyMLinesTail);
    tilingData_.dxhInputTiling.set_nLoop(dxhInputParam_.nLoop);
    tilingData_.dxhInputTiling.set_copyNLength(dxhInputParam_.copyNLength);
    tilingData_.dxhInputTiling.set_copyNLengthTail(dxhInputParam_.copyNLengthTail);
    tilingData_.dxhInputTiling.set_splitTaskPerCore(dxhInputParam_.splitTaskPerCore);
    tilingData_.dxhInputTiling.set_splitPreCore(dxhInputParam_.splitPreCore);

    tilingData_.dxhHiddenTiling.set_taskNum(dxhHiddenParam_.taskNum);
    tilingData_.dxhHiddenTiling.set_copyMLines(dxhHiddenParam_.copyMLines);
    tilingData_.dxhHiddenTiling.set_copyMLinesTail(dxhHiddenParam_.copyMLinesTail);
    tilingData_.dxhHiddenTiling.set_nLoop(dxhHiddenParam_.nLoop);
    tilingData_.dxhHiddenTiling.set_copyNLength(dxhHiddenParam_.copyNLength);
    tilingData_.dxhHiddenTiling.set_copyNLengthTail(dxhHiddenParam_.copyNLengthTail);
    tilingData_.dxhHiddenTiling.set_splitTaskPerCore(dxhHiddenParam_.splitTaskPerCore);
    tilingData_.dxhHiddenTiling.set_splitPreCore(dxhHiddenParam_.splitPreCore);

    tilingData_.xhInputTiling.set_taskNum(xhInputParam_.taskNum);
    tilingData_.xhInputTiling.set_copyMLines(xhInputParam_.copyMLines);
    tilingData_.xhInputTiling.set_copyMLinesTail(xhInputParam_.copyMLinesTail);
    tilingData_.xhInputTiling.set_nLoop(xhInputParam_.nLoop);
    tilingData_.xhInputTiling.set_copyNLength(xhInputParam_.copyNLength);
    tilingData_.xhInputTiling.set_copyNLengthTail(xhInputParam_.copyNLengthTail);
    tilingData_.xhInputTiling.set_splitTaskPerCore(xhInputParam_.splitTaskPerCore);
    tilingData_.xhInputTiling.set_splitPreCore(xhInputParam_.splitPreCore);

    tilingData_.xhHiddenTiling.set_taskNum(xhHiddenParam_.taskNum);
    tilingData_.xhHiddenTiling.set_copyMLines(xhHiddenParam_.copyMLines);
    tilingData_.xhHiddenTiling.set_copyMLinesTail(xhHiddenParam_.copyMLinesTail);
    tilingData_.xhHiddenTiling.set_nLoop(xhHiddenParam_.nLoop);
    tilingData_.xhHiddenTiling.set_copyNLength(xhHiddenParam_.copyNLength);
    tilingData_.xhHiddenTiling.set_copyNLengthTail(xhHiddenParam_.copyNLengthTail);
    tilingData_.xhHiddenTiling.set_splitTaskPerCore(xhHiddenParam_.splitTaskPerCore);
    tilingData_.xhHiddenTiling.set_splitPreCore(xhHiddenParam_.splitPreCore);

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
}

void SingleLayerLstmGradTiling::LogCutBatchTilingParam(const std::string& paramName, const CutBatchTilingParam& param)
{
    OP_LOGD(nodeName_, "%s.taskNum %ld.", paramName.c_str(), param.taskNum);
    OP_LOGD(nodeName_, "%s.copyMLines %ld.", paramName.c_str(), param.copyMLines);
    OP_LOGD(nodeName_, "%s.copyMLinesTail %ld.", paramName.c_str(), param.copyMLinesTail);
    OP_LOGD(nodeName_, "%s.nLoop %ld.", paramName.c_str(), param.nLoop);
    OP_LOGD(nodeName_, "%s.copyNLength %ld.", paramName.c_str(), param.copyNLength);
    OP_LOGD(nodeName_, "%s.copyNLengthTail %ld.", paramName.c_str(), param.copyNLengthTail);
    OP_LOGD(nodeName_, "%s.splitTaskPerCore %ld.", paramName.c_str(), param.splitTaskPerCore);
    OP_LOGD(nodeName_, "%s.splitPreCore %ld.", paramName.c_str(), param.splitPreCore);
}

void SingleLayerLstmGradTiling::PrintTilingData()
{
    OP_LOGD(nodeName_, "Start printing");
    OP_LOGD(nodeName_, "ubSize is %ld.", rnnParams_.ubSize);
    OP_LOGD(nodeName_, "timeStep is %ld.", rnnParams_.timeStep);
    OP_LOGD(nodeName_, "batch is %ld.", rnnParams_.batch);
    OP_LOGD(nodeName_, "inputSize is %ld.", rnnParams_.inputSize);
    OP_LOGD(nodeName_, "hiddenSize is %ld.", rnnParams_.hiddenSize);
    OP_LOGD(nodeName_, "isBias is %ld.", rnnParams_.isBias);
    OP_LOGD(nodeName_, "isSeqLength is %ld.", rnnParams_.isSeqLength);
    OP_LOGD(nodeName_, "singleCoreM is %ld.", rnnParams_.singleCoreM);
    OP_LOGD(nodeName_, "singleCoreMTail is %ld.", rnnParams_.singleCoreMTail);
    OP_LOGD(nodeName_, "singleCoreN is %ld.", rnnParams_.singleCoreN);
    OP_LOGD(nodeName_, "singleCoreNTail is %ld.", rnnParams_.singleCoreNTail);
    OP_LOGD(nodeName_, "baseN is %ld.", rnnParams_.baseN);
    OP_LOGD(nodeName_, "baseM is %ld.", rnnParams_.baseM);
    OP_LOGD(nodeName_, "mCnt is %ld.", rnnParams_.mCnt);
    OP_LOGD(nodeName_, "nCnt is %ld.", rnnParams_.nCnt);
    OP_LOGD(nodeName_, "singleCoreReduceN is %ld.", rnnParams_.singleCoreReduceN);
    OP_LOGD(nodeName_, "singleCoreReduceNTail is %ld.", rnnParams_.singleCoreReduceNTail);
    OP_LOGD(nodeName_, "baseReduceN is %ld.", rnnParams_.baseReduceN);
    OP_LOGD(nodeName_, "nReduceCnt is %ld.", rnnParams_.nReduceCnt);
    OP_LOGD(nodeName_, "maxReduceNumOnce is %ld.", rnnParams_.maxReduceNumOnce);
    OP_LOGD(nodeName_, "reduceBlockSize is %ld.", rnnParams_.reduceBlockSize);
    OP_LOGD(nodeName_, "gateOrder is %ld.", rnnParams_.gateOrder);
    OP_LOGD(nodeName_, "direction is %ld.", rnnParams_.direction);
    OP_LOGD(nodeName_, "cellClip is %f.", rnnParams_.cellClip);
    OP_LOGD(nodeName_, "forgetBias is %f.", rnnParams_.forgetBias);
    OP_LOGD(nodeName_, "inputSizeAligned is %ld.", rnnParams_.inputSizeAligned);
    OP_LOGD(nodeName_, "hiddenSizeAligned is %ld.", rnnParams_.hiddenSizeAligned);
    OP_LOGD(nodeName_, "oneLineAligned is %ld.", rnnParams_.oneLineAligned);

    LogCutBatchTilingParam("dxhInputParam_", dxhInputParam_);
    LogCutBatchTilingParam("dxhHiddenParam_", dxhHiddenParam_);
    LogCutBatchTilingParam("xhInputParam_", xhInputParam_);
    LogCutBatchTilingParam("xhHiddenParam_", xhHiddenParam_);

    OP_LOGD(nodeName_, "End printing");
    OP_LOGD(nodeName_, "tiling end running");
}

static ge::graphStatus TilingFunc4SingleLayerLstmGrad(gert::TilingContext* context)
{
    SingleLayerLstmGradTiling tilingObject(context);
    if (tilingObject.Init()!=ge::GRAPH_SUCCESS){
        OP_LOGE(context->GetNodeName(),  "Tiling Init failed!");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForSingleLayerLstmGrad(gert::TilingParseContext *context)
{
  return ge::GRAPH_SUCCESS;
}


IMPL_OP_OPTILING(SingleLayerLstmGrad)
    .Tiling(TilingFunc4SingleLayerLstmGrad)
    .TilingParse<SingleLayerLstmGradCompileInfo>(TilingPrepareForSingleLayerLstmGrad);
}