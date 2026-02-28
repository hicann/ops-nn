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
 * \file thnn_fused_lstm_cell_grad_tiling.cpp
 * \brief
 */

#include "thnn_fused_lstm_cell_grad_tiling.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"
#include "tiling/tiling_api.h"
#include "error_util.h"
#include "platform/platform_infos_def.h"
using namespace AscendC;

namespace optiling {
const std::string OP_NAME = "ThnnFusedLstmCellGrad";
const int32_t GATES_NUM = 4;
const int64_t CONST_TWO = 2;
const int64_t FP32_BYTES = 4;
const int64_t INPUT_NUM = 5;
const int64_t OUTPUT_NUM = 3;
const int64_t INPUT_DIM_NUM = 2;
const int64_t DEFAULT_UB_RESERVE_SIZE = 1024;
const int64_t BLOCK_BYTES = 32;
const int64_t DEFAULT_ALIGNED_FP32 = 8;
const int64_t DEFAULT_REDUCE_N_LIMIT = 128;
const int64_t HIDDEN_UB_NUM = 18;
const int64_t INPUT_DC_INDEX = 1;
const int64_t INPUT_CX_INDEX = 2;
const int64_t INPUT_CY_INDEX = 3;
const int64_t INPUT_STORAGE_INDEX = 4;
const int64_t OUTPUT_DGATES_INDEX = 0;
const int64_t OUTPUT_DC_PREV_INDEX = 1;
const int64_t OUTPUT_DB_INDEX = 2;

const int64_t BATCH_INDEX = 0;
const int64_t SIZE_INDEX = 1;
const int64_t MIN_BASE_SHAPE = 2048;

class ThnnFusedLstmCellGradTiling {
public:
    explicit ThnnFusedLstmCellGradTiling(gert::TilingContext* context) : context_(context){};
    ge::graphStatus Init();
    void SetTilingData();
    bool CheckParamsDtype();
    bool CheckParamsShape();
    ge::graphStatus CheckAttr();
    void PrintTilingData();
    void VectorBlockCalculate();
    void ReduceBlockCalculate();
    bool ValidateInputShape (int index, const std::vector<int64_t>& expected_dims);
    bool ValidateOutputShape (int index, const std::vector<int64_t>& expected_dims);
private:
    ThnnFusedLstmCellGradTilingData tilingData_;
    gert::TilingContext* context_ = nullptr;
    ThnnFusedLstmCellGradTilingParams rnnParams_;
    const char* nodeName_ = nullptr;
    int64_t alignedPara_ = DEFAULT_ALIGNED_FP32;
    int64_t inputDSize_ = FP32_BYTES;
    int64_t tilingKey_ = 0;
};

bool ThnnFusedLstmCellGradTiling::CheckParamsDtype()
{
    // dtype support list
    std::vector<ge::DataType> supportDtype = {ge::DT_FLOAT, ge::DT_FLOAT16};
    ge::DataType baseDtype = context_->GetInputDesc(0)->GetDataType();
    inputDSize_ = baseDtype == ge::DT_FLOAT ? FP32_BYTES : 2;
    alignedPara_ = 32 / inputDSize_;

    // input check
    for (int64_t inputIdx = 1; inputIdx < INPUT_NUM; inputIdx++) {
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

bool ThnnFusedLstmCellGradTiling::ValidateInputShape(int index, const std::vector<int64_t>& expected_dims)
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

bool ThnnFusedLstmCellGradTiling::ValidateOutputShape (int index, const std::vector<int64_t>& expected_dims)
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

bool ThnnFusedLstmCellGradTiling::CheckParamsShape()
{
    // get input shape
    auto dhyInput = context_->GetInputShape(0);
    OP_CHECK_IF(dhyInput == nullptr,
              OP_LOGE(nodeName_, "input grad_hy is nullptr"),
              return false);
    auto dhyShape = dhyInput->GetStorageShape();
    if (dhyShape.GetDimNum() != INPUT_DIM_NUM) {
        OP_LOGE(nodeName_, "Input grad_hy must be 2D tensor");
        return false;
    }
    rnnParams_.batch = dhyShape.GetDim(BATCH_INDEX);
    rnnParams_.hiddenSize = dhyShape.GetDim(SIZE_INDEX);
    OP_TILING_CHECK(
        rnnParams_.hiddenSize <= 0 || rnnParams_.batch <= 0,
        VECTOR_INNER_ERR_REPORT_TILIING(nodeName_,
        "ThnnFusedLstmCellGrad tensor shape not support 0 or negtive, please check."), return false);

    std::vector<int64_t> hDim = {rnnParams_.batch, rnnParams_.hiddenSize};
    std::vector<int64_t> biasDim = {GATES_NUM * rnnParams_.hiddenSize};
    std::vector<int64_t> gatesDim = {rnnParams_.batch, GATES_NUM * rnnParams_.hiddenSize};

    bool ret =
      ValidateInputShape(INPUT_DC_INDEX, hDim) &&
      ValidateInputShape(INPUT_CX_INDEX, hDim) &&
      ValidateInputShape(INPUT_CY_INDEX, hDim) &&
      ValidateInputShape(INPUT_STORAGE_INDEX, gatesDim) &&
      ValidateOutputShape(OUTPUT_DGATES_INDEX, gatesDim) &&
      ValidateOutputShape(OUTPUT_DC_PREV_INDEX, hDim);
    ret = rnnParams_.isBias ? ValidateOutputShape(OUTPUT_DB_INDEX, biasDim) && ret : ret;
    return ret;
}

ge::graphStatus ThnnFusedLstmCellGradTiling::CheckAttr()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    rnnParams_.isBias = *(attrs->GetAttrPointer<bool>(0)) == true ? 1 : 0;
    return ge::GRAPH_SUCCESS;
}

void ThnnFusedLstmCellGradTiling::VectorBlockCalculate()
{
    int64_t hiddenUbNum = HIDDEN_UB_NUM;
    int64_t sizeLimit = (rnnParams_.ubSize - DEFAULT_UB_RESERVE_SIZE) / hiddenUbNum / FP32_BYTES / alignedPara_ *
        alignedPara_;
    //cut M
    rnnParams_.singleCoreM = Ops::Base::CeilDiv(rnnParams_.batch, rnnParams_.sysAivCoreNum);
    rnnParams_.mCnt = Ops::Base::CeilDiv(rnnParams_.batch, rnnParams_.singleCoreM);
    rnnParams_.singleCoreMTail = rnnParams_.batch - (rnnParams_.mCnt - 1)  * rnnParams_.singleCoreM;
    rnnParams_.nCnt = rnnParams_.batch < rnnParams_.sysAivCoreNum ?
        Ops::Base::CeilDiv(rnnParams_.sysAivCoreNum, rnnParams_.mCnt) : 1;
    //cut N
    rnnParams_.singleCoreN = Ops::Base::CeilAlign(Ops::Base::CeilDiv(rnnParams_.hiddenSize, rnnParams_.nCnt),
                                                  alignedPara_);
    rnnParams_.nCnt = Ops::Base::CeilDiv(rnnParams_.hiddenSize, rnnParams_.singleCoreN);
    rnnParams_.singleCoreNTail = rnnParams_.hiddenSize - (rnnParams_.nCnt - 1) * rnnParams_.singleCoreN;
    rnnParams_.baseN = rnnParams_.singleCoreN < sizeLimit ?
                       Ops::Base::CeilAlign(rnnParams_.singleCoreN, alignedPara_) : sizeLimit;
    rnnParams_.baseM = sizeLimit / rnnParams_.baseN;
}

void ThnnFusedLstmCellGradTiling::ReduceBlockCalculate()
{
    rnnParams_.baseReduceN = Ops::Base::CeilDiv(
        Ops::Base::CeilDiv(rnnParams_.hiddenSize * GATES_NUM, rnnParams_.sysAivCoreNum), alignedPara_) * alignedPara_;
    rnnParams_.baseReduceN =  rnnParams_.baseReduceN > DEFAULT_REDUCE_N_LIMIT ? DEFAULT_REDUCE_N_LIMIT :
        rnnParams_.baseReduceN;
    rnnParams_.maxReduceNumOnce = (rnnParams_.ubSize - DEFAULT_UB_RESERVE_SIZE) / FP32_BYTES / rnnParams_.baseReduceN;
    rnnParams_.reduceBlockSize = rnnParams_.batch;

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

ge::graphStatus ThnnFusedLstmCellGradTiling::Init()
{
    nodeName_ = context_->GetNodeName();
    OP_LOGD(nodeName_, "ThnnFusedLstmCellGrad tiling starts running");
    auto platformInfo = context_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t aivCoreNum = ascendcPlatform.GetCoreNumAiv();
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
    OP_TILING_CHECK(CheckAttr() != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "check attr fail."),
                    return ge::GRAPH_FAILED);
    vector<int64_t> dims = {MIN_BASE_SHAPE};
    uint32_t minSize = 0;
    uint32_t maxSize = 0;
    GetTanhMaxMinTmpSize(ge::Shape(dims), CONST_TWO, false, minSize, maxSize);
    rnnParams_.ubSize = ubSizePlatForm;
    rnnParams_.sysAivCoreNum = aivCoreNum;
    VectorBlockCalculate();
    ReduceBlockCalculate();
    SetTilingData();
    if (rnnParams_.isBias) {
        context_->SetScheduleMode(1);
    }
    context_->SetBlockDim(rnnParams_.sysAivCoreNum);
    context_->SetTilingKey(tilingKey_);
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    int64_t workspaceReduce = rnnParams_.batch > rnnParams_.maxReduceNumOnce ? 
        GATES_NUM * rnnParams_.hiddenSize * FP32_BYTES * Ops::Base::CeilDiv(rnnParams_.batch, rnnParams_.maxReduceNumOnce) : 0;
    int64_t workspaceDgates = (inputDSize_ != FP32_BYTES && rnnParams_.isBias == 1) ?
                              GATES_NUM * rnnParams_.hiddenSize * FP32_BYTES * rnnParams_.batch : 0;
    currentWorkspace[0] = sysWorkspaceSize + workspaceReduce + workspaceDgates;
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

void ThnnFusedLstmCellGradTiling::SetTilingData()
{   
    tilingData_.set_ubSize(rnnParams_.ubSize);
    tilingData_.set_batch(rnnParams_.batch);
    tilingData_.set_hiddenSize(rnnParams_.hiddenSize);
    tilingData_.set_isBias(rnnParams_.isBias);
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

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
}

void ThnnFusedLstmCellGradTiling::PrintTilingData()
{
    OP_LOGD(nodeName_, "Start printing");
    OP_LOGD(nodeName_, "ubSize is %ld.", rnnParams_.ubSize);
    OP_LOGD(nodeName_, "batch is %ld.", rnnParams_.batch);
    OP_LOGD(nodeName_, "hiddenSize is %ld.", rnnParams_.hiddenSize);
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
    OP_LOGD(nodeName_, "isBias is %ld.", rnnParams_.isBias);

    OP_LOGD(nodeName_, "End printing");
    OP_LOGD(nodeName_, "tiling end running");
}

static ge::graphStatus TilingFunc4ThnnFusedLstmCellGrad(gert::TilingContext* context)
{
    ThnnFusedLstmCellGradTiling tilingObject(context);
    if (tilingObject.Init()!=ge::GRAPH_SUCCESS){
        OP_LOGE(context->GetNodeName(),  "Tiling Init failed!");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForThnnFusedLstmCellGrad([[maybe_unused]] gert::TilingParseContext *context)
{
  return ge::GRAPH_SUCCESS;
}


IMPL_OP_OPTILING(ThnnFusedLstmCellGrad)
    .Tiling(TilingFunc4ThnnFusedLstmCellGrad)
    .TilingParse<ThnnFusedLstmCellGradCompileInfo>(TilingPrepareForThnnFusedLstmCellGrad);
}