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
 * \file non_zero_tiling_arch35.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "non_zero_tiling_arch35.h"

using namespace Ops::Base;

namespace optiling {
constexpr size_t ATTR_TRANSPOSE_IDX = 0;
constexpr size_t ATTR_DTYPE_IDX = 1;
constexpr size_t INPUT_X_IDX = 0;
constexpr size_t OUTPUT_Y_IDX = 1;
constexpr size_t SHAPE_DIM_0 = 0;
constexpr size_t SHAPE_DIM_1 = 1;
constexpr size_t SHAPE_DIM_2 = 2;
constexpr size_t SHAPE_DIM_3 = 3;
constexpr size_t SHAPE_DIM_4 = 4;
constexpr size_t SHAPE_DIM_5 = 5;
constexpr size_t SHAPE_DIM_6 = 6;
constexpr size_t SHAPE_DIM_7 = 7;
constexpr size_t SHAPE_DIM_8 = 8;
constexpr size_t DIV_NUM = 2;
constexpr int64_t SHAPE_DIM_MAX = 7;
constexpr int64_t WORKSPACE_SIZE_OFFSET = 9216L; // 72 * 128
constexpr int64_t WORKSPACE_SIZE = 16 * 1024 * 1024 + WORKSPACE_SIZE_OFFSET;
constexpr int64_t TMP_UB_SIZE_SMALL = 128;
constexpr int64_t TMP_UB_SIZE_BIG = 2304L;    // 72 * 32
constexpr int64_t TMP_UB_SIZE_BIGMASK = 9216; // 72 * 128
constexpr int64_t ALIGN_UB_SIZE = 256;
constexpr int64_t ALIGN_UB_32 = 32;
constexpr int64_t ALIGN_NUM = 32;
constexpr int64_t ALIGN_NUM_8 = 8;
constexpr int64_t QUICK_DIV_NUM_32 = 32;
constexpr int64_t MASK_UB_DIV_NUM = 8;
constexpr int64_t B8_BYTES = 1;
constexpr int64_t B16_BYTES = 2;
constexpr int64_t B32_BYTES = 4;
constexpr int64_t B64_BYTES = 8;
constexpr int64_t UB_REG_SIZE = 32;
constexpr int64_t VLEN_SIZE = 256;
constexpr int64_t UNPACK_NUM = 4;
constexpr int64_t VSQZ_UB_SIZE = 4;
constexpr int64_t NUM_1 = 1;
constexpr int64_t NUM_2 = 2;
constexpr int64_t NUM_4 = 4;
constexpr int64_t NUM_8 = 8;
constexpr int64_t NUM_32 = 32;
constexpr int64_t NUM_64 = 64;
constexpr int64_t NUM_128 = 128;
constexpr int64_t NUM_256 = 256;
constexpr int64_t HALF_PARAM = 2;

constexpr int64_t DATA_TYPE_INT8 = 0;   /* *< int8 */
constexpr int64_t DATA_TYPE_INT16 = 1;  /* *< int16 */
constexpr int64_t DATA_TYPE_INT32 = 2;  /* *< int32 */
constexpr int64_t DATA_TYPE_FP16 = 3;   /* *< fp16 */
constexpr int64_t DATA_TYPE_FP32 = 4;   /* *< fp32 */
constexpr int64_t DATA_TYPE_INT64 = 5;  /* *< int64 */
constexpr int64_t DATA_TYPE_UINT64 = 6; /* *< uint64 */
constexpr int64_t DATA_TYPE_UINT8 = 7;  /* *< uint8 */
constexpr int64_t DATA_TYPE_UINT16 = 8; /* *< uint16 */
constexpr int64_t DATA_TYPE_UINT32 = 9; /* *< uint32 */
constexpr int64_t DATA_TYPE_FP64 = 10;  /* *< fp64 */
constexpr int64_t DATA_TYPE_BFP16 = 11; /* *< bfp16 */

constexpr float HALF_RATE = 0.5;
constexpr float QUARTER_RATE = 0.25;

constexpr int64_t TILING_KEY_BIG_MASK_1_DIM = 20001;
constexpr int64_t TILING_KEY_BIG_MASK_2_DIM = 20002;
constexpr int64_t TILING_KEY_BIG_MASK_3_DIM = 20003;
constexpr int64_t TILING_KEY_BIG_MASK_4_DIM = 20004;
constexpr int64_t TILING_KEY_BIG_MASK_5_DIM = 20005;
constexpr int64_t TILING_KEY_BIG_MASK_6_DIM = 20006;
constexpr int64_t TILING_KEY_BIG_MASK_7_DIM = 20007;
constexpr int64_t TILING_KEY_BIG_MASK_8_DIM = 20008;
constexpr int64_t TILING_KEY_NULL_INPUT_TENSOR = 30001;
constexpr size_t WORKSPACE_SIZE_NULL_TENSOR = 16 * 1024 * 1024;

constexpr int64_t TILING_KEY_FULL_LOAD_BASE = 40000;
constexpr int64_t FULL_LOAD_DIM_1_HRESHOLD = 16384; // 1维进入全载模板的数据量阈值 16K
constexpr int64_t FULL_LOAD_DIM_2_HRESHOLD = 8192;  // 2维进入全载模板的数据量阈值 8K
constexpr int64_t FULL_LOAD_DIM_3_HRESHOLD = 8192;  // 3维进入全载模板的数据量阈值 8K
constexpr int64_t FULL_LOAD_DIM_4_HRESHOLD = 4096;  // 4维进入全载模板的数据量阈值 4K

class NonZeroAscendCTilingImpl {
public:
    explicit NonZeroAscendCTilingImpl(gert::TilingContext* context) : context_(context) {};

    ge::graphStatus Init(const NonZeroCompileInfo* compileInfo);
    ge::graphStatus DoTiling();

private:
    ge::graphStatus CheckInputOutputSize();
    void MatchTilingStrategyAndCalcUbSizeInfo(int64_t inputDataSize);
    void MatchTilingStrategyForFullLoad(int64_t inputDataSize);
    void CalcMaskUbSize(int64_t inputDtypeSize);
    void CalcMaskLoopNum(int64_t numInput);
    void FillTilingData();
    void PrintTilingData();
    void CalcUbSizeInfoSmallMask();
    void CalcUbSizeInfoBigMask();
    void CalcMulInDim(std::vector<int64_t> shape, int64_t dimSize);
    ge::graphStatus CalcQuickDivParams();
    void PrintNullTensorTilingData();
    ge::graphStatus DoTilingForNullInputTensor();
    bool IsUseFullLoad() const;

private:
    int64_t inputDims_ = 0;
    int64_t realCoreNum_ = 0;
    int64_t numPerCore_ = 0;
    int64_t numTailCore_ = 0;
    int64_t ubFactorNum_ = 0;
    int64_t loopNumPerCore_ = 0;
    int64_t loopTailPerCore_ = 0;
    int64_t loopNumTailCore_ = 0;
    int64_t loopTailTailCore_ = 0;

    int64_t needTranspose_ = 0;
    int64_t tilingKey_ = 0;

    int64_t offsetInt32Trans_ = 0;
    int64_t offsetInt64_ = 0;
    int64_t maskLoopNum_ = 0;

    int64_t loopNumO_ = 0;
    int64_t beforeNumO_ = 0;
    int64_t loopTailO_ = 0;
    int64_t loopNumTo_ = 0;
    int64_t loopTailTo_ = 0;
    int64_t maskLoopNumO_ = 0;

    int64_t xInputSize_ = 0;
    int64_t maskSize_ = 0;

    int64_t ubSize_ = 0;
    int64_t coreNum_ = 0;
    int64_t inputUbSize_ = 0;
    int64_t maskUbSize_ = 0;
    int64_t intputDtypeSize_ = 0;
    int64_t outputDtypeSize_ = 0;

    int64_t dimSize_ = 0;
    int64_t vRegSize_ = 0;
    int64_t ubAlignNum_ = 0;
    int64_t maskUbSizeThres_ = 0; // 模版切分阈值
    int64_t innerSize_ = 1;

    int64_t mulInDimRList_[SHAPE_DIM_MAX] = {0, 0, 0, 0, 0, 0, 0};
    int64_t quickDivRKList_[SHAPE_DIM_MAX] = {0, 0, 0, 0, 0, 0, 0};
    int64_t quickDivRMList_[SHAPE_DIM_MAX] = {0, 0, 0, 0, 0, 0, 0};
    ge::DataType inputDtype_ = ge::DataType::DT_MAX;
    ge::DataType outputDtype_ = ge::DataType::DT_MAX;
    gert::Shape xShape_;
    gert::Shape yShape_;
    gert::TilingContext* context_ = nullptr;
    NonZeroTilingData tilingData_;
};

ge::graphStatus NonZeroAscendCTilingImpl::Init(const NonZeroCompileInfo* compileInfo)
{
    OP_LOGD(context_->GetNodeName(), "Enter NonZeroAscendCTilingImpl init.");
    coreNum_ = compileInfo->coreNum;
    ubSize_ = compileInfo->ubSize;
    vRegSize_ = compileInfo->vRegSize;
    OP_CHECK_IF(
        coreNum_ <= 0 || ubSize_ <= 0 || vRegSize_ <= 0,
        OP_LOGE(context_->GetNodeName(), "coreNum or ubSize or vRegSize is small than zero"), return ge::GRAPH_FAILED);

    // get attrs: transpose
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const bool* transposePtr = attrs->GetAttrPointer<bool>(ATTR_TRANSPOSE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, transposePtr);
    needTranspose_ = *transposePtr ? 0 : 1;

    // get dtype, format
    auto inputXDesc = context_->GetInputDesc(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXDesc);
    auto outputYDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYDesc);
    inputDtype_ = inputXDesc->GetDataType();
    outputDtype_ = outputYDesc->GetDataType();

    intputDtypeSize_ = GetSizeByDataType(inputDtype_);
    outputDtypeSize_ = GetSizeByDataType(outputDtype_);
    OP_CHECK_IF(intputDtypeSize_ <= 0, OP_LOGE(context_->GetNodeName(), "dtype size is 0"), return ge::GRAPH_FAILED);

    OP_LOGD(context_->GetNodeName(), "Exit NonZeroAscendCTilingImpl init.");
    return ge::GRAPH_SUCCESS;
}

void NonZeroAscendCTilingImpl::CalcMaskUbSize(int64_t inputDtypeSize)
{
    maskUbSize_ =
        CeilDiv(numPerCore_, vRegSize_ / inputDtypeSize) * (UNPACK_NUM / inputDtypeSize) * UB_REG_SIZE;
}

void NonZeroAscendCTilingImpl::MatchTilingStrategyAndCalcUbSizeInfo(int64_t inputDataSize)
{
    numPerCore_ = CeilDiv(inputDataSize, coreNum_);
    CalcMaskUbSize(intputDtypeSize_);
    if (maskUbSize_ < maskUbSizeThres_) {
        if (inputDims_ == SHAPE_DIM_1) {
            tilingKey_ = 1;
        } else if (needTranspose_ == 1) {
            tilingKey_ = inputDims_;
        } else {
            tilingKey_ = inputDims_ + SHAPE_DIM_MAX;
        }

        realCoreNum_ = CeilDiv(inputDataSize, numPerCore_);
        numTailCore_ = inputDataSize - (realCoreNum_ - 1) * numPerCore_;
        CalcUbSizeInfoSmallMask();
    } else {
        if (inputDims_ == SHAPE_DIM_1) {
            tilingKey_ = TILING_KEY_BIG_MASK_1_DIM;
        } else if (inputDims_ == SHAPE_DIM_2) {
            tilingKey_ = TILING_KEY_BIG_MASK_2_DIM;
        } else if (inputDims_ == SHAPE_DIM_3) {
            tilingKey_ = TILING_KEY_BIG_MASK_3_DIM;
        } else if (inputDims_ == SHAPE_DIM_4) {
            tilingKey_ = TILING_KEY_BIG_MASK_4_DIM;
        } else if (inputDims_ == SHAPE_DIM_5) {
            tilingKey_ = TILING_KEY_BIG_MASK_5_DIM;
        } else if (inputDims_ == SHAPE_DIM_6) {
            tilingKey_ = TILING_KEY_BIG_MASK_6_DIM;
        } else if (inputDims_ == SHAPE_DIM_7) {
            tilingKey_ = TILING_KEY_BIG_MASK_7_DIM;
        } else if (inputDims_ == SHAPE_DIM_8) {
            tilingKey_ = TILING_KEY_BIG_MASK_8_DIM;
        }
        numPerCore_ = inputDataSize / coreNum_;
        realCoreNum_ = inputDataSize >= coreNum_ ? coreNum_ : inputDataSize;
        numTailCore_ = inputDataSize - realCoreNum_ * numPerCore_;
        CalcUbSizeInfoBigMask();
    }
}

void NonZeroAscendCTilingImpl::MatchTilingStrategyForFullLoad(int64_t inputDataSize)
{
    realCoreNum_ = 1;
    numPerCore_ = inputDataSize;
    tilingKey_ = TILING_KEY_FULL_LOAD_BASE + inputDims_;
    CalcMaskUbSize(intputDtypeSize_);
    CalcUbSizeInfoSmallMask();
}

void NonZeroAscendCTilingImpl::FillTilingData()
{
    tilingData_.set_inputDims(inputDims_);
    tilingData_.set_realCoreNum(realCoreNum_);
    tilingData_.set_numPerCore(numPerCore_);
    tilingData_.set_numTailCore(numTailCore_);
    tilingData_.set_ubFactorNum(ubFactorNum_);
    tilingData_.set_loopNumPerCore(loopNumPerCore_);
    tilingData_.set_loopTailPerCore(loopTailPerCore_);
    tilingData_.set_loopNumTailCore(loopNumTailCore_);
    tilingData_.set_loopTailTailCore(loopTailTailCore_);

    tilingData_.set_needTranspose(needTranspose_);
    tilingData_.set_tilingKey(tilingKey_);
    tilingData_.set_offsetInt32Trans(offsetInt32Trans_);
    tilingData_.set_offsetInt64(offsetInt64_);
    tilingData_.set_maskLoopNum(maskLoopNum_);

    tilingData_.set_loopNumO(loopNumO_);
    tilingData_.set_beforeNumO(beforeNumO_);
    tilingData_.set_loopTailO(loopTailO_);
    tilingData_.set_loopNumTo(loopNumTo_);
    tilingData_.set_loopTailTo(loopTailTo_);
    tilingData_.set_maskLoopNumO(maskLoopNumO_);

    tilingData_.set_xInputSize(inputUbSize_);
    tilingData_.set_maskSize(maskUbSize_);

    tilingData_.set_mulInDimRList(mulInDimRList_);
    tilingData_.set_quickDivRKList(quickDivRKList_);
    tilingData_.set_quickDivRMList(quickDivRMList_);

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
}

void NonZeroAscendCTilingImpl::PrintTilingData()
{
    OP_LOGI(
        context_->GetNodeName(),
        "tilingData is tilingKey:%ld, inputDims:%ld,realCoreNum:%ld, numPerCore:%ld, numTailCore:%ld,\
          ubFactorNum:%ld, loopNumPerCore:%ld, loopTailPerCore:%ld, loopNumTailCore:%ld, loopTailTailCore:%ld,\
          needTranspose:%ld, offsetInt32Trans:%ld,offsetInt64:%ld, maskLoopNum:%ld,\
          loopNumO:%ld,beforeNumO:%ld, loopTailO:%ld, loopNumTo:%ld,\
          loopTailTo:%ld, maskLoopNumO:%ld, xInputSize:%ld, maskSize:%ld",
        tilingData_.get_tilingKey(), tilingData_.get_inputDims(), tilingData_.get_realCoreNum(),
        tilingData_.get_numPerCore(), tilingData_.get_numTailCore(), tilingData_.get_ubFactorNum(),
        tilingData_.get_loopNumPerCore(), tilingData_.get_loopTailPerCore(), tilingData_.get_loopNumTailCore(),
        tilingData_.get_loopTailTailCore(), tilingData_.get_needTranspose(),

        tilingData_.get_offsetInt32Trans(), tilingData_.get_offsetInt64(), tilingData_.get_maskLoopNum(),

        tilingData_.get_loopNumO(), tilingData_.get_beforeNumO(), tilingData_.get_loopTailO(),
        tilingData_.get_loopNumTo(), tilingData_.get_loopTailTo(), tilingData_.get_maskLoopNumO(),

        tilingData_.get_xInputSize(), tilingData_.get_maskSize());

    for (int64_t i = 0; i < SHAPE_DIM_MAX; i++) {
        OP_LOGI(
            context_->GetNodeName(), "mulInDimRList:%ld, quickDivRKList:%ld, quickDivRMList:%ld", mulInDimRList_[i],
            quickDivRKList_[i], quickDivRMList_[i]);
    }
}

void NonZeroAscendCTilingImpl::CalcMulInDim(std::vector<int64_t> shape, int64_t dimSize)
{
    int64_t tmpProduct = 1;
    std::vector<int64_t> innerMul(SHAPE_DIM_MAX, 1);
    for (int64_t i = dimSize - 1; i > 0; i--) {
        if (i == dimSize - 1) {
            innerMul[i - 1] = shape[i];
            tmpProduct = innerMul[i - 1];
            continue;
        }
        innerMul[i - 1] = shape[i] * tmpProduct;
        tmpProduct = innerMul[i - 1];
    }
    for (int64_t idx = 0; idx < SHAPE_DIM_MAX; idx++) {
        mulInDimRList_[idx] = innerMul[idx];
    }
}

ge::graphStatus NonZeroAscendCTilingImpl::CalcQuickDivParams()
{
    // calc quick div params rk and rm
    for (int64_t i = 0; i < SHAPE_DIM_MAX; i++) {
        uint64_t c = mulInDimRList_[i];
        OP_CHECK_IF(c <= 0, OP_LOGE(context_->GetNodeName(), "divisor c is small than zero"), return ge::GRAPH_FAILED);
        quickDivRKList_[i] = std::ceil(std::log2(c));
        quickDivRMList_[i] =
            std::ceil(std::exp2(quickDivRKList_[i] + QUICK_DIV_NUM_32) / c) - std::exp2(QUICK_DIV_NUM_32);
    }
    return ge::GRAPH_SUCCESS;
}

static std::vector<int64_t> GetXShape(const gert::Shape& shape)
{
    int32_t dim_num = static_cast<int32_t>(shape.GetDimNum());
    std::vector<int64_t> param(dim_num);
    for (int32_t i = 0; i < dim_num; i++) {
        param[i] = shape.GetDim(i);
    }
    return param;
}

void NonZeroAscendCTilingImpl::CalcMaskLoopNum(int64_t numInput)
{
    ubFactorNum_ = FloorDiv(numInput, (vRegSize_ / intputDtypeSize_)) * (vRegSize_ / intputDtypeSize_);
    maskLoopNum_ =
        (ubFactorNum_ / (vRegSize_ / intputDtypeSize_) * (UNPACK_NUM / intputDtypeSize_) * UB_REG_SIZE) / UNPACK_NUM;
}

void NonZeroAscendCTilingImpl::CalcUbSizeInfoSmallMask()
{
    inputUbSize_ = FloorDiv(((ubSize_ - TMP_UB_SIZE_BIG - maskUbSize_) / DIV_NUM), static_cast<uint64_t>(ALIGN_UB_32)) * ALIGN_UB_32;
    int64_t numInput = inputUbSize_ / intputDtypeSize_;

    CalcMaskLoopNum(numInput);
    loopNumPerCore_ = numPerCore_ / ubFactorNum_;
    if (numPerCore_ % ubFactorNum_ == 0) {
        loopNumPerCore_ = loopNumPerCore_ - 1;
        CalcMaskUbSize(intputDtypeSize_);
    }
    loopTailPerCore_ = numPerCore_ - loopNumPerCore_ * ubFactorNum_;
    loopNumTailCore_ = numTailCore_ / ubFactorNum_;
    if (numTailCore_ % ubFactorNum_ == 0) {
        loopNumTailCore_ = loopNumTailCore_ - 1;
    }
    loopTailTailCore_ = numTailCore_ - loopNumTailCore_ * ubFactorNum_;
    inputUbSize_ = FloorDiv(((ubSize_ - TMP_UB_SIZE_BIG - maskUbSize_) / DIV_NUM), static_cast<uint64_t>(ALIGN_UB_32)) * ALIGN_UB_32;
    int32_t inputDimsNew = inputDims_;
    // 输出是int32且需要转置场景
    if ((inputDims_ != 1) && outputDtypeSize_ == B32_BYTES && needTranspose_) {
        inputDimsNew = inputDims_ + 1;
    }

    int64_t numInputO = inputUbSize_ / outputDtypeSize_;
    beforeNumO_ = numInputO / inputDimsNew;
    beforeNumO_ = (beforeNumO_ / NUM_64) * NUM_64; // repeat对齐

    loopNumO_ = numPerCore_ / beforeNumO_;
    loopTailO_ = numPerCore_ - loopNumO_ * beforeNumO_;
    loopNumTo_ = numTailCore_ / beforeNumO_;
    loopTailTo_ = numTailCore_ - beforeNumO_ * loopNumTo_;
    maskLoopNumO_ = (beforeNumO_ / NUM_64) * NUM_8;

    int64_t TmpBeforeNumO = beforeNumO_;
    if (loopNumO_ == 0) {
        TmpBeforeNumO = CeilDiv(numPerCore_, ALIGN_NUM_8) * ALIGN_NUM_8;
        inputUbSize_ = TmpBeforeNumO * inputDimsNew * outputDtypeSize_;
    }

    if ((inputDims_ != 1) && outputDtypeSize_ == B64_BYTES) {
        offsetInt64_ = TmpBeforeNumO * inputDims_;
        offsetInt32Trans_ = 0;
    } else if ((inputDims_ != 1) && (outputDtypeSize_ == B32_BYTES) && (needTranspose_)) {
        offsetInt64_ = 0;
        offsetInt32Trans_ = TmpBeforeNumO * inputDims_;
    } else if (inputDims_ == 1 && outputDtypeSize_ == B64_BYTES) {
        offsetInt64_ = 0;
        offsetInt32Trans_ = TmpBeforeNumO;
    } else {
        offsetInt64_ = 0;
        offsetInt32Trans_ = 0;
    }
}

void NonZeroAscendCTilingImpl::CalcUbSizeInfoBigMask()
{
    ubFactorNum_ = (ubSize_ - TMP_UB_SIZE_BIGMASK) / static_cast<int64_t>(DIV_NUM) /
                   (intputDtypeSize_ + VSQZ_UB_SIZE + inputDims_ * outputDtypeSize_);
    // 按照256Byte向下对齐
    ubAlignNum_ = ALIGN_UB_SIZE / intputDtypeSize_;
    ubFactorNum_ = FloorDiv(ubFactorNum_, ubAlignNum_) * ubAlignNum_;

    loopNumPerCore_ = numPerCore_ / ubFactorNum_;
    loopNumTailCore_ = numPerCore_ % ubFactorNum_;
}

void NonZeroAscendCTilingImpl::PrintNullTensorTilingData()
{
    OP_LOGI(
        context_->GetNodeName(), "tilingData is inputDims:%ld, realCoreNum:%ld, needTranspose:%ld",
        tilingData_.get_inputDims(), tilingData_.get_realCoreNum(), tilingData_.get_needTranspose());
    return;
}

ge::graphStatus NonZeroAscendCTilingImpl::DoTilingForNullInputTensor()
{
    OP_LOGD(context_->GetNodeName(), "Enter DoTilingForNullInputTensor.");
    tilingData_.set_inputDims(inputDims_);
    tilingData_.set_realCoreNum(1);
    tilingData_.set_needTranspose(needTranspose_);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    PrintNullTensorTilingData();

    context_->SetBlockDim(1);
    context_->SetTilingKey(TILING_KEY_NULL_INPUT_TENSOR);

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = WORKSPACE_SIZE_NULL_TENSOR;

    OP_LOGD(context_->GetNodeName(), "Exit DoTilingForNullInputTensor.");
    return ge::GRAPH_SUCCESS;
}

bool NonZeroAscendCTilingImpl::IsUseFullLoad() const
{
    // 1维进入全载模板判断
    if (inputDims_ == SHAPE_DIM_1 && innerSize_ <= FULL_LOAD_DIM_1_HRESHOLD) {
        return true;
    }

    // 2维进入全载模板判断
    if (inputDims_ == SHAPE_DIM_2 && innerSize_ <= FULL_LOAD_DIM_2_HRESHOLD) {
        return true;
    }

    // 3维进入全载模板判断
    if (inputDims_ == SHAPE_DIM_3 && innerSize_ <= FULL_LOAD_DIM_3_HRESHOLD) {
        return true;
    }

    // 4维进入全载模板判断
    if (inputDims_ == SHAPE_DIM_4 && innerSize_ <= FULL_LOAD_DIM_4_HRESHOLD) {
        return true;
    }

    return false;
}

ge::graphStatus NonZeroAscendCTilingImpl::DoTiling()
{
    OP_LOGD(context_->GetNodeName(), "Enter NonZeroAscendCTilingImpl DoTiling.");
    // get xshape, yshape
    auto xStorage = context_->GetInputShape(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xStorage);
    xShape_ = EnsureNotScalar(xStorage->GetStorageShape());
    const std::vector<int64_t> xRunShape = GetXShape(xShape_);
    inputDims_ = static_cast<int64_t>(xShape_.GetDimNum());
    dimSize_ = xRunShape.size();

    // Null input tensor
    if (xShape_.GetShapeSize() == 0) {
        return DoTilingForNullInputTensor();
    }
    // calculate shape multiplier in the R dimension
    CalcMulInDim(xRunShape, dimSize_);
    // calculate and verify quick division parameters
    if (CalcQuickDivParams() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < xShape_.GetDimNum(); i++) {
        innerSize_ *= xShape_.GetDim(i);
    }
    maskUbSizeThres_ = (ubSize_ - TMP_UB_SIZE_BIGMASK) / HALF_PARAM;

    // match tiling strategy and calc Ub size info
    if (IsUseFullLoad()) {
        MatchTilingStrategyForFullLoad(innerSize_);
    } else {
        MatchTilingStrategyAndCalcUbSizeInfo(innerSize_);
    }

    // fill data
    FillTilingData();
    // print data
    PrintTilingData();
    // set block dim and tilingKey
    context_->SetBlockDim(realCoreNum_);
    context_->SetTilingKey(tilingData_.get_tilingKey());

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = WORKSPACE_SIZE;
    OP_LOGD(context_->GetNodeName(), "Exit NonZeroAscendCTilingImpl DoTiling.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4NonZero(gert::TilingContext* context)
{
    auto compileInfo = reinterpret_cast<const NonZeroCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    NonZeroAscendCTilingImpl tilingImpl = NonZeroAscendCTilingImpl(context);
    if (tilingImpl.Init(compileInfo) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Tiling4NonZero init failed.");
        return ge::GRAPH_FAILED;
    }

    if (tilingImpl.DoTiling() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Tiling4NonZero do tiling failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4NonZero(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<NonZeroCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->coreNum <= 0), OP_LOGE(context->GetNodeName(), "core num invalid."), return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(
        (compileInfo->ubSize <= 0), OP_LOGE(context->GetNodeName(), "ub size invalid."), return ge::GRAPH_FAILED);
    compileInfo->vRegSize = static_cast<int64_t>(GetVRegSize(context));
    OP_CHECK_IF(
        (compileInfo->vRegSize <= 0), OP_LOGE(context->GetNodeName(), "vRegSize size invalid."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(NonZero).Tiling(Tiling4NonZero).TilingParse<NonZeroCompileInfo>(TilingPrepare4NonZero);

} // namespace optiling
