/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/* !
 * \file group_norm_grad_regbase_tiling.cc
 * \brief
 */
#include "group_norm_grad_tiling_arch35.h"

namespace optiling {
static constexpr uint16_t INPUT_IDX_DY = 0;
static constexpr uint16_t INPUT_IDX_MEAN = 1;
static constexpr uint16_t INPUT_IDX_RSTD = 2;
static constexpr uint16_t INPUT_IDX_X = 3;
static constexpr uint16_t INPUT_IDX_GAMMA = 4;
static constexpr uint16_t OUTPUT_IDX_DX = 0;
static constexpr uint16_t OUTPUT_IDX_DGAMMA = 1;
static constexpr uint16_t OUTPUT_IDX_DBETA = 2;
static constexpr uint16_t DX_IS_REQUIRE_IDX = 2;
static constexpr uint16_t DGAMMA_IS_REQUIRE_IDX = 3;
static constexpr uint16_t DBETA_IS_REQUIRE_IDX = 4;
static constexpr uint16_t DIM0 = 0;
static constexpr uint16_t DIM1 = 1;
static constexpr uint16_t DIM2 = 2;
static constexpr uint16_t DIM3 = 3;
static constexpr uint32_t FLOAT_DTYPE_BYTES = 4;
static constexpr uint16_t BFLOAT16_DTYPE_BYTES = 2;
static constexpr uint16_t FLOAT16_DTYPE_BYTES = 2;
static constexpr int64_t RA_BINARY_ADD_THRESHOLD = 8;
static constexpr uint16_t BINARY_ADD_COEF = 2;
static constexpr int64_t SCALE_COEF_EIGHT = 8;
static constexpr uint16_t MIN_X_DIM = 2;
static constexpr uint32_t MIN_BLOCK_SIZE = 512;
static constexpr uint32_t SMALL_NG = 16;
static constexpr uint32_t STAGE0_C_PER_CORE = 2;
// the threshold for n*(d/c_hat)
static constexpr uint32_t SMALL_NG_ND_CHAT = 32;

enum class GNGDtypeKey : int
{
    FLOAT_FLOAT = 1,
    HALF_HALF = 2,
    BF16_BF16 = 3,
    HALF_FLOAT = 4,
    BF16_FLOAT = 5
};

static const std::unordered_map<ge::DataType, uint32_t> DATA_TYPE_TO_INT{
    {ge::DataType::DT_FLOAT, 1}, {ge::DataType::DT_FLOAT16, 2}, {ge::DataType::DT_BF16, 3}};

ge::graphStatus GroupNormGradRegBaseTiling::GetPlatformInfo()
{
    OP_LOGD(opName, "Tiling initing.");
    // Get compileInfo
    auto compileInfo = context_->GetCompileInfo<GroupNormGradCompileInfo>();
    ubSize_ = compileInfo->ubSizePlatForm;
    OP_TILING_CHECK(
        (ubSize_ <= 0), OP_LOGE(context_->GetNodeName(), "ubSize should be greater than zeor."),
        return ge::GRAPH_FAILED);
    coreNum_ = compileInfo->totalCoreNum;
    OP_TILING_CHECK(
        (coreNum_ <= 0), OP_LOGE(context_->GetNodeName(), "core num should be greater than zero."),
        return ge::GRAPH_FAILED);
    vectorLen_ = compileInfo->vectorLen / sizeof(float);
    OP_TILING_CHECK(
        (vectorLen_ <= 0), OP_LOGE(context_->GetNodeName(), "vectorLen should be greater than zero."),
        return ge::GRAPH_FAILED);
    blockSize_ = compileInfo->blockSize;
    OP_TILING_CHECK(
        (blockSize_ <= 0), OP_LOGE(context_->GetNodeName(), "blockSize should be greater than zero."),
        return ge::GRAPH_FAILED);
    sysWorkspaceSize_ = compileInfo->sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupNormGradRegBaseTiling::GetShapeAttrsInfo()
{
    auto dyDesc = context_->GetInputDesc(INPUT_IDX_DY);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dyDesc);
    tTypeStr_ = dyDesc->GetDataType();
    auto dyShape = context_->GetInputShape(INPUT_IDX_DY)->GetStorageShape();
    OP_TILING_CHECK(
        InputCheck(dyShape) == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "Input check failed."),
        return ge::GRAPH_FAILED);

    auto dimNum = dyShape.GetDimNum();
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    // Get ATTR
    const bool* dxRequire = attrs->GetAttrPointer<bool>(DX_IS_REQUIRE_IDX);
    dxIsRequire_ = (dxRequire == nullptr) ? true : *dxRequire;
    const bool* dgammaRequire = attrs->GetAttrPointer<bool>(DGAMMA_IS_REQUIRE_IDX);
    dgammaIsRequire_ = (dgammaRequire == nullptr) ? true : *dgammaRequire;
    const bool* dbetaRequire = attrs->GetAttrPointer<bool>(DBETA_IS_REQUIRE_IDX);
    dbetaIsRequire_ = (dbetaRequire == nullptr) ? true : *dbetaRequire;
    const int64_t* gValue = attrs->GetAttrPointer<int64_t>(0);
    OP_TILING_CHECK(
        (*gValue <= 0), OP_LOGE(context_->GetNodeName(), "numGroups must be bigger than 0."), return ge::GRAPH_FAILED);
    G_ = static_cast<int64_t>(*gValue);
    N_ = dyShape.GetDim(DIM0);
    C_ = dyShape.GetDim(DIM1);
    OP_TILING_CHECK(
        (this->N_ <= 0 || this->C_ <= 0), OP_LOGE(context_->GetNodeName(), "C, N must be bigger than 0."),
        return ge::GRAPH_FAILED);
    NxG_ = N_ * G_;
    NxC_ = N_ * C_;
    CPerG_ = C_ / G_;
    for (uint32_t dimIdx = 2; dimIdx < dimNum; dimIdx++) {
        HxW_ *= dyShape.GetDim(dimIdx);
    }
    OP_TILING_CHECK(
        ParamsCheck() == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "Params check failed."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool GroupNormGradRegBaseTiling::IsCapable()
{
    CalBinaryParams(this->HxW_, binaryAddQuotient_, binaryAddK_, binaryAddLastNum_);
    CalBinaryParams(this->CPerG_, binaryCGQuotient_, binaryCGK_, binaryCGLastNum_);
    return true;
}

ge::graphStatus GroupNormGradRegBaseTiling::InputCheck(gert::Shape& dyShape)
{
    auto xDesc = context_->GetInputDesc(INPUT_IDX_X);
    auto dxDesc = context_->GetOutputDesc(OUTPUT_IDX_DX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dxDesc);
    auto xShape = context_->GetInputShape(INPUT_IDX_X)->GetStorageShape();
    auto dxShape = context_->GetOutputShape(OUTPUT_IDX_DX)->GetStorageShape();
    ge::DataType xDtypeStr = xDesc->GetDataType();
    ge::DataType dxDtypeStr = dxDesc->GetDataType();
    OP_TILING_CHECK(
        (tTypeStr_ != dxDtypeStr || dxDtypeStr != xDtypeStr),
        OP_LOGE(context_->GetNodeName(), "x, dy, dx data type must be same."), return ge::GRAPH_FAILED);
    auto iter = DATA_TYPE_TO_INT.find(tTypeStr_);
    if (iter == DATA_TYPE_TO_INT.end()) {
        OP_LOGE(
            context_->GetNodeName(), "inputdtype [%d] must be in float32:[%d], float16:[%d], bfloat16:[%d]", tTypeStr_,
            ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT16, ge::DataType::DT_BF16);
        return ge::GRAPH_FAILED;
    }
    OP_TILING_CHECK(
        (dyShape != dxShape || dxShape != xShape), OP_LOGE(context_->GetNodeName(), "x, dy, dx shape must be same."),
        return ge::GRAPH_FAILED);
    for (uint32_t dimIdx = 0; dimIdx < dyShape.GetDimNum(); dimIdx++) {
        auto currShape = dyShape.GetDim(dimIdx);
        OP_TILING_CHECK(
            (currShape < 1), OP_LOGE(context_->GetNodeName(), "shape value must large than 1."),
            return ge::GRAPH_FAILED);
    }
    auto dimNum = dyShape.GetDimNum();
    OP_TILING_CHECK(
        (dimNum < MIN_X_DIM), OP_LOGE(context_->GetNodeName(), "x, dy, dx shape dim must large than 2."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupNormGradRegBaseTiling::ParamsCheck()
{
    // C / G must not be zero, and C must be an integer multiple of G
    // while also not exceeding the operator's current carrying capacity.
    if (CPerG_ == 0 || C_ % G_ != 0 || CPerG_ > UPPER_CARRYING_LIMIT) {
        OP_LOGE(context_->GetNodeName(), "Group_num is invalid");
        return ge::GRAPH_FAILED;
    }
    auto gammaDesc = context_->GetInputDesc(INPUT_IDX_GAMMA);
    auto meanDesc = context_->GetInputDesc(INPUT_IDX_MEAN);
    auto rstdDesc = context_->GetInputDesc(INPUT_IDX_RSTD);
    auto dGammaDesc = context_->GetOutputDesc(OUTPUT_IDX_DGAMMA);
    auto dBetaDesc = context_->GetOutputDesc(OUTPUT_IDX_DBETA);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gammaDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, meanDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, rstdDesc);
    auto gammaShape = context_->GetInputShape(INPUT_IDX_GAMMA)->GetStorageShape();
    auto meanShape = context_->GetInputShape(INPUT_IDX_MEAN)->GetStorageShape();
    auto rstdShape = context_->GetInputShape(INPUT_IDX_RSTD)->GetStorageShape();
    uTypeStr_ = gammaDesc->GetDataType();
    ge::DataType meanDtypeStr = meanDesc->GetDataType();
    ge::DataType rstdDtypeStr = rstdDesc->GetDataType();
    ge::DataType dGammaDtypeStr = dGammaDesc->GetDataType();
    ge::DataType dBetaDtypeStr = dBetaDesc->GetDataType();
    int64_t gammaSize = 1;
    for (uint32_t dimIdx = 0; dimIdx < gammaShape.GetDimNum(); dimIdx++) {
        gammaSize *= gammaShape.GetDim(dimIdx);
    }
    OP_TILING_CHECK(
        gammaShape.GetDimNum() != 1 || gammaSize != this->C_,
        OP_LOGE(
            context_->GetNodeName(), "the shape of gamma must be the same as Channel, current gammaSize is :%ld.",
            gammaSize),
        return ge::GRAPH_FAILED);
    int64_t meanSize = 1;
    for (uint32_t dimIdx = 0; dimIdx < meanShape.GetDimNum(); dimIdx++) {
        meanSize *= meanShape.GetDim(dimIdx);
    }
    int64_t rstdSize = 1;
    for (uint32_t dimIdx = 0; dimIdx < rstdShape.GetDimNum(); dimIdx++) {
        rstdSize *= rstdShape.GetDim(dimIdx);
    }
    auto iter = DATA_TYPE_TO_INT.find(uTypeStr_);
    if (iter == DATA_TYPE_TO_INT.end()) {
        OP_LOGE(
            context_->GetNodeName(), "inputdtype [%d] must be in float32:[%d], float16:[%d], bfloat16:[%d]", tTypeStr_,
            ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT16, ge::DataType::DT_BF16);
        return ge::GRAPH_FAILED;
    }
    OP_TILING_CHECK(
        (meanDtypeStr != rstdDtypeStr || rstdDtypeStr != dGammaDtypeStr || dGammaDtypeStr != dBetaDtypeStr ||
         dBetaDtypeStr != uTypeStr_),
        OP_LOGE(context_->GetNodeName(), "rstd, mean, gamma ,dbeta, dgamma data type must be same."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (tTypeStr_ == ge::DT_FLOAT && (uTypeStr_ == ge::DT_FLOAT16 || uTypeStr_ == ge::DT_BF16)),
        OP_LOGE(
            context_->GetNodeName(), "dy, x, dx is float, mean, rstd gamma, dgamma, dbeta only support float type."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        meanSize != rstdSize || meanSize != this->N_ * this->G_,
        OP_LOGE(
            context_->GetNodeName(), "the shape of mean [%ld], rstd [%ld] must equal N * G, current N * G is :%ld.",
            meanSize, rstdSize, this->N_ * this->G_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void GroupNormGradRegBaseTiling::CalBinaryParams(
    int64_t reduceNum, int64_t& binaryQuotient, uint32_t& binaryK, uint32_t& binaryLastNum)
{
    binaryQuotient = vectorLen_;
    while (binaryQuotient < reduceNum) {
        binaryQuotient *= BINARY_ADD_COEF;
    }
    binaryQuotient /= BINARY_ADD_COEF;                      // 二分折叠之后的长度
    int64_t quotientVcaddNum = binaryQuotient / vectorLen_; // 预留空间大小，存放二分折叠临时变量
    if (quotientVcaddNum <= vectorLen_) {
        binaryLastNum = static_cast<uint32_t>(quotientVcaddNum);
        binaryK = 0;
    } else {
        auto binaryAddNum = quotientVcaddNum / vectorLen_;
        binaryK = __builtin_ctzl(binaryAddNum);
        binaryLastNum = vectorLen_;
    }
}

uint32_t GroupNormGradRegBaseTiling::GetTypeSize(ge::DataType dtypeStr) const
{
    switch (dtypeStr) {
        case ge::DT_FLOAT:
            return FLOAT_DTYPE_BYTES;
        case ge::DT_BF16:
            return BFLOAT16_DTYPE_BYTES;
        case ge::DT_FLOAT16:
            return FLOAT16_DTYPE_BYTES;
        default:
            OP_LOGE(context_->GetNodeName(), "inputdtype must be in [float32, float16, bfloat16]");
            return 0;
    }
}

ge::graphStatus GroupNormGradRegBaseTiling::BlockTiling()
{
    this->tTypeBytes_ = GetTypeSize(this->tTypeStr_);
    OP_TILING_CHECK(
        this->tTypeBytes_ == 0, OP_LOGE(context_->GetNodeName(), "Calculated tTypeBytes_ is zero"),
        return ge::GRAPH_FAILED);

    int64_t cGFloatAlign = Ops::Base::CeilAlign(CPerG_, static_cast<int64_t>(this->blockSize_ / FLOAT_DTYPE_BYTES));
    int64_t cGHalfAlign = Ops::Base::CeilAlign(CPerG_, static_cast<int64_t>(this->blockSize_ / FLOAT16_DTYPE_BYTES));
    int64_t cGtempSpace = 0;
    // CGNum(4+2): gamma, dbeta, dgamma, ds, input_db, input_ds
    int64_t cgAllocSpace = 0;
    if (NxG_ <= SMALL_NG) {
        cgAllocSpace = cGFloatAlign * FLOAT_DTYPE_BYTES * (UB_COPIES_4 + UB_COPIES_2);
        if (uTypeStr_ == ge::DT_FLOAT16 || uTypeStr_ == ge::DT_BF16) {
            cGtempSpace = cGHalfAlign * FLOAT16_DTYPE_BYTES; // tBufGamma
        }
    } else {
        cgAllocSpace = cGFloatAlign * FLOAT_DTYPE_BYTES * UB_COPIES_4;
        if (uTypeStr_ == ge::DT_FLOAT16 || uTypeStr_ == ge::DT_BF16) {
            cGtempSpace = cGHalfAlign * UB_COPIES_2 * FLOAT16_DTYPE_BYTES; // tBufGamma, tBufDbeta
        }
    }

    reserveSpace_ = cgAllocSpace + cGtempSpace + UB_COPIES_2 * this->blockSize_; // 2 block for Mean, rstd buf

    // Allocate computing core
    if (NxG_ <= SMALL_NG) {
        stage0TaskNumPerCore_ = Ops::Base::CeilDiv(NxC_, static_cast<int64_t>(coreNum_));
        if (stage0TaskNumPerCore_ == 0) {
            OP_LOGE(context_->GetNodeName(), "stage0TaskNumPerCore_ is zero");
            return ge::GRAPH_FAILED;
        }
        stage0CoreUsed_ = (NxC_ - 1) / stage0TaskNumPerCore_ + 1;
        stage0TaskNumPerTailCore_ = stage0TaskNumPerCore_;
        stage0TailCore_ = stage0CoreUsed_;
        if (NxC_ % stage0CoreUsed_ != 0) {
            stage0TaskNumPerTailCore_ = stage0TaskNumPerCore_ - 1;
            stage0TailCore_ = NxC_ % stage0CoreUsed_;
        }
    }

    taskNumPerCore_ = Ops::Base::CeilAlign(NxG_, static_cast<int64_t>(coreNum_)) / coreNum_;
    stage1CoreUsed_ = (NxG_ - 1) / taskNumPerCore_ + 1;
    taskNumPerTailCore_ = taskNumPerCore_;
    tailCore_ = stage1CoreUsed_;

    if (NxG_ % stage1CoreUsed_ != 0) {
        taskNumPerTailCore_ = taskNumPerCore_ - 1;
        tailCore_ = NxG_ % stage1CoreUsed_;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupNormGradRegBaseTiling::UbTiling()
{
    OP_TILING_CHECK(
        ubSize_ < (reserveSpace_), OP_LOGE(context_->GetNodeName(), "Error:[UbTiling] ubSize less than reserveSpace"),
        return ge::GRAPH_FAILED);
    auto canUseUbSize = (ubSize_ - reserveSpace_) / DOUBLE_BUFFER;
    // mode0 x, dy 全部load成float32数据类型，放在UB中计算完dx 和 dgamma, dbeta
    int64_t tAlignFactor = this->blockSize_ / this->tTypeBytes_;
    int64_t mode0xDyDxSize = static_cast<int64_t>(Ops::Base::CeilAlign(CPerG_ * HxW_, tAlignFactor)) * this->tTypeBytes_ * UB_COPIES_3;
    // mode0 mean, rstd需要额外的空间
    mode0UbCapGNum_ = canUseUbSize / (mode0xDyDxSize + this->blockSize_ * UB_COPIES_2);
    mode1UbCapCNum_ = canUseUbSize / (static_cast<int64_t>(Ops::Base::CeilAlign(HxW_, tAlignFactor)) * this->tTypeBytes_ * UB_COPIES_3);
    // the conditions for branch small_ng
    if (mode0UbCapGNum_ > 0) {
        modeKey_ = MODE_0;
    } else if (mode1UbCapCNum_ > 0) {
        bool smallNGCase = (NxG_ <= SMALL_NG) && (stage0TaskNumPerCore_ >= STAGE0_C_PER_CORE) &&
                           ((N_ * (CPerG_ / mode1UbCapCNum_)) >= SMALL_NG_ND_CHAT);
        if (smallNGCase && (NxC_ <= MAX_C_SIZE)) {
            modeKey_ = MODE_3;
        } else {
            modeKey_ = MODE_1;
        }
    } else if (mode1UbCapCNum_ <= 0) {
        modeKey_ = MODE_2;
        return Mode2UbTiling();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupNormGradRegBaseTiling::Mode2UbTiling()
{
    // cache level is 3
    auto reserveCacheSize = CACHE_BUFF_SIZE * 3 * UB_COPIES_2 * sizeof(float);
    OP_TILING_CHECK(
        ubSize_ < (reserveSpace_ + reserveCacheSize),
        OP_LOGE(context_->GetNodeName(), "Error:[UbTiling] ubSize less than reserveSpace"), return ge::GRAPH_FAILED);
    uint32_t canUseUbSize = (ubSize_ - reserveSpace_ - reserveCacheSize) / DOUBLE_BUFFER;
    mode2UbCapEle_ = Ops::Base::FloorAlign(canUseUbSize / (this->tTypeBytes_ * UB_COPIES_3), this->blockSize_);
    // Loop Size vector Len align
    mode2OneLoopSize_ = this->vectorLen_;
    while (mode2OneLoopSize_ < (mode2UbCapEle_ / BINARY_ADD_COEF)) { // loop need split main/fold part
        mode2OneLoopSize_ *= BINARY_ADD_COEF;
    }
    mode2OneLoopSize_ /= BINARY_ADD_COEF;
    mode2MainLoopCnt_ = Ops::Base::CeilDiv(this->binaryAddQuotient_, static_cast<int64_t>(this->mode2OneLoopSize_));
    mode2FoldLoopCnt_ =
        Ops::Base::CeilDiv((this->HxW_ - this->binaryAddQuotient_), static_cast<int64_t>(this->mode2OneLoopSize_));
    OP_TILING_CHECK(
        this->binaryAddQuotient_ % this->mode2OneLoopSize_ != 0,
        OP_LOGE(
            context_->GetNodeName(),
            "Error:[UbTiling] the main block can't have tail. binaryAddQuotient = %ld, mode2OneLoopSize = %u.",
            this->binaryAddQuotient_, this->mode2OneLoopSize_),
        return ge::GRAPH_FAILED);
    mode2FoldTail_ = (this->HxW_ - this->binaryAddQuotient_) % this->mode2OneLoopSize_ == 0 ?
                         this->mode2OneLoopSize_ :
                         (this->HxW_ - this->binaryAddQuotient_) % this->mode2OneLoopSize_;
    OP_TILING_CHECK(
        this->mode2MainLoopCnt_ < 1 || this->mode2FoldLoopCnt_ < 1,
        OP_LOGE(context_->GetNodeName(), "Error:[UbTiling] mode2MainLoopCnt or mode2FoldLoopCnt less than 1"),
        return ge::GRAPH_FAILED);
    mode2UbCapEle_ = mode2OneLoopSize_ * BINARY_ADD_COEF;
    if (mode2UbCapEle_ == 0) {
        OP_LOGE(context_->GetNodeName(), "Error:[UbTiling] mode2UbCapEle_ less than 1");
        return ge::GRAPH_FAILED;
    }
    mode2UbIterNum_ = Ops::Base::CeilAlign(HxW_, static_cast<int64_t>(mode2UbCapEle_)) / mode2UbCapEle_;
    mode2UbTailNum_ = (HxW_ - mode2UbIterNum_ * mode2UbCapEle_ == 0) ? mode2UbCapEle_ :
                                                                       (HxW_ - (mode2UbIterNum_ - 1) * mode2UbCapEle_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupNormGradRegBaseTiling::Stage2Mode2Tiling()
{
    cNumMode2PerCore_ =
        Ops::Base::CeilAlign(Ops::Base::CeilDiv(C_, static_cast<int64_t>(coreNum_)), static_cast<int64_t>(vectorLen_));
    cNumMode2PerCore_ = std::min(C_, static_cast<int64_t>(cNumMode2PerCore_));
    stage2CoreUsed_ = Ops::Base::CeilDiv(C_, static_cast<int64_t>(cNumMode2PerCore_));
    tailcNumMode2PerCore_ = (stage2CoreUsed_ == 1) ? C_ : C_ - (stage2CoreUsed_ - 1) * cNumMode2PerCore_;
    // 至少需要2块内存，开doubleBuffer
    uint32_t ubCanUseSize = (ubSize_ - FLOAT16_DTYPE_BYTES * UB_COPIES_4) / DOUBLE_BUFFER;
    uint32_t elemSize = (uTypeStr_ == ge::DT_FLOAT) ? FLOAT_DTYPE_BYTES : FLOAT16_DTYPE_BYTES;
    uint32_t intputUbAlignSize = Ops::Base::CeilAlign(cNumMode2PerCore_ * FLOAT_DTYPE_BYTES, blockSize_);
    uint32_t outputUbSize = cNumMode2PerCore_ * elemSize;
    uint32_t outputUbAlignSize = Ops::Base::CeilAlign(outputUbSize, blockSize_);
    uint32_t ubAlignSum = outputUbAlignSize + intputUbAlignSize;
    if (ubCanUseSize > ubAlignSum) { // C方向不切分
        stage2LoopCnt_ = 1;
        cFactorAlign_ = intputUbAlignSize / FLOAT_DTYPE_BYTES; // C在UB中能全载
        // 重新计算UB，这里先减去输出UB，剩余的全部给输入
        ubCanUseSize = ubCanUseSize - Ops::Base::CeilAlign(outputUbAlignSize, blockSize_);
        stage2FactorN_ = ubCanUseSize / Ops::Base::CeilAlign(intputUbAlignSize, blockSize_);
    } else { // C方向切分， N方向一次只能载入一行
        stage2LoopCnt_ = ubAlignSum / Ops::Base::CeilAlign(ubCanUseSize, blockSize_);
        cFactorAlign_ = cNumMode2PerCore_ / stage2LoopCnt_; // UB一次读入多少个C
        stage2FactorN_ = 1;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupNormGradRegBaseTiling::Stage2Tiling()
{
    uint32_t elemSize = FLOAT_DTYPE_BYTES;
    if (uTypeStr_ == ge::DT_FLOAT16 || uTypeStr_ == ge::DT_BF16) {
        elemSize = FLOAT16_DTYPE_BYTES;
    }

    if (this->modeKey_ == MODE_3) {
        reduceNCnt_ = N_;
    } else {
        reduceNCnt_ = Ops::Base::CeilDiv(N_, static_cast<int64_t>(SPLIT_COUNT));
    }
    size_t ubCanUseSize = (((ubSize_ / SPLIT_COUNT / DOUBLE_BUFFER) / this->blockSize_) * this->blockSize_);

    auto ubSizePerC = (reduceNCnt_ + 1) * FLOAT_DTYPE_BYTES + FLOAT_DTYPE_BYTES + elemSize;
    if (reduceNCnt_ > SCALE_COEF_EIGHT) {
        ubSizePerC = ubSizePerC + Ops::Base::CeilDiv(reduceNCnt_, SCALE_COEF_EIGHT) * FLOAT_DTYPE_BYTES;
    }
    cFactor_ = ubCanUseSize / ubSizePerC;
    cFactorAlign_ = ((cFactor_ * elemSize / this->blockSize_) * this->blockSize_) / elemSize;
    stage2Mode_ = (cFactorAlign_ >= 1) ? STAGE2_MODE_1 : STAGE2_MODE_2;
    if (stage2Mode_ == STAGE2_MODE_1) {
        uint32_t theLeastAPerCore = this->blockSize_ / elemSize;
        cBlockFactor_ = (C_ + coreNum_ - 1) / coreNum_;
        cBlockFactor_ = (cBlockFactor_ < theLeastAPerCore) ? theLeastAPerCore : (C_ + coreNum_ - 1) / coreNum_;
        stage2CoreUsed_ = (C_ + cBlockFactor_ - 1) / cBlockFactor_;
        cTailBlockFactor_ = C_ - cBlockFactor_ * (stage2CoreUsed_ - 1);
        if (reduceNCnt_ <= RA_BINARY_ADD_THRESHOLD) {
            return ge::GRAPH_SUCCESS;
        }
        stage2BinaryAddQuotient_ = RA_BINARY_ADD_THRESHOLD;
        while (stage2BinaryAddQuotient_ < reduceNCnt_) {
            stage2BinaryAddQuotient_ *= BINARY_ADD_COEF;
        }
        stage2BinaryAddQuotient_ /= BINARY_ADD_COEF;
        uint32_t binaryAddNum = stage2BinaryAddQuotient_ / RA_BINARY_ADD_THRESHOLD;
        stage2BinaryAddK_ = 0;
        uint32_t curBinaryAddNum = 1;
        while (curBinaryAddNum < binaryAddNum) {
            stage2BinaryAddK_++;
            curBinaryAddNum *= BINARY_ADD_COEF;
        }
    } else {
        Stage2Mode2Tiling();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupNormGradRegBaseTiling::DoOpTiling()
{
    OP_TILING_CHECK(
        BlockTiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "fail to calculate block Tiling"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        UbTiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "fail to calculate UB Tiling"),
        return ge::GRAPH_FAILED);
    if (N_ > 1 || this->modeKey_ == MODE_3) {
        OP_TILING_CHECK(
            Stage2Tiling() != ge::GRAPH_SUCCESS,
            OP_LOGE(context_->GetNodeName(), "fail to calculate regbase Stage2 Tiling"), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupNormGradRegBaseTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupNormGradRegBaseTiling::GetTilingKey() const
{
    uint64_t tilingKey = this->modeKey_;
    if (tTypeStr_ == ge::DataType::DT_FLOAT) {
        tilingKey += static_cast<uint64_t>(GNGDtypeKey::FLOAT_FLOAT);
    } else if (tTypeStr_ == ge::DataType::DT_FLOAT16) {
        if (uTypeStr_ == ge::DataType::DT_FLOAT16) {
            tilingKey += static_cast<uint64_t>(GNGDtypeKey::HALF_HALF);
        } else if (uTypeStr_ == ge::DataType::DT_FLOAT) {
            tilingKey += static_cast<uint64_t>(GNGDtypeKey::HALF_FLOAT);
        }
    } else if (tTypeStr_ == ge::DataType::DT_BF16) {
        if (uTypeStr_ == ge::DataType::DT_BF16) {
            tilingKey += static_cast<uint64_t>(GNGDtypeKey::BF16_BF16);
        } else if (uTypeStr_ == ge::DataType::DT_FLOAT) {
            tilingKey += static_cast<uint64_t>(GNGDtypeKey::BF16_FLOAT);
        }
    }
    OP_LOGD(opName, "tilingKey = %lld.", tilingKey);
    return tilingKey;
}

ge::graphStatus GroupNormGradRegBaseTiling::GetWorkspaceSize()
{
    if (this->modeKey_ == MODE_3) {
        workSpaceSize_ = N_ * C_;
    } else {
        workSpaceSize_ = Ops::Base::CeilDiv(N_, static_cast<int64_t>(SPLIT_COUNT)) * C_;
    }
    auto workspaces = context_->GetWorkspaceSizes(1);
    auto usrWorkspaceSize = WORKSPACE_COPIES * workSpaceSize_ * FLOAT_DTYPE_BYTES;
    workspaces[0] = sysWorkspaceSize_ + usrWorkspaceSize;

    auto alignFactor = this->blockSize_ / FLOAT_DTYPE_BYTES;
    auto clrBlockFactor = Ops::Base::CeilDiv(workSpaceSize_, static_cast<int64_t>(coreNum_));
    auto clrBlockAlignFactor = Ops::Base::CeilDiv(clrBlockFactor, static_cast<int64_t>(alignFactor)) * alignFactor;
    clrBlockSize_ = std::max(static_cast<int64_t>(clrBlockAlignFactor), static_cast<int64_t>(MIN_BLOCK_SIZE));
    clrBlockNum_ = Ops::Base::CeilDiv(workSpaceSize_, clrBlockSize_);
    clrTailBlockSize_ = workSpaceSize_ - clrBlockSize_ * (clrBlockNum_ - 1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupNormGradRegBaseTiling::PostTiling()
{
    SetTilingData();
    PrintTilingData();
    if (this->modeKey_ == MODE_3) {
        context_->SetBlockDim(std::max(
            this->stage0CoreUsed_,
            std::max(std::max(this->stage1CoreUsed_, this->clrBlockNum_), this->stage2CoreUsed_)));
    } else {
        context_->SetBlockDim(std::max(std::max(this->stage1CoreUsed_, this->clrBlockNum_), this->stage2CoreUsed_));
    }
    context_->SetScheduleMode(1);
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void GroupNormGradRegBaseTiling::SetTilingData()
{
    OP_LOGD(opName, "Tiling start.");
    SetBaseTilingData();
    SetReduceTilingData();
    SetKernel2TilingData();
}

void GroupNormGradRegBaseTiling::SetBaseTilingData()
{
    OP_LOGD(opName, "Tiling start.");
    tilingData.gNGBaseParams.set_N(N_);                                               // 1
    tilingData.gNGBaseParams.set_C(C_);                                               // 2
    tilingData.gNGBaseParams.set_HXW(HxW_);                                           // 3
    tilingData.gNGBaseParams.set_G(G_);                                               // 4
    tilingData.gNGBaseParams.set_clrBlockSize(clrBlockSize_);                         // 5
    tilingData.gNGBaseParams.set_clrTailBlockSize(clrTailBlockSize_);                 // 6
    tilingData.gNGBaseParams.set_clrBlockNum(clrBlockNum_);                           // 7
    tilingData.gNGBaseParams.set_taskNumPerCore(taskNumPerCore_);                     // 8
    tilingData.gNGBaseParams.set_taskNumPerTailCore(taskNumPerTailCore_);             // 9
    tilingData.gNGBaseParams.set_tailCore(tailCore_);                                 // 10
    tilingData.gNGBaseParams.set_stage1CoreUsed(stage1CoreUsed_);                     // 11
    tilingData.gNGBaseParams.set_mode0UbCapGNum(mode0UbCapGNum_);                     // 12
    tilingData.gNGBaseParams.set_mode1UbCapCNum(mode1UbCapCNum_);                     // 13
    tilingData.gNGBaseParams.set_mode2UbCapEle(mode2UbCapEle_);                       // 14
    tilingData.gNGBaseParams.set_mode2UbIterNum(mode2UbIterNum_);                     // 15
    tilingData.gNGBaseParams.set_mode2UbTailNum(mode2UbTailNum_);                     // 16
    tilingData.gNGBaseParams.set_mode2MainLoopCnt(mode2MainLoopCnt_);                 // 17
    tilingData.gNGBaseParams.set_mode2FoldLoopCnt(mode2FoldLoopCnt_);                 // 18
    tilingData.gNGBaseParams.set_mode2OneLoopSize(mode2OneLoopSize_);                 // 19
    tilingData.gNGBaseParams.set_mode2FoldTail(mode2FoldTail_);                       // 20
    tilingData.gNGBaseParams.set_workSpaceSize(workSpaceSize_);                       // 21
    tilingData.gNGBaseParams.set_dxIsRequire(dxIsRequire_);                           // 22
    tilingData.gNGBaseParams.set_dgammaIsRequire(dgammaIsRequire_);                   // 23
    tilingData.gNGBaseParams.set_dbetaIsRequire(dbetaIsRequire_);                     // 24
    tilingData.gNGBaseParams.set_stage0TaskNumPerCore(stage0TaskNumPerCore_);         // 25
    tilingData.gNGBaseParams.set_stage0TaskNumPerTailCore(stage0TaskNumPerTailCore_); // 26
    tilingData.gNGBaseParams.set_stage0TailCore(stage0TailCore_);                     // 27
    tilingData.gNGBaseParams.set_stage0CoreUsed(stage0CoreUsed_);                     // 28
}

void GroupNormGradRegBaseTiling::SetReduceTilingData()
{
    tilingData.gNGReduceParams.set_binaryAddQuotient(binaryAddQuotient_); // 1
    tilingData.gNGReduceParams.set_binaryAddK(binaryAddK_);               // 2
    tilingData.gNGReduceParams.set_binaryAddLastNum(binaryAddLastNum_);   // 3
    tilingData.gNGReduceParams.set_binaryCGQuotient(binaryCGQuotient_);   // 4
    tilingData.gNGReduceParams.set_binaryCGK(binaryCGK_);                 // 5
    tilingData.gNGReduceParams.set_binaryCGLastNum(binaryCGLastNum_);     // 6
}

void GroupNormGradRegBaseTiling::SetKernel2TilingData()
{
    tilingData.gNGKernel2Params.set_stage2Mode(stage2Mode_);
    tilingData.gNGKernel2Params.set_cFactor(cFactorAlign_);
    tilingData.gNGKernel2Params.set_cBlockFactor(cBlockFactor_);
    tilingData.gNGKernel2Params.set_cTailBlockFactor(cTailBlockFactor_);
    tilingData.gNGKernel2Params.set_stage2CoreUsed(stage2CoreUsed_);
    tilingData.gNGKernel2Params.set_stage2BinaryAddQuotient(stage2BinaryAddQuotient_);
    tilingData.gNGKernel2Params.set_stage2BinaryAddK(stage2BinaryAddK_);
    tilingData.gNGKernel2Params.set_reduceNCnt(reduceNCnt_);
    tilingData.gNGKernel2Params.set_cNumMode2PerCore(cNumMode2PerCore_);
    tilingData.gNGKernel2Params.set_tailcNumMode2PerCore(tailcNumMode2PerCore_);
    tilingData.gNGKernel2Params.set_stage2LoopCnt(stage2LoopCnt_);
    tilingData.gNGKernel2Params.set_stage2FactorN(stage2FactorN_);
}

void GroupNormGradRegBaseTiling::PrintTilingData() const
{
    OP_LOGD(opName, "N:                       %ld.", N_);
    OP_LOGD(opName, "C:                       %ld.", C_);
    OP_LOGD(opName, "HXW:                     %ld.", HxW_);
    OP_LOGD(opName, "G:                       %ld.", G_);
    OP_LOGD(opName, "NXG:                     %ld.", NxG_);
    OP_LOGD(opName, "CPerG:                   %ld.", CPerG_);
    OP_LOGD(opName, "clrBlockSize_:           %ld.", clrBlockSize_);
    OP_LOGD(opName, "clrTailBlockSize_:       %ld.", clrTailBlockSize_);
    OP_LOGD(opName, "clrBlockNum_:            %d.", clrBlockNum_);
    OP_LOGD(opName, "taskNumPerCore:          %d.", taskNumPerCore_);
    OP_LOGD(opName, "taskNumPerTailCore:      %d.", taskNumPerTailCore_);
    OP_LOGD(opName, "tailCore:                %d.", tailCore_);
    OP_LOGD(opName, "stage1CoreUsed:          %d.", stage1CoreUsed_);
    OP_LOGD(opName, "stage0TaskNumPerCore_:   %d.", stage0TaskNumPerCore_);
    OP_LOGD(opName, "stage0TaskNumPerTailCore_:  %d.", stage0TaskNumPerTailCore_);
    OP_LOGD(opName, "stage0TailCore_:         %d.", stage0TailCore_);
    OP_LOGD(opName, "stage0CoreUsed:          %d.", stage0CoreUsed_);
    OP_LOGD(opName, "mode0UbCapGNum:          %d.", mode0UbCapGNum_);
    OP_LOGD(opName, "mode1UbCapCNum:          %d.", mode1UbCapCNum_);
    OP_LOGD(opName, "mode2UbCapEle:           %d.", mode2UbCapEle_);
    OP_LOGD(opName, "mode2UbIterNum:          %d.", mode2UbIterNum_);
    OP_LOGD(opName, "mode2UbTailNum:          %d.", mode2UbTailNum_);
    OP_LOGD(opName, "mode2MainLoopCnt:        %d.", mode2MainLoopCnt_);
    OP_LOGD(opName, "mode2FoldLoopCnt:        %d.", mode2FoldLoopCnt_);
    OP_LOGD(opName, "mode2OneLoopSize:        %d.", mode2OneLoopSize_);
    OP_LOGD(opName, "mode2FoldTail:           %d.", mode2FoldTail_);
    OP_LOGD(opName, "binaryAddQuotient:       %ld.", binaryAddQuotient_);
    OP_LOGD(opName, "binaryAddK:              %d.", binaryAddK_);
    OP_LOGD(opName, "binaryAddLastNum:        %d.", binaryAddLastNum_);
    OP_LOGD(opName, "workSpaceSize:           %ld.", workSpaceSize_);
    OP_LOGD(opName, "cFactor:                 %d.", cFactorAlign_);
    OP_LOGD(opName, "cBlockFactor:            %d.", cBlockFactor_);
    OP_LOGD(opName, "cTailBlockFactor:        %d.", cTailBlockFactor_);
    OP_LOGD(opName, "stage2CoreUsed:          %d.", stage2CoreUsed_);
    OP_LOGD(opName, "stage2BinaryAddQuotient: %d.", stage2BinaryAddQuotient_);
    OP_LOGD(opName, "stage2BinaryAddK:        %d.", stage2BinaryAddK_);
    OP_LOGD(opName, "reduceNCnt:              %ld.", reduceNCnt_);
    OP_LOGD(opName, "stage2Mode:              %d.", stage2Mode_);
    OP_LOGD(opName, "cNumMode2PerCore:        %d.", cNumMode2PerCore_);
    OP_LOGD(opName, "tailcNumMode2PerCore:    %d.", tailcNumMode2PerCore_);
    OP_LOGD(opName, "stage2LoopCnt:           %ld.", stage2LoopCnt_);
    OP_LOGD(opName, "stage2FactorN:           %ld.", stage2FactorN_);
    OP_LOGD(opName, "dxIsRequire:             %d.", dxIsRequire_);
    OP_LOGD(opName, "dgammaIsRequire:         %d.", dgammaIsRequire_);
    OP_LOGD(opName, "dbetaIsRequire:          %d.", dbetaIsRequire_);
    OP_LOGD(opName, "modeKey_:                %ld.", this->modeKey_);
}
} // namespace optiling
