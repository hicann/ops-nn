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
#include "op_host/tiling_templates_registry.h"

namespace optiling {
static constexpr uint16_t INPUT_IDX_DY = 0;
static constexpr uint16_t DX_IS_REQUIRE_IDX = 2;
static constexpr uint16_t DGAMMA_IS_REQUIRE_IDX = 3;
static constexpr uint16_t DBETA_IS_REQUIRE_IDX = 4;
static constexpr uint16_t DIM0 = 0;
static constexpr uint16_t DIM1 = 1;
static constexpr uint32_t FLOAT_DTYPE_BYTES = 4;
static constexpr uint16_t BFLOAT16_DTYPE_BYTES = 2;
static constexpr uint16_t FLOAT16_DTYPE_BYTES = 2;
static constexpr int64_t RA_BINARY_ADD_THRESHOLD = 8;
static constexpr uint16_t BINARY_ADD_COEF = 2;
static constexpr int64_t SCALE_COEF_EIGHT = 8;
static constexpr uint32_t MIN_BLOCK_SIZE = 512;
static constexpr uint32_t SMALL_NG = 16;
static constexpr uint32_t STAGE0_C_PER_CORE = 2;
// the threshold for n*(d/c_hat)
static constexpr uint32_t SMALL_NG_ND_CHAT = 32;

static constexpr int16_t CONST_ZERO = 0;
static constexpr int16_t CONST_ONE = 1;
static constexpr int16_t CONST_TWO = 2;
static constexpr int16_t CONST_THREE = 3;
static constexpr int16_t CONST_FOUR = 4;
static constexpr int16_t CONST_SIXTY_THREE = 63;
static constexpr int16_t CONST_SIXTY_FOUR = 64;

ge::graphStatus GroupNormGradRegBaseMixTiling::GetShapeAttrsInfo()
{
    auto dyDesc = context_->GetInputDesc(INPUT_IDX_DY);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dyDesc);
    commonParams.tTypeStr = dyDesc->GetDataType();
    auto dyShape = context_->GetInputShape(INPUT_IDX_DY)->GetStorageShape();
    OP_TILING_CHECK(InputCheck(dyShape) == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "Input check failed."),
                    return ge::GRAPH_FAILED);

    auto dimNum = dyShape.GetDimNum();
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    // Get ATTR
    const bool* dxRequire = attrs->GetAttrPointer<bool>(DX_IS_REQUIRE_IDX);
    commonParams.dxIsRequire = (dxRequire == nullptr) ? true : *dxRequire;
    const bool* dgammaRequire = attrs->GetAttrPointer<bool>(DGAMMA_IS_REQUIRE_IDX);
    commonParams.dgammaIsRequire = (dgammaRequire == nullptr) ? true : *dgammaRequire;
    const bool* dbetaRequire = attrs->GetAttrPointer<bool>(DBETA_IS_REQUIRE_IDX);
    commonParams.dbetaIsRequire = (dbetaRequire == nullptr) ? true : *dbetaRequire;
    const int64_t* gValue = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gValue);
    OP_TILING_CHECK((*gValue <= 0),
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "num_groups", std::to_string(*gValue).c_str(),
                                              "bigger than 0"),
                    return ge::GRAPH_FAILED);
    commonParams.G = static_cast<int64_t>(*gValue);
    commonParams.N = dyShape.GetDim(DIM0);
    commonParams.C = dyShape.GetDim(DIM1);
    OP_TILING_CHECK(
        (this->commonParams.N <= 0 || this->commonParams.C <= 0),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "dy", Ops::Base::ToString(dyShape).c_str(),
            "The N-dimension and C-dimension of dy must be greater than 0, where N-dimension is the first dim and "
            "C-dimension is the second dim"),
        return ge::GRAPH_FAILED);
    NxG_ = commonParams.N * commonParams.G;
    NxC_ = commonParams.N * commonParams.C;
    commonParams.CPerG = commonParams.C / commonParams.G;
    for (uint32_t dimIdx = 2; dimIdx < dimNum; dimIdx++) {
        commonParams.HxW *= dyShape.GetDim(dimIdx);
    }
    OP_TILING_CHECK(ParamsCheck() == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "Params check failed."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool GroupNormGradRegBaseMixTiling::IsCapable()
{
    CalBinaryParams(this->commonParams.HxW, binaryAddQuotient_, binaryAddK_, binaryAddLastNum_);
    CalBinaryParams(this->commonParams.CPerG, binaryCGQuotient_, binaryCGK_, binaryCGLastNum_);
    return true;
}

void GroupNormGradRegBaseMixTiling::CalBinaryParams(int64_t reduceNum, int64_t& binaryQuotient, uint32_t& binaryK,
                                                    uint32_t& binaryLastNum)
{
    binaryQuotient = commonParams.vectorLen;
    while (binaryQuotient < reduceNum) {
        binaryQuotient *= BINARY_ADD_COEF;
    }
    binaryQuotient /= BINARY_ADD_COEF;                                  // 二分折叠之后的长度
    int64_t quotientVcaddNum = binaryQuotient / commonParams.vectorLen; // 预留空间大小，存放二分折叠临时变量
    if (quotientVcaddNum <= commonParams.vectorLen) {
        binaryLastNum = static_cast<uint32_t>(quotientVcaddNum);
        binaryK = 0;
    } else {
        auto binaryAddNum = quotientVcaddNum / commonParams.vectorLen;
        binaryK = __builtin_ctzl(binaryAddNum);
        binaryLastNum = commonParams.vectorLen;
    }
}

uint32_t GroupNormGradRegBaseMixTiling::GetTypeSize(ge::DataType dtypeStr) const
{
    switch (dtypeStr) {
        case ge::DT_FLOAT:
            return FLOAT_DTYPE_BYTES;
        case ge::DT_BF16:
            return BFLOAT16_DTYPE_BYTES;
        case ge::DT_FLOAT16:
            return FLOAT16_DTYPE_BYTES;
        default:
            OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x",
                                      ge::TypeUtils::DataTypeToSerialString(dtypeStr).c_str(),
                                      "float32, float16 or bfloat16");
            return 0;
    }
}

ge::graphStatus GroupNormGradRegBaseMixTiling::BlockTiling()
{
    this->tTypeBytes_ = GetTypeSize(this->commonParams.tTypeStr);
    OP_TILING_CHECK(this->tTypeBytes_ == 0, OP_LOGE(context_->GetNodeName(), "Calculated tTypeBytes_ is zero"),
                    return ge::GRAPH_FAILED);

    int64_t cGFloatAlign = Ops::Base::CeilAlign(commonParams.CPerG,
                                                static_cast<int64_t>(this->commonParams.blockSize / FLOAT_DTYPE_BYTES));
    int64_t cGHalfAlign = Ops::Base::CeilAlign(
        commonParams.CPerG, static_cast<int64_t>(this->commonParams.blockSize / FLOAT16_DTYPE_BYTES));
    int64_t cGtempSpace = 0;
    // CGNum(4+2): gamma, dbeta, dgamma, ds, input_db, input_ds
    int64_t cgAllocSpace = 0;
    if (NxG_ <= SMALL_NG) {
        cgAllocSpace = cGFloatAlign * FLOAT_DTYPE_BYTES * (UB_COPIES_4 + UB_COPIES_2);
        if (commonParams.uTypeStr == ge::DT_FLOAT16 || commonParams.uTypeStr == ge::DT_BF16) {
            cGtempSpace = cGHalfAlign * FLOAT16_DTYPE_BYTES; // tBufGamma
        }
    } else {
        cgAllocSpace = cGFloatAlign * FLOAT_DTYPE_BYTES * UB_COPIES_4;
        if (commonParams.uTypeStr == ge::DT_FLOAT16 || commonParams.uTypeStr == ge::DT_BF16) {
            cGtempSpace = cGHalfAlign * UB_COPIES_2 * FLOAT16_DTYPE_BYTES; // tBufGamma, tBufDbeta
        }
    }

    reserveSpace_ = cgAllocSpace + cGtempSpace +
                    UB_COPIES_2 * this->commonParams.blockSize; // 2 block for Mean, rstd buf

    // Allocate computing core
    if (NxG_ <= SMALL_NG) {
        stage0TaskNumPerCore_ = Ops::Base::CeilDiv(NxC_, static_cast<int64_t>(commonParams.coreNum));
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

    taskNumPerCore_ = Ops::Base::CeilAlign(NxG_, static_cast<int64_t>(commonParams.coreNum)) / commonParams.coreNum;
    stage1CoreUsed_ = (NxG_ - 1) / taskNumPerCore_ + 1;
    taskNumPerTailCore_ = taskNumPerCore_;
    tailCore_ = stage1CoreUsed_;

    if (NxG_ % stage1CoreUsed_ != 0) {
        taskNumPerTailCore_ = taskNumPerCore_ - 1;
        tailCore_ = NxG_ % stage1CoreUsed_;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupNormGradRegBaseMixTiling::UbTiling()
{
    OP_TILING_CHECK(commonParams.ubSizePlatForm < (reserveSpace_),
                    OP_LOGE(context_->GetNodeName(), "Error:[UbTiling] ubSize less than reserveSpace"),
                    return ge::GRAPH_FAILED);
    auto canUseUbSize = (commonParams.ubSizePlatForm - reserveSpace_) / DOUBLE_BUFFER;
    // mode0 x, dy 全部load成float32数据类型，放在UB中计算完dx 和 dgamma, dbeta
    int64_t tAlignFactor = this->commonParams.blockSize / this->tTypeBytes_;
    int64_t mode0xDyDxSize = static_cast<int64_t>(
                                 Ops::Base::CeilAlign(commonParams.CPerG * commonParams.HxW, tAlignFactor)) *
                             this->tTypeBytes_ * UB_COPIES_3;
    // mode0 mean, rstd需要额外的空间
    mode0UbCapGNum_ = canUseUbSize / (mode0xDyDxSize + this->commonParams.blockSize * UB_COPIES_2);
    mode1UbCapCNum_ = canUseUbSize / (static_cast<int64_t>(Ops::Base::CeilAlign(commonParams.HxW, tAlignFactor)) *
                                      this->tTypeBytes_ * UB_COPIES_3);
    // the conditions for branch small_ng
    if (mode0UbCapGNum_ > 0) {
        modeKey_ = MODE_0;
    } else if (mode1UbCapCNum_ > 0) {
        bool smallNGCase = (NxG_ <= SMALL_NG) && (stage0TaskNumPerCore_ >= STAGE0_C_PER_CORE) &&
                           ((commonParams.N * (commonParams.CPerG / mode1UbCapCNum_)) >= SMALL_NG_ND_CHAT);
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

ge::graphStatus GroupNormGradRegBaseMixTiling::Mode2UbTiling()
{
    // cache level is 3
    auto reserveCacheSize = CACHE_BUFF_SIZE * 3 * UB_COPIES_2 * sizeof(float);
    OP_TILING_CHECK(commonParams.ubSizePlatForm < (reserveSpace_ + reserveCacheSize),
                    OP_LOGE(context_->GetNodeName(), "Error:[UbTiling] ubSize less than reserveSpace"),
                    return ge::GRAPH_FAILED);
    uint32_t canUseUbSize = (commonParams.ubSizePlatForm - reserveSpace_ - reserveCacheSize) / DOUBLE_BUFFER;
    mode2UbCapEle_ = Ops::Base::FloorAlign(canUseUbSize / (this->tTypeBytes_ * UB_COPIES_3),
                                           this->commonParams.blockSize);
    // Loop Size vector Len align
    mode2OneLoopSize_ = this->commonParams.vectorLen;
    while (mode2OneLoopSize_ < (mode2UbCapEle_ / BINARY_ADD_COEF)) { // loop need split main/fold part
        mode2OneLoopSize_ *= BINARY_ADD_COEF;
    }
    mode2OneLoopSize_ /= BINARY_ADD_COEF;
    mode2MainLoopCnt_ = Ops::Base::CeilDiv(this->binaryAddQuotient_, static_cast<int64_t>(this->mode2OneLoopSize_));
    mode2FoldLoopCnt_ = Ops::Base::CeilDiv((this->commonParams.HxW - this->binaryAddQuotient_),
                                           static_cast<int64_t>(this->mode2OneLoopSize_));
    OP_TILING_CHECK(
        this->binaryAddQuotient_ % this->mode2OneLoopSize_ != 0,
        OP_LOGE(context_->GetNodeName(),
                "Error:[UbTiling] the main block can't have tail. binaryAddQuotient = %ld, mode2OneLoopSize = %u.",
                this->binaryAddQuotient_, this->mode2OneLoopSize_),
        return ge::GRAPH_FAILED);
    mode2FoldTail_ = (this->commonParams.HxW - this->binaryAddQuotient_) % this->mode2OneLoopSize_ == 0 ?
                         this->mode2OneLoopSize_ :
                         (this->commonParams.HxW - this->binaryAddQuotient_) % this->mode2OneLoopSize_;
    OP_TILING_CHECK(
        this->mode2MainLoopCnt_ < 1 || this->mode2FoldLoopCnt_ < 1,
        OP_LOGE(context_->GetNodeName(), "Error:[UbTiling] mode2MainLoopCnt or mode2FoldLoopCnt less than 1"),
        return ge::GRAPH_FAILED);
    mode2UbCapEle_ = mode2OneLoopSize_ * BINARY_ADD_COEF;
    if (mode2UbCapEle_ == 0) {
        OP_LOGE(context_->GetNodeName(), "Error:[UbTiling] mode2UbCapEle_ less than 1");
        return ge::GRAPH_FAILED;
    }
    mode2UbIterNum_ = Ops::Base::CeilAlign(commonParams.HxW, static_cast<int64_t>(mode2UbCapEle_)) / mode2UbCapEle_;
    mode2UbTailNum_ = (commonParams.HxW - mode2UbIterNum_ * mode2UbCapEle_ == 0) ?
                          mode2UbCapEle_ :
                          (commonParams.HxW - (mode2UbIterNum_ - 1) * mode2UbCapEle_);
    return ge::GRAPH_SUCCESS;
}

int64_t GroupNormGradRegBaseMixTiling::FindNearestPower2(const int64_t value)
{
    if (value <= CONST_ONE) {
        return CONST_ZERO;
    } else if (value <= CONST_TWO) {
        return CONST_ONE;
    } else if (value <= CONST_FOUR) {
        return CONST_TWO;
    } else {
        const int64_t num = value - CONST_ONE;
        const int64_t pow = CONST_SIXTY_THREE - __builtin_clzl(num);
        return (CONST_ONE << pow);
    }
}

int64_t GroupNormGradRegBaseMixTiling::GetCacheID(const int64_t idx)
{
    return __builtin_popcountll(idx ^ (idx + CONST_ONE)) - CONST_ONE;
}

ge::graphStatus GroupNormGradRegBaseMixTiling::Stage2Mode2Tiling()
{
    constexpr int16_t ALIGN_FACTOR_BASE_BYTE_LEN = 128;
    constexpr int16_t TILE_NUM_BASE_LEN_B32 = 512;
    const uint32_t dTypeSize = FLOAT_DTYPE_BYTES;
    int16_t blockLenFp32 = this->commonParams.blockSize / dTypeSize;
    int64_t cBlockFactorBase = ALIGN_FACTOR_BASE_BYTE_LEN / dTypeSize;
    int64_t cBlockFactorAlignBase = ALIGN_FACTOR_BASE_BYTE_LEN / dTypeSize;
    if (commonParams.C <= blockLenFp32) {
        cBlockFactorBase = blockLenFp32;
        cBlockFactorAlignBase = blockLenFp32;
    }
    int64_t cBlockFactor = Ops::Base::CeilDiv(commonParams.C, static_cast<int64_t>(commonParams.coreNum));
    cBlockFactor = Ops::Base::CeilDiv(cBlockFactor, cBlockFactorAlignBase) * cBlockFactorAlignBase;
    cBlockFactor = std::max(cBlockFactorBase, cBlockFactor);
    stage2CoreUsed_ = Ops::Base::CeilDiv(commonParams.C, cBlockFactor);
    int64_t cTileBlockFactor = commonParams.C - (stage2CoreUsed_ - 1) * cBlockFactor;
    int64_t cTileNumBase = TILE_NUM_BASE_LEN_B32 / dTypeSize;
    if (cTileNumBase > cBlockFactor) {
        cTileNumBase = cBlockFactor;
    }

    std::vector<int64_t> nTileNumList = {128, 256, 512, 1024, 2048, 4096};
    for (const int64_t& nTileNumPossible : nTileNumList) {
        int64_t nTileNum = nTileNumPossible;
        int64_t nLoop = reduceNCnt_ / nTileNum;
        int64_t nTotalLoop = Ops::Base::CeilDiv(reduceNCnt_, nTileNum);
        int64_t nTail = reduceNCnt_ - nLoop * nTileNum;
        int64_t basicBlockLoop = FindNearestPower2(nTotalLoop);
        if (basicBlockLoop == 0) {
            break;
        }
        int64_t mainFoldCount = nLoop - basicBlockLoop;
        int64_t cacheBufferCount = CONST_ONE;
        int32_t resultCacheID = CONST_ZERO;
        if (basicBlockLoop != 0) {
            cacheBufferCount = CONST_SIXTY_FOUR - __builtin_clzl(basicBlockLoop);
            resultCacheID = GetCacheID(basicBlockLoop - 1);
        }

        int64_t cTileNum = 0;
        cTileNum = (commonParams.ubSizePlatForm - FLOAT16_DTYPE_BYTES * UB_COPIES_4) /
                   (CONST_THREE * nTileNum + CONST_TWO + cacheBufferCount) / sizeof(float);
        cTileNum = cTileNum / cTileNumBase * cTileNumBase;

        if ((cTileNum < cBlockFactor && nTileNum != nTileNumList[0]) || cTileNum <= 0) {
            break;
        }

        int64_t cLoopMain = Ops::Base::CeilDiv(cBlockFactor, cTileNum);
        int64_t cTailMain = cBlockFactor - (cLoopMain - 1) * cTileNum;
        int64_t cLoopTail = Ops::Base::CeilDiv(cTileBlockFactor, cTileNum);
        int64_t cTailTail = cTileBlockFactor - (cLoopTail - 1) * cTileNum;

        cTileNum_ = cTileNum;
        nTileNum_ = nTileNum;
        basicBlockLoop_ = basicBlockLoop;
        mainFoldCount_ = mainFoldCount;
        nTail_ = nTail;
        cacheBufferCount_ = cacheBufferCount;
        resultCacheID_ = resultCacheID;
        cLoopMain_ = cLoopMain;
        cTailMain_ = cTailMain;
        cLoopTail_ = cLoopTail;
        cTailTail_ = cTailTail;
    }

    OP_TILING_CHECK((cTileNum_ <= 0), VECTOR_INNER_ERR_REPORT_TILIING(opName, "The axis of C can not be Tiled"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupNormGradRegBaseMixTiling::Stage2Tiling()
{
    uint32_t elemSize = FLOAT_DTYPE_BYTES;
    if (commonParams.uTypeStr == ge::DT_FLOAT16 || commonParams.uTypeStr == ge::DT_BF16) {
        elemSize = FLOAT16_DTYPE_BYTES;
    }

    if (this->modeKey_ == MODE_3) {
        reduceNCnt_ = commonParams.N;
    } else {
        reduceNCnt_ = Ops::Base::CeilDiv(commonParams.N, static_cast<int64_t>(SPLIT_COUNT));
    }
    size_t ubCanUseSize = (((commonParams.ubSizePlatForm / SPLIT_COUNT / DOUBLE_BUFFER) /
                            this->commonParams.blockSize) *
                           this->commonParams.blockSize);

    auto ubSizePerC = (reduceNCnt_ + 1) * FLOAT_DTYPE_BYTES + FLOAT_DTYPE_BYTES + elemSize;
    if (reduceNCnt_ > SCALE_COEF_EIGHT) {
        ubSizePerC = ubSizePerC + Ops::Base::CeilDiv(reduceNCnt_, SCALE_COEF_EIGHT) * FLOAT_DTYPE_BYTES;
    }
    cFactor_ = ubCanUseSize / ubSizePerC;
    cFactorAlign_ = ((cFactor_ * elemSize / this->commonParams.blockSize) * this->commonParams.blockSize) / elemSize;
    stage2Mode_ = (cFactorAlign_ >= 1) ? STAGE2_MODE_1 : STAGE2_MODE_2;
    if (stage2Mode_ == STAGE2_MODE_1) {
        uint32_t theLeastAPerCore = this->commonParams.blockSize / elemSize;
        cBlockFactor_ = (commonParams.C + commonParams.coreNum - 1) / commonParams.coreNum;
        cBlockFactor_ = (cBlockFactor_ < theLeastAPerCore) ?
                            theLeastAPerCore :
                            (commonParams.C + commonParams.coreNum - 1) / commonParams.coreNum;
        stage2CoreUsed_ = (commonParams.C + cBlockFactor_ - 1) / cBlockFactor_;
        cTailBlockFactor_ = commonParams.C - cBlockFactor_ * (stage2CoreUsed_ - 1);
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

ge::graphStatus GroupNormGradRegBaseMixTiling::DoOpTiling()
{
    OP_TILING_CHECK(BlockTiling() != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "fail to calculate block Tiling"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(UbTiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "fail to calculate UB Tiling"),
                    return ge::GRAPH_FAILED);
    if (commonParams.N > 1 || this->modeKey_ == MODE_3) {
        OP_TILING_CHECK(Stage2Tiling() != ge::GRAPH_SUCCESS,
                        OP_LOGE(context_->GetNodeName(), "fail to calculate regbase Stage2 Tiling"),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupNormGradRegBaseMixTiling::GetTilingKey() const
{
    uint64_t tilingKey = this->modeKey_;
    if (commonParams.tTypeStr == ge::DataType::DT_FLOAT) {
        tilingKey += static_cast<uint64_t>(GNGDtypeKey::FLOAT_FLOAT);
    } else if (commonParams.tTypeStr == ge::DataType::DT_FLOAT16) {
        if (commonParams.uTypeStr == ge::DataType::DT_FLOAT16) {
            tilingKey += static_cast<uint64_t>(GNGDtypeKey::HALF_HALF);
        } else if (commonParams.uTypeStr == ge::DataType::DT_FLOAT) {
            tilingKey += static_cast<uint64_t>(GNGDtypeKey::HALF_FLOAT);
        }
    } else if (commonParams.tTypeStr == ge::DataType::DT_BF16) {
        if (commonParams.uTypeStr == ge::DataType::DT_BF16) {
            tilingKey += static_cast<uint64_t>(GNGDtypeKey::BF16_BF16);
        } else if (commonParams.uTypeStr == ge::DataType::DT_FLOAT) {
            tilingKey += static_cast<uint64_t>(GNGDtypeKey::BF16_FLOAT);
        }
    }
    OP_LOGD(opName, "tilingKey = %lld.", tilingKey);
    return tilingKey;
}

ge::graphStatus GroupNormGradRegBaseMixTiling::GetWorkspaceSize()
{
    if (this->modeKey_ == MODE_3) {
        workSpaceSize_ = commonParams.N * commonParams.C;
    } else {
        workSpaceSize_ = Ops::Base::CeilDiv(commonParams.N, static_cast<int64_t>(SPLIT_COUNT)) * commonParams.C;
    }
    auto workspaces = context_->GetWorkspaceSizes(1);
    auto usrWorkspaceSize = WORKSPACE_COPIES * workSpaceSize_ * FLOAT_DTYPE_BYTES;
    workspaces[0] = commonParams.sysWorkspaceSize + usrWorkspaceSize;

    auto alignFactor = this->commonParams.blockSize / FLOAT_DTYPE_BYTES;
    auto clrBlockFactor = Ops::Base::CeilDiv(workSpaceSize_, static_cast<int64_t>(commonParams.coreNum));
    auto clrBlockAlignFactor = Ops::Base::CeilDiv(clrBlockFactor, static_cast<int64_t>(alignFactor)) * alignFactor;
    clrBlockSize_ = std::max(static_cast<int64_t>(clrBlockAlignFactor), static_cast<int64_t>(MIN_BLOCK_SIZE));
    clrBlockNum_ = Ops::Base::CeilDiv(workSpaceSize_, clrBlockSize_);
    clrTailBlockSize_ = workSpaceSize_ - clrBlockSize_ * (clrBlockNum_ - 1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupNormGradRegBaseMixTiling::PostTiling()
{
    SetTilingData();
    PrintTilingData();
    if (this->modeKey_ == MODE_3) {
        context_->SetBlockDim(
            std::max(this->stage0CoreUsed_,
                     std::max(std::max(this->stage1CoreUsed_, this->clrBlockNum_), this->stage2CoreUsed_)));
    } else {
        context_->SetBlockDim(std::max(std::max(this->stage1CoreUsed_, this->clrBlockNum_), this->stage2CoreUsed_));
    }
    context_->SetScheduleMode(1);
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void GroupNormGradRegBaseMixTiling::SetTilingData()
{
    OP_LOGD(opName, "Tiling start.");
    SetBaseTilingData();
    SetReduceTilingData();
    SetKernel2TilingData();
}

void GroupNormGradRegBaseMixTiling::SetBaseTilingData()
{
    OP_LOGD(opName, "Tiling start.");
    tilingData.gNGBaseParams.set_N(commonParams.N);                                   // 1
    tilingData.gNGBaseParams.set_C(commonParams.C);                                   // 2
    tilingData.gNGBaseParams.set_HXW(commonParams.HxW);                               // 3
    tilingData.gNGBaseParams.set_G(commonParams.G);                                   // 4
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
    tilingData.gNGBaseParams.set_dxIsRequire(commonParams.dxIsRequire);               // 22
    tilingData.gNGBaseParams.set_dgammaIsRequire(commonParams.dgammaIsRequire);       // 23
    tilingData.gNGBaseParams.set_dbetaIsRequire(commonParams.dbetaIsRequire);         // 24
    tilingData.gNGBaseParams.set_stage0TaskNumPerCore(stage0TaskNumPerCore_);         // 25
    tilingData.gNGBaseParams.set_stage0TaskNumPerTailCore(stage0TaskNumPerTailCore_); // 26
    tilingData.gNGBaseParams.set_stage0TailCore(stage0TailCore_);                     // 27
    tilingData.gNGBaseParams.set_stage0CoreUsed(stage0CoreUsed_);                     // 28
}

void GroupNormGradRegBaseMixTiling::SetReduceTilingData()
{
    tilingData.gNGReduceParams.set_binaryAddQuotient(binaryAddQuotient_); // 1
    tilingData.gNGReduceParams.set_binaryAddK(binaryAddK_);               // 2
    tilingData.gNGReduceParams.set_binaryAddLastNum(binaryAddLastNum_);   // 3
    tilingData.gNGReduceParams.set_binaryCGQuotient(binaryCGQuotient_);   // 4
    tilingData.gNGReduceParams.set_binaryCGK(binaryCGK_);                 // 5
    tilingData.gNGReduceParams.set_binaryCGLastNum(binaryCGLastNum_);     // 6
}

void GroupNormGradRegBaseMixTiling::SetKernel2TilingData()
{
    // 共用
    tilingData.gNGKernel2Params.set_stage2CoreUsed(stage2CoreUsed_);
    tilingData.gNGKernel2Params.set_reduceNCnt(reduceNCnt_);
    // mode1
    tilingData.gNGKernel2Params.set_stage2Mode(stage2Mode_);
    tilingData.gNGKernel2Params.set_cFactor(cFactorAlign_);
    tilingData.gNGKernel2Params.set_cBlockFactor(cBlockFactor_);
    tilingData.gNGKernel2Params.set_cTailBlockFactor(cTailBlockFactor_);
    tilingData.gNGKernel2Params.set_stage2BinaryAddQuotient(stage2BinaryAddQuotient_);
    tilingData.gNGKernel2Params.set_stage2BinaryAddK(stage2BinaryAddK_);
    tilingData.gNGKernel2Params.set_cNumMode2PerCore(cNumMode2PerCore_);
    tilingData.gNGKernel2Params.set_tailcNumMode2PerCore(tailcNumMode2PerCore_);
    tilingData.gNGKernel2Params.set_stage2LoopCnt(stage2LoopCnt_);
    tilingData.gNGKernel2Params.set_stage2FactorN(stage2FactorN_);
    // mode2
    tilingData.gNGKernel2Params.set_cFactorStage2Mode2(cTileNum_);
    tilingData.gNGKernel2Params.set_nFactorStage2Mode2(nTileNum_);
    tilingData.gNGKernel2Params.set_nLoop(basicBlockLoop_);
    tilingData.gNGKernel2Params.set_nMainFlodCount(mainFoldCount_);
    tilingData.gNGKernel2Params.set_nTail(nTail_);
    tilingData.gNGKernel2Params.set_cacheBufferCount(cacheBufferCount_);
    tilingData.gNGKernel2Params.set_resultCacheId(resultCacheID_);
    tilingData.gNGKernel2Params.set_cLoopMainBlock(cLoopMain_);
    tilingData.gNGKernel2Params.set_cTailMainBlock(cTailMain_);
    tilingData.gNGKernel2Params.set_cLoopTailBlock(cLoopTail_);
    tilingData.gNGKernel2Params.set_cTailTailBlock(cTailTail_);
}

void GroupNormGradRegBaseMixTiling::PrintTilingData() const
{
    OP_LOGD(opName, "N:                       %ld.", commonParams.N);
    OP_LOGD(opName, "C:                       %ld.", commonParams.C);
    OP_LOGD(opName, "HXW:                     %ld.", commonParams.HxW);
    OP_LOGD(opName, "G:                       %ld.", commonParams.G);
    OP_LOGD(opName, "NXG:                     %ld.", NxG_);
    OP_LOGD(opName, "CPerG:                   %ld.", commonParams.CPerG);
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
    OP_LOGD(opName, "dxIsRequire:             %d.", commonParams.dxIsRequire);
    OP_LOGD(opName, "dgammaIsRequire:         %d.", commonParams.dgammaIsRequire);
    OP_LOGD(opName, "dbetaIsRequire:          %d.", commonParams.dbetaIsRequire);
    OP_LOGD(opName, "modeKey_:                %ld.", this->modeKey_);
    OP_LOGD(opName, "cFactorStage2Mode2       %ld.", cTileNum_);
    OP_LOGD(opName, "nFactorStage2Mode2       %ld.", nTileNum_);
    OP_LOGD(opName, "nLoop                    %ld.", basicBlockLoop_);
    OP_LOGD(opName, "nMainFoldCount           %ld.", mainFoldCount_);
    OP_LOGD(opName, "nTail                    %ld.", nTail_);
    OP_LOGD(opName, "cacheBufferCount         %ld.", cacheBufferCount_);
    OP_LOGD(opName, "resultCacheId            %d.", resultCacheID_);
    OP_LOGD(opName, "cLoopMainBlock           %ld.", cLoopMain_);
    OP_LOGD(opName, "cTailMainBlock           %ld.", cTailMain_);
    OP_LOGD(opName, "cLoopTailBlock           %ld.", cLoopTail_);
    OP_LOGD(opName, "cTailTailBlock           %ld.", cTailTail_);
}

// 注册 mix 模板 tiling（优先级 200，empty 优先级 100 在前，空tensor场景由 empty 接管）
REGISTER_TILING_TEMPLATE("GroupNormGrad", GroupNormGradRegBaseMixTiling, 200);
} // namespace optiling
