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
 * \file advance_step_tiling.cpp
 * \brief
 */
#include <vector>
#include <iostream>
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "platform/platform_infos_def.h"

// V1 多核启用阈值：每核行数低于该值时维持单核（短 kernel 标量存储丢失窗口，见
// Tiling4AdvanceStepLegacy 注释）；1000 为 950PR 实测已验证安全点
constexpr int64_t ADV_V1_MULTICORE_MIN_PERCORE = 1000;
#include "advance_step_tiling_def.h"
#include <string>
#include "util/shape_util.h"
#include "base/err_msg.h"

namespace optiling {

static constexpr uint16_t SOC_VERSION_310 = 200;
static constexpr uint16_t SOC_VERSION_910 = 220;
static constexpr int64_t ONE_BLOCK_BYTE = 32;
static constexpr int64_t UB_RESERVED_BYTE = 768;
static constexpr int64_t IDX_INPUT__INPUT_TOKENS = 0;
static constexpr int64_t IDX_INPUT__SAMPLED_TOKEN_IDS = 1;
static constexpr int64_t IDX_INPUT__INPUT_POSITIONS = 2;
static constexpr int64_t IDX_INPUT__SEQ_LENS = 3;
static constexpr int64_t IDX_INPUT__SLOT_MAPPING = 4;
static constexpr int64_t IDX_INPUT__BLOCK_TABLES = 5;
static constexpr int64_t IDX_OPTIONAL_INPUT__SPEC_TOKEN = 6;
static constexpr int64_t IDX_OPTIONAL_INPUT__ACCEPTED_NUM = 7;
static constexpr int64_t IDX_ATTR__NUM_SEQS = 0;
static constexpr int64_t IDX_ATTR__NUM_QUERIES = 1;
static constexpr int64_t IDX_ATTR__BLOCK_SIZE = 2;
static constexpr int64_t MAX_NUMSEQS = 200000000;

inline static int64_t AlignBytes(int64_t elementNum, int64_t bytes = sizeof(int64_t))
{
    return (elementNum * bytes + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
}

inline static int64_t CeilDiv(int64_t x, int64_t y)
{
    if (y != 0) {
        int64_t rtTmp = ((x % y != 0) ? (x / y + 1) : (x / y));
        int64_t rtCd = (y != 0 && x != 0) ? rtTmp : x;
        return rtCd;
    } else {
        return 0;
    }
}

inline static int64_t FloorDiv(int64_t x, int64_t y) { return y == 0 ? x : x / y; }

static bool CheckOptionalShapeExisting(const gert::StorageShape* storageShape)
{
    OP_CHECK_IF(nullptr == storageShape, OP_LOGD("CheckOptionalShapeExisting: Get nullptr storageShape"), return false);
    int64_t smoothShapeSize = storageShape->GetOriginShape().GetShapeSize();
    OP_CHECK_IF((smoothShapeSize <= 0), OP_LOGD("CheckOptionalShapeExisting", "Get empty storageShape"), return false);
    return true;
}

bool AdvanceStepTilingHelper::GetBaseInfo()
{
    OP_LOGD(context_, "Enter GetBaseInfo");
    auto platformInfo = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    this->aivNum_ = platformInfo.GetCoreNumAiv();
    this->socVersion_ = platformInfo.GetSocVersion();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, this->ubSize_);
    OP_LOGD(context_, "End GetBaseInfo");
    return true;
}

bool AdvanceStepTilingHelper::CheckAttrs()
{
    OP_LOGD(context_, "Enter CheckAttr");
    const gert::RuntimeAttrs* attrs = context_->GetAttrs();
    if (nullptr == attrs) {
        const std::string msgOpName = std::string(Ops::Base::GetOpInfo(context_));
        const std::string msgReason = (std::string(__func__) + ": " + "Invalid RuntimeAttrs, tiling failed.");
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", msgReason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{msgOpName.c_str(), msgReason.c_str()}));
        return false;
    };
    this->numSeqs_ = *(attrs->GetInt(IDX_ATTR__NUM_SEQS));
    this->numQueries_ = *(attrs->GetInt(IDX_ATTR__NUM_QUERIES));
    this->blockSize_ = *(attrs->GetInt(IDX_ATTR__BLOCK_SIZE));
    OP_CHECK_IF(
        this->numSeqs_ < 1,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "num_seqs", std::to_string(this->numSeqs_),
                                              "Non-positive attr num_seqs, tiling failed."),
        return false);
    OP_CHECK_IF(
        this->numQueries_ < 1,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "num_queries", std::to_string(this->numQueries_),
                                              "Non-positive attr num_queries, tiling failed."),
        return false);
    OP_CHECK_IF(
        this->blockSize_ < 1,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "block_size", std::to_string(this->blockSize_),
                                              "Non-positive attr block_size, tiling failed."),
        return false);
    OP_CHECK_IF(this->numSeqs_ >= MAX_NUMSEQS,
                OP_LOGE(context_, "numSeqs %ld should be smaller than 200000000.", this->numSeqs_), return false);
    OP_LOGD(context_, "End CheckAttr");
    return true;
}

bool AdvanceStepTilingHelper::GetInputsOutputs()
{
    auto inputTokensShapePtr = context_->GetInputShape(IDX_INPUT__INPUT_TOKENS); // [num_seqs * (1 + spec_num),]
    auto sampledTokenIdsShapePtr = context_->GetInputShape(IDX_INPUT__SAMPLED_TOKEN_IDS); // [num_seqs, (1 + spec_num)]
    auto inputPositionsShapePtr = context_->GetInputShape(IDX_INPUT__INPUT_POSITIONS); // [num_seqs * (1 + spec_num),]
    auto seqLensShapePtr = context_->GetInputShape(IDX_INPUT__SEQ_LENS);               // [num_seqs * (1 + spec_num),]
    auto slotMappingShapePtr = context_->GetInputShape(IDX_INPUT__SLOT_MAPPING);       // [num_seqs * (1 + spec_num),]
    auto blockTablesShapePtr = context_->GetInputShape(IDX_INPUT__BLOCK_TABLES);       // [num_seqs, seq_lens_maxblock]
    OP_CHECK_IF(inputTokensShapePtr == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    ((context_) == nullptr || (context_)->GetNodeName() == nullptr ? "advance_step" :
                                                                                     (context_)->GetNodeName()),
                    "input_tokens"),
                return false);
    OP_CHECK_IF(sampledTokenIdsShapePtr == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    ((context_) == nullptr || (context_)->GetNodeName() == nullptr ? "advance_step" :
                                                                                     (context_)->GetNodeName()),
                    "sampled_token_ids"),
                return false);
    OP_CHECK_IF(inputPositionsShapePtr == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    ((context_) == nullptr || (context_)->GetNodeName() == nullptr ? "advance_step" :
                                                                                     (context_)->GetNodeName()),
                    "input_positions"),
                return false);
    OP_CHECK_IF(seqLensShapePtr == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    ((context_) == nullptr || (context_)->GetNodeName() == nullptr ? "advance_step" :
                                                                                     (context_)->GetNodeName()),
                    "seq_lens"),
                return false);
    OP_CHECK_IF(slotMappingShapePtr == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    ((context_) == nullptr || (context_)->GetNodeName() == nullptr ? "advance_step" :
                                                                                     (context_)->GetNodeName()),
                    "slot_mapping"),
                return false);
    OP_CHECK_IF(blockTablesShapePtr == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    ((context_) == nullptr || (context_)->GetNodeName() == nullptr ? "advance_step" :
                                                                                     (context_)->GetNodeName()),
                    "block_tables"),
                return false);
    this->inputTokensShape_ = &inputTokensShapePtr->GetStorageShape();
    this->sampledTokenIdsShape_ = &sampledTokenIdsShapePtr->GetStorageShape();
    this->inputPositionsShape_ = &inputPositionsShapePtr->GetStorageShape();
    this->seqLensShape_ = &seqLensShapePtr->GetStorageShape();
    this->slotMappingShape_ = &slotMappingShapePtr->GetStorageShape();
    this->blockTablesShape_ = &blockTablesShapePtr->GetStorageShape();

    auto specTokenShapePtr = context_->GetOptionalInputShape(IDX_OPTIONAL_INPUT__SPEC_TOKEN); // [num_seqs, spec_num]
    auto acceptedNumShapePtr = context_->GetOptionalInputShape(IDX_OPTIONAL_INPUT__ACCEPTED_NUM); // [num_seqs, ]
    this->specTokenExist = CheckOptionalShapeExisting(specTokenShapePtr);
    this->acceptedNumExist = CheckOptionalShapeExisting(acceptedNumShapePtr);
    this->specTokenShape_ = specTokenExist ? &specTokenShapePtr->GetStorageShape() : nullptr;
    this->acceptedNumShape_ = acceptedNumExist ? &acceptedNumShapePtr->GetStorageShape() : nullptr;

    return true;
}

bool AdvanceStepTilingHelper::CheckOptionalInputs()
{
    OP_LOGD(context_, "Enter CheckOptionalInputs");
    auto attrs = context_->GetAttrs();
    this->numSeqs_ = *(attrs->GetInt(IDX_ATTR__NUM_SEQS));
    // check optional inputs if exist
    OP_CHECK_IF(this->specTokenExist &&
                    (this->specTokenShape_->GetDimNum() != 2 || this->specTokenShape_->GetDim(0) != this->numSeqs_),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "spec_token",
                                                      Ops::Base::ToString(*this->specTokenShape_),
                                                      "specToken shape is not [num_seqs, spec_num]."),
                return false);
    this->specNum_ = this->specTokenExist ? this->specTokenShape_->GetDim(1) : 0;
    this->tokenEachReqs_ = this->specNum_ + 1;
    OP_CHECK_IF(this->acceptedNumExist &&
                    (this->acceptedNumShape_->GetDimNum() != 1 || this->acceptedNumShape_->GetDim(0) != this->numSeqs_),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "accepted_num",
                                                      Ops::Base::ToString(*this->acceptedNumShape_),
                                                      "acceptedNum shape is not [num_seqs, ]."),
                return false);
    OP_LOGD(context_, "End CheckOptionalInputs");
    return true;
}

bool AdvanceStepTilingHelper::CheckInputOutputShape()
{
    OP_LOGD(context_, "Enter CheckInputOutputShape");
    OP_CHECK_IF(this->numSeqs_ != this->numQueries_,
                OP_LOGE(context_, "numSeqs %ld should be equal to numQueries %ld.", this->numSeqs_, this->numQueries_),
                return false);
    OP_CHECK_IF(this->inputTokensShape_->GetDimNum() != 1 ||
                    this->inputTokensShape_->GetDim(0) != this->numSeqs_ * this->tokenEachReqs_,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "input_tokens",
                                                      Ops::Base::ToString(*this->inputTokensShape_),
                                                      "inputTokens shape is not [num_seqs * (1 + spec_num),]."),
                return false);
    OP_CHECK_IF(this->sampledTokenIdsShape_->GetDimNum() != 2 ||
                    this->sampledTokenIdsShape_->GetDim(0) != this->numSeqs_ ||
                    this->sampledTokenIdsShape_->GetDim(1) != this->tokenEachReqs_,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "sampled_token_ids",
                                                      Ops::Base::ToString(*this->sampledTokenIdsShape_),
                                                      "sampledTokenIds shape is not [num_seqs, (1 + spec_num)]."),
                return false);
    OP_CHECK_IF(*this->inputPositionsShape_ != *this->inputTokensShape_,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "input_positions",
                                                      Ops::Base::ToString(*this->inputPositionsShape_),
                                                      "inputPosisiongs shape is not [num_seqs * (1 + spec_num),]."),
                return false);
    OP_CHECK_IF(*this->seqLensShape_ != *this->inputTokensShape_,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "seq_lens",
                                                      Ops::Base::ToString(*this->seqLensShape_),
                                                      "seqLens shape is not [num_seqs * (1 + spec_num),]."),
                return false);
    OP_CHECK_IF(*this->slotMappingShape_ != *this->inputTokensShape_,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "slot_mapping",
                                                      Ops::Base::ToString(*this->slotMappingShape_),
                                                      "slotMapping shape is not [num_seqs * (1 + spec_num),]."),
                return false);
    OP_CHECK_IF(this->blockTablesShape_->GetDimNum() != 2 || this->blockTablesShape_->GetDim(0) != this->numSeqs_,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "block_tables",
                                                      Ops::Base::ToString(*this->blockTablesShape_),
                                                      "blockTables shape 1st dim is not num_seqs."),
                return false);
    this->blockTablesStride_ = this->blockTablesShape_->GetDim(1);
    OP_CHECK_IF(this->blockTablesStride_ < 1,
                OP_LOGE(context_, "blockTables shape 2nd dim must be positive, empty tensor is not supported."),
                return false);
    OP_LOGD(context_, "End CheckInputOutputShape");
    return true;
}

bool AdvanceStepTilingHelper::CheckInputOutputFormat() const
{
    OP_LOGD(context_, "Enter CheckInputOutputFormat");
    // need no more null check for InputDesc
    OP_CHECK_IF(context_->GetInputDesc(IDX_INPUT__INPUT_TOKENS)->GetStorageFormat() != ge::FORMAT_ND,
                OP_LOGE(context_, "inputTokensFormat is not ND."), return false);
    OP_CHECK_IF(context_->GetInputDesc(IDX_INPUT__SAMPLED_TOKEN_IDS)->GetStorageFormat() != ge::FORMAT_ND,
                OP_LOGE(context_, "sampledTokenIdsFormat is not ND."), return false);
    OP_CHECK_IF(context_->GetInputDesc(IDX_INPUT__INPUT_POSITIONS)->GetStorageFormat() != ge::FORMAT_ND,
                OP_LOGE(context_, "inputPositionsFormat is not ND."), return false);
    OP_CHECK_IF(context_->GetInputDesc(IDX_INPUT__SEQ_LENS)->GetStorageFormat() != ge::FORMAT_ND,
                OP_LOGE(context_, "seqLensFormat is not ND."), return false);
    OP_CHECK_IF(context_->GetInputDesc(IDX_INPUT__SLOT_MAPPING)->GetStorageFormat() != ge::FORMAT_ND,
                OP_LOGE(context_, "slotMappingFormat is not ND."), return false);
    OP_CHECK_IF(context_->GetInputDesc(IDX_INPUT__BLOCK_TABLES)->GetStorageFormat() != ge::FORMAT_ND,
                OP_LOGE(context_, "blockTablesFormat is not ND."), return false);
    // check OptionalInputFormat if exist
    OP_CHECK_IF(this->specTokenExist &&
                    context_->GetOptionalInputDesc(IDX_OPTIONAL_INPUT__SPEC_TOKEN)->GetStorageFormat() != ge::FORMAT_ND,
                OP_LOGE(context_, "specTokenFormat is not ND."), return false);
    OP_CHECK_IF(
        this->acceptedNumExist &&
            context_->GetOptionalInputDesc(IDX_OPTIONAL_INPUT__ACCEPTED_NUM)->GetStorageFormat() != ge::FORMAT_ND,
        OP_LOGE(context_, "acceptedNumFormat is not ND."), return false);
    OP_LOGD(context_, "End CheckInputOutputFormat");
    return true;
}

bool AdvanceStepTilingHelper::CheckInputOutputDType() const
{
    OP_LOGD(context_, "Enter CheckInputOutputDType");
    // need no more null check for InputDesc
    OP_CHECK_IF(context_->GetInputDesc(IDX_INPUT__INPUT_TOKENS)->GetDataType() != ge::DT_INT64,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context_->GetNodeName(), "input_tokens",
                    std::to_string(static_cast<int>(context_->GetInputDesc(IDX_INPUT__INPUT_TOKENS)->GetDataType())),
                    "inputTokensDType is not DT_INT64."),
                return false);
    OP_CHECK_IF(
        context_->GetInputDesc(IDX_INPUT__SAMPLED_TOKEN_IDS)->GetDataType() != ge::DT_INT64,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context_->GetNodeName(), "sampled_token_ids",
            std::to_string(static_cast<int>(context_->GetInputDesc(IDX_INPUT__SAMPLED_TOKEN_IDS)->GetDataType())),
            "sampledTokenIdsDType is not DT_INT64."),
        return false);
    OP_CHECK_IF(context_->GetInputDesc(IDX_INPUT__INPUT_POSITIONS)->GetDataType() != ge::DT_INT64,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context_->GetNodeName(), "input_positions",
                    std::to_string(static_cast<int>(context_->GetInputDesc(IDX_INPUT__INPUT_POSITIONS)->GetDataType())),
                    "inputPositionsDType is not DT_INT64."),
                return false);
    OP_CHECK_IF(context_->GetInputDesc(IDX_INPUT__SEQ_LENS)->GetDataType() != ge::DT_INT64,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context_->GetNodeName(), "seq_lens",
                    std::to_string(static_cast<int>(context_->GetInputDesc(IDX_INPUT__SEQ_LENS)->GetDataType())),
                    "seqLensDType is not DT_INT64."),
                return false);
    OP_CHECK_IF(context_->GetInputDesc(IDX_INPUT__SLOT_MAPPING)->GetDataType() != ge::DT_INT64,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context_->GetNodeName(), "slot_mapping",
                    std::to_string(static_cast<int>(context_->GetInputDesc(IDX_INPUT__SLOT_MAPPING)->GetDataType())),
                    "slotMappingDType is not DT_INT64."),
                return false);
    OP_CHECK_IF(context_->GetInputDesc(IDX_INPUT__BLOCK_TABLES)->GetDataType() != ge::DT_INT64,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context_->GetNodeName(), "block_tables",
                    std::to_string(static_cast<int>(context_->GetInputDesc(IDX_INPUT__BLOCK_TABLES)->GetDataType())),
                    "blockTablesDType is not DT_INT64."),
                return false);
    // check OptionalInputDType if exist
    OP_CHECK_IF(this->specTokenExist &&
                    context_->GetOptionalInputDesc(IDX_OPTIONAL_INPUT__SPEC_TOKEN)->GetDataType() != ge::DT_INT64,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context_->GetNodeName(), "spec_token",
                    std::to_string(static_cast<int>(
                        context_->GetOptionalInputDesc(IDX_OPTIONAL_INPUT__SPEC_TOKEN)->GetDataType())),
                    "specTokenDType is not DT_INT64."),
                return false);
    OP_CHECK_IF(this->acceptedNumExist &&
                    context_->GetOptionalInputDesc(IDX_OPTIONAL_INPUT__ACCEPTED_NUM)->GetDataType() != ge::DT_INT64,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context_->GetNodeName(), "accepted_num",
                    std::to_string(static_cast<int>(
                        context_->GetOptionalInputDesc(IDX_OPTIONAL_INPUT__ACCEPTED_NUM)->GetDataType())),
                    "acceptedNumDType is not DT_INT64."),
                return false);
    OP_LOGD(context_, "End CheckInputOutputDType");
    return true;
}

bool AdvanceStepTilingHelper::Tiling4Seqs()
{
    OP_LOGD(context_, "Enter Tiling4Seqs");
    this->perCoreSeqs_ = CeilDiv(this->numSeqs_, static_cast<int64_t>(this->aivNum_));
    if (this->perCoreSeqs_ <= 0) {
        const std::string msgOpName = std::string(Ops::Base::GetOpInfo(context_));
        const std::string msgReason = (std::string(__func__) + ": " +
                                       std::string("Tiling produced no sequences per core: num_seqs=") +
                                       std::to_string(this->numSeqs_) +
                                       ", platformCores=" + std::to_string(this->aivNum_));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", msgReason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{msgOpName.c_str(), msgReason.c_str()}));
        return false;
    };
    this->needCoreNum_ = CeilDiv(this->numSeqs_, this->perCoreSeqs_);
    this->lastCoreSeqs_ = this->numSeqs_ - (this->needCoreNum_ - 1) * this->perCoreSeqs_;

    // V2 kernel 向量管线以 int32 参与 blockSize/stride 运算（static_cast<int32_t> 与
    // ArithProgression<int32_t>），属性超出 int32 表示域时截断为脏值甚至 0（除零），
    // host 侧直接拒绝；输入数据值的 int32 域仍为用户保证约束（REQUIREMENTS §5.3）
    constexpr int64_t INT32_DOMAIN_LIMIT = 1LL << 31;
    OP_CHECK_IF(this->blockSize_ >= INT32_DOMAIN_LIMIT,
                OP_LOGE(context_, "blockSize %ld must be smaller than 2^31 for the spec path.", this->blockSize_),
                return false);
    OP_CHECK_IF(this->blockTablesStride_ >= INT32_DOMAIN_LIMIT,
                OP_LOGE(context_, "blockTablesStride %ld must be smaller than 2^31 for the spec path.",
                        this->blockTablesStride_),
                return false);

    // input oneline buffers, use 4 int64 buffers and 1 int32 buffer with an addtional block
    // 每 seq 实际占用与 kernel InitBuffer 逐项对齐：4 x (AlignBytes(te,8) + 32B) + (AlignBytes(te,4) + 32B)。
    // int32 项此前按 AlignBytes(te + 4, 4) 估算，在 te mod 8 ∈ {1,2,3,4} 时比实际少 32B/seq，
    // 导致 perLoopMaxSeqs 偏大、大 numSeqs 时 UB 越界
    int64_t perCoreInBufSize = 4 * AlignBytes(this->tokenEachReqs_ +
                                              ONE_BLOCK_BYTE / static_cast<int64_t>(sizeof(int64_t))) +
                               AlignBytes(this->tokenEachReqs_, static_cast<int64_t>(sizeof(int32_t))) + ONE_BLOCK_BYTE;
    // 向量化行展开 scratch（kernel rowIdxBuf_/byteOffBuf_）：2 x rowStride x 4B + acc32 4B；
    // rowStride 按 align 上界 AlignBytes(te,8)/8 + 4 元素计（与 kernel InitBuffer 预留一致）
    perCoreInBufSize += 2 *
                            (AlignBytes(this->tokenEachReqs_, static_cast<int64_t>(sizeof(int64_t))) /
                                 static_cast<int64_t>(sizeof(int64_t)) +
                             ONE_BLOCK_BYTE / static_cast<int64_t>(sizeof(int64_t))) *
                            static_cast<int64_t>(sizeof(int32_t)) +
                        2 * static_cast<int64_t>(sizeof(int64_t)) + static_cast<int64_t>(sizeof(int32_t));
    // buffers to calc ReduceMin
    int64_t perCoreTempBufSize = AlignBytes(1 + this->tokenEachReqs_, static_cast<int64_t>(sizeof(float)));
    // calc: ubRequired = perCoreInBufSize + tempBufSize + UB_RESERVED_BYTE
    // 向量化 scratch 常量开销：各 buffer 尾部对齐余量（VEC_SCRATCH_ALIGN_SLACK×3 + buffer3 余量）
    // 与 colQue 256b 取整上溢，合计按 4KB 预留（与 kernel InitBuffer 常量项对齐）
    constexpr int64_t ADV_VEC_SCRATCH_CONST_BYTES = 4096;
    int64_t perCoreAvaliableSize = static_cast<int64_t>(this->ubSize_) - perCoreTempBufSize - UB_RESERVED_BYTE -
                                   ADV_VEC_SCRATCH_CONST_BYTES;
    OP_CHECK_IF(perCoreAvaliableSize <= perCoreInBufSize,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    context_->GetNodeName(), "spec_token", std::to_string(this->specNum_),
                    std::string("The per-sequence tile exceeds UB capacity: inputBytes=") +
                        std::to_string(perCoreInBufSize) + ", availableBytes=" + std::to_string(perCoreAvaliableSize)),
                return false);
    // calc perLoopSeqs
    this->perLoopMaxSeqs_ = FloorDiv(perCoreAvaliableSize, perCoreInBufSize);
    // V2 向量化 gather 预算：blockTables 分块搬运缓冲（固定 CAP）+ 每序列 valid01 掩码草稿
    // （行步长按 align 上界 AlignBytes(te,8)/8 + 4 元素计——元素槽数而非字节数，
    //  与 kernel ComputeSlotMapping 的 rowStride = Align(te,8)+4 逐项一致，
    //  常量须与 op_kernel/advance_step_common.h 保持一致）。
    // **仅 Ascend950 启用向量路径**：分块搬运+向量 gather 及配套向量化链路只在 950 上板
    // 验证过（910B4 CI 真机首跑流同步失败 507035）；其余 SoC（910B/910_93 等）走
    // 基线逐行标量路径（master 已验证），接受面与原实现一致。
    // 仅当预留后每循环仍 ≥1 序列且单行可分块时启用向量路径并收紧 perLoopMaxSeqs；
    // 否则维持原公式与标量路径——不新增任何拒绝，接受面与原实现完全一致。
    static constexpr int64_t VEC_BT_CHUNK_CAP_BYTES = 32 * 1024;
    static constexpr int64_t VEC_GATHER_MAX_LANES = 255 * 64;
    const bool socSupportsVecGather = (this->socVersion_ == platform_ascendc::SocVersion::ASCEND950);
    const int64_t rowStrideMax = AlignBytes(this->tokenEachReqs_, sizeof(int64_t)) / sizeof(int64_t) +
                                 ONE_BLOCK_BYTE / sizeof(int64_t);
    const int64_t tailPerSeqBytes = rowStrideMax * static_cast<int64_t>(sizeof(int32_t));
    const bool chunkFits = this->blockTablesStride_ * static_cast<int64_t>(sizeof(int64_t)) <= VEC_BT_CHUNK_CAP_BYTES;
    const int64_t availAfterCap = perCoreAvaliableSize - VEC_BT_CHUNK_CAP_BYTES;
    if (socSupportsVecGather && chunkFits && availAfterCap > perCoreInBufSize + tailPerSeqBytes) {
        int64_t perLoopVec = FloorDiv(availAfterCap, perCoreInBufSize + tailPerSeqBytes);
        const int64_t laneCap = FloorDiv(VEC_GATHER_MAX_LANES, rowStrideMax);
        if (perLoopVec > laneCap) {
            perLoopVec = laneCap;
        }
        // 每循环至少 1 序列：防止 perLoopMaxSeqs=0 传入 kernel 造成 perLoopSeqs_ 除零
        if (perLoopVec >= 1) {
            this->perLoopMaxSeqs_ = perLoopVec;
            this->vectorGather_ = true;
        }
    }
    // only supports full-load
    this->ubTilingPolicy_ = UB_TILING_POLICY::FULL_LOAD;
    OP_LOGD(context_, "End Tiling4Seqs");
    return true;
}

ge::graphStatus AdvanceStepTilingHelper::DoTiling()
{
    OP_LOGD(context_, "Enter DoTiling");
    OP_CHECK_IF(!GetBaseInfo(), OP_LOGE_WITHOUT_REPORT(context_, "GetBaseInfo falied, return false"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckAttrs(), OP_LOGE_WITHOUT_REPORT(context_, "CheckAttrs falied, return false"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!GetInputsOutputs(), OP_LOGE_WITHOUT_REPORT(context_, "GetInputs falied, return false"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckInputOutputFormat(),
                OP_LOGE_WITHOUT_REPORT(context_, "CheckInputOutputFormat falied, return false"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckInputOutputDType(),
                OP_LOGE_WITHOUT_REPORT(context_, "CheckInputOutputDType falied, return false"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckOptionalInputs(), OP_LOGE_WITHOUT_REPORT(context_, "CheckOptionalInputs falied, return false"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(this->specTokenExist && this->acceptedNumExist),
                OP_LOGI(context_, "No optional inputs, use legacy tiling."), return this->Tiling4AdvanceStepLegacy());
    OP_CHECK_IF(!CheckInputOutputShape(),
                OP_LOGE_WITHOUT_REPORT(context_, "CheckInputOutputShape falied, return false"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!Tiling4Seqs(), OP_LOGE_WITHOUT_REPORT(context_, "DoBlockTiling falied, return false"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context_, "End DoTiling");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdvanceStepTilingHelper::PostTiling(AdvanceStepTilingData* tilingData) const
{
    OP_LOGD(context_, "Enter PostTiling");
    // set tiling data
    context_->SetBlockDim(this->needCoreNum_);
    tilingData->set_needCoreNum(this->needCoreNum_);
    tilingData->set_blockTablesStride(this->blockTablesStride_);
    tilingData->set_numSeqs(this->numSeqs_);
    tilingData->set_numQueries(this->numQueries_);
    tilingData->set_blockSize(this->blockSize_);
    tilingData->set_perCoreSeqs(this->perCoreSeqs_);
    tilingData->set_lastCoreSeqs(this->lastCoreSeqs_);
    tilingData->set_perLoopMaxSeqs(this->perLoopMaxSeqs_);
    tilingData->set_specNum(this->specNum_);
    tilingData->set_vectorGather(this->vectorGather_ ? 1 : 0);
    OP_LOGD("AdvanceStepTilingHelper::PostTiling", "Tilingdata perCoreSeqs: %ld, lastCoreSeqs: %ld, specNum: %ld",
            this->perCoreSeqs_, this->lastCoreSeqs_, this->specNum_);

    // set tiling key
    uint32_t tilingKey = static_cast<uint32_t>(this->ubTilingPolicy_);
    context_->SetTilingKey(tilingKey);
    tilingData->SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData->GetDataSize());

    // set workspace size
    // kernel 侧不访问 workspace（Init 从未消费该参数），保留 32MB 仅造成每次执行
    // 无意义的 device 内存申请/释放，置 0 消除
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    OP_LOGD(context_, "End PostTiling");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdvanceStepTilingHelper::Tiling4AdvanceStepLegacy()
{
    auto input0_storage_shape = context_->GetInputShape(0)->GetStorageShape();
    auto input1_storage_shape = context_->GetInputShape(1)->GetStorageShape();
    auto input2_storage_shape = context_->GetInputShape(2)->GetStorageShape();
    auto input3_storage_shape = context_->GetInputShape(3)->GetStorageShape();
    auto input4_storage_shape = context_->GetInputShape(4)->GetStorageShape();
    auto input5_storage_shape = context_->GetInputShape(5)->GetStorageShape();
    OP_CHECK_IF(this->numSeqs_ <= this->numQueries_,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    "CheckAdvTiling", "num_seqs,num_queries",
                    std::to_string(this->numSeqs_) + "," + std::to_string(this->numQueries_),
                    ("numSeqs " + std::to_string(this->numSeqs_) + " should not be smaller or equal to numQueries " +
                     std::to_string(this->numQueries_) + " ")),
                return ge::GRAPH_FAILED);
    // rank 校验必须先于 GetDim 调用：rank 不足时 GetDim 读取 shape 维度数组外的
    // 未初始化数据，校验结果不确定（公开契约 inputTokens/inputPositions/seqLens/
    // slotMapping 为 1-D、sampledTokenIds 为 2-D、blockTables 为 2-D）
    OP_CHECK_IF(input0_storage_shape.GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    "CheckAdvTiling", "input_tokens", std::to_string(input0_storage_shape.GetDimNum()),
                    ("inputTokens must be 1-D, got rank " + std::to_string(input0_storage_shape.GetDimNum()) + " ")),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        input1_storage_shape.GetDimNum() != 2,
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            "CheckAdvTiling", "sampled_token_ids", std::to_string(input1_storage_shape.GetDimNum()),
            ("sampledTokenIds must be 2-D, got rank " + std::to_string(input1_storage_shape.GetDimNum()) + " ")),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(input2_storage_shape.GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    "CheckAdvTiling", "input_positions", std::to_string(input2_storage_shape.GetDimNum()),
                    ("inputPositions must be 1-D, got rank " + std::to_string(input2_storage_shape.GetDimNum()) + " ")),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(input3_storage_shape.GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    "CheckAdvTiling", "seq_lens", std::to_string(input3_storage_shape.GetDimNum()),
                    ("seqLens must be 1-D, got rank " + std::to_string(input3_storage_shape.GetDimNum()) + " ")),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(input4_storage_shape.GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    "CheckAdvTiling", "slot_mapping", std::to_string(input4_storage_shape.GetDimNum()),
                    ("slotMapping must be 1-D, got rank " + std::to_string(input4_storage_shape.GetDimNum()) + " ")),
                return ge::GRAPH_FAILED);
    // blockTables 契约为 2-D [numSeqs, blockTablesStride]（torch_npu 公开文档声明
    // 非投机/投机场景均为二维；spec.yaml rank_range [2,2]；与 V2 路径检查对称）。
    // 此前仅要求 rank>=1，1-D/3-D+ 输入被放行后 stride 按 numel/dim0 计算，
    // 与公开资料的 2-D 声明不一致，统一收紧并同步资料（2026-10-16）
    OP_CHECK_IF(input5_storage_shape.GetDimNum() != 2,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    "CheckAdvTiling", "block_tables", std::to_string(input5_storage_shape.GetDimNum()),
                    ("blockTables must be 2-D [numSeqs, blockTablesStride], got rank " +
                     std::to_string(input5_storage_shape.GetDimNum()) + " ")),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(input0_storage_shape.GetDim(0) != this->numSeqs_,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    "CheckAdvTiling", "input_tokens", std::to_string(input0_storage_shape.GetDim(0)),
                    ("numSeqs " + std::to_string(this->numSeqs_) + " should equal to inputTokens's dim " +
                     std::to_string(input0_storage_shape.GetDim(0)) + " ")),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(input2_storage_shape.GetDim(0) != this->numSeqs_,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    "CheckAdvTiling", "input_positions", std::to_string(input2_storage_shape.GetDim(0)),
                    ("numSeqs " + std::to_string(this->numSeqs_) + " should equal to inputPositions's dim " +
                     std::to_string(input2_storage_shape.GetDim(0)) + " ")),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(input3_storage_shape.GetDim(0) != this->numSeqs_,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    "CheckAdvTiling", "seq_lens", std::to_string(input3_storage_shape.GetDim(0)),
                    ("numSeqs " + std::to_string(this->numSeqs_) + " should equal to seqLens's dim " +
                     std::to_string(input3_storage_shape.GetDim(0)) + " ")),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(input4_storage_shape.GetDim(0) != this->numSeqs_,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    "CheckAdvTiling", "slot_mapping", std::to_string(input4_storage_shape.GetDim(0)),
                    ("numSeqs " + std::to_string(this->numSeqs_) + " should equal to slotMapping's dim " +
                     std::to_string(input4_storage_shape.GetDim(0)) + " ")),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(input5_storage_shape.GetDim(0) != this->numSeqs_,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    "CheckAdvTiling", "block_tables", std::to_string(input5_storage_shape.GetDim(0)),
                    ("numSeqs " + std::to_string(this->numSeqs_) + " should equal to blockTables's first dim " +
                     std::to_string(input5_storage_shape.GetDim(0)) + " ")),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        input1_storage_shape.GetDim(0) != this->numQueries_,
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            "CheckAdvTiling", "sampled_token_ids", std::to_string(input1_storage_shape.GetDim(0)),
            ("numQueries " + std::to_string(this->numQueries_) + " should equal to sampledTokenIds's first dim " +
             std::to_string(input1_storage_shape.GetDim(0)) + " ")),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(input1_storage_shape.GetDim(1) != 1,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON("CheckAdvTiling", "sampled_token_ids",
                                                          std::to_string(input1_storage_shape.GetDim(1)),
                                                          "sampledTokenIds's last dim should be 1"),
                return ge::GRAPH_FAILED);
    this->blockTablesStride_ = input5_storage_shape.GetShapeSize() / input5_storage_shape.GetDim(0);
    // 空 blockTables 防护：stride < 1（如 [numSeqs, 0] 使 GetShapeSize()==0）被放行时，
    // kernel 将以未受限下标读取该 0 元素张量并越界；空 Tensor 语义由可选输入承载
    OP_CHECK_IF(this->blockTablesStride_ < 1,
                OP_LOGE("CheckAdvTiling", "blockTables stride %ld must be positive, empty tensor is not supported.",
                        this->blockTablesStride_),
                return ge::GRAPH_FAILED);

    // V1 主循环多核切分（阈值启用）：多核短 kernel（每核行数过小）实测存在标量 GM 存储
    // 在栅格拆除时的丢失窗口（950PR 实测每核 ≤3 行时部分行写入丢失、每核 1000 行正常），
    // 仅当每核行数达到已验证安全量级（≥1000）时启用多核，否则维持单核全量循环（历史基线路径）。
    // kernel V1 分区为连续切块：核 b 处理 [b*perCoreSeqs, min((b+1)*perCoreSeqs, numQueries))；
    // padding 行由核 0 密集清零（issue #7889 修复），needCoreNum 不再承担稀疏标记步长契约
    this->perCoreSeqs_ = CeilDiv(this->numQueries_, static_cast<int64_t>(this->aivNum_));
    OP_CHECK_IF(this->perCoreSeqs_ <= 0, OP_LOGE("CheckAdvTiling", "Invalid perCoreQueries %ld", this->perCoreSeqs_),
                return ge::GRAPH_FAILED);
    if (this->perCoreSeqs_ >= ADV_V1_MULTICORE_MIN_PERCORE) {
        this->needCoreNum_ = CeilDiv(this->numQueries_, this->perCoreSeqs_);
    } else {
        // 单核全量：核 0 的行区间须覆盖 [0, numQueries)
        this->perCoreSeqs_ = this->numQueries_;
        this->needCoreNum_ = 1;
    }

    this->ubTilingPolicy_ = UB_TILING_POLICY::ONE_CORE_NO_SPEC;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4AdvanceStep(gert::TilingContext* context)
{
    if (nullptr == context) {
        OP_LOGE_WITHOUT_REPORT("AdvanceStep", "%s", "Context is null");
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{"AdvanceStep", "Context is null"}));
        return ge::GRAPH_FAILED;
    };
    OP_LOGD(context->GetNodeName(), "Begin the tiling process for Arch35 architecture");
    OP_LOGI(context, "Enter Tiling4AdvanceStep");
    AdvanceStepTilingData tiling;
    AdvanceStepTilingHelper instanceAdvanceStepTilingHelper(context);
    auto status = instanceAdvanceStepTilingHelper.DoTiling();
    OP_CHECK_IF(status != ge::GRAPH_SUCCESS, OP_LOGE_WITHOUT_REPORT(context, "DoTiling Failed, return Failed."),
                return ge::GRAPH_FAILED);
    status = instanceAdvanceStepTilingHelper.PostTiling(&tiling);

    OP_LOGI(context, "End Tiling4AdvanceStep");
    return status;
}

static ge::graphStatus TilingPrepare4AdvanceStep([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AdvanceStep).Tiling(Tiling4AdvanceStep).TilingParse<AdvanceStepCompileInfo>(TilingPrepare4AdvanceStep);
} // namespace optiling
