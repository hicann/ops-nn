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
 * \file kv_rms_norm_rope_cache_ds_tiling.cpp
 * \brief
 */
#include "kv_rms_norm_rope_cache_tiling.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"

namespace optiling {
constexpr int64_t RMS_NORM_LENGTH = 512;
constexpr int64_t ROPE_LENGTH = 64;
constexpr int64_t MAX_BLOCK_DIM = 65535;

constexpr uint64_t TLING_KEY_5011 = 5011;
constexpr uint64_t TLING_KEY_5010 = 5010;
constexpr uint64_t TLING_KEY_5000 = 5000;
constexpr uint64_t TLING_KEY_4011 = 4011;
constexpr uint64_t TLING_KEY_4001 = 4001;
constexpr uint64_t TLING_KEY_4010 = 4010;
constexpr uint64_t TLING_KEY_4000 = 4000;
constexpr uint64_t TLING_KEY_5001 = 5001;
constexpr uint64_t TLING_KEY_3001 = 3001;
constexpr uint64_t TLING_KEY_3000 = 3000;
constexpr uint64_t TLING_KEY_2000 = 2000;
constexpr uint64_t TLING_KEY_1000 = 1000;

bool KvRmsNormRopeCacheTilingDs::CheckScaleValid(const gert::TilingContext* context)
{
    auto scale1Shape = context->GetOptionalInputShape(K_ROPE_SCALE_IDX);
    auto scale2Shape = context->GetOptionalInputShape(C_KV_SCALE_IDX);

    bool isValid = true;
    isValid = isValid && ((scale1Shape != nullptr) || (scale2Shape != nullptr));
    if ((scale1Shape != nullptr) && (scale2Shape != nullptr)) {
        isValid = isValid && (scale1Shape->GetStorageShape().GetDimNum() == scale2Shape->GetStorageShape().GetDimNum());
    }
    isValid = isValid && (((scale1Shape != nullptr) && (scale1Shape->GetStorageShape().GetDimNum() <= DIM_TWO)) ||
                          ((scale2Shape != nullptr) && (scale2Shape->GetStorageShape().GetDimNum() <= DIM_TWO)));
    if (!isValid) {
        return false;
    }
    if (scale1Shape != nullptr && scale1Shape->GetStorageShape().GetDimNum() == DIM_ONE) {
        isValid = isValid && (scale1Shape->GetStorageShape().GetDim(0) == ROPE_LENGTH);
    } else if (scale1Shape != nullptr && scale1Shape->GetStorageShape().GetDimNum() == DIM_TWO) {
        isValid = isValid && (scale1Shape->GetStorageShape().GetDim(0) == DIM_ONE);
        isValid = isValid && (scale1Shape->GetStorageShape().GetDim(1) == ROPE_LENGTH);
    }

    if (scale2Shape != nullptr && scale2Shape->GetStorageShape().GetDimNum() == DIM_ONE) {
        isValid = isValid && (scale2Shape->GetStorageShape().GetDim(0) == RMS_NORM_LENGTH);
    } else if (scale2Shape != nullptr && scale2Shape->GetStorageShape().GetDimNum() == DIM_TWO) {
        isValid = isValid && (scale2Shape->GetStorageShape().GetDim(0) == DIM_ONE);
        isValid = isValid && (scale2Shape->GetStorageShape().GetDim(1) == RMS_NORM_LENGTH);
    }
    return isValid;
}

bool KvRmsNormRopeCacheTilingDs::IsCapable()
{
    return !isRegbase_;
}

void KvRmsNormRopeCacheTilingDs::DoOpTilingPaBlkNz()
{
    int64_t batchSize = tilingData_.get_batchSize();
    int64_t seqLen = tilingData_.get_seqLength();
    int64_t bs = batchSize * seqLen;
    int64_t blockFactor = (bs + coreNum_ - 1) / coreNum_;
    int64_t blockDim = (bs + blockFactor - 1) / blockFactor;
    tilingData_.set_blockFactor(blockFactor);
    tilingData_.set_blockDim(blockDim);
    constexpr static int64_t maxUbFactor = static_cast<int64_t>(16);
    constexpr static int64_t needUbSize = static_cast<int64_t>(170) * static_cast<int64_t>(1024);
    if (static_cast<int64_t>(ubSize_) >= static_cast<int64_t>(needUbSize)) {
        tilingData_.set_ubFactor(maxUbFactor);
    } else {
        tilingData_.set_ubFactor(1);
    }
}

ge::graphStatus KvRmsNormRopeCacheTilingDs::DoOpTiling()
{
    auto kvShapeTuple = GetShapeTuple(context_, KV_INDEX);
    tilingData_.set_batchSize(std::get<SHAPE_IDX_B>(kvShapeTuple));
    tilingData_.set_numHead(std::get<SHAPE_IDX_N>(kvShapeTuple));
    tilingData_.set_seqLength(std::get<SHAPE_IDX_S>(kvShapeTuple));
    tilingData_.set_cacheLength(cacheLength_);
    tilingData_.set_blockSize(blockSize_);
    tilingData_.set_reciprocal(reciprocal_);
    tilingData_.set_epsilon(epsilon_);
    if (isOutputKv_) {
        tilingData_.set_isOutputKv(1);
    } else {
        tilingData_.set_isOutputKv(0);
    }

    OP_CHECK_IF(
        (!isRegbase_) && (quantMode_ != NON_QUANT_MODE && quantMode_ != SYMMETRIC_QUANT_MODE),
        OP_LOGE(context_->GetNodeName(), "Only Support SYMMETRIC_QUANT or NON_QUANT."), return ge::GRAPH_FAILED);

    if ((!isRegbase_) && (quantMode_ > 0)) {
        OP_CHECK_IF(
            !CheckScaleValid(context_), OP_LOGE(context_->GetNodeName(), "quant scale shape check failed."),
            return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(
        (!isRegbase_) && (dk_ != ROPE_LENGTH), OP_LOGE(context_->GetNodeName(), "rope last dim only support 64."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (!isRegbase_) && (dv_ != RMS_NORM_LENGTH),
        OP_LOGE(context_->GetNodeName(), "rms_norm last dim only support 512."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (!isRegbase_) && (currentCacheMode_ == CacheMode::Norm) && (quantMode_ == SYMMETRIC_QUANT_MODE),
        OP_LOGE(context_->GetNodeName(), "CacheMode::Norm do not support quant!"), return ge::GRAPH_FAILED);

    auto scale1Shape = context_->GetOptionalInputShape(K_ROPE_SCALE_IDX);
    auto scale2Shape = context_->GetOptionalInputShape(C_KV_SCALE_IDX);
    if (scale1Shape != nullptr) {
        tilingData_.set_isKQuant(1);
    } else {
        tilingData_.set_isKQuant(0);
    }
    if (scale2Shape != nullptr) {
        tilingData_.set_isVQuant(1);
    } else {
        tilingData_.set_isVQuant(0);
    }

    if (currentCacheMode_ == CacheMode::PA && quantMode_ == SYMMETRIC_QUANT_MODE) {
        DoOpTilingPaBlkNz();
        tilingKey_ = TLING_KEY_5011;
        return ge::GRAPH_SUCCESS;
    }

    if (currentCacheMode_ == CacheMode::PA_BLK_BNSD) {
        DoOpTilingPaBlkNz();
        if (quantMode_ == SYMMETRIC_QUANT_MODE) {
            tilingKey_ = TLING_KEY_5010;
        } else {
            tilingKey_ = TLING_KEY_5000;
        }
        return ge::GRAPH_SUCCESS;
    }

    if (currentCacheMode_ == CacheMode::PA_NZ) {
        DoOpTilingPaBlkNz();
        if (quantMode_ == SYMMETRIC_QUANT_MODE) {
            tilingKey_ = TLING_KEY_4011;
        } else {
            tilingKey_ = TLING_KEY_4001;
        }
        return ge::GRAPH_SUCCESS;
    }

    if (currentCacheMode_ == CacheMode::PA_BLK_NZ) {
        DoOpTilingPaBlkNz();
        if (quantMode_ == SYMMETRIC_QUANT_MODE) {
            tilingKey_ = TLING_KEY_4010;
        } else {
            tilingKey_ = TLING_KEY_4000;
        }
        return ge::GRAPH_SUCCESS;
    }

    int8_t outputKvValue = tilingData_.get_isOutputKv();
    bool outputKv = outputKvValue == 0 ? false : true;
    if (outputKv && currentCacheMode_ == CacheMode::PA) {
        DoOpTilingPaBlkNz();
        tilingKey_ = TLING_KEY_5001;
        return ge::GRAPH_SUCCESS;
    }

    int64_t batchSize = tilingData_.get_batchSize();
    int64_t seqLen = tilingData_.get_seqLength();

    if (IsB1SD(context_)) {
        int64_t bs = batchSize * seqLen;
        int64_t blockFactor = (bs + coreNum_ - 1) / coreNum_;
        int64_t blockDim = (bs + blockFactor - 1) / blockFactor;
        tilingData_.set_blockFactor(blockFactor);
        tilingData_.set_blockDim(blockDim);
        constexpr static int64_t maxUbFactor = static_cast<int64_t>(16);
        constexpr static int64_t needUbSize = static_cast<int64_t>(170) * static_cast<int64_t>(1024);
        if (static_cast<int64_t>(ubSize_) >= static_cast<int64_t>(needUbSize)) {
            tilingData_.set_ubFactor(maxUbFactor);
        } else {
            tilingData_.set_ubFactor(1);
        }
    } else {
        if (batchSize % BATCHES_FOR_EACH_CORE == 0) {
            tilingData_.set_blockDim(batchSize / BATCHES_FOR_EACH_CORE);
            tilingData_.set_rowsPerBlock(BATCHES_FOR_EACH_CORE);
        } else {
            tilingData_.set_blockDim(batchSize);
            tilingData_.set_rowsPerBlock(1);
        }
        // Check blockDim <= MAX_BLOCK_DIM
        OP_CHECK_IF(
            tilingData_.get_blockDim() > MAX_BLOCK_DIM,
            OP_LOGE(context_->GetNodeName(), "blockDim must be smaller than 65535."), return ge::GRAPH_FAILED);
    }

    if (IsB1SD(context_)) {
        if (isPagedAttention_) {
            tilingKey_ = TLING_KEY_3001;
        } else {
            tilingKey_ = TLING_KEY_3000;
        }
    } else {
        if (isPagedAttention_) {
            tilingKey_ = TLING_KEY_2000;
        } else {
            tilingKey_ = TLING_KEY_1000;
        }
        if (isMTP_) {
            tilingKey_ += 1;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvRmsNormRopeCacheTilingDs::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(tilingData_.get_blockDim());
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = DEFAULT_WORKSPACE_SIZE;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("KvRmsNormRopeCache", KvRmsNormRopeCacheTilingDs, TEMPLATE_DS_PRIORITY);
} // namespace optiling
