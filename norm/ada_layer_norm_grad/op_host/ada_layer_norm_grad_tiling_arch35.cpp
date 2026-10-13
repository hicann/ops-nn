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
 * \file ada_layer_norm_grad_tiling_arch35.cpp
 * \brief A5 (ascend950) arch35 tiling implementation.
 */

#include "ada_layer_norm_grad_tiling.h"
#include <limits>

namespace optiling {

static constexpr uint32_t WORKSPACE_SIZE = 512;
static constexpr uint32_t SYNC_SPACE = 512;
static constexpr uint32_t REDUCE_BUF = 64 * 4;
static constexpr uint32_t BLOCK_ALIGN = 32;
static constexpr uint32_t BLOCK_NUMBER = BLOCK_ALIGN / sizeof(float);
static constexpr uint32_t B32_ALIGN_FACTOR = 8;
static constexpr uint32_t B16_ALIGN_FACTOR = 16;
static constexpr uint32_t FP32_SIZE = 4;
static constexpr uint32_t FP16_SIZE = 2;
static constexpr uint32_t FLOAT_DTYPE_KEY = 1;
static constexpr uint32_t FLOAT16_DTYPE_KEY = 2;
static constexpr uint32_t FLOAT16_FLOAT_DTYPE_KEY = 3;
static constexpr uint32_t BFLOAT16_DTYPE_KEY = 4;
static constexpr uint32_t BFLOAT16_FLOAT_DTYPE_KEY = 5;
static constexpr int64_t CUT_N_PER_ROW_FP32_BUFFER_NUM = 3;
static constexpr int64_t CUT_N_PER_ROW_RSTD_MEAN_BYTES = 2 * BLOCK_NUMBER * FP32_SIZE;
static constexpr int64_t CUT_N_SCALE_BUFFER_NUM = 3;
static constexpr int64_t CUT_N_FIXED_BUFFER_NUM = 9;
// cutD fixed part: rstd/mean scalar queues (2*BLOCK_NUMBER*FP32_SIZE) + tmpBuf (V_LENGTH*2+2 floats = 520 B).
// Both are independent of dInnerLength and covered by CUT_D_FIXED_BYTES.
static constexpr int64_t CUT_D_FIXED_BYTES = 2 * BLOCK_NUMBER * FP32_SIZE + 520;
// cutD per-chunk part (all dInnerLength-scaled). The kernel arch35 cutD InitQueue/InitOutputQueue/InitInputQueue
// allocates exactly 15 fp32-equivalent buffers per element, BUFFER_NUM == 1 (no double buffering):
//   - 6 fp32 accumulators: pdGammaBuf/pdBetaBuf/dScaleBuf/dShiftBuf/dyGStageBuf/yStageBuf (each
//     ROUND_UP(dInnerLength,8)*4)
//   - 6 T/U input/output queues: dyQue/xQue/scaleQue/gammaQue/betaQue/pdXQue (each BUFFER_NUM(1) *
//     ROUND_UP(dInnerLength, blockNumTdtype/Udtype)*sizeof(T/U))
//   - 3 cast-out temps: scaleOutTmp/shiftOutTmp/gammaOutTmp (each ROUND_UP(dInnerLength, blockNumTdtype/Udtype)
//     *sizeof(T/U); sized by dInnerLength since they only ever hold dInOnceUb elements)
// The fp32 worst case (T=U=float, align 8) makes every buffer exactly dInnerLength*4, so per-element bytes
// = 15 * FP32_SIZE = 60, which is >= every mixed b16/fp32 case (b16 buffers are <= their fp32-equivalent).
// Keep in sync with ada_layer_norm_grad_cut_d_arch35.h InitQueue/InitOutputQueue/InitInputQueue.
// (The deterministic phase re-inits the same pipe after Reset() with fixed ROW_TEMPLATE*COL_TEMPLATE buffers,
// ~186KB max, which is smaller than the cutD main phase, so the main phase is the binding constraint.)
static constexpr int64_t CUT_D_T_INPUT_NUM = 6;
static constexpr int64_t CUT_D_FP32_CHUNK_NUM = 6;
static constexpr int64_t CUT_D_CAST_OUT_NUM = 3;
static constexpr uint8_t SCHEDULE_MODE = 1;
static constexpr size_t MAX_DIM_NUM = 8;
static constexpr size_t MIN_DIM_NUM = 1;

static inline int64_t CeilDiv(int64_t value, int64_t factor)
{
    return factor == 0 ? value : (value + factor - 1) / factor;
}

static inline int64_t CeilAlign(int64_t value, int64_t factor)
{
    return factor == 0 ? value : (value + factor - 1) / factor * factor;
}

static inline int64_t FloorAlign(int64_t value, int64_t factor)
{
    return factor == 0 ? value : (value - factor + 1) / factor * factor;
}

// arch35-only input contract checks. These mirror the A2 aclnn ValidateScaleShiftShapes semantics so the
// GE/tiling entry rejects illegal scale and rstd/mean shapes instead of running the kernel on them. Kept
// in the arch35 tiling (never the shared base) so the A2 (ascend910b/910_93) behaviour is untouched.
static bool AdaLayerNormGradArch35CheckScaleShape(const gert::TilingContext* context, const gert::Shape& dyShape)
{
    auto scaleShapePtr = context->GetInputShape(4);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShapePtr);
    const gert::Shape& scaleShape = scaleShapePtr->GetStorageShape();
    const int64_t dyDim = dyShape.GetDimNum();
    if (dyDim < 2) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "dy", Ops::Base::ToString(dyShape).c_str(),
                                              "dy must have rank >= 2");
        return false;
    }
    const int64_t scaleDim = scaleShape.GetDimNum();
    const int64_t h = dyShape.GetDim(dyDim - 1);
    const int64_t batchDim = dyDim - 2;

    // Expect [B..., 1, H] or [B..., H] where B... matches dy's leading dims.
    bool expect1 = (scaleDim == batchDim + 2); // [B..., 1, H]
    for (int64_t i = 0; expect1 && i < batchDim; ++i) {
        expect1 = scaleShape.GetDim(i) == dyShape.GetDim(i);
    }
    expect1 = expect1 && scaleShape.GetDim(batchDim) == 1 && scaleShape.GetDim(batchDim + 1) == h;

    bool expect2 = (scaleDim == batchDim + 1); // [B..., H]
    for (int64_t i = 0; expect2 && i < batchDim; ++i) {
        expect2 = scaleShape.GetDim(i) == dyShape.GetDim(i);
    }
    expect2 = expect2 && scaleShape.GetDim(batchDim) == h;

    if (!expect1 && !expect2) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "scale", Ops::Base::ToString(scaleShape).c_str(),
                                              ("scale shape must be [B...,1," + std::to_string(h) + "] or [B...," +
                                               std::to_string(h) + "] matching dy leading dims")
                                                  .c_str());
        return false;
    }
    return true;
}

static bool AdaLayerNormGradArch35CheckReduceShape(const gert::TilingContext* context, const gert::Shape& dyShape,
                                                   size_t inputIndex, const char* inputName)
{
    auto shapePtr = context->GetInputShape(inputIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapePtr);
    const gert::Shape& shape = shapePtr->GetStorageShape();
    const int64_t dyDim = dyShape.GetDimNum();
    bool ok = (shape.GetDimNum() == dyDim) && (shape.GetDim(dyDim - 1) == 1);
    for (int64_t i = 0; ok && i < dyDim - 1; ++i) {
        ok = shape.GetDim(i) == dyShape.GetDim(i);
    }
    if (!ok) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), inputName, Ops::Base::ToString(shape).c_str(),
                                              "shape must equal dy with the last dimension fixed to 1");
        return false;
    }
    return true;
}

bool AdaLayerNormGradArch35Tiling::IsCapable()
{
    // only serves the regbase (ascend950) soc; the A2 four templates reject regbase
    return commonParams.isRegBase;
}

ge::graphStatus AdaLayerNormGradArch35Tiling::DoOpTiling()
{
    const int64_t batch = static_cast<int64_t>(commonParams.batchSize);
    const int64_t seq = static_cast<int64_t>(commonParams.seqSize);
    const int64_t row = static_cast<int64_t>(commonParams.rowSize);
    const int64_t col = static_cast<int64_t>(commonParams.colSize);
    const int64_t coreNum = static_cast<int64_t>(commonParams.coreNum);

    auto dyShapePtr = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dyShapePtr);
    const gert::Shape& dyShape = dyShapePtr->GetStorageShape();
    if (!AdaLayerNormGradArch35CheckScaleShape(context_, dyShape) ||
        !AdaLayerNormGradArch35CheckReduceShape(context_, dyShape, 2, "rstd") ||
        !AdaLayerNormGradArch35CheckReduceShape(context_, dyShape, 3, "mean")) {
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(row < MIN_DIM_NUM || col < MIN_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "dy",
                                                      "row=" + std::to_string(row) + ",col=" + std::to_string(col),
                                                      "row>=1 and col>=1"),
                return ge::GRAPH_FAILED);

    const int64_t colAlignV = CeilAlign(col, B32_ALIGN_FACTOR);
    const int64_t colAlignM = (commonParams.dyDtype == ge::DataType::DT_FLOAT) ? colAlignV :
                                                                                 CeilAlign(col, B16_ALIGN_FACTOR);

    const uint64_t maxUbSize = commonParams.ubSizePlatForm;
    const uint32_t actualAvailUb = (maxUbSize > WORKSPACE_SIZE + SYNC_SPACE + REDUCE_BUF) ?
                                       (maxUbSize - WORKSPACE_SIZE - SYNC_SPACE - REDUCE_BUF) :
                                       0;

    // ---- cutN capacity solve: rows per UB tile (fp32 worst-case accounting) ----
    const int64_t perRowBytes = CUT_N_PER_ROW_FP32_BUFFER_NUM * colAlignV * FP32_SIZE + CUT_N_PER_ROW_RSTD_MEAN_BYTES;
    const int64_t fixedBytes = CUT_N_FIXED_BUFFER_NUM * colAlignV * FP32_SIZE;
    int64_t nAvailInUb = 0;
    int64_t n = actualAvailUb / perRowBytes;
    while (n >= MIN_DIM_NUM) {
        int64_t scaleRows = CeilDiv(n, seq) + 1;
        int64_t scaleBytes = CUT_N_SCALE_BUFFER_NUM * scaleRows * colAlignV * FP32_SIZE;
        if (n * perRowBytes + scaleBytes + fixedBytes <= actualAvailUb) {
            nAvailInUb = n;
            break;
        }
        --n;
    }

    // ---- cutD chunk width (only reached when cutN cannot host a single row) ----
    int64_t dInnerLength = 0;
    bool cutDPath = (nAvailInUb < MIN_DIM_NUM);
    if (cutDPath) {
        // per-element bytes cover every dInnerLength-scaled UB buffer in the cutD kernel InitQueue/
        // InitOutputQueue/InitInputQueue (15 fp32-equivalent buffers; see the constant definitions for the
        // buffer-to-constant mapping). This keeps the host UB budget exactly in sync with the kernel
        // allocation so large-col cutD shapes cannot over-allocate UB.
        const int64_t perChunkBytes = (CUT_D_T_INPUT_NUM + CUT_D_FP32_CHUNK_NUM + CUT_D_CAST_OUT_NUM) * FP32_SIZE;
        int64_t availForChunk = (actualAvailUb > CUT_D_FIXED_BYTES) ? (actualAvailUb - CUT_D_FIXED_BYTES) : 0;
        dInnerLength = FloorAlign(availForChunk / perChunkBytes, B32_ALIGN_FACTOR);
        OP_CHECK_IF(dInnerLength < B32_ALIGN_FACTOR,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "col", std::to_string(col),
                                                          "col too large for the available UB capacity"),
                    return ge::GRAPH_FAILED);
    }

    // ---- core split: batch-split preferred; row-split fallback; shrink cores if row < coreNum ----
    bool isBatchSplit = false;
    int64_t blockNum = 0;
    int64_t blockFormer = 0;
    int64_t blockTail = 0;
    if (batch >= coreNum) {
        isBatchSplit = true;
        int64_t blockFormerBatch = CeilDiv(batch, coreNum);
        blockNum = CeilDiv(batch, blockFormerBatch);
        blockTail = batch - (blockNum - 1) * blockFormerBatch;
        blockFormer = blockFormerBatch * seq;
        blockTail = blockTail * seq;
    } else if (row >= coreNum) {
        isBatchSplit = false;
        blockFormer = CeilDiv(row, coreNum);
        blockNum = CeilDiv(row, blockFormer);
        blockTail = row - (blockNum - 1) * blockFormer;
    } else {
        isBatchSplit = false;
        blockNum = row;
        blockFormer = (row != 0) ? CeilDiv(row, blockNum) : 1;
        blockTail = row - (blockNum - 1) * blockFormer;
    }
    OP_CHECK_IF(blockNum < MIN_DIM_NUM,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "blockNum", std::to_string(blockNum), ">=1"),
                return ge::GRAPH_FAILED);

    const int64_t tier = (!cutDPath && isBatchSplit)  ? 1 :
                         (!cutDPath && !isBatchSplit) ? 2 :
                         (cutDPath && isBatchSplit)   ? 3 :
                                                        4;
    // TilingKey = tier*100 + dtypeKey*10, matching the registered AdaLayerNormGrad_110.._150 /
    // _210.._250 / _310.._350 / _410.._450 tiling-data classes and the arch35 kernel dispatch
    // (TILING_KEY_IS(110)..TILING_KEY_IS(450)).
    const int64_t dtypeKey = static_cast<int64_t>(commonParams.dtypeKey);
    tilingKey_ = static_cast<uint64_t>(tier * 100 + dtypeKey * 10);

    // b16 outputs keep per-batch scale/shift in the fp32 workspace even in batch-split mode so the
    // reduction is accumulated in fp32 and cast exactly once (see the kernel B16_OUTPUT handling).
    const bool b16Output = commonParams.dyDtype != ge::DataType::DT_FLOAT;
    const int64_t workspaceNum = (isBatchSplit && !b16Output) ? 2 : (2 + 2 * batch);
    const uint64_t usrWorkspaceSize = static_cast<uint64_t>(blockNum) * colAlignV * workspaceNum * FP32_SIZE;

    td_.set_batch(batch);
    td_.set_seq(seq);
    td_.set_row(row);
    td_.set_col(col);
    td_.set_colAlignV(colAlignV);
    td_.set_colAlignM(colAlignM);
    td_.set_blockNum(blockNum);
    td_.set_blockFormer(blockFormer);
    td_.set_blockTail(blockTail);
    td_.set_colRegime(cutDPath ? 1 : 0);
    td_.set_nAvailInUb(nAvailInUb);
    td_.set_dtypeKey(dtypeKey);
    td_.set_isDeterministicKey(1);
    td_.set_rstdElemNum(row);
    td_.set_usrWorkspaceSize(usrWorkspaceSize);

    if (!cutDPath) {
        int64_t ubLoopOfFormerBlock = CeilDiv(blockFormer, nAvailInUb);
        int64_t ubTailOfFormerBlock = blockFormer - (ubLoopOfFormerBlock - 1) * nAvailInUb;
        int64_t ubLoopOfTailBlock = CeilDiv(blockTail, nAvailInUb);
        int64_t ubTailOfTailBlock = blockTail - (ubLoopOfTailBlock - 1) * nAvailInUb;
        td_.set_ubLoopOfFormerBlock(ubLoopOfFormerBlock);
        td_.set_ubLoopOfTailBlock(ubLoopOfTailBlock);
        td_.set_ubTailOfFormerBlock(ubTailOfFormerBlock);
        td_.set_ubTailOfTailBlock(ubTailOfTailBlock);
        td_.set_dOuterLength(0);
        td_.set_dInnerLength(0);
        td_.set_dInnerLengthTail(0);
    } else {
        int64_t dOuterLength = CeilDiv(col, dInnerLength);
        int64_t dInnerLengthTail = col - dInnerLength * (dOuterLength - 1);
        td_.set_dOuterLength(dOuterLength);
        td_.set_dInnerLength(dInnerLength);
        td_.set_dInnerLengthTail(dInnerLengthTail);
        td_.set_ubLoopOfFormerBlock(0);
        td_.set_ubLoopOfTailBlock(0);
        td_.set_ubTailOfFormerBlock(0);
        td_.set_ubTailOfTailBlock(0);
    }

    OP_LOGI(context_->GetNodeName(),
            "AdaLayerNormGrad arch35 tiling: batch=%ld seq=%ld row=%ld col=%ld coreNum=%ld colAlignV=%ld "
            "colAlignM=%ld blockNum=%ld blockFormer=%ld blockTail=%ld cutDPath=%d nAvailInUb=%ld "
            "dInnerLength=%ld tier=%ld dtypeKey=%ld isDet=%ld usrWorkspaceSize=%lu",
            batch, seq, row, col, coreNum, colAlignV, colAlignM, blockNum, blockFormer, blockTail, cutDPath ? 1 : 0,
            nAvailInUb, dInnerLength, tier, dtypeKey, td_.get_isDeterministicKey(), usrWorkspaceSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaLayerNormGradArch35Tiling::GetWorkspaceSize()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = sysWorkspaceSize + td_.get_usrWorkspaceSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaLayerNormGradArch35Tiling::PostTiling()
{
    context_->SetBlockDim(td_.get_blockNum());
    context_->SetScheduleMode(SCHEDULE_MODE);
    td_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(td_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

uint64_t AdaLayerNormGradArch35Tiling::GetTilingKey() const { return tilingKey_; }

// highest priority (lowest value) so the arch35 template is tried first
REGISTER_TILING_TEMPLATE("AdaLayerNormGrad", AdaLayerNormGradArch35Tiling, 100);

} // namespace optiling
