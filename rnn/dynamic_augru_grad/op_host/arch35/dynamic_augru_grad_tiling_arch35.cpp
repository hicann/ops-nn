/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_augru_grad_tiling_arch35.cpp
 * \brief DynamicAUGRUGrad tiling实现（ascend950/arch35）
 *
 * 单kernel完成AUGRU反向BPTT：按T倒序逐时间步执行"向量阶段（门梯度+融合掩码）->
 * cube matmul（dh回传）->向量累加"，循环结束后做权重梯度/dx/bias归约的matmul与reduce。
 * tiling内容：4个matmul的TCubeTiling、向量分块（bTile/hTile/ubLength）、
 * bias归约切分与workspace大小。
 */

#include <algorithm>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "rnn/dynamic_augru_grad/op_kernel/arch35/dynamic_augru_grad_tiling_data.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorDiv;

namespace {
constexpr int64_t GATE_NUM = 3;
constexpr int64_t FP32_BYTES = 4;
constexpr int64_t FP32_ALIGN = 8; // fp32向量32B块对齐元素数
// 向量阶段fp32 UB缓冲个数（VF操作数9 + VF结果6），合计15
constexpr int64_t NUM_VEC_BUFFERS = 15;
constexpr int64_t UB_RESERVE_BYTES = 16384; // TPipe元数据与matmul预留UB
constexpr int64_t B32_REPEAT_ELEMS = 64;    // 一个向量repeat的fp32元素数(256B)，与kernel侧一致
constexpr int64_t MAX_ROW_ACC = 512;        // kernel侧vbRowAcc_/vbRowSum_元素数（bTile上限）
constexpr int64_t REDUCE_TMP_ROWS = 64;     // kernel侧vbReduceTmp_行数（行x64列）
constexpr int64_t MAX_UB_SEQ_BYTES = 65536; // seq_length本核分片UB上限，与kernel侧一致
// 辅助缓冲字节（与kernel侧InitBuffer逐项对账）：rowAcc(512*4)+rowSum(512*4)+reduceTmp(64*64*4)
constexpr int64_t AUX_BUF_BYTES = (2 * MAX_ROW_ACC + REDUCE_TMP_ROWS * B32_REPEAT_ELEMS) * FP32_BYTES;
constexpr int64_t MAX_UB_LENGTH = 2048;      // 单缓冲元素上限
constexpr int64_t MIN_UB_LENGTH = 16;        // 单缓冲最小元素数（过小无法承载最小tile）
constexpr int64_t DEFAULT_BUFFER_SPACE = -1; // matmul L1/L0缓冲由lib自动分配
constexpr int64_t REDUCE_N_LIMIT = 128;      // bias归约单核列数上限（启发式）
constexpr int64_t GRAD_SIDE_NUM = 2;         // 门梯度两侧：dGi(输入侧)/dGh(隐状态侧)
constexpr int64_t STAGING_QUEUE_NUM = 2;     // fp16 staging in/out两条暂存队列
constexpr int32_t MM_L1_SIZE = 128 * 1024;
constexpr int64_t MM_K_CHUNK = 48;
constexpr int64_t MM_CHUNK_TRAFFIC_LIMIT = 32LL * 1024 * 1024 * 1024;
constexpr int64_t MM_ACCUM_READS = 3;  // partial/sum/correction
constexpr int64_t MM_ACCUM_WRITES = 2; // sum/correction
constexpr int64_t MM_ACCUM_TRAFFIC_BYTES_PER_ELEMENT = (MM_ACCUM_READS + MM_ACCUM_WRITES) * FP32_BYTES;

// 属性索引（与proto/def定义序一致）
constexpr int64_t ATTR_DIRECTION = 0;
constexpr int64_t ATTR_CELL_DEPTH = 1;
constexpr int64_t ATTR_KEEP_PROB = 2;
constexpr int64_t ATTR_CELL_CLIP = 3;
constexpr int64_t ATTR_NUM_PROJ = 4;
constexpr int64_t ATTR_TIME_MAJOR = 5;
constexpr int64_t ATTR_GATE_ORDER = 6;
constexpr int64_t ATTR_RESET_AFTER = 7;

// 输入索引：x/wi/wh/att/y/init_h/h/dy/dh/update/update_att/reset/new/hidden_new/seq_length/mask
constexpr int64_t IDX_X = 0;
constexpr int64_t IDX_WI = 1;
constexpr int64_t IDX_WH = 2;
constexpr int64_t IDX_ATT = 3;
constexpr int64_t IDX_Y = 4;
constexpr int64_t IDX_INIT_H = 5;
constexpr int64_t IDX_H = 6;
constexpr int64_t IDX_DY = 7;
constexpr int64_t IDX_DH = 8;
constexpr int64_t IDX_UPDATE = 9;
constexpr int64_t IDX_UATT = 10;
constexpr int64_t IDX_RESET = 11;
constexpr int64_t IDX_NEW = 12;
constexpr int64_t IDX_HN = 13;
constexpr int64_t IDX_SEQ_LEN = 14;
constexpr int64_t IDX_MASK = 15;
} // namespace

struct DynamicAUGRUGradCompileInfo {};

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& aicCoreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    aicCoreNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicCoreNum == 0, OP_LOGE(context, "aic core num is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static bool IsWeightHiddenShapeValid(const gert::Shape& shape, int64_t hSize)
{
    int64_t threeH = GATE_NUM * hSize;
    if (shape.GetDimNum() == 2) {
        return shape.GetDim(0) == hSize && shape.GetDim(1) == threeH;
    }
    return shape.GetDimNum() == 3 && shape.GetDim(0) == 1 && shape.GetDim(1) == hSize && shape.GetDim(2) == threeH;
}

static ge::graphStatus BuildMatmulTiling(gert::TilingContext* context, AscendC::tiling::TCubeTiling& output,
                                         const char* name, int64_t m, int64_t n, int64_t k, int64_t shapeK,
                                         int64_t aicCoreNum, bool transposeA, bool transposeB)
{
    matmul_tiling::MultiCoreMatmulTiling mm;
    auto mmDataType = matmul_tiling::DataType::DT_FLOAT;
    OP_CHECK_IF(mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, transposeA) == -1,
                OP_LOGE(context, "%s SetAType fail.", name), return ge::GRAPH_FAILED);
    OP_CHECK_IF(mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, transposeB) == -1,
                OP_LOGE(context, "%s SetBType fail.", name), return ge::GRAPH_FAILED);
    OP_CHECK_IF(mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType) == -1,
                OP_LOGE(context, "%s SetCType fail.", name), return ge::GRAPH_FAILED);
    OP_CHECK_IF(mm.SetDim(aicCoreNum) == -1, OP_LOGE(context, "%s SetDim fail.", name), return ge::GRAPH_FAILED);
    OP_CHECK_IF(mm.SetOrgShape(m, n, k) == -1, OP_LOGE(context, "%s SetOrgShape fail.", name), return ge::GRAPH_FAILED);
    OP_CHECK_IF(mm.SetShape(m, n, k) == -1, OP_LOGE(context, "%s SetShape fail.", name), return ge::GRAPH_FAILED);
    OP_CHECK_IF(mm.SetBufferSpace(MM_L1_SIZE, DEFAULT_BUFFER_SPACE, DEFAULT_BUFFER_SPACE) == -1,
                OP_LOGE(context, "%s SetBufferSpace fail.", name), return ge::GRAPH_FAILED);
    if (shapeK > 0) {
        OP_CHECK_IF(mm.SetShape(m, n, shapeK) == -1, OP_LOGE(context, "%s chunk shape fail.", name),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(mm.GetTiling(output) == -1, OP_LOGE(context, "%s GetTiling fail.", name), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static void SetWeightChunking(DynamicAUGRUGradTilingData* tiling, int64_t tb, int64_t iSize, int64_t hPad,
                              int64_t threeHPad, int64_t& dwInputK, int64_t& dwHiddenK)
{
    tiling->mmChunkK = 0;
    tiling->mmChunkMask = 0;
    dwInputK = tb;
    dwHiddenK = tb;
    if (tb <= MM_K_CHUNK) {
        return;
    }
    int64_t chunkP = CeilDiv(tb, MM_K_CHUNK);
    int64_t chunkLen = CeilDiv(CeilDiv(tb, chunkP), MM_DIM_ALIGN) * MM_DIM_ALIGN;
    int64_t chunks = CeilDiv(tb, chunkLen);
    if (chunks * iSize * threeHPad * MM_ACCUM_TRAFFIC_BYTES_PER_ELEMENT <= MM_CHUNK_TRAFFIC_LIMIT) {
        tiling->mmChunkMask |= MM_CHUNK_DW_INPUT;
        dwInputK = chunkLen;
    }
    if (chunks * hPad * threeHPad * MM_ACCUM_TRAFFIC_BYTES_PER_ELEMENT <= MM_CHUNK_TRAFFIC_LIMIT) {
        tiling->mmChunkMask |= MM_CHUNK_DW_HIDDEN;
        dwHiddenK = chunkLen;
    }
    if (tiling->mmChunkMask != 0) {
        tiling->mmChunkK = chunkLen;
    }
}

static ge::graphStatus GetMatmulTilings(gert::TilingContext* context, DynamicAUGRUGradTilingData* tiling, int64_t tSize,
                                        int64_t bSize, int64_t hSize, int64_t iSize, int64_t aicCoreNum)
{
    int64_t tb = tSize * bSize;
    int64_t hPad = CeilDiv(hSize, MM_DIM_ALIGN) * MM_DIM_ALIGN;
    int64_t threeHPad = GATE_NUM * hPad;
    int64_t dwInputK = tb;
    int64_t dwHiddenK = tb;
    SetWeightChunking(tiling, tb, iSize, hPad, threeHPad, dwInputK, dwHiddenK);
    OP_CHECK_IF(BuildMatmulTiling(context, tiling->dgateMMParam, "dgateMM", bSize, hPad, threeHPad,
                                  std::min(hPad / 2, MM_RECURRENT_CHUNK), aicCoreNum, false, true) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "dgateMM tiling failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(BuildMatmulTiling(context, tiling->dwInputMMParam, "dwInputMM", iSize, threeHPad, dwInputK, 0,
                                  aicCoreNum, true, false) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "dwInputMM tiling failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(BuildMatmulTiling(context, tiling->dwHiddenMMParam, "dwHiddenMM", hPad, threeHPad, dwHiddenK, 0,
                                  aicCoreNum, true, false) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "dwHiddenMM tiling failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(BuildMatmulTiling(context, tiling->dxMMParam, "dxMM", tb, iSize, threeHPad,
                                  std::min(hPad, MM_RECURRENT_CHUNK), aicCoreNum, false, true) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "dxMM tiling failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

struct DynamicAUGRUGradParams {
    int64_t tSize = 0;
    int64_t bSize = 0;
    int64_t iSize = 0;
    int64_t hSize = 0;
    int64_t hPad = 0;
    int64_t aicCoreNum = 0;
    int64_t gateOrder = DYNAMIC_AUGRU_GRAD_GATE_ZRH;
    int64_t isSeqLength = 0;
    uint64_t ubSize = 0;
    ge::DataType dataType = ge::DT_UNDEFINED;
};

static ge::graphStatus ValidateBhAndTbhShapes(gert::TilingContext* context, const DynamicAUGRUGradParams& p)
{
    static const int64_t bhInputs[] = {IDX_INIT_H, IDX_DH};
    static const char* bhNames[] = {"init_h", "dh"};
    for (size_t k = 0; k < sizeof(bhInputs) / sizeof(bhInputs[0]); k++) {
        auto shapePtr = context->GetInputShape(bhInputs[k]);
        OP_CHECK_NULL_WITH_CONTEXT(context, shapePtr);
        const gert::Shape& shape = shapePtr->GetStorageShape();
        OP_CHECK_IF(shape.GetDimNum() != 2 || shape.GetDim(0) != p.bSize || shape.GetDim(1) != p.hSize,
                    OP_LOGE(context, "The shape of %s should be [%lld, %lld].", bhNames[k], p.bSize, p.hSize),
                    return ge::GRAPH_FAILED);
    }
    static const int64_t tbhInputs[] = {IDX_ATT, IDX_Y, IDX_DY, IDX_UPDATE, IDX_UATT, IDX_RESET, IDX_NEW, IDX_HN};
    static const char* tbhNames[] = {"weight_att", "y", "dy", "update", "update_att", "reset", "new", "hidden_new"};
    for (size_t k = 0; k < sizeof(tbhInputs) / sizeof(tbhInputs[0]); k++) {
        auto shapePtr = context->GetInputShape(tbhInputs[k]);
        OP_CHECK_NULL_WITH_CONTEXT(context, shapePtr);
        const gert::Shape& shape = shapePtr->GetStorageShape();
        OP_CHECK_IF(
            shape.GetDimNum() != 3 || shape.GetDim(0) != p.tSize || shape.GetDim(1) != p.bSize ||
                shape.GetDim(2) != p.hSize,
            OP_LOGE(context, "The shape of %s should be [%lld, %lld, %lld].", tbhNames[k], p.tSize, p.bSize, p.hSize),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAndValidateShapes(gert::TilingContext* context, DynamicAUGRUGradParams& p)
{
    auto xShapePtr = context->GetInputShape(IDX_X);
    auto hShapePtr = context->GetInputShape(IDX_H);
    auto wHiddenShapePtr = context->GetInputShape(IDX_WH);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, hShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, wHiddenShapePtr);
    const gert::Shape& xShape = xShapePtr->GetStorageShape();
    const gert::Shape& hShape = hShapePtr->GetStorageShape();
    OP_CHECK_IF(xShape.GetDimNum() != 3 || hShape.GetDimNum() != 3,
                OP_LOGE(context, "The dim num of x and h should be 3."), return ge::GRAPH_FAILED);
    p.tSize = xShape.GetDim(0);
    p.bSize = xShape.GetDim(1);
    p.iSize = xShape.GetDim(2);
    p.hSize = hShape.GetDim(2);
    OP_CHECK_IF(p.tSize <= 0 || p.bSize <= 0 || p.iSize <= 0 || p.hSize <= 0,
                OP_LOGE(context, "T/B/I/H should be positive."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(hShape.GetDim(0) != p.tSize || hShape.GetDim(1) != p.bSize,
                OP_LOGE(context, "The shape of h should be [%lld, %lld, %lld].", p.tSize, p.bSize, p.hSize),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsWeightHiddenShapeValid(wHiddenShapePtr->GetStorageShape(), p.hSize),
                OP_LOGE(context, "The shape of weight_hidden should be [%lld, %lld] or [1, %lld, %lld].", p.hSize,
                        GATE_NUM * p.hSize, p.hSize, GATE_NUM * p.hSize),
                return ge::GRAPH_FAILED);
    auto wInputShapePtr = context->GetInputShape(IDX_WI);
    OP_CHECK_NULL_WITH_CONTEXT(context, wInputShapePtr);
    const gert::Shape& wInputShape = wInputShapePtr->GetStorageShape();
    OP_CHECK_IF(
        wInputShape.GetDimNum() != 2 || wInputShape.GetDim(0) != p.iSize || wInputShape.GetDim(1) != GATE_NUM * p.hSize,
        OP_LOGE(context, "The shape of weight_input should be [%lld, %lld].", p.iSize, GATE_NUM * p.hSize),
        return ge::GRAPH_FAILED);
    return ValidateBhAndTbhShapes(context, p);
}

static ge::graphStatus ValidateInputDtypes(gert::TilingContext* context, DynamicAUGRUGradParams& p)
{
    auto inputDesc = context->GetInputDesc(IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    p.dataType = inputDesc->GetDataType();
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16};
    OP_CHECK_IF(supportedDtype.count(p.dataType) == 0, OP_LOGE(context, "invalid dtype of x"), return ge::GRAPH_FAILED);
    for (int64_t idx : {IDX_WI, IDX_WH, IDX_ATT, IDX_Y, IDX_INIT_H, IDX_H, IDX_DY, IDX_DH, IDX_UPDATE, IDX_UATT,
                        IDX_RESET, IDX_NEW, IDX_HN}) {
        auto desc = context->GetInputDesc(idx);
        OP_CHECK_NULL_WITH_CONTEXT(context, desc);
        OP_CHECK_IF(desc->GetDataType() != p.dataType,
                    OP_LOGE(context, "The dtype of input(%lld) should be the same as x.", idx),
                    return ge::GRAPH_FAILED);
    }
    p.hPad = CeilDiv(p.hSize, MM_DIM_ALIGN) * MM_DIM_ALIGN;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateCoreAttrs(gert::TilingContext* context, const gert::RuntimeAttrs* attrs,
                                         DynamicAUGRUGradParams& p)
{
    const char* directionStr = attrs->GetStr(ATTR_DIRECTION);
    const char* gateOrderStr = attrs->GetStr(ATTR_GATE_ORDER);
    OP_CHECK_NULL_WITH_CONTEXT(context, directionStr);
    OP_CHECK_NULL_WITH_CONTEXT(context, gateOrderStr);
    OP_CHECK_IF(strcmp(directionStr, "UNIDIRECTIONAL") != 0,
                OP_LOGE(context, "direction should be UNIDIRECTIONAL, but %s was obtained.", directionStr),
                return ge::GRAPH_FAILED);
    if (strcmp(gateOrderStr, "rzh") == 0) {
        p.gateOrder = DYNAMIC_AUGRU_GRAD_GATE_RZH;
    } else {
        OP_CHECK_IF(strcmp(gateOrderStr, "zrh") != 0,
                    OP_LOGE(context, "gate_order should be zrh or rzh, but %s was obtained.", gateOrderStr),
                    return ge::GRAPH_FAILED);
    }
    const int64_t* cellDepthPtr = attrs->GetAttrPointer<int64_t>(ATTR_CELL_DEPTH);
    const int64_t* numProjPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_PROJ);
    const bool* timeMajorPtr = attrs->GetAttrPointer<bool>(ATTR_TIME_MAJOR);
    OP_CHECK_NULL_WITH_CONTEXT(context, cellDepthPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, numProjPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, timeMajorPtr);
    OP_CHECK_IF(*cellDepthPtr != 1 || *numProjPtr != 0 || !*timeMajorPtr,
                OP_LOGE(context, "Only cell_depth=1, num_proj=0 and time_major=true are supported."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateOptionalAttrs(gert::TilingContext* context, const gert::RuntimeAttrs* attrs)
{
    const float* keepProbPtr = attrs->GetAttrPointer<float>(ATTR_KEEP_PROB);
    const float* cellClipPtr = attrs->GetAttrPointer<float>(ATTR_CELL_CLIP);
    const bool* resetAfterPtr = attrs->GetAttrPointer<bool>(ATTR_RESET_AFTER);
    OP_CHECK_NULL_WITH_CONTEXT(context, keepProbPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, cellClipPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, resetAfterPtr);
    OP_CHECK_IF(*keepProbPtr != 1.0f && *keepProbPtr != -1.0f,
                OP_LOGE(context, "keep_prob should be 1.0 or -1.0, but %f was obtained.", *keepProbPtr),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*cellClipPtr != -1.0f, OP_LOGE(context, "cell_clip should be -1.0, but %f was obtained.", *cellClipPtr),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!*resetAfterPtr, OP_LOGE(context, "reset_after should be true."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateAttrsAndOptionalInput(gert::TilingContext* context, DynamicAUGRUGradParams& p)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    OP_CHECK_IF(ValidateCoreAttrs(context, attrs, p) != ge::GRAPH_SUCCESS, OP_LOGE(context, "Invalid attributes."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateOptionalAttrs(context, attrs) != ge::GRAPH_SUCCESS, OP_LOGE(context, "Invalid attributes."),
                return ge::GRAPH_FAILED);
    auto seqLenShapePtr = context->GetOptionalInputShape(IDX_SEQ_LEN);
    auto seqLenDescPtr = context->GetOptionalInputDesc(IDX_SEQ_LEN);
    if (seqLenShapePtr != nullptr && seqLenDescPtr != nullptr && seqLenShapePtr->GetStorageShape().GetDimNum() != 0) {
        OP_CHECK_IF(seqLenShapePtr->GetStorageShape().GetDimNum() != 1 ||
                        seqLenShapePtr->GetStorageShape().GetDim(0) != p.bSize,
                    OP_LOGE(context, "The shape of seq_length should be [%lld].", p.bSize), return ge::GRAPH_FAILED);
        OP_CHECK_IF(seqLenDescPtr->GetDataType() != ge::DT_INT32,
                    OP_LOGE(context, "The dtype of seq_length should be int32."), return ge::GRAPH_FAILED);
        p.isSeqLength = 1;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetVectorTiling(gert::TilingContext* context, const DynamicAUGRUGradParams& p,
                                       DynamicAUGRUGradTilingData* tiling)
{
    int64_t ubBudget = static_cast<int64_t>(p.ubSize) - UB_RESERVE_BYTES;
    OP_CHECK_IF(ubBudget <= FP32_ALIGN * NUM_VEC_BUFFERS * FP32_BYTES, OP_LOGE(context, "ub size is too small."),
                return ge::GRAPH_FAILED);
    int64_t ubLength = std::min(MAX_UB_LENGTH, ubBudget / (NUM_VEC_BUFFERS * FP32_BYTES));
    ubLength = FloorDiv(ubLength, B32_REPEAT_ELEMS) * B32_REPEAT_ELEMS;
    OP_CHECK_IF(ubLength < MIN_UB_LENGTH, OP_LOGE(context, "ub size is too small."), return ge::GRAPH_FAILED);
    int64_t hAligned = CeilDiv(p.hSize, FP32_ALIGN) * FP32_ALIGN;
    int64_t bTile = 1;
    int64_t hTile = hAligned;
    if (hAligned <= ubLength) {
        bTile = std::min(std::min(ubLength / hAligned, MAX_ROW_ACC), p.bSize);
        OP_CHECK_IF(bTile < 1, OP_LOGE(context, "ub size is too small for bTile."), return ge::GRAPH_FAILED);
    } else {
        hTile = FloorDiv(ubLength, FP32_ALIGN) * FP32_ALIGN;
        OP_CHECK_IF(hTile < FP32_ALIGN, OP_LOGE(context, "hidden_size(%lld) is too large for UB.", p.hSize),
                    return ge::GRAPH_FAILED);
    }
    tiling->bTile = bTile;
    tiling->hTile = hTile;
    tiling->ubLength = ubLength;
    tiling->enablePipeline = (p.dataType == ge::DT_FLOAT) && (bTile >= CeilDiv(p.bSize, p.aicCoreNum)) &&
                             (hTile >= p.hSize) && (GATE_NUM * p.hPad == std::min(p.hPad / 2, MM_RECURRENT_CHUNK));
    int64_t dbInlineBytes = 2 * GRAD_SIDE_NUM * GATE_NUM * p.hPad * FP32_BYTES;
    int64_t dbRemainBytes = ubBudget - NUM_VEC_BUFFERS * ubLength * FP32_BYTES - AUX_BUF_BYTES;
    if (p.dataType == ge::DT_FLOAT16) {
        dbRemainBytes -= STAGING_QUEUE_NUM * ubLength * static_cast<int64_t>(sizeof(uint16_t));
    }
    if (p.isSeqLength == 1) {
        int64_t bPerCore = CeilDiv(p.bSize, p.aicCoreNum);
        int64_t seqUbBytes = CeilDiv(bPerCore, FP32_ALIGN) * FP32_ALIGN * static_cast<int64_t>(sizeof(int32_t));
        if (seqUbBytes <= MAX_UB_SEQ_BYTES) {
            dbRemainBytes -= seqUbBytes;
        }
    }
    tiling->enableDbInline = (dbRemainBytes >= dbInlineBytes) ? 1 : 0;
    int64_t threeH = GATE_NUM * p.hSize;
    tiling->singleCoreReduceN = std::min(CeilDiv(CeilDiv(threeH, p.aicCoreNum), FP32_ALIGN) * FP32_ALIGN,
                                         REDUCE_N_LIMIT);
    return ge::GRAPH_SUCCESS;
}

static int64_t GetBaseWorkspaceFloats(const DynamicAUGRUGradParams& p, const DynamicAUGRUGradTilingData* tiling,
                                      int64_t tb, int64_t tbPad, int64_t threeHPad)
{
    bool needPad = p.hPad != p.hSize;
    bool needCast = p.dataType == ge::DT_FLOAT16;
    int64_t size = GRAD_SIDE_NUM * tbPad * threeHPad + tbPad * p.hPad + p.bSize * p.hPad + p.bSize * p.hSize;
    if (needCast || needPad) {
        size += p.hPad * threeHPad + p.iSize * threeHPad;
    }
    if (needCast) {
        size += tbPad * p.iSize + p.iSize * threeHPad + p.hPad * threeHPad + tb * p.iSize;
    } else {
        if (needPad) {
            size += p.hPad * threeHPad + p.iSize * threeHPad;
        }
        if (tiling->mmChunkMask & MM_CHUNK_DW_INPUT) {
            size += tbPad * p.iSize;
        }
    }
    return size;
}

static int64_t GetReductionWorkspaceFloats(const DynamicAUGRUGradParams& p, const DynamicAUGRUGradTilingData* tiling,
                                           int64_t tb, int64_t threeH, int64_t threeHPad)
{
    int64_t size = (tiling->enableDbInline == 1) ? GRAD_SIDE_NUM * p.aicCoreNum * threeH : 0;
    if (tiling->mmChunkMask & MM_CHUNK_DW_INPUT) {
        size += p.iSize * threeHPad;
    }
    if (tiling->mmChunkMask & MM_CHUNK_DW_HIDDEN) {
        size += p.hPad * threeHPad;
    }
    int64_t correctionRows = (tiling->mmChunkMask & MM_CHUNK_DW_INPUT) ? p.iSize : 0;
    if (tiling->mmChunkMask & MM_CHUNK_DW_HIDDEN) {
        correctionRows = std::max(correctionRows, p.hPad);
    }
    size += correctionRows * threeHPad;
    size += 2 * std::max(p.bSize * p.hPad, tb * p.iSize);
    return size;
}

static ge::graphStatus SetWorkspaceAndSchedule(gert::TilingContext* context, const DynamicAUGRUGradParams& p,
                                               DynamicAUGRUGradTilingData* tiling)
{
    int64_t tb = p.tSize * p.bSize;
    int64_t tbPad = CeilDiv(tb, MM_DIM_ALIGN) * MM_DIM_ALIGN;
    if ((tiling->mmChunkMask & (MM_CHUNK_DW_INPUT | MM_CHUNK_DW_HIDDEN)) != 0) {
        tbPad = CeilDiv(tb, tiling->mmChunkK) * tiling->mmChunkK;
    }
    int64_t threeH = GATE_NUM * p.hSize;
    int64_t threeHPad = GATE_NUM * p.hPad;
    int64_t userWsFloats = GetBaseWorkspaceFloats(p, tiling, tb, tbPad, threeHPad) +
                           GetReductionWorkspaceFloats(p, tiling, tb, threeH, threeHPad);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(userWsFloats * FP32_BYTES + static_cast<int64_t>(sysWorkspaceSize));
    OP_CHECK_IF(context->SetScheduleMode(1) != ge::GRAPH_SUCCESS, OP_LOGE(context, "Failed to set ScheduleMode!"),
                return ge::GRAPH_FAILED);
    context->SetBlockDim(p.aicCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DynamicAUGRUGradTilingFunc(gert::TilingContext* context)
{
    DynamicAUGRUGradParams p;
    OP_CHECK_IF(GetPlatformInfo(context, p.ubSize, p.aicCoreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetAndValidateShapes(context, p) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "Input shape validation failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateInputDtypes(context, p) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "Input dtype validation failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateAttrsAndOptionalInput(context, p) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "Attribute or optional input validation failed."), return ge::GRAPH_FAILED);

    DynamicAUGRUGradTilingData* tiling = context->GetTilingData<DynamicAUGRUGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(DynamicAUGRUGradTilingData), 0, sizeof(DynamicAUGRUGradTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    tiling->timeStep = p.tSize;
    tiling->batchSize = p.bSize;
    tiling->hiddenSize = p.hSize;
    tiling->inputSize = p.iSize;
    tiling->hPad = p.hPad;
    tiling->isSeqLength = p.isSeqLength;
    tiling->gateOrder = p.gateOrder;

    OP_CHECK_IF(
        GetMatmulTilings(context, tiling, p.tSize, p.bSize, p.hSize, p.iSize, p.aicCoreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetMatmulTilings failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetVectorTiling(context, p, tiling) != ge::GRAPH_SUCCESS, OP_LOGE(context, "SetVectorTiling failed."),
                return ge::GRAPH_FAILED);
    return SetWorkspaceAndSchedule(context, p, tiling);
}
static ge::graphStatus TilingParseForDynamicAUGRUGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DynamicAUGRUGrad)
    .Tiling(DynamicAUGRUGradTilingFunc)
    .TilingParse<DynamicAUGRUGradCompileInfo>(TilingParseForDynamicAUGRUGrad);

} // namespace optiling
