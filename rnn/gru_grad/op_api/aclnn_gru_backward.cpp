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
 * \file aclnn_gru_backward.cpp
 * \brief GRU 反向 Level2 API 实现，支持多参数 TensorList 接口
 */

#include "aclnn_gru_backward.h"
#include "gru_grad.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/slice.h"
#include "aclnn_kernels/transpose.h"
#include "level0/add.h"
#include "level0/concat.h"
#include "index/reverse_v2/op_api/reverse_v2.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr int64_t DIM_ZERO = 0;
constexpr int64_t DIM_ONE = 1;
constexpr int64_t DIM_TWO = 2;
constexpr int64_t DIM_THREE = 3;
constexpr int64_t GATE_COUNT = 3; // GRU: reset, update, new
constexpr int64_t GATE_LIST_COUNT = 5;

// 参数列表中每层的参数个数
constexpr int64_t PARAMS_PER_LAYER_NO_BIAS_NO_BIDI = 2; // w_ih, w_hh
constexpr int64_t PARAMS_PER_LAYER_BIAS_OR_BIDI = 4;    // w_ih, w_hh, b_ih, b_hh
constexpr int64_t PARAMS_PER_LAYER_BIAS_AND_BIDI = 8; // w_ih, w_hh, b_ih, b_hh, w_ih_rev, w_hh_rev, b_ih_rev, b_hh_rev

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                       op::DataType::DT_FLOAT16};

static const std::initializer_list<op::Format> FORMAT_SUPPORT_LIST = {op::Format::FORMAT_ND, op::Format::FORMAT_NCL};

static bool IsFormatSupported(const aclTensor* t, const char* name)
{
    if (std::find(FORMAT_SUPPORT_LIST.begin(), FORMAT_SUPPORT_LIST.end(), t->GetStorageFormat()) ==
        FORMAT_SUPPORT_LIST.end()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s format only support ND/NCL, actual %s.", name,
                op::ToString(t->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

static bool CheckNotNull(const aclTensor* input, const aclTensor* hx, const aclTensorList* params, const aclTensor* dy,
                         const aclTensor* dh, const aclTensorList* r, const aclTensorList* z, const aclTensorList* n,
                         const aclTensorList* hn, const aclTensorList* h, aclTensor* dxOut, aclTensor* dhPrevOut,
                         aclTensorList* dparamsOut)
{
    OP_CHECK_NULL(input, return false);
    OP_CHECK_NULL(hx, return false);
    OP_CHECK_NULL(params, return false);
    OP_CHECK_NULL(dy, return false);
    OP_CHECK_NULL(dh, return false);
    OP_CHECK_NULL(r, return false);
    OP_CHECK_NULL(z, return false);
    OP_CHECK_NULL(n, return false);
    OP_CHECK_NULL(hn, return false);
    OP_CHECK_NULL(h, return false);
    OP_CHECK_NULL(dxOut, return false);
    OP_CHECK_NULL(dhPrevOut, return false);
    OP_CHECK_NULL(dparamsOut, return false);
    return true;
}

static bool CheckDtypeConsistent(const aclTensor* ref, const char* name, ge::DataType baseDtype)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(ref, DTYPE_SUPPORT_LIST, return false);
    if (ref->GetDataType() != baseDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s dtype inconsistent, expected %s, actual %s.", name,
                op::ToString(baseDtype).GetString(), op::ToString(ref->GetDataType()).GetString());
        return false;
    }
    if (!IsFormatSupported(ref, name)) {
        return false;
    }
    return true;
}

static bool CheckTensorListDtype(const aclTensorList* list, const char* name, ge::DataType baseDtype)
{
    OP_CHECK_NULL(list, return false);
    for (uint64_t i = 0; i < list->Size(); ++i) {
        auto* tensor = (*list)[i];
        OP_CHECK_NULL(tensor, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(tensor, DTYPE_SUPPORT_LIST, return false);
        if (tensor->GetDataType() != baseDtype) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s[%llu] dtype inconsistent, expected %s, actual %s.", name,
                    static_cast<unsigned long long>(i), op::ToString(baseDtype).GetString(),
                    op::ToString(tensor->GetDataType()).GetString());
            return false;
        }
        if (!IsFormatSupported(tensor, name)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s[%llu] format invalid", name, static_cast<unsigned long long>(i));
            return false;
        }
    }
    return true;
}

static bool CheckDtypeValid(const aclTensor* input, const aclTensor* hx, const aclTensorList* params,
                            const aclTensor* dy, const aclTensor* dh, const aclTensorList* r, const aclTensorList* z,
                            const aclTensorList* n, const aclTensorList* hn, const aclTensorList* h,
                            const aclTensor* batchSizes, aclTensor* dxOut, aclTensor* dhPrevOut,
                            aclTensorList* dparamsOut)
{
    ge::DataType baseDtype = input->GetDataType();
    if (!CheckDtypeConsistent(input, "input", baseDtype))
        return false;
    if (!CheckDtypeConsistent(hx, "hx", baseDtype))
        return false;
    if (!CheckDtypeConsistent(dy, "dy", baseDtype))
        return false;
    if (!CheckDtypeConsistent(dh, "dh", baseDtype))
        return false;
    if (!CheckDtypeConsistent(dxOut, "dxOut", baseDtype))
        return false;
    if (!CheckDtypeConsistent(dhPrevOut, "dhPrevOut", baseDtype))
        return false;

    if (!CheckTensorListDtype(params, "params", baseDtype))
        return false;
    if (!CheckTensorListDtype(r, "r", baseDtype))
        return false;
    if (!CheckTensorListDtype(z, "z", baseDtype))
        return false;
    if (!CheckTensorListDtype(n, "n", baseDtype))
        return false;
    if (!CheckTensorListDtype(hn, "hn", baseDtype))
        return false;
    if (!CheckTensorListDtype(h, "h", baseDtype))
        return false;
    if (!CheckTensorListDtype(dparamsOut, "dparamsOut", baseDtype))
        return false;

    if (batchSizes != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "batchSizes only accepts an empty tensor as input currently.");
        return false;
    }
    return true;
}

static bool ValidateShape(const aclTensor* tensor, const std::vector<int64_t>& expected, const char* name)
{
    auto shape = tensor->GetViewShape();
    if (shape.GetDimNum() != expected.size()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s has wrong dimension count, expected %zu, actual %zu.", name,
                expected.size(), shape.GetDimNum());
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != shape.GetDim(i)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s dim %zu mismatch, expected %ld, actual %ld.", name, i, expected[i],
                    shape.GetDim(i));
            return false;
        }
    }
    return true;
}

static bool CheckShapeValid(const aclTensor* input, const aclTensor* hx, const aclTensorList* params,
                            const aclTensor* dy, const aclTensor* dh, const aclTensorList* r, const aclTensorList* z,
                            const aclTensorList* n, const aclTensorList* hn, const aclTensorList* h, bool hasBias,
                            int64_t numLayers, bool bidirection, bool batchFirst, const aclTensor* dxOut,
                            const aclTensor* dhPrevOut, const aclTensorList* dparamsOut)
{
    // 从 input 推断 T, B, I
    auto xShape = input->GetViewShape();
    OP_CHECK_WRONG_DIMENSION(input, DIM_THREE, return false);
    int64_t T = xShape[batchFirst ? DIM_ONE : DIM_ZERO];
    int64_t B = xShape[batchFirst ? DIM_ZERO : DIM_ONE];
    int64_t I = xShape[DIM_TWO];
    CHECK_RET(T > 0 && B > 0 && I > 0, false);

    int64_t D = bidirection ? 2 : 1;

    auto hxShape = hx->GetViewShape();
    int64_t H = hxShape[hxShape.GetDimNum() - 1];
    CHECK_RET(H > 0, false);

    if (batchFirst) {
        std::vector<int64_t> expected = {B, T, I};
        if (!ValidateShape(input, expected, "input"))
            return false;
    } else {
        std::vector<int64_t> expected = {T, B, I};
        if (!ValidateShape(input, expected, "input"))
            return false;
    }

    std::vector<int64_t> hxExpected = {D * numLayers, B, H};
    if (!ValidateShape(hx, hxExpected, "hx"))
        return false;

    std::vector<int64_t> dyExpected;
    if (batchFirst) {
        dyExpected = {B, T, H * D};
    } else {
        dyExpected = {T, B, H * D};
    }
    if (!ValidateShape(dy, dyExpected, "dy"))
        return false;

    std::vector<int64_t> dhExpected = {numLayers * D, B, H};
    if (!ValidateShape(dh, dhExpected, "dh"))
        return false;

    uint64_t expectedGateLen = static_cast<uint64_t>(D * numLayers);
    const char* gateNames[] = {"r", "z", "n", "hn", "h"};
    const aclTensorList* gateLists[] = {r, z, n, hn, h};
    for (int g = 0; g < GATE_LIST_COUNT; ++g) {
        if (gateLists[g]->Size() != expectedGateLen) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s list size mismatch, expected %llu, actual %llu.", gateNames[g],
                    static_cast<unsigned long long>(expectedGateLen),
                    static_cast<unsigned long long>(gateLists[g]->Size()));
            return false;
        }
        std::vector<int64_t> gateExpected = {T, B, H};
        for (uint64_t i = 0; i < gateLists[g]->Size(); ++i) {
            if (!ValidateShape((*gateLists[g])[i], gateExpected, gateNames[g]))
                return false;
        }
    }

    int64_t paramsPerLayer = hasBias ? (bidirection ? PARAMS_PER_LAYER_BIAS_AND_BIDI : PARAMS_PER_LAYER_BIAS_OR_BIDI) :
                                       (bidirection ? PARAMS_PER_LAYER_BIAS_OR_BIDI : PARAMS_PER_LAYER_NO_BIAS_NO_BIDI);
    uint64_t expectedParamsLen = static_cast<uint64_t>(paramsPerLayer * numLayers);
    if (params->Size() != expectedParamsLen) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "params list size mismatch, expected %llu, actual %llu.",
                static_cast<unsigned long long>(expectedParamsLen), static_cast<unsigned long long>(params->Size()));
        return false;
    }

    std::vector<int64_t> xShapeVec(xShape.GetDimNum());
    for (size_t i = 0; i < xShape.GetDimNum(); ++i) {
        xShapeVec[i] = xShape.GetDim(i);
    }
    if (!ValidateShape(dxOut, xShapeVec, "dxOut"))
        return false;

    std::vector<int64_t> dhPrevOutExpected = {D * numLayers, B, H};
    if (!ValidateShape(dhPrevOut, dhPrevOutExpected, "dhPrevOut"))
        return false;

    if (dparamsOut->Size() != expectedParamsLen) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dparamsOut list size mismatch, expected %llu, actual %llu.",
                static_cast<unsigned long long>(expectedParamsLen),
                static_cast<unsigned long long>(dparamsOut->Size()));
        return false;
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor* input, const aclTensor* hx, const aclTensorList* params,
                               const aclTensor* dy, const aclTensor* dh, const aclTensorList* r, const aclTensorList* z,
                               const aclTensorList* n, const aclTensorList* hn, const aclTensorList* h,
                               const aclTensor* batchSizes, bool hasBias, int64_t numLayers, bool bidirection,
                               bool batchFirst, aclTensor* dxOut, aclTensor* dhPrevOut, aclTensorList* dparamsOut)
{
    CHECK_RET(CheckNotNull(input, hx, params, dy, dh, r, z, n, hn, h, dxOut, dhPrevOut, dparamsOut),
              ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(CheckDtypeValid(input, hx, params, dy, dh, r, z, n, hn, h, batchSizes, dxOut, dhPrevOut, dparamsOut),
              ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckShapeValid(input, hx, params, dy, dh, r, z, n, hn, h, hasBias, numLayers, bidirection, batchFirst,
                              dxOut, dhPrevOut, dparamsOut),
              ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static const aclTensorList* MakeContiguousList(const aclTensorList* list, aclOpExecutor* executor)
{
    std::vector<const aclTensor*> contiguousTensors;
    for (uint64_t i = 0; i < list->Size(); ++i) {
        auto* c = l0op::Contiguous((*list)[i], executor);
        CHECK_RET(c != nullptr, nullptr);
        contiguousTensors.push_back(c);
    }
    return executor->AllocTensorList(contiguousTensors.data(), contiguousTensors.size());
}

static const aclTensorList* CastListElements(const aclTensorList* list, op::DataType dstDtype, aclOpExecutor* executor)
{
    std::vector<const aclTensor*> castTensors;
    for (uint64_t i = 0; i < list->Size(); ++i) {
        auto* c = l0op::Cast((*list)[i], dstDtype, executor);
        CHECK_RET(c != nullptr, nullptr);
        castTensors.push_back(c);
    }
    return executor->AllocTensorList(castTensors.data(), castTensors.size());
}

// 单层单向 L0 输出
struct GruGradLayerOut {
    const aclTensor* dx{nullptr};     // [T, B, inputSize_l]
    const aclTensor* dhPrev{nullptr}; // [1, B, H]
    const aclTensor* dwInput{nullptr};
    const aclTensor* dwHidden{nullptr};
    const aclTensor* dbInput{nullptr};
    const aclTensor* dbHidden{nullptr};
};

// 从 [N, B, H] 第 i 行切出 [1, B, H]
static const aclTensor* SliceRowAt(const aclTensor* t, int64_t i, int64_t B, int64_t H, aclOpExecutor* executor)
{
    const int64_t offData[] = {i, 0, 0};
    aclIntArray* offs = executor->AllocIntArray(offData, DIM_THREE);
    const int64_t sizeData[] = {1, B, H};
    aclIntArray* size = executor->AllocIntArray(sizeData, DIM_THREE);
    return l0op::Slice(t, offs, size, executor);
}

static aclnnStatus GruBackwardSingleLayerDirec(int64_t layerIdx, int64_t directIdx, int64_t D, bool hasBias,
                                               const aclTensor* x, const aclTensor* dyDir, const aclTensor* dh,
                                               const aclTensor* hx, const aclTensorList* params, const aclTensorList* r,
                                               const aclTensorList* z, const aclTensorList* n, const aclTensorList* hn,
                                               const aclTensorList* h, const aclTensor* batchSizes,
                                               GruGradLayerOut& out, aclOpExecutor* executor)
{
    int64_t idx = layerIdx * D + directIdx;
    int64_t groupLen = hasBias ? 4 : 2;
    int64_t B = hx->GetViewShape().GetDim(DIM_ONE);
    int64_t H = hx->GetViewShape().GetDim(DIM_TWO);

    auto initH = SliceRowAt(hx, idx, B, H, executor);
    CHECK_RET(initH != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto dhDir = SliceRowAt(dh, idx, B, H, executor);
    CHECK_RET(dhDir != nullptr, ACLNN_ERR_INNER_NULLPTR);

    int64_t pOff = idx * groupLen;
    auto wInput = (*params)[pOff + 0];
    auto wHidden = (*params)[pOff + 1];
    CHECK_RET(wInput != nullptr && wHidden != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    auto resetGate = (*r)[idx];
    auto updateGate = (*z)[idx];
    auto newGate = (*n)[idx];
    auto hiddenNew = (*hn)[idx];
    auto outputH = (*h)[idx];
    CHECK_RET(resetGate != nullptr && updateGate != nullptr && newGate != nullptr && hiddenNew != nullptr &&
                  outputH != nullptr,
              ACLNN_ERR_PARAM_NULLPTR);

    const char* direction = (directIdx == 0) ? "UNIDIRECTIONAL" : "REDIRECTIONAL";
    auto l0out = l0op::GruGrad(x, wInput, wHidden, initH, outputH, resetGate, updateGate, newGate, hiddenNew, dyDir,
                               dhDir, batchSizes, direction, hasBias, false, executor);
    CHECK_RET(l0out[l0op::OUT_DX_INDEX] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0out[l0op::OUT_DH_PREV_INDEX] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0out[l0op::OUT_DW_INPUT_INDEX] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0out[l0op::OUT_DW_HIDDEN_INDEX] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0out[l0op::OUT_DB_INPUT_INDEX] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0out[l0op::OUT_DB_HIDDEN_INDEX] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    out.dx = l0out[l0op::OUT_DX_INDEX];
    out.dhPrev = l0out[l0op::OUT_DH_PREV_INDEX];
    out.dwInput = l0out[l0op::OUT_DW_INPUT_INDEX];
    out.dwHidden = l0out[l0op::OUT_DW_HIDDEN_INDEX];
    out.dbInput = l0out[l0op::OUT_DB_INPUT_INDEX];
    out.dbHidden = l0out[l0op::OUT_DB_HIDDEN_INDEX];
    return ACLNN_SUCCESS;
}

} // namespace
aclnnStatus aclnnGRUBackwardGetWorkspaceSize(const aclTensor* input, const aclTensorList* params, const aclTensor* hx,
                                             const aclTensor* dy, const aclTensor* dh, const aclTensorList* r,
                                             const aclTensorList* z, const aclTensorList* n, const aclTensorList* hn,
                                             const aclTensorList* h, const aclTensor* batchSizes, bool hasBias,
                                             int64_t numLayers, bool bidirection, bool batchFirst, aclTensor* dxOut,
                                             aclTensor* dhPrevOut, aclTensorList* dparamsOut, uint64_t* workspaceSize,
                                             aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnGRUBackward,
        DFX_IN(input, params, hx, dy, dh, r, z, n, hn, h, batchSizes, hasBias, numLayers, bidirection, batchFirst),
        DFX_OUT(dxOut, dhPrevOut, dparamsOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数校验
    auto ret = CheckParams(input, hx, params, dy, dh, r, z, n, hn, h, batchSizes, hasBias, numLayers, bidirection,
                           batchFirst, dxOut, dhPrevOut, dparamsOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // Contiguous 所有输入
    auto inputContiguous = l0op::Contiguous(input, uniqueExecutor.get());
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (batchFirst) {
        std::vector<int64_t> perm = {1, 0, 2};
        auto valuePerm = uniqueExecutor.get()->AllocIntArray(perm.data(), 3);
        inputContiguous = l0op::Transpose(inputContiguous, valuePerm, uniqueExecutor.get());
        CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto hxContiguous = l0op::Contiguous(hx, uniqueExecutor.get());
    CHECK_RET(hxContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto paramsContiguous = MakeContiguousList(params, uniqueExecutor.get());
    CHECK_RET(paramsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto dyContiguous = l0op::Contiguous(dy, uniqueExecutor.get());
    CHECK_RET(dyContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (batchFirst) {
        std::vector<int64_t> perm = {1, 0, 2};
        auto valuePerm = uniqueExecutor.get()->AllocIntArray(perm.data(), 3);
        dyContiguous = l0op::Transpose(dyContiguous, valuePerm, uniqueExecutor.get());
        CHECK_RET(dyContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto dhContiguous = l0op::Contiguous(dh, uniqueExecutor.get());
    CHECK_RET(dhContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto rContiguous = MakeContiguousList(r, uniqueExecutor.get());
    CHECK_RET(rContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto zContiguous = MakeContiguousList(z, uniqueExecutor.get());
    CHECK_RET(zContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto nContiguous = MakeContiguousList(n, uniqueExecutor.get());
    CHECK_RET(nContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto hnContiguous = MakeContiguousList(hn, uniqueExecutor.get());
    CHECK_RET(hnContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto hContiguous = MakeContiguousList(h, uniqueExecutor.get());
    CHECK_RET(hContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* batchSizesContiguous = nullptr;
    if (batchSizes != nullptr) {
        batchSizesContiguous = l0op::Contiguous(batchSizes, uniqueExecutor.get());
        CHECK_RET(batchSizesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    bool needCast = (input->GetDataType() == op::DataType::DT_FLOAT16);
    if (needCast) {
        inputContiguous = l0op::Cast(inputContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        hxContiguous = l0op::Cast(hxContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(hxContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        dyContiguous = l0op::Cast(dyContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(dyContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        dhContiguous = l0op::Cast(dhContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(dhContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        paramsContiguous = CastListElements(paramsContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(paramsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        rContiguous = CastListElements(rContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(rContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        zContiguous = CastListElements(zContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(zContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        nContiguous = CastListElements(nContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(nContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        hnContiguous = CastListElements(hnContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(hnContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        hContiguous = CastListElements(hContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(hContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    int64_t D = bidirection ? 2 : 1;
    int64_t H = hxContiguous->GetViewShape().GetDim(DIM_TWO);
    int64_t B = hxContiguous->GetViewShape().GetDim(DIM_ONE);
    int64_t groupLen = hasBias ? 4 : 2;
    const aclTensor* curDy = dyContiguous; // [T, B, H*D]; 逐层被本层 dx 覆盖
    std::vector<const aclTensor*> dhPrevVec(static_cast<size_t>(D * numLayers), nullptr);

    for (int64_t l = numLayers - 1; l >= 0; --l) {
        const aclTensor* xL = nullptr;
        if (l == 0) {
            xL = inputContiguous;
        } else {
            std::vector<const aclTensor*> xv;
            for (int64_t d = 0; d < D; ++d) {
                xv.push_back((*hContiguous)[(l - 1) * D + d]);
            }
            if (D == 1) {
                xL = xv[0];
            } else {
                auto tl = uniqueExecutor->AllocTensorList(xv.data(), xv.size());
                xL = l0op::ConcatD(tl, DIM_TWO, uniqueExecutor.get());
                CHECK_RET(xL != nullptr, ACLNN_ERR_INNER_NULLPTR);
            }
        }
        const aclTensor* dxLayer = nullptr;
        for (int64_t d = 0; d < D; ++d) {
            const aclTensor* dyDir = nullptr;
            if (D == 1) {
                dyDir = curDy;
            } else {
                int64_t T = curDy->GetViewShape().GetDim(DIM_ZERO);
                const int64_t offData[] = {0, 0, d * H};
                aclIntArray* offs = uniqueExecutor->AllocIntArray(offData, DIM_THREE);
                const int64_t sizeData[] = {T, B, H};
                aclIntArray* size = uniqueExecutor->AllocIntArray(sizeData, DIM_THREE);
                dyDir = l0op::Slice(curDy, offs, size, uniqueExecutor.get());
                CHECK_RET(dyDir != nullptr, ACLNN_ERR_INNER_NULLPTR);
            }

            GruGradLayerOut gout;
            ret = GruBackwardSingleLayerDirec(l, d, D, hasBias, xL, dyDir, dhContiguous, hxContiguous, paramsContiguous,
                                              rContiguous, zContiguous, nContiguous, hnContiguous, hContiguous,
                                              batchSizesContiguous, gout, uniqueExecutor.get());
            CHECK_RET(ret == ACLNN_SUCCESS, ret);

            dhPrevVec[static_cast<size_t>(l * D + d)] = gout.dhPrev;

            int64_t pOff = (l * D + d) * groupLen;
            auto dwInputOut = needCast ? l0op::Cast(gout.dwInput, op::DataType::DT_FLOAT16, uniqueExecutor.get()) :
                                         gout.dwInput;
            auto dwHiddenOut = needCast ? l0op::Cast(gout.dwHidden, op::DataType::DT_FLOAT16, uniqueExecutor.get()) :
                                          gout.dwHidden;
            CHECK_RET(dwInputOut != nullptr && dwHiddenOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
            CHECK_RET(l0op::ViewCopy(dwInputOut, (*dparamsOut)[pOff + 0], uniqueExecutor.get()) != nullptr,
                      ACLNN_ERR_INNER_NULLPTR);
            CHECK_RET(l0op::ViewCopy(dwHiddenOut, (*dparamsOut)[pOff + 1], uniqueExecutor.get()) != nullptr,
                      ACLNN_ERR_INNER_NULLPTR);
            if (hasBias) {
                auto dbInputOut = needCast ? l0op::Cast(gout.dbInput, op::DataType::DT_FLOAT16, uniqueExecutor.get()) :
                                             gout.dbInput;
                auto dbHiddenOut = needCast ?
                                       l0op::Cast(gout.dbHidden, op::DataType::DT_FLOAT16, uniqueExecutor.get()) :
                                       gout.dbHidden;
                CHECK_RET(dbInputOut != nullptr && dbHiddenOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
                CHECK_RET(l0op::ViewCopy(dbInputOut, (*dparamsOut)[pOff + 2], uniqueExecutor.get()) != nullptr,
                          ACLNN_ERR_INNER_NULLPTR);
                CHECK_RET(l0op::ViewCopy(dbHiddenOut, (*dparamsOut)[pOff + 3], uniqueExecutor.get()) != nullptr,
                          ACLNN_ERR_INNER_NULLPTR);
            }

            dxLayer = (dxLayer == nullptr) ? gout.dx : l0op::Add(dxLayer, gout.dx, uniqueExecutor.get());
            CHECK_RET(dxLayer != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        curDy = dxLayer;
    }

    if (batchFirst) {
        std::vector<int64_t> perm = {1, 0, 2};
        auto valuePerm = uniqueExecutor.get()->AllocIntArray(perm.data(), 3);
        curDy = l0op::Transpose(curDy, valuePerm, uniqueExecutor.get());
        CHECK_RET(curDy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (needCast) {
        curDy = l0op::Cast(curDy, op::DataType::DT_FLOAT16, uniqueExecutor.get());
        CHECK_RET(curDy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    CHECK_RET(l0op::ViewCopy(curDy, dxOut, uniqueExecutor.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto dhPrevList = uniqueExecutor->AllocTensorList(dhPrevVec.data(), dhPrevVec.size());
    CHECK_RET(dhPrevList != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* dhPrevConcat = l0op::ConcatD(dhPrevList, DIM_ZERO, uniqueExecutor.get());
    CHECK_RET(dhPrevConcat != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (needCast) {
        dhPrevConcat = l0op::Cast(dhPrevConcat, op::DataType::DT_FLOAT16, uniqueExecutor.get());
        CHECK_RET(dhPrevConcat != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    CHECK_RET(l0op::ViewCopy(dhPrevConcat, dhPrevOut, uniqueExecutor.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGRUBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGRUBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
