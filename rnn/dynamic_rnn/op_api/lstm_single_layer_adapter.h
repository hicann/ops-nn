/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_NN_LSTM_SINGLE_LAYER_ADAPTER_H
#define OPS_NN_LSTM_SINGLE_LAYER_ADAPTER_H

#include <array>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <vector>
#include "../../single_layer_lstm/op_api/single_layer_lstm.h"
#include "level0/add.h"
#include "level0/concat.h"
#include "level0/zero_op.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/slice.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

// Compose dense ACLNN LSTM through SingleLayerLstm nodes in the caller's executor.
namespace lstm_single_layer_adapter {
struct DenseInputs {
    const aclTensor* input;
    const aclTensorList* params;
    const aclTensorList* hx;
    int64_t layers;
    bool hasBias;
    bool train;
    bool bidirectional;
    bool batchFirst;
    double dropout;
};

struct DenseOutputs {
    const aclTensor* y;
    const aclTensor* hy;
    const aclTensor* cy;
    std::array<const aclTensorList*, 7> saved; // Public order: i,j,f,o,h,c,tanhc.
};

inline bool FitsBytes(std::initializer_list<int64_t> shape, int64_t width)
{
    int64_t bytes = width;
    for (const auto dim : shape) {
        if (dim < 0 || (dim != 0 && bytes > std::numeric_limits<int64_t>::max() / dim)) {
            return false;
        }
        bytes *= dim;
    }
    return true;
}

inline const aclTensor* Zeros(const op::Shape& shape, op::DataType dtype, aclOpExecutor* executor)
{
    auto* descriptor = executor->AllocTensor(shape, dtype, op::Format::FORMAT_ND);
    CHECK_RET(descriptor != nullptr, nullptr);
    // AllocTensor is uninitialized; only ZerosLike produces initialized data.
    return l0op::ZerosLike(descriptor, executor);
}

inline const aclTensor* Concat(const aclTensor* a, const aclTensor* b, int64_t axis, aclOpExecutor* executor)
{
    CHECK_RET(a != nullptr && b != nullptr, nullptr);
    const aclTensor* values[] = {a, b};
    auto* list = executor->AllocTensorList(values, 2);
    CHECK_RET(list != nullptr, nullptr);
    return l0op::ConcatD(list, axis, executor);
}

inline const aclTensor* Wide(const aclTensor* value, aclOpExecutor* executor)
{
    CHECK_RET(value != nullptr, nullptr);
    return value->GetDataType() == op::DataType::DT_FLOAT ? value : l0op::Cast(value, op::DataType::DT_FLOAT, executor);
}

inline const aclTensor* PadAxis(const aclTensor* value, size_t axis, int64_t extent, aclOpExecutor* executor)
{
    CHECK_RET(value != nullptr, nullptr);
    auto shape = value->GetViewShape();
    CHECK_RET(axis < shape.GetDimNum() && shape.GetDim(axis) <= extent, nullptr);
    if (shape.GetDim(axis) == extent) {
        return value;
    }
    shape.SetDim(axis, extent - shape.GetDim(axis));
    return Concat(value, Zeros(shape, value->GetDataType(), executor), axis, executor);
}

inline const aclTensor* Slice(const aclTensor* value, std::initializer_list<int64_t> offsets,
                              std::initializer_list<int64_t> sizes, aclOpExecutor* executor)
{
    CHECK_RET(value != nullptr, nullptr);
    auto* offsetArray = executor->AllocIntArray(offsets.begin(), offsets.size());
    auto* sizeArray = executor->AllocIntArray(sizes.begin(), sizes.size());
    CHECK_RET(offsetArray != nullptr && sizeArray != nullptr, nullptr);
    return l0op::Slice(value, offsetArray, sizeArray, executor);
}

inline const aclTensor* Transpose(const aclTensor* value, std::initializer_list<int64_t> perm, aclOpExecutor* executor)
{
    CHECK_RET(value != nullptr, nullptr);
    auto* array = executor->AllocIntArray(perm.begin(), perm.size());
    CHECK_RET(array != nullptr, nullptr);
    return l0op::Transpose(value, array, executor);
}

// [4H,K] -> [Kp,4Hp]. Reshape exposes the gate axis BEFORE padding so that
// [i,f,g,o] stays [i,f,j,o], with H live columns in each Hp-wide gate block.
inline const aclTensor* GateWeight(const aclTensor* weight, int64_t k, int64_t kp, int64_t h, int64_t hp,
                                   aclOpExecutor* executor)
{
    CHECK_RET(weight != nullptr, nullptr);
    if (k == 0) {
        return Zeros({kp, 4 * hp}, weight->GetDataType(), executor);
    }
    const aclTensor* result = l0op::Contiguous(weight, executor);
    result = Transpose(result, {1, 0}, executor);
    CHECK_RET(result != nullptr, nullptr);
    if (h != hp) {
        result = l0op::Reshape(result, op::Shape{k, 4, h}, executor);
        result = PadAxis(result, 2, hp, executor);
        CHECK_RET(result != nullptr, nullptr);
        result = l0op::Reshape(result, op::Shape{k, 4 * hp}, executor);
    }
    return PadAxis(result, 0, kp, executor);
}

inline const aclTensor* Bias(const DenseInputs& in, int64_t paramIndex, int64_t h, int64_t hp, aclOpExecutor* executor)
{
    if (!in.hasBias) {
        return Zeros({4 * hp}, in.input->GetDataType(), executor);
    }
    // Preserve each addend until the kernel can sum them in its FP32 workspace.
    const aclTensor* result = l0op::Contiguous((*in.params)[paramIndex], executor);
    CHECK_RET(result != nullptr, nullptr);
    if (h != hp) {
        result = l0op::Reshape(result, op::Shape{4, h}, executor);
        result = PadAxis(result, 1, hp, executor);
        CHECK_RET(result != nullptr, nullptr);
        result = l0op::Reshape(result, op::Shape{4 * hp}, executor);
    }
    return result;
}

inline const aclTensor* InitialState(const DenseInputs& in, size_t index, int64_t layer, int64_t batch, int64_t h,
                                     int64_t hp, aclOpExecutor* executor)
{
    if (in.hx == nullptr || in.hx->Size() == 0) {
        return Zeros({batch, hp}, in.input->GetDataType(), executor);
    }
    auto* contiguous = l0op::Contiguous((*in.hx)[index], executor);
    const aclTensor* result = Slice(contiguous, {layer, 0, 0}, {1, batch, h}, executor);
    CHECK_RET(result != nullptr, nullptr);
    result = l0op::Reshape(result, op::Shape{batch, h}, executor);
    return PadAxis(result, 1, hp, executor);
}

inline aclnnStatus CopyPublic(const aclTensor* value, const aclTensor* output, aclOpExecutor* executor)
{
    CHECK_RET(value != nullptr && output != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (value->GetDataType() != output->GetDataType()) {
        value = l0op::Cast(value, output->GetDataType(), executor);
        CHECK_RET(value != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    CHECK_RET(l0op::ViewCopy(value, output, executor) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

inline aclnnStatus BuildSingleLayerLstm(const DenseInputs& in, const DenseOutputs& out, aclOpExecutor* executor)
{
    CHECK_RET(executor != nullptr && in.input != nullptr && in.params != nullptr, ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK(!in.bidirectional && in.dropout == 0.0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ascend950 LSTM currently requires unidirectional and dropout=0; "
                                              "no DynamicRNN fallback is used."),
             return ACLNN_ERR_PARAM_INVALID);
    const auto& shape = in.input->GetViewShape();
    const int64_t t = shape.GetDim(in.batchFirst ? 1 : 0);
    const int64_t batch = shape.GetDim(in.batchFirst ? 0 : 1);
    const int64_t inputSize = shape.GetDim(2);
    const int64_t h = (*in.params)[0]->GetViewShape().GetDim(0) / 4;
    const auto dtype = in.input->GetDataType();
    constexpr int64_t MAX_DIM = std::numeric_limits<uint32_t>::max();
    OP_CHECK(t > 0 && batch >= 0 && inputSize >= 0 && h >= 0 && in.layers > 0 && t <= MAX_DIM && batch <= MAX_DIM &&
                 inputSize <= MAX_DIM - 7 && h <= MAX_DIM / 4 - 7,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ascend950 LSTM requires positive T and nonnegative B/I/H "
                                              "with storage extents representable by the kernel tiling."),
             return ACLNN_ERR_PARAM_INVALID);
    // B=0 or H=0 has empty public results. I=0 still has a nonempty recurrence,
    // so this branch deliberately does not use IsEmpty().
    if (batch == 0 || h == 0) {
        return ACLNN_SUCCESS;
    }
    const int64_t hp = ((h + 7) / 8) * 8;
    const int64_t ip0 = inputSize == 0 ? 8 : ((inputSize + 7) / 8) * 8;
    OP_CHECK(ip0 <= MAX_DIM - hp && t <= MAX_DIM / batch,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ascend950 LSTM physical I+H or T*B overflows tiling extents."),
             return ACLNN_ERR_PARAM_INVALID);
    const int64_t largestIp = ip0 > hp ? ip0 : hp;
    const int64_t width = dtype == op::DataType::DT_FLOAT ? 4 : 2;
    OP_CHECK(FitsBytes({t, batch, largestIp}, 4) && FitsBytes({in.layers, 8, t, batch, hp}, 4) &&
                 FitsBytes({in.layers, largestIp + hp, 4 * hp}, width) && FitsBytes({in.layers, batch, hp}, width),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ascend950 LSTM tensor/storage byte product exceeds int64."),
             return ACLNN_ERR_PARAM_INVALID);

    const aclTensor* layerInput = nullptr;
    if (inputSize == 0) {
        layerInput = Zeros({t, batch, ip0}, dtype, executor);
    } else {
        layerInput = l0op::Contiguous(in.input, executor);
        if (in.batchFirst) {
            layerInput = Transpose(layerInput, {1, 0, 2}, executor);
        }
        layerInput = PadAxis(layerInput, 2, ip0, executor);
    }
    CHECK_RET(layerInput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* lastY = nullptr;
    const aclTensor* hy = nullptr;
    const aclTensor* cy = nullptr;
    const int64_t parameterCount = in.hasBias ? 4 : 2;
    for (int64_t layer = 0; layer < in.layers; ++layer) {
        const int64_t logicalI = layer == 0 ? inputSize : h;
        const int64_t ip = layer == 0 ? ip0 : hp;
        const int64_t offset = layer * parameterCount;
        auto* wi = GateWeight((*in.params)[offset], logicalI, ip, h, hp, executor);
        auto* wh = GateWeight((*in.params)[offset + 1], h, hp, h, hp, executor);
        auto* weight = Concat(wi, wh, 0, executor);
        auto* bias = Bias(in, offset + 2, h, hp, executor);
        const auto* biasHh = in.hasBias ? Bias(in, offset + 3, h, hp, executor) : nullptr;
        CHECK_RET(!in.hasBias || biasHh != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto* h0 = InitialState(in, 0, layer, batch, h, hp, executor);
        auto* c0 = InitialState(in, 1, layer, batch, h, hp, executor);
        CHECK_RET(weight != nullptr && bias != nullptr && h0 != nullptr && c0 != nullptr, ACLNN_ERR_INNER_NULLPTR);
        const char* refusal = nullptr;
        OP_CHECK(l0op::SingleLayerLstmSupports(layerInput, h0, "UNIDIRECTIONAL", &refusal),
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ascend950 LSTM physical layer refused: %s",
                         refusal != nullptr ? refusal : "unknown reason"),
                 return ACLNN_ERR_PARAM_INVALID);

        // Native operator order: y,h,c,i,j,f,o,tanhc. All outputs follow the input dtype.
        std::array<aclTensor*, 8> planes{};
        for (size_t i = 0; i < planes.size(); ++i) {
            planes[i] = executor->AllocTensor(op::Shape{t, batch, hp}, dtype, op::Format::FORMAT_ND);
            CHECK_RET(planes[i] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        const auto result = l0op::SingleLayerLstm(layerInput, weight, bias, h0, c0, nullptr, "UNIDIRECTIONAL", "ifjo",
                                                  planes[0], planes[1], planes[2], planes[3], planes[4], planes[5],
                                                  planes[6], planes[7], executor, logicalI, h, biasHh);
        CHECK_RET(std::get<0>(result) != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // The next layer consumes the declared-dtype hidden output.
        layerInput = planes[1];
        lastY = planes[0];
        const aclTensor* layerH = Slice(planes[0], {t - 1, 0, 0}, {1, batch, h}, executor);
        const aclTensor* layerC = Slice(planes[2], {t - 1, 0, 0}, {1, batch, h}, executor);
        CHECK_RET(layerH != nullptr && layerC != nullptr, ACLNN_ERR_INNER_NULLPTR);
        hy = hy == nullptr ? layerH : Concat(hy, layerH, 0, executor);
        cy = cy == nullptr ? layerC : Concat(cy, layerC, 0, executor);
        CHECK_RET(hy != nullptr && cy != nullptr, ACLNN_ERR_INNER_NULLPTR);
        if (in.train) {
            constexpr std::array<size_t, 7> PUBLIC_ORDER = {3, 4, 5, 6, 1, 2, 7};
            for (size_t i = 0; i < PUBLIC_ORDER.size(); ++i) {
                CHECK_RET(out.saved[i] != nullptr, ACLNN_ERR_INNER_NULLPTR);
                const aclTensor* cropped = Slice(planes[PUBLIC_ORDER[i]], {0, 0, 0}, {t, batch, h}, executor);
                const auto status = CopyPublic(cropped, (*out.saved[i])[layer], executor);
                CHECK_RET(status == ACLNN_SUCCESS, status);
            }
        }
    }
    lastY = Slice(lastY, {0, 0, 0}, {t, batch, h}, executor);
    if (in.batchFirst) {
        lastY = Transpose(lastY, {1, 0, 2}, executor);
    }
    auto status = CopyPublic(lastY, out.y, executor);
    CHECK_RET(status == ACLNN_SUCCESS, status);
    status = CopyPublic(hy, out.hy, executor);
    CHECK_RET(status == ACLNN_SUCCESS, status);
    return CopyPublic(cy, out.cy, executor);
}
} // namespace lstm_single_layer_adapter
#endif // OPS_NN_LSTM_SINGLE_LAYER_ADAPTER_H
