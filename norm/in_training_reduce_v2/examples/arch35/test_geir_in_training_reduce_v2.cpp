/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <exception>
#include <map>
#include <string>
#include <vector>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "../../op_graph/in_training_reduce_v2_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::string;
using std::vector;

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return string(tmp);
}

struct ExampleOptions {
    DataType dtype = DT_FLOAT;
    Format format = FORMAT_NCHW;
    vector<int64_t> shape = {2, 3, 8, 8};
};

bool ParseOptions(int argc, char* argv[], ExampleOptions& options)
{
    if (argc > 3) {
        return false;
    }
    if (argc > 1) {
        const string dtype = argv[1];
        if (dtype != "fp16" && dtype != "fp32") {
            return false;
        }
        options.dtype = dtype == "fp16" ? DT_FLOAT16 : DT_FLOAT;
    }
    if (argc > 2) {
        const string format = argv[2];
        if (format == "NCHW") {
            options.format = FORMAT_NCHW;
        } else if (format == "NHWC") {
            options.format = FORMAT_NHWC;
            options.shape = {2, 8, 8, 3};
        } else if (format == "NDHWC") {
            options.format = FORMAT_NDHWC;
            options.shape = {2, 4, 4, 8, 3};
        } else if (format == "NCDHW") {
            options.format = FORMAT_NCDHW;
            options.shape = {2, 3, 4, 4, 8};
        } else if (format == "ND") {
            options.format = FORMAT_ND;
            options.shape = {2, 3, 64};
        } else {
            return false;
        }
    }
    return true;
}

int32_t GenFloatData(const ExampleOptions& options, Tensor& tensor, TensorDesc& desc)
{
    desc.SetRealDimCnt(options.shape.size());
    size_t count = 1;
    for (const auto dim : options.shape) {
        count *= static_cast<size_t>(dim);
    }
    const size_t elemBytes = options.dtype == DT_FLOAT16 ? sizeof(uint16_t) : sizeof(float);
    vector<uint8_t> data(count * elemBytes);
    // Exact IEEE binary16 encodings of {-4, -3, -2, -1, 0, 1, 2, 3, 4}.
    constexpr uint16_t halfValues[] = {0xc400, 0xc200, 0xc000, 0xbc00, 0, 0x3c00, 0x4000, 0x4200, 0x4400};
    for (size_t i = 0; i < count; ++i) {
        if (options.dtype == DT_FLOAT16) {
            std::memcpy(data.data() + i * elemBytes, &halfValues[i % 9], elemBytes);
        } else {
            const float value = static_cast<float>(static_cast<int>(i % 9) - 4);
            std::memcpy(data.data() + i * elemBytes, &value, elemBytes);
        }
    }
    if (tensor.SetTensorDesc(desc) != ge::GRAPH_SUCCESS ||
        tensor.SetData(data.data(), data.size()) != ge::GRAPH_SUCCESS) {
        printf("Set input tensor failed\n");
        return FAILED;
    }
    return SUCCESS;
}

bool CheckOutputs(const ExampleOptions& options, const vector<Tensor>& outputs)
{
    const bool channelLast = options.format == FORMAT_NHWC || options.format == FORMAT_NDHWC;
    const size_t channelAxis = channelLast ? options.shape.size() - 1 : 1;
    const int64_t batches = options.shape[0];
    const int64_t channels = options.shape[channelAxis];
    int64_t spatialSize = 1;
    vector<int64_t> expectedShape(options.shape.size(), 1);
    expectedShape[0] = batches;
    expectedShape[channelAxis] = channels;
    for (size_t axis = 1; axis < options.shape.size(); ++axis) {
        if (axis != channelAxis) {
            spatialSize *= options.shape[axis];
        }
    }
    vector<double> sums(batches * channels, 0.0);
    vector<double> squares(batches * channels, 0.0);
    for (int64_t n = 0; n < batches; ++n) {
        for (int64_t c = 0; c < channels; ++c) {
            for (int64_t r = 0; r < spatialSize; ++r) {
                const int64_t index = channelLast ? (n * spatialSize + r) * channels + c :
                                                    (n * channels + c) * spatialSize + r;
                const double value = static_cast<double>(index % 9 - 4);
                sums[n * channels + c] += value;
                squares[n * channels + c] += value * value;
            }
        }
    }
    if (outputs.size() != 2) {
        printf("Expected two outputs, got %zu\n", outputs.size());
        return false;
    }
    for (size_t outputIndex = 0; outputIndex < outputs.size(); ++outputIndex) {
        const auto desc = outputs[outputIndex].GetTensorDesc();
        const auto& expected = outputIndex == 0 ? sums : squares;
        if (desc.GetDataType() != DT_FLOAT || desc.GetFormat() != FORMAT_ND ||
            desc.GetShape().GetDims() != expectedShape || outputs[outputIndex].GetData() == nullptr ||
            outputs[outputIndex].GetSize() != expected.size() * sizeof(float)) {
            printf("Output %zu has an unexpected descriptor or buffer size\n", outputIndex);
            return false;
        }
        for (size_t i = 0; i < expected.size(); ++i) {
            float actual;
            std::memcpy(&actual, outputs[outputIndex].GetData() + i * sizeof(float), sizeof(float));
            // These small integer inputs, products and sums are exactly representable in float32.
            if (static_cast<double>(actual) != expected[i]) {
                printf("Output %zu[%zu]: actual=%g, expected=%g\n", outputIndex, i, actual, expected[i]);
                return false;
            }
            printf("%s[%zu] = %g\n", outputIndex == 0 ? "sum" : "square_sum", i, actual);
        }
    }
    return true;
}

int32_t RunExample(const ExampleOptions& options)
{
    Graph graph("tc_ge_irrun_in_training_reduce_v2");
    auto xData = op::Data("placeholder_x").set_attr_index(0);
    TensorDesc xDesc(ge::Shape(options.shape), options.format, options.dtype);
    xDesc.SetPlacement(ge::kPlacementHost);
    Tensor tensorX;
    if (GenFloatData(options, tensorX, xDesc) != SUCCESS || xData.update_input_desc_x(xDesc) != ge::GRAPH_SUCCESS ||
        xData.update_output_desc_y(xDesc) != ge::GRAPH_SUCCESS || graph.AddOp(xData) != ge::GRAPH_SUCCESS) {
        printf("Prepare graph input failed\n");
        return FAILED;
    }
    auto reduceOp = op::INTrainingReduceV2("in_training_reduce_v2");
    reduceOp.set_input_x(xData);
    const vector<Operator> inputs = {xData};
    const vector<Operator> outputs = {reduceOp};
    graph.SetInputs(inputs).SetOutputs(outputs);

    // Destroy the session and its graph before GEFinalize, including on every failure path.
    const std::map<AscendString, AscendString> buildOptions;
    ge::Session session(buildOptions);
    const uint32_t graphId = 0;
    Status ret = session.AddGraph(graphId, graph, buildOptions);
    if (ret != SUCCESS) {
        printf("AddGraph failed ret=%u\n", ret);
        return FAILED;
    }
    vector<Tensor> results;
    // RunGraph is synchronous; its returned tensors are ready for host-side checking.
    ret = session.RunGraph(graphId, {tensorX}, results);
    if (ret != SUCCESS) {
        const auto error = ge::GEGetErrorMsgV2();
        printf("RunGraph failed ret=%u: %s\n", ret, error.GetString());
        return FAILED;
    }
    const bool matches = CheckOutputs(options, results);
    ret = session.RemoveGraph(graphId);
    if (ret != SUCCESS) {
        printf("RemoveGraph failed ret=%u\n", ret);
        return FAILED;
    }
    return matches ? SUCCESS : FAILED;
}

int main(int argc, char* argv[])
{
    ExampleOptions options;
    if (!ParseOptions(argc, argv, options)) {
        printf("Usage: %s [fp16|fp32] [NCHW|NHWC|NDHWC|NCDHW|ND]\n", argv[0]);
        return FAILED;
    }
    printf("x dtype=%d format=%d shape=[", options.dtype, options.format);
    for (const auto dim : options.shape) {
        printf("%ld,", dim);
    }
    printf("]\n");
    const std::map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(globalOptions);
    if (ret != SUCCESS) {
        printf("GEInitialize failed ret=%u\n", ret);
        return FAILED;
    }
    int32_t runStatus = FAILED;
    try {
        runStatus = RunExample(options);
    } catch (const std::exception& error) {
        printf("Example failed: %s\n", error.what());
    }
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("GEFinalize failed ret=%u\n", ret);
        return FAILED;
    }
    printf("%s - %s\n", GetTime().c_str(), runStatus == SUCCESS ? "PASS" : "FAIL");
    return runStatus;
}
