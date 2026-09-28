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
 * \file test_geir_non_zero_with_value.cpp
 * \brief NonZeroWithValue GE IR (graph-mode) construction example.
 *   本算子无 aclnn 入口, 图模式是其调用方式。契约(见 README/op_host def):
 *     · x 严格 2D, dtype 支持 12 类(此处取 float32 / int32 / bool 三档);
 *     · 输出为**静态 max-size**(按全非零最坏情况申请), 非数据依赖 shape:
 *         value = [row*col](dtype 同 x)、index = [2*row*col](int32, 坐标主序
 *         前半段行号后半段列号)、count = [1](int32, 有效长度);
 *     · attr transpose 仅支持 true(坐标主序), attr dtype 仅支持 DT_INT32。
 */

#include <ctime>
#include <cstdint>
#include <iostream>
#include <map>
#include <new>
#include <string>
#include <vector>

#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "array_ops.h"
#include "tensor.h"
#include "types.h"
// 不 include 仓内 ../op_graph/non_zero_with_value_proto.h:
// NonZeroWithValue 的 IR 已由 CANN 内置头 array_ops.h 声明(与仓内 proto 逐字段一致:
// 同样的 12 类 x/value dtype、index/count 恒 int32、attr transpose/dtype), 而内置那份
// **没有包守卫宏**, 两边同时 include 会 "redefinition of class ge::op::NonZeroWithValue"。
// array_ops.h 本身又是 op::Data 的来源, 无法不引。用户实际调用也是用安装态头文件, 故以它为准。

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::string;
using std::vector;

namespace {
constexpr uint32_t BOOL_BYTE_SIZE = 1;
constexpr uint32_t FP16_BYTE_SIZE = 2;
constexpr uint32_t FP32_BYTE_SIZE = 4;
constexpr int64_t INDEX_COORD_NUM = 2; // index 逻辑形状 [2, row*col] 展平

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

int64_t GetShapeSize(const vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

uint32_t GetDataTypeSize(DataType dt)
{
    if (dt == ge::DT_BOOL || dt == ge::DT_INT8 || dt == ge::DT_UINT8) {
        return BOOL_BYTE_SIZE;
    }
    if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16 || dt == ge::DT_INT16 || dt == ge::DT_UINT16) {
        return FP16_BYTE_SIZE;
    }
    return FP32_BYTE_SIZE; // DT_FLOAT / DT_INT32 / DT_UINT32
}

// 造数只求"合法且含零与非零混合": 本样例验证图构建与执行通路, 数值正确性由 TTK 用例集覆盖。
// 每隔 3 个元素置 0, 其余非零 —— 这样 count 落在 (0, row*col) 之间, 三个输出都被真正用到。
int32_t GenInputData(const vector<int64_t>& shape, Tensor& tensor, TensorDesc& tensorDesc, DataType dataType)
{
    tensorDesc.SetRealDimCnt(shape.size());
    size_t elementNum = static_cast<size_t>(GetShapeSize(shape));
    uint32_t dtSize = GetDataTypeSize(dataType);
    size_t dataLen = elementNum * dtSize;
    uint8_t* data = new (std::nothrow) uint8_t[dataLen];
    if (data == nullptr) {
        printf("%s - ERROR - [XIR]: Alloc input data failed\n", GetTime().c_str());
        return FAILED;
    }
    for (size_t i = 0; i < dataLen; ++i) {
        data[i] = 0;
    }
    for (size_t e = 0; e < elementNum; ++e) {
        if (e % 3 == 0) {
            continue; // 留零
        }
        // 只写每个元素的首字节: 对整型/bool 即为非零值; 对 float32 得到一个极小的
        // 非规格化数, 同样满足 x != 0(本样例只需"非零"这一性质)。
        data[e * dtSize] = static_cast<uint8_t>(1 + (e % 7));
    }
    tensor = Tensor(tensorDesc, data, dataLen);
    delete[] data;
    return SUCCESS;
}

struct NonZeroWithValueCase {
    const char* name;
    DataType xDtype; // 12 类中取 3 档: 浮点 / 有符号整型 / bool
    int64_t row;
    int64_t col;
};

int CreateNonZeroWithValueGraph(const NonZeroWithValueCase& tc, Graph& graph, vector<Tensor>& input,
                                vector<Operator>& inputs, vector<Operator>& outputs)
{
    auto op = ge::op::NonZeroWithValue("non_zero_with_value");

    vector<int64_t> xShape = {tc.row, tc.col};
    const int64_t numel = tc.row * tc.col;

    TensorDesc xDesc(ge::Shape(xShape), FORMAT_ND, tc.xDtype);
    xDesc.SetPlacement(ge::kPlacementHost);
    Tensor xTensor;
    if (GenInputData(xShape, xTensor, xDesc, tc.xDtype) != SUCCESS) {
        return FAILED;
    }
    auto xData = op::Data("x").set_attr_index(0);
    xData.update_input_desc_x(xDesc);
    xData.update_output_desc_y(xDesc);
    graph.AddOp(xData);
    op.set_input_x(xData);
    input.push_back(xTensor);
    inputs.push_back(xData);

    // 静态 max-size 输出: value=[numel] 同 x dtype; index=[2*numel] int32; count=[1] int32
    TensorDesc valueDesc(ge::Shape(vector<int64_t>{numel}), FORMAT_ND, tc.xDtype);
    TensorDesc indexDesc(ge::Shape(vector<int64_t>{INDEX_COORD_NUM * numel}), FORMAT_ND, ge::DT_INT32);
    TensorDesc countDesc(ge::Shape(vector<int64_t>{1}), FORMAT_ND, ge::DT_INT32);
    op.update_output_desc_value(valueDesc);
    op.update_output_desc_index(indexDesc);
    op.update_output_desc_count(countDesc);

    // transpose 仅支持 true(坐标主序); dtype 仅支持 DT_INT32 —— 取值来自 README 约束说明
    op.set_attr_transpose(true);
    op.set_attr_dtype(ge::DT_INT32);

    outputs.push_back(op);
    return SUCCESS;
}

int RunNonZeroWithValueCase(const NonZeroWithValueCase& tc)
{
    printf("%s - INFO - [XIR]: Run %s\n", GetTime().c_str(), tc.name);

    Graph graph(tc.name);
    vector<Tensor> input;
    vector<Operator> inputs;
    vector<Operator> outputs;
    if (CreateNonZeroWithValueGraph(tc, graph, input, inputs, outputs) != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create graph failed\n", GetTime().c_str());
        return FAILED;
    }
    graph.SetInputs(inputs).SetOutputs(outputs);

    std::map<AscendString, AscendString> buildOptions = {};
    Session* session = new (std::nothrow) Session(buildOptions);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session failed\n", GetTime().c_str());
        return FAILED;
    }

    uint32_t graphId = 0;
    std::map<AscendString, AscendString> graphOptions = {};
    if (session->AddGraph(graphId, graph, graphOptions) != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph failed\n", GetTime().c_str());
        delete session;
        return FAILED;
    }

    vector<Tensor> output;
    if (session->RunGraph(graphId, input, output) != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run %s graph failed\n", GetTime().c_str(), tc.name);
        delete session;
        return FAILED;
    }

    for (size_t i = 0; i < output.size(); ++i) {
        const TensorDesc& desc = output[i].GetTensorDesc();
        std::cout << tc.name << " output " << i << " dtype: " << desc.GetDataType()
                  << " shape size = " << desc.GetShape().GetShapeSize() << std::endl;
    }
    printf("%s - INFO - [XIR]: Run %s graph success\n", GetTime().c_str(), tc.name);
    delete session;
    return SUCCESS;
}
} // namespace

int main(int argc, char* argv[])
{
    if (argc > 1) {
        std::cout << argv[1] << std::endl;
    }

    printf("%s - INFO - [XIR]: Start to initialize ge\n", GetTime().c_str());
    std::map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(globalOptions) != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge success\n", GetTime().c_str());

    vector<NonZeroWithValueCase> testCases = {
        {"fp32_4x8", ge::DT_FLOAT, 4, 8},
        {"int32_8x16", ge::DT_INT32, 8, 16},
        {"bool_4x8", ge::DT_BOOL, 4, 8},
    };

    for (const auto& tc : testCases) {
        if (RunNonZeroWithValueCase(tc) != SUCCESS) {
            (void)ge::GEFinalize();
            return FAILED;
        }
    }

    if (ge::GEFinalize() != SUCCESS) {
        printf("%s - ERROR - [XIR]: Finalize ge failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ge success\n", GetTime().c_str());
    return SUCCESS;
}
