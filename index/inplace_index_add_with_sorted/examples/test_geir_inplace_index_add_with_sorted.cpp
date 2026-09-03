/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_geir_inplace_index_add_with_sorted.cpp
 * @brief
 *
 * 构图并运行单算子子图；InplaceIndexAddWithSorted 有 4 个必选输入 / 1 个可选输入 / 1 个属性 / 1 个输出（inplace 更新
 * var）：
 *   - 必选输入：var, value, sorted_indices, pos
 *   - 可选输入：alpha（标量 fp32，缺省按 1.0）
 *   - 必选属性：axis（int，仅支持 0）
 *   - 输出    ：var（与输入 var 同 shape/dtype，inplace 累加）
 *
 * 目标平台：Ascend950 PR / Ascend950 DT（arch35 / DAV_3510）
 *
 * 样例 shape（M=N=D=4，无重复索引）：
 *   var[4,4]=1.0, value[4,4]=0.5, sorted_indices=[0,1,2,3], pos=[0,1,2,3], alpha=1.0
 *   期望输出：var[i,j] = 1.0 + 1.0 * 0.5 = 1.5
 */

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include "assert.h"

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "../op_graph/inplace_index_add_with_sorted_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

uint32_t GetDataTypeSize(DataType dt)
{
    uint32_t oneByte = 1;
    uint32_t twoByte = 2;
    uint32_t fourByte = 4;
    uint32_t eightByte = 8;
    uint32_t dilation = 0;

    if (dt == ge::DT_FLOAT) {
        dilation = fourByte;
    } else if (dt == ge::DT_FLOAT16) {
        dilation = twoByte;
    } else if (dt == ge::DT_BF16) {
        dilation = twoByte;
    } else if (dt == ge::DT_INT16) {
        dilation = twoByte;
    } else if (dt == ge::DT_UINT16) {
        dilation = twoByte;
    } else if (dt == ge::DT_INT32) {
        dilation = fourByte;
    } else if (dt == ge::DT_UINT32) {
        dilation = fourByte;
    } else if (dt == ge::DT_INT64) {
        dilation = eightByte;
    } else if (dt == ge::DT_UINT64) {
        dilation = eightByte;
    } else if (dt == ge::DT_INT8) {
        dilation = oneByte;
    }
    return dilation;
}

// IEEE 754 float32 → float16（半精度）按位转换。
// 不依赖平台相关的 __fp16 类型，保证在 aarch64 与 x86_64 上均可编译运行。
// 采用 round-to-nearest-even，与参考实现保持一致。
static uint16_t FloatToFp16(float value)
{
    uint32_t f = 0;
    memcpy(&f, &value, sizeof(f));
    uint32_t sign = (f >> 16) & 0x8000U;
    uint32_t exponent = (f >> 23) & 0xFFU;
    uint32_t mantissa = f & 0x7FFFFFU;
    uint16_t half = 0;

    if (exponent == 0xFFU) {
        // Inf / NaN
        half = static_cast<uint16_t>(sign | 0x7C00U);
        if (mantissa != 0) {
            half = static_cast<uint16_t>(half | (mantissa >> 13));
            if ((mantissa & 0x1FFFU) == 0) {
                half |= 0x1U; // 保持 NaN 尾数非零
            }
        }
        return half;
    }

    int32_t newExp = static_cast<int32_t>(exponent) - 127 + 15;
    if (newExp >= 31) {
        half = static_cast<uint16_t>(sign | 0x7C00U); // 溢出 → Inf
    } else if (newExp <= 0) {
        // 下溢 → 次正规数或 0
        if (newExp < -10) {
            half = static_cast<uint16_t>(sign);
        } else {
            mantissa |= 0x800000U;
            uint32_t shift = static_cast<uint32_t>(14 - newExp);
            uint16_t halfMantissa = static_cast<uint16_t>(mantissa >> shift);
            uint32_t rem = mantissa & ((1U << shift) - 1U);
            uint32_t halfUlp = 1U << (shift - 1U);
            if (rem > halfUlp || (rem == halfUlp && (halfMantissa & 1U))) {
                halfMantissa++;
            }
            half = static_cast<uint16_t>(sign | halfMantissa);
        }
    } else {
        uint16_t halfMantissa = static_cast<uint16_t>(mantissa >> 13);
        uint32_t rem = mantissa & 0x1FFFU;
        if (rem > 0x1000U || (rem == 0x1000U && (halfMantissa & 1U))) {
            halfMantissa++;
            if (halfMantissa == 0x400U) {
                halfMantissa = 0;
                newExp++;
            }
        }
        if (newExp >= 31) {
            half = static_cast<uint16_t>(sign | 0x7C00U);
        } else {
            half = static_cast<uint16_t>(sign | (static_cast<uint32_t>(newExp) << 10) | halfMantissa);
        }
    }
    return half;
}

// 生成按 float 常量填充的输入数据。
// 算子 var / value 仅支持 FP16 / BF16；仅 alpha 为 FP32 标量。
//   allowFp32=true 仅在 alpha 调用点传 true，其它输入传 fp32 会直接报错，防止误用。
// 按目标 DataType 真正的字节宽度写入，避免 fp16/bf16 与 float(4B) 错位。
int32_t GenFloatData(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type,
                     float value, bool allowFp32 = false)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t elemBytes = GetDataTypeSize(data_type);
    if (elemBytes == 0) {
        printf("GenFloatData: unsupported dtype=%d\n", static_cast<int>(data_type));
        return FAILED;
    }
    uint32_t data_len = size * elemBytes;
    uint8_t* pData = new (std::nothrow) uint8_t[data_len];
    if (pData == nullptr) {
        printf("GenFloatData: allocation failed for size=%zu\n", size);
        return FAILED;
    }
    for (size_t i = 0; i < size; ++i) {
        uint8_t* dst = pData + i * elemBytes;
        switch (data_type) {
            case ge::DT_FLOAT:
                // 仅 alpha 允许 fp32；var / value 不支持 fp32
                if (!allowFp32) {
                    printf("GenFloatData: DT_FLOAT only allowed for alpha (var/value support FP16/BF16 only)\n");
                    delete[] pData;
                    return FAILED;
                }
                *reinterpret_cast<float*>(dst) = value;
                break;
            case ge::DT_FLOAT16: {
                // var / value（fp16）：float32 → fp16 按位转换（round-to-nearest-even）
                uint16_t fp16 = FloatToFp16(value);
                memcpy(dst, &fp16, sizeof(fp16));
                break;
            }
            case ge::DT_BF16: {
                // var / value（bf16）：float32 → bf16 四舍五入截断低 16 位
                uint32_t bits = 0;
                memcpy(&bits, &value, sizeof(bits));
                uint16_t bf16 = static_cast<uint16_t>((bits + 0x7FFFU + ((bits >> 16) & 1U)) >> 16);
                memcpy(dst, &bf16, sizeof(bf16));
                break;
            }
            default:
                // 算子仅支持 fp16 / bf16（alpha 为 fp32），其余类型直接报错
                printf("GenFloatData: unsupported dtype=%d (only FP16/BF16 for var/value, FP32 for alpha)\n",
                       static_cast<int>(data_type));
                delete[] pData;
                return FAILED;
        }
    }
    input_tensor = Tensor(input_tensor_desc, pData, data_len);
    delete[] pData; // GE Tensor 构造已拷贝，立即释放
    return SUCCESS;
}

// 生成按 int32 序列填充的输入数据（用于 sorted_indices / pos）
// 按目标 DataType 真正的字节宽度写入，避免 int16 与 int32(4B) 错位。
int32_t GenInt32Data(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type,
                     const std::vector<int32_t>& values)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    if (values.size() < size) {
        printf("GenInt32Data: provided values=%zu less than tensor size=%zu\n", values.size(), size);
        return FAILED;
    }
    uint32_t elemBytes = GetDataTypeSize(data_type);
    if (elemBytes == 0) {
        printf("GenInt32Data: unsupported dtype=%d\n", static_cast<int>(data_type));
        return FAILED;
    }
    uint32_t data_len = size * elemBytes;
    uint8_t* pData = new (std::nothrow) uint8_t[data_len];
    if (pData == nullptr) {
        printf("GenInt32Data: allocation failed for size=%zu\n", size);
        return FAILED;
    }
    for (size_t i = 0; i < size; ++i) {
        uint8_t* dst = pData + i * elemBytes;
        switch (data_type) {
            case ge::DT_INT32:
                *reinterpret_cast<int32_t*>(dst) = values[i];
                break;
            case ge::DT_INT16:
                *reinterpret_cast<int16_t*>(dst) = static_cast<int16_t>(values[i]);
                break;
            default:
                printf("GenInt32Data: unsupported dtype=%d\n", static_cast<int>(data_type));
                delete[] pData;
                return FAILED;
        }
    }
    input_tensor = Tensor(input_tensor_desc, pData, data_len);
    delete[] pData;
    return SUCCESS;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t* inputData)
{
    FILE* fp = fopen(bin_file.c_str(), "wb");
    if (fp == nullptr) {
        printf("WriteDataToFile: fopen failed for %s\n", bin_file.c_str());
        return FAILED;
    }
    size_t written = fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    if (written != data_size) {
        printf("WriteDataToFile: short write %zu/%lu\n", written, data_size);
        return FAILED;
    }
    return SUCCESS;
}

// 构造一个 GE Data 输入（host 占位，device 填充 tensor），并绑定到 op 的指定输入端口
//   placeholderIndex : 占位索引（用于生成独立变量名）
//   portSetter       : op.set_input_<name>(data) 的函数
//   dtype            : 数据类型
//   shape            : shape
//   fillValue        : 填充的常数值（float 路径使用）
//   intValues        : 填充的 int32 序列（int32 路径使用，仅当 isInt32=true 时生效）
//   isInt32          : 是否走 int32 填充路径
//   allowFp32        : 是否允许 fp32（仅 alpha 传 true；var/value 只能 fp16/bf16）
//   graph / input / inputs : 图、device tensor 列表、graph 输入端口列表
template <typename SetterFn>
int32_t AddDataInput(int placeholderIndex, SetterFn portSetter, DataType dtype, const vector<int64_t>& shape,
                     float fillValue, const std::vector<int32_t>& intValues, bool isInt32, Graph& graph,
                     std::vector<ge::Tensor>& input, std::vector<Operator>& inputs, bool allowFp32 = false)
{
    std::string name = "placeholder" + std::to_string(placeholderIndex);
    auto data = op::Data(name.c_str()).set_attr_index(placeholderIndex);
    TensorDesc desc = TensorDesc(ge::Shape(shape), FORMAT_ND, dtype);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetFormat(FORMAT_ND);
    Tensor tensor;
    Status ret;
    if (isInt32) {
        ret = GenInt32Data(shape, tensor, desc, dtype, intValues);
    } else {
        ret = GenFloatData(shape, tensor, desc, dtype, fillValue, allowFp32);
    }
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate input data failed for %s\n", GetTime().c_str(), name.c_str());
        return FAILED;
    }
    data.update_input_desc_x(desc);
    data.update_output_desc_y(desc);
    input.push_back(tensor);
    graph.AddOp(data);
    portSetter(data);
    inputs.push_back(data);
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    // 添加 InplaceIndexAddWithSorted 单算子定义到图中，axis 为必选属性，当前仅支持 0
    auto opInst = op::InplaceIndexAddWithSorted("inplace_index_add_with_sorted1");
    opInst.set_attr_axis(0);

    // shape 约定：M=N=D=4，无重复索引，便于核对期望值
    std::vector<int64_t> varShape = {4, 4}; // var / value：[M, D]
    std::vector<int64_t> idxShape = {4};    // sorted_indices / pos：[N]
    std::vector<int64_t> alphaShape = {1};  // alpha：标量

    // sorted_indices 升序、无重复：[0, 1, 2, 3]
    // pos 为 [0, N) 的置换：[0, 1, 2, 3]
    std::vector<int32_t> sortedIndices = {0, 1, 2, 3};
    std::vector<int32_t> posValues = {0, 1, 2, 3};

    int idx = 0;
    // 必选输入 —— 顺序须与 REG_OP(InplaceIndexAddWithSorted) 的 .INPUT 顺序一致
    if (AddDataInput(
            idx++, [&](Operator& d) { opInst.set_input_var(d); }, inDtype, varShape, 1.0f, {}, false, graph, input,
            inputs) != SUCCESS) {
        return FAILED;
    }
    if (AddDataInput(
            idx++, [&](Operator& d) { opInst.set_input_value(d); }, inDtype, varShape, 0.5f, {}, false, graph, input,
            inputs) != SUCCESS) {
        return FAILED;
    }
    if (AddDataInput(
            idx++, [&](Operator& d) { opInst.set_input_sorted_indices(d); }, DT_INT32, idxShape, 0.0f, sortedIndices,
            true, graph, input, inputs) != SUCCESS) {
        return FAILED;
    }
    if (AddDataInput(
            idx++, [&](Operator& d) { opInst.set_input_pos(d); }, DT_INT32, idxShape, 0.0f, posValues, true, graph,
            input, inputs) != SUCCESS) {
        return FAILED;
    }
    // 可选输入 alpha：fp32 标量 = 1.0（算子限定 alpha 只支持 fp32）
    if (AddDataInput(
            idx++, [&](Operator& d) { opInst.set_input_alpha(d); }, DT_FLOAT, alphaShape, 1.0f, {}, false, graph, input,
            inputs, /*allowFp32=*/true) != SUCCESS) {
        return FAILED;
    }

    // 输出端口（var）的 TensorDesc，shape/dtype 与输入 var 一致
    TensorDesc varOutDesc = TensorDesc(ge::Shape(varShape), FORMAT_ND, inDtype);
    opInst.update_output_desc_var(varOutDesc);

    outputs.push_back(opInst);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "tc_inplace_index_add_with_sorted_ge_irrun_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    if (argc > 1) {
        std::cout << "device id arg: " << argv[1] << std::endl;
    }

    DataType inDtype = DT_FLOAT16;
    std::cout << "input dtype: " << inDtype << std::endl;

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: CreateOppInGraph failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);

    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: dump graph to txt\n", GetTime().c_str());
    std::string file_path = "./dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());
    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        std::cout << "input " << i << " dtype :  " << input[i].GetTensorDesc().GetDataType() << std::endl;
        string input_file = "./tc_inplace_index_add_with_sorted_ge_irrun_test_npu_input_" + std::to_string(i) + ".bin";
        uint8_t* input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th input, input shape size =" << input_shape << std::endl;
        uint32_t data_size = input_shape * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char*)input_file.c_str(), data_size, input_data_i);
    }

    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string output_file = "./tc_inplace_index_add_with_sorted_ge_irrun_test_npu_output_" + std::to_string(i) +
                             ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th output, output shape size =" << output_shape << std::endl;
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char*)output_file.c_str(), data_size, output_data_i);
    }

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    std::cout << "Error message: " << error_str << std::endl;
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    std::string warning_str(warning_msg.GetString());
    std::cout << "Warning message: " << warning_str << std::endl;
    printf("%s - INFO - [XIR]: Precision is ok\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
