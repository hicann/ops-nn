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
 * \file test_geir_add_example_aicpu.cpp
 * \brief AddExampleAicpu算子的GEIR测试程序
 * 
 * 本文件实现了使用GEIR(Graph Engine Intermediate Representation)API测试AddExampleAicpu算子的功能。
 * GEIR是华为Ascend芯片的图构建和执行框架，提供了一套C++ API来构建计算图并执行推理。
 * 
 * 测试流程：
 * 1. 初始化GE运行环境
 * 2. 构建计算图，包含Data节点和AddExampleAicpu算子节点
 * 3. 创建Session并添加计算图
 * 4. 生成输入数据并运行计算图
 * 5. 获取输出数据并验证结果
 * 6. 将输入输出数据导出为二进制文件用于分析
 * 7. 释放资源并清理环境
 * 
 * 使用方法：
 * ./test_geir_add_example_aicpu
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

#include "experiment_ops.h"
#include "nn_other.h"
#include "../op_graph/add_example_aicpu_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;
// 宏定义：添加动态输入节点到计算图
// 该宏为Data算子创建输入占位符，并生成全1的测试数据
// 参数说明：
//   - intputIndex: 输入索引，用于创建唯一的变量名
//   - intputName: 算子的输入端口名称
//   - intputDtype: 输入张量的数据类型
//   - inputShape: 输入张量的形状
#define ADD_INPUT(intputIndex, intputName, intputDtype, inputShape)                          \
    vector<int64_t> placeholder##intputIndex##_shape = inputShape;                           \
    auto placeholder##intputIndex = op::Data("placeholder" + intputIndex).set_attr_index(0); \
    TensorDesc placeholder##intputIndex##_desc =                                             \
        TensorDesc(ge::Shape(placeholder##intputIndex##_shape), FORMAT_ND, intputDtype);     \
    placeholder##intputIndex##_desc.SetPlacement(ge::kPlacementHost);                        \
    placeholder##intputIndex##_desc.SetFormat(FORMAT_ND);                                    \
    Tensor tensor_placeholder##intputIndex;                                                  \
    ret = GenOnesData(placeholder##intputIndex##_shape,                                      \
        tensor_placeholder##intputIndex,                                                     \
        placeholder##intputIndex##_desc,                                                     \
        intputDtype,                                                                         \
        2);                                                                                  \
    if (ret != SUCCESS) {                                                                    \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());       \
        return FAILED;                                                                       \
    }                                                                                        \
    placeholder##intputIndex.update_input_desc_x(placeholder##intputIndex##_desc);           \
    input.push_back(tensor_placeholder##intputIndex);                                        \
    graph.AddOp(placeholder##intputIndex);                                                   \
    add1.set_input_##intputName(placeholder##intputIndex);                                   \
    inputs.push_back(placeholder##intputIndex);

// 宏定义：添加常量输入节点到计算图
// 该宏为Const算子创建常量输入，并生成全1的测试数据
// 与ADD_INPUT的区别在于：Const算子直接在图中存储常量值，不需要在运行时提供输入
// 参数说明：
//   - intputIndex: 输入索引，用于创建唯一的变量名
//   - intputName: 算子的输入端口名称
//   - intputDtype: 输入张量的数据类型
//   - inputShape: 输入张量的形状
#define ADD_CONST_INPUT(intputIndex, intputName, intputDtype, inputShape)                    \
    vector<int64_t> placeholder##intputIndex##_shape = inputShape;                           \
    auto placeholder##intputIndex = op::Const("placeholder" + intputIndex);                  \
    TensorDesc placeholder##intputIndex##_desc =                                             \
        TensorDesc(ge::Shape(placeholder##intputIndex##_shape), FORMAT_ND, intputDtype);     \
    placeholder##intputIndex##_desc.SetPlacement(ge::kPlacementHost);                        \
    placeholder##intputIndex##_desc.SetFormat(FORMAT_ND);                                    \
    Tensor tensor_placeholder##intputIndex;                                                  \
    ret = GenOnesData(placeholder##intputIndex##_shape,                                      \
        tensor_placeholder##intputIndex,                                                     \
        placeholder##intputIndex##_desc,                                                     \
        intputDtype,                                                                         \
        2);                                                                                  \
    if (ret != SUCCESS) {                                                                    \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());       \
        return FAILED;                                                                       \
    }                                                                                        \
    placeholder##intputIndex.SetAttr("value", tensor_placeholder##intputIndex);              \
    placeholder##intputIndex.update_output_desc_y(placeholder##intputIndex##_desc);          \
    graph.AddOp(placeholder##intputIndex);                                                   \
    add1.set_input_##intputName(placeholder##intputIndex);                                   \
    add1.update_input_desc_##intputName(placeholder##intputIndex##_desc);                    \
    inputs.push_back(placeholder##intputIndex);

// 宏定义：添加输出描述到计算图
// 该宏为算子的输出端口设置张量描述符，指定输出张量的形状和数据类型
// 参数说明：
//   - outputIndex: 输出索引，用于创建唯一的变量名
//   - outputName: 算子的输出端口名称
//   - outputDtype: 输出张量的数据类型
//   - outputShape: 输出张量的形状
#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                        \
    TensorDesc outputName##outputIndex##_desc =                                              \
        TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype);                          \
    add1.update_output_desc_##outputName(outputName##outputIndex##_desc);             

/*!
 * \brief 获取当前时间的格式化字符串
 * 
 * 该函数返回当前系统时间，格式为"YYYY-MM-DD HH:MM:SS,000"。
 * 用于日志输出，方便追踪程序执行的时间点。
 * 
 * @return 格式化的时间字符串
 */
string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

/*!
 * \brief 获取数据类型对应的字节大小
 * 
 * 该函数根据GE的数据类型枚举值返回对应的字节大小，用于计算张量的内存占用。
 * 支持的数据类型包括：
 * - 浮点类型：FLOAT(4字节), FLOAT16(2字节), BF16(2字节)
 * - 整型：INT8(1字节), INT16(2字节), INT32(4字节), INT64(8字节)
 * - 无符号整型：UINT16(2字节), UINT32(4字节), UINT64(8字节)
 * 
 * @param dt GE数据类型枚举值
 * @return 数据类型的字节大小
 */
uint32_t GetDataTypeSize(DataType dt)
{
    uint32_t dilation = 1;
    uint32_t oneByte = 1;
    uint32_t twoByte = 2;
    uint32_t fourByte = 4;
    uint32_t eightByte = 8;

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

/*!
 * \brief 生成全为指定值的Float32类型测试数据
 * 
 * 该函数创建一个Float32类型的张量，所有元素的值都为指定的value。
 * 主要用于生成测试输入数据。
 * 
 * @param shapes 输出张量的形状
 * @param input_tensor 输出张量对象
 * @param input_tensor_desc 输出张量的描述符
 * @param value 所有元素的初始值
 * @return SUCCESS表示成功，FAILED表示失败
 */
int32_t GenOnesDataFloat32(vector<int64_t> shapes, Tensor &input_tensor, TensorDesc &input_tensor_desc, float value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t byteSizeFloat32 = 4;
    uint32_t data_len = size * byteSizeFloat32;
    float *pData = new (std::nothrow) float[size];

    for (size_t i = 0; i < size; ++i) {
        *(pData + i) = value;
    }
    input_tensor = Tensor(input_tensor_desc, (uint8_t *)pData, data_len);
    return SUCCESS;
}

/*!
 * \brief 生成全为指定值的测试数据(通用版本)
 * 
 * 该函数创建一个指定类型的张量，所有元素的值都为指定的value。
 * 支持多种数据类型，通过GetDataTypeSize函数获取每个数据类型的字节大小。
 * 
 * @param shapes 输出张量的形状
 * @param input_tensor 输出张量对象
 * @param input_tensor_desc 输出张量的描述符
 * @param data_type 张量的数据类型
 * @param value 所有元素的初始值(转换为对应数据类型)
 * @return SUCCESS表示成功，FAILED表示失败
 */
int32_t GenOnesData(
    vector<int64_t> shapes, Tensor &input_tensor, TensorDesc &input_tensor_desc, DataType data_type, int value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t data_len = size * GetDataTypeSize(data_type);
    int32_t *pData = new (std::nothrow) int32_t[data_len];
    for (uint32_t i = 0; i < size; ++i) {
        *(pData + i) = value;
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t *>(pData), data_len);
    return SUCCESS;
}

/*!
 * \brief 将数据写入二进制文件
 * 
 * 该函数将内存中的数据写入指定的二进制文件，用于保存输入输出数据以便后续分析。
 * 
 * @param bin_file 输出文件路径
 * @param data_size 数据的字节大小
 * @param inputData 指向数据的内存指针
 * @return SUCCESS表示成功，FAILED表示失败
 */
int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t *inputData)
{
    FILE *fp;
    fp = fopen(bin_file.c_str(), "w");
    fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return SUCCESS;
}

/*!
 * \brief 在计算图中构建AddExampleAicpu算子
 * 
 * 该函数创建一个包含AddExampleAicpu算子的计算图，包括：
 * 1. 创建AddExampleAicpu算子节点
 * 2. 创建两个输入张量占位符(Data节点)，形状为[32,4,4,4]
 * 3. 设置输出张量的描述符
 * 4. 将算子节点添加到图的输出列表
 * 
 * @param inDtype 输入张量的数据类型
 * @param input 输入张量数据的容器引用
 * @param inputs 图输入节点的容器引用
 * @param outputs 图输出节点的容器引用
 * @param graph 计算图对象的引用
 * @return SUCCESS表示成功，FAILED表示失败
 */
int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor> &input, std::vector<Operator> &inputs,
    std::vector<Operator> &outputs, Graph &graph)
{
    Status ret = SUCCESS;
    
    // 创建AddExampleAicpu算子节点，节点名称为"add1"
    auto add1 = op::AddExampleAicpu("add1");
    
    // 定义输入张量的形状为[32,4,4,4]
    std::vector<int64_t> xShape = {32,4,4,4};
    
    // 创建两个输入张量占位符，每个张量填充值为2
    ADD_INPUT(1, x1, inDtype, xShape);
    ADD_INPUT(2, x2, inDtype, xShape);
    
    // 设置输出张量的描述符，形状与输入相同
    ADD_OUTPUT(1, y, inDtype, xShape);

    // 将算子节点添加到图的输出列表
    outputs.push_back(add1);
    
    return SUCCESS;
}

/*!
 * \brief 主函数：GEIR测试程序的入口点
 * 
 * 该函数执行完整的GEIR测试流程：
 * 1. 初始化GE运行环境，指定设备ID和运行模式
 * 2. 创建计算图，添加AddExampleAicpu算子和输入节点
 * 3. 创建Session，将计算图添加到Session
 * 4. 运行计算图，获取输出结果
 * 5. 将输入输出数据导出为二进制文件
 * 6. 打印错误和警告信息
 * 7. 清理资源并退出
 * 
 * @return SUCCESS表示成功，FAILED表示失败
 */
int main(int argc, char *argv[])
{
    // 定义计算图名称
    const char *graph_name = "tc_ge_irrun_test";
    
    // 创建计算图对象
    Graph graph(graph_name);
    
    // 创建输入张量容器
    std::vector<ge::Tensor> input;

    // 初始化GE运行环境
    // ge.exec.deviceId: 指定使用的NPU设备ID为0
    // ge.graphRunMode: 1表示运行模式(0为编译模式)
    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

    // 创建输入输出算子节点容器
    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    std::cout << argv[1] << std::endl;
    char *endptr;
    DataType inDtype = DT_FLOAT;
    std::cout << inDtype << std::endl;

    // 在计算图中构建AddExampleAicpu算子
    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }

    // 设置计算图的输入输出节点
    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    // 创建Session，用于管理和执行计算图
    std::map<AscendString, AscendString> build_options = {};
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    ge::Session *session = new Session(build_options);

    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
    
    // 将计算图添加到Session
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());
    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);

    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());
    
    // 导出计算图到文本文件，用于调试和分析
    printf("%s - INFO - [XIR]: dump graph to txt\n", GetTime().c_str());
    std::string file_path = "./dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());
    
    // 运行计算图
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

    // 将输入数据导出为二进制文件
    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        std::cout << "input " << i << " dtype :  " << input[i].GetTensorDesc().GetDataType() << std::endl;
        string input_file = "./tc_ge_irrun_test_0008_npu_input_" + std::to_string(i) + ".bin";
        uint8_t *input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th input, input shape size =" << input_shape << std::endl;
        uint32_t data_size = input_shape * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char *)input_file.c_str(), data_size, input_data_i);
    }

    // 将输出数据导出为二进制文件
    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string output_file = "./tc_ge_irrun_test_0008_npu_output_" + std::to_string(i) + ".bin";
        uint8_t *output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th output, output shape size =" << output_shape << std::endl;
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char *)output_file.c_str(), data_size, output_data_i);
    }

    // 打印错误和警告信息
    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    std::cout << "Error message: " << error_str << std::endl;
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    std::string warning_str(warning_msg.GetString());
    std::cout << "Warning message: " << warning_str << std::endl;
    printf("%s - INFO - [XIR]: Precision is ok\n", GetTime().c_str());
    
    // 清理GE运行环境
    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
