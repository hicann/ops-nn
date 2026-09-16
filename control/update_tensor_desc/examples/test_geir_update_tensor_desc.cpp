
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* UpdateTensorDesc 图模式（GE IR）调用样例。
 * proto 与 canndev 保持一致（REG_OP 随 CANN 包安装于 array_ops.h），
 * 此处直接使用 array_ops.h 中的注册，避免与本地 proto 头重复定义。
 * 验证口径：y dtype 恒为 INT64，y.shape = attr(shape)，
 * 且固定槽位 y[3] = N、y[4+i] = shape[i]（N = len(shape)，即 kernel RMW 写入区）。
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
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

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

enum RunMode { RUN_MODE_S = 0, RUN_MODE_D = 1 };

struct CaseResult {
    std::string case_name;
    bool build_ok;
    bool run_ok;
    bool output_exists;
    bool verify_ok;
    int output_count;
    std::string err_msg;
};

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

    if (dt == ge::DT_FLOAT) {
        return fourByte;
    }
    if (dt == ge::DT_FLOAT16) {
        return twoByte;
    }
    if (dt == ge::DT_INT16 || dt == ge::DT_UINT16) {
        return twoByte;
    }
    if (dt == ge::DT_INT32 || dt == ge::DT_UINT32) {
        return fourByte;
    }
    if (dt == ge::DT_INT64 || dt == ge::DT_UINT64 || dt == ge::DT_DOUBLE) {
        return eightByte;
    }
    return oneByte; // int8/uint8/bool
}

// x 为占位输入（kernel 不读取其数据），按 dtype 分配缓冲并填充固定字节模式即可
int32_t GenPlaceholderData(const vector<int64_t>& shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc,
                           DataType data_type)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (size_t i = 0; i < shapes.size(); i++) {
        size *= static_cast<size_t>(shapes[i]);
    }
    size_t data_len = size * GetDataTypeSize(data_type);
    // Tensor 构造函数对入参数据做深拷贝，vector 在函数返回后自动释放，无泄漏
    vector<uint8_t> data(data_len, 1);
    input_tensor = Tensor(input_tensor_desc, data.data(), data_len);
    return SUCCESS;
}

int CreateOppInGraph(RunMode mode, DataType inDtype, const vector<int64_t>& xShape, const vector<int64_t>& attrShape,
                     std::vector<ge::Tensor>& input, std::vector<Operator>& inputs, std::vector<Operator>& outputs,
                     Graph& graph)
{
    // ---- 输入 x（Data 占位）：host 侧构造数据，S/D 模式分离 graph desc 与 real desc
    vector<int64_t> x_graph_shape = (mode == RUN_MODE_D) ? vector<int64_t>(xShape.size(), -1) : xShape;
    auto placeholder1 = op::Data("placeholder1").set_attr_index(0);
    TensorDesc placeholder1_desc_graph = TensorDesc(ge::Shape(x_graph_shape), FORMAT_ND, inDtype);
    placeholder1_desc_graph.SetPlacement(ge::kPlacementHost);
    placeholder1_desc_graph.SetFormat(FORMAT_ND);
    placeholder1.update_input_desc_x(placeholder1_desc_graph);
    placeholder1.update_output_desc_y(placeholder1_desc_graph);

    TensorDesc placeholder1_desc_real = TensorDesc(ge::Shape(xShape), FORMAT_ND, inDtype);
    placeholder1_desc_real.SetPlacement(ge::kPlacementHost);
    placeholder1_desc_real.SetFormat(FORMAT_ND);
    Tensor tensor_placeholder1;
    if (GenPlaceholderData(xShape, tensor_placeholder1, placeholder1_desc_real, inDtype) != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());
        return FAILED;
    }
    input.push_back(tensor_placeholder1);
    graph.AddOp(placeholder1);
    inputs.push_back(placeholder1);

    // ---- UpdateTensorDesc 节点：x + 必填属性 shape -> y
    auto add1 = op::UpdateTensorDesc("add1");
    add1.set_input_x(placeholder1);
    add1.set_attr_shape(attrShape);
    add1.update_input_desc_x(placeholder1_desc_graph);

    // 输出 y：dtype 恒为 INT64，shape 由属性 shape 推导（D 模式传 -1 由 infershape 推导）
    vector<int64_t> y_graph_shape = (mode == RUN_MODE_D) ? vector<int64_t>(attrShape.size(), -1) : attrShape;
    TensorDesc y_desc = TensorDesc(ge::Shape(y_graph_shape), FORMAT_ND, DT_INT64);
    add1.update_output_desc_y(y_desc);

    graph.AddOp(add1);
    outputs.push_back(add1);
    return SUCCESS;
}

// 校验：输出数量、dtype、shape、RMW 写入区（y[3] = N，y[4+i] = shape[i]）
bool VerifyOutput(const std::vector<ge::Tensor>& output, const vector<int64_t>& attrShape, std::string& err_msg)
{
    if (output.size() != 1U) {
        err_msg = "output count=" + std::to_string(output.size());
        return false;
    }
    const ge::Tensor& y = output[0];
    if (y.GetTensorDesc().GetDataType() != ge::DT_INT64) {
        err_msg = "y dtype=" + std::to_string(static_cast<int>(y.GetTensorDesc().GetDataType()));
        return false;
    }
    auto dims = y.GetTensorDesc().GetShape().GetDims();
    if (dims != vector<int64_t>(attrShape.begin(), attrShape.end())) {
        err_msg = "y shape mismatch";
        return false;
    }
    int64_t numel = 1;
    for (auto d : dims) {
        numel *= d;
    }
    if (y.GetData() == nullptr || y.GetSize() != static_cast<size_t>(numel) * sizeof(int64_t)) {
        err_msg = "y buffer size mismatch";
        return false;
    }
    const int64_t* yData = reinterpret_cast<const int64_t*>(y.GetData());
    const int64_t rank = static_cast<int64_t>(attrShape.size());
    if (yData[3] != rank) {
        err_msg = "y[3]=" + std::to_string(yData[3]) + ", expect rank=" + std::to_string(rank);
        return false;
    }
    for (int64_t i = 0; i < rank; i++) {
        if (yData[4 + i] != attrShape[static_cast<size_t>(i)]) {
            err_msg = "y[" + std::to_string(4 + i) + "]=" + std::to_string(yData[4 + i]) + ", expect " +
                      std::to_string(attrShape[static_cast<size_t>(i)]);
            return false;
        }
    }
    return true;
}

CaseResult RunOneCase(ge::Session* session, uint32_t graph_id, RunMode mode, DataType dtype,
                      const vector<int64_t>& xShape, const vector<int64_t>& attrShape, const std::string& case_name)
{
    CaseResult r;
    r.case_name = case_name;
    r.build_ok = false;
    r.run_ok = false;
    r.output_exists = false;
    r.verify_ok = false;
    r.output_count = 0;
    r.err_msg = "";

    std::string graph_name = "tc_ge_irrun_test_" + std::to_string(graph_id);
    Graph graph(graph_name.c_str());
    std::vector<ge::Tensor> input;
    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    if (CreateOppInGraph(mode, dtype, xShape, attrShape, input, inputs, outputs, graph) != SUCCESS) {
        r.err_msg = "CreateOppInGraph failed";
        return r;
    }
    graph.SetInputs(inputs).SetOutputs(outputs);

    std::map<AscendString, AscendString> graph_options = {};
    if (session->AddGraph(graph_id, graph, graph_options) != SUCCESS) {
        r.err_msg = "AddGraph failed";
        return r;
    }
    r.build_ok = true;

    std::vector<ge::Tensor> output;
    Status ret = session->RunGraph(graph_id, input, output);
    session->RemoveGraph(graph_id);
    if (ret != SUCCESS) {
        r.err_msg = "RunGraph failed, ret=" + std::to_string(ret);
        return r;
    }
    r.run_ok = true;
    r.output_count = static_cast<int>(output.size());
    r.output_exists = (output.size() > 0);
    r.verify_ok = VerifyOutput(output, attrShape, r.err_msg);
    return r;
}

void PrintReport(const std::vector<CaseResult>& results)
{
    printf("\n");
    printf("====================================================================================================\n");
    printf("| %-28s | %-8s | %-9s | %-12s | %-9s | %-24s\n", "Case", "Build", "RunGraph", "OutputExists", "Verify",
           "ErrMsg");
    printf("----------------------------------------------------------------------------------------------------\n");
    int pass_cnt = 0;
    int total = static_cast<int>(results.size());
    for (const auto& r : results) {
        bool pass = r.build_ok && r.run_ok && r.output_exists && r.verify_ok;
        if (pass)
            pass_cnt++;
        printf("| %-28s | %-8s | %-9s | %-12s | %-9s | %-24s\n", r.case_name.c_str(), r.build_ok ? "OK" : "FAIL",
               r.run_ok ? "OK" : "FAIL", r.output_exists ? "OK" : "FAIL", r.verify_ok ? "OK" : "FAIL",
               r.err_msg.empty() ? "-" : r.err_msg.c_str());
    }
    printf("====================================================================================================\n");
    printf("Summary: %d/%d passed\n", pass_cnt, total);
}

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {
        {"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}, {"ge.exec.precision_mode", "must_keep_origin_dtype"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

    // 用例矩阵（5 条）：抽样覆盖浮点 / 整型 / bool、rank1 最小 numel / rank3 / rank5、S（静态）/ D（-1 动态
    // desc）双模式
    struct CaseEntry {
        DataType dt;
        std::string dtName;
        std::vector<int64_t> attrShape;
        std::string shapeName;
        RunMode mode;
    };
    std::vector<CaseEntry> case_list = {
        {DT_FLOAT, "FP32", {4, 8, 4}, "rank3", RUN_MODE_S},      {DT_INT64, "INT64", {128}, "rank1_min", RUN_MODE_S},
        {DT_FLOAT16, "FP16", {4, 8, 4}, "rank3", RUN_MODE_D},    {DT_INT32, "INT32", {128}, "rank1_min", RUN_MODE_D},
        {DT_BOOL, "BOOL", {2, 2, 2, 2, 8}, "rank5", RUN_MODE_S},
    };

    // x 为占位输入，shape 任意
    vector<int64_t> x_shape = {2, 3};

    std::map<AscendString, AscendString> build_options = {};
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: create session failed\n", GetTime().c_str());
        ge::GEFinalize();
        return FAILED;
    }

    std::vector<CaseResult> results;
    uint32_t graph_id = 0;
    for (const auto& c : case_list) {
        std::string mode_name = (c.mode == RUN_MODE_S) ? "S" : "D";
        std::string case_name = c.dtName + "_" + c.shapeName + "_" + mode_name;
        printf("\n%s - INFO - [XIR]: ===== %s =====\n", GetTime().c_str(), case_name.c_str());
        CaseResult r = RunOneCase(session, graph_id, c.mode, c.dt, x_shape, c.attrShape, case_name);
        results.push_back(r);
        graph_id++;
    }

    PrintReport(results);

    bool all_pass = true;
    for (const auto& r : results) {
        if (!r.build_ok || !r.run_ok || !r.output_exists || !r.verify_ok) {
            all_pass = false;
        }
    }
    if (all_pass) {
        printf("\n%s - INFO - [XIR]: ALL CASES PASSED\n", GetTime().c_str());
    } else {
        printf("\n%s - ERROR - [XIR]: SOME CASES FAILED, see report above\n", GetTime().c_str());
    }

    delete session;
    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return all_pass ? SUCCESS : FAILED;
}
