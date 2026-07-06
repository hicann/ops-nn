// test_geir for ScatterDiv (GEIR graph-mode verification). Style follows the lamb test_geir templates.
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <new>
#include <ctime>
#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "ge_ir_build.h"
#include "array_ops.h"
#include "../op_graph/scatter_div_proto.h"

using namespace ge;
using std::map;
using std::string;
using std::vector;

static string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

static int32_t GenFloatOnes(vector<int64_t> shape, Tensor& t, TensorDesc& desc, float v)
{
    desc.SetRealDimCnt(shape.size());
    size_t n = 1;
    for (auto d : shape)
        n *= d;
    float* p = new (std::nothrow) float[n];
    for (size_t i = 0; i < n; ++i)
        p[i] = v;
    t = Tensor(desc, (uint8_t*)p, n * sizeof(float));
    delete[] p;
    return SUCCESS;
}

static int32_t GenInt32Idx(vector<int64_t> shape, Tensor& t, TensorDesc& desc, int32_t mod)
{
    desc.SetRealDimCnt(shape.size());
    size_t n = 1;
    for (auto d : shape)
        n *= d;
    int32_t* p = new (std::nothrow) int32_t[n];
    for (size_t i = 0; i < n; ++i)
        p[i] = (int32_t)(i % mod); // valid indices in [0, mod)
    t = Tensor(desc, (uint8_t*)p, n * sizeof(int32_t));
    delete[] p;
    return SUCCESS;
}

int CreateOppInGraph(vector<Tensor>& input, vector<Operator>& inputs, vector<Operator>& outputs, Graph& graph)
{
    auto op1 = op::ScatterDiv("ScatterDiv1");
    vector<int64_t> varShape = {16, 4};
    vector<int64_t> idxShape = {4};
    vector<int64_t> updShape = {4, 4}; // indices.shape + var.shape[1:]

    // var (float32)
    auto dVar = op::Data("data_var").set_attr_index(0);
    TensorDesc descVar(ge::Shape(varShape), FORMAT_ND, DT_FLOAT);
    descVar.SetPlacement(ge::kPlacementHost);
    Tensor tVar;
    GenFloatOnes(varShape, tVar, descVar, 1.0f);
    dVar.update_input_desc_x(descVar);
    dVar.update_output_desc_y(descVar);
    input.push_back(tVar);
    graph.AddOp(dVar);
    inputs.push_back(dVar);
    op1.set_input_var(dVar);

    // indices (int32)
    auto dIdx = op::Data("data_indices").set_attr_index(1);
    TensorDesc descIdx(ge::Shape(idxShape), FORMAT_ND, DT_INT32);
    descIdx.SetPlacement(ge::kPlacementHost);
    Tensor tIdx;
    GenInt32Idx(idxShape, tIdx, descIdx, 16);
    dIdx.update_input_desc_x(descIdx);
    dIdx.update_output_desc_y(descIdx);
    input.push_back(tIdx);
    graph.AddOp(dIdx);
    inputs.push_back(dIdx);
    op1.set_input_indices(dIdx);

    // updates (float32)
    auto dUpd = op::Data("data_updates").set_attr_index(2);
    TensorDesc descUpd(ge::Shape(updShape), FORMAT_ND, DT_FLOAT);
    descUpd.SetPlacement(ge::kPlacementHost);
    Tensor tUpd;
    GenFloatOnes(updShape, tUpd, descUpd, 2.0f);
    dUpd.update_input_desc_x(descUpd);
    dUpd.update_output_desc_y(descUpd);
    input.push_back(tUpd);
    graph.AddOp(dUpd);
    inputs.push_back(dUpd);
    op1.set_input_updates(dUpd);

    TensorDesc outDesc(ge::Shape(varShape), FORMAT_ND, DT_FLOAT);
    op1.update_output_desc_var(outDesc);
    outputs.push_back(op1);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    Graph graph("tc_ge_irrun_test");
    vector<ge::Tensor> input;
    printf("%s - INFO - [XIR]: Start to initialize ge\n", GetTime().c_str());
    map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(global_options) != SUCCESS) {
        printf("GEInit failed\n");
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge success\n", GetTime().c_str());

    vector<Operator> inputs{}, outputs{};
    if (CreateOppInGraph(input, inputs, outputs, graph) != SUCCESS) {
        printf("create graph failed\n");
        return FAILED;
    }
    if (!inputs.empty() && !outputs.empty())
        graph.SetInputs(inputs).SetOutputs(outputs);

    map<AscendString, AscendString> build_options = {};
    Session* session = new Session(build_options);
    if (session == nullptr) {
        printf("session null\n");
        return FAILED;
    }
    map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    session->AddGraph(graph_id, graph, graph_options);
    printf("%s - INFO - [XIR]: Session add graph success\n", GetTime().c_str());

    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    vector<ge::Tensor> output;
    if (session->RunGraph(graph_id, input, output) != SUCCESS) {
        printf("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());
    delete session;
    ge::GEFinalize();
    return SUCCESS;
}
