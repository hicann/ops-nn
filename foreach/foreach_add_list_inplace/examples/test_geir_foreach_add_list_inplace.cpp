// test_geir for ForeachAddListInplace (GEIR graph-mode verify, dynamic tensor-list inputs).
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
#include "../op_graph/foreach_add_list_inplace_proto.h"

using namespace ge;
using std::map; using std::vector; using std::string;

static string GetTime(){ time_t t; time(&t); char b[64]; strftime(b,sizeof(b),"%Y-%m-%d %H:%M:%S,000",localtime(&t)); return b; }

static Operator MakeData(const string& name, int idx, vector<int64_t> shape, DataType dt, vector<Tensor>& input)
{
    auto d = op::Data(name.c_str()).set_attr_index(idx);
    TensorDesc desc(ge::Shape(shape), FORMAT_ND, dt);
    size_t n=1; for(auto s:shape) n*=s;
    if (dt == DT_FLOAT) {
        float* p=new(std::nothrow) float[n]; for(size_t i=0;i<n;++i)p[i]=1.0f;
        Tensor t(desc,(uint8_t*)p,n*sizeof(float)); input.push_back(t); delete[] p;
    }
    d.update_input_desc_x(desc); d.update_output_desc_y(desc);
    return d;
}

int CreateOppInGraph(vector<Tensor>& input, vector<Operator>& inputs, vector<Operator>& outputs, Graph& graph)
{
    auto op1 = op::ForeachAddListInplace("foreachAddListInplace1");
    const int N = 2;                         // 2 tensors per list
    vector<int64_t> tShape = {4};
    int idx = 0;
    op1.create_dynamic_input_x1(N);
    for (int i=0;i<N;++i){ auto d=MakeData("x1_"+std::to_string(i),idx++,tShape,DT_FLOAT,input); graph.AddOp(d); inputs.push_back(d); op1.set_dynamic_input_x1(i,d); }
    op1.create_dynamic_input_x2(N);
    for (int i=0;i<N;++i){ auto d=MakeData("x2_"+std::to_string(i),idx++,tShape,DT_FLOAT,input); graph.AddOp(d); inputs.push_back(d); op1.set_dynamic_input_x2(i,d); }
    auto dA=MakeData("alpha",idx++,{1},DT_FLOAT,input); graph.AddOp(dA); inputs.push_back(dA); op1.set_input_alpha(dA);
    outputs.push_back(op1);                  // inplace: op as graph output
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    Graph graph("tc_ge_irrun_test");
    vector<ge::Tensor> input;
    map<AscendString, AscendString> global_options = {{"ge.exec.deviceId","0"},{"ge.graphRunMode","1"}};
    if (ge::GEInitialize(global_options)!=SUCCESS){ printf("GEInit failed\n"); return FAILED; }
    printf("%s - INFO - [XIR]: Initialize ge success\n", GetTime().c_str());
    vector<Operator> inputs{}, outputs{};
    if (CreateOppInGraph(input,inputs,outputs,graph)!=SUCCESS){ printf("create failed\n"); return FAILED; }
    if(!inputs.empty()&&!outputs.empty()) graph.SetInputs(inputs).SetOutputs(outputs);
    map<AscendString, AscendString> opts={};
    Session* session=new Session(opts);
    if(session==nullptr){ printf("session null\n"); return FAILED; }
    uint32_t gid=0; session->AddGraph(gid,graph,opts);
    printf("%s - INFO - [XIR]: Session add graph success\n", GetTime().c_str());
    vector<ge::Tensor> output;
    if(session->RunGraph(gid,input,output)!=SUCCESS){ printf("%s - INFO - [XIR]: Run graph failed\n",GetTime().c_str()); delete session; GEFinalize(); return FAILED; }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());
    delete session; ge::GEFinalize();
    return SUCCESS;
}
