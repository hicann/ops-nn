// test_geir for ForeachBinaryOp (GEIR graph-mode verify; 2 dynamic input lists + dynamic output + op_code attr).
// ForeachBinaryOp is a fused graph-internal op (no aclnn entry); it is exercised through the GE IR graph path.
// y[i] = x1[i] <op> x2[i], where <op> is selected by attr op_code: 0=add, 1=sub, 2=mul, 3=div.
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
#include "../../op_graph/foreach_binary_op_proto.h"

using namespace ge;
using std::map; using std::vector; using std::string;
static string GetTime(){ time_t t; time(&t); char b[64]; strftime(b,sizeof(b),"%Y-%m-%d %H:%M:%S,000",localtime(&t)); return b; }

static Operator MkF32(const string& name, int idx, vector<int64_t> shape, vector<Tensor>& input)
{
    auto d = op::Data(name.c_str()).set_attr_index(idx);
    TensorDesc desc(ge::Shape(shape), FORMAT_ND, DT_FLOAT);
    size_t n=1; for(auto s:shape) n*=s;
    float* p=new(std::nothrow) float[n]; for(size_t i=0;i<n;++i)p[i]=2.0f;
    Tensor t(desc,(uint8_t*)p,n*sizeof(float)); input.push_back(t); delete[] p;
    d.update_input_desc_x(desc); d.update_output_desc_y(desc);
    return d;
}

int CreateOppInGraph(vector<Tensor>& input, vector<Operator>& inputs, vector<Operator>& outputs, Graph& graph)
{
    auto op1 = op::ForeachBinaryOp("foreachBinaryOp1");
    const int N = 2;                          // 2 tensors per list
    vector<int64_t> tShape = {256};
    int idx = 0;
    op1.set_attr_op_code(0);                  // 0=add (1=sub, 2=mul, 3=div)
    op1.create_dynamic_input_x1(N);
    for(int i=0;i<N;++i){ auto d=MkF32("x1_"+std::to_string(i),idx++,tShape,input); graph.AddOp(d); inputs.push_back(d); op1.set_dynamic_input_x1(i,d); }
    op1.create_dynamic_input_x2(N);
    for(int i=0;i<N;++i){ auto d=MkF32("x2_"+std::to_string(i),idx++,tShape,input); graph.AddOp(d); inputs.push_back(d); op1.set_dynamic_input_x2(i,d); }
    op1.create_dynamic_output_y(N);
    for(int i=0;i<N;++i){ TensorDesc yd(ge::Shape(tShape),FORMAT_ND,DT_FLOAT); op1.update_dynamic_output_desc_y(i,yd); }
    outputs.push_back(op1);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    Graph graph("tc_ge_irrun_test");
    vector<ge::Tensor> input;
    map<AscendString, AscendString> go = {{"ge.exec.deviceId","0"},{"ge.graphRunMode","1"}};
    if (ge::GEInitialize(go)!=SUCCESS){ printf("GEInit failed\n"); return FAILED; }
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
