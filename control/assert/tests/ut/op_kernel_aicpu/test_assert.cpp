#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_ASSERT_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, sum)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();      \
  NodeDefBuilder(node_def.get(), "Assert", "Assert")                    \
      .Input({"input_condition", data_types[0], shapes[0], datas[0]})   \
      .Input({"input_data", data_types[1], shapes[1], datas[1]})        \
      .Attr("summarize", sum);

#define ADD_CASE(base_type, aicpu_type, condition, shapes, num, sum_attr)    \
  TEST_F(TEST_ASSERT_UT, Test_##aicpu_type##condition##num##sum_attr) {      \
    size_t input0 = shapes[0].size();                                        \
    size_t dims = shapes[1].size();                                          \
    vector<DataType> data_types = {DT_BOOL, aicpu_type};                     \
    base_type input[num];                                                    \
    SetRandomValue<base_type>(input, num);                                   \
    vector<void *> datas = {(void *)&condition, (void *)input};              \
    CREATE_NODEDEF(shapes, data_types, datas, sum_attr);                     \
    uint32_t re = KERNEL_STATUS_OK;                                          \
    if (!condition) { re = KERNEL_STATUS_PARAM_INVALID; }                    \
    if (input0 != 0) { re = KERNEL_STATUS_PARAM_INVALID; }                   \
    RUN_KERNEL(node_def, HOST, re);                                          \
  }

TEST_F(TEST_ASSERT_UT, ExpInputNull) {
  vector<DataType> data_types = {DT_BOOL, DT_INT32};
  bool input0 = false;
  vector<vector<int64_t>> shapes = {{}, {2, 11}};
  vector<void *> datas = {(void *)&input0, (void *)nullptr};
  CREATE_NODEDEF(shapes, data_types, datas, 3);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ASSERT_UT, InputString) {
  vector<DataType> data_types = {DT_BOOL, DT_STRING};
  bool input0 = false;
  string input1[4] = {"input","is","string","type"};
  vector<vector<int64_t>> shapes = {{}, {2, 2}};
  vector<void *> datas = {(void *)&input0, (void *)input1};
  CREATE_NODEDEF(shapes, data_types, datas, 3);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

bool input_condition_true = true;
bool input_condition_false = false;
vector<vector<int64_t>> input_shapes1 = {{}, {4}};
ADD_CASE(int8_t, DT_INT8, input_condition_true, input_shapes1, 4, 3)
ADD_CASE(int8_t, DT_INT8, input_condition_false, input_shapes1, 4, 3)

ADD_CASE(bool, DT_BOOL, input_condition_true, input_shapes1, 4, 3)
ADD_CASE(bool, DT_BOOL, input_condition_false, input_shapes1, 4, 3)

vector<vector<int64_t>> input_shapes2 = {{}, {2, 2}};
ADD_CASE(uint8_t, DT_UINT8, input_condition_true, input_shapes2, 4, 4)
ADD_CASE(uint8_t, DT_UINT8, input_condition_false, input_shapes2, 4, 4)

vector<vector<int64_t>> input_shapes3 = {{}, {1, 3, 4}};
ADD_CASE(int32_t, DT_INT32, input_condition_true, input_shapes3, 12, 20)
ADD_CASE(int32_t, DT_INT32, input_condition_false, input_shapes3, 12, 20)

vector<vector<int64_t>> input_shapes4 = {{}, {2, 2, 2, 3}};
ADD_CASE(int64_t, DT_INT64, input_condition_true, input_shapes4, 24, 20)
ADD_CASE(int64_t, DT_INT64, input_condition_false, input_shapes4, 24, 20)

vector<vector<int64_t>> input_shapes5 = {{}, {4}};
ADD_CASE(float, DT_FLOAT, input_condition_true, input_shapes5, 4, 4)
ADD_CASE(float, DT_FLOAT, input_condition_false, input_shapes5, 4, 4)

ADD_CASE(Eigen::half, DT_FLOAT16, input_condition_true, input_shapes5, 4, 2)
ADD_CASE(Eigen::half, DT_FLOAT16, input_condition_false, input_shapes5, 4, 2)

vector<vector<int64_t>> input_shapes6 = {{}, {4, 1}};
ADD_CASE(double, DT_DOUBLE, input_condition_true, input_shapes6, 4, 4)
ADD_CASE(double, DT_DOUBLE, input_condition_false, input_shapes6, 4, 4)

vector<vector<int64_t>> input_shapes7 = {{}, {2, 2, 2}};
ADD_CASE(uint32_t, DT_UINT32, input_condition_true, input_shapes7, 8, 10)
ADD_CASE(uint32_t, DT_UINT32, input_condition_false, input_shapes7, 8, 10)

vector<vector<int64_t>> input_shapes8 = {{}, {2, 2, 1, 4}};
ADD_CASE(uint64_t, DT_UINT64, input_condition_true, input_shapes8, 16, 20)
ADD_CASE(uint64_t, DT_UINT64, input_condition_false, input_shapes8, 16, 20)

ADD_CASE(int16_t, DT_INT16, input_condition_true, input_shapes8, 16, 16)
ADD_CASE(int16_t, DT_INT16, input_condition_false, input_shapes8, 16, 16)

vector<vector<int64_t>> input_shapes9 = {{}, {2, 2, 1, 4}};
ADD_CASE(uint16_t, DT_UINT16, input_condition_true, input_shapes9, 16, 10)
ADD_CASE(uint16_t, DT_UINT16, input_condition_false, input_shapes9, 16, 10)

vector<vector<int64_t>> input_shapes10 = {{1}, {2, 3}};
ADD_CASE(uint16_t, DT_UINT16, input_condition_true, input_shapes10, 6, 6)
ADD_CASE(uint16_t, DT_UINT16, input_condition_false, input_shapes10, 6, 6)