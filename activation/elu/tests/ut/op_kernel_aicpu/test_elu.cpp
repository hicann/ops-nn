#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "utils/aicpu_test_utils.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "cpu_types.h"
#undef private
#undef protected

using namespace std;
using namespace aicpu;

class TEST_ELU_UT : public testing::Test {};
TEST_F(TEST_ELU_UT, TestElu_double_comparewith_tf) {
  vector<int64_t> shapes{2, 3};
  double input[6] =
    {-4.951530312134153, 0.6762052328103874, 5.362240961653322, -1.3938825500140126, -3.1986461909792396, 4.301564834639343};
  double output[6] = {0};

  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "Elu", "Elu")
    .Input({"x", DT_DOUBLE, shapes, input})
    .Output({"y", DT_DOUBLE, shapes, output})
    .Attr("alpha", 1.0)
    .Attr("scale", 1.0)
    .Attr("input_scale", 1.0);

  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  double output_exp[6] =
    {-0.9929274226076533, 0.6762052328103874, 5.362240961653322, -0.7518898678182324, -0.9591825744108585, 4.301564834639343};
  bool compare = CompareResult(output, output_exp, 6);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_ELU_UT, TestElu_float32_comparewith_pytorch) {
  vector<int64_t> shapes{2, 3};
  float input[6] = {4.9935164, 8.458385, 0.39824128, 0.9179628, -4.5891247, -4.2628508};
  float output[6] = {0};

  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "Elu", "Elu")
    .Input({"x", DT_FLOAT, shapes, input})
    .Output({"y", DT_FLOAT, shapes, output})
    .Attr("alpha", 0.2)
    .Attr("scale", 1.0)
    .Attr("input_scale", 1.0);

  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  float output_exp[6] = {4.9935164, 8.458385, 0.39824128, 0.9179628, -0.19796765, -0.1971836};
  bool compare = CompareResult(output, output_exp, 6);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_ELU_UT, TestElu_attr_not_default_with_bf16) {
  Eigen::bfloat16 input[2] = {Eigen::bfloat16(-2), Eigen::bfloat16(1)};
  Eigen::bfloat16 output[2];
  vector<int64_t> shape = {2};

  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "Elu", "Elu")
    .Input({"x", DT_BFLOAT16, shape, input})
    .Output({"y", DT_BFLOAT16, shape, output})
    .Attr("alpha", 0.1)
    .Attr("scale", 0.2)
    .Attr("input_scale", 0.3);

  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_ELU_UT, TestElu_empty_tensor) {
  double input[2] = {-2, -1};
  double output[2];
  vector<int64_t> shape = {0};

  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "Elu", "Elu")
    .Input({"x", DT_DOUBLE, shape, input})
    .Output({"y", DT_DOUBLE, shape, output})
    .Attr("alpha", 0.1)
    .Attr("scale", 0.2)
    .Attr("input_scale", 0.3);

  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_ELU_UT, TestElu_not_support_int8) {
  int8_t input[2] = {-2, 1};
  int8_t output[2] = {0};
  vector<int64_t> shape = {2};

  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
  NodeDefBuilder(node_def.get(), "Elu", "Elu")
    .Input({"x", DT_INT8, shape, input})
    .Output({"y", DT_INT8, shape, output})
    .Attr("alpha", 0.1)
    .Attr("scale", 0.2)
    .Attr("input_scale", 0.3);

  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}