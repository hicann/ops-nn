#include <array>
#include <vector>
#include "gtest/gtest.h"

#include <unistd.h>

using namespace std;

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"

// #include "increDebug.h"
#include "../data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

#include "kernel_tiling/kernel_tiling.h"
#include "test_masked_softmax_with_relposbias_tiling.h"
// #include "kernel_operator.h"
using namespace std;

extern "C" __global__ __aicore__ void masked_softmax_with_rel_pos_bias(GM_ADDR x, GM_ADDR atten_mask, GM_ADDR bias,
                                                                       GM_ADDR y, GM_ADDR work_space, GM_ADDR tiling);

class masked_softmax_with_relposbias_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "masked_softmax_with_relposbias_test SetUp\n" << endl;
  }
  static void TearDownTestCase() {
    cout << "masked_softmax_with_relposbias_test TearDown\n" << endl;
  }
};

void run_masked_softmax_with_relposbias_case(uint32_t BS, uint32_t W, uint32_t N, uint32_t S1,
                   uint32_t S2, std::string& dtype, uint32_t dtypeSize,
                   uint32_t tiling_key, uint32_t shape_index) {
  system(
      "cp -r "
      "../../../../../../../ops/built-in/tests/ut/fast_op_test/masked_softmax_with_relposbias/"
      "masked_softmax_with_relposbias_data ./ && chmod -R 755 ./masked_softmax_with_relposbias_data/");
  system("ls -lh ./masked_softmax_with_relposbias_data/");
  system("cd ./masked_softmax_with_relposbias_data/ && rm -rf ./*.bin");

  std::string gen_data_cmd =
      "cd ./masked_softmax_with_relposbias_data/ && python3 gen_data.py " +
      std::to_string(BS) + std::string(" ") + std::to_string(W) +
      std::string(" ") + std::to_string(N) + std::string(" ") +
      std::to_string(S1) + std::string(" ") + std::to_string(S2) +
      std::string(" ") + dtype + std::string(" ") + std::to_string(tiling_key);
  system(gen_data_cmd.c_str());

  std::string gen_tiling_cmd =
      "cd ./masked_softmax_with_relposbias_data/ && python3 gen_tiling.py " + std::string("case") +
      std::to_string(tiling_key) + std::to_string(shape_index);
  system(gen_tiling_cmd.c_str());

  system("ls -lh ./masked_softmax_with_relposbias_data/");

  size_t xSize = BS * W * N * S1 * S2 * dtypeSize;
  size_t attenMaskSize = W * S1 * S2 * dtypeSize;
  size_t biasSize = N * S1 * S2 * dtypeSize;
  size_t ySize = BS * W * N * S1 * S2 * dtypeSize;
  size_t expect_ySize = BS * W * N * S1 * S2 * dtypeSize;

  uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
  uint8_t* atten_mask = (uint8_t*)AscendC::GmAlloc(attenMaskSize);
  uint8_t* bias = (uint8_t*)AscendC::GmAlloc(biasSize);
  uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
  uint8_t* expect_y = (uint8_t*)AscendC::GmAlloc(expect_ySize);

  uint8_t* workSpace = nullptr;
  size_t tilingSize = sizeof(MaskedSoftmaxWithRelPosBiasTilingData);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
  MaskedSoftmaxWithRelPosBiasTilingData* tilingDatafromBin =
      reinterpret_cast<MaskedSoftmaxWithRelPosBiasTilingData*>(tiling);
  string curPath = "./";
  ReadFile(curPath + "masked_softmax_with_relposbias_data/x.bin", xSize, x, xSize);
  ReadFile(curPath + "masked_softmax_with_relposbias_data/atten_mask.bin", attenMaskSize, atten_mask, attenMaskSize);
  ReadFile(curPath + "masked_softmax_with_relposbias_data/bias.bin", biasSize, bias, biasSize);
  ReadFile(curPath + "masked_softmax_with_relposbias_data/tiling.bin", tilingSize, tilingDatafromBin, tilingSize);
  ReadFile(curPath + "masked_softmax_with_relposbias_data/golden_y.bin", expect_ySize, expect_y, expect_ySize);
  ICPU_SET_TILING_KEY(tiling_key);

  uint32_t blockDim = 24;
  ICPU_RUN_KF(masked_softmax_with_rel_pos_bias, blockDim, x, atten_mask, bias, y, workSpace,
              (uint8_t*)(tilingDatafromBin));
  WriteFile(curPath + "/masked_softmax_with_relposbias_data/y.bin", y, xSize);

  std::string compare_cmd = "cd ./masked_softmax_with_relposbias_data/ && python3 compare_data.py " + std::string(" ") + dtype;
  EXPECT_EQ(system(compare_cmd.c_str()), 0);
  AscendC::GmFree((void*)y);
  AscendC::GmFree((void*)expect_y);
  AscendC::GmFree((void*)x);
  AscendC::GmFree((void*)atten_mask);
  AscendC::GmFree((void*)bias);
  AscendC::GmFree((void*)tiling);
}

TEST_F(masked_softmax_with_relposbias_test, test_case_0) {
  uint32_t BS = 48;
  uint32_t W = 1;
  uint32_t N = 1;
  uint32_t S1 = 8;
  uint32_t S2 = 8;
  std::string caseIdx = "case502";
  std::string dtype = "float32";
  uint32_t dtypeSize = 4;
  //run_test_case(BS, W, N, S1, S2, dtype, caseIdx, dtypeSize, 502);
  caseIdx = "case512";
  dtype = "half";
  dtypeSize = 2;
  //run_test_case(BS, W, N, S1, S2, dtype, caseIdx, dtypeSize, 512);
  caseIdx = "case522";
  dtype = "bfloat16";
  dtypeSize = 2;
  //run_test_case(BS, W, N, S1, S2, dtype, caseIdx, dtypeSize, 512);
}

// TEST_F(masked_softmax_with_relposbias_test, test_case_50) {
//   uint32_t BS = 48;
//   uint32_t W = 1;
//   uint32_t N = 1;
//   uint32_t S1 = 16;
//   uint32_t S2 = 16;

//   //下面4个循环，代表每个类型的4个tiling key，从1到4, 
//   //最后一个输入，代表当前tilingkey的第几个测试，每个tilingkey需要测试多个shape，例如：
//   //测试模板4的，float32，第一个tilingkey，第一种shape，则gen_tiling.py中的caseidx应该写成"case4010"，第二种shape则为"case4011"
//   std::string dtype = "float32";
//   for (uint32_t i = 1; i <= 4; i++) {
//     run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(float), 5000 + i, 0);
//   }
  
//   dtype = "half";
//   for (uint32_t i = 1; i <= 4; i++) {
//     run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(half), 5010 + i, 0);
//   }

//   dtype = "bfloat16";
//   for (uint32_t i = 1; i <= 4; i++) {
//     run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(bfloat16_t), 5020 + i, 0);
//   }
// }

// TEST_F(masked_softmax_with_relposbias_test, test_case_51) {
//   uint32_t BS = 48;
//   uint32_t W = 1;
//   uint32_t N = 1;
//   uint32_t S1 = 32;
//   uint32_t S2 = 31;

//   //下面4个循环，代表每个类型的4个tiling key，从1到4, 
//   //最后一个输入，代表当前tilingkey的第几个测试，每个tilingkey需要测试多个shape，例如：
//   //测试模板4的，float32，第一个tilingkey，第一种shape，则gen_tiling.py中的caseidx应该写成"case4010"，第二种shape则为"case4011"
//   std::string dtype = "float32";
//   for (uint32_t i = 1; i <= 4; i++) {
//     run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(float), 5100 + i, 0);
//   }
  
//   dtype = "half";
//   for (uint32_t i = 1; i <= 4; i++) {
//     run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(half), 5110 + i, 0);
//   }

//   dtype = "bfloat16";
//   for (uint32_t i = 1; i <= 4; i++) {
//     run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(bfloat16_t), 5120 + i, 0);
//   }
// }

TEST_F(masked_softmax_with_relposbias_test, test_case_52) {
  uint32_t BS = 1023;
  uint32_t W = 4;
  uint32_t N = 2;
  uint32_t S1 = 16;
  uint32_t S2 = 16;

  //下面4个循环，代表每个类型的4个tiling key，从1到4, 
  //最后一个输入，代表当前tilingkey的第几个测试，每个tilingkey需要测试多个shape，例如：
  //测试模板4的，float32，第一个tilingkey，第一种shape，则gen_tiling.py中的caseidx应该写成"case4010"，第二种shape则为"case4011"
  std::string dtype = "float32";
  for (uint32_t i = 1; i <= 4; i++) {
    run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(float), 5000 + i, 1);
  }
  
  dtype = "half";
  for (uint32_t i = 1; i <= 4; i++) {
    run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(half), 5010 + i, 1);
  }

  dtype = "bfloat16";
  for (uint32_t i = 1; i <= 4; i++) {
    run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(bfloat16_t), 5020 + i, 1);
  }
}

// TEST_F(masked_softmax_with_relposbias_test, test_case_53) {
//   uint32_t BS = 49;
//   uint32_t W = 3;
//   uint32_t N = 5;
//   uint32_t S1 = 4;
//   uint32_t S2 = 3;

//   //下面4个循环，代表每个类型的4个tiling key，从1到4, 
//   //最后一个输入，代表当前tilingkey的第几个测试，每个tilingkey需要测试多个shape，例如：
//   //测试模板4的，float32，第一个tilingkey，第一种shape，则gen_tiling.py中的caseidx应该写成"case4010"，第二种shape则为"case4011"
//   std::string dtype = "float32";
//   for (uint32_t i = 1; i <= 4; i++) {
//     run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(float), 5100 + i, 1);
//   }
  
//   dtype = "half";
//   for (uint32_t i = 1; i <= 4; i++) {
//     run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(half), 5110 + i, 1);
//   }

//   dtype = "bfloat16";
//   for (uint32_t i = 1; i <= 4; i++) {
//     run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(bfloat16_t), 5120 + i, 1);
//   }
// }

TEST_F(masked_softmax_with_relposbias_test, test_case_54) {
  uint32_t BS = 525;
  uint32_t W = 3;
  uint32_t N = 5;
  uint32_t S1 = 4;
  uint32_t S2 = 3;

  //下面4个循环，代表每个类型的4个tiling key，从1到4, 
  //最后一个输入，代表当前tilingkey的第几个测试，每个tilingkey需要测试多个shape，例如：
  //测试模板4的，float32，第一个tilingkey，第一种shape，则gen_tiling.py中的caseidx应该写成"case4010"，第二种shape则为"case4011"
  std::string dtype = "float32";
  for (uint32_t i = 1; i <= 4; i++) {
    run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(float), 5100 + i, 2);
  }
  
  dtype = "half";
  for (uint32_t i = 1; i <= 4; i++) {
    run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(half), 5110 + i, 2);
  }

  dtype = "bfloat16";
  for (uint32_t i = 1; i <= 4; i++) {
    run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(bfloat16_t), 5120 + i, 2);
  }
}

TEST_F(masked_softmax_with_relposbias_test, test_case_40) {
  uint32_t BS = 2;
  uint32_t W = 95;
  uint32_t N = 1;
  uint32_t S1 = 8;
  uint32_t S2 = 8;

  //下面4个循环，代表每个类型的4个tiling key，从1到4, 
  //最后一个输入，代表当前tilingkey的第几个测试，每个tilingkey需要测试多个shape，例如：
  //测试模板4的，float32，第一个tilingkey，第一种shape，则gen_tiling.py中的caseidx应该写成"case4010"，第二种shape则为"case4011"
  std::string dtype = "float32";
  for (uint32_t i = 1; i <= 4; i++) {
    run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(float), 400 + i, 0);
  }
  
  dtype = "half";
  for (uint32_t i = 1; i <= 4; i++) {
    run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(half), 410 + i, 0);
  }

  dtype = "bfloat16";
  for (uint32_t i = 1; i <= 4; i++) {
    run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(bfloat16_t), 420 + i, 0);
  }
}

TEST_F(masked_softmax_with_relposbias_test, test_case_41) {
  uint32_t BS = 2;
  uint32_t W = 96;
  uint32_t N = 1;
  uint32_t S1 = 8;
  uint32_t S2 = 8;

  //下面4个循环，代表每个类型的4个tiling key，从1到4, 
  //最后一个输入，代表当前tilingkey的第几个测试，每个tilingkey需要测试多个shape，例如：
  //测试模板4的，float32，第一个tilingkey，第一种shape，则gen_tiling.py中的caseidx应该写成"case4010"，第二种shape则为"case4011"
  std::string dtype = "float32";
  for (uint32_t i = 1; i <= 4; i++) {
    run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(float), 400 + i, 1);
  }
  
  dtype = "half";
  for (uint32_t i = 1; i <= 4; i++) {
    run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(half), 410 + i, 1);
  }

  dtype = "bfloat16";
  for (uint32_t i = 1; i <= 4; i++) {
    run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(bfloat16_t), 420 + i, 1);
  }
}

TEST_F(masked_softmax_with_relposbias_test, test_case_42) {
  uint32_t BS = 2;
  uint32_t W = 96;
  uint32_t N = 5;
  uint32_t S1 = 8;
  uint32_t S2 = 7;

  //下面4个循环，代表每个类型的4个tiling key，从1到4, 
  //最后一个输入，代表当前tilingkey的第几个测试，每个tilingkey需要测试多个shape，例如：
  //测试模板4的，float32，第一个tilingkey，第一种shape，则gen_tiling.py中的caseidx应该写成"case4010"，第二种shape则为"case4011"
  std::string dtype = "float32";
  for (uint32_t i = 1; i <= 4; i++) {
    run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(float), 400 + i, 2);
  }
  
  dtype = "half";
  for (uint32_t i = 1; i <= 4; i++) {
    run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(half), 410 + i, 2);
  }

  dtype = "bfloat16";
  for (uint32_t i = 1; i <= 4; i++) {
    run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(bfloat16_t), 420 + i, 2);
  }
}

TEST_F(masked_softmax_with_relposbias_test, test_case_20) {
  uint32_t BS = 2;
  uint32_t W = 1;
  uint32_t N = 16;
  uint32_t S1 = 144;
  uint32_t S2 = 144;

  //下面4个循环，代表每个类型的4个tiling key，从1到4, 
  //最后一个输入，代表当前tilingkey的第几个测试，每个tilingkey需要测试多个shape，例如：
  //测试模板4的，float32，第一个tilingkey，第一种shape，则gen_tiling.py中的caseidx应该写成"case4010"，第二种shape则为"case4011"
  std::string dtype = "float32";
  for (uint32_t i = 3; i <= 4; i++) {
    run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(float), 200 + i, 1);
  }
  
  dtype = "half";
  for (uint32_t i = 3; i <= 4; i++) {
    run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(half), 210 + i, 1);
  }

  dtype = "bfloat16";
  for (uint32_t i = 3; i <= 4; i++) {
    run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(bfloat16_t), 220 + i, 1);
  }
}

// TEST_F(masked_softmax_with_relposbias_test, test_case_21) {
//   uint32_t BS = 2;
//   uint32_t W = 1;
//   uint32_t N = 16;
//   uint32_t S1 = 144;
//   uint32_t S2 = 145;

//   //下面4个循环，代表每个类型的4个tiling key，从1到4, 
//   //最后一个输入，代表当前tilingkey的第几个测试，每个tilingkey需要测试多个shape，例如：
//   //测试模板4的，float32，第一个tilingkey，第一种shape，则gen_tiling.py中的caseidx应该写成"case4010"，第二种shape则为"case4011"
//   std::string dtype = "float32";
//   for (uint32_t i = 3; i <= 4; i++) {
//     run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(float), 2100 + i, 1);
//   }
  
//   dtype = "half";
//   for (uint32_t i = 3; i <= 4; i++) {
//     run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(half), 2110 + i, 1);
//   }

//   dtype = "bfloat16";
//   for (uint32_t i = 3; i <= 4; i++) {
//     run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(bfloat16_t), 2120 + i, 1);
//   }
// }

// TEST_F(masked_softmax_with_relposbias_test, test_case_22) {
//   uint32_t BS = 2;
//   uint32_t W = 1;
//   uint32_t N = 1;
//   uint32_t S1 = 4;
//   uint32_t S2 = 144;

//   //下面4个循环，代表每个类型的4个tiling key，从1到4, 
//   //最后一个输入，代表当前tilingkey的第几个测试，每个tilingkey需要测试多个shape，例如：
//   //测试模板4的，float32，第一个tilingkey，第一种shape，则gen_tiling.py中的caseidx应该写成"case4010"，第二种shape则为"case4011"
//   std::string dtype = "float32";
//   for (uint32_t i = 3; i <= 4; i++) {
//     run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(float), 200 + i, 2);
//   }
  
//   dtype = "half";
//   for (uint32_t i = 3; i <= 4; i++) {
//     run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(half), 210 + i, 2);
//   }

//   dtype = "bfloat16";
//   for (uint32_t i = 3; i <= 4; i++) {
//     run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(bfloat16_t), 220 + i, 2);
//   }
// }

// TEST_F(masked_softmax_with_relposbias_test, test_case_23) {
//   uint32_t BS = 1;
//   uint32_t W = 1;
//   uint32_t N = 1;
//   uint32_t S1 = 49;
//   uint32_t S2 = 145;

//   //下面4个循环，代表每个类型的4个tiling key，从1到4, 
//   //最后一个输入，代表当前tilingkey的第几个测试，每个tilingkey需要测试多个shape，例如：
//   //测试模板4的，float32，第一个tilingkey，第一种shape，则gen_tiling.py中的caseidx应该写成"case4010"，第二种shape则为"case4011"
//   std::string dtype = "float32";
//   for (uint32_t i = 3; i <= 4; i++) {
//     run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(float), 2100 + i, 2);
//   }
  
//   dtype = "half";
//   for (uint32_t i = 3; i <= 4; i++) {
//     run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(half), 2110 + i, 2);
//   }

//   dtype = "bfloat16";
//   for (uint32_t i = 3; i <= 4; i++) {
//     run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(bfloat16_t), 2120 + i, 2);
//   }
// }

// TEST_F(masked_softmax_with_relposbias_test, test_case_24) {
//   uint32_t BS = 1;
//   uint32_t W = 1;
//   uint32_t N = 1;
//   uint32_t S1 = 1;
//   uint32_t S2 = 16;

//   //下面4个循环，代表每个类型的4个tiling key，从1到4, 
//   //最后一个输入，代表当前tilingkey的第几个测试，每个tilingkey需要测试多个shape，例如：
//   //测试模板4的，float32，第一个tilingkey，第一种shape，则gen_tiling.py中的caseidx应该写成"case4010"，第二种shape则为"case4011"
//   std::string dtype = "float32";
//   for (uint32_t i = 3; i <= 4; i++) {
//     run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(float), 200 + i, 3);
//   }
  
//   dtype = "half";
//   for (uint32_t i = 3; i <= 4; i++) {
//     run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(half), 210 + i, 3);
//   }

//   dtype = "bfloat16";
//   for (uint32_t i = 3; i <= 4; i++) {
//     run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(bfloat16_t), 220 + i, 3);
//   }
// }

// TEST_F(masked_softmax_with_relposbias_test, test_case_25) {
//   uint32_t BS = 1;
//   uint32_t W = 1;
//   uint32_t N = 1;
//   uint32_t S1 = 1;
//   uint32_t S2 = 7;

//   //下面4个循环，代表每个类型的4个tiling key，从1到4, 
//   //最后一个输入，代表当前tilingkey的第几个测试，每个tilingkey需要测试多个shape，例如：
//   //测试模板4的，float32，第一个tilingkey，第一种shape，则gen_tiling.py中的caseidx应该写成"case4010"，第二种shape则为"case4011"
//   std::string dtype = "float32";
//   for (uint32_t i = 3; i <= 4; i++) {
//     run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(float), 2100 + i, 3);
//   }
  
//   dtype = "half";
//   for (uint32_t i = 3; i <= 4; i++) {
//     run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(half), 2110 + i, 3);
//   }

//   dtype = "bfloat16";
//   for (uint32_t i = 3; i <= 4; i++) {
//     run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(bfloat16_t), 2120 + i, 3);
//   }
// }

TEST_F(masked_softmax_with_relposbias_test, test_case_4) {
  uint32_t BS = 1;
  uint32_t W = 128;
  uint32_t N = 3;
  uint32_t S1 = 8;
  uint32_t S2 = 8;
  std::string caseIdx = "case4";
  std::string dtype = "float32";
  uint32_t dtypeSize = 4;
  
  //run_test_case<float>(BS, W, N, S1, S2, dtype, caseIdx, dtypeSize);
  caseIdx = "case8";
  dtype = "half";
  dtypeSize = 2;
  //run_test_case<half>(BS, W, N, S1, S2, dtype, caseIdx, dtypeSize);
  caseIdx = "case9";
  dtype = "bfloat16";
  dtypeSize = 2;
  //run_test_case<bfloat16_t>(BS, W, N, S1, S2, dtype, caseIdx, dtypeSize);
}

TEST_F(masked_softmax_with_relposbias_test, test_case_3) {
  uint32_t BS = 20;
  uint32_t W = 38;
  uint32_t N = 3;
  uint32_t S1 = 7;
  uint32_t S2 = 7;
  std::string caseIdx = "case10";
  std::string dtype = "float32";
  uint32_t dtypeSize = 4;
  //run_test_case<float>(BS, W, N, S1, S2, dtype, caseIdx, dtypeSize);
  caseIdx = "case11";
  dtype = "half";
  dtypeSize = 2;
  //run_test_case<half>(BS, W, N, S1, S2, dtype, caseIdx, dtypeSize);
  caseIdx = "case12";
  dtype = "bfloat16";
  dtypeSize = 2;
  //run_test_case<bfloat16_t>(BS, W, N, S1, S2, dtype, caseIdx, dtypeSize);
}

TEST_F(masked_softmax_with_relposbias_test, test_case_30) {
  uint32_t BS = 2;
  uint32_t W = 2;
  uint32_t N = 128;
  uint32_t S1 = 8;
  uint32_t S2 = 8;

  //下面4个循环，代表每个类型的4个tiling key，从1到4, 
  //最后一个输入，代表当前tilingkey的第几个测试，每个tilingkey需要测试多个shape，例如：
  //测试模板3的，float32，第一个tilingkey，第一种shape，则gen_tiling.py中的caseidx应该写成"case3010"，第二种shape则为"case3011"

  std::string dtype = "float32";
  for (uint32_t i = 1; i <= 4; i++) {
    run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(float), 300 + i, 0);
  }
  
  dtype = "half";
  // ut 报在softmax内部，npu执行正常
  for (uint32_t i = 1; i <= 1; i++) {
    run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(half), 310 + i, 0);
  }

  dtype = "bfloat16";
  for (uint32_t i = 1; i <= 2; i++) {
    run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(bfloat16_t), 320 + i, 0);
  }
}

TEST_F(masked_softmax_with_relposbias_test, test_case_31) {
  uint32_t BS = 2;
  uint32_t W = 3;
  uint32_t N = 127;
  uint32_t S1 = 5;
  uint32_t S2 = 5;

  //下面4个循环，代表每个类型的4个tiling key，从1到4, 
  //最后一个输入，代表当前tilingkey的第几个测试，每个tilingkey需要测试多个shape，例如：
  //测试模板3的，float32，第一个tilingkey，第一种shape，则gen_tiling.py中的caseidx应该写成"case3010"，第二种shape则为"case3011"

  std::string dtype = "float32";
  for (uint32_t i = 1; i <= 4; i++) {
    run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(float), 300 + i, 1);
  }
  
  dtype = "half";
  for (uint32_t i = 1; i <= 1; i++) {
    run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(half), 310 + i, 1);
  }

  dtype = "bfloat16";
  for (uint32_t i = 1; i <= 2; i++) {
    run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(bfloat16_t), 320 + i, 1);
  }
}

TEST_F(masked_softmax_with_relposbias_test, test_case_20_12) {
  uint32_t BS = 2;
  uint32_t W = 3;
  uint32_t N = 5;
  uint32_t S1 = 36;
  uint32_t S2 = 16;

  //下面4个循环，代表每个类型的4个tiling key，从1到4, 
  //最后一个输入，代表当前tilingkey的第几个测试，每个tilingkey需要测试多个shape，例如：
  //测试模板4的，float32，第一个tilingkey，第一种shape，则gen_tiling.py中的caseidx应该写成"case4010"，第二种shape则为"case4011"
  std::string dtype = "float32";
  for (uint32_t i = 1; i <= 2; i++) {
    run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(float), 200 + i, 0);
  }
  
  dtype = "half";
  for (uint32_t i = 1; i <= 2; i++) {
    run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(half), 210 + i, 0);
  }

  dtype = "bfloat16";
  for (uint32_t i = 1; i <= 2; i++) {
    run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(bfloat16_t), 220 + i, 0);
  }
}

TEST_F(masked_softmax_with_relposbias_test, test_case_21_12) {
  uint32_t BS = 2;
  uint32_t W = 3;
  uint32_t N = 5;
  uint32_t S1 = 36;
  uint32_t S2 = 15;

  //下面4个循环，代表每个类型的4个tiling key，从1到4, 
  //最后一个输入，代表当前tilingkey的第几个测试，每个tilingkey需要测试多个shape，例如：
  //测试模板4的，float32，第一个tilingkey，第一种shape，则gen_tiling.py中的caseidx应该写成"case4010"，第二种shape则为"case4011"
  std::string dtype = "float32";
  for (uint32_t i = 1; i <= 2; i++) {
    run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(float), 200 + i, 1);
  }
  
  dtype = "half";
  for (uint32_t i = 1; i <= 2; i++) {
    run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(half), 210 + i, 1);
  }

  dtype = "bfloat16";
  for (uint32_t i = 1; i <= 2; i++) {
    run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(bfloat16_t), 220 + i, 1);
  }
}

TEST_F(masked_softmax_with_relposbias_test, test_case_s2_1) {
  uint32_t BS = 4;
  uint32_t W = 256;
  uint32_t N = 128;
  uint32_t S1 = 1;
  uint32_t S2 = 1;

  std::string dtype = "float32";
  run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(float), 100, 0);
  dtype = "half";
  run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(half), 110, 0);
  dtype = "bfloat16";
  run_masked_softmax_with_relposbias_case(BS, W, N, S1, S2, dtype, sizeof(bfloat16_t), 120, 0);
}

// TEST_F(masked_softmax_with_relposbias_test, B_testcase_with_tail) {
//   uint32_t B = 64;
//   uint32_t W = 5;
//   uint32_t N = 1;
//   uint32_t S1 = 1;
//   uint32_t S2 = 128;

//   //下面4个循环，代表每个类型的4个tiling key，从1到4, 
//   //最后一个输入，代表当前tilingkey的第几个测试，每个tilingkey需要测试多个shape，例如：
//   //测试模板5的，float32，第一个tilingkey，第一种shape，则gen_tiling.py中的caseidx应该写成"case50012"，第二种shape则为"case50022"

//   std::string dtype = "float32";
//   for (uint32_t i = 1; i <= 4; i++) {
//     run_masked_softmax_with_relposbias_case(B, W, N, S1, S2, dtype, sizeof(float), 5000 + i, 2);
//   }
  
//   dtype = "half";
//   for (uint32_t i = 1; i <= 4; i++) {
//     run_masked_softmax_with_relposbias_case(B, W, N, S1, S2, dtype, sizeof(half), 5010 + i, 2);
//   }

//   dtype = "bfloat16";
//   for (uint32_t i = 1; i <= 4; i++) {
//     run_masked_softmax_with_relposbias_case(B, W, N, S1, S2, dtype, sizeof(bfloat16_t), 5020 + i, 2);
//   }
// }