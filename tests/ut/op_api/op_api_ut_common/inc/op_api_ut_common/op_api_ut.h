#ifndef _OP_API_UT_COMMON_UT_CASE_H_
#define _OP_API_UT_COMMON_UT_CASE_H_

#include <string>
#include <fstream>
#include <map>

#include "gtest/gtest.h"
#include "nlohmann/json.hpp"

#include "opdev/op_executor.h"

#include "op_api_ut_common/inner/c_shell.h"
#include "op_api_ut_common/inner/file_io.h"
#include "op_api_ut_common/tuple_utils.h"

using namespace std;
using namespace nlohmann;

////////////  reload tensor data  ///////////////

template<typename T>
int ReloadDataFromBinaryFile([[maybe_unused]] T arg, [[maybe_unused]] size_t index,
                             [[maybe_unused]] size_t total_input, [[maybe_unused]] const string& file_prefix) {
  return 0;
}

template<size_t index, typename... Ts>
struct ReloadIterator {
  int operator() (tuple<Ts...>& t, size_t input_count, const string& file_prefix) {
    auto ret = ReloadDataFromBinaryFile(get<index>(t), index, input_count, file_prefix);
    if (ret != 0) {
      return ret;
    }
    return ReloadIterator<index - 1, Ts...>{}(t, input_count, file_prefix);
  }
};

template<typename... Ts>
struct ReloadIterator<0, Ts...> {
  int operator() (tuple<Ts...>& t, size_t input_count, const string& file_prefix) {
    return ReloadDataFromBinaryFile(get<0>(t), 0, input_count, file_prefix);
  }
};

template<typename... Ts>
int ReloadTensorData(tuple<Ts...>& t, size_t input_count, const string& file_prefix) {
  const auto size = tuple_size<tuple<Ts...>>::value;
  ReloadIterator<size - 1, Ts...> it;
  return it(t, input_count, file_prefix);
}

////////////  save result      ///////////////

[[maybe_unused]]static short int GetInplaceOutputIdx(size_t index, size_t input_count,
                                     const map<short int, short int>& inplace_map_reserve) {
  if (index >= input_count) {
    return -1;
  }
  auto it = inplace_map_reserve.find(static_cast<short int>(index));
  return it != inplace_map_reserve.end() ? it->second : -1;
}

template<typename T>
int SaveResultToBinaryFile([[maybe_unused]] T arg, [[maybe_unused]] size_t index,
                           [[maybe_unused]] size_t total_input,
                           [[maybe_unused]] const string& file_prefix,
                           [[maybe_unused]] const set<short int>& inplace_output) {
  return 0;
}

template<typename T>
int SaveResultToBinaryFile([[maybe_unused]] T arg, [[maybe_unused]] size_t index,
                           [[maybe_unused]] const string& file_prefix) {
  return 0;
}

template<size_t index, typename... Ts>
struct SaveIterator {
  int operator() (tuple<Ts...>& t, size_t input_count, const string& file_prefix,
                  const map<short int, short int>& inplace_map_reserve, const set<short int>& inplace_output) {
    short int inplace_output_idx = GetInplaceOutputIdx(index, input_count, inplace_map_reserve);
    int ret = 0;
    if (inplace_output_idx < 0) {
      ret = SaveResultToBinaryFile(get<index>(t), index, input_count, file_prefix, inplace_output);
    } else {
      ret = SaveResultToBinaryFile(get<index>(t), inplace_output_idx, file_prefix);
    }

    if (ret != 0) {
      return ret;
    }
    return SaveIterator<index - 1, Ts...>{}(t, input_count, file_prefix, inplace_map_reserve, inplace_output);
  }
};

template<typename... Ts>
struct SaveIterator<0, Ts...> {
  int operator() (tuple<Ts...>& t, size_t input_count, const string& file_prefix,
                  const map<short int, short int>& inplace_map_reserve, const set<short int>& inplace_output) {
    short int inplace_output_idx = GetInplaceOutputIdx(0, input_count, inplace_map_reserve);
    if (inplace_output_idx < 0) {
      return SaveResultToBinaryFile(get<0>(t), 0, input_count, file_prefix, inplace_output);
    } else {
      return SaveResultToBinaryFile(get<0>(t), inplace_output_idx, file_prefix);
    }
  }
};

template<typename... Ts>
int SaveResult(tuple<Ts...>& t, size_t input_count, const string& file_prefix,
               const map<short int, short int>& inplace_map_reserve, const set<short int>& inplace_output) {
  const auto size = tuple_size<tuple<Ts...>>::value;
  SaveIterator<size - 1, Ts...> it;
  return it(t, input_count, file_prefix, inplace_map_reserve, inplace_output);
}

//////////////////////////////////////////////

struct Enter {
  Enter(const string& info, void(*cleanup_func)(const string&), const string& cleanup_args)
    : info_(info), cleanup_func_(cleanup_func), cleanup_args_(cleanup_args) {
    cout << "=== ENTER " << info_ << " ===" << endl;
  }
  ~Enter() {
    cleanup_func_(cleanup_args_);
    setenv("NEED_AICPU_SIMULATOR", "0", 1);
    cout << "=== LEAVE " << info_ << " ===" << endl;
  }
  private:
    string info_;
    void (*cleanup_func_)(const string&);
    string cleanup_args_;
};

#define ENTER(info) Enter enter(info, DeleteUtTmpFiles, tmp_file_prefix_)

///////////////////////////

template<typename GET_F, typename API_F,
         typename INPUT_T, typename OUTPUT_T, typename ACL_ARGS_T,
         typename EXTRA_T>
class OpApiUt {
  public:
    OpApiUt(const string& test_suite_name, const string& test_case_name, const string& op_name,
            GET_F get_workspace_size_func, API_F api_func,
            const INPUT_T& inputs, const OUTPUT_T& outputs,
            [[maybe_unused]] const ACL_ARGS_T& wrapper_args,
            const EXTRA_T& extra_args)
      : op_name_(op_name), case_name_(test_suite_name + "_" + test_case_name),
        api_func_(api_func), get_workspace_size_func_(get_workspace_size_func),
        inputs_(inputs), outputs_(outputs), extra_args_(extra_args) {
      tmp_file_prefix_ = op_name_ + "_" + case_name_;
      case_location_ = GetOpUtSrcPath() + "/" + op_name_;
      input_count_ = tuple_size<INPUT_T>::value;
    }

    ~OpApiUt() {
      if (all_args_set_) {
        ReleaseAclTypes(all_args_);
      }
    }

    OpApiUt& InputGenFunc(const string& input_gen_func) {
      input_gen_func_ = input_gen_func;
      return *this;
    }

    aclnnStatus TestGetWorkspaceSize(uint64_t* workspace_size) {
      aclOpExecutor *executor = nullptr;
      aclnnStatus ret = GetWorkspaceSize(workspace_size, &executor);
      if (executor != nullptr) {
        delete executor;
      }
      return ret;
    }

    aclnnStatus TestGetWorkspaceSizeWithNNopbaseInner(uint64_t* workspace_size,
                                                      aclOpExecutor *executor) {
      aclnnStatus ret = GetWorkspaceSize(workspace_size, &executor);
      return ret;
    }

    void TestPrecision() {
    }

  private:
    aclnnStatus GetWorkspaceSize(uint64_t* workspace_size, aclOpExecutor **executor) {
      ACL_ARGS_T args = GetConvertedAclArgs();
      return InvokeGetWorkspaceSizeFunc(get_workspace_size_func_, args, extra_args_, workspace_size, executor);
    }

    ACL_ARGS_T GetConvertedAclArgs() {
      if (!all_args_set_) {
        all_args_ = ConvertDescToAclTypes(tuple_cat(inputs_, outputs_));
        all_args_set_ = true;
      }
      return all_args_;
    }

  private:
    bool golden_exists_ = true;
    string op_name_;
    string case_name_;
    string input_gen_func_;
    string tmp_file_prefix_;
    string case_location_;
    // actually it is difficult to describe inplace map in condition: 3 outputs & 1 inplace input (how to define the
    // output index? 1 possible soultion is that: consider output index as which in the golden output file index.)
    // Note: If no OUTPUT is specified, default inplace map will be output0 -> input0 only.
    map<string, short int> inplace_map_; // output_index -> input_index
    map<short int, short int> inplace_map_reverse_; // input_index -> output_index
    set<short int> inplace_output_; // inplace output_index

    API_F api_func_;
    GET_F get_workspace_size_func_;
    INPUT_T inputs_;
    OUTPUT_T outputs_;
    ACL_ARGS_T all_args_;  // only inputs + outputs_.
    EXTRA_T extra_args_;
    bool all_args_set_ = false;
    size_t input_count_;
};

#define OP_API_UT(api, input, output, ...)                     \
  OpApiUt(testing::UnitTest::GetInstance()->current_test_info()->test_case_name(),     \
          testing::UnitTest::GetInstance()->current_test_info()->name(), #api,         \
          api##GetWorkspaceSize, api, input, output,                                   \
          InferAclTypes(tuple_cat(input, output)),                                     \
          make_tuple(__VA_ARGS__))

#define INPUT(...) make_tuple(__VA_ARGS__)
#define OUTPUT(...) make_tuple(__VA_ARGS__)

#endif
