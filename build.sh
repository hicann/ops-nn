#!/bin/bash
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

set -e
RELEASE_TARGETS=("ophost" "opapi")
SUPPORT_COMPUTE_UNIT_SHORT=("ascend910b" "ascend910_93" "ascend310p" "ascend910")
TRIGER_UTS=()

# 所有支持的短选项
SUPPORTED_SHORT_OPTS="hj:vO:uf:-:"

# 所有支持的长选项
SUPPORTED_LONG_OPTS=(
  "help" "ops=" "soc=" "vendor_name=" "debug" "noexec" "opkernel" "jit" "pkg"
  "disable_asan" "make_clean_all" "make_clean"
  "ophost" "opapi" "run_example" "example_name=" "genop=" "genop_aicpu="
)

in_array() {
  local needle="$1"
  shift
  local haystack=("$@")
  for item in "${haystack[@]}"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

# 检查参数是否合法
check_option_validity() {
  local arg="$1"

  # 检查短选项
  if [[ "$arg" =~ ^-[^-] ]]; then
    local opt_chars=${arg:1}

    # 获取需要参数的短选项列表
    local needs_arg_opts=$(echo "$SUPPORTED_SHORT_OPTS" | grep -o "[a-zA-Z]:" | tr -d ':')

    # 逐个检查短选项字符
    local i=0
    while [ $i -lt ${#opt_chars} ]; do
      local char="${opt_chars:$i:1}"

      # 检查是否为支持的短选项
      if [[ ! "$SUPPORTED_SHORT_OPTS" =~ "$char" ]]; then
        print_error "Invalid short option: -$char"
        return 1
      fi

      # 如果这个选项需要参数，跳过参数部分
      if [[ "$needs_arg_opts" =~ "$char" ]]; then
        # 跳过参数部分（可能是数字或其他字符）
        while [ $i -lt ${#opt_chars} ] && [[ "${opt_chars:$i:1}" =~ [0-9a-zA-Z] ]]; do
          i=$((i + 1))
        done
      else
        i=$((i + 1))
      fi
    done
    return 0
  fi

  # 检查长选项
  if [[ "$arg" =~ ^-- ]]; then
    local long_opt="${arg:2}"
    local opt_name="${long_opt%%=*}"

    # 检查是否在支持的长选项列表中
    for supported_opt in "${SUPPORTED_LONG_OPTS[@]}"; do
      # 处理带=的长选项
      if [[ "$supported_opt" =~ =$ ]]; then
        local base_opt="${supported_opt%=}"
        if [[ "$opt_name" == "$base_opt" ]]; then
          return 0
        fi
      else
        # 处理不带=的长选项
        if [[ "$opt_name" == "$supported_opt" ]]; then
          return 0
        fi
      fi
    done

    print_error "Invalid long option: --$opt_name"
    return 1
  fi

  # 如果不是选项，可能是其他参数
  return 0
}

dotted_line="----------------------------------------------------------------"
export BASE_PATH=$(
  cd "$(dirname $0)"
  pwd
)
export BUILD_PATH="${BASE_PATH}/build"
export BUILD_OUT_PATH="${BASE_PATH}/build_out"
REPOSITORY_NAME="nn"

CORE_NUMS=$(cat /proc/cpuinfo | grep "processor" | wc -l)
ARCH_INFO=$(uname -m)

export INCLUDE_PATH="${ASCEND_HOME_PATH}/include"
export ACLNN_INCLUDE_PATH="${INCLUDE_PATH}/aclnn"
export COMPILER_INCLUDE_PATH="${ASCEND_HOME_PATH}/compiler/include"
export GRAPH_INCLUDE_PATH="${COMPILER_INCLUDE_PATH}/graph"
export GE_INCLUDE_PATH="${COMPILER_INCLUDE_PATH}/ge"
export INC_INCLUDE_PATH="${ASCEND_OPP_PATH}/built-in/op_proto/inc"
export LINUX_INCLUDE_PATH="${ASCEND_HOME_PATH}/${ARCH_INFO}-linux/include"
export EAGER_LIBRARY_OPP_PATH="${ASCEND_OPP_PATH}/lib64"
export EAGER_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64"
export GRAPH_LIBRARY_STUB_PATH="${ASCEND_HOME_PATH}/compiler/lib64/stub"
export GRAPH_LIBRARY_PATH="${ASCEND_HOME_PATH}/compiler/lib64"

# print usage message
usage() {
  local specific_help="$1"

  if [[ -n "$specific_help" ]]; then
    case "$specific_help" in
      pkg)
        echo "pkg Build Options:"
        echo $dotted_line
        echo "    --pkg                  Build run pkg with kernel bin"
        echo "    --jit                  Build run pkg without kernel bin"
        echo "    --soc=soc_version      Compile for specified Ascend SoC (comma-separated for multiple)"
        echo "    --vendor_name=name     Specify custom operator pkg vendor name"
        echo "    --ops=op1,op2,...      Compile specified operators (comma-separated for multiple)"
        echo "    -j[n]                  Compile thread nums, default is 8"
        echo "    -O[n]                  Compile optimization options, support [O0 O1 O2 O3]"
        echo "    --debug                Build with debug mode"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --pkg --soc=ascend910b --vendor_name=customize -j16 -O3"
        echo "    bash build.sh --pkg --ops=add,sub --debug"
        return
        ;;
      opkernel)
        echo "Opkernel Build Options:"
        echo $dotted_line
        echo "    --opkernel             Build binary kernel"
        echo "    --soc=soc_version      Compile for specified Ascend SoC (comma-separated for multiple)"
        echo "    --ops=op1,op2,...      Compile specified operators (comma-separated for multiple)"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --opkernel --soc=ascend310p --ops=add,sub"
        return
        ;;
      test)
        echo "Test Options:"
        echo $dotted_line
        echo "    -u                     Build and run all unit tests"
        echo "    --noexec               Only compile ut, do not execute"
        echo "    --disable_asan         Disable ASAN (Address Sanitizer)"
        echo "    --ophost -u            Same as ophost test"
        echo "    --opapi -u             Same as opapi test"
        echo "    --opkernel -u          Same as opkernel test"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh -u"
        echo "    bash build.sh -u --ophost"
        return
        ;;
      clean)
        echo "Clean Options:"
        echo $dotted_line
        echo "    --make_clean_all       Clean all build artifacts and related files"
        echo "    --make_clean           Clean build artifacts"
        echo $dotted_line
        return
        ;;
      ophost)
        echo "Ophost Build Options:"
        echo $dotted_line
        echo "    --ophost               Build ophost library"
        echo "    -j[n]                  Compile thread nums, default is 8"
        echo "    -O[n]                  Compile optimization options, support [O0 O1 O2 O3]"
        echo "    --debug                Build with debug mode"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --ophost -j16 -O3"
        echo "    bash build.sh --ophost --debug"
        return
        ;;
      opapi)
        echo "Opapi Build Options:"
        echo $dotted_line
        echo "    --opapi                Build opapi library"
        echo "    -j[n]                  Compile thread nums, default is 8"
        echo "    -O[n]                  Compile optimization options, support [O0 O1 O2 O3]"
        echo "    --debug                Build with debug mode"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --opapi -j16 -O3"
        echo "    bash build.sh --opapi --debug"
        return
        ;;
      run_example)
        echo "Run example Options:"
        echo $dotted_line
        echo "    --run_example op_name  mode[eager:graph] [pkg_mode --vendor_name=name --example_name=name]      Compile and execute the test_aclnn_xxx.cpp/test_geir_xxx.cpp"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --run_example mat_mul_v3 eager"
        echo "    bash build.sh --run_example mat_mul_v3 graph"
        echo "    bash build.sh --run_example mat_mul_v3 eager --example_name=mm"
        echo "    bash build.sh --run_example mat_mul_v3 eager cust"
        echo "    bash build.sh --run_example mat_mul_v3 eager cust --vendor_name=custom"
        return
        ;;
      genop)
        echo "Gen Op Directory Options:"
        echo $dotted_line
        echo "    --genop=op_class/op_name      Create the initial directory for op_name undef op_class"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --genop=op_class/op_name"
        return
        ;;
      genop_aicpu)
        echo "Gen Op Directory Options:"
        echo $dotted_line
        echo "    --genop_aicpu=op_class/op_name      Create the initial directory for op_name undef op_class"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --genop_aicpu=op_class/op_name"
        return
        ;;
    esac
  fi
  echo "build script for ops-nn repository"
  echo "Usage:"
  echo "    bash build.sh [-h] [-j[n]] [-v] [-O[n]] [-u] "
  echo ""
  echo ""
  echo "Options:"
  echo $dotted_line
  echo "    Build parameters "
  echo $dotted_line
  echo "    --help, -h Print usage"
  echo "    -j[n] Compile thread nums, default is 8"
  echo "    -v Cmake compile verbose"
  echo "    -O[n] Compile optimization options, support [O0 O1 O2 O3]"
  echo "    -u Compile all ut, default run ophost opapi test"
  echo $dotted_line
  echo "    example, Build ophost test with O0 level compilation optimization and do not execute."
  echo "    ./build.sh -u --ophost --noexec -O0 -j8"
  echo $dotted_line
  echo "    The following are all supported arguments:"
  echo $dotted_line
  echo "    --debug build with debug mode"
  echo "    --noexec Only compile ut, do not execute the compiled executable file"
  echo "    --make_clean make clean"
  echo "    --make_clean_all make clean and delete related file"
  echo "    --disable_asan disable asan"
  echo ""
  echo "    --ops Compile specified operator, use snake name, like: --ops=add,add_lora, use ',' to separate different operator"
  echo "    --soc Compile binary with specified Ascend SoC, like: --soc=ascend910b"
  echo "    --vendor_name Specify the custom operator pkg vendor name, like: --vendor_name=customize, default to customize-nn"
  echo "    --opapi build opapi_nn.so"
  echo "    --ophost build ophost_nn.so"
  echo "    --opkernel build binary kernel"
  echo "    --pkg build run pkg"
  echo "    --jit build run pkg without kernel bin"
  echo "    --run_example Compile and execute the example. use --run_example --help for more detail"
  echo "    --genop Create the initial directory for op, like: --genop=op_class/op_name"
  echo "    --genop_aicpu Create the initial directory for AI CPU op, like: --genop_aicpu=op_class/op_name"
  echo "to be continued ..."
}

# 检查--help 前的组合参数是否非法
check_help_combinations() {
  local args=("$@")
  local has_u=false
  local has_test_command=false
  local has_build_command=false
  local has_pkg=false
  local has_opkernel=false

  for arg in "${args[@]}"; do
    case "$arg" in
      -u) has_u=true ;;
      --ophost | --opapi)
        has_test_command=true
        has_build_command=true
        ;;
      --pkg) has_pkg=true ;;
      --opkernel) has_opkernel=true ;;
      --help | -h) ;;
    esac
  done

  # 检查help中的无效命令组合
  if [[ "$has_pkg" == "true" && ("$has_test_command" == "true" || "$has_u" == "true") ]]; then
    print_error "--pkg cannot be used with test(-u, etc.), --ophost, --opapi"
    return 1
  fi

  if [[ "$has_opkernel" == "true" && ("$has_test_command" == "true" || "$has_u" == "true") ]]; then
    print_error "--opkernel cannot be used with test(-u, etc.), --ophost, --opapi"
    return 1
  fi

  return 0
}

check_param() {
  # --ops不能与--ophost，--opapi同时存在，如果带U则可以
  if [[ -n "$COMPILED_OPS" && "$ENABLE_TEST" == "FALSE" ]] && [[ "$OP_HOST" == "TRUE" || "$OP_API" == "TRUE" ]]; then
    print_error "--ops cannot be used with --ophost, --opapi"
    exit 1
  fi

  # --pkg不能与-u（UT模式，包含_test的参数）或者--ophost，--opapi同时存在
  if [[ "$ENABLE_PACKAGE" == "TRUE" ]]; then
    if [[ "$ENABLE_TEST" == "TRUE" ]]; then
      print_error "--pkg cannot be used with test(-u, --ophost_test, etc.)"
      exit 1
    fi

    if [[ "$OP_HOST" == "TRUE" || "$OP_API" == "TRUE" ]]; then
      print_error "--pkg cannot be used with --ophost, --opapi"
      exit 1
    fi
  fi

  if $(echo ${USE_CMD} | grep -wq "opkernel") && $(echo ${USE_CMD} | grep -wq "jit"); then
    print_error "--opkernel cannot be used with --jit"
    exit 1
  fi
}

COLOR_RESET="\033[0m"
COLOR_GREEN="\033[32m"
COLOR_RED="\033[31m"
print_success() {
  echo
  echo $dotted_line
  local msg="$1"
  echo -e "${COLOR_GREEN}[SUCCESS] ${msg}${COLOR_RESET}"
  echo $dotted_line
  echo
}

print_error() {
  echo
  echo $dotted_line
  local msg="$1"
  echo -e "${COLOR_RED}[ERROR] ${msg}${COLOR_RESET}"
  echo $dotted_line
  echo
}

set_create_libs() {
  if [[ "$ENABLE_TEST" == "TRUE" ]]; then
    return
  fi
  if [[ "$ENABLE_PACKAGE" == "TRUE" && "$ENABLE_CUSTOM" != "TRUE" ]]; then
    BUILD_LIBS=("ophost_${REPOSITORY_NAME}" "opapi_${REPOSITORY_NAME}")
    ENABLE_CREATE_LIB=TRUE
  else
    if [[ "$OP_HOST" == "TRUE" ]]; then
      BUILD_LIBS+=("ophost_${REPOSITORY_NAME}")
      ENABLE_CREATE_LIB=TRUE
    fi
    if [[ "$OP_API" == "TRUE" ]]; then
      BUILD_LIBS+=("opapi_${REPOSITORY_NAME}")
      ENABLE_CREATE_LIB=TRUE
    fi
  fi
}

set_ut_mode() {
  if [[ "$ENABLE_TEST" != "TRUE" ]]; then
    return
  fi
  ENABLE_CUSTOM=FALSE
  UT_TEST_ALL=TRUE
  if [[ "$OP_HOST" == "TRUE" ]]; then
    OP_HOST_UT=TRUE
    UT_TEST_ALL=FALSE
  fi
  if [[ "$OP_API" == "TRUE" ]]; then
    OP_API_UT=TRUE
    UT_TEST_ALL=FALSE
  fi
  if [[ "$OP_KERNEL" == "TRUE" ]]; then
    OP_KERNEL_UT=TRUE
    UT_TEST_ALL=FALSE
  fi
  # 检查测试项，至少有一个
  if [[ "$UT_TEST_ALL" == "FALSE" && "$OP_HOST_UT" == "FALSE" && "$OP_API_UT" == "FALSE" && "$OP_KERNEL_UT" == "FALSE" ]]; then
    print_error "At least one test target must be specified (ophost test, opapi test, opgraph test, opkernel test)"
    usage
    exit 1
  fi

  if [[ "$UT_TEST_ALL" == "TRUE" ]] || [[ "$OP_HOST_UT" == "TRUE" ]]; then
    UT_TARGES+=("${REPOSITORY_NAME}_op_host_ut")
  fi
  if [[ "$UT_TEST_ALL" == "TRUE" ]] || [[ "$OP_API_UT" == "TRUE" ]]; then
    UT_TARGES+=("${REPOSITORY_NAME}_op_api_ut")
  fi
  if [[ "$UT_TEST_ALL" == "TRUE" ]] || [[ "$OP_KERNEL_UT" == "TRUE" ]]; then
    UT_TARGES+=("${REPOSITORY_NAME}_op_kernel_ut")
  fi
}

make_clean() {
  if [ -d "$BUILD_PATH" ]; then
    cd $BUILD_PATH
    make clean
    print_success "make clean success!"
  fi
}

make_clean_all() {
  if [ -d "$BUILD_PATH" ]; then
    cd $BUILD_PATH
    rm -rf ./*
  fi
  [ -d "$BUILD_OUT_PATH" ] && rm -rf $BUILD_OUT_PATH
  print_success "make clean all success!"
}

checkopts() {
  THREAD_NUM=8
  VERBOSE=""
  BUILD_MODE=""
  COMPILED_OPS=""
  UT_TEST_ALL=FALSE
  CHANGED_FILES=""
  CI_MODE=FALSE
  COMPUTE_UNIT=""
  VENDOR_NAME=""
  SHOW_HELP=""
  OP_NAME=""
  EXAMPLE_NAME=""
  EXAMPLE_MODE=""
  USE_CMD="$*"

  ENABLE_DEBUG=FALSE
  ENABLE_CONVERAGE=FALSE
  ENABLE_UT_EXEC=TRUE
  ENABLE_ASAN=FALSE
  ENABLE_VALGRIND=FALSE
  ENABLE_BINARY=FALSE
  ENABLE_CUSTOM=FALSE
  ENABLE_PACKAGE=FALSE
  ENABLE_JIT=FALSE
  ENABLE_TEST=FALSE
  AICPU_ONLY=FALSE
  OP_API_UT=FALSE
  OP_HOST_UT=FALSE
  OP_GRAPH_UT=FALSE
  OP_KERNEL_UT=FALSE
  OP_API=FALSE
  OP_HOST=FALSE
  OP_KERNEL=FALSE
  ENABLE_CREATE_LIB=FALSE
  ENABLE_RUN_EXAMPLE=FALSE
  BUILD_LIBS=()
  UT_TARGES=()

  ENABLE_GENOP=FALSE
  ENABLE_GENOP_AICPU=FALSE
  GENOP_TYPE=""
  GENOP_NAME=""

  if [ $# -eq 0 ]; then
    usage "$SHOW_HELP"
    exit 0
  fi

  # 首先检查所有参数是否合法
  for arg in "$@"; do
    if [[ "$arg" =~ ^- ]]; then # 只检查以-开头的参数
      if ! check_option_validity "$arg"; then
        usage
        exit 1
      fi
    fi
  done

  # 检查并处理--help
  for arg in "$@"; do
    if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
      # 检查帮助信息中的组合参数
      check_help_combinations "$@"
      local comb_result=$?
      if [ $comb_result -eq 1 ]; then
        exit 1
      fi
      SHOW_HELP="general"

      # 检查 --help 前面的命令
      for prev_arg in "$@"; do
        case "$prev_arg" in
          --pkg) SHOW_HELP="pkg" ;;
          --opkernel) SHOW_HELP="opkernel" ;;
          -u) SHOW_HELP="test" ;;
          --make_clean_all | --make_clean) SHOW_HELP="clean" ;;
          --ophost) SHOW_HELP="ophost" ;;
          --opapi) SHOW_HELP="opapi" ;;
          --run_example) SHOW_HELP="run_example" ;;
          --genop) SHOW_HELP="genop" ;;
        esac
      done

      usage "$SHOW_HELP"
      exit 0
    fi
  done

  process_genop() {
    local opt_name=$1
    local genop_value=$2

    if [[ "$opt_name" == "genop" ]]; then
      ENABLE_GENOP=TRUE
    elif [[ "$opt_name" == "genop_aicpu" ]]; then
      ENABLE_GENOP_AICPU=TRUE
    else
      usage "genop"
      exit 1
    fi

    if [[ "$genop_value" != *"/"* ]] || [[ "$genop_value" == *"/"*"/"* ]]; then
      usage "$opt_name"
      exit 1
    fi

    GENOP_TYPE=$(echo "$genop_value" | cut -d'/' -f1)
    GENOP_NAME=$(echo "$genop_value" | cut -d'/' -f2)
  }

  # Process the options
  while getopts $SUPPORTED_SHORT_OPTS opt; do
    case "${opt}" in
      h)
        usage
        exit 0
        ;;
      j) THREAD_NUM=$OPTARG ;;
      v) VERBOSE="VERBOSE=1" ;;
      O) BUILD_MODE="-O$OPTARG" ;;
      u) ENABLE_TEST=TRUE ;;
      f)
        CHANGED_FILES=$OPTARG
        CI_MODE=TRUE
        ;;
      -) case $OPTARG in
        help)
          usage
          exit 0
          ;;
        ops=*)
          COMPILED_OPS=${OPTARG#*=}
          ENABLE_CUSTOM=TRUE
          ;;
        genop=*)
          process_genop "genop" "${OPTARG#*=}"
          ;;
        genop_aicpu=*)
          process_genop "genop_aicpu" "${OPTARG#*=}"
          ;;
        soc=*)
          COMPUTE_UNIT=${OPTARG#*=}
          ;;
        vendor_name=*)
          VENDOR_NAME=${OPTARG#*=}
          ENABLE_CUSTOM=TRUE
          ;;
        example_name=*) EXAMPLE_NAME=${OPTARG#*=} ;;
        debug) ENABLE_DEBUG=TRUE ;;
        noexec) ENABLE_UT_EXEC=FALSE ;;
        opkernel)
          ENABLE_BINARY=TRUE
          OP_KERNEL=TRUE
          ;;
        pkg)
          ENABLE_PACKAGE=TRUE
          ENABLE_BINARY=TRUE
          ;;
        jit)
          ENABLE_JIT=TRUE
          ;;
        disable_asan) ENABLE_ASAN=FALSE ;;
        run_example) ENABLE_RUN_EXAMPLE=TRUE
          step=0
          set_example_opt $2 $3 $4
          shift $step
          ;;
        make_clean_all) make_clean_all
                        exit 0 ;;
        make_clean) make_clean
                    exit 0 ;;
        *)
          ## 如果不在RELEASE_TARGETS，不做处理
          if ! in_array "$OPTARG" "${RELEASE_TARGETS[@]}"; then
            print_error "Invalid option: --$OPTARG"
            usage
            exit 1
          fi

          if [[ "$OPTARG" == "ophost" ]]; then
            OP_HOST=TRUE
          elif [[ "$OPTARG" == "opapi" ]]; then
            OP_API=TRUE
          elif [[ "$OPTARG" == "opkernel" ]]; then
            OP_KERNEL=TRUE
          else
            usage
            exit 1
          fi
          ;;
      esac ;;
      *)
        print_error "Undefined option: ${opt}"
        usage
        exit 1
        ;;
    esac
  done

  if [[ "$ENABLE_TEST" == "TRUE" && "$OP_HOST" == "FALSE" && "$OP_API" == "FALSE" && "$OP_KERNEL" == "FALSE" ]]; then
    OP_HOST=TRUE
    OP_API=TRUE
    OP_KERNEL=TRUE
  fi
  if [[ "$ENABLE_JIT" == "TRUE" ]]; then
    ENABLE_PACKAGE=TRUE
    ENABLE_BINARY=FALSE
  fi
  check_param
  set_create_libs
  set_ut_mode
}

set_example_opt() {
  if [[ -n $1 && $1 != -* ]]; then
    OP_NAME=$1
    step=$((step + 1))
  fi
  if [[ -n $2 && $2 != -* ]]; then
    EXAMPLE_MODE=$2
    step=$((step + 1))
  fi
  if [[ -n $3 && $3 != -* ]]; then
    PKG_MODE=$3
    step=$((step + 1))
  fi
}

custom_cmake_args() {
  if [[ -n $COMPILED_OPS ]]; then
    COMPILED_OPS="${COMPILED_OPS//,/;}"
    CMAKE_ARGS="$CMAKE_ARGS -DASCEND_OP_NAME=${COMPILED_OPS}"
  else
    CMAKE_ARGS="$CMAKE_ARGS -UASCEND_OP_NAME"
  fi
  if [[ -n $VENDOR_NAME ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DVENDOR_NAME=${VENDOR_NAME}"
  else
    CMAKE_ARGS="$CMAKE_ARGS -UVENDOR_NAME"
  fi
}

assemble_cmake_args() {
  CMAKE_ARGS="-DBUILD_PATH=${BUILD_PATH}"
  CMAKE_ARGS="$CMAKE_ARGS -DENABLE_ASAN=${ENABLE_ASAN}"
  CMAKE_ARGS="$CMAKE_ARGS -DENABLE_VALGRIND=${ENABLE_VALGRIND}"
  CMAKE_ARGS="$CMAKE_ARGS -DENABLE_DEBUG=${ENABLE_DEBUG}"
  CMAKE_ARGS="$CMAKE_ARGS -DENABLE_TEST=${ENABLE_TEST}"
  CMAKE_ARGS="$CMAKE_ARGS -DENABLE_UT_EXEC=${ENABLE_UT_EXEC}"
  CMAKE_ARGS="$CMAKE_ARGS -DENABLE_CUSTOM=${ENABLE_CUSTOM}"
  custom_cmake_args
  if [[ "$ENABLE_CUSTOM" == "TRUE" ]]; then
    ENABLE_BINARY=TRUE
  fi
  CMAKE_ARGS="$CMAKE_ARGS -DENABLE_BINARY=${ENABLE_BINARY}"
  CMAKE_ARGS="$CMAKE_ARGS -DENABLE_PACKAGE=${ENABLE_PACKAGE}"
  CMAKE_ARGS="$CMAKE_ARGS -DBUILD_MODE=${BUILD_MODE}"
  CMAKE_ARGS="$CMAKE_ARGS -DOP_HOST_UT=${OP_HOST_UT}"
  CMAKE_ARGS="$CMAKE_ARGS -DOP_API_UT=${OP_API_UT}"
  CMAKE_ARGS="$CMAKE_ARGS -DOP_KERNEL_UT=${OP_KERNEL_UT}"
  CMAKE_ARGS="$CMAKE_ARGS -DUT_TEST_ALL=${UT_TEST_ALL}"
  if [[ -n $COMPUTE_UNIT ]]; then
    IFS=',' read -ra COMPUTE_UNIT <<<"$COMPUTE_UNIT"
    COMPUTE_UNIT_SHORT=""
    for unit in "${COMPUTE_UNIT[@]}"; do
      for support_unit in "${SUPPORT_COMPUTE_UNIT_SHORT[@]}"; do
        lowercase_word=$(echo "$unit" | tr '[:upper:]' '[:lower:]')
        if [[ "$lowercase_word" == *"$support_unit"* ]]; then
          COMPUTE_UNIT_SHORT="$COMPUTE_UNIT_SHORT$support_unit;"
          break
        fi
      done
    done

    if [[ "$COMPUTE_UNIT_SHORT" == "ascend310p;" && "$ENABLE_PACKAGE" == "TRUE" && "$ENABLE_CUSTOM" == "FALSE" ]]; then
      print_error "The soc [${COMPUTE_UNIT}] need to be used with --vendor_name."
      usage
      exit 1
    fi
    if [[ -z $COMPUTE_UNIT_SHORT ]]; then
      print_error "The soc [${COMPUTE_UNIT}] is not support."
      usage
      exit 1
    fi

    echo "COMPUTE_UNIT: ${COMPUTE_UNIT_SHORT}"
    CMAKE_ARGS="$CMAKE_ARGS -DASCEND_COMPUTE_UNIT=$COMPUTE_UNIT_SHORT"
  else
    CMAKE_ARGS="$CMAKE_ARGS -UASCEND_COMPUTE_UNIT"
  fi
}

build_lib() {
  echo $dotted_line
  echo "Start to build libs ${BUILD_LIBS[@]}"

  git submodule init && git submodule update
  cd "${BUILD_PATH}" && cmake ${CMAKE_ARGS} ..
  local all_targets=$(cmake --build . --target help)
  for lib in "${BUILD_LIBS[@]}"; do
    if echo "${all_targets}" | grep -wq "$lib"; then
      echo "Building target ${lib}"
      cmake --build . --target ${lib} -- ${VERBOSE} -j $THREAD_NUM
    else
      echo "No need to build target ${lib}"
    fi
  done

  if [[ "$ENABLE_CONVERAGE" =~ "TRUE" ]]; then
    cmake --build . --target generate_ops_cpp_cov -- -j $THREAD_NUM
  fi
  print_success "Build libs ${BUILD_LIBS[@]} success!"
}

build_binary() {
  echo $dotted_line
  echo "Start to build binary"

  mkdir -p "${BUILD_PATH}"
  echo "--------------- build tiling start ---------------"
  local tilingLib="ophost_${REPOSITORY_NAME}"
  local tilingSo="${BUILD_PATH}/lib${tilingLib}.so"
  rm -f ${tilingSo}  # 临时方案，简单实现，强制先删再编，tiling代码不变时仅浪费链接时间
  local TMP_BUILD_LIBS=${BUILD_LIBS}
  local TMP_CMAKE_ARGS=${CMAKE_ARGS}
  CMAKE_ARGS="-DBUILD_PATH=$BUILD_PATH -DENABLE_CUSTOM=FALSE"
  BUILD_LIBS+=("${tilingLib}")
  build_lib
  BUILD_LIBS=${TMP_BUILD_LIBS}
  CMAKE_ARGS=${TMP_CMAKE_ARGS}

  export ASCEND_OPP_PATH=${BUILD_PATH}/opp
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${BUILD_PATH}/common/stub/op_tiling
  local fakeTilingSoPath=${BUILD_PATH}/opp/op_impl/built-in/ai_core/tbe/op_tiling/lib/linux/$(arch)
  mkdir -p ${fakeTilingSoPath}
  ln -sf ${tilingSo} ${fakeTilingSoPath}/liboptiling.so
  ln -sf ${tilingSo} ${fakeTilingSoPath}/libopmaster_ct.so
  rm -f ${BUILD_PATH}/opp/scene.info
  echo "os=linux"     >> ${BUILD_PATH}/opp/scene.info
  echo "os_version="  >> ${BUILD_PATH}/opp/scene.info
  echo "arch=$(arch)" >> ${BUILD_PATH}/opp/scene.info
  echo "--------------- build tiling end ---------------"

  cd "${BUILD_PATH}" && cmake .. ${CMAKE_ARGS}

  echo "--------------- prepare build start ---------------"
  local all_targets=$(cmake --build . --target help)
  if echo "${all_targets}" | grep -wq "ascendc_impl_gen"; then
    cmake --build . --target ascendc_impl_gen -- ${VERBOSE} -j $THREAD_NUM
    if [ $? -ne 0 ]; then exit 1; fi
  fi
  if echo "${all_targets}" | grep -wq "gen_bin_scripts"; then
    cmake --build . --target gen_bin_scripts -- ${VERBOSE} -j $THREAD_NUM
    if [ $? -ne 0 ]; then exit 1; fi
  fi
  echo "--------------- prepare build end ---------------"

  echo "--------------- binary build start ---------------"
  if echo "${all_targets}" | grep -wq "binary"; then
    cmake --build . --target binary -- ${VERBOSE} -j1
    if [ $? -ne 0 ]; then
      print_error "Kernel compile failed!" && exit 1
    fi
  fi
  if echo "${all_targets}" | grep -wq "gen_bin_info_config"; then
    cmake --build . --target gen_bin_info_config -- ${VERBOSE} -j $THREAD_NUM
    if [ $? -ne 0 ]; then exit 1; fi
  fi

  print_success "Build binary success!"
}

build_pkg() {
  echo "--------------- build pkg start ---------------"
  if [[ "$ENABLE_BINARY" == "FALSE" ]]; then # for jit need dynamic py
    local all_targets=$(cmake --build . --target help)
    if echo "${all_targets}" | grep -wq "ascendc_impl_gen"; then
      cmake --build . --target ascendc_impl_gen -- ${VERBOSE} -j $THREAD_NUM
      if [ $? -ne 0 ]; then exit 1; fi
    fi
  fi
  cd "${BUILD_PATH}" && cmake ${CMAKE_ARGS} ..
  cmake --build . --target package -- ${VERBOSE} -j $THREAD_NUM
  echo "--------------- build pkg end ---------------"
}

parse_op_dependencies() {
  echo $dotted_line
  echo "Start to parse op dependencies for ${COMPILED_OPS}"

  cd "${BUILD_PATH}"
  python3 ${BASE_PATH}/scripts/util/dependency_parser.py --ops ${COMPILED_OPS} -p ${BUILD_PATH}
  echo $dotted_line
  echo "End to parse op dependencies"
}

set_ci_mode() {
  cd ${BASE_PATH}
  if [[ "$ENABLE_TEST" == "TRUE" ]]; then
    {
      result=$(python3 ${BASE_PATH}/scripts/util/parse_changed_files.py ${CHANGED_FILES})
    } || {
      echo $result && exit 1
    }
    for line in $result; do
      if [[ $line == op_* ]]; then
        TRIGER_UTS+=($line)
      fi
    done
  else
    {
      result=$(python3 ${BASE_PATH}/scripts/util/parse_compile_changed_files.py ${CHANGED_FILES})
    } || {
      echo $result && exit 1
    }
    args=(${result//:/ })
    CMAKE_ARGS="$CMAKE_ARGS -DASCEND_OP_NAME=${args[0]} -DASCEND_COMPILE_OPS=${args[1]}"
  fi
}

build_ut() {
  echo $dotted_line
  echo "Start to build ut"

  git submodule init && git submodule update
  if [ ! -d "${BUILD_PATH}" ]; then
    mkdir -p "${BUILD_PATH}"
  fi
  cd "${BUILD_PATH}" && cmake ${CMAKE_ARGS} ..
  if [[ "$CI_MODE" == "TRUE" ]]; then
    # ci 模式
    for triger_option in "${TRIGER_UTS[@]}"; do
      ut_args=(${triger_option//:/ })
      if [[ "${UT_TARGES[@]}" =~ "${ut_args[0]}" ]]; then
        echo "Triger Ut: ${ut_args[0]} for ops: ${ut_args[1]}"
        cmake ${CMAKE_ARGS} -DASCEND_OP_NAME=${ut_args[1]} -DASCEND_COMPILE_OPS=${ut_args[2]} ..
        cmake --build . --target ${REPOSITORY_NAME}_${ut_args[0]} -- ${VERBOSE} -j $THREAD_NUM
      else
        echo "Not need triger Ut: ${ut_args[0]}"
      fi
    done
  else
    cmake --build . --target ${UT_TARGES[@]} -- ${VERBOSE} -j $THREAD_NUM
  fi
}

build_single_example() {
  echo "Start to run example,op_name:${OP_NAME} example_name:${example} mode:${EXAMPLE_MODE}."

  if [[ "${EXAMPLE_MODE}" == "eager" ]]; then
    if [[ "${PKG_MODE}" == "cust" ]]; then
      if [[ "${VENDOR_NAME}" == "" ]]; then
        VENDOR_NAME="custom"
      fi
      export CUST_LIBRARY_PATH="${ASCEND_HOME_PATH}/opp/vendors/${VENDOR_NAME}_nn/op_api/lib"     # 仅自定义算子需要
      export CUST_INCLUDE_PATH="${ASCEND_HOME_PATH}/opp/vendors/${VENDOR_NAME}_nn/op_api/include" # 仅自定义算子需要
      g++ ${file} -I ${INCLUDE_PATH} -I ${INCLUDE_PATH}/aclnnop -I ${CUST_INCLUDE_PATH} -L ${CUST_LIBRARY_PATH} -L ${EAGER_LIBRARY_PATH} -lopapi -lcust_opapi -lascendcl -lnnopbase -o test_aclnn_${example} -Wl,-rpath=${CUST_LIBRARY_PATH}
    elif [[ "${PKG_MODE}" == "" ]]; then
      g++ ${file} -I ${INCLUDE_PATH} -I ${INCLUDE_PATH}/aclnnop -I ${ACLNN_INCLUDE_PATH} -L ${EAGER_LIBRARY_OPP_PATH} -L ${EAGER_LIBRARY_PATH} -lopapi -lopapi_nn -lascendcl -lnnopbase -o test_aclnn_${example}
    else
      usage "run_example"
      exit 1
    fi
  elif [[ "${EXAMPLE_MODE}" == "graph" ]]; then
    g++ ${file} -I ${GRAPH_INCLUDE_PATH} -I ${GE_INCLUDE_PATH} -I ${LINUX_INCLUDE_PATH} -I ${INC_INCLUDE_PATH} -L ${GRAPH_LIBRARY_STUB_PATH} -L ${GRAPH_LIBRARY_PATH} -lgraph -lge_runner -lgraph_base -o test_geir_${example}
  fi
  ${BUILD_PATH}/"${pattern}${example}" && success_example+=(${example})
}

build_example() {
  echo $dotted_line

  if [ ! -d "${BUILD_PATH}" ]; then
    mkdir -p "${BUILD_PATH}"
  fi
  cd "${BUILD_PATH}"
  if [[ "${EXAMPLE_MODE}" == "eager" ]]; then
    pattern="test_aclnn_"
  elif [[ "${EXAMPLE_MODE}" == "graph" ]]; then
    pattern="test_geir_"
  else
    usage "run_example"
    exit 1
  fi

  OLDIFS=$IFS
  IFS=$'\n'
  files=($(find ../ -path "*/${OP_NAME}/examples/*" -name "${pattern}*.cpp"))
  IFS=$OLDIFS

  failed_example=()
  success_example=()
  examples=()
  for file in "${files[@]}"; do
    example=${file#*"${pattern}"}
    example=${example%.*}
    examples+=($example)
    if [[ $EXAMPLE_NAME == "" || $EXAMPLE_NAME == $example ]]; then
      build_single_example || {
        echo "run ${pattern}${example} failed."
        failed_example+=(${example})
      }
    fi
  done

  examples=$(IFS=,; echo "${examples[*]}")
  failed_example=$(IFS=,; echo "${failed_example[*]}")
  success_example=$(IFS=,; echo "${success_example[*]}")

  if [[ ${failed_example} != "" || ${success_example} == "" ]]; then
    echo "Run example failed, op_name: ${OP_NAME}, exist examples: ${examples}, success examples: ${success_example},\
 failed_example: ${failed_example[@]}, mode: ${EXAMPLE_MODE}."
    exit 1
  fi
}

gen_op() {
  if [[ -z "$GENOP_NAME" ]] || [[ -z "$GENOP_TYPE" ]]; then
    print_error "op_class or op_name is not set."
    usage "genop"
    exit 1
  fi

  echo $dotted_line
  echo "Start to create the initial directory for ${GENOP_NAME} under ${GENOP_TYPE}"

  if [ ! -d "${GENOP_TYPE}" ]; then
    mkdir -p "${GENOP_TYPE}"
    cp examples/CMakeLists.txt "${GENOP_TYPE}/CMakeLists.txt"
    sed -i '/add_category_subdirectory()/a add_subdirectory('"${GENOP_TYPE}"')' CMakeLists.txt
  fi

  BASE_DIR=${GENOP_TYPE}/${GENOP_NAME}
  mkdir -p "${BASE_DIR}"

  cp -r examples/add_example/* "${BASE_DIR}/"

  rm -rf "${BASE_DIR}/examples"
  rm -rf "${BASE_DIR}/op_host/config"

  for file in $(find "${BASE_DIR}" -name "*.h" -o -name "*.cpp"); do
    head -n 14 "$file" > "${file}.tmp"
    cat "${file}.tmp" > "$file"
    rm "${file}.tmp"
  done

  for file in $(find "${BASE_DIR}" -type f); do
    sed -i "s/add_example/${GENOP_NAME}/g" "$file"
  done

  cd ${BASE_DIR}
  for file in $(find ./ -name "add_example*"); do
    new_file=$(echo "$file" | sed "s/add_example/${GENOP_NAME}/g")
    mv "$file" "$new_file"
  done

  echo "Create the initial directory for ${GENOP_NAME} under ${GENOP_TYPE} success"
}

gen_aicpu_op() {
if [[ -z "$GENOP_NAME" ]] || [[ -z "$GENOP_TYPE" ]]; then
    print_error "op_class or op_name is not set."
    usage "genop"
    exit 1
  fi

  echo $dotted_line
  echo "Start to create the AI CPU initial directory for ${GENOP_NAME} under ${GENOP_TYPE}"

  if [ ! -d "${GENOP_TYPE}" ]; then
    mkdir -p "${GENOP_TYPE}"
    cp examples/CMakeLists.txt "${GENOP_TYPE}/CMakeLists.txt"
    sed -i '/add_category_subdirectory()/a add_subdirectory('"${GENOP_TYPE}"')' CMakeLists.txt
  fi

  BASE_DIR=${GENOP_TYPE}/${GENOP_NAME}
  mkdir -p "${BASE_DIR}"

  cp -r examples/add_example_aicpu/* "${BASE_DIR}/"

  rm -rf "${BASE_DIR}/examples"
  rm -rf "${BASE_DIR}/op_host/config"

  for file in $(find "${BASE_DIR}" -name "*.h" -o -name "*.cpp"); do
    head -n 14 "$file" >"${file}.tmp"
    cat "${file}.tmp" >"$file"
    rm "${file}.tmp"
  done

  for file in $(find "${BASE_DIR}" -type f); do
    sed -i "s/add_example_aicpu/${GENOP_NAME}/g" "$file"
  done

  cd ${BASE_DIR}
  for file in $(find ./ -name "add_example_aicpu*"); do
    new_file=$(echo "$file" | sed "s/add_example_aicpu/${GENOP_NAME}_aicpu/g")
    mv "$file" "$new_file"
  done

  for file in $(find ./ -name "add_example*"); do
    new_file=$(echo "$file" | sed "s/add_example/${GENOP_NAME}/g")
    mv "$file" "$new_file"
  done

  echo "Create the AI CPU initial directory for ${GENOP_NAME} under ${GENOP_TYPE} success"
}

main() {
  checkopts "$@"
  if [ "$THREAD_NUM" -gt "$CORE_NUMS" ]; then
    echo "compile thread num:$THREAD_NUM over core num:$CORE_NUMS, adjust to core num"
    THREAD_NUM=$CORE_NUMS
  fi
  if [[ "$ENABLE_GENOP" == "TRUE" ]]; then
    gen_op
    exit $?
  fi
  if [[ "$ENABLE_GENOP_AICPU" == "TRUE" ]]; then
    gen_aicpu_op
    exit $?
  fi
  assemble_cmake_args
  echo "CMAKE_ARGS: ${CMAKE_ARGS}"
  mkdir -p "${BUILD_PATH}"
  cd "${BUILD_PATH}" && rm -f CMakeCache.txt && cmake ..

  if [[ "$CI_MODE" == "TRUE" ]]; then
    set_ci_mode
  elif [[ -n $COMPILED_OPS ]]; then
    parse_op_dependencies
  fi

  cd "${BUILD_PATH}" && cmake ${CMAKE_ARGS} ..

  if [[ "$ENABLE_TEST" == "TRUE" ]]; then
    build_ut
    exit $?
  fi

  if [[ "$ENABLE_RUN_EXAMPLE" == "TRUE" ]]; then
    build_example
    exit $?
  fi
  if [ "$ENABLE_CREATE_LIB" == "TRUE" ]; then
    build_lib
  fi
  if [[ "$ENABLE_BINARY" == "TRUE" || "$ENABLE_CUSTOM" == "TRUE" ]]; then
    build_binary
  fi
  if [[ "$ENABLE_PACKAGE" == "TRUE" ]]; then
    build_pkg
  fi
}

set -o pipefail
main "$@" | gawk '{print strftime("[%Y-%m-%d %H:%M:%S]"), $0}'
