#!/bin/bash


cd ${WORKSPACE}
echo $(grep -E "^VERSION_ID=" /etc/os-release | cut -d'"' -f2)
if [[ "${task_name}" == *ubuntu24* ]]; then
    sudo update-alternatives --set gcc /usr/bin/gcc-14
else
    if [[ -f "/opt/rh/devtoolset-7/enable" ]]; then
        echo "source devtoolset"
        source /opt/rh/devtoolset-7/enable
    fi
fi
gcc --version
source /home/jenkins/Ascend/cann/bin/setenv.bash
set +e
case "${task_name}" in
    Pre_compile)
        bash build.sh --pkg --ops="fatrelu_mul" --cann_3rd_lib_path=/home/jenkins/opensource
        echo "build fatrelu_mul"
        ls build_out
        mv build_out/*.run ${WORKSPACE}/build_out/cann-ops-nn-fatrelu_mul_linux-aarch64.run
        ls build_out
        exit 0
        ;;
    compile_single*)
        if [ "${target_branch}" = "master" ];then
            export ASCEND_3RD_LIB_PATH=/home/jenkins/opensource
            bash scripts/ci/check_pkg.sh "pr_filelist.txt" "-j16"
            echo "exec cmd: [bash scripts/ci/check_pkg.sh pr_filelist.txt]"
        else
            echo "not need build single"
            mkdir single
            touch single.tar.gz
        fi
        ;;
    arm_compile*)
        bash build.sh --pkg --jit --cann_3rd_lib_path=/home/jenkins/opensource -j16
        echo "exec cmd: [bash build.sh --pkg --jit --cann_3rd_lib_path=/home/jenkins/opensource -j16]"
        exit 0
        ;;  
    Compile_Ascend_experimental)
        sh scripts/ci/check_experimental_pkg.sh "pr_filelist.txt"
        echo "exec cmd: [sh scripts/ci/check_experimental_pkg.sh pr_filelist.txt]"
        if [ ! -f "build_out/"*.run ]; then
            mkdir -p build_out
            touch build_out/cann-ops-nn-experimental_linux-aarch64.run
        fi
        exit 0
        ;;
    Compile_Ascend_ARM_950)
        export ASCEND_3RD_LIB_PATH=/home/jenkins/opensource
        bash scripts/ci/compile_ascend950_pkg.sh "pr_filelist.txt" "-j16" "-force_jit" "--no_force"
        compile_package_name=$(ls "${WORKSPACE}/build_out/" |grep -E "*.run$"|head -n1)
        if [[ -z "${compile_package_name}" ]]; then
            echo "not need build 950"
            mkdir build_out
            touch build_out/cann-ops-nn-950_linux-aarch64.run
        fi
        exit 0
        ;;
esac

if [ ! -f "build_out/"*.run ]; then
    mkdir -p build_out
    touch build_out/cann-ops-nn-test_linux-aarch64.run
fi