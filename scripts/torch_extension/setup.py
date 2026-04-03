#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import os
import shutil
import subprocess
import logging
import sys
from setuptools import setup, find_packages, Distribution, Command
from wheel.bdist_wheel import bdist_wheel

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
PACKAGE_NAME = "ascend_ops_nn"
VERSION = "1.0.0"
DESCRIPTION = "PyTorch extensions for AscendC"


class CleanCommand(Command):
    """
    usage: python setup.py clean
    """
    description = "Clean build artifacts from the source tree"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        folders_to_remove = ['build', 'dist', f'{PACKAGE_NAME}.egg-info']
        for folder in folders_to_remove:
            if os.path.exists(folder):
                shutil.rmtree(folder)
                logging.info(f"Removed folder: {folder}")
        for root, _, files in os.walk('.'):
            for file in files:
                if file.endswith(('.pyc', '.pyo')):
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    logging.info(f"Removed file: {file_path}")
        logging.info("Cleaned build artifacts.")


class BinaryDistribution(Distribution):
    """
    Make this wheel not a pure python package
    """
    def is_pure(self):
        return False

    def has_ext_modules(self):
        return True


class ABI3Wheel(bdist_wheel):
    """
    Force to use abi3 tag for wheel, this wheel supports multiple python versions >= 3.8
    """
    def get_tag(self):
        python, abi, plat = super().get_tag()
        python = f"cp{sys.version_info.major}{sys.version_info.minor}"
        abi = "abi3"
        return python, abi, plat

    def run(self):
        self.run_command('cmake_build')
        super().run()


class CMakeBuildCommand(Command):
    """
    Custom command to build CMake extensions
    """
    description = "Build CMake extensions"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """
        This file `setup.py` and the CMakeLists.txt are in the same directory.
        Use multi-core to speed up compilation.
        """
        build_jobs = os.environ.get('PE_BUILD_JOBS', '8')
        logging.info(f"build jobs: {build_jobs}")
        # Get Torch and Torch NPU paths
        import torch
        torch_cmake_path = torch.utils.cmake_prefix_path
        torch_dir = os.path.join(torch_cmake_path, "Torch")
        logging.info(f"Using Torch path: {torch_dir}")
        import torch_npu
        torch_npu_path = os.path.dirname(torch_npu.__file__)
        logging.info(f"Using Torch NPU path: {torch_npu_path}")

        # Get NPU_ARCH from environment variable or set default
        npu_arch = os.environ.get('NPU_ARCH', 'ascend910b')
        logging.info(f"Using NPU_ARCH: {npu_arch}")

        # Build the CMake project
        build_temp = os.path.join(os.getcwd(), 'build')
        cmake_config_command = ['cmake', '-S', os.getcwd(), '-B', build_temp,
                                '-DCMAKE_BUILD_TYPE=Release',
                                f'-DTorch_DIR={torch_dir}',
                                f'-DTORCH_NPU_PATH={torch_npu_path}',
                                f'-DNPU_ARCH={npu_arch}'
                                ]
        subprocess.check_call(cmake_config_command, cwd=os.getcwd())
        subprocess.check_call(
            ['cmake', '--build', build_temp, '--config', 'Release', '--parallel', build_jobs], cwd=os.getcwd())
        logging.info("CMake extensions built successfully.")


cmdclass = {
    'clean': CleanCommand,
    'bdist_wheel': ABI3Wheel,
    'cmake_build': CMakeBuildCommand,
}


setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    packages=find_packages(),
    package_data={PACKAGE_NAME: ['*.so']},
    distclass=BinaryDistribution,
    cmdclass=cmdclass,
    zip_safe=False,
    install_requires=[
        "torch",
        "torch_npu"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires='>=3.8',
)
