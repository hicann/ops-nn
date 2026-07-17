# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import shutil
import glob
import logging
from setuptools import setup, find_packages
from setuptools import Command
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.sdist import sdist as _sdist

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:
    _bdist_wheel = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DESCRIPTION = "AscendOpsNn"
VERSION = "1.0.0"
PACKAGE_NAME = "cann_ops_nn"

HERE = os.path.dirname(os.path.abspath(__file__))
OPS_NN_ROOT = os.path.normpath(os.path.join(HERE, ".."))
_src_path = os.path.join(HERE, PACKAGE_NAME)

_extra_packages = []
for _subdir in ("csrc", "common"):
    _root = os.path.join(_src_path, _subdir)
    if os.path.isdir(_root):
        for root, _, _ in os.walk(_root):
            rel = os.path.relpath(root, HERE).replace(os.sep, ".")
            _extra_packages.append(rel)


def _non_python_files(directory):
    paths = []
    for path, dirs, filenames in os.walk(directory):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for filename in filenames:
            if filename.endswith((".h", ".cpp", ".md")):
                paths.append(os.path.join(path, filename))
    return paths


# --- 收集 ops-nn/ 下所有 torch_extension/ 中的算子文件 ---
_op_category_inits = {}
_op_py_files = []
_op_cpp_files = []

for cat in sorted(os.listdir(OPS_NN_ROOT)):
    cat_path = os.path.join(OPS_NN_ROOT, cat)
    if not os.path.isdir(cat_path) or cat.startswith((".", "_")):
        continue

    cat_init_src = os.path.join(cat_path, "__init__.py")
    if os.path.isfile(cat_init_src):
        _op_category_inits[cat] = cat_init_src

    for name in sorted(os.listdir(cat_path)):
        torch_extension = os.path.join(cat_path, name, "torch_extension")
        if not os.path.isdir(torch_extension):
            continue

        for f in ("__init__.py", "%s.py" % name):
            f_src = os.path.join(torch_extension, f)
            if os.path.isfile(f_src):
                _op_py_files.append((os.path.join("ops", cat, name, f), f_src))

        graph_src = os.path.join(torch_extension, "graph_convert_%s.py" % name)
        if os.path.isfile(graph_src):
            _op_py_files.append(
                (
                    os.path.join("ops", cat, name, "graph_convert_%s.py" % name),
                    graph_src,
                )
            )

        csrc_dir = os.path.join(torch_extension, "csrc")
        if os.path.isdir(csrc_dir):
            for cpp in os.listdir(csrc_dir):
                if cpp.endswith(".cpp"):
                    _op_cpp_files.append(
                        (os.path.join("csrc", cat, cpp), os.path.join(csrc_dir, cpp))
                    )


class Clean(Command):
    description = "clean build artifacts"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for pattern in ("build", "dist", "*.egg-info"):
            for path in glob.glob(os.path.join(HERE, pattern)):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    logger.info("removing %s", path)


class BuildPyWithOps(_build_py):
    def run(self):
        _clean_build_artifacts()
        super().run()
        build_pkg = os.path.join(self.build_lib, PACKAGE_NAME)
        for category, init_src in _op_category_inits.items():
            dst = os.path.join(build_pkg, "ops", category, "__init__.py")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(init_src, dst)
        for rel_path, src_abs in _op_py_files:
            dst = os.path.join(build_pkg, rel_path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src_abs, dst)
        for rel_path, src_abs in _op_cpp_files:
            dst = os.path.join(build_pkg, rel_path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src_abs, dst)


class SdistWithClean(_sdist):
    def run(self):
        _clean_build_artifacts()
        super().run()


def _clean_build_artifacts():
    for pattern in ("build", "*.egg-info"):
        for path in glob.glob(os.path.join(HERE, pattern)):
            if os.path.isdir(path):
                shutil.rmtree(path)
                logger.info("removing %s", path)


if _bdist_wheel is not None:

    class BdistWheelWithClean(_bdist_wheel):
        def run(self):
            super().run()
            _clean_build_artifacts()
else:
    BdistWheelWithClean = None


_cmdclass = {
    "build_py": BuildPyWithOps,
    "sdist": SdistWithClean,
    "clean": Clean,
}
if BdistWheelWithClean is not None:
    _cmdclass["bdist_wheel"] = BdistWheelWithClean


setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    author="CANN",
    packages=find_packages() + _extra_packages,
    include_package_data=True,
    package_data={PACKAGE_NAME: _non_python_files(_src_path)},
    cmdclass=_cmdclass,
    zip_safe=False,
)
