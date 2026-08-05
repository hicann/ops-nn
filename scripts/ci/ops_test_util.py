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
import argparse
import csv
import logging
import os
import sys
from dataclasses import dataclass

# 日志配置：错误/警告输出到stderr
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stderr
)
logger = logging.getLogger(__name__)

# 表格专用logger：输出到stdout，保持表格格式
table_logger = logging.getLogger("table_output")
table_handler = logging.StreamHandler(sys.stdout)
table_handler.setFormatter(logging.Formatter("%(message)s"))
table_logger.addHandler(table_handler)
table_logger.setLevel(logging.INFO)
table_logger.propagate = False


@dataclass
class SummaryRowData:
    """汇总行数据封装

    用于封装写入汇总时所需的参数，避免函数参数过多
    """

    rows: list
    op_name: str
    test_type: str
    result_csv: str
    summary_file: str
    precision_idx: int


@dataclass
class SingleRowData:
    """单行数据封装

    用于封装写入单行时所需的参数
    """

    out_f: object
    row: list
    op_name: str
    test_type: str
    result_csv: str
    precision_idx: int


@dataclass
class TableRowData:
    """表格行数据封装

    用于封装打印表格行时所需的参数
    """

    op: str
    testcase: str
    test_type: str
    status: str


class OpTestUtil:
    """OPS测试工具主类

    整合了精度检查、结果汇总和表格打印功能

    日志设计：
    - logger: 用于错误/警告信息，输出到stderr
    - table_logger: 用于表格可视化输出，输出到stdout
    """

    col_widths = {
        "op": 20,
        "testcase": 70,
        "type": 8,
        "status": 8,
    }

    def __init__(self):
        pass

    @staticmethod
    def check(result_csv, op_name, testcase_name):
        """检查精度状态

        Args:
            result_csv: 结果CSV文件路径
            op_name: 算子名称
            testcase_name: 测试用例名称

        Returns:
            int: 0表示全部通过，1表示有失败用例
        """
        if not OpTestUtil._validate_file(result_csv):
            return 1

        try:
            with open(result_csv, "r") as f:
                reader = csv.reader(f)
                headers = next(reader)
                precision_idx = OpTestUtil._find_precision_column(headers)

                if precision_idx == -1:
                    logger.warning("precision_status column not found in result csv")
                    return 1

                total_cases, passed_cases = OpTestUtil._count_results(
                    reader, precision_idx
                )

                if total_cases - passed_cases > 0:
                    return 1
                return 0
        except Exception as e:
            logger.error(f"Failed to parse result csv: {e}")
            return 1

    @staticmethod
    def summarize(result_csv, op_name, test_type, summary_file):
        """汇总测试结果

        Args:
            result_csv: 结果CSV文件路径
            op_name: 算子名称
            test_type: 测试类型 (kernel/aclnn/e2e)
            summary_file: 汇总CSV文件路径
        """
        if not os.path.exists(result_csv):
            return

        OpTestUtil._ensure_summary_file(summary_file)
        OpTestUtil._process_csv(result_csv, op_name, test_type, summary_file)

    @staticmethod
    def print_table(log_path):
        """打印可视化表格

        Args:
            log_path: 日志目录路径
        """
        summary_files = ["kernel_summary.csv", "aclnn_summary.csv", "e2e_summary.csv"]

        all_rows = OpTestUtil._load_summary_data(log_path, summary_files)

        if not all_rows:
            logger.warning("No summary data found")
            return

        total = len(all_rows)
        passed = sum(1 for r in all_rows if r.get("status", "").upper() == "PASS")
        failed = total - passed

        OpTestUtil._print_title_section()

        if failed > 0:
            OpTestUtil._print_failed_rows(all_rows)

        OpTestUtil._print_summary(total, passed, failed)

    @staticmethod
    def check_precision(result_csv, op_name, testcase_name):
        """检查精度状态

        Args:
            result_csv: 结果CSV文件路径
            op_name: 算子名称
            testcase_name: 测试用例名称

        Returns:
            int: 0表示全部通过，1表示有失败用例
        """
        return OpTestUtil.check(result_csv, op_name, testcase_name)

    @staticmethod
    def summarize_results(result_csv, op_name, test_type, summary_file):
        """汇总测试结果

        Args:
            result_csv: 结果CSV文件路径
            op_name: 算子名称
            test_type: 测试类型 (kernel/aclnn/e2e)
            summary_file: 汇总CSV文件路径
        """
        OpTestUtil.summarize(result_csv, op_name, test_type, summary_file)

    @staticmethod
    def print_summary_table(log_path):
        """打印可视化表格

        Args:
            log_path: 日志目录路径
        """
        OpTestUtil.print_table(log_path)

    @staticmethod
    def _validate_file(result_csv):
        """验证文件是否存在

        Args:
            result_csv: 结果CSV文件路径

        Returns:
            bool: 文件存在返回True
        """
        if not os.path.exists(result_csv):
            logger.warning(f"Result csv file not found: {result_csv}")
            return False
        return True

    @staticmethod
    def _find_precision_column(headers):
        """查找precision_status列索引

        Args:
            headers: CSV表头列表

        Returns:
            int: 列索引，未找到返回-1
        """
        try:
            return headers.index("precision_status")
        except ValueError:
            return -1

    @staticmethod
    def _count_results(reader, precision_idx):
        """统计测试结果

        Args:
            reader: CSV reader对象
            precision_idx: precision_status列索引

        Returns:
            tuple: (总数, 通过数)
        """
        total_cases = 0
        passed_cases = 0

        for row in reader:
            if len(row) <= precision_idx:
                continue
            total_cases += 1
            if row[precision_idx] == "PASS":
                passed_cases += 1

        return total_cases, passed_cases

    @staticmethod
    def _ensure_summary_file(summary_file):
        """确保汇总文件存在并写入表头

        Args:
            summary_file: 汇总文件路径
        """
        if not os.path.exists(summary_file):
            summary_header = "op_name,testcase_name,test_type,result_csv,status"
            with open(summary_file, "w") as f:
                f.write(summary_header + "\n")

    @staticmethod
    def _read_csv_rows(result_csv):
        """读取CSV文件

        Args:
            result_csv: CSV文件路径

        Returns:
            tuple: (headers, rows) 或 (None, None)
        """
        try:
            with open(result_csv, "r") as f:
                reader = csv.reader(f)
                headers = next(reader)
                rows = list(reader)
                return headers, rows
        except Exception as e:
            logger.error(f"Failed to read {result_csv}: {e}")
            return None, None

    @staticmethod
    def _find_column_indices(headers):
        """查找precision_status列索引

        Args:
            headers: CSV表头列表

        Returns:
            int: precision_status列索引，未找到返回-1
        """
        try:
            return headers.index("precision_status")
        except ValueError:
            return -1

    @staticmethod
    def _get_status(row, precision_idx):
        """获取状态值

        Args:
            row: 数据行
            precision_idx: precision_status列索引

        Returns:
            str: 状态值
        """
        if precision_idx == -1:
            return "PASS"
        if precision_idx >= 0 and len(row) > precision_idx:
            return row[precision_idx]
        return "FAIL"

    @staticmethod
    def _write_single_row(single_data):
        """写入单行数据

        Args:
            single_data: SingleRowData数据封装对象
        """
        tc_name = single_data.row[0]
        status = OpTestUtil._get_status(single_data.row, single_data.precision_idx)

        single_data.out_f.write(
            f"{single_data.op_name},{tc_name},{single_data.test_type},"
            f"{single_data.result_csv},{status}\n"
        )

    @staticmethod
    def _process_csv(result_csv, op_name, test_type, summary_file):
        """处理CSV文件并写入汇总

        Args:
            result_csv: 结果CSV文件路径
            op_name: 算子名称
            test_type: 测试类型
            summary_file: 汇总文件路径
        """
        try:
            headers, rows = OpTestUtil._read_csv_rows(result_csv)
            if headers is None:
                return

            precision_idx = OpTestUtil._find_column_indices(headers)

            row_data = SummaryRowData(
                rows=rows,
                op_name=op_name,
                test_type=test_type,
                result_csv=result_csv,
                summary_file=summary_file,
                precision_idx=precision_idx,
            )
            OpTestUtil._write_summary_rows(row_data)
        except Exception as e:
            logger.error(f"Failed to process {result_csv}: {e}")

    @staticmethod
    def _write_summary_rows(row_data):
        """写入汇总行数据

        Args:
            row_data: SummaryRowData数据封装对象
        """
        with open(row_data.summary_file, "a") as out_f:
            for row in row_data.rows:
                if len(row) == 0:
                    continue

                single_data = SingleRowData(
                    out_f=out_f,
                    row=row,
                    op_name=row_data.op_name,
                    test_type=row_data.test_type,
                    result_csv=row_data.result_csv,
                    precision_idx=row_data.precision_idx,
                )
                OpTestUtil._write_single_row(single_data)

    @staticmethod
    def _read_summary_file(filepath):
        """读取单个汇总文件

        Args:
            filepath: 文件路径

        Returns:
            list: 数据行列表
        """
        try:
            with open(filepath, "r") as f:
                reader = csv.DictReader(f)
                return list(reader)
        except Exception as e:
            logger.error(f"Failed to read {filepath}: {e}")
            return []

    @staticmethod
    def _print_title_section():
        """打印标题区域

        使用table_logger输出到stdout，保持表格格式
        """
        width = sum(OpTestUtil.col_widths.values()) + len(OpTestUtil.col_widths) + 1
        table_logger.info("")
        table_logger.info("=" * width)
        table_logger.info("{:^{}}".format("PRECISION TEST RESULTS SUMMARY", width - 2))
        table_logger.info("=" * width)

    @staticmethod
    def _load_summary_data(log_path, summary_files):
        """加载汇总数据

        Args:
            log_path: 日志目录路径
            summary_files: 汇总文件列表

        Returns:
            list: 所有数据行
        """
        all_rows = []
        for sf in summary_files:
            filepath = os.path.join(log_path, sf)
            if os.path.exists(filepath):
                rows = OpTestUtil._read_summary_file(filepath)
                all_rows.extend(rows)
        return all_rows

    @staticmethod
    def _print_separator():
        """打印分隔线

        使用table_logger输出到stdout
        """
        line = "+" + "+".join("-" * w for w in OpTestUtil.col_widths.values()) + "+"
        table_logger.info(line)

    @staticmethod
    def _print_header():
        """打印表头

        使用table_logger输出到stdout
        """
        OpTestUtil._print_separator()
        header = "| {:^18} | {:^68} | {:^6} | {:^6} |".format(
            "Op Name",
            "Testcase Name",
            "Type",
            "Status",
        )
        table_logger.info(header)
        OpTestUtil._print_separator()

    @staticmethod
    def _print_row(row_data):
        """打印单行数据

        Args:
            row_data: TableRowData数据封装对象

        使用table_logger输出到stdout
        """
        if row_data.status.upper() == "PASS":
            status_display = "\033[32mPASS\033[0m"
        else:
            status_display = "\033[31mFAIL\033[0m"

        tc_display = (
            row_data.testcase
            if len(row_data.testcase) <= OpTestUtil.col_widths["testcase"]
            else row_data.testcase[:35] + "..." + row_data.testcase[-32:]
        )

        row = "| {:<18} | {:<68} | {:^6} | {:^6} |".format(
            row_data.op,
            tc_display,
            row_data.test_type,
            status_display,
        )
        table_logger.info(row)

    @staticmethod
    def _print_failed_rows(all_rows):
        """打印失败的行数据

        Args:
            all_rows: 所有数据行列表

        使用table_logger输出到stdout
        """
        OpTestUtil._print_header()

        failed_rows = [r for r in all_rows if r.get("status", "").upper() != "PASS"]
        for row in failed_rows:
            row_data = TableRowData(
                op=row.get("op_name", ""),
                testcase=row.get("testcase_name", ""),
                test_type=row.get("test_type", ""),
                status=row.get("status", ""),
            )
            OpTestUtil._print_row(row_data)

    @staticmethod
    def _print_summary(total, passed, failed):
        """打印汇总统计

        Args:
            total: 总数
            passed: 通过数
            failed: 失败数

        使用table_logger输出到stdout
        """
        OpTestUtil._print_separator()
        pass_rate = (passed / total * 100) if total > 0 else 0.0
        content = f"TOTAL: {total:>3}  | PASSED: {passed:>3}  | FAILED: {failed:>3}  | PASS RATE: {pass_rate:.2f}%"
        inner_width = (
            sum(OpTestUtil.col_widths.values()) + len(OpTestUtil.col_widths) - 1
        )
        summary_line = f"| {content:<{inner_width}} |"
        table_logger.info(summary_line)
        OpTestUtil._print_separator()


def main():
    parser = argparse.ArgumentParser(description="OPS Test Utilities")
    parser.add_argument(
        "--action",
        required=True,
        choices=["check_precision", "summarize", "print_table"],
        help="Action to perform",
    )
    parser.add_argument("--result_csv", help="Result CSV file path")
    parser.add_argument("--op_name", help="Operator name")
    parser.add_argument("--testcase_name", help="Testcase name")
    parser.add_argument("--test_type", help="Test type (kernel/aclnn/e2e)")
    parser.add_argument("--summary_file", help="Summary CSV file path")
    parser.add_argument("--log_path", help="Log directory path")

    args = parser.parse_args()

    if args.action == "check_precision":
        ret = OpTestUtil.check_precision(
            args.result_csv, args.op_name, args.testcase_name
        )
        sys.exit(ret)
    elif args.action == "summarize":
        OpTestUtil.summarize_results(
            args.result_csv, args.op_name, args.test_type, args.summary_file
        )
    elif args.action == "print_table":
        OpTestUtil.print_summary_table(args.log_path)


if __name__ == "__main__":
    main()
