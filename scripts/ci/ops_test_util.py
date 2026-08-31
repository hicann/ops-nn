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

# Diagnostics go to stderr; the summary table goes to stdout
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stderr
)
logger = logging.getLogger(__name__)

table_handler = logging.StreamHandler(sys.stdout)
table_handler.setFormatter(logging.Formatter("%(message)s"))
table_logger = logging.getLogger("table_output")
table_logger.addHandler(table_handler)
table_logger.setLevel(logging.INFO)
table_logger.propagate = False

SUMMARY_HEADER = "op_name,testcase_name,test_type,result_csv,status"
SUMMARY_FILES = ["kernel_summary.csv", "aclnn_summary.csv", "e2e_summary.csv"]
COL_WIDTHS = {"op": 20, "testcase": 70, "type": 8, "status": 8}


def _read_csv(csv_path):
    """Return (headers, rows) of a csv file, or (None, None) on error."""
    try:
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            headers = next(reader)
            return headers, list(reader)
    except Exception as e:
        logger.error(f"Failed to read {csv_path}: {e}")
        return None, None


def _precision_index(headers):
    """Return index of the precision_status column, or -1 if missing."""
    try:
        return headers.index("precision_status")
    except ValueError:
        return -1


def check_precision(result_csv, op_name=None, testcase_name=None):
    """Check precision statuses of a result csv.

    Returns 0 if every case passes, 1 otherwise. op_name and testcase_name are
    accepted for CLI compatibility with the callers and not used directly.
    """
    if not os.path.exists(result_csv):
        logger.warning(f"Result csv file not found: {result_csv}")
        return 1

    headers, rows = _read_csv(result_csv)
    if headers is None:
        return 1

    idx = _precision_index(headers)
    if idx == -1:
        logger.warning("precision_status column not found in result csv")
        return 1

    total = 0
    passed = 0
    for row in rows:
        if len(row) <= idx:
            continue
        total += 1
        if row[idx] == "PASS":
            passed += 1
    return 0 if passed == total else 1


def summarize(result_csv, op_name, test_type, summary_file):
    """Append rows of a result csv to the type-specific summary csv."""
    if not os.path.exists(result_csv):
        return

    headers, rows = _read_csv(result_csv)
    if headers is None:
        return

    idx = _precision_index(headers)
    if not os.path.exists(summary_file):
        with open(summary_file, "w") as f:
            f.write(SUMMARY_HEADER + "\n")

    with open(summary_file, "a") as f:
        for row in rows:
            if len(row) == 0:
                continue
            status = "PASS" if idx == -1 else row[idx] if len(row) > idx else "FAIL"
            f.write(f"{op_name},{row[0]},{test_type},{result_csv},{status}\n")


def print_table(log_path):
    """Print a formatted table of failed cases to stdout."""
    all_rows = []
    for sf in SUMMARY_FILES:
        filepath = os.path.join(log_path, sf)
        if not os.path.exists(filepath):
            continue
        try:
            with open(filepath, "r") as f:
                all_rows.extend(list(csv.DictReader(f)))
        except Exception as e:
            logger.error(f"Failed to read {filepath}: {e}")

    if not all_rows:
        logger.warning("No summary data found")
        return

    total = len(all_rows)
    passed = sum(1 for r in all_rows if r.get("status", "").upper() == "PASS")
    failed = total - passed

    _print_title()
    if failed > 0:
        _print_failed_rows(all_rows)
    _print_totals(total, passed, failed)


def _print_title():
    width = sum(COL_WIDTHS.values()) + len(COL_WIDTHS) + 1
    table_logger.info("")
    table_logger.info("=" * width)
    table_logger.info("{:^{}}".format("PRECISION TEST RESULTS SUMMARY", width - 2))
    table_logger.info("=" * width)


def _print_separator():
    line = "+" + "+".join("-" * w for w in COL_WIDTHS.values()) + "+"
    table_logger.info(line)


def _print_header():
    _print_separator()
    header = "| {:^18} | {:^68} | {:^6} | {:^6} |".format(
        "Op Name", "Testcase Name", "Type", "Status"
    )
    table_logger.info(header)
    _print_separator()


def _print_row(op, testcase, test_type, status):
    if status.upper() == "PASS":
        status_display = "\033[32mPASS\033[0m"
    else:
        status_display = "\033[31mFAIL\033[0m"

    limit = COL_WIDTHS["testcase"]
    if len(testcase) <= limit:
        tc_display = testcase
    else:
        tc_display = testcase[:35] + "..." + testcase[-32:]

    row = "| {:<18} | {:<68} | {:^6} | {:^6} |".format(
        op, tc_display, test_type, status_display
    )
    table_logger.info(row)


def _print_failed_rows(all_rows):
    _print_header()
    for row in all_rows:
        if row.get("status", "").upper() == "PASS":
            continue
        _print_row(
            row.get("op_name", ""),
            row.get("testcase_name", ""),
            row.get("test_type", ""),
            row.get("status", ""),
        )


def _print_totals(total, passed, failed):
    _print_separator()
    pass_rate = (passed / total * 100) if total > 0 else 0.0
    content = f"TOTAL: {total:>3}  | PASSED: {passed:>3}  | FAILED: {failed:>3}  | PASS RATE: {pass_rate:.2f}%"
    inner_width = sum(COL_WIDTHS.values()) + len(COL_WIDTHS) - 1
    summary_line = f"| {content:<{inner_width}} |"
    table_logger.info(summary_line)
    _print_separator()


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
        sys.exit(check_precision(args.result_csv, args.op_name, args.testcase_name))
    elif args.action == "summarize":
        summarize(args.result_csv, args.op_name, args.test_type, args.summary_file)
    elif args.action == "print_table":
        print_table(args.log_path)


if __name__ == "__main__":
    main()
