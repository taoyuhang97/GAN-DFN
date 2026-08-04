#!/usr/bin/env python3
"""Periodically remind Codex to continue work from a task document."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_FILE = REPO_ROOT / "docs" / "codex_active_task.md"


TASK_TEMPLATE = """# Codex Active Task

Status: active
Updated: 2026-07-02

## Current Goal

把这里改成当前需要 Codex 持续推进的目标。

## Context

- 工作目录：`/home/tyh/projects/petroleum/code/GAN-DFN`
- 关键文件/目录：
- 重要约束：

## Done

- [ ] 这里记录已经完成的步骤，避免恢复后重复做。

## Next Step

1. 这里写 Codex 下一次恢复后应该直接执行的步骤。

## Verification

- 需要运行的检查命令：
- 已运行的检查结果：
- 尚未验证的风险：

## Resume Prompt

请继续执行本文件中的任务。先读取本文档，确认 `Current Goal`、`Done`、`Next Step` 和 `Verification`，然后从 `Next Step` 继续推进。每完成一个阶段后，更新本文档中的状态、已完成内容、下一步和验证结果。若任务已经完成，把 `Status` 改为 `done`。
"""


def parse_interval(value: str) -> float:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*", value.lower())
    if not match:
        raise argparse.ArgumentTypeError(
            "interval must look like 30s, 10m, 2h, or 1d"
        )
    amount = float(match.group(1))
    unit = match.group(2) or "s"
    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    seconds = amount * multipliers[unit]
    if seconds <= 0:
        raise argparse.ArgumentTypeError("interval must be positive")
    return seconds


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def ensure_task_file(path: Path, force: bool) -> None:
    if path.exists() and not force:
        print(f"[{now_text()}] task file already exists: {relative_or_absolute(path)}")
        print("Use --force to overwrite it.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TASK_TEMPLATE, encoding="utf-8")
    print(f"[{now_text()}] wrote task file: {relative_or_absolute(path)}")


def read_task_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"task file does not exist: {path}. Run `python scripts/codex_task_reminder.py init` first."
        )
    return path.read_text(encoding="utf-8")


def read_task_status(path: Path) -> str:
    content = read_task_file(path)
    for line in content.splitlines():
        match = re.match(r"^\s*Status\s*:\s*(\S+)\s*$", line, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().lower()
    return ""


def build_prompt(path: Path, include_task_text: bool, note: str) -> str:
    resolved = path.resolve()
    prompt_lines = [
        "请继续执行当前任务。",
        f"任务状态文档：{resolved}",
        "",
        "执行要求：",
        "1. 先读取任务状态文档，确认 Current Goal、Done、Next Step 和 Verification。",
        "2. 从 Next Step 继续，不要重复 Done 中已经完成的内容。",
        "3. 如果发现对话是从中断状态恢复，先做一次简短的上下文 sanity check。",
        "4. 每完成一个阶段后，更新任务状态文档中的 Done、Next Step、Verification 和 Updated。",
        "5. 若任务已经完成，把 Status 改为 done，并在最终回复中说明验证结果。",
    ]
    if note:
        prompt_lines.extend(["", "补充提醒：", note])
    if include_task_text:
        prompt_lines.extend(
            [
                "",
                "任务文档当前内容如下：",
                "```markdown",
                read_task_file(path).strip(),
                "```",
            ]
        )
    return "\n".join(prompt_lines).rstrip() + "\n"


def deliver_print(prompt: str) -> int:
    print(f"\n[{now_text()}] Codex reminder prompt:\n")
    print(prompt)
    return 0


def deliver_tmux(prompt: str, target: str) -> int:
    if not target:
        print("tmux mode requires --tmux-target, for example: main:0.1", file=sys.stderr)
        return 2
    load = subprocess.run(["tmux", "load-buffer", "-"], input=prompt, text=True)
    if load.returncode != 0:
        return load.returncode
    paste = subprocess.run(["tmux", "paste-buffer", "-t", target, "-d"])
    if paste.returncode != 0:
        return paste.returncode
    enter = subprocess.run(["tmux", "send-keys", "-t", target, "Enter"])
    return enter.returncode


def deliver_codex_exec_resume(
    prompt: str,
    session: str | None,
    all_sessions: bool,
    codex_cwd: Path,
) -> int:
    cmd = ["codex", "exec", "resume"]
    if all_sessions:
        cmd.append("--all")
    if session:
        cmd.append(session)
    else:
        cmd.append("--last")
    cmd.append("-")
    print(f"[{now_text()}] running: {' '.join(cmd)}")
    completed = subprocess.run(cmd, input=prompt, text=True, cwd=str(codex_cwd))
    return completed.returncode


def deliver(args: argparse.Namespace, prompt: str) -> int:
    if args.dry_run:
        return deliver_print(prompt)
    if args.mode == "print":
        return deliver_print(prompt)
    if args.mode == "tmux":
        return deliver_tmux(prompt, args.tmux_target)
    if args.mode == "codex-exec-resume":
        return deliver_codex_exec_resume(
            prompt=prompt,
            session=args.session,
            all_sessions=args.all_sessions,
            codex_cwd=Path(args.codex_cwd).resolve(),
        )
    print(f"unknown mode: {args.mode}", file=sys.stderr)
    return 2


def add_delivery_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--task-file",
        default=str(DEFAULT_TASK_FILE),
        help=f"Markdown task file to read. Default: {relative_or_absolute(DEFAULT_TASK_FILE)}",
    )
    parser.add_argument(
        "--mode",
        choices=["print", "tmux", "codex-exec-resume"],
        default="print",
        help="Where to send the reminder.",
    )
    parser.add_argument(
        "--include-task-text",
        action="store_true",
        help="Include the full task document content in the prompt.",
    )
    parser.add_argument("--note", default="", help="Extra one-off note appended to the prompt.")
    parser.add_argument(
        "--tmux-target",
        default="",
        help="tmux pane target used by --mode tmux, for example: main:0.1",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Codex session id/name for --mode codex-exec-resume. Default uses --last.",
    )
    parser.add_argument(
        "--all-sessions",
        action="store_true",
        help="With codex-exec-resume, allow selecting the latest session outside the current cwd filter.",
    )
    parser.add_argument(
        "--codex-cwd",
        default=str(REPO_ROOT),
        help="Working directory passed to codex exec resume.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print instead of sending.")


def run_once(args: argparse.Namespace) -> int:
    task_file = Path(args.task_file).resolve()
    read_task_file(task_file)
    prompt = build_prompt(task_file, args.include_task_text, args.note)
    return deliver(args, prompt)


def run_watch(args: argparse.Namespace) -> int:
    task_file = Path(args.task_file).resolve()
    read_task_file(task_file)
    run_count = 0
    try:
        if not args.run_immediately:
            time.sleep(args.interval)
        while args.max_runs is None or run_count < args.max_runs:
            if args.stop_when_done and read_task_status(task_file) == "done":
                print(f"[{now_text()}] task status is done; stopping reminders")
                break
            prompt = build_prompt(task_file, args.include_task_text, args.note)
            code = deliver(args, prompt)
            run_count += 1
            if code != 0:
                print(f"[{now_text()}] reminder delivery failed with exit code {code}", file=sys.stderr)
                if args.stop_on_error:
                    return code
            if args.max_runs is not None and run_count >= args.max_runs:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\n[{now_text()}] stopped by user")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remind Codex to continue from a Markdown task document."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create the task document template.")
    init_parser.add_argument(
        "--task-file",
        default=str(DEFAULT_TASK_FILE),
        help=f"Task file to create. Default: {relative_or_absolute(DEFAULT_TASK_FILE)}",
    )
    init_parser.add_argument("--force", action="store_true", help="Overwrite an existing task file.")

    once_parser = subparsers.add_parser("once", help="Send one reminder.")
    add_delivery_args(once_parser)

    watch_parser = subparsers.add_parser("watch", help="Send reminders on an interval.")
    add_delivery_args(watch_parser)
    watch_parser.add_argument(
        "--interval",
        type=parse_interval,
        default=parse_interval("10m"),
        help="Reminder interval, for example 30s, 10m, 2h. Default: 10m.",
    )
    watch_parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Stop after this many reminders. Default: run until Ctrl-C.",
    )
    watch_parser.add_argument(
        "--run-immediately",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send one reminder immediately before waiting for the interval.",
    )
    watch_parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the watch loop if reminder delivery fails.",
    )
    watch_parser.add_argument(
        "--stop-when-done",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop the watch loop when the task document contains `Status: done`.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "init":
        ensure_task_file(Path(args.task_file).resolve(), args.force)
        return 0
    if args.command == "once":
        return run_once(args)
    if args.command == "watch":
        return run_watch(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
