"""CLI entry point for explicit, offline Phoenix trace evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import TextIO

from adaptive.observability.phoenix_eval import (
    MAX_SPANS_PER_RUN,
    PhoenixEvalSettings,
    PhoenixEvaluationError,
    TraceSelection,
    evaluate_selected_traces,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adaptive-phoenix-eval",
        description="Explicitly evaluate selected content-bearing Phoenix traces.",
    )
    parser.add_argument("--project", required=True, help="Phoenix project identifier")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--trace-id",
        dest="trace_ids",
        action="append",
        help="Trace ID to evaluate; repeat for multiple traces",
    )
    selection.add_argument(
        "--since",
        help="Inclusive ISO-8601 start for a bounded time selection (requires --until)",
    )
    parser.add_argument(
        "--until",
        help="Exclusive ISO-8601 end for a bounded time selection (requires --since)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=MAX_SPANS_PER_RUN,
        help=f"Maximum source spans to read (1-{MAX_SPANS_PER_RUN})",
    )
    return parser


def parse_selection(args: argparse.Namespace) -> TraceSelection:
    if args.trace_ids:
        if args.until:
            raise PhoenixEvaluationError("trace_ids_or_time_range_required")
        return TraceSelection(
            project=args.project,
            trace_ids=tuple(args.trace_ids),
            limit=args.limit,
        )
    if not args.since or not args.until:
        raise PhoenixEvaluationError("bounded_time_range_required")
    return TraceSelection(
        project=args.project,
        start_time=_parse_time(args.since),
        end_time=_parse_time(args.until),
        limit=args.limit,
    )


def _parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PhoenixEvaluationError("invalid_time_range") from exc
    if parsed.tzinfo is None:
        raise PhoenixEvaluationError("timezone_aware_time_range_required")
    return parsed.astimezone(UTC)


def main(
    argv: Sequence[str] | None = None,
    *,
    settings_loader: Callable[[], PhoenixEvalSettings] = PhoenixEvalSettings.from_environment,
    output: TextIO | None = None,
) -> int:
    """Run the evaluator and print a summary that contains no trace content."""

    stream = output or sys.stdout
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        summary = evaluate_selected_traces(parse_selection(args), settings_loader())
    except PhoenixEvaluationError as exc:
        print(
            json.dumps({"status": "error", "category": exc.category}, sort_keys=True),
            file=stream,
        )
        return 2
    except Exception:
        print(
            json.dumps({"status": "error", "category": "evaluation_failed"}, sort_keys=True),
            file=stream,
        )
        return 2
    print(json.dumps({"status": "ok", **summary.as_dict()}, sort_keys=True), file=stream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
