from __future__ import annotations

import argparse
from pathlib import Path

from samosbor.server_autonomy_config import (
    DAILY_AUTONOMY_RESEARCH_ARRAY_LIMITS,
    build_offline_autonomy_config_text,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an offline nightly-autonomy config from the current effective runtime config."
    )
    parser.add_argument("--source", required=True, help="Source config path.")
    parser.add_argument("--output", required=True, help="Output autonomy config path.")
    parser.add_argument(
        "--parquet-dir",
        required=True,
        help="Parquet directory path to inject into the [data] section.",
    )
    args = parser.parse_args()

    source_path = Path(args.source).resolve()
    output_path = Path(args.output).resolve()
    source_text = source_path.read_text(encoding="utf-8-sig")
    rendered = build_offline_autonomy_config_text(
        source_text,
        parquet_dir_path=args.parquet_dir,
        research_array_limits=DAILY_AUTONOMY_RESEARCH_ARRAY_LIMITS,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
