"""Partitioned Parquet data lake writer.

Data is organized as: data/lake/<symbol>/<YYYY-MM-DD>/ticks.parquet
This layout enables efficient date-range and symbol-filtered queries via PyArrow.
"""

from __future__ import annotations

import datetime
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)


class ParquetLake:
    """
    Batched Parquet writer for research-grade data storage.

    Flushes to disk when:
    - `batch_size` records have accumulated per symbol, OR
    - `flush()` is called explicitly (e.g. at end-of-day)
    """

    DEFAULT_BASE = Path("data/lake")

    SCHEMA = pa.schema([
        pa.field("timestamp_ns", pa.int64()),
        pa.field("symbol",       pa.string()),
        pa.field("price",        pa.float64()),
        pa.field("qty",          pa.float64()),
        pa.field("side",         pa.string()),
        pa.field("sequence_id",  pa.int64()),
    ])

    def __init__(self, base_path: Path = DEFAULT_BASE, batch_size: int = 10_000) -> None:
        self._base = base_path
        self._batch_size = batch_size
        self._buffers: dict[str, list[dict]] = defaultdict(list)

    def write(self, tick: dict) -> None:
        """Buffer a tick. Auto-flushes when batch_size is reached."""
        sym = tick["symbol"]
        self._buffers[sym].append(tick)
        if len(self._buffers[sym]) >= self._batch_size:
            self._flush_symbol(sym)

    def flush(self) -> None:
        """Force-flush all buffered symbols to disk."""
        for sym in list(self._buffers.keys()):
            self._flush_symbol(sym)

    def _flush_symbol(self, symbol: str) -> None:
        rows = self._buffers.pop(symbol, [])
        if not rows:
            return

        date_str = datetime.date.today().isoformat()
        out_dir = self._base / symbol / date_str
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "ticks.parquet"

        df = pd.DataFrame(rows)
        table = pa.Table.from_pandas(df, schema=self.SCHEMA, safe=False)

        # Append to existing file if present (for incremental intraday writes)
        if out_path.exists():
            existing = pq.read_table(out_path)
            table = pa.concat_tables([existing, table])

        pq.write_table(table, out_path, compression="snappy")
        log.debug("Flushed %d rows for %s → %s", len(rows), symbol, out_path)

    def read(self, symbol: str, date: str) -> pd.DataFrame:
        """Read a day's data from the lake."""
        path = self._base / symbol / date / "ticks.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pq.read_table(path).to_pandas()
