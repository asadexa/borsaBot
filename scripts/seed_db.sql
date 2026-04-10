-- Seed TimescaleDB: create tables and hypertables.

-- Ticks hypertable
CREATE TABLE IF NOT EXISTS ticks (
    time         TIMESTAMPTZ      NOT NULL,
    symbol       TEXT             NOT NULL,
    price        DOUBLE PRECISION,
    qty          DOUBLE PRECISION,
    side         CHAR(1),
    sequence_id  BIGINT           NOT NULL
);

SELECT create_hypertable('ticks', 'time', if_not_exists => TRUE);

CREATE UNIQUE INDEX IF NOT EXISTS ticks_dedup_idx
    ON ticks (symbol, time, sequence_id);

-- Order book snapshots
CREATE TABLE IF NOT EXISTS order_book_snapshots (
    time        TIMESTAMPTZ      NOT NULL,
    symbol      TEXT             NOT NULL,
    update_id   BIGINT           NOT NULL,
    bids        JSONB,
    asks        JSONB
);

SELECT create_hypertable('order_book_snapshots', 'time', if_not_exists => TRUE);

-- OHLCV continuous aggregate (1-minute bars)
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_1m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time) AS bucket,
    symbol,
    FIRST(price, time)            AS open,
    MAX(price)                    AS high,
    MIN(price)                    AS low,
    LAST(price, time)             AS close,
    SUM(qty)                      AS volume
FROM ticks
GROUP BY bucket, symbol
WITH NO DATA;

-- Orders table (order lifecycle tracking)
CREATE TABLE IF NOT EXISTS orders (
    order_id         TEXT        PRIMARY KEY,
    client_order_id  TEXT,
    broker_order_id  TEXT,
    symbol           TEXT        NOT NULL,
    side             TEXT        NOT NULL,
    order_type       TEXT        NOT NULL,
    quantity         DOUBLE PRECISION,
    filled_qty       DOUBLE PRECISION DEFAULT 0,
    avg_price        DOUBLE PRECISION DEFAULT 0,
    status           TEXT        NOT NULL DEFAULT 'created',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Portfolio Snapshots (State Recovery)
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    time          TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    nav           DOUBLE PRECISION NOT NULL,
    total_exposure DOUBLE PRECISION NOT NULL,
    positions     JSONB            NOT NULL
);

SELECT create_hypertable('portfolio_snapshots', 'time', if_not_exists => TRUE);
