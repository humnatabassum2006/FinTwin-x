-- ======================================================================================
-- FinTwin-X · PostgreSQL schema (transactional + serving layer)
-- DuckDB/Parquet remains the analytical warehouse; Postgres serves the product.
-- Run automatically by docker-compose (mounted into /docker-entrypoint-initdb.d).
-- ======================================================================================
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ------------------------------------------------------------------ identity
CREATE TABLE IF NOT EXISTS users (
    user_id              INTEGER PRIMARY KEY,
    persona              TEXT        NOT NULL,
    age                  SMALLINT    NOT NULL CHECK (age BETWEEN 18 AND 90),
    household_size       SMALLINT    NOT NULL DEFAULT 1,
    city                 TEXT        NOT NULL,
    city_cost_index      REAL        NOT NULL,
    income_type          TEXT        NOT NULL,
    employment_type      TEXT        NOT NULL,
    financial_goal       TEXT        NOT NULL,
    risk_preference      TEXT        NOT NULL CHECK (risk_preference IN ('conservative','balanced','aggressive')),
    monthly_income_base  NUMERIC(14,2) NOT NULL CHECK (monthly_income_base > 0),
    opening_cash_balance NUMERIC(14,2) NOT NULL DEFAULT 0,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- PII minimisation: no names, no CNIC, no contact details are stored.
    pii_hash             BYTEA                 -- salted hash, used only for dedup
);

-- ------------------------------------------------------------------ ledger
CREATE TABLE IF NOT EXISTS income (
    income_id    BIGINT       PRIMARY KEY,
    user_id      INTEGER      NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    txn_date     DATE         NOT NULL,
    amount       NUMERIC(14,2) NOT NULL CHECK (amount > 0),
    source       TEXT         NOT NULL,
    income_type  TEXT         NOT NULL,
    recurring    BOOLEAN      NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_income_user_date ON income(user_id, txn_date);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id   BIGINT       PRIMARY KEY,
    user_id          INTEGER      NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    ts               TIMESTAMPTZ  NOT NULL,
    amount           NUMERIC(14,2) NOT NULL CHECK (amount > 0),
    category         TEXT         NOT NULL,
    merchant         TEXT         NOT NULL,
    payment_method   TEXT         NOT NULL,
    transaction_type TEXT         NOT NULL CHECK (transaction_type IN ('debit','credit')),
    is_anomaly       SMALLINT     NOT NULL DEFAULT 0,
    anomaly_type     TEXT         NOT NULL DEFAULT 'none',
    anomaly_score    REAL
);
CREATE INDEX IF NOT EXISTS idx_tx_user_ts   ON transactions(user_id, ts);
CREATE INDEX IF NOT EXISTS idx_tx_category  ON transactions(category);
CREATE INDEX IF NOT EXISTS idx_tx_anomaly   ON transactions(is_anomaly) WHERE is_anomaly = 1;

CREATE TABLE IF NOT EXISTS debts (
    loan_id           BIGSERIAL PRIMARY KEY,
    user_id           INTEGER  NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    loan_type         TEXT     NOT NULL,
    principal         NUMERIC(14,2) NOT NULL CHECK (principal > 0),
    interest_rate     REAL     NOT NULL CHECK (interest_rate BETWEEN 0 AND 1),
    monthly_payment   NUMERIC(14,2) NOT NULL CHECK (monthly_payment >= 0),
    remaining_balance NUMERIC(14,2) NOT NULL CHECK (remaining_balance >= 0),
    start_date        DATE     NOT NULL,
    end_date          DATE     NOT NULL,
    term_months       INTEGER  NOT NULL CHECK (term_months > 0),
    CONSTRAINT chk_loan_dates CHECK (end_date > start_date)
);
CREATE INDEX IF NOT EXISTS idx_debt_user ON debts(user_id);

CREATE TABLE IF NOT EXISTS investments (
    investment_id   BIGSERIAL PRIMARY KEY,
    user_id         INTEGER  NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    asset           TEXT     NOT NULL,
    quantity        NUMERIC(18,6) NOT NULL CHECK (quantity > 0),
    purchase_price  NUMERIC(14,4) NOT NULL CHECK (purchase_price > 0),
    current_price   NUMERIC(14,4) NOT NULL CHECK (current_price > 0),
    purchase_date   DATE     NOT NULL,
    as_of_date      DATE     NOT NULL,
    liquidity_score REAL     NOT NULL CHECK (liquidity_score BETWEEN 0 AND 1),
    expected_return REAL,
    volatility      REAL
);
CREATE INDEX IF NOT EXISTS idx_inv_user ON investments(user_id);

CREATE TABLE IF NOT EXISTS financial_goals (
    goal_id              BIGSERIAL PRIMARY KEY,
    user_id              INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    goal_name            TEXT    NOT NULL,
    target_amount        NUMERIC(14,2) NOT NULL CHECK (target_amount > 0),
    current_amount       NUMERIC(14,2) NOT NULL DEFAULT 0,
    target_date          DATE    NOT NULL,
    monthly_contribution NUMERIC(14,2) NOT NULL DEFAULT 0,
    priority             SMALLINT NOT NULL DEFAULT 1,
    status               TEXT    NOT NULL DEFAULT 'active'
);
CREATE INDEX IF NOT EXISTS idx_goal_user ON financial_goals(user_id);

-- ------------------------------------------------------------------ analytics marts
CREATE TABLE IF NOT EXISTS monthly_panel (
    user_id        INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    year_month     DATE    NOT NULL,
    income         NUMERIC(14,2),
    expense        NUMERIC(14,2),
    emi            NUMERIC(14,2),
    net_cash_flow  NUMERIC(14,2),
    cash_balance   NUMERIC(14,2),
    invest_value   NUMERIC(14,2),
    debt_balance   NUMERIC(14,2),
    liquid_assets  NUMERIC(14,2),
    tx_count       INTEGER,
    PRIMARY KEY (user_id, year_month)
);

CREATE TABLE IF NOT EXISTS financial_features (
    user_id         INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    as_of           DATE    NOT NULL,
    features        JSONB   NOT NULL,     -- full engineered feature vector
    dna             JSONB   NOT NULL,     -- 9 Financial DNA dimensions
    segment         TEXT,
    PRIMARY KEY (user_id, as_of)
);

CREATE TABLE IF NOT EXISTS financial_risk_scores (
    score_id     BIGSERIAL PRIMARY KEY,
    user_id      INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    as_of        DATE    NOT NULL,
    risk_score   REAL    NOT NULL CHECK (risk_score BETWEEN 0 AND 100),
    band         TEXT    NOT NULL,
    model_name   TEXT    NOT NULL,
    model_version TEXT   NOT NULL,
    contributions JSONB,                  -- SHAP risk points
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_risk_user ON financial_risk_scores(user_id, as_of);

CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id  BIGSERIAL PRIMARY KEY,
    user_id      INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    as_of        DATE    NOT NULL,
    horizon      INTEGER NOT NULL,
    target       TEXT    NOT NULL,        -- expense | income | cashflow
    point        NUMERIC(14,2) NOT NULL,
    low          NUMERIC(14,2),
    high         NUMERIC(14,2),
    model_name   TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------------ simulation
CREATE TABLE IF NOT EXISTS scenarios (
    scenario_id  BIGSERIAL PRIMARY KEY,
    user_id      INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    name         TEXT    NOT NULL,
    label        TEXT,
    params       JSONB   NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS simulations (
    simulation_id BIGSERIAL PRIMARY KEY,
    user_id       INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    scenario_id   BIGINT  REFERENCES scenarios(scenario_id) ON DELETE SET NULL,
    n_paths       INTEGER NOT NULL,
    horizon       INTEGER NOT NULL,
    seed          INTEGER NOT NULL,
    assumptions   JSONB,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS simulation_results (
    simulation_id BIGINT NOT NULL REFERENCES simulations(simulation_id) ON DELETE CASCADE,
    metric        TEXT   NOT NULL,
    value         DOUBLE PRECISION,
    PRIMARY KEY (simulation_id, metric)
);

-- ------------------------------------------------------------------ agents / RAG / audit
CREATE TABLE IF NOT EXISTS ai_conversations (
    conversation_id BIGSERIAL PRIMARY KEY,
    user_id         INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    question        TEXT   NOT NULL,
    intent          TEXT,
    answer          TEXT,
    tool_calls      JSONB,
    trace           JSONB,
    model           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS recommendations (
    recommendation_id BIGSERIAL PRIMARY KEY,
    user_id           INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    kind              TEXT NOT NULL,
    payload           JSONB NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    model_name    TEXT NOT NULL,
    model_version TEXT NOT NULL,
    entity_id     TEXT NOT NULL,
    as_of         DATE NOT NULL,
    prediction    JSONB NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS audit_log (
    audit_id   BIGSERIAL PRIMARY KEY,
    ts         TIMESTAMPTZ NOT NULL DEFAULT now(),
    actor      TEXT NOT NULL,
    action     TEXT NOT NULL,
    resource   TEXT,
    status     TEXT,
    meta       JSONB
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts DESC);

-- ------------------------------------------------------------------ application users
CREATE TABLE IF NOT EXISTS app_users (
    username        TEXT PRIMARY KEY,
    email           TEXT NOT NULL,
    password_hash   TEXT NOT NULL,
    salt            TEXT NOT NULL,
    role            TEXT NOT NULL DEFAULT 'analyst',
    fintwin_user_id INTEGER REFERENCES users(user_id) ON DELETE SET NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
