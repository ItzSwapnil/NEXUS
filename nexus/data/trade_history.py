import ast
import os
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List

try:
    import duckdb
except ModuleNotFoundError:  # SQLite trade reconciliation does not require DuckDB.
    duckdb = None  # type: ignore[assignment]


class TradeHistory:
    def __init__(self, db_path: str = "data/trade_history.db"):
        self.db_path = db_path
        Path(os.path.dirname(self.db_path) or ".").mkdir(parents=True, exist_ok=True)
        self._initialize_db()

    def _initialize_db(self):
        """Initialize the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset TEXT,
                direction TEXT,
                amount REAL,
                result TEXT,
                profit REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # Durable broker reconciliation fields.  Keep the original columns for
        # backwards compatibility with existing installs.
        existing = {row[1] for row in cursor.execute("PRAGMA table_info(trades)")}
        migrations = {
            "local_id": "TEXT",
            "broker_order_id": "TEXT",
            "status": "TEXT",
            "outcome": "TEXT",
            "expiration": "INTEGER",
            "error": "TEXT",
            "updated_at": "TEXT",
        }
        for name, kind in migrations.items():
            if name not in existing:
                cursor.execute(f"ALTER TABLE trades ADD COLUMN {name} {kind}")
        cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_trades_local_id ON trades(local_id)"
        )
        cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_trades_broker_order_id "
            "ON trades(broker_order_id) WHERE broker_order_id IS NOT NULL AND broker_order_id <> ''"
        )
        conn.commit()
        conn.close()

    def log_trade(self, trade: Dict):
        """Log a trade to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO trades (asset, direction, amount, result, profit)
            VALUES (:asset, :direction, :amount, :result, :profit)
            """,
            trade,
        )
        conn.commit()
        conn.close()

    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """Retrieve the most recent trade history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, local_id, broker_order_id, asset, direction, amount, result,
                   profit, status, outcome, expiration, error, timestamp, updated_at
            FROM trades
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        conn.close()
        def broker_id(value: object) -> str:
            """Render legacy nested broker responses as their actual ID."""
            text = str(value or "").strip()
            if text.startswith("{"):
                try:
                    parsed = ast.literal_eval(text)
                    if isinstance(parsed, dict):
                        value = parsed.get("id") or parsed.get("order_id") or ""
                except (SyntaxError, ValueError):
                    return ""
            return str(value or "").strip()

        return [
            {
                "db_id": row[0],
                "id": row[1] or f"DB-{row[0]}",
                "local_id": row[1],
                "broker_order_id": broker_id(row[2]),
                "asset": row[3],
                "direction": row[4],
                "amount": row[5],
                "result": row[6],
                "profit": float(row[7] or 0.0),
                "status": row[8] or ("SETTLED" if row[6] else "UNKNOWN"),
                "outcome": row[9] or row[6] or "UNVERIFIED",
                "expiration": row[10] or 0,
                "error": row[11],
                "timestamp": row[12],
                "updated_at": row[13],
            }
            for row in rows
        ]

    def record_placement(self, trade: Dict) -> None:
        """Persist a placement before it can be displayed as an active trade."""
        now = datetime.now(UTC).isoformat()
        payload = {
            "local_id": trade.get("local_id"),
            "broker_order_id": trade.get("broker_order_id") or None,
            "asset": trade.get("asset"),
            "direction": trade.get("direction"),
            "amount": float(trade.get("amount", 0.0)),
            "result": "PENDING",
            "profit": None,
            "status": trade.get("status", "PLACED"),
            "outcome": None,
            "expiration": int(trade.get("expiration", 0)),
            "error": trade.get("error"),
            "timestamp": trade.get("timestamp") or now,
            "updated_at": now,
        }
        with threading.Lock():
            conn = sqlite3.connect(self.db_path, timeout=15)
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO trades
                    (local_id, broker_order_id, asset, direction, amount, result,
                     profit, status, outcome, expiration, error, timestamp, updated_at)
                    VALUES (:local_id, :broker_order_id, :asset, :direction, :amount,
                            :result, :profit, :status, :outcome, :expiration, :error,
                            :timestamp, :updated_at)""",
                    payload,
                )
                conn.commit()
            finally:
                conn.close()

    def record_settlement(self, local_id: str, outcome: str, profit: float, **extra) -> None:
        """Atomically attach broker feedback to the original placement."""
        now = datetime.now(UTC).isoformat()
        conn = sqlite3.connect(self.db_path, timeout=15)
        try:
            conn.execute(
                """UPDATE trades SET result=?, outcome=?, profit=?, status=?, error=?, updated_at=?
                   WHERE local_id=?""",
                (
                    outcome,
                    outcome,
                    float(profit),
                    extra.get("status", "SETTLED"),
                    extra.get("error"),
                    now,
                    local_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_unresolved(self, limit: int = 500) -> List[Dict]:
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """SELECT * FROM trades
                   WHERE status IN ('PLACED', 'PENDING', 'UNVERIFIED')
                   ORDER BY timestamp DESC LIMIT ?""",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def clear_history(self, include_unresolved: bool = False) -> int:
        """Delete ledger rows, retaining unresolved orders by default.

        Unresolved rows are kept unless explicitly requested so clearing the
        UI cannot destroy the broker reconciliation trail for an open order.
        """
        conn = sqlite3.connect(self.db_path, timeout=15)
        try:
            if include_unresolved:
                cursor = conn.execute("DELETE FROM trades")
            else:
                cursor = conn.execute(
                    "DELETE FROM trades WHERE status NOT IN ('PLACED', 'PENDING', 'UNVERIFIED')"
                )
            conn.commit()
            return int(cursor.rowcount)
        finally:
            conn.close()


class AdvancedDataStore:
    """
    Advanced data store for NEXUS supporting SQLite, DuckDB, and model/feature storage.
    """

    def __init__(
        self, sqlite_path: str = "data/trade_history.db", duckdb_path: str = "data/analytics.duckdb"
    ):
        self.sqlite_path = sqlite_path
        self.duckdb_path = duckdb_path
        self._sqlite_lock = threading.Lock()
        self._duckdb_lock = threading.Lock()
        self._init_duckdb()

    def _init_duckdb(self):
        if duckdb is None:
            return
        os.makedirs(os.path.dirname(self.duckdb_path), exist_ok=True)
        with self._duckdb_lock:
            con = duckdb.connect(self.duckdb_path)
            con.execute("""
                CREATE TABLE IF NOT EXISTS model_checkpoints (
                    id BIGINT,
                    model_name VARCHAR,
                    checkpoint_path VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS feature_matrices (
                    id BIGINT,
                    asset VARCHAR,
                    timeframe INT,
                    features BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS reward_curves (
                    id BIGINT,
                    strategy VARCHAR,
                    rewards BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS risk_models (
                    id BIGINT,
                    model_name VARCHAR,
                    params BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS strategy_trees (
                    id BIGINT,
                    tree_name VARCHAR,
                    tree_data BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            con.close()

    def log_model_checkpoint(self, model_name: str, checkpoint_path: str):
        with self._duckdb_lock:
            con = duckdb.connect(self.duckdb_path)
            con.execute(
                """
                INSERT INTO model_checkpoints (model_name, checkpoint_path) VALUES (?, ?)
            """,
                (model_name, checkpoint_path),
            )
            con.close()

    def log_feature_matrix(self, asset: str, timeframe: int, features: bytes):
        with self._duckdb_lock:
            con = duckdb.connect(self.duckdb_path)
            con.execute(
                """
                INSERT INTO feature_matrices (asset, timeframe, features) VALUES (?, ?, ?)
            """,
                (asset, timeframe, features),
            )
            con.close()

    def log_reward_curve(self, strategy: str, rewards: bytes):
        with self._duckdb_lock:
            con = duckdb.connect(self.duckdb_path)
            con.execute(
                """
                INSERT INTO reward_curves (strategy, rewards) VALUES (?, ?)
            """,
                (strategy, rewards),
            )
            con.close()

    def log_risk_model(self, model_name: str, params: bytes):
        with self._duckdb_lock:
            con = duckdb.connect(self.duckdb_path)
            con.execute(
                """
                INSERT INTO risk_models (model_name, params) VALUES (?, ?)
            """,
                (model_name, params),
            )
            con.close()

    def log_strategy_tree(self, tree_name: str, tree_data: bytes):
        with self._duckdb_lock:
            con = duckdb.connect(self.duckdb_path)
            con.execute(
                """
                INSERT INTO strategy_trees (tree_name, tree_data) VALUES (?, ?)
            """,
                (tree_name, tree_data),
            )
            con.close()

    # Retrieval/query methods for analytics and dashboards can be added as needed.
