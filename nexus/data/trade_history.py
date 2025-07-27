import sqlite3
from typing import List, Dict
import duckdb
import threading
import os

class TradeHistory:
    def __init__(self, db_path: str = "data/trade_history.db"):
        self.db_path = db_path
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
            trade
        )
        conn.commit()
        conn.close()

    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """Retrieve the most recent trade history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT asset, direction, amount, result, profit, timestamp
            FROM trades
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "asset": row[0],
                "direction": row[1],
                "amount": row[2],
                "result": row[3],
                "profit": row[4],
                "timestamp": row[5],
            }
            for row in rows
        ]

class AdvancedDataStore:
    """
    Advanced data store for NEXUS supporting SQLite, DuckDB, and model/feature storage.
    """
    def __init__(self, sqlite_path: str = "data/trade_history.db", duckdb_path: str = "data/analytics.duckdb"):
        self.sqlite_path = sqlite_path
        self.duckdb_path = duckdb_path
        self._sqlite_lock = threading.Lock()
        self._duckdb_lock = threading.Lock()
        self._init_duckdb()

    def _init_duckdb(self):
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
            con.execute("""
                INSERT INTO model_checkpoints (model_name, checkpoint_path) VALUES (?, ?)
            """, (model_name, checkpoint_path))
            con.close()

    def log_feature_matrix(self, asset: str, timeframe: int, features: bytes):
        with self._duckdb_lock:
            con = duckdb.connect(self.duckdb_path)
            con.execute("""
                INSERT INTO feature_matrices (asset, timeframe, features) VALUES (?, ?, ?)
            """, (asset, timeframe, features))
            con.close()

    def log_reward_curve(self, strategy: str, rewards: bytes):
        with self._duckdb_lock:
            con = duckdb.connect(self.duckdb_path)
            con.execute("""
                INSERT INTO reward_curves (strategy, rewards) VALUES (?, ?)
            """, (strategy, rewards))
            con.close()

    def log_risk_model(self, model_name: str, params: bytes):
        with self._duckdb_lock:
            con = duckdb.connect(self.duckdb_path)
            con.execute("""
                INSERT INTO risk_models (model_name, params) VALUES (?, ?)
            """, (model_name, params))
            con.close()

    def log_strategy_tree(self, tree_name: str, tree_data: bytes):
        with self._duckdb_lock:
            con = duckdb.connect(self.duckdb_path)
            con.execute("""
                INSERT INTO strategy_trees (tree_name, tree_data) VALUES (?, ?)
            """, (tree_name, tree_data))
            con.close()

    # Retrieval/query methods for analytics and dashboards can be added as needed.
