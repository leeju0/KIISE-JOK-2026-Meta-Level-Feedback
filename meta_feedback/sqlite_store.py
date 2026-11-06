"""
Iterative Reasoning with Meta-Level Feedback - SQLite Store
Database operations for node, chain_link, run, and log tables
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from .schemas import (
    NodeRecord, ChainLinkRecord, RunRecord,
    MLogRecord, CLogRecord, StateSummaryLogRecord, KVStateRecord,
    NodeType, SessionRole, LinkType, RunStatus
)


class SQLiteStore:
    """SQLite 기반 체인 저장소"""
    
    def __init__(self, db_path: str = "outputs/reasoning.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    @contextmanager
    def _get_connection(self):
        """Thread-safe connection context manager"""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize database tables"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # node table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS node (
                    node_id TEXT PRIMARY KEY,
                    text_label TEXT NOT NULL,
                    node_type TEXT,
                    step_index INTEGER,
                    prev_node_id TEXT NULL,
                    created_at TEXT,
                    session_role TEXT
                )
            """)
            
            # chain_link table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chain_link (
                    from_node_id TEXT,
                    to_node_id TEXT,
                    link_type TEXT,
                    step_index INTEGER,
                    created_at TEXT,
                    PRIMARY KEY (from_node_id, to_node_id)
                )
            """)
            
            # run table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS run (
                    run_id TEXT PRIMARY KEY,
                    global_goal TEXT,
                    current_status TEXT,
                    final_answer TEXT NULL,
                    started_at TEXT,
                    finished_at TEXT NULL
                )
            """)
            
            # m_log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS m_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    step_index INTEGER,
                    input_summary TEXT,
                    chain_state_summary TEXT,
                    prior_chain_json TEXT NULL,
                    situation_analysis TEXT NULL,
                    sub_goal TEXT NULL,
                    reasoning_direction TEXT NULL,
                    decision TEXT NULL,
                    decision_reason TEXT NULL,
                    raw_response TEXT
                )
            """)
            
            # c_log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS c_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    step_index INTEGER,
                    prompt_text TEXT,
                    raw_response TEXT,
                    extracted_answer TEXT NULL
                )
            """)
            
            # state_summary_log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS state_summary_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    step_index INTEGER,
                    prev_input_text TEXT,
                    prev_output_text TEXT,
                    prev_state_summary TEXT NULL,
                    new_state_summary TEXT,
                    raw_response TEXT,
                    compression_ratio REAL NULL
                )
            """)
            
            # kv_state table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS kv_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    role TEXT,
                    step_index INTEGER,
                    state_ref TEXT,
                    token_count INTEGER NULL,
                    byte_size INTEGER NULL,
                    created_at TEXT
                )
            """)
            
            conn.commit()
    
    # =========================================================================
    # Run Operations
    # =========================================================================
    
    def create_run(self, run_id: str, global_goal: str) -> RunRecord:
        """Create a new run"""
        record = RunRecord(
            run_id=run_id,
            global_goal=global_goal,
            current_status=RunStatus.RUNNING,
            started_at=datetime.now().isoformat()
        )
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO run (run_id, global_goal, current_status, final_answer, started_at, finished_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                record.run_id, record.global_goal, record.current_status.value,
                record.final_answer, record.started_at, record.finished_at
            ))
        
        return record
    
    def update_run_status(self, run_id: str, status: RunStatus, final_answer: Optional[str] = None):
        """Update run status"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if status == RunStatus.STOP and final_answer:
                cursor.execute("""
                    UPDATE run SET current_status = ?, final_answer = ?, finished_at = ?
                    WHERE run_id = ?
                """, (status.value, final_answer, datetime.now().isoformat(), run_id))
            else:
                cursor.execute("""
                    UPDATE run SET current_status = ? WHERE run_id = ?
                """, (status.value, run_id))
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get run by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM run WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_all_runs(self) -> List[Dict[str, Any]]:
        """Get all runs"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM run ORDER BY started_at DESC")
            return [dict(row) for row in cursor.fetchall()]
    
    # =========================================================================
    # Node Operations
    # =========================================================================
    
    def insert_node(self, record: NodeRecord):
        """Insert a node"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO node (node_id, text_label, node_type, step_index, prev_node_id, created_at, session_role)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                record.node_id, record.text_label, record.node_type.value,
                record.step_index, record.prev_node_id, record.created_at, record.session_role.value
            ))
    
    def get_nodes_by_run(self, run_id: str) -> List[Dict[str, Any]]:
        """Get all nodes for a run (by node_id prefix)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM node WHERE node_id LIKE ? ORDER BY step_index, created_at
            """, (f"{run_id}%",))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM node WHERE node_id = ?", (node_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    # =========================================================================
    # Chain Link Operations
    # =========================================================================
    
    def insert_chain_link(self, record: ChainLinkRecord):
        """Insert a chain link"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO chain_link (from_node_id, to_node_id, link_type, step_index, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                record.from_node_id, record.to_node_id, record.link_type.value,
                record.step_index, record.created_at
            ))
    
    def get_chain_links_by_run(self, run_id: str) -> List[Dict[str, Any]]:
        """Get all chain links for a run"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM chain_link WHERE from_node_id LIKE ? ORDER BY step_index
            """, (f"{run_id}%",))
            return [dict(row) for row in cursor.fetchall()]
    
    # =========================================================================
    # M Log Operations
    # =========================================================================
    
    def insert_m_log(self, record: MLogRecord):
        """Insert M session log"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO m_log (run_id, step_index, input_summary, chain_state_summary,
                    prior_chain_json, situation_analysis, sub_goal, reasoning_direction,
                    decision, decision_reason, raw_response)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.run_id, record.step_index, record.input_summary, record.chain_state_summary,
                record.prior_chain_json, record.situation_analysis, record.sub_goal,
                record.reasoning_direction, record.decision, record.decision_reason, record.raw_response
            ))
    
    def get_m_logs_by_run(self, run_id: str) -> List[Dict[str, Any]]:
        """Get all M logs for a run"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM m_log WHERE run_id = ? ORDER BY step_index
            """, (run_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    # =========================================================================
    # C Log Operations
    # =========================================================================
    
    def insert_c_log(self, record: CLogRecord):
        """Insert C session log"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO c_log (run_id, step_index, prompt_text, raw_response, extracted_answer)
                VALUES (?, ?, ?, ?, ?)
            """, (
                record.run_id, record.step_index, record.prompt_text,
                record.raw_response, record.extracted_answer
            ))
    
    def get_c_logs_by_run(self, run_id: str) -> List[Dict[str, Any]]:
        """Get all C logs for a run"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM c_log WHERE run_id = ? ORDER BY step_index
            """, (run_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    # =========================================================================
    # State Summary Log Operations
    # =========================================================================
    
    def insert_state_summary_log(self, record: StateSummaryLogRecord):
        """Insert state summary log"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO state_summary_log (run_id, step_index, prev_input_text, prev_output_text,
                    prev_state_summary, new_state_summary, raw_response, compression_ratio)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.run_id, record.step_index, record.prev_input_text, record.prev_output_text,
                record.prev_state_summary, record.new_state_summary, record.raw_response,
                record.compression_ratio
            ))
    
    def get_state_summary_logs_by_run(self, run_id: str) -> List[Dict[str, Any]]:
        """Get all state summary logs for a run"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM state_summary_log WHERE run_id = ? ORDER BY step_index
            """, (run_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_latest_state_summary(self, run_id: str) -> Optional[str]:
        """Get the latest state summary for a run"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT new_state_summary FROM state_summary_log
                WHERE run_id = ? ORDER BY step_index DESC LIMIT 1
            """, (run_id,))
            row = cursor.fetchone()
            return row['new_state_summary'] if row else None
    
    # =========================================================================
    # KV State Operations
    # =========================================================================
    
    def insert_kv_state(self, record: KVStateRecord):
        """Insert KV state record"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO kv_state (run_id, role, step_index, state_ref, token_count, byte_size, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                record.run_id, record.role, record.step_index,
                record.state_ref, record.token_count, record.byte_size, record.created_at
            ))
    
    def get_latest_kv_state(self, run_id: str, role: str) -> Optional[Dict[str, Any]]:
        """Get the latest KV state for a role"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM kv_state WHERE run_id = ? AND role = ?
                ORDER BY step_index DESC LIMIT 1
            """, (run_id, role))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    # =========================================================================
    # Chain Visualization Helper
    # =========================================================================
    
    def get_chain_for_visualization(self, run_id: str) -> Dict[str, Any]:
        """Get complete chain data for visualization"""
        nodes = self.get_nodes_by_run(run_id)
        links = self.get_chain_links_by_run(run_id)
        run = self.get_run(run_id)
        m_logs = self.get_m_logs_by_run(run_id)
        c_logs = self.get_c_logs_by_run(run_id)
        state_summaries = self.get_state_summary_logs_by_run(run_id)
        
        return {
            "run": run,
            "nodes": nodes,
            "links": links,
            "m_logs": m_logs,
            "c_logs": c_logs,
            "state_summaries": state_summaries
        }


    def clear_all(self):
        """Clear all data from all tables"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM node")
            cursor.execute("DELETE FROM chain_link")
            cursor.execute("DELETE FROM run")
            cursor.execute("DELETE FROM m_log")
            cursor.execute("DELETE FROM c_log")
            cursor.execute("DELETE FROM state_summary_log")
            cursor.execute("DELETE FROM kv_state")
            conn.commit()


# Singleton instance
_store_instance: Optional[SQLiteStore] = None

def get_store(db_path: str = "outputs/reasoning.db") -> SQLiteStore:
    """Get or create SQLite store instance"""
    global _store_instance
    if _store_instance is None:
        _store_instance = SQLiteStore(db_path)
    return _store_instance
