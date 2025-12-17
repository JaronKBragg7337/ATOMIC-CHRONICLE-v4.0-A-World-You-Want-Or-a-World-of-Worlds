#!/usr/bin/env python3
"""
ATOMIC CHRONICLE v4.0 (A world you want or a world of worlds.)

- Canonical Atomic Chronicle core (blocks, RUS, worlds, carrier)
- Transparency Economy (optional)
- Energy / Robotics / Game / DAO worlds
- GameControl -> RobotWorld -> RewardWorld pipeline
- Dynamic feature toggles:
    * Loaded from atomic_chronicle_config.json (if present)
    * Overridable via CLI flags
    * Runtime toggling via FastAPI /toggle endpoints (if enabled)

This file is intentionally single-file and offline-capable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Tuple

# =====================================================
# OPTIONAL DEPENDENCIES
# =====================================================

try:
    import pygame
    from pygame.locals import *
    PYGAME_AVAILABLE = True
except Exception:
    pygame = None
    PYGAME_AVAILABLE = False

try:
    import serial
    SERIAL_AVAILABLE = True
except Exception:
    serial = None
    SERIAL_AVAILABLE = False

try:
    from fastapi import FastAPI, WebSocket
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    FASTAPI_AVAILABLE = True
except Exception:
    FASTAPI_AVAILABLE = False


# =====================================================
# AUTHORIZATION / ID TAGGING
# =====================================================

def authorized_label(module_name: str, pattern: str = "AEIOU4x3") -> dict:
    """
    Symbolic ID tags for major subsystems or entities.
    These are metadata only — they do not grant privileges.
    """
    return {
        "module": module_name,
        "authorization_pattern": pattern,
        "issued_at": time.time(),
    }

CORE_ID         = authorized_label("AtomicChronicleCore", "AEIOU4x3")
TRANSPARENCY_ID = authorized_label("TransparencyEconomy", "DoReMi4x3")
ENERGY_ID       = authorized_label("EnergyHarvesterWorld", "111222333444")

# =====================================================
# FEATURE TOGGLES (CONFIG + CLI + RUNTIME)
# =====================================================

# 1️⃣ UPDATED DEFAULT_FEATURES
DEFAULT_FEATURES: Dict[str, bool] = {
    "enable_transparency": True,
    "enable_energy_world": True,
    "enable_flywheel_world": True,
    "enable_windup_world": True,
    "enable_robot_world": True,
    "enable_game_world": True,
    "enable_game_control_world": True,
    "enable_reward_world": True,
    "enable_mystery_world": True,
    "enable_public_commons": True,
    "enable_private_consortium": True,
    "enable_carrier_heartbeat": False,
    "enable_fastapi_server": True,
    "enable_omni_world": True,  # NEW: OmniWorld master orchestrator
}

CONFIG_PATH = "atomic_chronicle_config.json"


def load_features(config_path: str = CONFIG_PATH) -> Dict[str, bool]:
    """
    Load feature flags from JSON file if present, else defaults.
    File format example:
      { "enable_energy_world": true, "enable_robot_world": false }
    """
    features = dict(DEFAULT_FEATURES)
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                if k in features:
                    features[k] = bool(v)
        except Exception as e:
            print(f"[WARN] Failed to load {config_path}: {e}")
    return features


def apply_cli_overrides(features: Dict[str, bool]) -> Dict[str, bool]:
    """
    CLI overrides: --set-feature key=value
    Example:
        --set-feature enable_robot_world=false
        --set-feature enable_fastapi_server=true
    """
    parser = argparse.ArgumentParser(description="Atomic Chronicle v4.0")
    parser.add_argument(
        "--set-feature",
        action="append",
        default=[],
        help="Override features: key=true|false (may repeat)",
    )
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Disable FastAPI server regardless of config.",
    )
    args, _ = parser.parse_known_args()

    for item in args.set_feature:
        if "=" not in item:
            print(f"[WARN] Ignoring malformed feature override: {item}")
            continue
        key, val = item.split("=", 1)
        key = key.strip()
        val = val.strip().lower()
        if key not in features:
            print(f"[WARN] Unknown feature key: {key}")
            continue
        if val in ("1", "true", "yes", "on"):
            features[key] = True
        elif val in ("0", "false", "no", "off"):
            features[key] = False
        else:
            print(f"[WARN] Unknown boolean for {key}: {val}")

    if args.no_server:
        features["enable_fastapi_server"] = False

    return features


# Runtime in-memory feature set (mutated by /toggle endpoints)
RUNTIME_FEATURES: Dict[str, bool] = {}


def set_feature_runtime(key: str, value: bool) -> None:
    if key in RUNTIME_FEATURES:
        print(f"[TOGGLE] {key} -> {value}")
        RUNTIME_FEATURES[key] = value
    else:
        print(f"[WARN] Attempt to toggle unknown feature at runtime: {key}")


# =====================================================
# UTILITIES / JSON / RUS
# =====================================================

def _dumps(o: Any) -> str:
    return json.dumps(o, separators=(",", ":"), sort_keys=True, default=str)


def rus_validate(inputs: Any, state: Any, receipt: Any) -> bool:
    """
    Reality Update Substrate (RUS) structural validation.
    System enforces *structure*, not meaning.
    """
    return (
        isinstance(inputs, dict)
        and isinstance(state, dict)
        and "_ts" in state
        and isinstance(receipt, dict)
        and "type" in receipt
    )


# =====================================================
# BLOCK / LEDGER CORE
# =====================================================

BLOCK_ID = authorized_label("BlockCore", "Block3333")

@dataclass(frozen=True)
class Block:
    id: str
    ts: float
    prev: Optional[str]
    world: str
    runtime: str
    inputs: str
    state: str
    receipt: str
    witness: Optional[str]
    nonce: str
    cost: float


def new_block(
    *,
    prev: Optional[str],
    world: str,
    runtime: str,
    inputs: Any,
    state: Any,
    receipt: Any,
    witness: Optional[Any] = None,
    nonce: str = "0",
    cost: float = 0.0,
) -> Block:
    return Block(
        id=str(uuid.uuid4()),
        ts=time.time(),
        prev=prev,
        world=world,
        runtime=runtime,
        inputs=_dumps(inputs),
        state=_dumps(state),
        receipt=_dumps(receipt),
        witness=_dumps(witness) if witness is not None else None,
        nonce=str(nonce),
        cost=float(cost),
    )


# =====================================================
# PERSISTENCE (SQLite)
# =====================================================

PERSISTENCE_ID = authorized_label("Persistence", "Persist3333")


class Persistence:
    def __init__(self, db: str = "atomic_unified.db"):
        self.db = db
        self.lock = threading.RLock()
        self.token_price = 0.00001
        self.pow_prefix = "000"      # placeholder
        self.pow_iters = 200_000     # placeholder
        self._init()

    def _c(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db, check_same_thread=False)
        c.execute("PRAGMA journal_mode=WAL;")
        return c

    def _init(self) -> None:
        with self._c() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS blocks(
                    id TEXT PRIMARY KEY,
                    ts REAL,
                    prev TEXT,
                    world TEXT,
                    runtime TEXT,
                    inputs TEXT,
                    state TEXT,
                    receipt TEXT,
                    witness TEXT,
                    nonce TEXT,
                    cost REAL
                )"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS wallets(
                    id TEXT PRIMARY KEY,
                    bal REAL
                )"""
            )

    # --- wallets ---

    def wallet(self, pid: str) -> float:
        with self._c() as c:
            r = c.execute(
                "SELECT bal FROM wallets WHERE id=?",
                (pid,),
            ).fetchone()
            return r[0] if r else 0.0

    def reward(self, pid: str, amt: float) -> None:
        with self._c() as c:
            c.execute(
                """INSERT INTO wallets(id, bal) VALUES (?,?)
                   ON CONFLICT(id) DO UPDATE SET bal=bal+excluded.bal""",
                (pid, amt),
            )

    def charge(self, pid: str, cost: float) -> bool:
        with self.lock:
            bal = self.wallet(pid)
            if bal < cost:
                return False
            self.reward(pid, -cost)
            return True

    # --- blocks / chain ---

    def head(self, world: str) -> Optional[str]:
        with self._c() as c:
            r = c.execute(
                "SELECT id FROM blocks WHERE world=? ORDER BY ts DESC LIMIT 1",
                (world,),
            ).fetchone()
            return r[0] if r else None

    def append(self, b: Block) -> None:
        with self.lock, self._c() as c:
            c.execute(
                "INSERT INTO blocks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    b.id,
                    b.ts,
                    b.prev,
                    b.world,
                    b.runtime,
                    b.inputs,
                    b.state,
                    b.receipt,
                    b.witness,
                    b.nonce,
                    b.cost,
                ),
            )

    def verify_chain(self, world: str) -> bool:
        with self._c() as c:
            rows = c.execute(
                "SELECT id, prev FROM blocks WHERE world=? ORDER BY ts ASC",
                (world,),
            ).fetchall()
        last_id = None
        for cid, prev_id in rows:
            if prev_id != last_id:
                return False
            last_id = cid
        return True


# =====================================================
# TRANSPARENCY ECONOMY
# =====================================================

TRANSPARENCY_CORE_LABEL = authorized_label("TransparencyConsensus", "TransCore")


class TransparencyConsensus:
    def __init__(self):
        self.registrations: Dict[str, Dict[str, Any]] = {}
        self.unknown_registry: Dict[str, Dict[str, Any]] = {}

    def register_entity(
        self,
        entity_id: str,
        transparency: str = "unknown",
        willingness_to_reveal: float = 0.5,
        terms: Optional[dict] = None,
        registered_by: str = "system",
    ) -> Dict[str, Any]:
        if transparency not in ("public", "private", "unknown", "consensus_pending"):
            transparency = "unknown"

        reg = {
            "transparency": transparency,
            "willingness": float(willingness_to_reveal),
            "terms": terms or {},
            "registered_at": time.time(),
            "reputation": 0.0,
            "tokens_earned": 0.0,
            "last_reveal_attempt": None,
            "registered_by": registered_by,
        }
        self.registrations[entity_id] = reg

        if transparency == "unknown":
            self.unknown_registry[entity_id] = {
                "excitement_level": random.uniform(0.7, 1.0),
                "hint": self._hint(),
                "potential_value": random.uniform(100, 10_000),
                "discovery_reward": 1000.0 * float(willingness_to_reveal),
            }
        return reg

    def get_commercial_rights(self, entity_id: str, use_type: str) -> Dict[str, Any]:
        reg = self.registrations.get(entity_id, {})
        transparency = reg.get("transparency", "unknown")

        if transparency == "public":
            fee = reg.get("terms", {}).get("commercial", {}).get("fee_per_use", 0.0)
            return {"commercial_allowed": True, "fee_tokens": fee}
        if transparency == "private":
            return {"commercial_allowed": True, "fee_tokens": 0.0}

        return {"commercial_allowed": False, "fee_tokens": 0.0}

    def attempt_reveal(self, entity_id: str, revealer_id: str, evidence: dict) -> Dict[str, Any]:
        if entity_id not in self.unknown_registry:
            return {"success": False, "reason": "Entity not unknown or already revealed."}

        unknown = self.unknown_registry[entity_id]
        success_chance = (len(_dumps(evidence)) / 1000.0) * unknown["excitement_level"] * 0.5

        if random.random() < success_chance:
            reward = unknown["discovery_reward"]
            reg = self.registrations[entity_id]
            reg["transparency"] = "public"
            reg["reputation"] = 0.1
            del self.unknown_registry[entity_id]

            return {
                "success": True,
                "reward_tokens": reward,
                "new_transparency": "public",
                "message": f"Entity {entity_id} successfully revealed and is now public.",
            }

        self.registrations[entity_id]["last_reveal_attempt"] = time.time()
        unknown["excitement_level"] = min(1.0, unknown["excitement_level"] + 0.1)
        return {"success": False, "reason": "Insufficient evidence or poor timing."}

    def _hint(self) -> str:
        hints = [
            "Capability exceeds current physics models",
            "Consciousness type previously unclassified",
            "Energy source appears self-sustaining",
            "Uses non-EM spectrum communication",
            "Temporal signature beyond expected bounds",
            "Information density near quantum threshold",
            "Indicates cooperative intelligence pattern",
            "Efficiency exceeds known maxima",
        ]
        return random.choice(hints)


TRANSPARENCY_ECON_ID = authorized_label("TransparencyEconomy", "TransEcon")


class TransparencyEconomy(Persistence):
    def __init__(self, db: str = "atomic_unified.db"):
        super().__init__(db)
        self.transparency = TransparencyConsensus()
        self.commercial_ledger: Dict[str, float] = {}
        self.revelation_rewards: Dict[str, List[Dict[str, Any]]] = {}

        self.transparency_multipliers = {
            "public": 1.0,
            "private": 0.3,
            "unknown": 0.0,
            "consensus_pending": 0.1,
        }

    def reward_with_transparency(
        self, pid: str, amt: float, entity_id: Optional[str] = None
    ) -> float:
        if entity_id is None:
            super().reward(pid, amt)
            return amt

        reg = self.transparency.registrations.get(entity_id, {})
        transparency = reg.get("transparency", "unknown")
        mult = self.transparency_multipliers.get(transparency, 0.0)
        actual = amt * mult

        if actual <= 0.0:
            if entity_id in self.transparency.unknown_registry:
                u = self.transparency.unknown_registry[entity_id]
                u["excitement_level"] = min(1.0, u["excitement_level"] * 1.1)
                u["discovery_reward"] *= 1.2
            return 0.0

        super().reward(pid, actual)

        if transparency == "public":
            self.commercial_ledger[entity_id] = self.commercial_ledger.get(entity_id, 0.0) + actual
            reg["tokens_earned"] += actual
            reg["reputation"] = min(1.0, self.commercial_ledger[entity_id] / 10_000.0)

        return actual

    def charge_with_transparency(
        self, pid: str, cost: float, entity_id: Optional[str] = None
    ) -> bool:
        total = cost
        if entity_id:
            rights = self.transparency.get_commercial_rights(entity_id, "usage")
            if not rights.get("commercial_allowed", False):
                return False
            fee = rights.get("fee_tokens", 0.0)
            if fee > 0.0:
                owner = self.transparency.registrations.get(entity_id, {}).get("registered_by", "system")
                super().reward(owner, fee)
            total = cost + fee
        return super().charge(pid, total)

    def reveal_entity(
        self, entity_id: str, revealer_id: str, evidence: Optional[dict] = None
    ) -> Dict[str, Any]:
        res = self.transparency.attempt_reveal(entity_id, revealer_id, evidence or {})
        if res.get("success"):
            reward = res.get("reward_tokens", 1000.0)
            super().reward(revealer_id, reward)
            self.revelation_rewards.setdefault(entity_id, []).append(
                {"revealer": revealer_id, "reward": reward, "timestamp": time.time()}
            )
            res["commercial_status"] = "now_available"
        return res


# =====================================================
# WORLD BASE CLASSES
# =====================================================

WORLD_CORE_ID = authorized_label("WorldBase", "WorldBase")


class World:
    def __init__(
        self,
        name: str,
        persist: Persistence,
        player: str,
        runtime: str,
        charge: bool = False,
    ):
        self.name = name
        self.p = persist
        self.pid = player
        self.runtime = runtime
        self.charge = charge
        self.state: Dict[str, Any] = {"_ts": time.time()}
        self.subs: List[Callable[[Block], None]] = []

    def subscribe(self, fn: Callable[[Block], None]) -> None:
        self.subs.append(fn)

    def cost(self, i: Any) -> float:
        return len(_dumps(i).encode()) * self.p.token_price if self.charge else 0.0

    def execute_logic(self, i: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ts = time.time()
        s = dict(self.state, _ts=ts)
        if isinstance(i, dict):
            s.update(i)
        r = {"type": "event", "ts": ts}
        return s, r

    def step(self, i: Dict[str, Any], witness: Optional[Dict[str, Any]] = None) -> Block:
        c = self.cost(i)
        if c and not self.p.charge(self.pid, c):
            raise ValueError(f"Insufficient tokens for player {self.pid}")
        prev = self.p.head(self.name)
        s, r = self.execute_logic(i)
        if not rus_validate(i, s, r):
            raise ValueError("RUS validation failed")
        b = new_block(
            prev=prev,
            world=self.name,
            runtime=self.runtime,
            inputs=i,
            state=s,
            receipt=r,
            witness=witness,
            cost=c,
        )
        self.state = s
        self.p.append(b)
        for fn in self.subs:
            try:
                fn(b)
            except Exception as e:
                print(f"[WARN] subscriber failed in world {self.name}: {e}")
        return b


class TransparencyAwareWorld(World):
    def __init__(
        self,
        name: str,
        persist: Persistence,
        player: str,
        runtime: str,
        charge: bool = False,
        default_transparency: str = "unknown",
    ):
        super().__init__(name, persist, player, runtime, charge)
        self.default_transparency = default_transparency
        self.entity_registry: Dict[str, Dict[str, Any]] = {}
        self.transparency_economy = isinstance(persist, TransparencyEconomy)

    def register_entity(
        self,
        entity_id: str,
        entity_type: str,
        transparency: Optional[str] = None,
        registered_by: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        transparency = transparency or self.default_transparency
        reg_by = registered_by or self.pid

        if self.transparency_economy:
            registration = self.p.transparency.register_entity(
                entity_id, transparency, registered_by=reg_by, **kwargs
            )
        else:
            registration = {"transparency": transparency, "registered_at": time.time()}

        self.entity_registry[entity_id] = {
            "type": entity_type,
            "registration": registration,
            "created_at": time.time(),
            "usage_count": 0,
        }
        return registration

    def execute_logic(self, i: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ts = time.time()
        s = dict(self.state, _ts=ts)
        r: Dict[str, Any] = {"type": "event", "ts": ts, "transparency_aware": True}

        if isinstance(i, dict) and "entities_used" in i and self.transparency_economy:
            entities = i["entities_used"]
            commercial = bool(i.get("commercial", False))
            if commercial:
                checks: Dict[str, bool] = {}
                for e in entities:
                    rights = self.p.transparency.get_commercial_rights(e, "usage")
                    checks[e] = rights.get("commercial_allowed", False)
                r["commercial_checks"] = checks
                if not all(checks.values()):
                    r["commercial_blocked"] = True
                    r["blocked_entities"] = [e for e, ok in checks.items() if not ok]

        return s, r


# =====================================================
# DOMAIN WORLDS
# =====================================================

class EnergyHarvesterWorld(World):
    def __init__(self, p: Persistence):
        super().__init__("EnergyHarvesterWorld", p, "harvester", "EnergyHarvester", True)
        self.total = 0.0
        self.state = {"total_kwh": self.total, "_ts": time.time()}

    def execute_logic(self, i: Dict[str, Any]):
        ts = time.time()
        e = float(i.get("energy_produced_kwh", 0.0))
        self.total += e
        s = {"total_kwh": self.total, "_ts": ts}
        r = {"type": "energy_output", "ts": ts, "kwh": e}
        return s, r


class FlywheelPlantWorld(World):
    def __init__(self, p: Persistence):
        super().__init__("FlywheelPlantWorld", p, "plant_ops", "FlywheelPlant", True)
        self.state = {
            "rpm": 0.0,
            "stored_kwh": 0.0,
            "mode": "idle",
            "last_cmd": None,
            "_ts": time.time(),
        }

    def execute_logic(self, i: Dict[str, Any]):
        ts = time.time()
        s = dict(self.state, _ts=ts)
        cmd = i.get("cmd")

        if cmd == "set_mode":
            new_mode = i.get("mode", s["mode"])
            s["mode"] = new_mode if new_mode in ("idle", "charge", "discharge", "fault") else "fault"
        elif cmd == "adjust_rpm":
            rpm_delta = float(i.get("rpm_delta", 0.0))
            s["rpm"] = max(0.0, s["rpm"] + rpm_delta)
        elif cmd == "trip":
            s["mode"] = "fault"
        elif cmd == "reset" and s["mode"] == "fault":
            s["mode"] = "idle"

        if "kwh_delta" in i:
            s["stored_kwh"] = max(0.0, s["stored_kwh"] + float(i["kwh_delta"]))

        s["last_cmd"] = cmd
        r = {
            "type": "flywheel_event",
            "ts": ts,
            "cmd": cmd,
            "rpm": s["rpm"],
            "stored_kwh": s["stored_kwh"],
            "mode": s["mode"],
        }
        return s, r


class WindupGeneratorWorld(World):
    def __init__(self, p: Persistence):
        super().__init__("WindupGeneratorWorld", p, "mech_ops", "WindupGenerator", True)
        self.state = {
            "tension_level": 0.0,
            "available_kwh": 0.0,
            "status": "idle",
            "last_cmd": None,
            "_ts": time.time(),
        }

    def execute_logic(self, i: Dict[str, Any]):
        ts = time.time()
        s = dict(self.state, _ts=ts)
        cmd = i.get("cmd")

        if cmd == "wind":
            amt = float(i.get("amount", 0.0))
            s["tension_level"] = max(0.0, min(100.0, s["tension_level"] + amt))
            s["status"] = "winding"
            s["available_kwh"] = s["tension_level"] * 0.01
        elif cmd == "generate":
            kwh = float(i.get("kwh_generated", 0.0))
            s["available_kwh"] = max(0.0, s["available_kwh"] - kwh)
            s["tension_level"] = max(0.0, s["tension_level"] - kwh * 10.0)
            s["status"] = "generating"
        elif cmd == "stop":
            s["status"] = "idle"
        elif cmd == "trip":
            s["status"] = "fault"
        elif cmd == "reset" and s["status"] == "fault":
            s["status"] = "idle"

        s["last_cmd"] = cmd
        r = {
            "type": "windup_event",
            "ts": ts,
            "cmd": cmd,
            "tension_level": s["tension_level"],
            "available_kwh": s["available_kwh"],
            "status": s["status"],
        }
        return s, r


class AsteroidsGameWorld(World):
    def __init__(self, p: Persistence):
        super().__init__("AsteroidsGameWorld", p, "player1", "AsteroidsGame", True)
        self.score = 0
        self.ship = {"x": 500, "y": 350, "a": 0, "t": 0, "b": []}
        self.asts: List[Dict[str, Any]] = []
        self.spawn()

        if PYGAME_AVAILABLE:
            pygame.init()
            self.scr = pygame.display.set_mode((1000, 700))
            self.clk = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

    def spawn(self, n: int = 5):
        for _ in range(n):
            self.asts.append(
                {
                    "x": random.randint(0, 1000),
                    "y": random.randint(0, 700),
                    "dx": random.uniform(-2, 2),
                    "dy": random.uniform(-2, 2),
                    "s": random.randint(20, 40),
                }
            )

    def execute_logic(self, i: Dict[str, Any]):
        ts = time.time()
        s = dict(self.state, _ts=ts)
        r: Dict[str, Any] = {"type": "game_event", "ts": ts}
        action = i.get("action")
        if action == "shoot":
            self.score += 10
        s["score"] = self.score
        s["ship"] = self.ship
        s["asteroids"] = self.asts
        r["action"] = action
        r["score"] = self.score
        return s, r


class MysteryTechnologyWorld(TransparencyAwareWorld):
    def __init__(self, p: Persistence):
        super().__init__("MysteryTechnologyWorld", p, "unknown_inventor", "MysteryTech", True, "unknown")

    def execute_logic(self, i: Dict[str, Any]):
        ts = time.time()
        s, r = super().execute_logic(i)
        action = i.get("action", "observe")
        tech_id = i.get("technology_id", f"mystery_{int(time.time())}")

        if action == "discover":
            reg = self.register_entity(
                tech_id,
                "mystery_technology",
                transparency="unknown",
                willingness_to_reveal=i.get("willingness", 0.3),
                terms={"excitement_guarantee": True},
            )
            info = {}
            if isinstance(self.p, TransparencyEconomy):
                info = self.p.transparency.unknown_registry.get(tech_id, {})
            s.update(
                {
                    "mystery_technologies": list(self.entity_registry.keys()),
                    "latest_discovery": tech_id,
                    "excitement_level": info.get("excitement_level", 0.8),
                    "_ts": ts,
                }
            )
            r.update(
                {
                    "type": "mystery_discovery",
                    "tech_id": tech_id,
                    "hint": info.get("hint", "Capabilities unknown"),
                    "reveal_reward": info.get("discovery_reward", 0.0),
                    "ts": ts,
                }
            )
            return s, r
        elif action == "attempt_reveal" and isinstance(self.p, TransparencyEconomy):
            result = self.p.reveal_entity(tech_id, i.get("revealer_id", "anon"), i.get("evidence", {}))
            s.update({"_ts": ts, "revelation_attempt": tech_id})
            r.update(result)
            return s, r
        else:
            avg_exc = 0.5
            if isinstance(self.p, TransparencyEconomy) and self.p.transparency.unknown_registry:
                vals = [v.get("excitement_level", 0.5) for v in self.p.transparency.unknown_registry.values()]
                avg_exc = sum(vals) / len(vals)
            s.update({"observing_mysteries": True, "_ts": ts})
            r.update({"type": "mystery_observation", "average_excitement": avg_exc, "ts": ts})
            return s, r


class PublicCommonsWorld(TransparencyAwareWorld):
    def __init__(self, p: Persistence):
        super().__init__("PublicCommonsWorld", p, "commons_manager", "PublicCommons", True, "public")

    def execute_logic(self, i: Dict[str, Any]):
        ts = time.time()
        s, r = super().execute_logic(i)
        action = i.get("action", "contribute")
        eid = i.get("entity_id", f"public_{int(time.time())}")

        if action == "contribute":
            reg = self.register_entity(
                eid,
                i.get("entity_type", "knowledge"),
                transparency="public",
                willingness_to_reveal=1.0,
                terms={
                    "commercial": {
                        "fee_per_use": i.get("license_fee", 0.0),
                        "attribution": i.get("require_attribution", True),
                    },
                    "open_source": i.get("open_source", True),
                },
            )
            total_val = 0.0
            if isinstance(self.p, TransparencyEconomy):
                total_val = sum(
                    self.p.commercial_ledger.get(e, 0.0) for e in self.entity_registry.keys()
                )
            s.update(
                {
                    "public_entities": list(self.entity_registry.keys()),
                    "latest_addition": eid,
                    "total_commercial_value": total_val,
                    "_ts": ts,
                }
            )
            r.update(
                {
                    "type": "public_contribution",
                    "entity_id": eid,
                    "license_terms": reg.get("terms", {}),
                    "estimated_value": i.get("estimated_value", 100.0),
                    "ts": ts,
                }
            )
            return s, r

        s.update({"status": "public_commons_operational", "_ts": ts})
        r.update({"type": "commons_report", "ts": ts})
        return s, r


class PrivateConsortiumWorld(TransparencyAwareWorld):
    def __init__(self, p: Persistence, members: Optional[List[str]] = None):
        super().__init__("PrivateConsortiumWorld", p, "consortium_admin", "PrivateConsortium", True, "private")
        self.consortium_members = members or ["member_1", "member_2"]
        self.shared_secrets: Dict[str, Dict[str, Any]] = {}

    def execute_logic(self, i: Dict[str, Any]):
        ts = time.time()
        s, r = super().execute_logic(i)
        member = i.get("member_id")
        if member not in self.consortium_members:
            s.update({"access_denied": True, "_ts": ts})
            r.update({"type": "consortium_access_denied", "reason": "not_a_member", "ts": ts})
            return s, r

        action = i.get("action", "share_secret")
        if action == "share_secret":
            sid = i.get("secret_id", f"secret_{int(time.time())}")
            content = i.get("content", {})
            self.register_entity(
                sid,
                "consortium_secret",
                transparency="private",
                terms={"access": self.consortium_members},
                registered_by=member,
            )
            self.shared_secrets[sid] = {"content": content, "owner": member, "_ts": ts}
            s.update({"shared": sid, "_ts": ts})
            r.update({"type": "secret_shared", "sid": sid, "ts": ts})
            return s, r

        s.update({"idle": True, "_ts": ts})
        r.update({"type": "consortium_idle", "ts": ts})
        return s, r


# =====================================================
# GAME CONTROL -> ROBOT -> REWARD PIPELINE
# =====================================================

GAME_CONTROL_LABEL = authorized_label("GameControlWorld", "GA123CTRL")


class GameControlWorld(World):
    """
    Bridges gamer input to robot commands.
    """
    def __init__(self, p: Persistence, player: str = "player1"):
        super().__init__("GameControlWorld", p, player, "GameControl", True)
        self.state = {
            "last_action": None,
            "last_command": None,
            "last_job_id": None,
            "_ts": time.time(),
        }

    def execute_logic(self, i: Dict[str, Any]):
        ts = time.time()
        s = dict(self.state, _ts=ts)

        action = i.get("action")
        job_id = i.get("job_id")

        cmd = None
        if action == "move_forward":
            cmd = {"type": "move", "direction": "forward", "distance": i.get("distance", 0.5)}
        elif action == "move_back":
            cmd = {"type": "move", "direction": "back", "distance": i.get("distance", 0.5)}
        elif action == "turn_left":
            cmd = {"type": "turn", "direction": "left", "angle_deg": i.get("angle_deg", 15)}
        elif action == "turn_right":
            cmd = {"type": "turn", "direction": "right", "angle_deg": i.get("angle_deg", 15)}
        elif action == "grab":
            cmd = {"type": "manipulator", "op": "grab"}
        elif action == "release":
            cmd = {"type": "manipulator", "op": "release"}
        elif action == "custom":
            cmd = i.get("command")

        s["last_action"] = action
        s["last_command"] = cmd
        s["last_job_id"] = job_id

        receipt = {
            "type": "game_control_event",
            "ts": ts,
            "player": self.pid,
            "action": action,
            "job_id": job_id,
            "robot_command": cmd,
        }
        self.state = s
        return s, receipt


class RobotWorld(World):
    """
    Robot controller.
    Receives commands (possibly from GameControlWorld via carrier).
    """
    def __init__(self, p: Persistence, port: str = "/dev/ttyUSB0"):
        super().__init__("RobotWorld", p, "robot", "Robot")
        self.ser = None
        if SERIAL_AVAILABLE:
            try:
                self.ser = serial.Serial(port, 9600, timeout=1)
                time.sleep(2)
            except Exception as e:
                print(f"[WARN] Could not open serial port {port}: {e}")
                self.ser = None

        self.state = {
            "last_cmd": None,
            "last_job_id": None,
            "last_status": "idle",
            "_ts": time.time(),
        }

    def execute_logic(self, i: Dict[str, Any]):
        ts = time.time()
        s = dict(self.state, _ts=ts)
        r: Dict[str, Any] = {"type": "robot_action", "ts": ts}

        cmd = i.get("command")
        job_id = i.get("job_id")
        status = i.get("status", "pending")
        player = i.get("player")  # if propagated from upstream

        if self.ser and cmd is not None:
            self.ser.write((_dumps(cmd) + "\n").encode())
            r["sent_to_hardware"] = True

        s["last_cmd"] = cmd
        s["last_job_id"] = job_id
        s["last_status"] = status
        s["last_player"] = player

        r["command"] = cmd
        r["job_id"] = job_id
        r["status"] = status
        r["player"] = player

        self.state = s
        return s, r

    # subscriber from GameControlWorld blocks
    def handle_game_block(self, b: Block):
        receipt = json.loads(b.receipt)
        if receipt.get("type") != "game_control_event":
            return
        cmd = receipt.get("robot_command")
        job_id = receipt.get("job_id")
        player = receipt.get("player")
        if cmd is None:
            return
        self.step(
            {
                "command": cmd,
                "job_id": job_id,
                "status": "pending",
                "player": player,
            }
        )


class RewardWorld(World):
    """
    Pays players when robot jobs complete.
    """
    def __init__(self, p: Persistence, runtime: str = "RewardEngine"):
        super().__init__("RewardWorld", p, "reward_system", runtime, False)
        self.state = {"total_rewards_issued": 0.0, "_ts": time.time()}

    def execute_logic(self, i: Dict[str, Any]):
        ts = time.time()
        s = dict(self.state, _ts=ts)
        receipt = {
            "type": "reward_event",
            "ts": ts,
            "player_id": i.get("player_id"),
            "amount": i.get("amount"),
            "job_id": i.get("job_id"),
        }
        self.state = s
        return s, receipt

    def reward_player(self, player_id: str, amount: float, job_id: Optional[str] = None):
        ts = time.time()
        self.p.reward(player_id, amount)
        self.state["total_rewards_issued"] += amount
        self.state["_ts"] = ts
        self.step(
            {
                "event": "reward",
                "player_id": player_id,
                "amount": amount,
                "job_id": job_id,
            }
        )

    # subscriber to RobotWorld completion
    def handle_robot_block(self, b: Block):
        receipt = json.loads(b.receipt)
        if receipt.get("type") != "robot_action":
            return
        status = receipt.get("status")
        job_id = receipt.get("job_id")
        player = receipt.get("player")
        if status == "completed" and player:
            self.reward_player(player, amount=10.0, job_id=job_id)


# =====================================================
# OMNIWORLD — MULTI-WORLD ORCHESTRATOR
# =====================================================

OMNIWORLD_LABEL = authorized_label("OmniWorld", "OMNI999ALL")


class OmniWorld(World):
    """
    OmniWorld: multi-world orchestrator.

    - Has its own ON/OFF state.
    - Has per-module ON/OFF for each major world.
    - Can route a single input to any other world (if allowed).
    - Logs its own block describing what it did.

    It does NOT duplicate the other worlds' logic.
    Instead, it calls their .step(...) methods via the Carrier,
    which keeps the substrate neutral and modular.
    """

    def __init__(
        self,
        p: Persistence,
        carrier,
        player: str = "omni_admin",
        runtime: str = "OmniWorld",
        enabled: bool = True,
        module_flags: Optional[Dict[str, bool]] = None,
    ):
        super().__init__("OmniWorld", p, player, runtime, False)
        self.carrier = carrier
        self.enabled = enabled

        # Default: everything known is enabled (you can turn off individually).
        default_modules = {
            "EnergyHarvesterWorld": True,
            "FlywheelPlantWorld": True,
            "WindupGeneratorWorld": True,
            "RobotWorld": True,
            "AsteroidsGameWorld": True,
            "GameControlWorld": True,
            "RewardWorld": True,
            "MysteryTechnologyWorld": True,
            "PublicCommonsWorld": True,
            "PrivateConsortiumWorld": True,
        }
        if module_flags is not None:
            default_modules.update(module_flags)

        self.modules: Dict[str, bool] = default_modules
        self.state.update(
            {
                "enabled": self.enabled,
                "modules": dict(self.modules),
                "last_route": None,
                "_ts": time.time(),
            }
        )

    def _set_enabled(self, value: bool):
        self.enabled = bool(value)
        self.state["enabled"] = self.enabled

    def _set_module(self, name: str, value: bool):
        self.modules[name] = bool(value)
        self.state["modules"] = dict(self.modules)

    def execute_logic(self, i: Dict[str, Any]):
        """
        Supported patterns:

        1) Toggle OmniWorld itself:
           { "op": "set_enabled", "value": true/false }

        2) Toggle a specific module (per world):
           { "op": "set_module", "module": "RobotWorld", "value": false }

        3) Route a command to another world:
           {
               "op": "route",
               "target_world": "RobotWorld",
               "payload": { ... original inputs for that world ... }
           }

        If 'op' is omitted, defaults to "route".
        """
        ts = time.time()
        s = dict(self.state, _ts=ts)
        op = i.get("op", "route")

        # Base receipt
        r: Dict[str, Any] = {
            "type": "omni_event",
            "ts": ts,
            "op": op,
        }

        # --- 1) Enable / disable OmniWorld itself ---
        if op == "set_enabled":
            value = bool(i.get("value", True))
            self._set_enabled(value)
            r["enabled"] = self.enabled
            s["enabled"] = self.enabled
            self.state = s
            return s, r

        # --- 2) Enable / disable a specific module/world ---
        if op == "set_module":
            mod = i.get("module")
            if not isinstance(mod, str):
                r["error"] = "module_name_required"
                self.state = s
                return s, r
            value = bool(i.get("value", True))
            self._set_module(mod, value)
            r["module"] = mod
            r["value"] = value
            s["modules"] = dict(self.modules)
            self.state = s
            return s, r

        # --- 3) Route a command to another world ---
        if op == "route":
            target_name = i.get("target_world")
            payload = i.get("payload", {})

            s["last_route"] = {
                "target_world": target_name,
                "ts": ts,
            }

            # If OmniWorld itself is disabled, do nothing except log.
            if not self.enabled:
                r["status"] = "omni_disabled"
                r["target_world"] = target_name
                self.state = s
                return s, r

            if not isinstance(target_name, str):
                r["error"] = "target_world_required"
                self.state = s
                return s, r

            # Check module permission for that world
            if not self.modules.get(target_name, False):
                r["status"] = "blocked_by_module_flag"
                r["target_world"] = target_name
                self.state = s
                return s, r

            target_world = self.carrier.get(target_name)
            if target_world is None:
                r["status"] = "target_not_found"
                r["target_world"] = target_name
                self.state = s
                return s, r

            # Call the target world's step, let it create its own block
            try:
                target_block = target_world.step(payload)
                r["status"] = "routed"
                r["target_world"] = target_world.name
                r["target_block_id"] = target_block.id
                r["target_receipt"] = json.loads(target_block.receipt)
            except Exception as e:
                r["status"] = "route_failed"
                r["target_world"] = target_name
                r["error"] = str(e)

            self.state = s
            return s, r

        # --- Unknown operation: just log it structurally ---
        r["status"] = "unknown_op"
        self.state = s
        return s, r


# =====================================================
# CARRIER (BLOCK BUS)
# =====================================================

class Carrier:
    """
    Block-level broadcast bus.
    Worlds subscribe via normal world.step (through monkey-patch wrap).
    """
    def __init__(self):
        self.w: Dict[str, World] = {}

    def reg(self, w: World) -> None:
        self.w[w.name] = w

    def get(self, name: str) -> Optional[World]:
        return self.w.get(name)

    def broadcast_block(self, b: Block) -> None:
        world_name = b.world

        if world_name == "GameControlWorld":
            robot_world = self.get("RobotWorld")
            if isinstance(robot_world, RobotWorld):
                robot_world.handle_game_block(b)

        if world_name == "RobotWorld":
            reward_world = self.get("RewardWorld")
            if isinstance(reward_world, RewardWorld):
                reward_world.handle_robot_block(b)


def wrap_world_step(world: World, carrier: Carrier):
    orig_step = world.step

    def wrapped(i: Dict[str, Any], witness: Optional[Dict[str, Any]] = None):
        b = orig_step(i, witness)
        carrier.broadcast_block(b)
        return b

    world.step = wrapped  # type: ignore[attr-defined]


# =====================================================
# FASTAPI SERVER (OPTIONAL)
# =====================================================

def build_fastapi_app(
    carrier: Carrier,
    persistence: Persistence,
    features: Dict[str, bool],
) -> FastAPI:
    app = FastAPI(title="Atomic Chronicle v4.0", version="4.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # expose feature set for runtime toggling
    @app.get("/")
    def root():
        return {
            "message": "Atomic Chronicle Unified Ecosystem Online",
            "version": "4.0",
            "worlds": list(carrier.w.keys()),
            "features": RUNTIME_FEATURES,
        }

    @app.get("/wallet/{pid}")
    def get_wallet(pid: str):
        return {"pid": pid, "balance": persistence.wallet(pid)}

    @app.post("/worlds/{world_name}/step")
    async def step_world(world_name: str, data: Dict[str, Any]):
        w = carrier.get(world_name)
        if w is None:
            return {"error": f"World '{world_name}' not found"}
        try:
            blk = w.step(data)
            return {
                "block_id": blk.id,
                "world": w.name,
                "world_state": w.state,
                "receipt": json.loads(blk.receipt),
            }
        except Exception as e:
            return {"error": str(e)}

    @app.post("/broadcast")
    async def broadcast_event(event: dict):
        """
        Broadcast as 'ecosystem heartbeat' (does NOT create blocks by itself;
        individual worlds decide if they want to log it).
        """
        ts = time.time()
        event = dict(event, ts=ts)
        # simple example: send into all worlds as a step with zero cost
        for w in carrier.w.values():
            try:
                w.step({"event": "broadcast", "payload": event})
            except Exception as e:
                print(f"[WARN] broadcast to {w.name} failed: {e}")
        return {"status": "broadcasted", "event": event}

    # runtime toggling of features
    @app.post("/toggle/{feature_key}")
    async def toggle_feature(feature_key: str, body: Dict[str, Any]):
        if feature_key not in RUNTIME_FEATURES:
            return {"error": f"Unknown feature {feature_key}"}
        value = bool(body.get("value", True))
        set_feature_runtime(feature_key, value)
        return {"feature": feature_key, "value": value}

    # websocket for block stream (lightweight demo)
    event_subscribers: List[WebSocket] = []

    async def notify_subscribers(message: dict):
        to_remove = []
        for ws in event_subscribers:
            try:
                await ws.send_json(message)
            except Exception:
                to_remove.append(ws)
        for ws in to_remove:
            if ws in event_subscribers:
                event_subscribers.remove(ws)

    def forwarder(block: Block):
        msg = {
            "type": "NEW_BLOCK",
            "world": block.world,
            "block_id": block.id,
            "ts": block.ts,
            "cost": block.cost,
        }
        try:
            loop = asyncio.get_event_loop()
            asyncio.run_coroutine_threadsafe(notify_subscribers(msg), loop)
        except RuntimeError:
            # no running loop (e.g. when not under uvicorn)
            pass

    # subscribe all worlds to forwarder
    for w in carrier.w.values():
        w.subscribe(forwarder)

    @app.websocket("/ws")
    async def websocket_feed(ws: WebSocket):
        await ws.accept()
        event_subscribers.append(ws)
        await ws.send_json({"welcome": "Connected to Atomic Chronicle v4.0 stream"})
        try:
            while True:
                await ws.receive_text()
        finally:
            if ws in event_subscribers:
                event_subscribers.remove(ws)

    # background heartbeat (if enabled)
    if features.get("enable_carrier_heartbeat", False):

        async def background_autoloop():
            while True:
                await asyncio.sleep(10.0)
                heartbeat = {"auto_event": "heartbeat", "ts": time.time()}
                for w in carrier.w.values():
                    try:
                        w.step(heartbeat)
                    except Exception as e:
                        print(f"[WARN] heartbeat to {w.name} failed: {e}")

        @app.on_event("startup")
        async def startup_event():
            asyncio.create_task(background_autoloop())

    return app


# =====================================================
# MAIN INITIALIZATION
# =====================================================

def main():
    print("\n=== ATOMIC CHRONICLE ECOSYSTEM v4.0 ===\n")

    # 1) Load features (config + CLI)
    features = load_features(CONFIG_PATH)
    features = apply_cli_overrides(features)

    # copy into runtime-visible dict
    RUNTIME_FEATURES.clear()
    RUNTIME_FEATURES.update(features)

    # 2) Choose persistence: transparency or plain
    if features.get("enable_transparency", True):
        persistence: Persistence = TransparencyEconomy("atomic_unified.db")
        print("[INIT] Using TransparencyEconomy persistence")
    else:
        persistence = Persistence("atomic_unified.db")
        print("[INIT] Using plain Persistence")

    carrier = Carrier()

    # 3) Instantiate worlds depending on feature flags
    if features["enable_energy_world"]:
        energy_world = EnergyHarvesterWorld(persistence)
        carrier.reg(energy_world)
        wrap_world_step(energy_world, carrier)
        persistence.reward("harvester", 100.0)

    if features["enable_flywheel_world"]:
        flywheel_world = FlywheelPlantWorld(persistence)
        carrier.reg(flywheel_world)
        wrap_world_step(flywheel_world, carrier)

    if features["enable_windup_world"]:
        windup_world = WindupGeneratorWorld(persistence)
        carrier.reg(windup_world)
        wrap_world_step(windup_world, carrier)

    if features["enable_robot_world"]:
        robot_world = RobotWorld(persistence)
        carrier.reg(robot_world)
        wrap_world_step(robot_world, carrier)

    if features["enable_game_world"]:
        game_world = AsteroidsGameWorld(persistence)
        carrier.reg(game_world)
        wrap_world_step(game_world, carrier)

    if features["enable_game_control_world"]:
        game_control_world = GameControlWorld(persistence, player="player1")
        carrier.reg(game_control_world)
        wrap_world_step(game_control_world, carrier)

    if features["enable_reward_world"]:
        reward_world = RewardWorld(persistence)
        carrier.reg(reward_world)
        wrap_world_step(reward_world, carrier)

    if features["enable_mystery_world"]:
        mystery_world = MysteryTechnologyWorld(persistence)
        carrier.reg(mystery_world)
        wrap_world_step(mystery_world, carrier)

    if features["enable_public_commons"]:
        public_world = PublicCommonsWorld(persistence)
        carrier.reg(public_world)
        wrap_world_step(public_world, carrier)

    if features["enable_private_consortium"]:
        private_world = PrivateConsortiumWorld(persistence)
        carrier.reg(private_world)
        wrap_world_step(private_world, carrier)

    # 3️⃣ Instantiate OmniWorld
    if features.get("enable_omni_world", True):
        omni_world = OmniWorld(persistence, carrier)
        carrier.reg(omni_world)
        wrap_world_step(omni_world, carrier)

    print(f"[INIT] Worlds registered: {', '.join(carrier.w.keys())}")

    # seed some balances
    persistence.reward("researcher_X", 1000.0)

    # 4) Quick CLI demo transactions (non-interactive)
    if features.get("enable_energy_world"):
        blk = carrier.get("EnergyHarvesterWorld").step({"energy_produced_kwh": 2.4})
        print(f"[DEMO] Energy block: {blk.id} | Harvester balance: {persistence.wallet('harvester'):.4f}")

    if features.get("enable_mystery_world") and isinstance(persistence, TransparencyEconomy):
        mystery_world = carrier.get("MysteryTechnologyWorld")
        mystery_world.step({"action": "discover", "technology_id": "mystic_core_v1"})
        reveal_block = mystery_world.step(
            {
                "action": "attempt_reveal",
                "technology_id": "mystic_core_v1",
                "revealer_id": "researcher_X",
                "evidence": {"description": "Quantum harmonic emitter"},
            }
        )
        receipt = json.loads(reveal_block.receipt)
        print(f"[DEMO] Revelation success: {receipt.get('success', False)} | Researcher_X: {persistence.wallet('researcher_X'):.4f}")

    if features.get("enable_public_commons"):
        public_world = carrier.get("PublicCommonsWorld")
        public_world.step(
            {
                "action": "contribute",
                "entity_id": "open_battery_design",
                "license_fee": 0.1,
            }
        )
        commons_block = public_world.step({})
        receipt = json.loads(commons_block.receipt)
        print(f"[DEMO] Commons receipt type: {receipt.get('type')}")

    # 5) Optionally run FastAPI server
    if FASTAPI_AVAILABLE and features.get("enable_fastapi_server", True):
        app = build_fastapi_app(carrier, persistence, features)
        print("[SERVER] Starting FastAPI server at http://0.0.0.0:8080")
        uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
    else:
        print("[SERVER] FastAPI server disabled or dependency not available.")
        print("[DONE] CLI demo completed.")


if __name__ == "__main__":
    main()
