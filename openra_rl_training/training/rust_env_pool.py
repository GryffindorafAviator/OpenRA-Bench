"""Pool of Rust-backed OpenRA environments for fast in-process rollout.

Mirrors the surface of `env_pool.EnvPool` but swaps the gRPC-backed
`OpenRAEnvironment` for the native `openra_train.OpenRAEnv` (a Rust
deterministic simulator built via maturin/PyO3).

Key differences from the gRPC pool:
  * No game server / port allocation. Each `OpenRAEnv` is a
    self-contained Rust object — instantiation is microseconds.
  * Episodes are deterministic given (scenario_path, seed). The pool
    accepts a `seed_generator` (defaults to a monotonic counter) so
    callers can reseed each acquire if desired.
  * `step` accepts a list of `openra_train.Command` objects (build
    them with `Command.move_units(...)`, `Command.attack_unit(...)`,
    `Command.observe()`).

The pool is process-local; for honest parallelism, fan out via
`concurrent.futures.ProcessPoolExecutor` and have each worker own its
own `RustEnvPool` (or just instantiate `OpenRAEnv` directly).

Drop-in for the existing `env_pool.EnvPool`:
  * `acquire(timeout=...) -> env`
  * `release(env)`
  * `update_scenario(path)` — refreshes the default scenario for new
    envs and resets the seed counter.
  * `shutdown()` — drops references; Rust GC frees the worlds.
"""

from __future__ import annotations

import itertools
import logging
import queue
import threading
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)


def _default_seed_generator(start: int = 0) -> Iterator[int]:
    return itertools.count(start)


class RustEnvHandle:
    """Thin wrapper to give the Rust env a uniform `reset / step / close`
    interface that mirrors the gRPC env without leaking PyO3 types
    upward."""

    def __init__(self, scenario_path: str, seed: int):
        # Lazy import so import-time failures don't break the rest of
        # the training package on machines without the wheel built.
        import openra_train

        self._cls_command = openra_train.Command
        self._env = openra_train.OpenRAEnv(scenario_path, int(seed))
        self.scenario_path = scenario_path
        self.seed = int(seed)

    @property
    def Command(self):
        """Expose `openra_train.Command` for callers that want to
        construct Move/Attack/Observe payloads without re-importing."""
        return self._cls_command

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None and int(seed) != self.seed:
            # Re-instantiate to pick up the new seed (the underlying
            # Rust env owns the world; reset() re-uses the original
            # seed). Cheap — Rust instantiation is sub-millisecond.
            import openra_train
            self._env = openra_train.OpenRAEnv(self.scenario_path, int(seed))
            self.seed = int(seed)
        return self._env.reset()

    def step(self, commands: list[Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Apply a list of `openra_train.Command` objects, returns
        (obs, reward, done, info)."""
        return self._env.step(commands)

    def close(self) -> None:
        # No external resources to release; the Rust world is freed
        # when this handle is dropped.
        self._env = None


class RustEnvPool:
    """Thread-safe pool of Rust-backed environments.

    Args:
        size: Number of environment instances.
        scenario_path: Path to the rush-hour-style scenario YAML.
        seed_generator: Iterator yielding seeds for each new env. If
            None, defaults to `itertools.count(0)`.
        env_factory: Optional override; receives `(scenario_path, seed)`
            and returns an env-like object exposing `reset(...)` /
            `step(...)`. Useful for testing.
    """

    def __init__(
        self,
        size: int = 4,
        scenario_path: str = "",
        seed_generator: Iterator[int] | None = None,
        env_factory: Callable[[str, int], Any] | None = None,
    ):
        if size < 1:
            raise ValueError(f"RustEnvPool size must be >=1, got {size}")
        if not scenario_path:
            raise ValueError("RustEnvPool requires a non-empty scenario_path")

        self._size = size
        self._scenario_path = scenario_path
        self._seed_gen = seed_generator or _default_seed_generator()
        self._factory = env_factory or (lambda path, seed: RustEnvHandle(path, seed))
        self._pool: queue.Queue = queue.Queue()
        self._envs: list = []
        self._lock = threading.Lock()

        for _ in range(size):
            seed = next(self._seed_gen)
            env = self._factory(scenario_path, seed)
            self._envs.append(env)
            self._pool.put(env)

    def acquire(self, timeout: float = 30.0):
        """Get an available environment (blocks if all busy).

        The Rust env is in-process and deterministic, so the timeout
        only applies if all envs are checked out by other threads.
        """
        return self._pool.get(timeout=timeout)

    def release(self, env) -> None:
        """Return an environment to the pool."""
        self._pool.put(env)

    def update_scenario(self, scenario_path: str) -> None:
        """Replace the scenario used for newly-instantiated envs.

        Existing pooled envs keep their current scenario until released
        and re-acquired with `acquire(reset=True)` (callers should use
        this method in conjunction with explicit env replacement).
        """
        with self._lock:
            self._scenario_path = scenario_path

    @property
    def scenario_path(self) -> str:
        return self._scenario_path

    @property
    def size(self) -> int:
        return self._size

    @property
    def available(self) -> int:
        return self._pool.qsize()

    def shutdown(self) -> None:
        """Drop all env references and drain the pool."""
        with self._lock:
            for env in self._envs:
                try:
                    if hasattr(env, "close"):
                        env.close()
                except Exception:
                    logger.exception("Error closing Rust env")
            self._envs.clear()
            while not self._pool.empty():
                try:
                    self._pool.get_nowait()
                except queue.Empty:
                    break
