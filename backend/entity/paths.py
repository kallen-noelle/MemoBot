import os
import re
import shutil
from pathlib import Path

_SAFE_THREAD_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


class Paths:
    """
    Centralized path configuration for MemoBot application data.

    Directory layout (host side):
        {uid}/
        ├── memory.json
        ├── USER.md          <-- global user profile 
        ├── files.json         <-- list of uploaded files

    BaseDir resolution (in priority order):
        1. Constructor argument `base_dir`
        2. DEER_FLOW_HOME environment variable
        3. Local dev fallback: cwd/.MemoBot/  (when cwd is the backend/ dir)
        4. Default: $HOME/.MemoBot/
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self._base_dir = Path(base_dir).resolve() if base_dir is not None else None

   
    def host_base_dir(self) -> Path:
        """Host-visible base dir for Docker volume mount sources.

        When running inside Docker with a mounted Docker socket (DooD), the Docker
        daemon runs on the host and resolves mount paths against the host filesystem.
        Set DEER_FLOW_HOST_BASE_DIR to the host-side path that corresponds to this
        container's base_dir so that sandbox container volume mounts work correctly.

        Falls back to base_dir when the env var is not set (native/local execution).
        """
        if env := os.getenv("MEMO_BOT_HOST_BASE_DIR"):
            return Path(env)
        
        cwd = Path.cwd()
        if (cwd / "backend").exists():
            return cwd / "MemoBot"
        return Path.home() / "MemoBot"

    def base_dir(self, uid: str) -> Path:
        """Root directory for all application data."""
        if self._base_dir is not None:
            return self._base_dir

        if env_home := os.getenv("MEMO_BOT_HOME"):
            return Path(env_home).resolve()
        return Path.home() / uid
    def memory_file(self, uid: str) -> Path:
        """Path to the persisted memory file: `{base_dir}/memory.json`."""
        return self.base_dir(uid) / "memory.json"

    def files_file(self, uid: str) -> Path:
        """Path to the list of uploaded files file: `{base_dir}/files.json`."""
        return self.base_dir(uid) / "files.json"

    def user_md_file(self, uid: str) -> Path:

        """Path to the global user profile file: `{base_dir}/USER.md`."""
        return self.base_dir(uid) / "USER.md"


# ── Singleton ────────────────────────────────────────────────────────────

_paths: Paths | None = None


def get_paths() -> Paths:
    """Return the global Paths singleton (lazy-initialized)."""
    global _paths
    if _paths is None:
        _paths = Paths()
    return _paths


def resolve_path(path: str) -> Path:
    """Resolve *path* to an absolute ``Path``.

    Relative paths are resolved relative to the application base directory.
    Absolute paths are returned as-is (after normalisation).
    """
    p = Path(path)
    if not p.is_absolute():
        p = get_paths().host_base_dir() / path
    return p.resolve()
