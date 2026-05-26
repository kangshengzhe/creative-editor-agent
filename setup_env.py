"""Interactive secret-setup script for the Creative Editor Agent.

Why this exists
---------------
The agent reads its LLM credentials from a local ``.env`` file. Typing those
credentials into a normal terminal echoes them onto the screen and into
shell history, and pasting them into chat windows / commit messages is a
real-world way that secrets leak. This script avoids both:

* The API key is read with :func:`getpass.getpass`, which **does not echo**
  the input to the terminal and is not stored in shell history.
* The resulting ``.env`` is written with restrictive permissions (chmod 600
  on POSIX, ACL "owner read/write only" on Windows) so other users on the
  same machine cannot read it.
* Existing ``.env`` files are never silently overwritten — the script asks
  for confirmation first.

Usage
-----
::

    python setup_env.py

The script will prompt for:

* TOKENPONY_API_KEY   (hidden input)
* TOKENPONY_BASE_URL  (default: https://token-plan-cn.xiaomimimo.com/v1)
* TOKENPONY_MODEL     (default: mimo-v2.5-pro)

Re-run it any time you rotate the key.
"""

from __future__ import annotations

import getpass
import os
import stat
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

ROOT: Path = Path(__file__).resolve().parent
ENV_PATH: Path = ROOT / ".env"
GITIGNORE_PATH: Path = ROOT / ".gitignore"

DEFAULT_BASE_URL: str = "https://token-plan-cn.xiaomimimo.com/v1"
DEFAULT_MODEL: str = "mimo-v2.5-pro"

# Friendly identifier prefix expected by TokenPony — used purely for a soft
# sanity check, not security. The real validation happens server-side.
EXPECTED_KEY_PREFIX: str = "tp-"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the interactive setup. Returns a process exit code."""
    print("=" * 60)
    print("Creative Editor Agent — secure .env setup")
    print("=" * 60)
    print()

    if not _verify_gitignore_protects_env():
        print(
            "WARNING: .gitignore does not appear to ignore .env. "
            "Add a line containing '.env' before continuing."
        )
        if not _confirm("Continue anyway?", default=False):
            return 1

    if ENV_PATH.exists():
        print(f"An existing .env was found at {ENV_PATH}.")
        print("Re-running this script will overwrite it.")
        if not _confirm("Overwrite?", default=False):
            print("Aborted; existing .env left untouched.")
            return 0
        print()

    api_key = _prompt_api_key()
    base_url = _prompt_with_default(
        "TOKENPONY_BASE_URL", DEFAULT_BASE_URL
    )
    model = _prompt_with_default("TOKENPONY_MODEL", DEFAULT_MODEL)

    _write_env_file(
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    _lock_down_permissions(ENV_PATH)

    print()
    print(f"Wrote {ENV_PATH}")
    print("Permissions restricted to the current user only.")
    print()
    print("Next:")
    print("  1. Verify the file is git-ignored:  git check-ignore .env")
    print("  2. Activate your venv and run a smoke test.")
    print()
    print("Reminder: rotate this key immediately if it ever appears in")
    print("a chat window, screenshot, log file, or commit history.")
    return 0


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def _prompt_api_key() -> str:
    """Read the API key with a hidden prompt; reject obvious mistakes."""
    while True:
        key = getpass.getpass("TOKENPONY_API_KEY (hidden): ").strip()
        if not key:
            print("  -> empty key; please paste the value from TokenPony.")
            continue

        if len(key) < 20:
            print(
                "  -> that looks shorter than a real TokenPony key "
                f"({len(key)} chars). Try again."
            )
            continue

        if not key.startswith(EXPECTED_KEY_PREFIX):
            print(
                f"  -> warning: the key does not start with "
                f"'{EXPECTED_KEY_PREFIX}'. Continue only if you are sure."
            )
            if not _confirm("Use this value anyway?", default=False):
                continue

        # Belt-and-braces: confirm by asking for the last 4 characters again.
        # This avoids a clipboard mishap where the user pasted a stale value.
        confirm_tail = getpass.getpass(
            "Re-enter the LAST 4 characters of the key for verification: "
        ).strip()
        if confirm_tail != key[-4:]:
            print("  -> mismatch; let's start over.")
            continue
        return key


def _prompt_with_default(name: str, default: str) -> str:
    """Prompt with a default — empty input keeps the default."""
    raw = input(f"{name} [{default}]: ").strip()
    return raw if raw else default


def _confirm(question: str, *, default: bool) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    raw = input(f"{question} {suffix} ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes"}


# ---------------------------------------------------------------------------
# File writing & locking
# ---------------------------------------------------------------------------


def _write_env_file(*, api_key: str, base_url: str, model: str) -> None:
    """Write a fresh .env without echoing secrets to the terminal."""
    contents = (
        "# Auto-generated by setup_env.py — never commit this file.\n"
        "# Re-run setup_env.py to rotate the key.\n"
        f"TOKENPONY_API_KEY={api_key}\n"
        f"TOKENPONY_BASE_URL={base_url}\n"
        f"TOKENPONY_MODEL={model}\n"
    )
    # Write atomically: write to a sibling tmp, lock down its permissions,
    # then replace. Ensures no window where a world-readable .env exists.
    tmp_path = ENV_PATH.with_suffix(".env.tmp")
    tmp_path.write_text(contents, encoding="utf-8")
    _lock_down_permissions(tmp_path)
    os.replace(tmp_path, ENV_PATH)


def _lock_down_permissions(path: Path) -> None:
    """Restrict the file so only the current user can read or write it."""
    if os.name == "nt":
        _windows_lock(path)
    else:
        # POSIX: 0600 — owner read/write, no group / world access.
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def _windows_lock(path: Path) -> None:
    """Best-effort Windows ACL lockdown using icacls.

    Removes inherited ACEs and grants only the current user full control.
    Falls back silently if ``icacls`` is unavailable — the file is still
    inside the user's profile, which limits the blast radius.
    """
    user = os.environ.get("USERNAME") or getpass.getuser()
    if not user:
        return

    try:
        # Disable inheritance and remove all inherited ACEs.
        subprocess.run(
            ["icacls", str(path), "/inheritance:r"],
            check=True,
            capture_output=True,
        )
        # Grant the current user full control. Anyone else (including
        # Administrators that aren't this user) is implicitly denied.
        subprocess.run(
            ["icacls", str(path), "/grant:r", f"{user}:F"],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        # Don't crash the setup over a missing icacls — log and move on.
        print(
            f"NOTE: could not tighten ACL on {path.name} "
            f"({type(exc).__name__}). The file is still inside your user "
            "profile, but consider locking it manually."
        )


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------


def _verify_gitignore_protects_env() -> bool:
    """Return True iff the project's .gitignore mentions ``.env``."""
    if not GITIGNORE_PATH.exists():
        return False
    text = GITIGNORE_PATH.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == ".env" or stripped.startswith(".env"):
            return True
    return False


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(130)
