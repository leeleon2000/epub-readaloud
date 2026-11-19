"""One UI job: one or more `readaloud` commands, one after the other, stopping at the first that does not succeed.

    python -m pipeline.ui.runner --books ROOT -- run alice --until voices -- run alice --stages synth --docs 2

A command that starts with `script` runs one of the repository's scripts instead (the ones in SCRIPTS, which take
`--books` as the command line does): `-- script youtube_export.py alice --parts 3`.

Each command is printed before it starts, so the job's log reads like a terminal session. The exit code is the
failing command's (3 = paused by a warning), or 0.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = ("youtube_export.py", "pdnc_overrides.py")


def split(argv: list[str]) -> tuple[str, list[list[str]]]:
    if len(argv) < 3 or argv[0] != "--books":
        raise SystemExit("usage: python -m pipeline.ui.runner --books ROOT -- <readaloud args> [-- <readaloud args> ...]")
    root, cmds, cur = argv[1], [], None
    for a in argv[2:]:
        if a == "--":
            cur = []
            cmds.append(cur)
        elif cur is None:
            raise SystemExit("commands are separated by --")
        else:
            cur.append(a)
    cmds = [c for c in cmds if c]
    if not cmds:
        raise SystemExit("no command given")
    return root, cmds


def main(argv: list[str]) -> int:
    root, cmds = split(argv)
    for args in cmds:
        if args[0] == "script":
            if len(args) < 2 or args[1] not in SCRIPTS:
                print(f"not a script the UI runs: {' '.join(args[1:2])}", flush=True)
                return 1
            print(f"$ python scripts/{args[1]} " + " ".join(args[2:]), flush=True)
            cmd = [sys.executable, "-u", str(REPO / "scripts" / args[1]), *args[2:], "--books", root]
        else:
            print("$ readaloud " + " ".join(args), flush=True)
            cmd = [sys.executable, "-u", "-m", "pipeline.cli", "--books", root, *args]
        rc = subprocess.call(cmd, cwd=REPO)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
