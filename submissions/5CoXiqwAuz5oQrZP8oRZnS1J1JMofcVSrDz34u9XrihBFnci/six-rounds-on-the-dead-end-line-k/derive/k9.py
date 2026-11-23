"""Shared paths and small helpers of this method's scripts (no GPU work here).

Working files of every stage go to $WSP_WORK/k9/ (samples, traces, rows, fits, reports); the case's edit specs stay in
$WSP_CASES/<case>/edits/ as the toolkit keeps them. The run's deadline (from BRIEF.md) is kept in k9/deadline.json so
that chain9.py and up9.py can tell how much time is left.
"""
import json
import os
import re
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIMIT = 4_194_304          # rules.json edit.limits.max_payload_bytes (wsp.edits.spec_size of the spec's own ops)


def work():
    d = Path(os.environ.get("WSP_WORK", ".work")) / "k9"
    d.mkdir(parents=True, exist_ok=True)
    return d


def queue_logs():
    return Path(os.environ.get("WSP_WORK", ".work")) / "k4q" / "logs"


def cases():
    return Path(os.environ.get("WSP_CASES", "cases"))


def edits(case):
    return cases() / case / "edits"


def conf(case):
    return json.loads((cases() / case / "case.json").read_text())


def parent_ref(case):
    """The "parent" every spec of this method carries: the case's start, as the toolkit keeps it (<sid>/edit/)."""
    par = conf(case).get("parent")
    if not par:
        raise SystemExit(f"case {case} has no start: init it with --parent parent/edit/edit.json")
    return "../" + par


def start_sid(case):
    """The name of the directory that holds the start's edit/ in the case (in a task: its sid)."""
    return Path(conf(case)["parent"]).parent.parent.name


def model_path(case, model):
    from wsp import case as C  # noqa: PLC0415
    return C.model_path(case, model)


def note(text):
    """Append a line to k9/log.md (the report draws on it)."""
    with open(work() / "log.md", "a") as f:
        f.write(f"- {time.strftime('%H:%M:%S', time.gmtime())} UTC {text}\n")


def set_deadline(text):
    """text: 'YYYY-MM-DD HH:MM:SS UTC' as BRIEF.md prints it."""
    import calendar  # noqa: PLC0415
    t = calendar.timegm(time.strptime(text.replace("UTC", "").strip(), "%Y-%m-%d %H:%M:%S"))
    (work() / "deadline.json").write_text(json.dumps({"deadline": t, "text": text, "set": time.time()}))
    return t


def minutes_left():
    f = work() / "deadline.json"
    if not f.exists():
        return None
    return (json.loads(f.read_text())["deadline"] - time.time()) / 60


def job_log(name):
    p = queue_logs() / f"{name}.log"
    return p.read_text(errors="replace").replace("\r", "\n") if p.exists() else ""


def gen_rates(names):
    """Output tokens per second per GPU of finished sampling jobs ('[gen] ... (R tok/s)' lines of wsp.gen)."""
    out = {}
    for n in names:
        m = re.findall(r"\[gen\][^\n]*?\((\d+) tok/s\)", job_log(n))
        if m:
            out[n] = float(m[-1])
    return out
