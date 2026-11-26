import json
import os
import re

import numpy as np

from wsp import case as wsp_case
from wsp import upload
from wsp.rules import RULES, doc, payload_limit

P, SID = "qwen35-0.8b-think", "0009538688-0019"


class FakeS3:
    def __init__(self):
        self.objects = {}

    def upload_file(self, path, bucket, key):
        self.objects[key] = open(path, "rb").read()

    def get_object(self, Bucket, Key):  # noqa: N803
        import io
        return {"Body": io.BytesIO(self.objects[Key])}

    def head_object(self, Bucket, Key):  # noqa: N803
        return {"ContentLength": len(self.objects[Key])}


def test_package_and_upload(tmp_path, monkeypatch):
    case = tmp_path / "cases" / "c"
    (case / "edits").mkdir(parents=True)
    np.savez(case / "edits" / "e.npz", u15=np.ones(4, np.float32), cols15=np.arange(3))
    (case / "edits" / "e.json").write_text(json.dumps({"name": "e", "base": P, "meta": {"directions": "x.npz"}, "ops": [
        {"op": "dir_ablate", "tensor": "t", "u": "e.npz#u15", "cols": "e.npz#cols15", "alpha": 6}]}))
    (case / "NOTES.md").write_text("notes")
    monkeypatch.setattr(upload, "CASES", tmp_path / "cases")
    monkeypatch.setattr(wsp_case, "CASES", tmp_path / "cases")
    monkeypatch.setattr(upload, "WORK", tmp_path / "work")
    export = upload.export_upload

    def leaves_a_tree(spec, edit_dir, *args):  # e.g. a check killed while it built in the package's directory
        r = export(spec, edit_dir, *args)
        (edit_dir.parent / "tmpq" / "out").mkdir(parents=True)
        (edit_dir.parent / "tmpq" / "out" / "edit.json").write_text("{}")
        return r
    monkeypatch.setattr(upload, "export_upload", leaves_a_tree)
    out, man = upload.package("c", "e")
    assert [f["path"] for f in man["files"]] == ["edit/edit.json", "edit/edit.npz", "edit/REPORT.md"]  # only these
    monkeypatch.setattr(upload, "export_upload", export)
    spec = json.loads((out / "edit" / "edit.json").read_text())
    assert spec["ops"][0]["u"] == "edit.npz#u15" and "meta" not in spec
    assert spec["parent"] == "Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17"  # the base, by its full name
    s3 = FakeS3()
    keys = upload.upload(out, man, "e", "bucket", "staging/sid/", (None, None, None), client=s3)
    assert keys[-1] == "staging/sid/manifest.json" and list(s3.objects)[-1] == keys[-1]  # manifest last
    assert json.loads(s3.objects["staging/sid/manifest.json"])["schema"] == RULES["upload"]["schema"]


def test_upload_references_parent_without_copying_it(tmp_path, monkeypatch):
    case = tmp_path / "cases" / "c"
    (case / "edits").mkdir(parents=True)
    (case / SID / "edit").mkdir(parents=True)
    (case / SID / "edit" / "edit.json").write_text(json.dumps({"base": P, "ops": []}))
    (case / "edits" / "e.json").write_text(json.dumps({"name": "e", "base": P, "parent": f"../{SID}/edit/edit.json",
                                                       "ops": []}))
    (case / "NOTES.md").write_text("n")
    monkeypatch.setattr(upload, "CASES", tmp_path / "cases")
    monkeypatch.setattr(wsp_case, "CASES", tmp_path / "cases")
    monkeypatch.setattr(upload, "WORK", tmp_path / "work")
    out, man = upload.package("c", "e")
    assert {f["path"] for f in man["files"]} == {"edit/edit.json", "edit/REPORT.md"}  # nothing else
    assert json.loads((out / "edit" / "edit.json").read_text()) == {"base": P, "parent": SID, "ops": []}


def test_upload_references_the_start_by_its_sid(tmp_path, monkeypatch):
    """A case started from a task's <sid>/edit/ (init --parent) uploads "<sid>/edit.json", checked against the task's
    starts; a start the task does not offer is refused before anything is uploaded."""
    from wsp.edits import copy_parent, start_dir
    sids = [SID, "0009538688-0004"]  # e.g. the entry's parent and the champion
    task = tmp_path / "task"
    for s in sids:
        (task / s / "edit").mkdir(parents=True)
        (task / s / "edit" / "edit.json").write_text(json.dumps({"base": P, "ops": []}))
    (task / "task.json").write_text(json.dumps({"starts": [{"sid": sids[0], "role": "parent"},
                                                           {"sid": sids[1], "role": "champion"}],
                                                "base_name": "Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17"}))
    monkeypatch.setattr(upload, "CASES", tmp_path / "cases")
    monkeypatch.setattr(wsp_case, "CASES", tmp_path / "cases")
    monkeypatch.setattr(upload, "WORK", tmp_path / "work")
    for s in sids:
        case = tmp_path / "cases" / s
        start = copy_parent(task / s / "edit" / "edit.json", case / start_dir(task / s / "edit" / "edit.json"))
        assert start == case / s / "edit" / "edit.json"  # kept as the task keeps it
        (case / "edits").mkdir()
        (case / "edits" / "e.json").write_text(json.dumps({"name": "e", "base": P, "ops": [],
                                                           "parent": os.path.relpath(start, case / "edits")}))
        (case / "NOTES.md").write_text("n")
        out, _ = upload.package(s, "e", task=task / "task.json")
        assert json.loads((out / "edit" / "edit.json").read_text())["parent"] == s
    (task / "task.json").write_text(json.dumps({"starts": [{"sid": sids[1], "role": "champion"}]}))
    import pytest
    with pytest.raises(SystemExit, match="task offers"):
        upload.package(sids[0], "e", task=task / "task.json")


def test_published_parent_is_copied_verbatim(tmp_path):
    pub = tmp_path / "published" / "edit"  # a published submission's edit/ directory
    (pub / "derive").mkdir(parents=True)
    np.savez(pub / "edit.npz", u15=np.ones(4, np.float32))
    (pub / "edit.json").write_text(json.dumps({"name": "edit", "base": P, "ops": [
        {"op": "dir_ablate", "tensor": "t", "u": "edit.npz#u15"}]}))
    (pub / "REPORT.md").write_text("r")
    (pub / "derive" / "d.py").write_text("x = 1")
    case = tmp_path / "cases" / "c"
    (case / "edits").mkdir(parents=True)
    from wsp.edits import copy_parent
    copy_parent(pub / "edit.json", case / SID)
    for p in pub.rglob("*"):  # a case starts from a byte-identical copy of the published edit
        if p.is_file():
            assert upload._sha(case / SID / "edit" / p.relative_to(pub)) == upload._sha(p)


def test_protocol():
    pat = RULES["submission"]["path_pattern"]
    assert re.match(pat, "submissions/5" + "A" * 47 + "/my-entry/")
    assert not re.match(pat, "submissions/team/my-entry/")
    assert set(RULES["submission"]["required"]) <= set(RULES["submission"]["files"])
    assert set(RULES["upload"]["required"]) <= set(RULES["upload"]["files"])
    text = doc()
    assert "payload" in text and RULES["upload"]["schema"] in text


def test_payload_limit_grows_with_the_base_width(tmp_path):
    """4 MiB on the 0.8B (hidden 1024), 8 MiB on the 2B (hidden 2048): read from the base's config."""
    for hidden in (1024, 2048):
        (tmp_path / "config.json").write_text(json.dumps({"text_config": {"hidden_size": hidden}}))
        assert payload_limit(tmp_path) == hidden * 4096
    assert payload_limit(tmp_path) == 8 << 20


def test_report_carries_every_result_and_status_reads_the_upload_back(tmp_path, monkeypatch):
    """The packaged REPORT.md ends with the case's results table, where a holdout run made before its edit was frozen
    is marked; --status tells this package from another one."""
    case = tmp_path / "cases" / "c"
    (case / "edits").mkdir(parents=True)
    (case / "results").mkdir()
    (case / "case.json").write_text(json.dumps({"profile": "p", "contrast": "loop", "frozen": "e"}))
    (case / "edits" / "e.json").write_text(json.dumps({"name": "e", "base": P, "ops": []}))
    (case / "NOTES.md").write_text("notes")
    paired = {"base_acc": 0.5, "edit_acc": 0.6, "gain": 0.1, "se": 0.02, "rescues": 3, "losses": 1, "base_cut": 0.2,
              "edit_cut": 0.1}
    for model, frozen in (("e", None), ("f", "e")):  # e ran on holdout before it was frozen; f is another edit
        (case / "results" / f"{model}__holdout256.json").write_text(json.dumps(
            {"summary": {}, "paired": paired, "frozen": frozen}))
    monkeypatch.setattr(upload, "CASES", tmp_path / "cases")
    monkeypatch.setattr(wsp_case, "CASES", tmp_path / "cases")
    monkeypatch.setattr(upload, "WORK", tmp_path / "work")
    out, man = upload.package("c", "e")
    report = (out / "edit" / "REPORT.md").read_text()
    assert report.startswith("notes") and report.count("holdout256 (not frozen)") == 2
    s3 = FakeS3()
    assert upload.status(man, "e", "bucket", "p", (None,) * 3, client=s3)[0] is None  # nothing uploaded yet
    upload.upload(out, man, "e", "bucket", "p", (None,) * 3, client=s3)
    assert "this package" in upload.status(man, "e", "bucket", "p", (None,) * 3, client=s3)[1]
    other = {**man, "files": man["files"][:1]}
    assert "another package" in upload.status(other, "e", "bucket", "p", (None,) * 3, client=s3)[1]
