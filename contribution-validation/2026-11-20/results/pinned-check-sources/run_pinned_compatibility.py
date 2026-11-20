"""Re-run only the six requested existing regressions with isolated dependency pins."""
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import argparse
import gc
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import runpy
import shlex
import subprocess
import sys
import time
import traceback

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parents[2])
parser.add_argument("--checks-dir", type=Path, help="Directory containing the three existing regression scripts")
options = parser.parse_args()
ROOT = options.workspace_root.resolve()
CHECKS = options.checks_dir.resolve() if options.checks_dir else ROOT / "audit/validation"
OUT = ROOT / "audit/followup"
PINNED = OUT / "pinned-deps"
TORCH_PATH = ROOT / "torch26"
PYTHON = ROOT / ".venv/bin/python"
ENV = {
    "PYTHONPATH": f"{PINNED}:{TORCH_PATH}",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
SCRIPTS = {
    "serving": CHECKS / "serving_regressions.py",
    "model": CHECKS / "model_dataset_regressions.py",
    "root": CHECKS / "root_checks.py",
}
CASES = [
    ("serving", "finish", "api-length-finish"),
    ("serving", "arguments", "api-tool-arguments-json"),
    ("serving", "text_parts", "api-text-content-parts"),
    ("serving", "budget", "web-generation-budget"),
    ("model", "agent-history-tool-call-json", "agent-history-tool-call-json"),
    ("root", "agent_eos", "agent-train-turn-endings"),
]


def command(args):
    env = " ".join(f"{key}={shlex.quote(value)}" for key, value in ENV.items())
    return env + " " + shlex.join([str(PYTHON), "-u", *map(str, args)])


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def clear_repository_modules():
    for name in list(sys.modules):
        if name == "eval_llm" or name.split(".")[0] in {"model", "scripts", "trainer", "dataset"}:
            del sys.modules[name]


def main():
    started = time.monotonic()
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Six existing regression classes, each on its unchanged baseline and unchanged PR worktree",
        "installation": {
            "command": shlex.join([str(PYTHON), "-m", "pip", "install", "--disable-pip-version-check",
                                    "--target", str(PINNED), "pydantic==2.11.5", "jinja2==3.1.2"]),
            "target": str(PINNED),
            "log": str(OUT / "pinned-install.log"),
            "existing_virtualenv_or_torch26_modified": False,
        },
        "runner_command": command([Path(__file__).resolve(), "--workspace-root", ROOT, "--checks-dir", CHECKS]),
        "environment": ENV,
        "limitations": {
            "device": "CPU",
            "gpu_tested": False,
            "python_3_10_tested": False,
            "full_requirements_environment_recreated": False,
            "note": "This rerun specifically pins pydantic/Jinja2 and their resolved dependencies; the remaining existing runtime is reused.",
        },
        "test_scripts": {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                         for name, path in SCRIPTS.items()},
        "results": [],
    }
    report_file = OUT / "pinned-compatibility.json"
    def save():
        report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save()

    print("Loading isolated pinned runtime", flush=True)
    import pydantic
    import pydantic_core
    import jinja2
    import torch
    import transformers
    torch.set_num_threads(1)
    assert pydantic.__version__ == "2.11.5"
    assert pydantic_core.__version__ == "2.33.2"
    assert jinja2.__version__ == "3.1.2"
    for module in (pydantic, pydantic_core, jinja2):
        assert Path(module.__file__).resolve().is_relative_to(PINNED.resolve()), module.__file__
    assert Path(torch.__file__).resolve().is_relative_to(TORCH_PATH.resolve()), torch.__file__
    package_names = ["pydantic", "pydantic-core", "Jinja2", "MarkupSafe", "annotated-types",
                     "typing-extensions", "typing-inspection", "torch", "transformers", "datasets",
                     "numpy", "fastapi", "starlette", "httpx", "openai", "tokenizers", "pyarrow"]
    versions = {}
    for name in package_names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report["runtime"] = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "packages": versions,
        "module_paths": {module.__name__: module.__file__ for module in
                         (pydantic, pydantic_core, jinja2, torch, transformers)},
    }
    print(json.dumps(report["runtime"], ensure_ascii=False), flush=True)
    save()

    pr_map = {item["slug"]: item for item in json.loads((OUT / "pr-map.json").read_text())}
    original_path, original_argv = sys.path[:], sys.argv[:]
    for suite, case, slug in CASES:
        for stage, repo in (("before", ROOT / "minimind"), ("after", OUT / "pr-heads" / slug)):
            head = git(repo, "rev-parse", "HEAD")
            assert not git(repo, "diff", "HEAD", "--name-only"), f"Tracked worktree changes in {repo}"
            if stage == "after":
                assert head == pr_map[slug]["expected_sha"], (slug, head, pr_map[slug]["expected_sha"])
            script = SCRIPTS[suite]
            arguments = (["--repo", str(repo), "--case", case] if suite == "model" else [str(repo), case])
            clear_repository_modules()
            sys.path[:] = original_path
            sys.argv = [str(script), *arguments]
            capture = io.StringIO()
            exit_code, error_type = 0, None
            case_start = time.monotonic()
            print("START", slug, stage, flush=True)
            try:
                with redirect_stdout(capture), redirect_stderr(capture):
                    runpy.run_path(str(script), run_name="__main__")
            except SystemExit as error:
                exit_code = int(error.code or 0)
                if exit_code:
                    error_type = "SystemExit"
            except Exception as error:
                exit_code, error_type = 1, type(error).__name__
                traceback.print_exc(file=capture)
            output = capture.getvalue()
            log = OUT / f"pinned-{slug}-{stage}.log"
            log.write_text(output, encoding="utf-8")
            success = exit_code == 0
            if stage == "before":
                if suite == "model":
                    intended_failure = ("FAIL: test_serialized_calls_match_structured_history" in output
                                        and "AssertionError:" in output and "ERROR:" not in output)
                else:
                    intended_failure = error_type == "AssertionError"
                expected_outcome = not success and intended_failure
            else:
                expected_outcome = success
            unchanged = (head == git(repo, "rev-parse", "HEAD")
                         and not git(repo, "diff", "HEAD", "--name-only"))
            record = {
                "suite": suite, "case": case, "slug": slug, "stage": stage,
                "repository": str(repo), "commit": head,
                "pr_urls": [item["url"] for item in pr_map[slug]["prs"]],
                "command": command([script, *arguments]),
                "executed_in_shared_process": True,
                "exit_code": exit_code, "success": success, "error_type": error_type,
                "expected_outcome": expected_outcome, "production_head_and_files_unchanged": bool(unchanged),
                "elapsed_seconds": round(time.monotonic() - case_start, 3),
                "log": str(log), "output": output,
            }
            report["results"].append(record)
            save()
            print("END", slug, stage, "PASS" if success else "FAIL", "expected=" + str(expected_outcome), flush=True)
            gc.collect()
    sys.path[:], sys.argv[:] = original_path, original_argv
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["summary"] = {
        "case_pairs": len(CASES), "observations": len(report["results"]),
        "expected_baseline_failures": sum(r["stage"] == "before" and r["expected_outcome"] for r in report["results"]),
        "fixed_passes": sum(r["stage"] == "after" and r["success"] for r in report["results"]),
        "all_outcomes_expected": all(r["expected_outcome"] for r in report["results"]),
        "all_production_heads_and_files_unchanged": all(r["production_head_and_files_unchanged"] for r in report["results"]),
    }
    save()
    print(json.dumps(report["summary"]), flush=True)
    return 0 if report["summary"]["all_outcomes_expected"] and report["summary"]["all_production_heads_and_files_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
