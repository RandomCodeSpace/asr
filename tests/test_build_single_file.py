import subprocess
import sys
from pathlib import Path


def test_build_produces_app_py(tmp_path):
    repo_root = Path(__file__).parent.parent
    out_path = repo_root / "dist" / "app.py"
    if out_path.exists():
        out_path.unlink()
    res = subprocess.run(
        [sys.executable, "scripts/build_single_file.py"],
        cwd=repo_root, capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    assert out_path.exists()
    content = out_path.read_text()
    # Sanity checks — key symbols present
    assert "class Orchestrator" in content
    assert "class IncidentStore" in content
    assert "def main()" in content
    assert "from orchestrator." not in content, "intra-package imports should be rewritten"


def test_built_file_imports_cleanly(tmp_path):
    repo_root = Path(__file__).parent.parent
    out_path = repo_root / "dist" / "app.py"
    if not out_path.exists():
        subprocess.run([sys.executable, "scripts/build_single_file.py"],
                       cwd=repo_root, check=True)
    res = subprocess.run(
        [sys.executable, "-c", f"import importlib.util, sys; "
         f"spec = importlib.util.spec_from_file_location('app', '{out_path}'); "
         f"mod = importlib.util.module_from_spec(spec); "
         f"sys.modules['app'] = mod; spec.loader.exec_module(mod); "
         f"print('ok')"],
        capture_output=True, text=True,
    )
    assert "ok" in res.stdout, res.stderr
