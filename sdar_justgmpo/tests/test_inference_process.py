import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from inference_process import TokenProgress, run_inference


class InferenceProcessTests(unittest.TestCase):
    def test_empty_events_never_keep_a_stalled_engine_alive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'progress'
            progress = TokenProgress(path)
            progress(0)
            self.assertFalse(path.exists())
            with patch('inference_process.time.monotonic', side_effect=[10, 10.1, 12]):
                progress(3)
                progress(2)
                self.assertEqual(path.read_text(), '3')
                progress(1)
                self.assertEqual(path.read_text(), '6')

    def test_real_progress_allows_a_job_longer_than_the_idle_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root/'worker.py'
            script.write_text('import pathlib, time\n'
                f'p = pathlib.Path({str(root / "progress")!r})\n'
                'for i in range(8):\n    p.write_text(str(i))\n    time.sleep(0.2)\n')
            run_inference([sys.executable, str(script)], env=dict(os.environ),
                log=root/'worker.log', progress=root/'progress', timeout=0.8)

    def test_stall_kills_a_child_that_ignores_term_even_after_its_parent_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root/'child.py'
            child.write_text('import os, pathlib, signal, time\n'
                'signal.signal(signal.SIGTERM, signal.SIG_IGN)\n'
                f'pathlib.Path({str(root / "child.pid")!r}).write_text(str(os.getpid()))\n'
                'time.sleep(60)\n')
            parent = root/'parent.py'
            parent.write_text('import subprocess, sys, time\n'
                f'subprocess.Popen([sys.executable, {str(child)!r}])\n'
                'time.sleep(60)\n')
            started = time.monotonic()
            try:
                with self.assertRaisesRegex(TimeoutError, 'No inference token progress'):
                    run_inference([sys.executable, str(parent)], env=dict(os.environ),
                        log=root/'worker.log', progress=root/'progress', timeout=0.5)
                pid = int((root/'child.pid').read_text())
                # An unreaped zombie holds no CUDA context and cannot execute.
                state = Path(f'/proc/{pid}/stat')
                deadline = time.monotonic()+1
                while state.exists() and state.read_text().split()[2] != 'Z' and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(not state.exists() or state.read_text().split()[2] == 'Z')
                self.assertLess(time.monotonic()-started, 8)
            finally:
                if (root/'child.pid').exists():
                    try:
                        os.kill(int((root/'child.pid').read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
