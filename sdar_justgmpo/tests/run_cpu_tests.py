"""Install CPU kernel oracles before importing the real SDAR implementation."""
import os
import sys
import types
import unittest
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
module = types.ModuleType('flash_attn.ops.triton.layer_norm')
module.rms_norm_fn = lambda *a, **kw: (_ for _ in ()).throw(AssertionError('Use dense RMSNorm'))
sys.modules[module.__name__] = module
suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern='test_*.py')
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
