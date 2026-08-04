"""PEP 517 shim that builds against the interpreter's installed torch.

A PyTorch C++/CUDA extension has to be compiled and linked against the exact
torch that will later import it, so the build cannot happen in an isolated
environment holding a different (or no) torch. Rather than forcing every caller
to remember ``--no-build-isolation``, this backend puts the interpreter's own
site-packages back on ``sys.path`` and then defers everything to setuptools.
"""

import site
import sys
import sysconfig

_candidates = [sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]]
try:
    _candidates += site.getsitepackages()
except AttributeError:  # pragma: no cover - not present in every environment
    pass

for _path in _candidates:
    if _path and _path not in sys.path:
        sys.path.append(_path)

from setuptools.build_meta import *  # noqa: F401,F403,E402
