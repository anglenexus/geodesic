"""
fi_trace.py - opt-in capture of FlashInfer's paged-KV index arrays from a live server.

Not run directly. run_phase4.sh links it as `sitecustomize.py` in a scratch folder under the results
directory and puts that folder on the server's PYTHONPATH; Python then imports it automatically at
startup. It does nothing unless FI_TRACE_DIR is set. When FlashInfer is first imported, it wraps the
plan()/begin_forward() methods of the batch decode and prefill wrappers, and every
FI_TRACE_EVERY-th call (default 50) saves the call's indptr / indices / last_page_len arrays
to FI_TRACE_DIR as .npz, up to FI_TRACE_MAX files per kind and process (default 300).
Any error disables capture and never affects serving. Manual equivalent of run_phase4.sh's trace step:

    mkdir -p /tmp/fihook && ln -sf "$PWD/fi_trace.py" /tmp/fihook/sitecustomize.py
    SERVER_ENV="PYTHONPATH=/tmp/fihook FI_TRACE_DIR=$HOME/qwen-serve/results/fi/trace" ./serve.sh restart
"""
import os
import sys

_DIR = os.environ.get("FI_TRACE_DIR")

if _DIR:
    import importlib.abc
    import importlib.util
    import inspect
    import time

    _EVERY = int(os.environ.get("FI_TRACE_EVERY", "50"))
    _MAX = int(os.environ.get("FI_TRACE_MAX", "300"))
    _count, _saved = {}, {}

    def _wrap(orig, kind):
        sig = inspect.signature(orig)

        def wrapped(self, *args, **kwargs):
            try:
                n = _count[kind] = _count.get(kind, 0) + 1
                if n % _EVERY == 0 and _saved.get(kind, 0) < _MAX:
                    import numpy as np
                    bound = sig.bind_partial(self, *args, **kwargs).arguments
                    arrays = {"t": np.array(time.time())}
                    for name, v in bound.items():
                        if hasattr(v, "detach") and any(s in name for s in ("indptr", "indices", "last_page_len")):
                            arrays[name] = v.detach().to("cpu").numpy()
                        elif name in ("page_size", "num_qo_heads", "num_kv_heads", "head_dim") and isinstance(v, int):
                            arrays[name] = np.array(v)
                    os.makedirs(_DIR, exist_ok=True)
                    np.savez_compressed(os.path.join(_DIR, f"{kind}_{os.getpid()}_{n:07d}.npz"), **arrays)
                    _saved[kind] = _saved.get(kind, 0) + 1
            except Exception as e:  # never break serving
                print(f"[fi_trace] capture error, disabling {kind}: {e}", file=sys.stderr)
                _saved[kind] = _MAX
            return orig(self, *args, **kwargs)

        return wrapped

    def _patch(mod):
        for cls_name, kind in (("BatchDecodeWithPagedKVCacheWrapper", "decode"),
                               ("BatchPrefillWithPagedKVCacheWrapper", "prefill")):
            cls = getattr(mod, cls_name, None)
            if cls is None:
                continue
            for meth in ("plan", "begin_forward"):
                orig = cls.__dict__.get(meth)
                if callable(orig):
                    setattr(cls, meth, _wrap(orig, kind))
        print(f"[fi_trace] capturing FlashInfer plan() inputs every {_EVERY} calls to {_DIR}", file=sys.stderr)

    class _Finder(importlib.abc.MetaPathFinder):
        """Patches flashinfer right after its first import, without importing it early."""

        def find_spec(self, name, path=None, target=None):
            if name != "flashinfer":
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            if spec is None or spec.loader is None:
                return spec
            exec_orig = spec.loader.exec_module

            def exec_module(module):
                exec_orig(module)
                try:
                    _patch(module)
                except Exception as e:
                    print(f"[fi_trace] patch failed: {e}", file=sys.stderr)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
