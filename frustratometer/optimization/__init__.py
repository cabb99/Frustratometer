def __getattr__(name):
    import importlib
    _opt = importlib.import_module(".optimization", __name__)
    _ip = importlib.import_module(".inner_product", __name__)

    _ns = {}
    for _mod in (_opt, _ip):
        for _k in dir(_mod):
            if not _k.startswith("_"):
                _ns[_k] = getattr(_mod, _k)
    globals().update(_ns)

    if name in _ns:
        return _ns[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")