"""
curl_cffi 补丁 — macOS 12 上 _iconv 符号缺失问题。

用法: 在 import akshare 之前执行:
    patch_akshare()  # 注册 stub, import akshare, 然后卸载 stub
"""
import sys
import types


def patch_akshare():
    """
    为 akshare 提供 curl_cffi stub, import akshare, 然后清理 stub,
    避免影响 yfinance 等其他库。

    用法:
        import src.utils.curl_cffi_patch as ccp
        ak = ccp.patch_akshare()
    """
    _register_stub()
    try:
        import akshare
        return akshare
    finally:
        _unregister_stub()


def _register_stub():
    """注册 curl_cffi stub 到 sys.modules"""
    if "curl_cffi" in sys.modules:
        return  # 已加载

    curl_cffi = types.ModuleType("curl_cffi")
    curl_cffi.__version__ = "0.0.0-stub"
    curl_cffi.__curl_version__ = "8.0-stub"
    curl_cffi.__description__ = "stub"
    curl_cffi.__title__ = "curl_cffi"

    # requests 子模块 (轻量, 仅满足 import 不报错)
    requests_mod = types.ModuleType("curl_cffi.requests")

    # Session
    class _StubSession:
        pass

    requests_mod.Session = _StubSession

    # 顶层 get/post
    def _stub(*args, **kwargs):
        raise NotImplementedError("curl_cffi stub")
    for f in ["get", "post", "request", "head", "put", "delete", "patch"]:
        setattr(requests_mod, f, _stub)

    curl_cffi.requests = requests_mod
    sys.modules["curl_cffi"] = curl_cffi
    sys.modules["curl_cffi.requests"] = requests_mod


def _unregister_stub():
    """从 sys.modules 中移除 stub, 恢复干净状态"""
    for k in list(sys.modules):
        if k.startswith("curl_cffi"):
            del sys.modules[k]
