"""测试全局卫生：按测试快照/还原模块级单例状态。

跨测试污染：RuntimeContext.initialize() 会把
config / aux provider 注入 agent.hook_exec 的模块级全局（cli.py:722），
测试结束不还原会改变后续测试的行为——例如 http handler 被 feature flag
静默门控（config 非 None 后 _is_handler_allowed 走 flag 判定），
tests/test_hooks.py::test_http_hook_posts_and_parses 在特定文件顺序下
KeyError: 'url'（hook 被跳过、fake_post 从未被调）。

这里统一按测试快照/还原，保证测试顺序无关。
"""
import pytest


@pytest.fixture(autouse=True)
def _restore_hook_exec_providers():
    """每个测试前后快照/还原 hook_exec 的 config/aux 全局 provider。"""
    import agent.hook_exec as he
    saved_config = he._CONFIG_PROVIDER
    saved_aux = he._AUX_ROUTER_PROVIDER
    yield
    he._CONFIG_PROVIDER = saved_config
    he._AUX_ROUTER_PROVIDER = saved_aux
