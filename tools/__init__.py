"""tools 包：工具注册表与各工具模块。

工具模块在 import 时通过 registry.register() 自注册。
model_tools.ensure_tools_discovered() 会触发自动发现。
"""
