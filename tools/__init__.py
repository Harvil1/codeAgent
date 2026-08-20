"""tools 包：所有给 LLM 用的工具都放这里。

工作方式：每个工具模块在被 import 的那一刻，就自己调用
registry.register() 把自己登记到中央工具注册表（不需要谁来集中登记）。
而 model_tools.ensure_tools_discovered() 负责把这些模块扫一遍触发 import，
注册就自动完成了。
"""
