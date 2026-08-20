"""内置技能数据包（里面全是 Markdown 技能文件，没有 Python 逻辑）。

这个文件存在的唯一目的：让打包工具（setuptools）把 skills/ 目录当成一个
Python 包，从而能把 SKILL.md、模板、脚本一起塞进安装包（wheel）。运行时
真正用来找技能目录的函数是 constants.builtin_skills_dir()，所以这里
不需要写任何代码。
"""
