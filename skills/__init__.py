"""内置技能数据包。

让 setuptools 能把 skills/ 作为包打包进 wheel（配合 pyproject 的
package-data 一起带 SKILL.md / 模板 / 脚本）。运行时定位技能用的是
constants.builtin_skills_dir()，这里不需要任何逻辑。
"""
