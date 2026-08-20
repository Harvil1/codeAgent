"""scripts/check_comments_only.py——检查「只改了注释」的验证脚本。

背景：全项目注释改写时，改注释的工人只许动注释和 docstring。本脚本用机器
验证这条铁律：把新旧两版代码都剥掉注释和 docstring，规范化后对比，代码部分
必须一字不差。

原理：Python 的 ast.parse 天然丢弃 # 注释；docstring 在语法树里是函数/类/
模块体开头的字符串表达式节点，专门删掉；剩下的语法树用 ast.unparse 还原成
文本再比。

用法（项目根目录）：
    uv run python scripts/check_comments_only.py 文件1 文件2 ...
对比「git 上一版（HEAD）」和「工作区当前版」；参数也可以传目录，会自动
展开成目录下（含子目录）的所有 .py 文件。
退出码：0 = 全部只动了注释；1 = 有文件动了代码或解析失败。
"""

import ast
import subprocess
import sys
from pathlib import Path


def expand_paths(args: list[str]) -> list[str]:
    """把参数里的目录展开成它下面（含子目录）的所有 .py 文件路径。"""
    files: list[str] = []
    for arg in args:
        p = Path(arg)
        if p.is_dir():
            files.extend(str(f).replace("\\", "/") for f in sorted(p.rglob("*.py")))
        else:
            files.append(arg)
    return files


class _DocstringRemover(ast.NodeTransformer):
    """删掉模块/类/函数体开头的 docstring 节点。

    函数体删空时补一个 pass 占位，保证 ast.unparse 输出仍是合法代码。
    注意：只删「体开头第一个语句是字符串」的情况——那才是 docstring；
    赋值给变量的字符串不是，不能碰。
    """

    def _strip(self, node):
        body = node.body
        first_is_doc = (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        )
        if first_is_doc:
            node.body = [ast.Pass()] if len(body) == 1 else body[1:]
        return node

    def visit_Module(self, node):
        self.generic_visit(node)
        return self._strip(node)

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        return self._strip(node)

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        return self._strip(node)

    def visit_AsyncFunctionDef(self, node):
        self.generic_visit(node)
        return self._strip(node)


def strip_comments_and_docstrings(source: str) -> str:
    """剥掉注释和 docstring，返回规范化后的代码文本。"""
    tree = ast.parse(source)
    tree = _DocstringRemover().visit(tree)
    return ast.unparse(tree)


def load_head_version(path: str) -> str | None:
    """取文件在 git 上一版（HEAD）的内容；不在 git 里返回 None。

    git show 要正斜杠路径（Windows 反斜杠它不认）。
    """
    git_path = path.replace("\\", "/")
    result = subprocess.run(
        ["git", "show", f"HEAD:{git_path}"],
        capture_output=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        return None
    return result.stdout


def check_file(path: str) -> list[str]:
    """检查单个文件，返回问题列表（空列表 = 只动了注释，通过）。"""
    head = load_head_version(path)
    if head is None:
        return [f"不在 git 里（无法对比）: {path}"]
    try:
        with open(path, encoding="utf-8") as fh:
            current = fh.read()
    except (OSError, UnicodeDecodeError) as exc:
        return [f"读不了文件 {path}: {exc}"]
    try:
        old_stripped = strip_comments_and_docstrings(head)
    except SyntaxError:
        return [f"旧版解析失败: {path}"]
    try:
        new_stripped = strip_comments_and_docstrings(current)
    except SyntaxError:
        return [f"新版解析失败（可能改坏了语法）: {path}"]
    if old_stripped == new_stripped:
        return []
    old_lines = old_stripped.splitlines()
    new_lines = new_stripped.splitlines()
    for i, (a, b) in enumerate(zip(old_lines, new_lines)):
        if a != b:
            return [
                f"代码被改动 {path}: 剥掉注释后第 {i + 1} 行不同\n  旧: {a}\n  新: {b}"
            ]
    return [
        f"代码被改动 {path}: 行数不同（旧 {len(old_lines)} 行 / 新 {len(new_lines)} 行）"
    ]


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    paths = expand_paths(argv[1:])
    all_problems: list[str] = []
    for path in paths:
        all_problems.extend(check_file(path))
    if all_problems:
        print("\n".join(all_problems))
        print(f"\n共 {len(all_problems)} 个问题——有文件的代码被动了，不是只改注释")
        return 1
    print(f"全部 {len(paths)} 个文件都只动了注释，代码一字未变")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
