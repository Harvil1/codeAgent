---
name: workflow-dsl
description: "workflow 工具的编排脚本 DSL 写法：agent/parallel/pipeline/phase 四个原语、args/budget 注入、结构化输出与断点续跑约定。当要写 workflow(action=\"run\", script=...) 或创建 .omnimate/workflows/*.py 时使用。"
---

# Workflow DSL（受约束 Python）

脚本必须定义 `async def main()`。引擎注入这些名字：

- `await agent(prompt)` —— 跑一个子代理（leaf 角色，minimal 工具集），返回最终文本；失败返回 None
- `await parallel([lambda: agent('a'), lambda: agent('b')])` —— 并发执行，单项失败该项为 None
- `await pipeline(items, [stage1, stage2])` —— 每个 item 依次过各 stage（stage 可 async），item 之间并发
- `with phase('阶段名'):` —— 纯日志分组
- `log(...)` / `args`（run 传入的参数 dict）/ `budget`（.spent/.remaining）

约束：禁 import/exec/eval/open/dunder（校验直接拒）；可用基础 builtins（len/range/sorted/json 除外——如需 JSON 用 agent 返回文本自行约定）。

示例——并发扫描+汇总：

```python
async def main():
    files = ['a.py', 'b.py', 'c.py']
    with phase('扫描'):
        reviews = await pipeline(
            files,
            [lambda f: agent(f'审查 {f} 的安全问题，只输出一行结论')],
        )
    with phase('汇总'):
        report = await agent(
            '汇总以下审查结论为一份报告：\n' + '\n'.join(r or '(失败)' for r in reviews))
    return report
```

断点续跑：同 run_id resume 时，prompt+参数相同的 agent() 调用直接回放 journal 结果不重跑；改脚本会导致整体重跑（hash 失配截断）。
