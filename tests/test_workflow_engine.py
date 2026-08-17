"""R28 W1：workflow 引擎核心测试（mock agent_runner，不碰 IO/LLM）。

pytest-asyncio asyncio_mode=auto，async 测试直接写。
"""

MAIN = "async def main():\n    return 42\n"


class TestValidateScript:
    def test_valid(self):
        from agent.workflow_engine import validate_script
        assert validate_script(MAIN) is None

    def test_missing_main(self):
        from agent.workflow_engine import validate_script
        assert validate_script("x = 1\n") is not None

    def test_forbidden_import(self):
        from agent.workflow_engine import validate_script
        assert "import" in validate_script("import os\n" + MAIN)

    def test_forbidden_exec_open(self):
        from agent.workflow_engine import validate_script
        assert validate_script("async def main():\n    exec('1')\n") is not None
        assert validate_script("async def main():\n    open('/etc/passwd')\n") is not None

    def test_forbidden_dunder(self):
        from agent.workflow_engine import validate_script
        assert validate_script(
            "async def main():\n    return object().__class__\n") is not None

    def test_syntax_error(self):
        from agent.workflow_engine import validate_script
        assert validate_script("def broken(:\n") is not None


class TestRunWorkflow:
    async def _run(self, source, runner=None, **kw):
        from agent.workflow_engine import run_workflow
        if runner is None:
            async def runner(prompt):
                return f"echo:{prompt}"
        return await run_workflow(source, agent_runner=runner, **kw)

    async def test_return_value(self):
        out = await self._run(MAIN)
        assert out["ok"] is True and out["return"] == 42

    async def test_agent_primitive(self):
        out = await self._run(
            "async def main():\n    r = await agent('hi')\n"
            "    return r\n")
        assert out["ok"] is True and out["return"] == "echo:hi"
        assert out["stats"]["calls"] == 1

    async def test_parallel(self):
        out = await self._run(
            "async def main():\n"
            "    rs = await parallel([lambda: agent('a'), lambda: agent('b')])\n"
            "    return rs\n")
        assert sorted(out["return"]) == ["echo:a", "echo:b"]
        assert out["stats"]["calls"] == 2

    async def test_parallel_null_on_error(self):
        out = await self._run(
            "async def _boom():\n    raise RuntimeError('x')\n"
            "async def main():\n"
            "    rs = await parallel([lambda: agent('a'), lambda: _boom()])\n"
            "    return rs\n")
        assert out["return"][0] == "echo:a" and out["return"][1] is None
        assert out["ok"] is True

    async def test_agent_dead_returns_none(self):
        async def runner(prompt):
            return None
        from agent.workflow_engine import run_workflow
        out = await run_workflow(
            "async def main():\n    return await agent('x')\n",
            agent_runner=runner,
        )
        assert out["ok"] is True and out["return"] is None
        assert out["stats"]["dead"] == 1

    async def test_pipeline(self):
        out = await self._run(
            "async def main():\n"
            "    rs = await pipeline(['a','b'], [lambda v: agent(v), lambda v: agent(v + '!')])\n"
            "    return rs\n")
        assert sorted(out["return"]) == ["echo:a!", "echo:b!"]

    async def test_phase_and_log_and_args(self):
        out = await self._run(
            "async def main():\n"
            "    with phase('扫描'):\n"
            "        log('开始')\n"
            "        return args['k']\n",
            args={"k": "v"})
        assert out["return"] == "v"

    async def test_budget_exceeded(self):
        from agent.workflow_engine import run_workflow

        async def runner(prompt):
            return "x" * 4000  # ~1000 估算 token
        out = await run_workflow(
            "async def main():\n    return await agent('a')\n",
            agent_runner=runner, budget_total=100,
        )
        assert out["ok"] is False and out["error_type"] == "budget_exceeded"

    async def test_concurrency_cap(self):
        import asyncio
        from agent.workflow_engine import run_workflow
        peak = 0
        cur = 0
        lock = asyncio.Lock()

        async def runner(prompt):
            nonlocal peak, cur
            async with lock:
                cur += 1
                peak = max(peak, cur)
            await asyncio.sleep(0.05)
            async with lock:
                cur -= 1
            return "r"

        factories = "["
        for i in range(6):
            factories += f"lambda: agent('t{i}'), "
        factories += "]"
        out = await run_workflow(
            f"async def main():\n    return await parallel({factories})\n",
            agent_runner=runner, max_concurrency=2,
        )
        assert out["ok"] is True and peak <= 2

    async def test_cancel_event(self):
        import asyncio
        from agent.workflow_engine import run_workflow
        ev = asyncio.Event()

        async def runner(prompt):
            ev.set()  # 第一次调用即请求取消
            return "r"
        out = await run_workflow(
            "async def main():\n    return await agent('x')\n",
            agent_runner=runner, cancel_event=ev,
        )
        assert out["ok"] is False and out["error_type"] == "cancelled"

    async def test_validator_retry_then_dead(self):
        """validator 两次不过 → dead → None。"""
        from agent.workflow_engine import run_workflow
        calls = []

        async def runner(prompt):
            calls.append(prompt)
            return "not-json"
        out = await run_workflow(
            "async def main():\n    return await agent('x')\n",
            agent_runner=runner,
            validator=lambda text: False,
        )
        assert out["return"] is None
        assert len(calls) == 2  # 重试一次
        assert out["stats"]["dead"] == 1

    async def test_script_error_reported(self):
        out = await self._run("async def main():\n    1/0\n")
        assert out["ok"] is False and out["error_type"] == "script_error"
