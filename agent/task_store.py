"""持久化任务仓库（Task System）——把"待办清单"存成文件，关掉程序也不丢。

在项目里的位置：被 tools/task_tools.py 包成工具给 LLM 用，底层只依赖文件系统。

和 TodoWrite（内存待办清单）的区别，打个比方：
  - TodoWrite 像一张便签纸：写在内存里，关会话就没了，也没有先后依赖
  - Task System 像一个项目看板：每个任务是一个 JSON 文件存在 ~/.codeAgent/.tasks/ 下，
    跨会话保留，任务之间还能声明"先做完 A 才能做 B"（DAG 依赖——就是一张
    "谁挡着谁"的关系网，不能有循环）

本文件提供的能力：
  - 依赖追踪（blocked_by 字段：这个任务被哪些任务挡着）
  - 状态机（pending 待办 → in_progress 做着 → completed 做完；另有 deleted/blocked/triage）
  - 所有权认领（owner：谁在负责这个任务）
  - 自动解锁检查（can_start：挡路的都做完了吗）
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


VALID_STATUSES = {"pending", "in_progress", "completed", "deleted", "blocked", "triage"}

# 同一类阻塞反复出现达到这个次数，就把任务升级到 triage（分诊/等人工处理）状态，
# 避免它被一遍遍捞出来重试、原地打转。
TRIAGE_THRESHOLD = 3
VALID_BLOCK_KINDS = {"dependency", "needs_input", "capability", "transient"}

# in_progress 任务心跳超过这个分钟数视为"认领人已死"（find_ready 顺手回收）。
# 传 0 可关闭。
STALE_HEARTBEAT_MINUTES = 30.0


def _now_iso() -> str:
    """返回当前时间的 UTC 标准格式字符串（用在任务的创建/更新时间戳上）。"""
    return datetime.now(timezone.utc).isoformat()


def _tasks_dir(codeagent_home=None) -> Path:
    """拿到存放任务 JSON 文件的目录（~/.codeAgent/.tasks/），没有就顺手建一个。

    参数：
        codeagent_home：CodeAgent 的数据根目录；不传就用默认的 ~/.codeAgent
            （测试里常传一个临时目录来隔离）。
    返回：目录的 Path 对象（已确保存在）。
    """
    if codeagent_home:
        home = Path(codeagent_home)
    else:
        try:
            from constants import get_codeagent_home
            home = get_codeagent_home()
        except Exception:
            home = Path.home() / ".codeAgent"
    d = home / ".tasks"
    d.mkdir(parents=True, exist_ok=True)
    return d


class TaskStore:
    """任务仓库本体：每个任务存成一个 JSON 文件，读写都走这里。

    一般不直接 new，用文件底部的 get_task_store() 拿缓存实例。
    """

    def __init__(self, codeagent_home=None):
        """记下任务目录（建目录的活儿由 _tasks_dir 干）。

        参数：
            codeagent_home：数据根目录，不传用默认 ~/.codeAgent。
        """
        self._dir = _tasks_dir(codeagent_home)

    def _task_file(self, task_id: str) -> Path:
        """由任务 id 拼出它的 JSON 文件路径。"""
        return self._dir / f"{task_id}.json"

    def _write(self, task_id: str, task: dict) -> None:
        """把任务 dict 落盘（先写临时文件再改名，写一半断电不会留半个文件）。

        参数：
            task_id：任务 id（决定文件名）。
            task：完整的任务 dict。
        """
        from agent.atomic_io import atomic_write_text
        f = self._task_file(task_id)
        atomic_write_text(f, json.dumps(task, ensure_ascii=False, indent=2))

    # ------------------------------------------------------------------
    # 增删改查（CRUD：Create / Read / Update / Delete）
    # ------------------------------------------------------------------

    def create(
        self,
        subject: str,
        description: str = "",
        blocked_by: Optional[List[str]] = None,
        owner: Optional[str] = None,
    ) -> dict:
        """新建一个任务并落盘，初始状态是 pending（待办）。

        参数：
            subject：任务标题（一句话说清要干什么）。
            description：任务详情，可空。
            blocked_by：这个任务被哪些任务挡着——列表里是那些任务的 id，
                它们全做完这个才能开工。
            owner：认领人名字，可空（还没人认领）。
        返回：新建好的完整任务 dict（含分配的 id）。
        """
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        task = {
            "id": task_id,
            "subject": subject,
            "description": description,
            "status": "pending",
            "owner": owner,
            "blocked_by": list(blocked_by or []),
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "last_heartbeat_at": None,
            "comments": [],
            "artifacts": [],
            "block_reason": None,
            "block_kind": None,
        }
        self._write(task_id, task)
        logger.info("创建任务 %s: %s", task_id, subject)
        return task

    def get(self, task_id: str) -> Optional[dict]:
        """按 id 读一个任务。

        参数：
            task_id：任务 id。
        返回：任务 dict；文件不存在或内容坏了（JSON 解析失败）返回 None。
        """
        f = self._task_file(task_id)
        if not f.exists():
            return None
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return None

    def update(self, task_id: str, **fields) -> Optional[dict]:
        """改任务的任意字段（id 不许改，改了会指向别的任务）。

        参数：
            task_id：要改的任务 id。
            **fields：要改的字段名=新值，如 status="completed"。
        返回：改完的任务 dict；任务不存在返回 None。
        """
        task = self.get(task_id)
        if task is None:
            return None
        for k, v in fields.items():
            if k != "id":
                task[k] = v
        task["updated_at"] = _now_iso()
        self._write(task_id, task)
        return task

    def set_status(self, task_id: str, status: str) -> Optional[dict]:
        """改任务状态（只接受 VALID_STATUSES 里列的那几种，防手滑写错）。

        参数：
            task_id：任务 id。
            status：新状态（pending/in_progress/completed/deleted/blocked/triage）。
        返回：更新后的任务 dict；状态名不合法或任务不存在返回 None。
        """
        if status not in VALID_STATUSES:
            return None
        return self.update(task_id, status=status)

    # ------------------------------------------------------------------
    # 阻塞记录与升级：任务卡住了就记一笔，反复卡住就升级等人工
    # ------------------------------------------------------------------

    def mark_blocked(
        self,
        task_id: str,
        *,
        kind: str = "transient",
        reason: str = "",
    ) -> dict:
        """记一笔"这个任务卡住了"，卡太多次会自动升级成 triage（分诊态）。

        同一类阻塞攒够 TRIAGE_THRESHOLD（3）次就升级到 triage，find_ready
        从此跳过它（只标 blocked 会被调度器反复捞出重试、原地打转），
        等人工或上级 agent 来处理。

        参数：
            task_id：任务 id。
            kind：阻塞类别，只能是 dependency（被依赖卡）/ needs_input（缺输入）/
                capability（干不了）/ transient（临时故障）之一，别的值一律按
                transient 处理。
            reason：这次为什么卡住，一句话描述。
        返回：{"status": "blocked" 或 "triage", "block_count": 该类阻塞累计次数, "kind": 类别}
        """
        if kind not in VALID_BLOCK_KINDS:
            kind = "transient"

        task = self.get(task_id)
        if task is None:
            raise KeyError(f"task not found: {task_id}")

        history = task.setdefault("block_history", [])
        counts = task.setdefault("block_count_by_kind", {})
        counts[kind] = counts.get(kind, 0) + 1
        history.append({
            "kind": kind,
            "reason": reason,
            "at": _now_iso(),
        })
        # 历史只留最近 20 条，超过就丢最老的——不然跑几个月的文件越写越大
        if len(history) > 20:
            del history[: len(history) - 20]

        # 攒够次数就升级到 triage
        new_status = "blocked"
        triage_reason: Optional[str] = None
        if counts[kind] >= TRIAGE_THRESHOLD:
            new_status = "triage"
            triage_reason = (
                f"task 阻塞超过 {TRIAGE_THRESHOLD} 次（kind={kind}），"
                f"已升级到 triage，等待人工或上级介入"
            )

        task["status"] = new_status
        task["block_kind"] = kind
        task["block_reason"] = triage_reason or reason
        task["updated_at"] = _now_iso()
        self._write(task_id, task)
        logger.info(
            "task %s 标记阻塞 kind=%s count=%d status=%s",
            task_id, kind, counts[kind], new_status,
        )
        return {
            "status": new_status,
            "block_count": counts[kind],
            "kind": kind,
        }

    def claim(self, task_id: str, owner: str) -> Optional[dict]:
        """认领任务：写上自己的名字，同时把状态切成 in_progress（开工了）。

        参数：
            task_id：任务 id。
            owner：认领人名字（哪个 agent/子代理在负责）。
        返回：更新后的任务 dict；任务不存在返回 None。
        """
        return self.update(task_id, owner=owner, status="in_progress")

    def complete(self, task_id: str) -> Optional[dict]:
        """把任务标成 completed（做完）。其他等着它的任务会因此解锁。

        参数：
            task_id：任务 id。
        返回：更新后的任务 dict；任务不存在返回 None。
        """
        return self.set_status(task_id, "completed")

    def delete(self, task_id: str) -> Optional[dict]:
        """软删除：只把状态标成 deleted，JSON 文件保留在盘上（想恢复还能恢复）。

        参数：
            task_id：任务 id。
        返回：更新后的任务 dict；任务不存在返回 None。
        """
        return self.set_status(task_id, "deleted")

    def list_all(self, status: Optional[str] = None) -> List[dict]:
        """列出所有任务，按创建时间从早到晚排。

        参数：
            status：只留这个状态的任务；不传就是全部。
        返回：任务 dict 的列表（坏了的 JSON 文件直接跳过不报错）。
        """
        tasks = []
        for f in sorted(self._dir.glob("*.json")):
            try:
                t = json.loads(f.read_text(encoding="utf-8"))
                if status is None or t.get("status") == status:
                    tasks.append(t)
            except Exception:
                continue
        tasks.sort(key=lambda t: t.get("created_at", ""))
        return tasks

    # ------------------------------------------------------------------
    # 依赖：判断任务能不能开工、哪些被挡着
    # ------------------------------------------------------------------

    def can_start(self, task_id: str) -> bool:
        """检查挡在这个任务前面的依赖是不是都完成了，决定它能否开工。

        被软删除或 JSON 文件缺失的依赖直接当作已满足（自动解链，
        不让任务永远卡在 blocked）。

        参数：
            task_id：任务 id。
        返回：True=依赖全部满足可以开工；False=还有依赖没完成（任务本身
            不存在也返回 False）。
        """
        task = self.get(task_id)
        if task is None:
            return False
        for dep_id in task.get("blocked_by", []):
            dep = self.get(dep_id)
            if dep is None or dep.get("status") == "deleted":
                logger.info(
                    "任务 %s 的依赖 %s 已删除/缺失，视为已满足（自动解链）",
                    task_id, dep_id,
                )
                continue
            if dep.get("status") != "completed":
                return False
        return True

    def find_ready(
        self, *, reclaim_stale_minutes: float = STALE_HEARTBEAT_MINUTES,
    ) -> List[dict]:
        """挑出"可以马上开工"的任务：状态是 pending 且挡路的都完成了。

        派活前先顺手回收心跳超时的僵尸 in_progress（认领人崩了没人
        重派就是调度死区——心跳只写不读的债在这里还上）。
        参数：reclaim_stale_minutes 见 reclaim_stale（0 = 不回收）。
        返回：可开工任务 dict 的列表。
        注意：triage 状态的任务不在此列（它们在等人工介入，不自动调度）。
        """
        try:
            self.reclaim_stale(reclaim_stale_minutes)
        except Exception as e:
            logger.debug("reclaim_stale 失败（fail-open，不影响派活）: %s", e)
        ready = []
        for t in self.list_all(status="pending"):
            if self.can_start(t["id"]):
                ready.append(t)
        return ready

    def find_blocked(self) -> List[dict]:
        """反向清单：还在排队等依赖的 pending 任务。

        参数：无。
        返回：被挡住的任务 dict 的列表。
        """
        blocked = []
        for t in self.list_all(status="pending"):
            if not self.can_start(t["id"]):
                blocked.append(t)
        return blocked

    # ------------------------------------------------------------------
    # 看板小功能：心跳（我还活着）/ 评论 / 产出物清单
    # ------------------------------------------------------------------

    def heartbeat(self, task_id: str) -> Optional[dict]:
        """报个心跳：把 last_heartbeat_at 刷成当前时间，证明"这活儿有人在做"。

        参数：
            task_id：任务 id。
        返回：更新后的任务 dict；任务不存在返回 None。
        """
        return self.update(task_id, last_heartbeat_at=_now_iso())

    def reclaim_stale(self, timeout_minutes: float = 30.0) -> List[dict]:
        """回收"心跳超时"的 in_progress 任务：打回 pending 等人重认领。

        为什么需要它：认领人（团队 worker/子代理）崩溃后任务永远卡在
        in_progress——find_ready 只捞 pending，没人重派就是调度死区。
        心跳字段此前只写不读，这里补上读的一端："距上次心跳超过
        timeout 分钟"就视为认领人已死，回炉重造。

        不误伤的设计：
        - 从没报过心跳的任务不回收（auto_heartbeat 之外的任务没有
          心跳语义，比如主代理亲自领的活）
        - timeout_minutes <= 0 视为关闭回收

        参数：
            timeout_minutes：判定"死了"的心跳超时分钟数。
        返回：被回收的任务 dict 列表（空列表 = 没有僵死任务）。
        """
        if timeout_minutes <= 0:
            return []
        now = datetime.now(timezone.utc)
        reclaimed: List[dict] = []
        for t in self.list_all(status="in_progress"):
            hb = t.get("last_heartbeat_at")
            if not hb:
                continue
            try:
                hb_dt = datetime.fromisoformat(str(hb))
            except (ValueError, TypeError):
                continue
            if hb_dt.tzinfo is None:
                hb_dt = hb_dt.replace(tzinfo=timezone.utc)
            if (now - hb_dt).total_seconds() < timeout_minutes * 60:
                continue
            updated = self.update(
                t["id"],
                status="pending",
                owner=None,
                stale_reclaimed_at=_now_iso(),
            )
            if updated is not None:
                logger.warning(
                    "task %s 心跳超时（>%s 分钟无心跳），已回收为 pending "
                    "等重认领（原 owner=%s）",
                    t["id"], timeout_minutes, t.get("owner"),
                )
                reclaimed.append(updated)
        return reclaimed

    def add_comment(
        self, task_id: str, *, author: str, content: str,
    ) -> Optional[dict]:
        """给任务追加一条评论（像留言板，只往上加、不删旧的）。

        参数：
            task_id：任务 id。
            author：谁说的（agent 名/用户）。
            content：留言内容。
        返回：更新后的任务 dict；任务不存在返回 None。
        """
        task = self.get(task_id)
        if task is None:
            return None
        task.setdefault("comments", []).append({
            "author": author,
            "content": content,
            "created_at": _now_iso(),
        })
        task["updated_at"] = _now_iso()
        self._write(task_id, task)
        return task

    def add_artifacts(self, task_id: str, paths: List[str]) -> Optional[dict]:
        """把这个任务产出的文件路径记到 artifacts 清单里（产出物索引，方便事后查看；重复的不记）。

        参数：
            task_id：任务 id。
            paths：产出文件的路径列表。
        返回：更新后的任务 dict；任务不存在返回 None。
        """
        task = self.get(task_id)
        if task is None:
            return None
        existing = task.setdefault("artifacts", [])
        for p in paths:
            if p not in existing:
                existing.append(p)
        task["updated_at"] = _now_iso()
        self._write(task_id, task)
        return task

    def remove_artifacts(self, task_id: str, paths: List[str]) -> Optional[dict]:
        """从产出物清单里划掉一些路径（记错了、文件挪走了）。

        参数：
            task_id：任务 id。
            paths：要移除的路径列表。
        返回：更新后的任务 dict；任务不存在返回 None。
        """
        task = self.get(task_id)
        if task is None:
            return None
        task["artifacts"] = [
            p for p in task.get("artifacts", []) if p not in paths
        ]
        task["updated_at"] = _now_iso()
        self._write(task_id, task)
        return task

    # ------------------------------------------------------------------
    # 依赖图安全：加依赖边之前先确认不会绕成一个圈
    # ------------------------------------------------------------------

    def has_path(self, start_id: str, target_id: str) -> bool:
        """沿着"被谁挡着"的关系一路问下去：start 最终（直接或间接）依赖 target 吗？

        打个比方：A 等 B，B 等 C，那从 A 出发沿"等"的箭头走能到 C。
        这是在依赖图（DAG）上做深度优先搜索（DFS——一条路走到黑再回头换路）。

        参数：
            start_id：出发的任务 id。
            target_id：要找的任务 id。
        返回：True=能走到（start 直接或间接依赖 target）；False=到不了。
        """
        visited = set()
        stack = [start_id]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            cur_task = self.get(cur)
            if cur_task is None:
                continue
            for dep in cur_task.get("blocked_by", []):
                if dep == target_id:
                    return True
                stack.append(dep)
        return False

    def add_dependency(
        self, child_id: str, parent_id: str,
        *, validate: bool = True,
    ) -> Optional[dict]:
        """加一条"child 要等 parent"的依赖边。

        默认先做成环检查（A 等 B、B 又等 A 就互相等死）：parent 如果已经
        直接或间接依赖 child，这条边会被拒。

        参数：
            child_id：被挡的任务 id（它要等别人）。
            parent_id：挡路的任务 id（先做完它）。
            validate：True（默认）做成环检查；False 跳过（信任调用方时用）。
        返回：更新后的 child 任务 dict；child 不存在返回 None。
        异常：自己等自己（self-link）任何时候都直接抛 ValueError；validate
            检出成环也抛 ValueError。
        """
        if parent_id == child_id:
            raise ValueError("self-link forbidden")
        child = self.get(child_id)
        if child is None:
            return None
        if validate:
            if self.has_path(parent_id, child_id):
                raise ValueError(
                    f"cycle detected: {parent_id} 已经依赖 {child_id}，"
                    f"再加 {child_id} → {parent_id} 边会成环"
                )
        blocked_by = child.setdefault("blocked_by", [])
        if parent_id not in blocked_by:
            blocked_by.append(parent_id)
        child["updated_at"] = _now_iso()
        self._write(child_id, child)
        return child


# 按 home 路径分格缓存的实例池：同一个 home 复用同一个实例，不同 home
# 各用各的、互不可见（不做全局唯一实例，否则传过 home 的调用会污染
# 后面不传 home 的调用拿到的目录）。
_task_stores: Dict[str, TaskStore] = {}


def get_task_store(codeagent_home=None) -> TaskStore:
    """拿 TaskStore 实例：同一个 home 永远给同一个（省得反复重建、也防串目录）。

    参数：
        codeagent_home：数据根目录，不传用默认 ~/.codeAgent（也是按这个做缓存键）。
    返回：该 home 对应的 TaskStore 实例（首次调用时创建并缓存）。
    """
    key = str(Path(codeagent_home).resolve()) if codeagent_home else ""
    store = _task_stores.get(key)
    if store is None:
        store = TaskStore(codeagent_home)
        _task_stores[key] = store
    return store
