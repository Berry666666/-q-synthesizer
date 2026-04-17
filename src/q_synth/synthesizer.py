import json
import random
import uuid
from urllib import error, request
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SubGoal:
    id: str
    title: str
    objective: str
    dependencies: List[str]
    required_tools: List[str]
    estimated_hours: int
    estimated_budget_k: int
    layer: int


@dataclass
class DynamicEvent:
    id: str
    trigger_phase: int
    description: str
    impact_level: str
    required_action: str


@dataclass
class TaskInstance:
    task_id: str
    profile: str
    domain: str
    org: str
    focus: str
    background: str
    goal: str
    team_size: int
    timeline_weeks: int
    budget_k: int
    parallel_limit: int
    subgoals: List[SubGoal]
    hard_constraints: List[str]
    soft_preferences: List[str]
    dynamic_events: List[DynamicEvent]
    deliverables: List[str]
    noise_context: List[str]
    available_tools: List[str]


class QSynthesizer:
    """Generate high-quality long-horizon task questions (Q) with solver-backed checks."""

    def __init__(self, config: Dict[str, Any], seed: int = 42) -> None:
        self.config = config
        self.rnd = random.Random(seed)
        defaults = self.config.get("defaults", {}) if isinstance(self.config, dict) else {}
        raw_pool = defaults.get("fallback_action_pool", []) if isinstance(defaults, dict) else []
        self.action_pool = [str(x).strip() for x in raw_pool if str(x).strip()]
        self.impact_pool = ["low", "medium", "high"]

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        t = text.strip()
        if t.startswith("```") and t.endswith("```"):
            lines = t.splitlines()
            if len(lines) >= 2:
                t = "\n".join(lines[1:-1]).strip()
        return t

    def _extract_json_obj_from_content(self, content: str) -> Dict[str, Any]:
        raw = self._strip_code_fence(content)

        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = raw[start : end + 1]
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

        raise RuntimeError("Cannot parse JSON object from LLM content")

    def _llm_messages(self, inst: TaskInstance, draft_q: str, mode: str) -> List[Dict[str, str]]:
        subgoals = [
            {
                "id": sg.id,
                "title": sg.title,
                "dependencies": sg.dependencies,
                "tools": sg.required_tools,
            }
            for sg in inst.subgoals[:12]
        ]
        events = [
            {
                "id": ev.id,
                "trigger_phase": ev.trigger_phase,
                "description": ev.description,
                "impact": ev.impact_level,
            }
            for ev in inst.dynamic_events[:6]
        ]

        payload = {
            "org": inst.org,
            "domain": inst.domain,
            "focus": inst.focus,
            "goal": inst.goal,
            "team_size": inst.team_size,
            "timeline_weeks": inst.timeline_weeks,
            "budget_k": inst.budget_k,
            "available_tools": inst.available_tools,
            "hard_constraints": inst.hard_constraints,
            "soft_preferences": inst.soft_preferences,
            "deliverables": inst.deliverables,
            "noise_context": inst.noise_context,
            "subgoals": subgoals,
            "dynamic_events": events,
        }

        system_prompt = (
            "你是高水平任务设计专家。请根据给定结构化上下文，产出一个高复杂度、长程、多约束、"
            "可执行且更具语言多样性的中文任务问题Q。"
            "必须覆盖：背景、核心目标、资源与环境、复杂子目标、硬约束、软偏好、动态事件、输出要求。"
            "输出必须是JSON对象，且只有一个字段：Q。"
        )

        if mode == "hybrid":
            user_prompt = (
                "请基于下面上下文，重写并显著丰富已有草稿Q，保留事实约束但提升表述多样性与泛化性。"
                "不要删掉关键约束与事件。\n\n"
                f"结构化上下文:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
                f"草稿Q:\n{draft_q}\n\n"
                "请仅输出JSON对象：{\"Q\":\"...\"}"
            )
        else:
            user_prompt = (
                "请根据下面结构化上下文直接生成全新Q（不要复述固定模板句式），"
                "要求长程、复杂、具备真实业务感与多步骤依赖。\n\n"
                f"结构化上下文:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
                "请仅输出JSON对象：{\"Q\":\"...\"}"
            )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _call_openai_compatible(self, llm_cfg: Dict[str, Any], messages: List[Dict[str, str]]) -> str:
        base_url = str(llm_cfg.get("base_url", "")).rstrip("/")
        if not base_url:
            raise ValueError("llm_cfg.base_url is required")

        model = str(llm_cfg.get("model", "")).strip()
        if not model:
            raise ValueError("llm_cfg.model is required")

        api_key = str(llm_cfg.get("api_key", "")).strip()
        if not api_key:
            raise ValueError("llm_cfg.api_key is required")

        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(llm_cfg.get("temperature", 0.9)),
            "top_p": float(llm_cfg.get("top_p", 0.95)),
            "max_tokens": int(llm_cfg.get("max_tokens", 2200)),
        }

        req = request.Request(
            url=f"{base_url}/chat/completions",
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        )

        timeout_sec = float(llm_cfg.get("timeout_sec", 60))
        try:
            with request.urlopen(req, timeout=timeout_sec) as resp:
                body = resp.read().decode("utf-8")
        except error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM HTTPError {e.code}: {detail[:500]}") from e
        except error.URLError as e:
            raise RuntimeError(f"LLM URLError: {e}") from e

        data = json.loads(body)
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"LLM response missing choices: {str(data)[:500]}")

        msg = choices[0].get("message", {})
        content = msg.get("content", "")
        if isinstance(content, list):
            # Some compatible APIs may return structured segments.
            content = "".join(str(x.get("text", "")) for x in content if isinstance(x, dict))

        content = str(content).strip()
        if not content:
            raise RuntimeError("LLM returned empty content")
        return content

    def _extract_q_from_llm_content(self, content: str) -> str:
        raw = self._strip_code_fence(content)

        try:
            obj = self._extract_json_obj_from_content(content)
            q_text = str(obj.get("Q", "")).strip()
            if q_text:
                return q_text
        except Exception:
            pass

        # Last resort: accept plain text if reasonably long.
        if len(raw) >= 120:
            return raw

        raise RuntimeError("Cannot parse usable Q from LLM content")

    @staticmethod
    def _normalize_subgoal_text(text: Any, max_len: int) -> str:
        val = " ".join(str(text).replace("\n", " ").split())
        if len(val) > max_len:
            val = val[:max_len].rstrip()
        return val

    def _subgoal_llm_messages(
        self,
        mode: str,
        domain: str,
        org: str,
        focus: str,
        goal: str,
        available_tools: List[str],
        subgoals: List[SubGoal],
    ) -> List[Dict[str, str]]:
        subgoal_payload = [
            {
                "id": sg.id,
                "layer": sg.layer,
                "dependencies": sg.dependencies,
                "required_tools": sg.required_tools,
                "draft_title": sg.title,
                "draft_objective": sg.objective,
            }
            for sg in subgoals
        ]

        payload = {
            "domain": domain,
            "org": org,
            "focus": focus,
            "goal": goal,
            "available_tools": available_tools,
            "subgoals": subgoal_payload,
        }

        system_prompt = (
            "你是资深任务分解专家。请输出JSON对象，且只包含一个字段subgoals。"
            "subgoals必须是数组；每项必须包含id,title,objective。"
            "禁止新增或删除id；禁止输出任何解释性文字和Markdown。"
            "title应简洁具体，objective应可执行并可验收。"
        )

        if mode == "hybrid":
            user_prompt = (
                "请基于下面结构化上下文，重写每个子目标的title/objective，保留id、层级、依赖和工具约束。"
                "强调行业语义、动作多样性与可执行性，不要使用同质化模板句式。\n\n"
                f"结构化上下文:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
                "请仅输出JSON对象："
                "{\"subgoals\":[{\"id\":\"SG-01\",\"title\":\"...\",\"objective\":\"...\"}]}"
            )
        else:
            user_prompt = (
                "请根据下面结构化上下文，直接为每个id生成新的title/objective。"
                "必须覆盖全部id；不要复述固定动作词，不要输出模板化套话。\n\n"
                f"结构化上下文:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
                "请仅输出JSON对象："
                "{\"subgoals\":[{\"id\":\"SG-01\",\"title\":\"...\",\"objective\":\"...\"}]}"
            )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _extract_subgoal_updates_from_llm_content(self, content: str, expected_ids: List[str]) -> Dict[str, Dict[str, str]]:
        obj = self._extract_json_obj_from_content(content)
        raw_items = obj.get("subgoals", [])
        if not isinstance(raw_items, list):
            raise RuntimeError("LLM subgoal payload missing 'subgoals' list")

        expected = set(expected_ids)
        updates: Dict[str, Dict[str, str]] = {}

        for item in raw_items:
            if not isinstance(item, dict):
                continue

            sid = str(item.get("id", "")).strip()
            if sid not in expected or sid in updates:
                continue

            title = self._normalize_subgoal_text(item.get("title", ""), max_len=60)
            objective = self._normalize_subgoal_text(item.get("objective", ""), max_len=180)
            if title and objective:
                updates[sid] = {"title": title, "objective": objective}

        if not updates:
            raise RuntimeError("LLM returned no usable subgoal updates")

        return updates

    def _render_subgoals_with_llm(
        self,
        subgoals: List[SubGoal],
        domain: str,
        org: str,
        focus: str,
        goal: str,
        available_tools: List[str],
        mode: str,
        llm_cfg: Dict[str, Any],
    ) -> List[SubGoal]:
        messages = self._subgoal_llm_messages(
            mode=mode,
            domain=domain,
            org=org,
            focus=focus,
            goal=goal,
            available_tools=available_tools,
            subgoals=subgoals,
        )

        call_cfg = dict(llm_cfg)
        if "subgoal_temperature" in llm_cfg:
            call_cfg["temperature"] = float(llm_cfg.get("subgoal_temperature"))
        if "subgoal_top_p" in llm_cfg:
            call_cfg["top_p"] = float(llm_cfg.get("subgoal_top_p"))
        if "subgoal_max_tokens" in llm_cfg:
            call_cfg["max_tokens"] = int(llm_cfg.get("subgoal_max_tokens"))

        max_retries = int(llm_cfg.get("subgoal_max_retries", llm_cfg.get("max_retries", 2)))

        last_err: Optional[Exception] = None
        for _ in range(max_retries + 1):
            try:
                content = self._call_openai_compatible(call_cfg, messages)
                updates = self._extract_subgoal_updates_from_llm_content(content, [sg.id for sg in subgoals])

                rewritten: List[SubGoal] = []
                for sg in subgoals:
                    patch = updates.get(sg.id)
                    if patch is None:
                        rewritten.append(sg)
                        continue
                    rewritten.append(
                        replace(
                            sg,
                            title=patch["title"],
                            objective=patch["objective"],
                        )
                    )

                return rewritten
            except Exception as e:  # noqa: PERF203
                last_err = e

        raise RuntimeError(f"LLM subgoal generation failed after retries: {last_err}")

    def _render_question_with_llm(self, inst: TaskInstance, draft_q: str, mode: str, llm_cfg: Dict[str, Any]) -> str:
        max_retries = int(llm_cfg.get("max_retries", 2))
        messages = self._llm_messages(inst, draft_q=draft_q, mode=mode)

        last_err: Optional[Exception] = None
        for _ in range(max_retries + 1):
            try:
                content = self._call_openai_compatible(llm_cfg, messages)
                q_text = self._extract_q_from_llm_content(content)
                if len(q_text) < 120:
                    raise RuntimeError("LLM Q text too short")
                return q_text
            except Exception as e:  # noqa: PERF203
                last_err = e

        raise RuntimeError(f"LLM generation failed after retries: {last_err}")

    def _randint(self, left_right: List[int]) -> int:
        left, right = left_right
        return self.rnd.randint(left, right)

    def _choose_many(self, items: List[str], count: int) -> List[str]:
        if not items:
            return []
        count = max(0, min(count, len(items)))
        return self.rnd.sample(items, count)

    def _sample_domain(self) -> Dict[str, Any]:
        domains = self.config["domains"]
        return self.rnd.choice(domains)

    def _sample_profile(self, profile: str) -> Dict[str, Any]:
        profiles = self.config["complexity_profiles"]
        if profile not in profiles:
            raise ValueError(f"Unknown profile: {profile}")
        return profiles[profile]

    def _build_layers(self, n_subgoals: int, depth_range: List[int]) -> List[int]:
        depth = min(n_subgoals, self._randint(depth_range))
        layer_counts = [1] * depth
        remain = n_subgoals - depth
        for _ in range(remain):
            layer_counts[self.rnd.randrange(depth)] += 1

        layers = []
        for layer_idx, cnt in enumerate(layer_counts):
            layers.extend([layer_idx] * cnt)
        self.rnd.shuffle(layers)
        return layers

    def _build_subgoals(
        self,
        profile_cfg: Dict[str, Any],
        domain_cfg: Dict[str, Any],
        org: str,
        focus: str,
        goal: str,
        available_tools: List[str],
        llm_cfg: Optional[Dict[str, Any]] = None,
        subgoal_generation_mode: str = "rule",
    ) -> Tuple[List[SubGoal], str]:
        mode = str(subgoal_generation_mode or "rule").lower()
        if mode not in {"rule", "llm", "hybrid"}:
            raise ValueError(f"Unknown subgoal_generation_mode: {mode}")

        n_subgoals = self._randint(profile_cfg["subgoals"])
        layers = self._build_layers(n_subgoals, profile_cfg["depth"])

        max_dep = self._randint(self.config["defaults"]["max_dependencies_per_subgoal"])
        req_tool_range = self.config["defaults"]["tool_requirements"]

        subgoals: List[SubGoal] = []
        by_layer: Dict[int, List[str]] = {}

        # In llm/hybrid mode, draft text intentionally avoids fixed action templates
        # to reduce lexical anchoring before model rewriting.
        use_action_templates = mode == "rule" and bool(self.action_pool)

        for i in range(n_subgoals):
            sid = f"SG-{i + 1:02d}"
            if use_action_templates:
                action = self.rnd.choice(self.action_pool)
                title = f"{action}（{focus}）"
            else:
                title = f"围绕{focus}推进关键子目标{i + 1}"
            objective = f"在受限资源下完成与“{focus}”相关的第{i + 1}个关键动作，形成可验收结果。"
            layer = layers[i]

            prev_candidates: List[str] = []
            for ly in range(layer):
                prev_candidates.extend(by_layer.get(ly, []))

            if prev_candidates:
                dep_count = min(len(prev_candidates), self.rnd.randint(1, max_dep))
                dependencies = self.rnd.sample(prev_candidates, dep_count)
            else:
                dependencies = []

            tool_count = min(len(available_tools), self._randint(req_tool_range))
            required_tools = self.rnd.sample(available_tools, tool_count)

            estimated_hours = self.rnd.randint(8, 40)
            estimated_budget_k = self.rnd.randint(3, 18)

            node = SubGoal(
                id=sid,
                title=title,
                objective=objective,
                dependencies=dependencies,
                required_tools=required_tools,
                estimated_hours=estimated_hours,
                estimated_budget_k=estimated_budget_k,
                layer=layer,
            )
            subgoals.append(node)
            by_layer.setdefault(layer, []).append(sid)

        source = "rule-template"
        if mode != "rule":
            if not llm_cfg:
                raise ValueError("llm_cfg is required when subgoal_generation_mode is llm or hybrid")
            try:
                subgoals = self._render_subgoals_with_llm(
                    subgoals=subgoals,
                    domain=str(domain_cfg.get("name", "unknown")),
                    org=org,
                    focus=focus,
                    goal=goal,
                    available_tools=available_tools,
                    mode=mode,
                    llm_cfg=llm_cfg,
                )
                source = "llm-generate" if mode == "llm" else "llm-hybrid"
            except Exception as e:
                fallback = bool(llm_cfg.get("subgoal_fallback_to_rule", llm_cfg.get("fallback_to_rule", False)))
                if fallback:
                    source = "rule-fallback"
                else:
                    raise RuntimeError(f"LLM subgoal generation failed: {e}") from e

        return subgoals, source

    def _build_dynamic_events(self, profile_cfg: Dict[str, Any], domain_cfg: Dict[str, Any], n_phases: int) -> List[DynamicEvent]:
        event_count = self._randint(profile_cfg["events"])
        sampled = self._choose_many(domain_cfg["dynamic_event_pool"], event_count)
        events: List[DynamicEvent] = []

        for i, desc in enumerate(sampled):
            trigger = self.rnd.randint(1, max(1, n_phases))
            impact = self.rnd.choice(self.impact_pool)
            action = "触发应急分支：重排优先级、收缩范围、并保留合规与回滚能力。"
            events.append(
                DynamicEvent(
                    id=f"EV-{i + 1:02d}",
                    trigger_phase=trigger,
                    description=desc,
                    impact_level=impact,
                    required_action=action,
                )
            )
        return events

    def _build_instance(
        self,
        profile: str,
        llm_cfg: Optional[Dict[str, Any]] = None,
        subgoal_generation_mode: str = "rule",
    ) -> Tuple[TaskInstance, str]:
        domain_cfg = self._sample_domain()
        profile_cfg = self._sample_profile(profile)

        org = self.rnd.choice(domain_cfg["org_pool"])
        focus = self.rnd.choice(domain_cfg["focus_pool"])
        background = self.rnd.choice(domain_cfg["background_templates"]).format(org=org, focus=focus)
        goal = self.rnd.choice(domain_cfg["goal_templates"])

        team_size = self.rnd.randint(4, 12)
        timeline_weeks = self.rnd.randint(4, 14)
        budget_k = self.rnd.randint(120, 800)
        parallel_limit = self.rnd.randint(profile_cfg["branch_factor"][0], profile_cfg["branch_factor"][1] + 2)

        avail_tool_count = self.rnd.randint(6, min(10, len(domain_cfg["tool_pool"])))
        available_tools = self._choose_many(domain_cfg["tool_pool"], avail_tool_count)

        subgoals, subgoal_source = self._build_subgoals(
            profile_cfg,
            domain_cfg,
            org,
            focus,
            goal,
            available_tools,
            llm_cfg=llm_cfg,
            subgoal_generation_mode=subgoal_generation_mode,
        )
        max_layer = max(sg.layer for sg in subgoals)
        dynamic_events = self._build_dynamic_events(profile_cfg, domain_cfg, max_layer + 1)

        target_hard_count = self._randint(profile_cfg["constraints"])
        hard_constraints = self._choose_many(
            domain_cfg["hard_constraint_pool"],
            min(len(domain_cfg["hard_constraint_pool"]), target_hard_count),
        )

        extra_templates = [
            f"至少{self.rnd.randint(2, 5)}个关键节点需要双人复核。",
            f"跨团队阻塞问题必须在{self.rnd.randint(4, 24)}小时内升级处理。",
            f"每周必须完成不少于{self.rnd.randint(2, 6)}次阶段性质量抽检。",
            f"所有高风险任务必须预留至少{self.rnd.randint(10, 30)}%缓冲时间。",
            f"关键路径任务不得连续延期超过{self.rnd.randint(1, 2)}次。",
            f"每个阶段至少保留{self.rnd.randint(1, 3)}个可替代执行方案。",
            f"对外依赖接口故障恢复时间不得超过{self.rnd.randint(30, 180)}分钟。",
            f"关键里程碑评审必须覆盖不少于{self.rnd.randint(3, 6)}类风险维度。",
            f"资源调度冲突在同一工作日内必须闭环处理。",
            f"高优先级告警确认时延不得超过{self.rnd.randint(10, 60)}分钟。",
        ]

        while len(hard_constraints) < target_hard_count and extra_templates:
            c = self.rnd.choice(extra_templates)
            extra_templates.remove(c)
            if c not in hard_constraints:
                hard_constraints.append(c)

        # If still below target due template depletion, add deterministic filler constraints.
        filler_idx = 1
        while len(hard_constraints) < target_hard_count:
            hard_constraints.append(f"附加硬约束-{filler_idx}: 不得弱化主目标验收标准。")
            filler_idx += 1

        # Inject numeric constraints for solver validation.
        hard_constraints.extend(
            [
                f"总预算上限为{budget_k}k，超出必须给出明确削减路径。",
                f"项目总工期不超过{timeline_weeks}周。",
                f"每周最多并行{parallel_limit}个高风险任务。",
                f"核心里程碑最迟在第{max(2, timeline_weeks - 2)}周完成。",
            ]
        )

        soft_count = self.rnd.randint(2, min(4, len(domain_cfg["soft_preference_pool"])))
        soft_preferences = self._choose_many(domain_cfg["soft_preference_pool"], soft_count)

        deliver_count = self.rnd.randint(4, min(6, len(domain_cfg["deliverable_requirements"])))
        deliverables = self._choose_many(domain_cfg["deliverable_requirements"], deliver_count)

        noise_count = self._randint(self.config["defaults"]["noise_sentences"])
        noise_context = self._choose_many(domain_cfg["noise_pool"], noise_count)

        task_id = f"Q-{profile}-{uuid.uuid4().hex[:10]}"
        return TaskInstance(
            task_id=task_id,
            profile=profile,
            domain=domain_cfg["name"],
            org=org,
            focus=focus,
            background=background,
            goal=goal,
            team_size=team_size,
            timeline_weeks=timeline_weeks,
            budget_k=budget_k,
            parallel_limit=parallel_limit,
            subgoals=subgoals,
            hard_constraints=hard_constraints,
            soft_preferences=soft_preferences,
            dynamic_events=dynamic_events,
            deliverables=deliverables,
            noise_context=noise_context,
            available_tools=available_tools,
        ), subgoal_source

    def _dependency_depth(self, subgoals: List[SubGoal]) -> int:
        node_map = {sg.id: sg for sg in subgoals}
        memo: Dict[str, int] = {}

        def depth(node_id: str) -> int:
            if node_id in memo:
                return memo[node_id]
            node = node_map[node_id]
            if not node.dependencies:
                memo[node_id] = 1
                return 1
            val = 1 + max(depth(dep) for dep in node.dependencies)
            memo[node_id] = val
            return val

        return max(depth(sg.id) for sg in subgoals)

    def _is_feasible(self, inst: TaskInstance) -> Tuple[bool, Dict[str, Any]]:
        total_hours = sum(sg.estimated_hours for sg in inst.subgoals)
        total_budget = sum(sg.estimated_budget_k for sg in inst.subgoals)

        # Assume effective throughput: each person contributes around 24 focused hours/week.
        capacity_hours = inst.team_size * inst.timeline_weeks * 24

        feasible_hours = total_hours <= capacity_hours * 1.1
        feasible_budget = total_budget <= inst.budget_k * 0.95

        details = {
            "total_hours": total_hours,
            "total_budget_k": total_budget,
            "capacity_hours": capacity_hours,
            "budget_limit_k": inst.budget_k,
            "feasible_hours": feasible_hours,
            "feasible_budget": feasible_budget,
        }
        return (feasible_hours and feasible_budget), details

    def _quality_score(self, inst: TaskInstance, feasible: bool, feasible_detail: Dict[str, Any]) -> Dict[str, Any]:
        depth = self._dependency_depth(inst.subgoals)
        n_subgoals = len(inst.subgoals)
        n_constraints = len(inst.hard_constraints)
        n_events = len(inst.dynamic_events)
        n_tools = len(inst.available_tools)

        f_subgoals = min(1.0, n_subgoals / 16.0)
        f_depth = min(1.0, depth / 8.0)
        f_constraints = min(1.0, n_constraints / 18.0)
        f_events = min(1.0, n_events / 4.0)
        f_tools = min(1.0, n_tools / 10.0)

        base = (
            f_subgoals * 0.24
            + f_depth * 0.24
            + f_constraints * 0.20
            + f_events * 0.16
            + f_tools * 0.16
        )

        penalty = 0.0
        if not feasible:
            penalty += 0.20

        score = max(0.0, min(1.0, base - penalty))

        return {
            "score": round(score, 4),
            "components": {
                "subgoal_complexity": round(f_subgoals, 4),
                "dependency_depth": round(f_depth, 4),
                "constraint_density": round(f_constraints, 4),
                "dynamic_events": round(f_events, 4),
                "tool_diversity": round(f_tools, 4),
            },
            "feasibility": feasible_detail,
            "graph_depth": depth,
        }

    def _render_question(self, inst: TaskInstance) -> str:
        subgoal_preview = [f"{sg.id}. {sg.title}" for sg in inst.subgoals[: min(10, len(inst.subgoals))]]
        if len(inst.subgoals) > 10:
            subgoal_preview.append(f"... 其余{len(inst.subgoals) - 10}个子任务需要你自行统筹与编排。")

        event_lines = [
            f"{ev.id}（在第{ev.trigger_phase}阶段触发，影响等级:{ev.impact_level}）: {ev.description}"
            for ev in inst.dynamic_events
        ]
        if not event_lines:
            event_lines = ["本任务默认无显式外部突发事件，但你仍需保留应急冗余。"]

        noise_lines = [f"- {n}" for n in inst.noise_context] if inst.noise_context else ["- 无额外噪声信息"]

        lines = [
            f"你是{inst.org}的总负责人。",
            "",
            "【背景】",
            inst.background,
            "",
            "【核心目标】",
            inst.goal,
            "",
            "【当前资源与环境】",
            f"- 团队规模: {inst.team_size}人",
            f"- 时间窗口: {inst.timeline_weeks}周",
            f"- 可用预算上限: {inst.budget_k}k",
            f"- 可调用工具: {', '.join(inst.available_tools)}",
            "",
            "【任务要求（复杂子目标）】",
            *[f"- {s}" for s in subgoal_preview],
            "",
            "【硬约束（必须满足）】",
            *[f"- {c}" for c in inst.hard_constraints],
            "",
            "【软偏好（尽量满足）】",
            *[f"- {p}" for p in inst.soft_preferences],
            "",
            "【动态变化事件】",
            *[f"- {e}" for e in event_lines],
            "",
            "【输出要求】",
            *[f"- {d}" for d in inst.deliverables],
            "- 需要给出分阶段长程规划、依赖关系、关键里程碑、资源分配、风险处置与回滚方案。",
            "- 请说明在突发事件触发后如何重排优先级，且保证目标不失控。",
            "",
            "【补充上下文（可能包含噪声信息，请自行判断相关性）】",
            *noise_lines,
        ]
        return "\n".join(lines)

    def _render_oracle_plan(self, inst: TaskInstance) -> str:
        phase_map: Dict[int, List[SubGoal]] = {}
        for sg in inst.subgoals:
            phase_map.setdefault(sg.layer, []).append(sg)

        lines = [
            "一、总体策略",
            "先统一口径与关键依赖，再分阶段推进执行；每个阶段设置可观测指标、风险门禁与回滚条件。",
            "",
            "二、分阶段执行计划",
        ]

        for phase in sorted(phase_map.keys()):
            nodes = phase_map[phase]
            lines.append(f"阶段{phase + 1}（目标: 收敛关键不确定性并推进里程碑）")
            for sg in nodes:
                dep = "无" if not sg.dependencies else ", ".join(sg.dependencies)
                lines.append(
                    f"- {sg.id} {sg.title}: 依赖[{dep}]；动作= {sg.objective}；工具= {', '.join(sg.required_tools)}；预估工时={sg.estimated_hours}h；预算={sg.estimated_budget_k}k"
                )
            lines.append("")

        if inst.dynamic_events:
            event_lines = [
                f"- {ev.id}: 当第{ev.trigger_phase}阶段触发“{ev.description}”时，执行动作: {ev.required_action}"
                for ev in inst.dynamic_events
            ]
        else:
            event_lines = ["- 无显式事件，默认按高风险场景保留10%容量与预算弹性。"]

        lines.extend(
            [
                "三、动态事件应对",
                *event_lines,
                "",
                "四、治理与验收",
                "- 每阶段输出里程碑结果、风险状态、预算消耗与下一阶段准入判断。",
                "- 若出现关键约束冲突，优先保护合规与可回滚能力，再做范围裁剪。",
            ]
        )

        return "\n".join(lines)

    def generate_one(
        self,
        profile: str,
        q_only: bool = False,
        q_generation_mode: str = "rule",
        subgoal_generation_mode: str = "rule",
        llm_cfg: Optional[Dict[str, Any]] = None,
        sample_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        mode = str(q_generation_mode or "rule").lower()
        if mode not in {"rule", "llm", "hybrid"}:
            raise ValueError(f"Unknown q_generation_mode: {mode}")

        subgoal_mode = str(subgoal_generation_mode or "rule").lower()
        if subgoal_mode not in {"rule", "llm", "hybrid"}:
            raise ValueError(f"Unknown subgoal_generation_mode: {subgoal_mode}")

        if sample_seed is None:
            inst, subgoal_source = self._build_instance(
                profile,
                llm_cfg=llm_cfg,
                subgoal_generation_mode=subgoal_mode,
            )
        else:
            # Use an isolated RNG stream per sample to improve scenario diversity while keeping reproducibility.
            prev_rnd = self.rnd
            self.rnd = random.Random(int(sample_seed))
            try:
                inst, subgoal_source = self._build_instance(
                    profile,
                    llm_cfg=llm_cfg,
                    subgoal_generation_mode=subgoal_mode,
                )
            finally:
                self.rnd = prev_rnd

        feasible, feasible_detail = self._is_feasible(inst)

        template_q = self._render_question(inst)
        q_source = "rule-template"
        if mode == "rule":
            q_text = template_q
        else:
            if not llm_cfg:
                raise ValueError("llm_cfg is required when q_generation_mode is llm or hybrid")
            try:
                q_text = self._render_question_with_llm(inst, draft_q=template_q, mode=mode, llm_cfg=llm_cfg)
                q_source = "llm-generate" if mode == "llm" else "llm-hybrid"
            except Exception as e:
                if bool(llm_cfg.get("fallback_to_rule", False)):
                    q_text = template_q
                    q_source = "rule-fallback"
                else:
                    raise RuntimeError(f"LLM question generation failed: {e}") from e

        quality = self._quality_score(inst, feasible, feasible_detail)

        record: Dict[str, Any] = {
            "id": inst.task_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "profile": profile,
            "domain": inst.domain,
            "Q": q_text,
            "quality": quality,
            "context": {
                "background": inst.background,
                "goal": inst.goal,
                "hard_constraints": inst.hard_constraints,
                "soft_preferences": inst.soft_preferences,
                "dynamic_events": [asdict(ev) for ev in inst.dynamic_events],
                "deliverables": inst.deliverables,
                "noise_context": inst.noise_context,
                "available_tools": inst.available_tools,
            },
            "meta": {
                "org": inst.org,
                "focus": inst.focus,
                "team_size": inst.team_size,
                "timeline_weeks": inst.timeline_weeks,
                "budget_k": inst.budget_k,
                "parallel_limit": inst.parallel_limit,
                "subgoal_count": len(inst.subgoals),
                "constraint_count": len(inst.hard_constraints),
                "event_count": len(inst.dynamic_events),
                "tool_count": len(inst.available_tools),
                "q_generation_mode": mode,
                "q_source": q_source,
                "subgoal_generation_mode": subgoal_mode,
                "subgoal_source": subgoal_source,
            },
            "task_graph": {
                "nodes": [asdict(sg) for sg in inst.subgoals],
                "edges": [
                    {"from": dep, "to": sg.id}
                    for sg in inst.subgoals
                    for dep in sg.dependencies
                ],
            },
        }

        if llm_cfg and (mode in {"llm", "hybrid"} or subgoal_mode in {"llm", "hybrid"}):
            record["meta"]["llm_model"] = str(llm_cfg.get("model", ""))

        if sample_seed is not None:
            record["meta"]["sample_seed"] = int(sample_seed)

        if not q_only:
            record["A"] = self._render_oracle_plan(inst)

        return record

    def get_quality_threshold(self, profile: str, override_threshold: Optional[float] = None) -> float:
        if override_threshold is not None:
            return override_threshold
        return float(self.config["defaults"]["quality_threshold"][profile])

    def to_jsonl_line(self, record: Dict[str, Any]) -> str:
        return json.dumps(record, ensure_ascii=False)
