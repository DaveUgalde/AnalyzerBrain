"""
AgentOrchestrator - Coordinates multiple agents for complex tasks.
"""

from typing import Dict, List, Optional, Any
import asyncio
import uuid
from datetime import datetime

from ..core.exceptions import AgentException, ValidationError
from .base_agent import BaseAgent, AgentInput, AgentOutput
from .agent_factory import AgentFactory


class AgentOrchestrator:
    """
    Coordinates multiple agents for complex tasks.
    """

    def __init__(self, agent_factory: AgentFactory):
        self.agent_factory = agent_factory
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.agent_performance: Dict[str, List[Dict[str, Any]]] = {}
        self.collaboration_history: List[Dict[str, Any]] = []
        self._is_running = False
        self._task_workers: List[asyncio.Task] = []

    # ------------------------------------------------------------------
    # Task assignment & coordination
    # ------------------------------------------------------------------

    async def assign_task(
        self,
        task: Dict[str, Any],
        preferred_agents: Optional[List[str]] = None,
        timeout: Optional[int] = None,
    ) -> str:
        """Assign a task to one or more suitable agents."""
        task_id = str(uuid.uuid4())

        required_capabilities = task.get("required_capabilities", [])

        suitable_agents = await self._find_suitable_agents(
            required_capabilities, preferred_agents
        )

        if not suitable_agents:
            raise AgentException(
                f"No suitable agents found for capabilities: {required_capabilities}"
            )

        self.active_tasks[task_id] = {
            "task": task,
            "assigned_agents": [a.config.agent_id for a in suitable_agents],
            "status": "pending",
            "created_at": datetime.now(),
            "timeout": timeout,
            "results": {},
            "errors": [],
            "completion_callback": task.get("completion_callback"),
        }

        await self.task_queue.put(
            {
                "task_id": task_id,
                "agents": suitable_agents,
                "task_data": task,
            }
        )

        return task_id

    async def coordinate_agents(
        self, agent_ids: List[str], task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Coordinate multiple agents working together."""
        coordination_id = str(uuid.uuid4())

        agents: List[BaseAgent] = []
        for agent_id in agent_ids:
            agent = self.agent_factory.get_agent(agent_id)
            if not agent:
                raise AgentException(f"Agent not found: {agent_id}")
            agents.append(agent)

        coordination = {
            "coordination_id": coordination_id,
            "agents": agent_ids,
            "task": task,
            "status": "in_progress",
            "created_at": datetime.now(),
            "round": 0,
            "agent_responses": {},
            "consensus_reached": False,
            "final_result": None,
        }

        self.collaboration_history.append(coordination)

        result = await self._execute_coordination_protocol(
            agents, task, coordination
        )

        coordination["final_result"] = result
        coordination["status"] = "completed"
        coordination["completed_at"] = datetime.now()

        return coordination

    # ------------------------------------------------------------------
    # Conflict resolution
    # ------------------------------------------------------------------

    def resolve_agent_conflicts(
        self, agent_responses: Dict[str, AgentOutput]
    ) -> AgentOutput:
        """Resolve conflicts between multiple agent responses."""
        if not agent_responses:
            raise ValidationError("No agent responses to resolve")

        if len(agent_responses) == 1:
            return next(iter(agent_responses.values()))

        best_agent_id = max(
            agent_responses,
            key=lambda aid: agent_responses[aid].confidence,
        )

        best_response = agent_responses[best_agent_id]
        best_response.metadata = best_response.metadata or {}
        best_response.metadata["conflict_resolution"] = {
            "method": "highest_confidence",
            "best_agent_id": best_agent_id,
            "total_responses": len(agent_responses),
        }

        return best_response

    # ------------------------------------------------------------------
    # Monitoring & optimization
    # ------------------------------------------------------------------

    async def balance_agent_load(self) -> Dict[str, Any]:
        """Balance load among agents."""
        agents = list(self.agent_factory.agent_pool.values())

        stats = {
            "total_agents": len(agents),
            "active_agents": 0,
            "idle_agents": 0,
            "error_agents": 0,
        }

        for agent in agents:
            state = getattr(agent.state, "value", None)
            if state == "processing":
                stats["active_agents"] += 1
            elif state == "ready":
                stats["idle_agents"] += 1
            elif state == "error":
                stats["error_agents"] += 1

        return stats

    async def monitor_agent_performance(
        self, timeframe_hours: int = 1
    ) -> Dict[str, Any]:
        """Monitor performance of all agents."""
        agents = list(self.agent_factory.agent_pool.values())

        report = {
            "timestamp": datetime.now(),
            "timeframe_hours": timeframe_hours,
            "agents": {},
        }

        for agent in agents:
            evaluation = await agent.evaluate()
            report["agents"][agent.config.agent_id] = evaluation

        return report

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------

    async def start(self, num_workers: int = 3) -> None:
        """Start orchestrator workers."""
        self._is_running = True
        self._task_workers = [
            asyncio.create_task(self._task_worker(f"worker-{i}"))
            for i in range(num_workers)
        ]

    async def stop(self) -> None:
        """Stop orchestrator and workers."""
        self._is_running = False
        for worker in self._task_workers:
            worker.cancel()
        await asyncio.gather(*self._task_workers, return_exceptions=True)
        self._task_workers.clear()

    async def _task_worker(self, worker_id: str) -> None:
        """Worker loop."""
        while self._is_running:
            try:
                task_item = await asyncio.wait_for(
                    self.task_queue.get(), timeout=1.0
                )

                task_id = task_item["task_id"]
                agents = task_item["agents"]
                task_data = task_item["task_data"]

                task_record = self.active_tasks[task_id]
                task_record["status"] = "processing"
                task_record["started_at"] = datetime.now()

                for agent in agents:
                    agent_input = AgentInput(
                        request_id=str(uuid.uuid4()),
                        data=task_data.get("data", {}),
                        context={
                            "task_id": task_id,
                            "worker_id": worker_id,
                        },
                        priority=task_data.get("priority", 1),
                    )

                    try:
                        result = await agent.process(agent_input)
                        task_record["results"][agent.config.agent_id] = result.dict()
                    except Exception as e:
                        task_record["errors"].append(
                            {
                                "agent_id": agent.config.agent_id,
                                "error": str(e),
                                "timestamp": datetime.now(),
                            }
                        )

                task_record["status"] = (
                    "failed"
                    if len(task_record["errors"]) == len(task_record["assigned_agents"])
                    else "completed"
                )
                task_record["completed_at"] = datetime.now()

                self.task_queue.task_done()

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[{worker_id}] Worker error: {e}")
                await asyncio.sleep(1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _find_suitable_agents(
        self,
        required_capabilities: List[str],
        preferred_agents: Optional[List[str]] = None,
    ) -> List[BaseAgent]:
        """Find agents with required capabilities."""
        agents = list(self.agent_factory.agent_pool.values())
        suitable: List[BaseAgent] = []

        for agent in agents:
            if getattr(agent.state, "value", None) != "ready":
                continue

            agent_caps = agent.config.capabilities or []
            if all(cap in agent_caps for cap in required_capabilities):
                suitable.append(agent)

        if preferred_agents:
            suitable = [
                a for a in suitable if a.config.agent_id in preferred_agents
            ]

        suitable.sort(
            key=lambda a: a.metrics.get("success_rate", 0.0),
            reverse=True,
        )

        return suitable
