# scripts/monitor_system.py
"""
Script para monitorear el sistema Project Brain.
Muestra métricas, estado de salud, y permite configurar alertas.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import asyncio
import json
from datetime import datetime, timedelta
import time
import psutil
import yaml
from tabulate import tabulate

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

# ---------------------------------------------------------------------
# Imports internos
# ---------------------------------------------------------------------
from core.orchestrator import BrainOrchestrator
from core.config_manager import ConfigManager
from utils.logging_config import setup_logging

# ---------------------------------------------------------------------
# SystemMonitor
# ---------------------------------------------------------------------
class SystemMonitor:
    """Clase para monitorear el sistema Project Brain."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or str(BASE_DIR / "config" / "system.yaml")
        self.config: Optional[Dict[str, Any]] = None
        self.orchestrator: Optional[BrainOrchestrator] = None
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> bool:
        """Inicializa el monitor."""
        try:
            setup_logging({
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": str(BASE_DIR / "logs" / "monitor_system.log")
            })

            config_manager = ConfigManager(self.config_path)
            self.config = config_manager.get_config()

            self.orchestrator = BrainOrchestrator(self.config_path)
            await self.orchestrator.initialize()

            self.logger.info("Monitor del sistema inicializado correctamente")
            return True

        except Exception as e:
            logging.exception("Error inicializando el monitor")
            return False

    # -----------------------------------------------------------------
    # Health check
    # -----------------------------------------------------------------
    async def check_system_health(self, detailed: bool = False) -> Dict[str, Any]:
        self.logger.info("Verificando salud del sistema")

        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall": "healthy",
            "components": {},
            "metrics": {},
            "issues": []
        }

        # Orchestrator
        try:
            metrics = await self.orchestrator.get_metrics()
            health_status["components"]["orchestrator"] = {
                "status": "healthy",
                "details": {
                    "operations_completed": metrics.get("operations_completed", 0),
                    "active_operations": metrics.get("active_operations", 0)
                }
            }
        except Exception as e:
            health_status["components"]["orchestrator"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["issues"].append(f"Orchestrator: {e}")

        # Databases
        health_status["components"]["databases"] = await self._check_databases()

        # Resources
        health_status["components"]["resources"] = self._check_system_resources()

        # Overall status
        unhealthy = [
            name for name, comp in health_status["components"].items()
            if comp.get("status") in {"unhealthy", "critical"}
        ]

        if unhealthy:
            health_status["overall"] = "unhealthy" if len(unhealthy) > 1 else "degraded"
            health_status["issues"].append(
                f"Componentes afectados: {', '.join(unhealthy)}"
            )

        health_status["metrics"] = await self._collect_system_metrics()
        return health_status

    # -----------------------------------------------------------------
    # Databases
    # -----------------------------------------------------------------
    async def _check_databases(self) -> Dict[str, Any]:
        status: Dict[str, Any] = {}

        # PostgreSQL
        try:
            import psycopg2
            cfg = self.config["databases"]["postgresql"]

            conn = psycopg2.connect(
                host=cfg["host"],
                port=cfg["port"],
                user=cfg["username"],
                password=os.getenv("DB_PASSWORD", cfg.get("password")),
                database=cfg["database"],
                connect_timeout=5
            )
            conn.close()
            status["postgresql"] = {"status": "healthy"}
        except Exception as e:
            status["postgresql"] = {"status": "unhealthy", "error": str(e)}

        # Redis
        try:
            import redis
            cfg = self.config["databases"]["redis"]

            r = redis.Redis(
                host=cfg["host"],
                port=cfg["port"],
                password=os.getenv("REDIS_PASSWORD", cfg.get("password")),
                socket_timeout=5
            )

            start = time.time()
            r.ping()
            latency = (time.time() - start) * 1000

            status["redis"] = {
                "status": "healthy",
                "latency_ms": round(latency, 2)
            }
        except Exception as e:
            status["redis"] = {"status": "unhealthy", "error": str(e)}

        return status

    # -----------------------------------------------------------------
    # System resources
    # -----------------------------------------------------------------
    def _check_system_resources(self) -> Dict[str, Any]:
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        issues = []
        status = "healthy"

        if cpu > 90:
            status = "critical"
            issues.append("CPU > 90%")
        elif cpu > 80:
            status = "degraded"
            issues.append("CPU > 80%")

        if mem.percent > 90:
            status = "critical"
            issues.append("Memory > 90%")
        elif mem.percent > 80:
            status = "degraded"
            issues.append("Memory > 80%")

        if disk.percent > 90:
            status = "critical"
            issues.append("Disk > 90%")

        return {
            "status": status,
            "details": {
                "cpu": {"percent": cpu},
                "memory": {
                    "percent": mem.percent,
                    "available_gb": round(mem.available / (1024 ** 3), 2)
                },
                "disk": {
                    "percent": disk.percent,
                    "free_gb": round(disk.free / (1024 ** 3), 2)
                }
            },
            "issues": issues
        }

    # -----------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}

        if self.orchestrator:
            orch = await self.orchestrator.get_metrics()
            metrics["orchestrator"] = {
                "operations_completed": orch.get("operations_completed", 0),
                "operations_failed": orch.get("operations_failed", 0),
                "success_rate": orch.get("success_rate", 1.0),
                "avg_response_time_ms": orch.get("avg_response_time_ms", 0)
            }

        return metrics

    # -----------------------------------------------------------------
    # Export helpers
    # -----------------------------------------------------------------
    @staticmethod
    def export_metrics_csv(data: Dict[str, Any], path: Path) -> None:
        import csv

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value", "timestamp"])

            writer.writerow([
                "health_overall",
                data["health"].get("overall"),
                data["timestamp"]
            ])

            resources = data["health"]["components"]["resources"]["details"]
            writer.writerow(["cpu_percent", resources["cpu"]["percent"], data["timestamp"]])
            writer.writerow(["memory_percent", resources["memory"]["percent"], data["timestamp"]])

# ---------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------
async def export_metrics_command(args):
    monitor = SystemMonitor(args.config)

    if not await monitor.initialize():
        print("No se pudo inicializar el monitor")
        return 1

    try:
        health = await monitor.check_system_health(detailed=True)
        metrics = await monitor._collect_system_metrics()

        export_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "health": health,
            "metrics": metrics
        }

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)

        if args.format == "json":
            output.write_text(json.dumps(export_data, indent=2))
        elif args.format == "yaml":
            output.write_text(yaml.dump(export_data))
        elif args.format == "csv":
            SystemMonitor.export_metrics_csv(export_data, output)
        else:
            print("Formato no soportado")
            return 1

        print(f"Métricas exportadas a {output}")
        return 0

    finally:
        if monitor.orchestrator:
            await monitor.orchestrator.shutdown()

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser("Monitor Project Brain")
    parser.add_argument("--config", default=str(BASE_DIR / "config" / "system.yaml"))

    sub = parser.add_subparsers(dest="command")

    export = sub.add_parser("export")
    export.add_argument("output")
    export.add_argument("--format", choices=["json", "yaml", "csv"], default="json")

    args = parser.parse_args()

    if args.command == "export":
        return asyncio.run(export_metrics_command(args))

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
