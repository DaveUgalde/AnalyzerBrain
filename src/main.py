"""
Módulo principal de entrada de ANALYZERBRAIN.

Este módulo proporciona la CLI principal y el sistema central de ANALYZERBRAIN,
un sistema inteligente de análisis de código. Maneja la inicialización del sistema,
gestión de componentes, análisis de proyectos y operaciones de consulta.

Dependencias Previas:
    1. src.core.config_manager - Gestión de configuración
    2. src.utils.logging_config - Sistema de logging estructurado
    3. src.core.health_check - Verificación de salud del sistema

Módulos Importados:
    - sys: Funcionalidades del sistema
    - signal: Manejo de señales
    - pathlib: Manejo de rutas
    - typing: Anotaciones de tipo
    - datetime: Manejo de fechas y horas
    - enum: Enumeraciones
    - threading: Hilos

Autor: ANALYZERBRAIN Team
Fecha: 2024
Versión: 1.0.0
Licencia: Propietario
"""

import sys
import signal
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
import threading

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)

# from rich.tree import Tree
from rich import box

from importlib.util import find_spec
from .core.config_manager import config
from .utils.logging_config import StructuredLogger
from .utils.file_utils import FileUtils
from .core.exceptions import AnalyzerBrainError
from .core.health_check import SystemHealthChecker

logger = StructuredLogger.get_logger(__name__)
console = Console()


# ─────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────


class SystemStatus(Enum):
    """Estado del sistema ANALYZERBRAIN."""

    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initiaizing"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


# ─────────────────────────────────────────────────────────────
# Sistema principal
# ─────────────────────────────────────────────────────────────


class AnalyzerBrainSystem:
    """Sistema principal de ANALYZERBRAIN.

    Esta clase maneja el ciclo de vida completo del sistema, incluyendo
    inicialización, análisis de proyectos, consultas y apagado controlado.

    Atributos:
        status (SystemStatus): Estado actual del sistema
        start_time (Optional[datetime]): Hora de inicio del sistema
        health_checker (Optional[SystemHealthChecker]): Verificador de salud
        _shutdown_flag (bool): Bandera de apagado
        components (Dict[str, Any]): Componentes del sistema registrados
    """

    def __init__(self) -> None:
        """Inicializa una nueva instancia del sistema.

        Configura el estado inicial y prepara los componentes del sistema.
        """
        self.status = SystemStatus.UNINITIALIZED
        self.start_time: Optional[datetime] = None
        self.health_checker: Optional[SystemHealthChecker] = None
        self._shutdown_flag = False
        self.components: Dict[str, Any] = {}

    # ───────────── Utils internos ─────────────

    @staticmethod
    def _run_async(coro: Any) -> Any:
        """Ejecuta una coroutine de forma segura.

        Args:
            coro: Coroutine a ejecutar

        Returns:
            Resultado de la coroutine o tarea creada

        Note:
            Esta función maneja automáticamente diferentes escenarios de event loop.
        """
        import asyncio

        # Verificar si ya hay un loop en ejecución
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Si ya hay un loop corriendo, crear tarea
                return asyncio.create_task(coro)
            else:
                # Si no hay loop corriendo, usar run_until_complete
                return loop.run_until_complete(coro)
        except RuntimeError:
            # No hay loop, crear uno nuevo
            return asyncio.run(coro)

    # ───────────── Banner ─────────────

    def print_banner(self) -> None:
        """Imprime el banner de presentación del sistema.

        Muestra el logo de ANALYZERBRAIN junto con información de versión
        y entorno en un panel visualmente atractivo.
        """
        banner = """
╔══════════════════════════════════════════════════════════╗
║   ░█████╗░███╗░░██╗░█████╗░██╗░░░░░███████╗██╗░░░██╗   ║
║   ██╔══██╗████╗░██║██╔══██╗██║░░░░░██╔════╝╚██╗░██╔╝   ║
║   ███████║██╔██╗██║███████║██║░░░░░█████╗░░░╚████╔╝░   ║
║   ██╔══██║██║╚████║██╔══██║██║░░░░░██╔══╝░░░░╚██╔╝░░   ║
║   ██║░░██║██║░╚███║██║░░██║███████╗███████╗░░░██║░░░   ║
║   ╚═╝░░╚═╝╚═╝░░╚══╝╚═╝░░╚═╝╚══════╝╚══════╝░░░╚═╝░░░   ║
║   ░██████╗░██████╗░░█████╗░██╗░█████╗░███╗░░██╗        ║
╚══════════════════════════════════════════════════════════╝
        """
        version_info = (
            f"Versión: {config.get('system.version', '0.1.0')} | " f"Entorno: {config.environment}"
        )
        env_color = "green" if config.is_development else "yellow"

        console.print(
            Panel.fit(
                banner,
                title="[bold cyan]ANALYZERBRAIN[/bold cyan]",
                subtitle=f"[{env_color}]{version_info}[/{env_color}]",
                border_style="cyan",
            )
        )

    # ───────────── Requisitos ─────────────

    def check_system_requirements(self) -> Dict[str, bool]:
        """Verifica requisitos básicos del sistema.

        Returns:
            Dict[str, bool]: Diccionario con el estado de cada requisito:
                - python_version: Versión de Python >= 3.9
                - directories: Directorios creados correctamente
                - write_permissions: Permisos de escritura
                - configuration: Configuración crítica presente
                - dependencies: Dependencias Python instaladas

        Raises:
            RuntimeError: Si hay problemas críticos con los requisitos
        """
        requirements: Dict[str, bool] = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Verificando requisitos...", total=100)

            # Python
            progress.update(task, advance=10)
            requirements["python_version"] = (
                sys.version_info.major == 3 and sys.version_info.minor >= 9
            )

            # Directorios
            progress.update(task, advance=20)
            try:
                # Obtener directorios desde configuración
                data_dir_str = config.get('storage.data_dir', './data')
                log_dir_str = config.get('storage.log_dir', './logs')

                data_dir = Path(data_dir_str)
                log_dir = Path(log_dir_str)

                data_dir.mkdir(parents=True, exist_ok=True)
                log_dir.mkdir(parents=True, exist_ok=True)
                requirements["directories"] = True

                # Ahora podemos usar data_dir aquí porque está definido
                test_file = data_dir / ".write_test"
                test_file.touch()
                test_file.unlink()
                requirements["write_permissions"] = True

            except Exception as e:
                logger.error(f"Error con directorios: {e}")
                requirements["directories"] = False
                requirements["write_permissions"] = False

            # Config crítica
            progress.update(task, advance=20)
            requirements["configuration"] = all(
                config.get(k) is not None
                for k in (
                    "environment",
                    "storage.data_dir",
                    "api.host",
                    "api.port",
                )
            )

            # Dependencias Python
            progress.update(task, advance=30)
            requirements["dependencies"] = all(
                find_spec(pkg) is not None for pkg in ("pydantic", "loguru", "rich")
            )

        return requirements

    # ───────────── Inicialización ─────────────

    async def initialize(self) -> bool:
        """Inicializa el sistema completo.

        Returns:
            bool: True si la inicialización fue exitosa, False en caso contrario

        Raises:
            RuntimeError: Si el sistema ya está en un estado incompatible
        """
        if self.status not in (SystemStatus.UNINITIALIZED, SystemStatus.ERROR):
            logger.warning(f"Sistema ya está en estado: {self.status}")
            return True

        self.status = SystemStatus.INITIALIZING
        self.start_time = datetime.now()

        try:
            self.print_banner()

            # 1. Verificación de requisitos
            console.print("\n[bold]1. Verificación de requisitos[/bold]")
            requirements = self.check_system_requirements()

            if not all(requirements.values()):
                console.print("\n[bold red]❌ Requisitos del sistema no cumplidos:[/bold red]")
                for req, status in requirements.items():
                    status_icon = "✅" if status else "❌"
                    console.print(f"  {status_icon} {req}")
                self.status = SystemStatus.ERROR
                return False

            console.print("[bold green]✅ Requisitos verificados[/bold green]")

            # 2. Inicialización de componentes
            console.print("\n[bold]2. Inicialización de componentes[/bold]")

            # Health Checker
            console.print("  • Inicializando Health Checker...")
            self.health_checker = SystemHealthChecker()

            try:
                health = await self.health_checker.check_all()
                if not health.get("overall", False):
                    console.print(
                        f"[bold red]❌ Health Check falló: {health.get('status')}[/bold red]"
                    )
                    self._print_health_report(health)
                    self.status = SystemStatus.ERROR
                    return False
                console.print("[bold green]  ✅ Health Check completado[/bold green]")
            except Exception as e:
                logger.error(f"Health check falló: {e}", exc_info=True)
                self.status = SystemStatus.ERROR
                return False

            # 3. Configurar componentes restantes
            self._setup_signal_handlers()

            # 4. Registrar componentes
            self.components = {
                "config_manager": config,
                "health_checker": self.health_checker,
                "logging": "configured",
                "file_utils": FileUtils(),
                "event_bus": "pending",  # Para Fase 1
                "system_state": "pending",  # Para Fase 1
                "orchestrator": "pending",  # Para Fase 2
                "indexer": "pending",  # Para Fase 2
                "graph": "pending",  # Para Fase 3
                "agents": "pending",  # Para Fase 4
                "embeddings": "pending",  # Para Fase 5
                "api": "pending",  # Para Fase 6
            }

            self.status = SystemStatus.READY

            console.print("\n[bold green]✅ Sistema listo[/bold green]")
            self._print_system_summary()
            return True

        except Exception as e:
            logger.error("Error crítico en inicialización", exc_info=True)
            self.status = SystemStatus.ERROR
            console.print(f"[bold red]❌ Error en inicialización: {e}[/bold red]")
            return False

    # ───────────── Resumen del sistema ─────────────

    def _print_system_summary(self) -> None:
        """Imprime un resumen del estado del sistema.

        Muestra información clave como versión, entorno, directorios
        y estado de los componentes en una tabla formateada.
        """
        console.print("\n" + "=" * 60)
        console.print("[bold cyan]RESUMEN DEL SISTEMA[/bold cyan]")
        console.print("=" * 60)

        table = Table(show_header=False, box=box.SIMPLE)
        table.add_column("", style="cyan", width=20)
        table.add_column("", style="white")

        table.add_row("Estado", f"[green]{self.status.value}[/green]")
        table.add_row("Versión", config.get('system.version', '0.1.0'))
        table.add_row("Entorno", config.environment)
        table.add_row("Directorio Datos", str(config.get('storage.data_dir', './data')))
        table.add_row("Directorio Logs", str(config.get('storage.log_dir', './logs')))

        if self.health_checker:
            health_status = self.health_checker.get_status()
            table.add_row(
                "Health Check", f"[green]{health_status.get('status', 'unknown')}[/green]"
            )

        console.print(table)
        console.print("=" * 60)

    def _print_health_report(self, health_result: Dict[str, Any]) -> None:
        """Imprime un reporte detallado del health check.

        Args:
            health_result: Resultados del health check con estructura:
                {
                    "checks": List[Dict],
                    "overall": bool,
                    "summary": Dict
                }
        """
        console.print("\n[bold red]REPORTE DE SALUD - FALLAS DETECTADAS[/bold red]")
        console.print("=" * 60)

        for check in health_result.get("checks", []):
            if check.get("status") != "healthy":
                status_icon = {
                    "healthy": "✅",
                    "warning": "⚠️",
                    "unhealthy": "❌",
                    "error": "💥",
                }.get(check.get("status"), "❓")

                console.print(f"{status_icon} [bold]{check.get('name')}[/bold]")
                console.print(f"  Mensaje: {check.get('message')}")

                details = check.get("details", {})
                if "error" in details:
                    console.print(f"  Error: {details['error']}")

        console.print("=" * 60)

    # ───────────── Señales ─────────────

    def _setup_signal_handlers(self) -> None:
        """Configura handlers para señales del sistema.

        Establece manejadores para SIGINT y SIGTERM para permitir
        un apagado controlado del sistema.
        """
        if threading.current_thread() is not threading.main_thread():
            return

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        console.print("[dim]  ✅ Handlers de señal configurados[/dim]")

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handler para señales de terminación.

        Args:
            signum: Número de la señal recibida
            frame: Marco de ejecución actual
        """
        logger.info(f"Señal {signum} recibida")
        console.print(f"\n[bold yellow]⚠️  Señal {signum} recibida, apagando...[/bold yellow]")
        self.shutdown()

    # ───────────── Operaciones ─────────────

    async def analyze_project(self, project_path: Path) -> Dict[str, Any]:
        """Analiza un proyecto de código.

        Args:
            project_path: Ruta al directorio del proyecto a analizar

        Returns:
            Dict con resultados del análisis:
                - project: Ruta del proyecto
                - status: Estado del análisis
                - message: Mensaje descriptivo
                - files_analyzed: Número de archivos analizados
                - entities_found: Entidades encontradas
                - analysis_time: Tiempo de análisis
                - warnings: Lista de advertencias
                - errors: Lista de errores

        Raises:
            AnalyzerBrainError: Si el sistema no está listo o el proyecto no existe
        """
        if self.status != SystemStatus.READY:
            raise AnalyzerBrainError("Sistema no está listo")

        if not project_path.exists():
            raise AnalyzerBrainError(f"Proyecto no encontrado: {project_path}")

        console.print(f"\n[bold]🔍 Analizando proyecto: {project_path}[/bold]")

        # Simulación de análisis (será implementado en Fase 2)
        import time

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Analizando..."),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("", total=100)
            for _ in range(10):
                time.sleep(0.1)  # Usar time.sleep en lugar de asyncio.sleep para simplificar
                progress.update(task, advance=10)

        return {
            "project": str(project_path),
            "status": "success",
            "message": "Análisis simulado completado",
            "phase": "Fase 2 - Indexer (pendiente de implementación)",
            "files_analyzed": 0,
            "entities_found": 0,
            "analysis_time": 1.0,
            "warnings": ["Funcionalidad pendiente de implementación"],
            "errors": [],
            "next_steps": [
                "Implementar src/indexer/project_scanner.py",
                "Implementar src/indexer/multi_language_parser.py",
                "Conectar con grafo de conocimiento",
            ],
        }

    async def query_system(self, query: str) -> Dict[str, Any]:
        """Consulta el sistema de conocimiento.

        Args:
            query: Consulta a realizar al sistema

        Returns:
            Dict con resultados de la consulta:
                - query: Consulta original
                - status: Estado de la consulta
                - results: Lista de resultados
                - sources: Fuentes consultadas
                - timestamp: Marca de tiempo

        Raises:
            AnalyzerBrainError: Si el sistema no está listo
        """
        if self.status != SystemStatus.READY:
            raise AnalyzerBrainError("Sistema no está listo")

        return {
            "query": query,
            "status": "success",
            "results": [
                {
                    "type": "info",
                    "content": f"Consulta: '{query}'",
                    "confidence": 0.8,
                    "source": "sistema_de_conocimiento",
                    "phase": "Fase 3 - Graph (pendiente de implementación)",
                }
            ],
            "sources": ["knowledge_graph_pending"],
            "timestamp": datetime.now().isoformat(),
            "note": "La funcionalidad de consulta será implementada en Fase 3",
        }

    def shutdown(self) -> None:
        """Apaga el sistema de manera controlada.

        Libera recursos, cambia el estado y notifica a los componentes
        del apagado inminente.
        """
        if self.status in (SystemStatus.SHUTTING_DOWN, SystemStatus.UNINITIALIZED):
            return

        self.status = SystemStatus.SHUTTING_DOWN
        logger.info("Apagando sistema…")

        console.print("\n[bold yellow]🔌 Apagando ANALYZERBRAIN...[/bold yellow]")

        # Limpiar recursos
        self._shutdown_flag = True

        console.print("[bold green]✅ Sistema apagado correctamente[/bold green]")
        self.status = SystemStatus.UNINITIALIZED

    def get_status(self) -> Dict[str, Any]:
        """Obtiene el estado actual del sistema.

        Returns:
            Dict con información completa del estado:
                - status: Estado actual
                - uptime_seconds: Tiempo activo
                - environment: Entorno configurado
                - version: Versión del sistema
                - start_time: Hora de inicio
                - components_ready: Estado de componentes
                - health: Estado de salud (si disponible)
        """
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0

        status_dict: Dict[str, Any] = {
            "status": self.status.value,
            "uptime_seconds": uptime,
            "environment": config.environment,
            "version": config.get("system.version", "0.1.0"),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "components_ready": {
                name: status != "pending" for name, status in self.components.items()
            },
        }

        if self.health_checker:
            status_dict["health"] = self.health_checker.get_status()

        return status_dict


# ─────────────────────────────────────────────────────────────
# CLI Principal
# ─────────────────────────────────────────────────────────────

system = AnalyzerBrainSystem()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version="0.1.0", prog_name="ANALYZERBRAIN")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """ANALYZERBRAIN - Sistema inteligente de análisis de código.

    Comandos disponibles:
        init    - Inicializa el sistema
        analyze - Analiza un proyecto
        query   - Consulta el sistema
        status  - Muestra el estado
        health  - Verifica salud del sistema
    """
    ctx.ensure_object(dict)
    ctx.obj["system"] = system


@cli.command()
def init() -> None:
    """Inicializa el sistema ANALYZERBRAIN.

    Realiza todas las verificaciones y configura los componentes
    necesarios para el funcionamiento del sistema.
    """
    import asyncio

    try:
        # Usar asyncio.run directamente
        success = asyncio.run(system.initialize())

        if not success:
            console.print("[bold red]❌ Falló la inicialización del sistema[/bold red]")
            sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]❌ Error durante inicialización: {e}[/bold red]")
        sys.exit(1)


@cli.command()
@click.argument("project_path", type=click.Path(exists=True, path_type=Path))
def analyze(project_path: Path) -> None:
    """Analiza un proyecto de código.

    Args:
        project_path: Ruta al directorio del proyecto a analizar
    """
    import asyncio

    try:
        # Inicializar si no está listo
        if system.status != SystemStatus.READY:
            console.print("[yellow]⚠️  Sistema no inicializado, inicializando...[/yellow]")
            if not asyncio.run(system.initialize()):
                console.print("[bold red]❌ No se pudo inicializar el sistema[/bold red]")
                sys.exit(1)

        # Ejecutar análisis
        result: dict[str, Union[str, int, bool, list[str]]]
        result = asyncio.run(system.analyze_project(project_path))

        # Mostrar resultados
        console.print("\n" + "=" * 60)
        console.print("[bold green]📊 RESULTADOS DEL ANÁLISIS[/bold green]")
        console.print("=" * 60)

        for key, value in result.items():
            if isinstance(value, list):
                console.print(f"[cyan]{key}:[/cyan]")
                for item in value:
                    console.print(f"  • {item}")
            else:
                console.print(f"[cyan]{key}:[/cyan] {value}")

        console.print("=" * 60)

    except Exception as e:
        console.print(f"[bold red]❌ Error durante análisis: {e}[/bold red]")
        sys.exit(1)


@cli.command()
@click.argument("query", type=str)
def query(query: str) -> None:
    """Consulta el sistema de conocimiento.

    Args:
        query: Consulta a realizar al sistema
    """
    import asyncio

    try:
        if system.status != SystemStatus.READY:
            console.print("[yellow]⚠️  Sistema no inicializado, inicializando...[/yellow]")
            if not asyncio.run(system.initialize()):
                console.print("[bold red]❌ No se pudo inicializar el sistema[/bold red]")
                sys.exit(1)

        result = asyncio.run(system.query_system(query))

        console.print("\n" + "=" * 60)
        console.print("[bold green]🤖 RESPUESTA DEL SISTEMA[/bold green]")
        console.print("=" * 60)
        console.print(f"[cyan]Consulta:[/cyan] {result['query']}")

        for res in result.get("results", []):
            console.print(f"\n[bold]{res['type'].upper()}:[/bold] {res['content']}")
            console.print(f"  Confianza: {res['confidence']}")
            console.print(f"  Fuente: {res['source']}")

        console.print("=" * 60)

    except Exception as e:
        console.print(f"[bold red]❌ Error durante consulta: {e}[/bold red]")
        sys.exit(1)


@cli.command()
def status() -> None:
    """Muestra el estado del sistema."""
    try:
        status_info = system.get_status()

        console.print("\n" + "=" * 60)
        console.print("[bold cyan]📈 ESTADO DEL SISTEMA[/bold cyan]")
        console.print("=" * 60)

        for key, value in status_info.items():
            if key == "components_ready":
                console.print(f"[cyan]{key}:[/cyan]")
                for comp, ready in value.items():
                    status_icon = "✅" if ready else "⏳"
                    console.print(f"  {status_icon} {comp}")
            elif key == "health":
                console.print(f"[cyan]{key}:[/cyan]")
                for hkey, hvalue in value.items():
                    console.print(f"  {hkey}: {hvalue}")
            else:
                console.print(f"[cyan]{key}:[/cyan] {value}")

        console.print("=" * 60)

    except Exception as e:
        console.print(f"[bold red]❌ Error obteniendo estado: {e}[/bold red]")
        sys.exit(1)


@cli.command()
def health() -> None:
    """Ejecuta un health check del sistema."""
    import asyncio

    try:
        if system.health_checker is None:
            system.health_checker = SystemHealthChecker()

        console.print("\n[bold]🧪 Ejecutando Health Check...[/bold]")
        result = asyncio.run(system.health_checker.check_all())

        console.print("\n" + "=" * 60)

        if result.get("overall", False):
            console.print("[bold green]✅ HEALTH CHECK: PASSED[/bold green]")
        else:
            console.print("[bold red]❌ HEALTH CHECK: FAILED[/bold red]")

        console.print("=" * 60)

        summary = result.get("summary", {})
        console.print(f"Total checks: {summary.get('total_checks', 0)}")
        console.print(f"✅ Saludables: {summary.get('healthy', 0)}")
        console.print(f"⚠️  Advertencias: {summary.get('warnings', 0)}")
        console.print(f"❌ No saludables: {summary.get('unhealthy', 0)}")
        console.print(f"💥 Errores: {summary.get('errors', 0)}")
        console.print(f"Tasa éxito: {summary.get('success_rate', 0)}%")

        console.print("=" * 60)

    except Exception as e:
        console.print(f"[bold red]❌ Error en health check: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    cli()
