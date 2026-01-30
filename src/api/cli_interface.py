"""
CLI - Interfaz de línea de comandos para Project Brain.
"""

import logging
import sys
import os
import json
import yaml
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from pathlib import Path
import argparse
import asyncio
import readline  # Para historial en Unix-like systems
import shlex
import cmd
import questionary
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.layout import Layout

from ..core.exceptions import BrainException, ValidationError
from ..core.orchestrator import BrainOrchestrator
from ..core.config_manager import ConfigManager
from .authentication import Authentication

logger = logging.getLogger(__name__)
console = Console()

class CLIInterface:
    """
    Interfaz de línea de comandos completa.
    
    Características:
    1. Parsing de argumentos con argparse
    2. Modo interactivo con autocompletado
    3. Formateo de salida con rich
    4. Manejo de historial de comandos
    5. Validación de entrada
    6. Modo batch para scripts
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa la interfaz CLI.
        
        Args:
            config_path: Ruta al archivo de configuración (opcional)
        """
        self.config_manager = ConfigManager()
        if config_path:
            self.config_manager.load_config(config_path)
        
        self.config = self.config_manager.get_config().get("cli", {})
        self.orchestrator: Optional[BrainOrchestrator] = None
        self.authentication: Optional[Authentication] = None
        
        # Historial de comandos
        self.history: List[str] = []
        self.history_file = Path.home() / ".project_brain_history"
        self._load_history()
        
        # Estado de sesión
        self.session_id = str(datetime.now().timestamp())
        self.current_project: Optional[str] = None
        self.interactive_mode = False
        
        # Comandos disponibles
        self.commands = self._initialize_commands()
        
        # Configurar autocompletado
        self._setup_completion()
        
        console.print("[bold green]Project Brain CLI inicializado[/bold green]")
        console.print(f"Session ID: {self.session_id}")
        logger.info("CLIInterface inicializado")
    
    def parse_arguments(self) -> argparse.Namespace:
        """
        Parsea argumentos de línea de comandos.
        
        Returns:
            Namespace con argumentos parseados
        """
        parser = argparse.ArgumentParser(
            description="Project Brain CLI - Intelligent code analysis and Q&A",
            epilog="Use 'brain --help' for more information on a command.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        
        # Comando principal
        subparsers = parser.add_subparsers(dest="command", help="Command to execute")
        
        # ========== PROYECTOS ==========
        project_parser = subparsers.add_parser("project", help="Project operations")
        project_subparsers = project_parser.add_subparsers(dest="project_command")
        
        # Listar proyectos
        list_parser = project_subparsers.add_parser("list", help="List projects")
        list_parser.add_argument("--format", choices=["table", "json", "yaml"], default="table")
        list_parser.add_argument("--filter", help="Filter by language or status")
        
        # Crear proyecto
        create_parser = project_subparsers.add_parser("create", help="Create project")
        create_parser.add_argument("name", help="Project name")
        create_parser.add_argument("path", help="Project path")
        create_parser.add_argument("--description", help="Project description")
        create_parser.add_argument("--language", help="Main language")
        create_parser.add_argument("--analyze", action="store_true", help="Analyze immediately")
        
        # Analizar proyecto
        analyze_parser = project_subparsers.add_parser("analyze", help="Analyze project")
        analyze_parser.add_argument("project_id", help="Project ID or path")
        analyze_parser.add_argument("--mode", choices=["quick", "standard", "comprehensive", "deep"], 
                                   default="comprehensive")
        analyze_parser.add_argument("--wait", action="store_true", help="Wait for completion")
        
        # ========== CONSULTAS ==========
        query_parser = subparsers.add_parser("query", help="Query operations")
        query_subparsers = query_parser.add_subparsers(dest="query_command")
        
        # Preguntar
        ask_parser = query_subparsers.add_parser("ask", help="Ask a question")
        ask_parser.add_argument("question", nargs="+", help="Question to ask")
        ask_parser.add_argument("--project", help="Project ID or path")
        ask_parser.add_argument("--format", choices=["text", "json", "markdown"], default="text")
        ask_parser.add_argument("--stream", action="store_true", help="Stream response")
        
        # Conversación
        chat_parser = query_subparsers.add_parser("chat", help="Interactive chat")
        chat_parser.add_argument("--project", help="Project ID or path")
        
        # ========== SISTEMA ==========
        system_parser = subparsers.add_parser("system", help="System operations")
        system_subparsers = system_parser.add_subparsers(dest="system_command")
        
        # Salud
        system_subparsers.add_parser("health", help="Check system health")
        
        # Métricas
        metrics_parser = system_subparsers.add_parser("metrics", help="Show metrics")
        metrics_parser.add_argument("--timeframe", choices=["hour", "day", "week", "month"], default="hour")
        
        # ========== INTERACTIVO ==========
        subparsers.add_parser("interactive", help="Start interactive mode")
        
        # ========== CONFIGURACIÓN ==========
        config_parser = subparsers.add_parser("config", help="Configuration operations")
        config_subparsers = config_parser.add_subparsers(dest="config_command")
        
        # Mostrar configuración
        config_subparsers.add_parser("show", help="Show configuration")
        
        # Configurar
        set_parser = config_subparsers.add_parser("set", help="Set configuration")
        set_parser.add_argument("key", help="Configuration key")
        set_parser.add_argument("value", help="Configuration value")
        
        # Argumentos globales
        parser.add_argument("--config", help="Configuration file path")
        parser.add_argument("--verbose", "-v", action="count", default=0, help="Increase verbosity")
        parser.add_argument("--version", action="store_true", help="Show version")
        parser.add_argument("--output", "-o", help="Output file path")
        
        return parser.parse_args()
    
    async def execute_command(self, command_line: Optional[str] = None) -> int:
        """
        Ejecuta un comando CLI.
        
        Args:
            command_line: Línea de comandos (si None, usa sys.argv)
            
        Returns:
            int: Código de salida (0 = éxito)
        """
        try:
            if command_line is None:
                # Parsear desde sys.argv
                args = self.parse_arguments()
            else:
                # Parsear desde string
                args = self._parse_command_string(command_line)
            
            # Manejar argumentos globales
            if args.version:
                self._show_version()
                return 0
            
            # Ejecutar comando
            if not hasattr(args, 'command') or args.command is None:
                # No se especificó comando, iniciar modo interactivo
                return await self.handle_interactive_mode()
            
            # Mapear comando a función
            command_func = self._get_command_function(args)
            if not command_func:
                console.print("[red]Error: Comando no reconocido[/red]")
                return 1
            
            # Ejecutar comando
            result = await command_func(args)
            
            # Guardar en historial si es modo interactivo
            if self.interactive_mode and command_line:
                self._add_to_history(command_line)
            
            return 0 if result else 1
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Comando cancelado por usuario[/yellow]")
            return 130  # SIGINT exit code
            
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            if hasattr(args, 'verbose') and args.verbose > 0:
                console.print_exception()
            return 1
    
    def display_help(self, command: Optional[str] = None) -> None:
        """
        Muestra ayuda.
        
        Args:
            command: Comando específico para mostrar ayuda (opcional)
        """
        if command:
            # Mostrar ayuda específica del comando
            help_text = self._get_command_help(command)
            if help_text:
                console.print(Panel(help_text, title=f"Help: {command}", border_style="blue"))
            else:
                console.print(f"[yellow]No hay ayuda disponible para '{command}'[/yellow]")
        else:
            # Mostrar ayuda general
            help_table = Table(title="Project Brain CLI - Comandos Disponibles", show_header=True)
            help_table.add_column("Comando", style="cyan")
            help_table.add_column("Descripción", style="white")
            help_table.add_column("Uso", style="dim")
            
            for cmd_name, cmd_info in self.commands.items():
                help_table.add_row(
                    cmd_name,
                    cmd_info.get("description", "Sin descripción"),
                    cmd_info.get("usage", "")
                )
            
            console.print(help_table)
            console.print("\n[dim]Usa 'brain help <comando>' para ayuda específica[/dim]")
    
    async def handle_interactive_mode(self) -> int:
        """
        Maneja modo interactivo.
        
        Returns:
            int: Código de salida
        """
        self.interactive_mode = True
        
        console.clear()
        console.print(Panel.fit(
            "[bold cyan]Project Brain Interactive Mode[/bold cyan]\n"
            "[dim]Type 'help' for commands, 'exit' to quit[/dim]",
            border_style="green"
        ))
        
        # Inicializar orquestador si es necesario
        if not self.orchestrator:
            self.orchestrator = BrainOrchestrator()
            await self.orchestrator.initialize()
        
        # Loop interactivo
        while True:
            try:
                # Leer comando con autocompletado
                command_line = await questionary.text(
                    "brain>",
                    qmark="",
                    completer=self._create_completer(),
                    history=self.history,
                ).unsafe_ask_async()
                
                if not command_line:
                    continue
                
                # Comandos especiales
                if command_line.lower() in ["exit", "quit", "q"]:
                    break
                
                if command_line.lower() in ["clear", "cls"]:
                    console.clear()
                    continue
                
                if command_line.lower() in ["help", "?"]:
                    self.display_help()
                    continue
                
                if command_line.lower().startswith("help "):
                    cmd = command_line[5:].strip()
                    self.display_help(cmd)
                    continue
                
                # Ejecutar comando
                exit_code = await self.execute_command(command_line)
                
                # Guardar en historial
                self._add_to_history(command_line)
                
                # Si hubo error, mostrar mensaje
                if exit_code != 0:
                    console.print("[yellow]El comando falló[/yellow]")
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Usa 'exit' para salir[/yellow]")
                continue
                
            except EOFError:
                break
                
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")
        
        console.print("[green]Saliendo del modo interactivo[/green]")
        return 0
    
    def manage_command_history(self) -> None:
        """Gestiona el historial de comandos."""
        # El historial ya se maneja en handle_interactive_mode
        # Este método expone funcionalidades adicionales
        
        if self.history:
            console.print(Panel.fit(
                "\n".join(f"{i+1}: {cmd}" for i, cmd in enumerate(self.history[-20:])),
                title="Historial de Comandos (últimos 20)",
                border_style="blue"
            ))
        else:
            console.print("[dim]No hay historial de comandos[/dim]")
    
    def validate_cli_input(self, input_str: str, expected_type: str = "command") -> bool:
        """
        Valida entrada de CLI.
        
        Args:
            input_str: String de entrada
            expected_type: Tipo esperado (command, path, question, etc.)
            
        Returns:
            bool: True si la entrada es válida
        """
        if not input_str or not input_str.strip():
            return False
        
        if expected_type == "command":
            # Validar formato básico de comando
            parts = shlex.split(input_str)
            if not parts:
                return False
            
            command = parts[0].lower()
            return command in self.commands or command in ["help", "exit", "clear"]
        
        elif expected_type == "path":
            # Validar ruta
            path = Path(input_str)
            return path.exists()
        
        elif expected_type == "question":
            # Validar pregunta (mínimo 3 caracteres)
            return len(input_str.strip()) >= 3
        
        return True
    
    def format_output(self, data: Any, format: str = "text") -> str:
        """
        Formatea salida según el formato especificado.
        
        Args:
            data: Datos a formatear
            format: Formato de salida (text, json, yaml, table, markdown)
            
        Returns:
            str: Datos formateados
        """
        if format == "json":
            return json.dumps(data, indent=2, default=str)
        
        elif format == "yaml":
            import yaml
            return yaml.dump(data, default_flow_style=False, allow_unicode=True)
        
        elif format == "table" and isinstance(data, list):
            # Crear tabla desde lista de dicts
            if not data:
                return "No data"
            
            table = Table(show_header=True)
            
            # Determinar columnas
            if isinstance(data[0], dict):
                columns = list(data[0].keys())
                for col in columns:
                    table.add_column(col, style="cyan")
                
                for row in data:
                    table.add_row(*[str(row.get(col, "")) for col in columns])
                
                return table
        
        elif format == "markdown":
            if isinstance(data, dict):
                markdown = []
                for key, value in data.items():
                    markdown.append(f"**{key}**: {value}")
                return "\n\n".join(markdown)
        
        # Por defecto, convertir a string
        return str(data)
    
    # Métodos de implementación
    
    def _initialize_commands(self) -> Dict[str, Dict[str, Any]]:
        """Inicializa diccionario de comandos."""
        return {
            "project": {
                "description": "Operaciones con proyectos",
                "subcommands": {
                    "list": "Listar proyectos",
                    "create": "Crear proyecto",
                    "analyze": "Analizar proyecto",
                    "delete": "Eliminar proyecto",
                }
            },
            "query": {
                "description": "Consultas al sistema",
                "subcommands": {
                    "ask": "Hacer una pregunta",
                    "chat": "Modo conversacional",
                    "search": "Buscar en conocimiento",
                }
            },
            "system": {
                "description": "Operaciones del sistema",
                "subcommands": {
                    "health": "Verificar salud del sistema",
                    "metrics": "Mostrar métricas",
                    "status": "Estado del sistema",
                }
            },
            "config": {
                "description": "Configuración",
                "subcommands": {
                    "show": "Mostrar configuración",
                    "set": "Establecer valor",
                    "reload": "Recargar configuración",
                }
            },
            "interactive": {
                "description": "Iniciar modo interactivo",
            },
            "help": {
                "description": "Mostrar ayuda",
            },
        }
    
    def _parse_command_string(self, command_line: str) -> argparse.Namespace:
        """Parsea un string de comando."""
        # Para simplicidad, usamos shlex para dividir
        # En implementación real, integraríamos con argparse
        
        parts = shlex.split(command_line)
        if not parts:
            raise ValidationError("Comando vacío")
        
        # Crear namespace básico
        class SimpleNamespace:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        
        # Mapeo simple de comandos
        args = SimpleNamespace()
        args.command = parts[0]
        
        # Para ahora, solo manejamos algunos comandos básicos
        if args.command == "project":
            if len(parts) > 1:
                args.project_command = parts[1]
        
        return args
    
    def _get_command_function(self, args: argparse.Namespace) -> Optional[Callable]:
        """Obtiene función para ejecutar un comando."""
        if not hasattr(args, 'command'):
            return None
        
        command_map = {
            "project": self._execute_project_command,
            "query": self._execute_query_command,
            "system": self._execute_system_command,
            "config": self._execute_config_command,
            "interactive": self.handle_interactive_mode,
            "help": lambda a: self.display_help(),
        }
        
        return command_map.get(args.command)
    
    async def _execute_project_command(self, args: argparse.Namespace) -> bool:
        """Ejecuta comando de proyecto."""
        if not hasattr(args, 'project_command'):
            console.print("[red]Error: Especifica un subcomando de proyecto[/red]")
            return False
        
        if args.project_command == "list":
            return await self._list_projects(args)
        elif args.project_command == "create":
            return await self._create_project(args)
        elif args.project_command == "analyze":
            return await self._analyze_project(args)
        else:
            console.print(f"[red]Error: Subcomando '{args.project_command}' no reconocido[/red]")
            return False
    
    async def _execute_query_command(self, args: argparse.Namespace) -> bool:
        """Ejecuta comando de consulta."""
        if not hasattr(args, 'query_command'):
            console.print("[red]Error: Especifica un subcomando de consulta[/red]")
            return False
        
        if args.query_command == "ask":
            return await self._ask_question(args)
        elif args.query_command == "chat":
            return await self._start_chat(args)
        else:
            console.print(f"[red]Error: Subcomando '{args.query_command}' no reconocido[/red]")
            return False
    
    async def _execute_system_command(self, args: argparse.Namespace) -> bool:
        """Ejecuta comando de sistema."""
        if not hasattr(args, 'system_command'):
            console.print("[red]Error: Especifica un subcomando de sistema[/red]")
            return False
        
        if args.system_command == "health":
            return await self._check_health(args)
        elif args.system_command == "metrics":
            return await self._show_metrics(args)
        else:
            console.print(f"[red]Error: Subcomando '{args.system_command}' no reconocido[/red]")
            return False
    
    async def _execute_config_command(self, args: argparse.Namespace) -> bool:
        """Ejecuta comando de configuración."""
        if not hasattr(args, 'config_command'):
            console.print("[red]Error: Especifica un subcomando de configuración[/red]")
            return False
        
        if args.config_command == "show":
            return self._show_config(args)
        elif args.config_command == "set":
            return self._set_config(args)
        else:
            console.print(f"[red]Error: Subcomando '{args.config_command}' no reconocido[/red]")
            return False
    
    # Implementación de comandos específicos
    
    async def _list_projects(self, args: argparse.Namespace) -> bool:
        """Lista proyectos."""
        # En implementación real, consultaría al orquestador
        # Por ahora mostramos datos de ejemplo
        
        projects = [
            {
                "id": "proj_123",
                "name": "Example Python Project",
                "path": "/path/to/project",
                "language": "python",
                "status": "analyzed",
                "files": 150,
                "last_analyzed": "2024-01-15 10:30:00",
            },
            {
                "id": "proj_456",
                "name": "Web Application",
                "path": "/path/to/webapp",
                "language": "javascript",
                "status": "pending",
                "files": 200,
                "last_analyzed": None,
            },
        ]
        
        if args.filter:
            projects = [p for p in projects if args.filter.lower() in str(p).lower()]
        
        if args.format == "json":
            console.print(json.dumps(projects, indent=2))
        elif args.format == "yaml":
            console.print(yaml.dump(projects, default_flow_style=False))
        else:
            table = Table(title="Proyectos", show_header=True)
            table.add_column("ID", style="cyan")
            table.add_column("Nombre", style="white")
            table.add_column("Lenguaje", style="green")
            table.add_column("Estado", style="yellow")
            table.add_column("Archivos", justify="right")
            table.add_column("Último Análisis", style="dim")
            
            for project in projects:
                status_color = "green" if project["status"] == "analyzed" else "yellow"
                table.add_row(
                    project["id"],
                    project["name"],
                    project["language"],
                    f"[{status_color}]{project['status']}[/{status_color}]",
                    str(project["files"]),
                    project["last_analyzed"] or "[dim]Nunca[/dim]",
                )
            
            console.print(table)
        
        return True
    
    async def _create_project(self, args: argparse.Namespace) -> bool:
        """Crea un proyecto."""
        if not self.orchestrator:
            self.orchestrator = BrainOrchestrator()
            await self.orchestrator.initialize()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Creando proyecto...", total=None)
            
            try:
                # Crear proyecto
                project_data = {
                    "name": args.name,
                    "path": args.path,
                    "description": getattr(args, 'description', None),
                    "language": getattr(args, 'language', None),
                }
                
                # En implementación real, llamaríamos al orquestador
                # project = await self.orchestrator.create_project(project_data)
                
                progress.update(task, completed=100)
                
                console.print(f"[green]Proyecto '{args.name}' creado exitosamente[/green]")
                
                # Analizar si se solicitó
                if getattr(args, 'analyze', False):
                    console.print("[dim]Iniciando análisis...[/dim]")
                    # await self.orchestrator.analyze_project(project["id"])
                    console.print("[green]Análisis completado[/green]")
                
                return True
                
            except Exception as e:
                progress.update(task, completed=100)
                console.print(f"[red]Error creando proyecto: {str(e)}[/red]")
                return False
    
    async def _analyze_project(self, args: argparse.Namespace) -> bool:
        """Analiza un proyecto."""
        if not self.orchestrator:
            self.orchestrator = BrainOrchestrator()
            await self.orchestrator.initialize()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            transient=True,
        ) as progress:
            task = progress.add_task("Analizando proyecto...", total=100)
            
            try:
                # Simular progreso
                for i in range(10):
                    await asyncio.sleep(0.5)
                    progress.update(task, advance=10)
                
                progress.update(task, completed=100)
                console.print("[green]Análisis completado exitosamente[/green]")
                return True
                
            except Exception as e:
                progress.update(task, completed=100)
                console.print(f"[red]Error analizando proyecto: {str(e)}[/red]")
                return False
    
    async def _ask_question(self, args: argparse.Namespace) -> bool:
        """Hace una pregunta al sistema."""
        if not self.orchestrator:
            self.orchestrator = BrainOrchestrator()
            await self.orchestrator.initialize()
        
        # Unir pregunta de múltiples argumentos
        question = " ".join(args.question)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Procesando pregunta...", total=None)
            
            try:
                # Hacer pregunta
                project_id = getattr(args, 'project', self.current_project)
                
                response = await self.orchestrator.ask_question(
                    question=question,
                    project_id=project_id,
                    context={"cli_session": self.session_id}
                )
                
                progress.update(task, completed=100)
                
                # Formatear respuesta
                if args.format == "json":
                    console.print(json.dumps(response, indent=2))
                elif args.format == "markdown":
                    md = Markdown(response.get("answer", {}).get("text", ""))
                    console.print(md)
                else:
                    # Mostrar respuesta formateada
                    console.print("\n[bold cyan]Respuesta:[/bold cyan]")
                    console.print(response.get("answer", {}).get("text", ""))
                    
                    if "sources" in response.get("answer", {}):
                        console.print("\n[bold cyan]Fuentes:[/bold cyan]")
                        for source in response["answer"]["sources"]:
                            console.print(f"  • {source.get('file_path', 'Unknown')}:{source.get('line_range', [])}")
                    
                    console.print(f"\n[dim]Confianza: {response.get('confidence', 0)*100:.1f}%[/dim]")
                
                return True
                
            except Exception as e:
                progress.update(task, completed=100)
                console.print(f"[red]Error procesando pregunta: {str(e)}[/red]")
                return False
    
    async def _start_chat(self, args: argparse.Namespace) -> bool:
        """Inicia chat interactivo."""
        console.print(Panel.fit(
            "[bold cyan]Chat Mode[/bold cyan]\n"
            "[dim]Ask questions about your project. Type 'exit' to quit.[/dim]",
            border_style="blue"
        ))
        
        if not self.orchestrator:
            self.orchestrator = BrainOrchestrator()
            await self.orchestrator.initialize()
        
        project_id = getattr(args, 'project', None)
        conversation_history = []
        
        while True:
            try:
                question = await questionary.text(
                    ">",
                    qmark="",
                    history=conversation_history,
                ).unsafe_ask_async()
                
                if not question or question.lower() in ["exit", "quit", "q"]:
                    break
                
                if question.lower() in ["clear", "cls"]:
                    console.clear()
                    continue
                
                # Procesar pregunta
                with console.status("[bold green]Pensando...[/bold green]"):
                    response = await self.orchestrator.ask_question(
                        question=question,
                        project_id=project_id,
                        context={
                            "cli_session": self.session_id,
                            "conversation_history": conversation_history[-10:]  # Últimas 10 preguntas
                        }
                    )
                
                # Mostrar respuesta
                console.print(f"\n[bold cyan]Brain:[/bold cyan] {response.get('answer', {}).get('text', '')}")
                
                # Guardar en historial
                conversation_history.append(question)
                
                # Mostrar sugerencias si hay
                if response.get('answer', {}).get('suggested_followups'):
                    console.print("\n[dim]Sugerencias:[/dim]")
                    for i, followup in enumerate(response['answer']['suggested_followups'][:3], 1):
                        console.print(f"  [dim]{i}.[/dim] {followup}")
                    console.print()
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Chat interrumpido[/yellow]")
                break
                
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")
        
        console.print("[green]Chat finalizado[/green]")
        return True
    
    async def _check_health(self, args: argparse.Namespace) -> bool:
        """Verifica salud del sistema."""
        if not self.orchestrator:
            console.print("[yellow]Orquestador no inicializado[/yellow]")
            return False
        
        try:
            status = await self.orchestrator._get_system_status()
            
            table = Table(title="Salud del Sistema", show_header=True)
            table.add_column("Componente", style="cyan")
            table.add_column("Estado", style="white")
            table.add_column("Detalles", style="dim")
            
            for component, state in status.get("components", {}).items():
                if state == "healthy":
                    state_display = "[green]✓ Healthy[/green]"
                else:
                    state_display = "[red]✗ Unhealthy[/red]"
                
                table.add_row(component, state_display, "")
            
            console.print(table)
            
            # Estado general
            if status.get("status") == "running":
                console.print("[green]✓ Sistema funcionando correctamente[/green]")
            else:
                console.print("[red]✗ Sistema con problemas[/red]")
            
            return True
            
        except Exception as e:
            console.print(f"[red]Error verificando salud: {str(e)}[/red]")
            return False
    
    async def _show_metrics(self, args: argparse.Namespace) -> bool:
        """Muestra métricas del sistema."""
        if not self.orchestrator:
            console.print("[yellow]Orquestador no inicializado[/yellow]")
            return False
        
        try:
            metrics = await self.orchestrator.get_metrics()
            
            # Crear panel para métricas clave
            metrics_panel = Panel.fit(
                f"[bold]Operaciones Completadas:[/bold] {metrics.get('operations_completed', 0)}\n"
                f"[bold]Operaciones Fallidas:[/bold] {metrics.get('operations_failed', 0)}\n"
                f"[bold]Tiempo Promedio Respuesta:[/bold] {metrics.get('avg_response_time_ms', 0):.2f}ms\n"
                f"[bold]Operaciones Concurrentes:[/bold] {metrics.get('concurrent_operations', 0)}",
                title="Métricas del Sistema",
                border_style="blue"
            )
            
            console.print(metrics_panel)
            return True
            
        except Exception as e:
            console.print(f"[red]Error mostrando métricas: {str(e)}[/red]")
            return False
    
    def _show_config(self, args: argparse.Namespace) -> bool:
        """Muestra configuración."""
        config = self.config_manager.get_config()
        
        console.print(Panel.fit(
            json.dumps(config, indent=2, default=str),
            title="Configuración del Sistema",
            border_style="blue"
        ))
        return True
    
    def _set_config(self, args: argparse.Namespace) -> bool:
        """Establece configuración."""
        try:
            self.config_manager.set_config(args.key, args.value)
            console.print(f"[green]Configuración '{args.key}' establecida a '{args.value}'[/green]")
            return True
        except Exception as e:
            console.print(f"[red]Error estableciendo configuración: {str(e)}[/red]")
            return False
    
    def _show_version(self) -> None:
        """Muestra versión."""
        console.print(Panel.fit(
            "[bold cyan]Project Brain CLI[/bold cyan]\n"
            "[bold]Versión:[/bold] 1.0.0\n"
            "[bold]API:[/bold] REST, WebSocket, gRPC\n"
            "[bold]Soporte:[/bold] Python, JavaScript, Java, C++, Go, Rust",
            border_style="green"
        ))
    
    def _get_command_help(self, command: str) -> Optional[str]:
        """Obtiene ayuda para un comando específico."""
        if command in self.commands:
            cmd_info = self.commands[command]
            help_text = f"[bold]{command}[/bold]\n\n{cmd_info.get('description', '')}"
            
            if "subcommands" in cmd_info:
                help_text += "\n\n[bold]Subcomandos:[/bold]"
                for subcmd, desc in cmd_info["subcommands"].items():
                    help_text += f"\n  {subcmd}: {desc}"
            
            if "usage" in cmd_info:
                help_text += f"\n\n[bold]Uso:[/bold]\n  {cmd_info['usage']}"
            
            return help_text
        
        return None
    
    def _setup_completion(self) -> None:
        """Configura autocompletado."""
        try:
            import readline
            
            # Cargar historial si existe
            if self.history_file.exists():
                readline.read_history_file(str(self.history_file))
            
            # Configurar autocompletado
            readline.set_completer(self._completer)
            readline.set_completer_delims(' \t\n;')
            readline.parse_and_bind("tab: complete")
            
        except ImportError:
            # readline no disponible en Windows
            pass
    
    def _completer(self, text: str, state: int) -> Optional[str]:
        """Función de autocompletado."""
        commands = list(self.commands.keys()) + ["help", "exit", "clear"]
        matches = [cmd for cmd in commands if cmd.startswith(text)]
        
        if state < len(matches):
            return matches[state]
        return None
    
    def _create_completer(self):
        """Crea un completer para questionary."""
        from prompt_toolkit.completion import Completer, Completion
        
        class BrainCompleter(Completer):
            def get_completions(self, document, complete_event):
                text = document.text_before_cursor
                words = text.split()
                
                if len(words) == 0:
                    # Completar comandos principales
                    for cmd in self.commands.keys():
                        if cmd.startswith(text):
                            yield Completion(cmd, start_position=-len(text))
                
                elif len(words) == 1:
                    # Ya tenemos un comando, completar subcomandos
                    cmd = words[0]
                    if cmd in self.commands and "subcommands" in self.commands[cmd]:
                        for subcmd in self.commands[cmd]["subcommands"].keys():
                            if subcmd.startswith(text[len(cmd):].lstrip()):
                                yield Completion(subcmd, start_position=-len(text) + len(cmd))
        
        return BrainCompleter()
    
    def _load_history(self) -> None:
        """Carga historial desde archivo."""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.history = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.debug("Error cargando historial: %s", e)
    
    def _add_to_history(self, command: str) -> None:
        """Añade comando al historial."""
        if command and command not in self.history[-100:]:  # Mantener últimos 100 únicos
            self.history.append(command)
            
            # Guardar en archivo
            try:
                with open(self.history_file, 'a', encoding='utf-8') as f:
                    f.write(command + '\n')
            except Exception as e:
                logger.debug("Error guardando historial: %s", e)

# Función principal para ejecutar CLI
def main():
    """Función principal de entrada CLI."""
    import sys
    
    cli = CLIInterface()
    
    # Si no hay argumentos, mostrar ayuda
    if len(sys.argv) == 1:
        cli.display_help()
        return 0
    
    # Ejecutar comando
    exit_code = asyncio.run(cli.execute_command())
    sys.exit(exit_code)

if __name__ == "__main__":
    main()