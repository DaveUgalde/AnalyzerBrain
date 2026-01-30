"""
Web UI - Interfaz web para Project Brain usando Streamlit.
Proporciona una interfaz gráfica completa para interactuar con el sistema.
"""

import logging
import streamlit as st
import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from ..core.exceptions import BrainException
from ..core.orchestrator import BrainOrchestrator
from ..core.config_manager import ConfigManager
from .authentication import Authentication

logger = logging.getLogger(__name__)

# Configurar página de Streamlit
def setup_page_config():
    """Configura la página de Streamlit."""
    st.set_page_config(
        page_title="Project Brain",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/project-brain',
            'Report a bug': 'https://github.com/project-brain/issues',
            'About': '# Project Brain - Intelligent Code Analysis System'
        }
    )

class WebUI:
    """
    Interfaz web completa para Project Brain.
    
    Características:
    1. Dashboard principal con métricas
    2. Gestión de proyectos
    3. Consultas interactivas
    4. Visualización de análisis
    5. Configuración del sistema
    6. Autenticación de usuarios
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa la interfaz web.
        
        Args:
            config_path: Ruta al archivo de configuración (opcional)
        """
        self.config_manager = ConfigManager()
        if config_path:
            self.config_manager.load_config(config_path)
        
        self.config = self.config_manager.get_config().get("web_ui", {})
        self.orchestrator: Optional[BrainOrchestrator] = None
        self.authentication: Optional[Authentication] = None
        
        # Estado de sesión
        self.session_state = st.session_state
        self._initialize_session_state()
        
        # Configurar página
        setup_page_config()
        
        logger.info("WebUI inicializada")
    
    def _initialize_session_state(self):
        """Inicializa el estado de sesión de Streamlit."""
        if 'initialized' not in self.session_state:
            self.session_state.initialized = True
            self.session_state.current_project = None
            self.session_state.projects = []
            self.session_state.conversation_history = []
            self.session_state.analysis_results = {}
            self.session_state.user_authenticated = False
            self.session_state.user_info = {}
            self.session_state.sidebar_expanded = True
    
    def render_interface(self):
        """
        Renderiza la interfaz web completa.
        """
        # Barra lateral
        self._render_sidebar()
        
        # Contenido principal basado en navegación
        if self.session_state.get('current_page', 'dashboard') == 'dashboard':
            self._render_dashboard()
        elif self.session_state.current_page == 'projects':
            self._render_projects_page()
        elif self.session_state.current_page == 'query':
            self._render_query_page()
        elif self.session_state.current_page == 'analysis':
            self._render_analysis_page()
        elif self.session_state.current_page == 'knowledge':
            self._render_knowledge_page()
        elif self.session_state.current_page == 'settings':
            self._render_settings_page()
    
    def handle_user_interaction(self):
        """
        Maneja las interacciones del usuario.
        Esta función se llama en cada recarga de la página.
        """
        # Verificar si hay acciones pendientes
        if 'action' in self.session_state:
            action = self.session_state.action
            del self.session_state.action
            
            if action == 'analyze_project':
                self._handle_analyze_project()
            elif action == 'ask_question':
                self._handle_ask_question()
            elif action == 'create_project':
                self._handle_create_project()
    
    def update_ui_state(self):
        """
        Actualiza el estado de la UI basado en cambios en el sistema.
        """
        # Actualizar lista de proyectos si es necesario
        if 'refresh_projects' in self.session_state:
            self._load_projects()
            del self.session_state.refresh_projects
        
        # Actualizar métricas del sistema
        if 'refresh_metrics' in self.session_state:
            self.session_state.system_metrics = self._get_system_metrics()
            del self.session_state.refresh_metrics
    
    def manage_sessions(self):
        """
        Gestiona sesiones de usuario.
        """
        # Implementar gestión de sesiones
        # Por ahora, solo verificamos autenticación básica
        
        if not self.session_state.user_authenticated:
            self._render_login_page()
            return False
        
        return True
    
    def stream_updates(self):
        """
        Maneja actualizaciones en tiempo real.
        """
        # Placeholder para actualizaciones en tiempo real
        # En una implementación real, usaríamos WebSockets
        
        if 'stream_updates' in self.session_state:
            # Mostrar notificaciones de actualizaciones
            pass
    
    def optimize_ui_performance(self):
        """
        Optimiza el rendimiento de la UI.
        """
        # Implementar optimizaciones como:
        # - Caché de datos
        # - Lazy loading
        # - Debouncing de actualizaciones
        
        # Por ahora, solo un placeholder
        pass
    
    def generate_ui_analytics(self) -> Dict[str, Any]:
        """
        Genera analíticas de uso de la UI.
        
        Returns:
            Dict con métricas de uso
        """
        return {
            "page_views": self.session_state.get('page_views', {}),
            "user_actions": self.session_state.get('user_actions', []),
            "session_duration": self._calculate_session_duration(),
            "timestamp": datetime.now().isoformat(),
        }
    
    # Métodos de renderizado
    
    def _render_sidebar(self):
        """Renderiza la barra lateral."""
        with st.sidebar:
            st.title("🧠 Project Brain")
            
            # Menú de navegación
            st.subheader("Navegación")
            
            menu_options = {
                "📊 Dashboard": "dashboard",
                "📁 Proyectos": "projects",
                "💬 Consultas": "query",
                "🔍 Análisis": "analysis",
                "🧠 Conocimiento": "knowledge",
                "⚙️ Configuración": "settings",
            }
            
            for label, page in menu_options.items():
                if st.button(label, key=f"nav_{page}", use_container_width=True):
                    self.session_state.current_page = page
            
            st.divider()
            
            # Información del proyecto actual
            if self.session_state.current_project:
                st.subheader("Proyecto Actual")
                st.info(f"**{self.session_state.current_project.get('name', 'Sin nombre')}**")
                
                if st.button("Cambiar Proyecto", use_container_width=True):
                    self.session_state.current_project = None
            
            # Información del sistema
            st.divider()
            st.subheader("Sistema")
            
            if self.orchestrator:
                status = "🟢 En línea"  # Simulado
                st.markdown(f"Estado: {status}")
            
            # Versión
            st.caption(f"v1.0.0 · {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    def _render_dashboard(self):
        """Renderiza el dashboard principal."""
        st.title("Dashboard de Project Brain")
        
        # Métricas rápidas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Proyectos",
                value=len(self.session_state.projects),
                delta="+2" if len(self.session_state.projects) > 0 else "0"
            )
        
        with col2:
            st.metric(
                label="Consultas Hoy",
                value=self._get_today_queries(),
                delta="+5"
            )
        
        with col3:
            st.metric(
                label="Análisis Activos",
                value=self._get_active_analyses(),
                delta="-1"
            )
        
        with col4:
            st.metric(
                label="Confianza Promedio",
                value=f"{self._get_avg_confidence():.1f}%",
                delta="+2.3%"
            )
        
        # Gráficos
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_projects_chart()
        
        with col2:
            self._render_queries_chart()
        
        # Actividad reciente
        st.subheader("Actividad Reciente")
        self._render_recent_activity()
        
        # Proyectos destacados
        st.subheader("Proyectos Destacados")
        self._render_featured_projects()
    
    def _render_projects_page(self):
        """Renderiza la página de proyectos."""
        st.title("Gestión de Proyectos")
        
        # Barra de acciones
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search_query = st.text_input("Buscar proyectos...", placeholder="Nombre, lenguaje, etc.")
        
        with col2:
            if st.button("🔄 Refrescar", use_container_width=True):
                self.session_state.refresh_projects = True
        
        with col3:
            if st.button("➕ Nuevo Proyecto", type="primary", use_container_width=True):
                self._show_new_project_modal()
        
        # Lista de proyectos
        if self.session_state.projects:
            self._render_projects_table(search_query)
        else:
            st.info("No hay proyectos aún. ¡Crea tu primer proyecto!")
            
            if st.button("Comenzar con un proyecto de ejemplo", type="secondary"):
                self._create_example_project()
    
    def _render_query_page(self):
        """Renderiza la página de consultas."""
        st.title("Consulta Inteligente")
        
        # Selección de proyecto
        col1, col2 = st.columns([3, 1])
        
        with col1:
            project_options = [p['name'] for p in self.session_state.projects]
            if project_options:
                selected_project = st.selectbox(
                    "Proyecto",
                    options=project_options,
                    index=0,
                    help="Selecciona el proyecto sobre el que quieres consultar"
                )
                self.session_state.selected_project = selected_project
            else:
                st.warning("No hay proyectos disponibles. Crea uno primero.")
        
        with col2:
            st.write("")  # Espaciador
            if st.button("📝 Ver historial", use_container_width=True):
                self._show_conversation_history()
        
        # Área de consulta
        st.subheader("Haz tu pregunta")
        
        question = st.text_area(
            "Describe lo que quieres saber sobre el código...",
            height=100,
            placeholder="Ej: ¿Qué hace la función process_data? ¿Dónde se define calculate_stats? ¿Hay algún patrón Singleton en el proyecto?",
            key="question_input"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            detail_level = st.select_slider(
                "Nivel de detalle",
                options=["Breve", "Normal", "Detallado"],
                value="Normal"
            )
        
        with col2:
            include_code = st.checkbox("Incluir código", value=True)
        
        with col3:
            include_sources = st.checkbox("Incluir fuentes", value=True)
        
        # Botón de enviar
        if st.button("🧠 Analizar y Responder", type="primary", disabled=not question):
            if question:
                self.session_state.question_context = {
                    "question": question,
                    "detail_level": detail_level,
                    "include_code": include_code,
                    "include_sources": include_sources,
                    "project": self.session_state.selected_project
                }
                self.session_state.action = "ask_question"
        
        # Mostrar respuesta si existe
        if 'last_response' in self.session_state:
            self._render_response(self.session_state.last_response)
    
    def _render_analysis_page(self):
        """Renderiza la página de análisis."""
        st.title("Análisis de Código")
        
        if not self.session_state.current_project:
            st.warning("Selecciona un proyecto para ver análisis")
            return
        
        # Pestañas de análisis
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Métricas", 
            "🔍 Issues", 
            "🎯 Patrones", 
            "📈 Evolución",
            "🛡️ Seguridad"
        ])
        
        with tab1:
            self._render_metrics_analysis()
        
        with tab2:
            self._render_issues_analysis()
        
        with tab3:
            self._render_patterns_analysis()
        
        with tab4:
            self._render_evolution_analysis()
        
        with tab5:
            self._render_security_analysis()
    
    def _render_knowledge_page(self):
        """Renderiza la página de conocimiento."""
        st.title("Base de Conocimiento")
        
        # Búsqueda en conocimiento
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_term = st.text_input(
                "Buscar en conocimiento...",
                placeholder="Términos, conceptos, nombres de funciones..."
            )
        
        with col2:
            search_type = st.selectbox(
                "Tipo",
                ["Semántica", "Keyword", "Híbrido"]
            )
        
        # Resultados de búsqueda
        if search_term:
            results = self._search_knowledge(search_term, search_type)
            self._render_knowledge_results(results)
        else:
            # Vista general del grafo de conocimiento
            self._render_knowledge_overview()
    
    def _render_settings_page(self):
        """Renderiza la página de configuración."""
        st.title("Configuración del Sistema")
        
        # Pestañas de configuración
        tab1, tab2, tab3, tab4 = st.tabs([
            "⚙️ General", 
            "🔐 Seguridad", 
            "📊 Monitoreo",
            "🔄 Integraciones"
        ])
        
        with tab1:
            self._render_general_settings()
        
        with tab2:
            self._render_security_settings()
        
        with tab3:
            self._render_monitoring_settings()
        
        with tab4:
            self._render_integration_settings()
    
    def _render_login_page(self):
        """Renderiza la página de login."""
        st.title("Project Brain - Inicio de Sesión")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            with st.form("login_form"):
                st.subheader("Acceso al Sistema")
                
                username = st.text_input("Usuario")
                password = st.text_input("Contraseña", type="password")
                
                col1, col2 = st.columns(2)
                with col1:
                    remember = st.checkbox("Recordarme")
                with col2:
                    st.write("")  # Espaciador
                
                if st.form_submit_button("Iniciar Sesión", type="primary"):
                    if self._authenticate_user(username, password):
                        self.session_state.user_authenticated = True
                        st.success("¡Inicio de sesión exitoso!")
                        st.rerun()
                    else:
                        st.error("Credenciales incorrectas")
            
            st.divider()
            st.caption("¿Primera vez? Contacta al administrador para crear una cuenta.")
    
    # Métodos de renderizado específicos
    
    def _render_projects_chart(self):
        """Renderiza gráfico de proyectos por lenguaje."""
        if not self.session_state.projects:
            st.info("No hay proyectos para mostrar")
            return
        
        # Agrupar por lenguaje
        language_counts = {}
        for project in self.session_state.projects:
            lang = project.get('language', 'Desconocido')
            language_counts[lang] = language_counts.get(lang, 0) + 1
        
        # Crear gráfico
        fig = go.Figure(data=[
            go.Pie(
                labels=list(language_counts.keys()),
                values=list(language_counts.values()),
                hole=.3
            )
        ])
        
        fig.update_layout(
            title="Proyectos por Lenguaje",
            showlegend=True,
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_queries_chart(self):
        """Renderiza gráfico de consultas por tipo."""
        # Datos de ejemplo
        query_types = ['Código', 'Arquitectura', 'Seguridad', 'Performance', 'Otros']
        query_counts = [45, 23, 18, 12, 7]
        
        fig = go.Figure(data=[
            go.Bar(
                x=query_types,
                y=query_counts,
                marker_color='lightblue'
            )
        ])
        
        fig.update_layout(
            title="Consultas por Tipo (última semana)",
            xaxis_title="Tipo de Consulta",
            yaxis_title="Número de Consultas",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_recent_activity(self):
        """Renderiza actividad reciente."""
        activities = [
            {"time": "Hace 5 min", "user": "admin", "action": "Analizó proyecto 'web-app'"},
            {"time": "Hace 15 min", "user": "dev1", "action": "Preguntó sobre función 'authenticate'"},
            {"time": "Hace 30 min", "user": "admin", "action": "Creó proyecto 'mobile-api'"},
            {"time": "Hace 2 horas", "user": "dev2", "action": "Encontró 3 issues críticos"},
            {"time": "Hace 4 horas", "user": "dev1", "action": "Exportó conocimiento a JSON"},
        ]
        
        for activity in activities:
            with st.container():
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.caption(activity['time'])
                with col2:
                    st.write(f"**{activity['user']}** {activity['action']}")
                st.divider()
    
    def _render_featured_projects(self):
        """Renderiza proyectos destacados."""
        if not self.session_state.projects:
            return
        
        # Tomar primeros 3 proyectos como destacados
        featured = self.session_state.projects[:3]
        
        for project in featured:
            with st.expander(f"📁 {project.get('name', 'Sin nombre')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Archivos", project.get('file_count', 0))
                    st.metric("Issues", project.get('issue_count', 0))
                
                with col2:
                    st.metric("Complejidad", f"{project.get('avg_complexity', 0):.1f}")
                    st.metric("Mantenibilidad", f"{project.get('maintainability', 0):.0f}/100")
                
                if st.button("Analizar", key=f"analyze_{project.get('id')}"):
                    self.session_state.current_project = project
                    self.session_state.current_page = 'analysis'
    
    def _render_projects_table(self, search_query: str = ""):
        """Renderiza tabla de proyectos."""
        # Filtrar proyectos si hay búsqueda
        projects = self.session_state.projects
        if search_query:
            projects = [
                p for p in projects
                if search_query.lower() in str(p).lower()
            ]
        
        # Crear DataFrame para la tabla
        df_data = []
        for project in projects:
            df_data.append({
                'Nombre': project.get('name', ''),
                'Lenguaje': project.get('language', ''),
                'Archivos': project.get('file_count', 0),
                'Issues': project.get('issue_count', 0),
                'Último Análisis': project.get('last_analyzed', 'Nunca'),
                'Estado': project.get('status', 'Desconocido'),
            })
        
        if df_data:
            df = pd.DataFrame(df_data)
            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    'Estado': st.column_config.ProgressColumn(
                        'Estado',
                        help="Estado del análisis",
                        format="%f",
                        min_value=0,
                        max_value=100,
                    ),
                },
                hide_index=True,
            )
        else:
            st.info("No se encontraron proyectos que coincidan con la búsqueda")
    
    def _render_response(self, response: Dict[str, Any]):
        """Renderiza una respuesta del sistema."""
        st.subheader("📝 Respuesta")
        
        # Mostrar confianza
        confidence = response.get('confidence', 0) * 100
        if confidence > 80:
            confidence_color = "green"
        elif confidence > 60:
            confidence_color = "orange"
        else:
            confidence_color = "red"
        
        st.markdown(f"**Confianza:** :{confidence_color}[{confidence:.1f}%]")
        
        # Mostrar respuesta principal
        answer = response.get('answer', {})
        if 'text' in answer:
            st.markdown("### 📋 Resumen")
            st.markdown(answer['text'])
        
        # Mostrar código si está disponible
        if 'code_examples' in answer and answer['code_examples']:
            st.markdown("### 💻 Ejemplos de Código")
            for example in answer['code_examples'][:2]:  # Máximo 2 ejemplos
                with st.expander(f"Ejemplo: {example.get('description', 'Código')}"):
                    st.code(example.get('code', ''), language=example.get('language', 'python'))
        
        # Mostrar fuentes
        if 'sources' in answer and answer['sources']:
            st.markdown("### 📚 Fuentes")
            for source in answer['sources'][:3]:  # Máximo 3 fuentes
                source_type = source.get('type', 'desconocido')
                if source_type == 'code':
                    st.markdown(f"**{source.get('file_path', 'Archivo')}** (líneas {source.get('line_range', [])})")
                    if 'excerpt' in source:
                        with st.expander("Ver código fuente"):
                            st.code(source['excerpt'], language='python')
                else:
                    st.markdown(f"**{source.get('title', 'Fuente')}** - {source.get('type', '')}")
        
        # Mostrar razonamiento si está disponible
        if 'reasoning_chain' in response and response['reasoning_chain']:
            with st.expander("🧠 Proceso de razonamiento"):
                for i, step in enumerate(response['reasoning_chain'], 1):
                    st.markdown(f"{i}. {step}")
        
        # Sugerencias de seguimiento
        if 'suggested_followups' in answer and answer['suggested_followups']:
            st.markdown("### 🤔 ¿Qué más quieres saber?")
            cols = st.columns(2)
            for i, followup in enumerate(answer['suggested_followups'][:4]):
                with cols[i % 2]:
                    if st.button(followup, use_container_width=True):
                        st.session_state.question_input = followup
                        st.rerun()
    
    def _render_metrics_analysis(self):
        """Renderiza análisis de métricas."""
        if not self.session_state.current_project:
            return
        
        project = self.session_state.current_project
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Complejidad Promedio", f"{project.get('avg_complexity', 0):.1f}")
        
        with col2:
            st.metric("Mantenibilidad", f"{project.get('maintainability', 0):.0f}/100")
        
        with col3:
            st.metric("Líneas de Código", project.get('total_lines', 0))
        
        with col4:
            st.metric("Cobertura de Tests", f"{project.get('test_coverage', 0):.0f}%")
        
        # Gráficos de distribución
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribución de complejidad
            complexity_data = project.get('complexity_distribution', [1, 3, 5, 2, 1])
            fig = go.Figure(data=[
                go.Histogram(
                    x=complexity_data,
                    nbinsx=10,
                    marker_color='coral'
                )
            ])
            fig.update_layout(
                title="Distribución de Complejidad",
                xaxis_title="Complejidad Ciclomática",
                yaxis_title="Número de Funciones",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Distribución de tamaño de archivos
            file_sizes = project.get('file_size_distribution', [100, 200, 150, 300, 250])
            fig = go.Figure(data=[
                go.Box(
                    y=file_sizes,
                    name="Tamaño de archivos (líneas)",
                    boxpoints='all',
                    marker_color='lightblue'
                )
            ])
            fig.update_layout(
                title="Distribución de Tamaño de Archivos",
                yaxis_title="Líneas de Código",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_issues_analysis(self):
        """Renderiza análisis de issues."""
        if not self.session_state.current_project:
            return
        
        # Issues por severidad
        issues_by_severity = {
            'Crítico': 3,
            'Alto': 7,
            'Medio': 12,
            'Bajo': 20,
            'Info': 15
        }
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            for severity, count in issues_by_severity.items():
                color = 'red' if severity == 'Crítico' else 'orange' if severity == 'Alto' else 'yellow'
                st.metric(f"Issues {severity}", count, delta_color="inverse")
        
        with col2:
            fig = go.Figure(data=[
                go.Pie(
                    labels=list(issues_by_severity.keys()),
                    values=list(issues_by_severity.values()),
                    hole=.3,
                    marker_colors=['red', 'orange', 'yellow', 'lightblue', 'lightgreen']
                )
            ])
            fig.update_layout(
                title="Issues por Severidad",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Lista de issues críticos/altos
        st.subheader("Issues Prioritarios")
        
        critical_issues = [
            {"id": 1, "description": "Inyección SQL posible en línea 45", "file": "database.py", "severity": "Crítico"},
            {"id": 2, "description": "Hardcoded password en línea 23", "file": "config.py", "severity": "Crítico"},
            {"id": 3, "description": "Memory leak en función process_data", "file": "utils.py", "severity": "Alto"},
        ]
        
        for issue in critical_issues:
            with st.expander(f"🚨 {issue['description']}"):
                st.markdown(f"**Archivo:** {issue['file']}")
                st.markdown(f"**Severidad:** {issue['severity']}")
                if st.button("Ver sugerencia de fix", key=f"fix_{issue['id']}"):
                    st.info("Sugerencia de fix aparecería aquí...")
    
    # Métodos de manejo de eventos
    
    def _handle_analyze_project(self):
        """Maneja análisis de proyecto."""
        project_id = self.session_state.get('project_to_analyze')
        if not project_id:
            return
        
        try:
            with st.spinner("Analizando proyecto..."):
                # Simular análisis
                import time
                time.sleep(2)
                
                # Actualizar proyecto
                for project in self.session_state.projects:
                    if project.get('id') == project_id:
                        project['status'] = 'analyzed'
                        project['last_analyzed'] = datetime.now().strftime("%Y-%m-%d %H:%M")
                        break
                
                st.success("¡Análisis completado!")
                self.session_state.refresh_projects = True
        
        except Exception as e:
            st.error(f"Error analizando proyecto: {str(e)}")
    
    def _handle_ask_question(self):
        """Maneja pregunta del usuario."""
        context = self.session_state.get('question_context')
        if not context:
            return
        
        try:
            with st.spinner("Procesando pregunta..."):
                # Simular procesamiento
                import time
                time.sleep(1)
                
                # Respuesta de ejemplo
                response = {
                    "answer": {
                        "text": f"Basado en mi análisis del proyecto '{context['project']}', encontré que la función mencionada se define en el archivo `utils.py` entre las líneas 45-78. Esta función procesa datos de entrada aplicando transformaciones específicas del dominio.",
                        "sources": [
                            {
                                "type": "code",
                                "file_path": "src/utils.py",
                                "line_range": [45, 78],
                                "excerpt": "def process_data(input_data, config=None):\n    \"\"\"Procesa datos con configuración opcional.\"\"\"\n    if config:\n        return apply_transformations(input_data, config)\n    return standard_transform(input_data)",
                                "confidence": 0.95
                            }
                        ],
                        "code_examples": [
                            {
                                "code": "result = process_data(raw_data, config=transformation_config)\nprint(f\"Datos procesados: {result}\")",
                                "language": "python",
                                "description": "Ejemplo de uso"
                            }
                        ],
                        "suggested_followups": [
                            "¿Qué parámetros acepta esta función?",
                            "¿Dónde se llama a esta función?",
                            "¿Hay funciones similares en el proyecto?"
                        ]
                    },
                    "confidence": 0.88,
                    "reasoning_chain": [
                        "Identificada referencia a función 'process_data'",
                        "Buscada en base de conocimiento del proyecto",
                        "Encontrada definición en utils.py",
                        "Analizado contexto de uso",
                        "Generada respuesta con ejemplos"
                    ]
                }
                
                self.session_state.last_response = response
        
        except Exception as e:
            st.error(f"Error procesando pregunta: {str(e)}")
    
    def _handle_create_project(self):
        """Maneja creación de proyecto."""
        project_data = self.session_state.get('new_project_data')
        if not project_data:
            return
        
        try:
            with st.spinner("Creando proyecto..."):
                # Simular creación
                import time
                time.sleep(1)
                
                # Añadir proyecto a la lista
                new_project = {
                    "id": f"proj_{len(self.session_state.projects) + 1}",
                    "name": project_data.get('name', 'Nuevo Proyecto'),
                    "path": project_data.get('path', ''),
                    "language": project_data.get('language', 'python'),
                    "file_count": 0,
                    "issue_count": 0,
                    "status": "pending",
                    "created": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                
                self.session_state.projects.append(new_project)
                st.success(f"¡Proyecto '{new_project['name']}' creado exitosamente!")
                
                # Limpiar datos temporales
                del self.session_state.new_project_data
                self.session_state.refresh_projects = True
        
        except Exception as e:
            st.error(f"Error creando proyecto: {str(e)}")
    
    # Métodos auxiliares
    
    def _show_new_project_modal(self):
        """Muestra modal para nuevo proyecto."""
        with st.form("new_project_form"):
            st.subheader("Nuevo Proyecto")
            
            name = st.text_input("Nombre del Proyecto*")
            path = st.text_input("Ruta del Proyecto*")
            description = st.text_area("Descripción")
            language = st.selectbox(
                "Lenguaje Principal",
                ["python", "javascript", "typescript", "java", "cpp", "go", "rust", "otros"]
            )
            
            col1, col2 = st.columns(2)
            with col1:
                analyze_now = st.checkbox("Analizar inmediatamente")
            with col2:
                st.write("")  # Espaciador
            
            if st.form_submit_button("Crear Proyecto", type="primary"):
                if name and path:
                    self.session_state.new_project_data = {
                        "name": name,
                        "path": path,
                        "description": description,
                        "language": language,
                        "analyze_now": analyze_now
                    }
                    self.session_state.action = "create_project"
                    st.rerun()
                else:
                    st.error("Nombre y ruta son requeridos")
    
    def _show_conversation_history(self):
        """Muestra historial de conversación."""
        if not self.session_state.conversation_history:
            st.info("No hay historial de conversación")
            return
        
        with st.expander("📜 Historial de Conversación", expanded=True):
            for entry in self.session_state.conversation_history[-10:]:  # Últimas 10 entradas
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.caption(entry.get('timestamp', ''))
                with col2:
                    if entry.get('type') == 'question':
                        st.markdown(f"**Tú:** {entry.get('content', '')}")
                    else:
                        st.markdown(f"**Brain:** {entry.get('content', '')}")
                st.divider()
    
    def _create_example_project(self):
        """Crea un proyecto de ejemplo."""
        example_project = {
            "id": "proj_example",
            "name": "Proyecto de Ejemplo",
            "path": "/path/to/example",
            "language": "python",
            "file_count": 25,
            "issue_count": 3,
            "avg_complexity": 2.5,
            "maintainability": 85,
            "status": "analyzed",
            "last_analyzed": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "created": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        
        self.session_state.projects.append(example_project)
        st.success("¡Proyecto de ejemplo creado!")
        self.session_state.refresh_projects = True
    
    def _load_projects(self):
        """Carga proyectos desde el sistema."""
        # En implementación real, esto cargaría desde el orquestador
        # Por ahora, usamos datos de ejemplo
        
        if not self.session_state.projects:
            self.session_state.projects = [
                {
                    "id": "proj_1",
                    "name": "API Backend",
                    "path": "/projects/api-backend",
                    "language": "python",
                    "file_count": 150,
                    "issue_count": 12,
                    "avg_complexity": 3.2,
                    "maintainability": 78,
                    "status": "analyzed",
                    "last_analyzed": "2024-01-15 14:30:00",
                    "created": "2024-01-10 09:00:00"
                },
                {
                    "id": "proj_2",
                    "name": "Frontend Web",
                    "path": "/projects/frontend",
                    "language": "typescript",
                    "file_count": 200,
                    "issue_count": 8,
                    "avg_complexity": 2.8,
                    "maintainability": 82,
                    "status": "analyzed",
                    "last_analyzed": "2024-01-14 16:45:00",
                    "created": "2024-01-05 11:30:00"
                }
            ]
    
    def _search_knowledge(self, term: str, search_type: str) -> List[Dict]:
        """Busca en la base de conocimiento."""
        # Simular resultados de búsqueda
        return [
            {
                "id": "ent_1",
                "type": "function",
                "name": "process_data",
                "score": 0.95,
                "content": "Función que procesa datos de entrada con transformaciones configurables.",
                "metadata": {
                    "file": "src/utils.py",
                    "lines": [45, 78],
                    "language": "python"
                }
            },
            {
                "id": "ent_2",
                "type": "class",
                "name": "DataProcessor",
                "score": 0.87,
                "content": "Clase principal para procesamiento de datos con múltiples estrategias.",
                "metadata": {
                    "file": "src/processing.py",
                    "lines": [120, 250],
                    "language": "python"
                }
            }
        ]
    
    def _authenticate_user(self, username: str, password: str) -> bool:
        """Autentica usuario."""
        # En implementación real, usaría el sistema de autenticación
        # Por ahora, credenciales simples
        return username == "admin" and password == "admin"
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas del sistema."""
        return {
            "uptime": "5 días, 3 horas",
            "memory_usage": "65%",
            "cpu_usage": "42%",
            "active_projects": len(self.session_state.projects),
            "queries_today": 24,
        }
    
    def _get_today_queries(self) -> int:
        """Obtiene número de consultas hoy."""
        return 24  # Simulado
    
    def _get_active_analyses(self) -> int:
        """Obtiene número de análisis activos."""
        return 1  # Simulado
    
    def _get_avg_confidence(self) -> float:
        """Obtiene confianza promedio."""
        return 85.5  # Simulado
    
    def _calculate_session_duration(self) -> str:
        """Calcula duración de sesión."""
        return "30 minutos"  # Simulado
    
    # Métodos para renderizar otras secciones (simplificados)
    
    def _render_patterns_analysis(self):
        st.info("Análisis de patrones - En construcción")
    
    def _render_evolution_analysis(self):
        st.info("Análisis de evolución - En construcción")
    
    def _render_security_analysis(self):
        st.info("Análisis de seguridad - En construcción")
    
    def _render_knowledge_overview(self):
        st.info("Vista general del conocimiento - En construcción")
    
    def _render_knowledge_results(self, results: List[Dict]):
        for result in results:
            with st.expander(f"{result['type']}: {result['name']} (score: {result['score']:.2f})"):
                st.write(result['content'])
                if 'metadata' in result:
                    st.json(result['metadata'])
    
    def _render_general_settings(self):
        st.info("Configuración general - En construcción")
    
    def _render_security_settings(self):
        st.info("Configuración de seguridad - En construcción")
    
    def _render_monitoring_settings(self):
        st.info("Configuración de monitoreo - En construcción")
    
    def _render_integration_settings(self):
        st.info("Configuración de integraciones - En construcción")

# Función principal para ejecutar la UI web
def run_web_ui(config_path: Optional[str] = None):
    """
    Función principal para ejecutar la interfaz web.
    
    Args:
        config_path: Ruta al archivo de configuración (opcional)
    """
    # Crear instancia de WebUI
    ui = WebUI(config_path)
    
    # Gestionar sesión
    if not ui.manage_sessions():
        return
    
    # Renderizar interfaz
    ui.render_interface()
    
    # Manejar interacciones
    ui.handle_user_interaction()
    
    # Actualizar estado
    ui.update_ui_state()
    
    # Stream updates
    ui.stream_updates()
    
    # Optimizar rendimiento
    ui.optimize_ui_performance()

if __name__ == "__main__":
    # Para ejecutar: streamlit run src/api/web_ui.py
    run_web_ui()