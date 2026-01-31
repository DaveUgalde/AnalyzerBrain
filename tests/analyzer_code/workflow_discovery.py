#!/usr/bin/env python3
"""
Analizador de Proyectos - Identificador de Workflows
Analiza un proyecto de código para identificar todos los workflows posibles,
sus requerimientos y configuraciones necesarias.
"""

import os
import json
import yaml
import re
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import platform

class ProjectWorkflowAnalyzer:
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.workflows = []
        self.project_type = None
        self.dependencies = []
        self.config_files = []
        
    def detect_project_type(self) -> str:
        """Detecta el tipo de proyecto basado en archivos existentes"""
        files = [f.name for f in self.project_path.iterdir() if f.is_file()]
        
        if 'package.json' in files:
            return 'nodejs'
        elif 'requirements.txt' in files or 'setup.py' in files:
            return 'python'
        elif 'pom.xml' in files:
            return 'java-maven'
        elif 'build.gradle' in files:
            return 'java-gradle'
        elif 'go.mod' in files:
            return 'go'
        elif 'Cargo.toml' in files:
            return 'rust'
        elif 'Gemfile' in files:
            return 'ruby'
        elif 'composer.json' in files:
            return 'php'
        elif 'docker-compose.yml' in files:
            return 'docker'
        elif '.github' in [d.name for d in self.project_path.iterdir() if d.is_dir()]:
            return 'github-actions'
        elif 'Makefile' in files:
            return 'make'
        else:
            return 'generic'
    
    def find_config_files(self):
        """Encuentra archivos de configuración en el proyecto"""
        config_patterns = [
            '*.json', '*.yaml', '*.yml', '*.toml', '*.ini',
            '*.cfg', '*.conf', '*.config', 'Dockerfile*',
            'docker-compose*.yml', '.env*', 'Makefile',
            'package.json', 'requirements.txt', 'pyproject.toml',
            'pom.xml', 'build.gradle', 'webpack.config.*',
            'tsconfig.json', 'babel.config.*', '.eslintrc*',
            '.prettierrc*', 'jest.config.*', 'karma.conf.*',
            '.gitlab-ci.yml', '.travis.yml', 'azure-pipelines.yml',
            'Jenkinsfile', 'bitbucket-pipelines.yml'
        ]
        
        for pattern in config_patterns:
            for config_file in self.project_path.rglob(pattern):
                if config_file.is_file():
                    self.config_files.append(str(config_file.relative_to(self.project_path)))
    
    def analyze_package_json(self) -> Dict[str, Any]:
        """Analiza package.json para Node.js proyectos"""
        package_json_path = self.project_path / 'package.json'
        if not package_json_path.exists():
            return {}
        
        try:
            with open(package_json_path, 'r') as f:
                data = json.load(f)
            
            workflows = []
            
            # Workflow: Desarrollo local
            if 'scripts' in data:
                dev_workflow = {
                    'nombre': 'Desarrollo Local',
                    'descripcion': 'Entorno de desarrollo con hot-reload',
                    'requerimientos': ['Node.js', 'npm/yarn/pnpm'],
                    'configuraciones': [
                        'Variables de entorno en .env',
                        'Configuración en package.json'
                    ],
                    'comandos': list(data['scripts'].keys()),
                    'dependencias': list(data.get('dependencies', {}).keys()) + 
                                   list(data.get('devDependencies', {}).keys())
                }
                workflows.append(dev_workflow)
            
            # Workflow: Build de producción
            build_workflow = {
                'nombre': 'Build de Producción',
                'descripcion': 'Compilación y optimización para producción',
                'requerimientos': ['Node.js', 'npm/yarn/pnpm', 'Build tool (Webpack/Vite/etc)'],
                'configuraciones': [
                    'Configuración de build en package.json scripts',
                    'Archivos de configuración del bundler'
                ],
                'comandos': ['npm run build', 'npm run build:prod', 'yarn build'],
                'optimizaciones': ['Minificación', 'Tree-shaking', 'Code splitting']
            }
            workflows.append(build_workflow)
            
            return {'workflows': workflows, 'package_info': data}
        except Exception as e:
            print(f"Error analizando package.json: {e}")
            return {}
    
    def analyze_python_project(self) -> List[Dict[str, Any]]:
        """Analiza proyectos Python"""
        workflows = []
        
        # Workflow: Entorno virtual y dependencias
        venv_workflow = {
            'nombre': 'Configuración de Entorno Python',
            'descripcion': 'Creación de entorno virtual e instalación de dependencias',
            'requerimientos': ['Python 3.6+', 'pip', 'virtualenv/venv'],
            'configuraciones': [
                'requirements.txt o pyproject.toml',
                '.env para variables de entorno',
                'setup.cfg o setup.py'
            ],
            'comandos': [
                'python -m venv venv',
                'source venv/bin/activate  # Linux/Mac',
                'venv\\Scripts\\activate  # Windows',
                'pip install -r requirements.txt'
            ]
        }
        workflows.append(venv_workflow)
        
        # Workflow: Testing
        test_workflow = {
            'nombre': 'Testing y QA',
            'descripcion': 'Ejecución de tests y análisis de calidad',
            'requerimientos': ['pytest/unittest', 'coverage.py', 'flake8/pylint'],
            'configuraciones': [
                'pytest.ini o setup.cfg',
                '.coveragerc',
                '.flake8 o .pylintrc'
            ],
            'comandos': [
                'pytest',
                'coverage run -m pytest',
                'coverage report',
                'flake8 .',
                'pylint src/'
            ]
        }
        workflows.append(test_workflow)
        
        # Verificar si es Django
        if (self.project_path / 'manage.py').exists():
            django_workflow = {
                'nombre': 'Desarrollo Django',
                'descripcion': 'Workflow completo para desarrollo con Django',
                'requerimientos': ['Django', 'Database (PostgreSQL/MySQL/SQLite)'],
                'configuraciones': [
                    'settings.py',
                    'urls.py',
                    '.env para secret key y database URL'
                ],
                'comandos': [
                    'python manage.py migrate',
                    'python manage.py runserver',
                    'python manage.py createsuperuser',
                    'python manage.py collectstatic'
                ],
                'subworkflows': [
                    'Migraciones de base de datos',
                    'Panel de administración',
                    'API REST (si usa Django REST Framework)'
                ]
            }
            workflows.append(django_workflow)
        
        # Verificar si es FastAPI
        if self.contains_file_pattern('main.py', 'fastapi'):
            fastapi_workflow = {
                'nombre': 'API FastAPI',
                'descripcion': 'Desarrollo y despliegue de API con FastAPI',
                'requerimientos': ['FastAPI', 'uvicorn', 'pydantic'],
                'configuraciones': [
                    'Configuración CORS',
                    'Variables de entorno para conexiones DB',
                    'Configuración de logging'
                ],
                'comandos': [
                    'uvicorn main:app --reload',
                    'uvicorn main:app --host 0.0.0.0 --port 8000'
                ]
            }
            workflows.append(fastapi_workflow)
        
        return workflows
    
    def analyze_docker_files(self) -> List[Dict[str, Any]]:
        """Analiza archivos Docker"""
        workflows = []
        
        if (self.project_path / 'Dockerfile').exists():
            docker_workflow = {
                'nombre': 'Build de Imagen Docker',
                'descripcion': 'Construcción y gestión de imágenes Docker',
                'requerimientos': ['Docker Engine', 'Docker CLI'],
                'configuraciones': ['Dockerfile', '.dockerignore'],
                'comandos': [
                    'docker build -t nombre-imagen .',
                    'docker run -p 8080:80 nombre-imagen',
                    'docker push registry/nombre-imagen'
                ]
            }
            workflows.append(docker_workflow)
        
        if (self.project_path / 'docker-compose.yml').exists():
            compose_workflow = {
                'nombre': 'Orquestación con Docker Compose',
                'descripcion': 'Despliegue multi-contenedor con Docker Compose',
                'requerimientos': ['Docker Compose'],
                'configuraciones': ['docker-compose.yml', 'Variables en .env'],
                'servicios': self.extract_docker_compose_services(),
                'comandos': [
                    'docker-compose up -d',
                    'docker-compose down',
                    'docker-compose logs -f',
                    'docker-compose build'
                ]
            }
            workflows.append(compose_workflow)
        
        return workflows
    
    def analyze_ci_cd(self) -> List[Dict[str, Any]]:
        """Analiza archivos de CI/CD"""
        workflows = []
        
        # GitHub Actions
        github_actions_path = self.project_path / '.github' / 'workflows'
        if github_actions_path.exists():
            for workflow_file in github_actions_path.glob('*.yml'):
                workflows.extend(self.analyze_github_action(workflow_file))
        
        # GitLab CI
        gitlab_ci_path = self.project_path / '.gitlab-ci.yml'
        if gitlab_ci_path.exists():
            gitlab_workflow = {
                'nombre': 'Pipeline GitLab CI/CD',
                'descripcion': 'Integración y despliegue continuo con GitLab',
                'requerimientos': ['Runner de GitLab', 'Docker (opcional)'],
                'configuraciones': ['.gitlab-ci.yml'],
                'etapas': self.extract_gitlab_stages(gitlab_ci_path)
            }
            workflows.append(gitlab_workflow)
        
        # Jenkins
        jenkins_path = self.project_path / 'Jenkinsfile'
        if jenkins_path.exists():
            jenkins_workflow = {
                'nombre': 'Pipeline Jenkins',
                'descripcion': 'Automation con Jenkins',
                'requerimientos': ['Servidor Jenkins', 'Agentes/workers'],
                'configuraciones': ['Jenkinsfile', 'Credenciales en Jenkins'],
                'etapas': ['build', 'test', 'deploy']
            }
            workflows.append(jenkins_workflow)
        
        return workflows
    
    def analyze_github_action(self, workflow_file: Path) -> List[Dict[str, Any]]:
        """Analiza un archivo de GitHub Actions"""
        workflows = []
        try:
            with open(workflow_file, 'r') as f:
                content = yaml.safe_load(f)
            
            workflow_name = content.get('name', workflow_file.stem)
            gh_workflow = {
                'nombre': f'GitHub Actions: {workflow_name}',
                'descripcion': content.get('name', 'Workflow de CI/CD'),
                'trigger': content.get('on', {}),
                'jobs': list(content.get('jobs', {}).keys()),
                'configuraciones': [str(workflow_file.relative_to(self.project_path))],
                'requerimientos': ['Cuenta GitHub', 'Secrets configuradas'],
                'runs_on': self.extract_github_runners(content)
            }
            workflows.append(gh_workflow)
        except Exception as e:
            print(f"Error analizando GitHub Action {workflow_file}: {e}")
        
        return workflows
    
    def analyze_database_configs(self) -> List[Dict[str, Any]]:
        """Analiza configuraciones de base de datos"""
        workflows = []
        
        # Buscar configuraciones de DB
        db_config_patterns = ['*database*', '*db*', '*sql*', '*mongo*', '*redis*']
        db_files = []
        
        for pattern in db_config_patterns:
            for file in self.project_path.rglob(pattern):
                if file.is_file() and file.suffix in ['.yml', '.yaml', '.json', '.js', '.ts', '.py']:
                    db_files.append(str(file.relative_to(self.project_path)))
        
        if db_files:
            db_workflow = {
                'nombre': 'Gestión de Base de Datos',
                'descripcion': 'Migraciones, seeds y mantenimiento de BD',
                'requerimientos': ['Cliente de BD', 'Herramientas de migración'],
                'configuraciones': db_files,
                'tareas_comunes': [
                    'Migraciones',
                    'Seeding de datos',
                    'Backup/restore',
                    'Optimización de queries'
                ]
            }
            workflows.append(db_workflow)
        
        return workflows
    
    def analyze_testing_workflows(self) -> List[Dict[str, Any]]:
        """Analiza workflows de testing"""
        workflows = []
        
        test_configs = []
        test_patterns = ['*test*', '*spec*', 'jest*', 'pytest*', '.eslint*', '.prettier*']
        
        for pattern in test_patterns:
            for file in self.project_path.rglob(pattern):
                if file.is_file():
                    test_configs.append(str(file.relative_to(self.project_path)))
        
        test_workflow = {
            'nombre': 'Testing Automatizado',
            'descripcion': 'Workflows de testing y calidad de código',
            'requerimientos': ['Framework de testing', 'Runner de tests'],
            'configuraciones': test_configs,
            'tipos_tests': self.detect_test_types()
        }
        workflows.append(test_workflow)
        
        return workflows
    
    def analyze_deployment_workflows(self) -> List[Dict[str, Any]]:
        """Analiza workflows de despliegue"""
        workflows = []
        
        # Despliegue en servidores
        server_deploy = {
            'nombre': 'Despliegue en Servidor',
            'descripcion': 'Despliegue tradicional en servidor VPS/dedicado',
            'requerimientos': ['SSH access', 'Servidor configurado', 'Nginx/Apache'],
            'configuraciones': [
                'Scripts de deploy (deploy.sh)',
                'Configuración de servidor web',
                'Supervisor/systemd configs'
            ],
            'pasos': [
                'Build de la aplicación',
                'Transferencia via SCP/rsync',
                'Restart de servicios'
            ]
        }
        workflows.append(server_deploy)
        
        # Despliegue en Kubernetes
        k8s_files = list(self.project_path.rglob('*k8s*')) + list(self.project_path.rglob('*kubernetes*'))
        if k8s_files or (self.project_path / 'k8s').exists():
            k8s_workflow = {
                'nombre': 'Despliegue en Kubernetes',
                'descripcion': 'Despliegue en cluster Kubernetes',
                'requerimientos': ['kubectl', 'Cluster K8s', 'Helm (opcional)'],
                'configuraciones': [
                    'Deployment.yaml',
                    'Service.yaml',
                    'Ingress.yaml',
                    'ConfigMap.yaml',
                    'Secrets.yaml'
                ],
                'comandos': [
                    'kubectl apply -f k8s/',
                    'kubectl rollout status deployment/app',
                    'kubectl get pods'
                ]
            }
            workflows.append(k8s_workflow)
        
        # Despliegue en serverless
        serverless_patterns = ['serverless*', 'sls*', 'zappa*', 'vercel*', 'netlify*']
        for pattern in serverless_patterns:
            if list(self.project_path.rglob(pattern)):
                serverless_workflow = {
                    'nombre': 'Despliegue Serverless',
                    'descripcion': 'Despliegue en plataformas serverless',
                    'requerimientos': ['CLI de la plataforma', 'Cuenta configurada'],
                    'configuraciones': ['serverless.yml', 'vercel.json', 'netlify.toml'],
                    'plataformas': self.detect_serverless_platforms()
                }
                workflows.append(serverless_workflow)
                break
        
        return workflows
    
    def analyze_monitoring_workflows(self) -> List[Dict[str, Any]]:
        """Analiza workflows de monitoreo"""
        workflows = []
        
        monitoring_files = []
        for pattern in ['*prometheus*', '*grafana*', '*sentry*', '*newrelic*', '*datadog*']:
            monitoring_files.extend(self.project_path.rglob(pattern))
        
        if monitoring_files:
            monitoring_workflow = {
                'nombre': 'Monitoreo y Observabilidad',
                'descripcion': 'Workflows para monitoreo de aplicación',
                'requerimientos': ['Herramientas de monitoring', 'Dashboards'],
                'configuraciones': [
                    str(f.relative_to(self.project_path)) for f in monitoring_files
                ],
                'metricas': self.detect_monitoring_metrics()
            }
            workflows.append(monitoring_workflow)
        
        return workflows
    
    def analyze_security_workflows(self) -> List[Dict[str, Any]]:
        """Analiza workflows de seguridad"""
        workflows = []
        
        security_workflow = {
            'nombre': 'Seguridad y Auditoría',
            'descripcion': 'Workflows de seguridad y análisis de vulnerabilidades',
            'requerimientos': ['Herramientas de seguridad', 'Dependabot/Snyk'],
            'tareas': [
                'Scan de dependencias',
                'Análisis de código estático',
                'Penetration testing',
                'Auditoría de secrets'
            ],
            'herramientas_recomendadas': [
                'OWASP ZAP',
                'snyk',
                'trivy',
                'gitleaks',
                'bandit (Python)',
                'npm audit (Node.js)'
            ]
        }
        workflows.append(security_workflow)
        
        return workflows
    
    def analyze_documentation_workflows(self) -> List[Dict[str, Any]]:
        """Analiza workflows de documentación"""
        workflows = []
        
        docs_path = self.project_path / 'docs'
        if docs_path.exists():
            docs_workflow = {
                'nombre': 'Generación de Documentación',
                'descripcion': 'Workflow para generar y publicar documentación',
                'requerimientos': ['Generador de docs', 'Servidor de docs'],
                'herramientas': self.detect_doc_tools(),
                'comandos': [
                    'npm run docs:build',
                    'mkdocs build',
                    'sphinx-build docs/ build/'
                ],
                'plataformas_despliegue': ['GitHub Pages', 'Read the Docs', 'Netlify']
            }
            workflows.append(docs_workflow)
        
        return workflows
    
    def contains_file_pattern(self, filename: str, content_pattern: str = None) -> bool:
        """Verifica si un archivo existe y contiene un patrón"""
        file_path = self.project_path / filename
        if not file_path.exists():
            return False
        
        if content_pattern:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                return content_pattern.lower() in content.lower()
            except:
                return False
        
        return True
    
    def extract_docker_compose_services(self) -> List[str]:
        """Extrae servicios de docker-compose.yml"""
        services = []
        compose_path = self.project_path / 'docker-compose.yml'
        
        if compose_path.exists():
            try:
                with open(compose_path, 'r') as f:
                    data = yaml.safe_load(f)
                services = list(data.get('services', {}).keys())
            except:
                pass
        
        return services
    
    def extract_gitlab_stages(self, gitlab_ci_path: Path) -> List[str]:
        """Extrae etapas de .gitlab-ci.yml"""
        stages = []
        try:
            with open(gitlab_ci_path, 'r') as f:
                data = yaml.safe_load(f)
            stages = data.get('stages', [])
        except:
            pass
        
        return stages
    
    def extract_github_runners(self, workflow_content: Dict) -> List[str]:
        """Extrae runners de GitHub Actions"""
        runners = set()
        
        for job_name, job_config in workflow_content.get('jobs', {}).items():
            runs_on = job_config.get('runs-on', 'ubuntu-latest')
            if isinstance(runs_on, list):
                runners.update(runs_on)
            else:
                runners.add(runs_on)
        
        return list(runners)
    
    def detect_test_types(self) -> List[str]:
        """Detecta tipos de tests usados en el proyecto"""
        test_types = []
        
        # Buscar patrones en archivos
        test_patterns = {
            'unit': ['*test*.js', '*test*.py', '*spec*.js'],
            'integration': ['*integration*', '*e2e*'],
            'e2e': ['cypress*', '*e2e*', '*playwright*', '*puppeteer*'],
            'performance': ['*performance*', '*load*', '*stress*'],
            'security': ['*security*', '*penetration*']
        }
        
        for test_type, patterns in test_patterns.items():
            for pattern in patterns:
                if list(self.project_path.rglob(pattern)):
                    test_types.append(test_type)
                    break
        
        return list(set(test_types))
    
    def detect_serverless_platforms(self) -> List[str]:
        """Detecta plataformas serverless configuradas"""
        platforms = []
        
        if list(self.project_path.rglob('serverless.yml')):
            platforms.append('AWS Lambda')
        if list(self.project_path.rglob('vercel.json')):
            platforms.append('Vercel')
        if list(self.project_path.rglob('netlify.toml')):
            platforms.append('Netlify')
        if list(self.project_path.rglob('firebase.json')):
            platforms.append('Firebase')
        
        return platforms
    
    def detect_monitoring_metrics(self) -> List[str]:
        """Detecta métricas de monitoreo configuradas"""
        metrics = []
        
        # Buscar configuraciones de métricas
        metric_patterns = {
            'performance': ['response_time', 'throughput', 'latency'],
            'errors': ['error_rate', 'exceptions'],
            'resources': ['cpu', 'memory', 'disk'],
            'business': ['users', 'transactions', 'revenue']
        }
        
        # Analizar archivos de configuración comunes
        config_files = ['docker-compose.yml', 'prometheus.yml', 'newrelic.js']
        
        for config_file in config_files:
            file_path = self.project_path / config_file
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    for metric_type, metric_list in metric_patterns.items():
                        for metric in metric_list:
                            if metric in content.lower():
                                metrics.append(metric_type)
                                break
                except:
                    pass
        
        return list(set(metrics))
    
    def detect_doc_tools(self) -> List[str]:
        """Detecta herramientas de documentación"""
        tools = []
        
        doc_patterns = {
            'mkdocs': ['mkdocs.yml'],
            'sphinx': ['conf.py', 'index.rst'],
            'docusaurus': ['docusaurus.config.js'],
            'vuepress': ['.vuepress'],
            'gitbook': ['book.json', 'README.md'],
            'jsdoc': ['jsdoc.json']
        }
        
        for tool, files in doc_patterns.items():
            for file_pattern in files:
                if list(self.project_path.rglob(file_pattern)):
                    tools.append(tool)
                    break
        
        return tools
    
    def analyze(self) -> Dict[str, Any]:
        """Ejecuta el análisis completo del proyecto"""
        print(f"Analizando proyecto en: {self.project_path}")
        
        # Detectar tipo de proyecto
        self.project_type = self.detect_project_type()
        print(f"Tipo de proyecto detectado: {self.project_type}")
        
        # Encontrar archivos de configuración
        self.find_config_files()
        
        # Analizar workflows específicos por tipo de proyecto
        all_workflows = []
        
        # Workflows generales (siempre presentes)
        general_workflows = [
            {
                'nombre': 'Control de Versiones',
                'descripcion': 'Workflow de Git para colaboración',
                'requerimientos': ['Git', 'Repositorio remoto (GitHub/GitLab/etc)'],
                'configuraciones': ['.gitignore', '.gitattributes'],
                'branching_strategy': self.detect_branching_strategy(),
                'comandos_comunes': [
                    'git clone <repo>',
                    'git checkout -b feature/nueva-funcionalidad',
                    'git commit -m "mensaje"',
                    'git push origin feature/nueva-funcionalidad',
                    'git merge feature/nueva-funcionalidad'
                ]
            },
            {
                'nombre': 'Setup de Desarrollo',
                'descripcion': 'Configuración inicial del entorno de desarrollo',
                'requerimientos': self.detect_development_requirements(),
                'configuraciones': self.config_files[:10],  # Primeros 10 archivos
                'pasos': [
                    'Clonar repositorio',
                    'Instalar dependencias',
                    'Configurar variables de entorno',
                    'Iniciar base de datos (si aplica)',
                    'Ejecutar migraciones (si aplica)'
                ]
            }
        ]
        all_workflows.extend(general_workflows)
        
        # Workflows por tipo de proyecto
        if self.project_type == 'nodejs':
            node_analysis = self.analyze_package_json()
            all_workflows.extend(node_analysis.get('workflows', []))
        
        elif self.project_type == 'python':
            all_workflows.extend(self.analyze_python_project())
        
        # Workflows de Docker (si existen)
        all_workflows.extend(self.analyze_docker_files())
        
        # Workflows de CI/CD
        all_workflows.extend(self.analyze_ci_cd())
        
        # Workflows de base de datos
        all_workflows.extend(self.analyze_database_configs())
        
        # Workflows de testing
        all_workflows.extend(self.analyze_testing_workflows())
        
        # Workflows de despliegue
        all_workflows.extend(self.analyze_deployment_workflows())
        
        # Workflows de monitoreo
        all_workflows.extend(self.analyze_monitoring_workflows())
        
        # Workflows de seguridad
        all_workflows.extend(self.analyze_security_workflows())
        
        # Workflows de documentación
        all_workflows.extend(self.analyze_documentation_workflows())
        
        self.workflows = all_workflows
        
        return {
            'project_info': {
                'path': str(self.project_path),
                'type': self.project_type,
                'config_files': self.config_files,
                'total_workflows': len(all_workflows)
            },
            'workflows': all_workflows
        }
    
    def detect_branching_strategy(self) -> str:
        """Detecta la estrategia de branching"""
        gitflow_files = ['.gitflow', 'gitflow']
        for file in gitflow_files:
            if (self.project_path / file).exists():
                return 'GitFlow'
        
        # Buscar en README o docs
        readme_path = self.project_path / 'README.md'
        if readme_path.exists():
            try:
                with open(readme_path, 'r') as f:
                    content = f.read().lower()
                if 'gitflow' in content:
                    return 'GitFlow'
                elif 'trunk' in content:
                    return 'Trunk-based'
                elif 'github flow' in content:
                    return 'GitHub Flow'
            except:
                pass
        
        return 'Estándar (master/feature/hotfix)'
    
    def detect_development_requirements(self) -> List[str]:
        """Detecta requerimientos de desarrollo"""
        requirements = []
        
        # Dependiendo del tipo de proyecto
        if self.project_type == 'nodejs':
            requirements.extend(['Node.js', 'npm o yarn', 'Editor/IDE'])
        elif self.project_type == 'python':
            requirements.extend(['Python', 'pip', 'virtualenv/venv', 'Editor/IDE'])
        elif self.project_type == 'docker':
            requirements.extend(['Docker', 'Docker Compose'])
        
        # Requerimientos generales
        requirements.extend(['Git', 'Terminal/CLI'])
        
        return requirements
    
    def generate_report(self, output_format: str = 'markdown') -> str:
        """Genera un reporte del análisis"""
        if not self.workflows:
            self.analyze()
        
        if output_format == 'markdown':
            return self._generate_markdown_report()
        elif output_format == 'json':
            return json.dumps({
                'project_info': {
                    'path': str(self.project_path),
                    'type': self.project_type,
                    'config_files': self.config_files
                },
                'workflows': self.workflows
            }, indent=2, ensure_ascii=False)
        else:
            return self._generate_text_report()
    
    def _generate_markdown_report(self) -> str:
        """Genera reporte en formato Markdown"""
        report = []
        
        report.append(f"# Análisis de Workflows - {self.project_path.name}")
        report.append(f"\n**Fecha de análisis:** {self._get_current_date()}")
        report.append(f"**Tipo de proyecto:** {self.project_type}")
        report.append(f"**Total de workflows identificados:** {len(self.workflows)}\n")
        
        report.append("## 📋 Tabla de Workflows Identificados")
        report.append("\n| # | Workflow | Descripción | Requerimientos | Configuraciones |")
        report.append("|---|---|---|---|---|")
        
        for i, workflow in enumerate(self.workflows, 1):
            nombre = workflow.get('nombre', 'N/A')
            descripcion = workflow.get('descripcion', '')[:100] + "..."
            requerimientos = ", ".join(workflow.get('requerimientos', []))[:50]
            configuraciones = ", ".join(workflow.get('configuraciones', []))[:50]
            
            report.append(f"| {i} | {nombre} | {descripcion} | {requerimientos} | {configuraciones} |")
        
        report.append("\n## 🔧 Workflows Detallados\n")
        
        for i, workflow in enumerate(self.workflows, 1):
            report.append(f"### {i}. {workflow.get('nombre')}")
            report.append(f"\n**Descripción:** {workflow.get('descripcion', 'N/A')}")
            
            if workflow.get('requerimientos'):
                report.append(f"\n**Requerimientos:**")
                for req in workflow['requerimientos']:
                    report.append(f"- {req}")
            
            if workflow.get('configuraciones'):
                report.append(f"\n**Configuraciones necesarias:**")
                for config in workflow['configuraciones']:
                    report.append(f"- `{config}`")
            
            if workflow.get('comandos'):
                report.append(f"\n**Comandos principales:**")
                for cmd in workflow['comandos']:
                    report.append(f"```bash\n{cmd}\n```")
            
            if workflow.get('pasos'):
                report.append(f"\n**Pasos a seguir:**")
                for j, paso in enumerate(workflow['pasos'], 1):
                    report.append(f"{j}. {paso}")
            
            report.append("\n---\n")
        
        # Resumen
        report.append("## 📊 Resumen del Proyecto")
        report.append(f"\n- **Archivos de configuración encontrados:** {len(self.config_files)}")
        report.append(f"- **Workflows de desarrollo:** {len([w for w in self.workflows if 'desarrollo' in w.get('nombre', '').lower() or 'dev' in w.get('nombre', '').lower()])}")
        report.append(f"- **Workflows de testing:** {len([w for w in self.workflows if 'test' in w.get('nombre', '').lower()])}")
        report.append(f"- **Workflows de despliegue:** {len([w for w in self.workflows if 'deploy' in w.get('nombre', '').lower() or 'despliegue' in w.get('nombre', '').lower()])}")
        report.append(f"- **Workflows de CI/CD:** {len([w for w in self.workflows if 'ci' in w.get('nombre', '').lower() or 'cd' in w.get('nombre', '').lower()])}")
        
        return "\n".join(report)
    
    def _generate_text_report(self) -> str:
        """Genera reporte en texto plano"""
        report = []
        
        report.append(f"=" * 80)
        report.append(f"ANÁLISIS DE WORKFLOWS - {self.project_path.name}")
        report.append(f"=" * 80)
        report.append(f"\nTipo de proyecto: {self.project_type}")
        report.append(f"Total workflows: {len(self.workflows)}")
        
        for i, workflow in enumerate(self.workflows, 1):
            report.append(f"\n{'='*60}")
            report.append(f"WORKFLOW {i}: {workflow.get('nombre')}")
            report.append(f"{'='*60}")
            report.append(f"Descripción: {workflow.get('descripcion', 'N/A')}")
            
            if workflow.get('requerimientos'):
                report.append(f"\nRequerimientos:")
                for req in workflow['requerimientos']:
                    report.append(f"  • {req}")
            
            if workflow.get('configuraciones'):
                report.append(f"\nConfiguraciones:")
                for config in workflow['configuraciones']:
                    report.append(f"  • {config}")
        
        return "\n".join(report)
    
    def _get_current_date(self) -> str:
        """Obtiene la fecha actual formateada"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    parser = argparse.ArgumentParser(description='Analizador de Workflows de Proyectos')
    parser.add_argument('project_path', help='Ruta al proyecto a analizar')
    parser.add_argument('--format', choices=['markdown', 'json', 'text'], 
                       default='markdown', help='Formato de salida')
    parser.add_argument('--output', help='Archivo de salida (opcional)')
    
    args = parser.parse_args()
    
    # Verificar que el proyecto existe
    if not os.path.exists(args.project_path):
        print(f"Error: La ruta '{args.project_path}' no existe.")
        sys.exit(1)
    
    # Crear analizador y ejecutar análisis
    analyzer = ProjectWorkflowAnalyzer(args.project_path)
    result = analyzer.analyze()
    
    # Generar reporte
    report = analyzer.generate_report(args.format)
    
    # Mostrar o guardar reporte
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Reporte guardado en: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()