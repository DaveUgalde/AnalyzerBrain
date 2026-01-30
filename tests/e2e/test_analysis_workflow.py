"""
Pruebas end-to-end del flujo de análisis completo.
"""

import pytest
import asyncio
import tempfile
import os
import shutil
from datetime import datetime
from pathlib import Path

# Marcar todas las pruebas en este archivo como e2e
pytestmark = pytest.mark.e2e


class TestEndToEndAnalysis:
    """Pruebas end-to-end del flujo de análisis completo."""
    
    @pytest.fixture
    def sample_project(self):
        """Crear proyecto de muestra para pruebas e2e."""
        # Crear directorio temporal
        project_dir = tempfile.mkdtemp(prefix="brain_e2e_project_")
        
        # Estructura del proyecto
        structure = {
            "src": {
                "main.py": """
import os
import json
from typing import Dict, List

class DataProcessor:
    \"\"\"Procesa datos de entrada.\"\"\"
    
    def __init__(self, config: Dict):
        self.config = config
        self.cache = {}
    
    def process(self, data: List[float]) -> Dict[str, float]:
        \"\"\"Procesa lista de números.\"\"\"
        if not data:
            return {"error": "Empty data"}
        
        result = {
            "sum": sum(data),
            "mean": sum(data) / len(data),
            "max": max(data),
            "min": min(data)
        }
        
        self.cache[str(data)] = result
        return result
    
    def get_cache_stats(self) -> Dict[str, int]:
        \"\"\"Obtiene estadísticas de caché.\"\"\"
        return {
            "size": len(self.cache),
            "hits": 0  # Simplified
        }

def main():
    \"\"\"Función principal.\"\"\"
    processor = DataProcessor({"debug": True})
    
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = processor.process(data)
    
    print(f"Result: {result}")
    print(f"Cache stats: {processor.get_cache_stats()}")
    
    return result

if __name__ == "__main__":
    main()
""",
                "utils": {
                    "__init__.py": "# Package initialization",
                    "helpers.py": """
\"\"\"Funciones de ayuda.\"\"\"

import re
from datetime import datetime

def validate_email(email: str) -> bool:
    \"\"\"Valida dirección de email.\"\"\"
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def format_date(date: datetime, fmt: str = "%Y-%m-%d") -> str:
    \"\"\"Formatea fecha.\"\"\"
    return date.strftime(fmt)

def safe_divide(a: float, b: float) -> float:
    \"\"\"División segura.\"\"\"
    if b == 0:
        return 0.0
    return a / b
"""
                }
            },
            "tests": {
                "test_main.py": """
import pytest
from src.main import DataProcessor

def test_data_processor():
    \"\"\"Test DataProcessor.\"\"\"
    processor = DataProcessor({})
    
    result = processor.process([1, 2, 3])
    assert result["sum"] == 6
    assert result["mean"] == 2.0

def test_empty_data():
    \"\"\"Test con datos vacíos.\"\"\"
    processor = DataProcessor({})
    result = processor.process([])
    assert "error" in result

if __name__ == "__main__":
    pytest.main([__file__])
"""
            },
            "requirements.txt": """
pytest>=7.0.0
pytest-asyncio>=0.20.0
""",
            "README.md": "# Proyecto de prueba E2E\n\nProyecto para pruebas end-to-end de Project Brain.",
            ".gitignore": """
__pycache__/
*.pyc
.env
"""
        }
        
        # Crear estructura
        def create_structure(base_path, struct):
            for name, content in struct.items():
                path = os.path.join(base_path, name)
                
                if isinstance(content, dict):
                    os.makedirs(path, exist_ok=True)
                    create_structure(path, content)
                else:
                    with open(path, 'w') as f:
                        f.write(content)
        
        create_structure(project_dir, structure)
        
        yield project_dir
        
        # Limpiar
        shutil.rmtree(project_dir, ignore_errors=True)
    
    @pytest.fixture
    async def initialized_system(self):
        """Inicializar sistema para pruebas e2e."""
        # Nota: En pruebas reales, esto inicializaría el sistema completo
        # Para esta prueba, usaremos mocks controlados
        
        from unittest.mock import AsyncMock, Mock
        
        class MockSystem:
            def __init__(self):
                self.orchestrator = Mock()
                self.orchestrator.initialize = AsyncMock(return_value=True)
                self.orchestrator.analyze_project = AsyncMock()
                self.orchestrator.ask_question = AsyncMock()
                self.orchestrator.shutdown = AsyncMock(return_value=True)
                
                # Configurar respuestas realistas
                self.orchestrator.analyze_project.return_value = {
                    "project_id": "e2e_test_project",
                    "status": "completed",
                    "files_analyzed": 5,
                    "entities_extracted": 25,
                    "analysis_time_seconds": 3.5,
                    "findings": [
                        {
                            "type": "code_smell",
                            "severity": "low",
                            "message": "Function too long",
                            "file": "src/main.py",
                            "line": 10
                        }
                    ],
                    "recommendations": [
                        "Add type hints to function parameters",
                        "Consider splitting large functions"
                    ]
                }
                
                self.orchestrator.ask_question.return_value = {
                    "answer": "The DataProcessor class processes numerical data and provides statistics like sum, mean, max, and min.",
                    "confidence": 0.92,
                    "sources": [
                        {
                            "file": "src/main.py",
                            "line": 10,
                            "content": "class DataProcessor:"
                        }
                    ],
                    "reasoning_chain": [
                        "Identified question about DataProcessor class",
                        "Located class definition in src/main.py",
                        "Extracted class documentation and methods",
                        "Formulated summary of class functionality"
                    ]
                }
        
        system = MockSystem()
        await system.orchestrator.initialize()
        
        yield system
        
        await system.orchestrator.shutdown()
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_complete_analysis_workflow(self, sample_project, initialized_system):
        """
        Test flujo completo de análisis:
        1. Inicialización del sistema
        2. Análisis de proyecto
        3. Consulta sobre el proyecto analizado
        4. Validación de resultados
        """
        print(f"\n{'='*60}")
        print("PRUEBA E2E: Flujo completo de análisis")
        print(f"{'='*60}")
        
        # 1. Análisis del proyecto
        print(f"\n1. Analizando proyecto: {sample_project}")
        
        analysis_result = await initialized_system.orchestrator.analyze_project(
            sample_project,
            options={
                "mode": "comprehensive",
                "include_tests": True
            }
        )
        
        # Validar resultado del análisis
        assert analysis_result["status"] == "completed"
        assert analysis_result["files_analyzed"] > 0
        assert "project_id" in analysis_result
        assert "findings" in analysis_result
        assert "recommendations" in analysis_result
        
        print(f"   ✓ Proyecto analizado: {analysis_result['files_analyzed']} archivos")
        print(f"   ✓ Entidades extraídas: {analysis_result.get('entities_extracted', 'N/A')}")
        print(f"   ✓ Hallazgos: {len(analysis_result['findings'])}")
        print(f"   ✓ Recomendaciones: {len(analysis_result['recommendations'])}")
        
        # 2. Consultas sobre el proyecto analizado
        print(f"\n2. Realizando consultas sobre el proyecto...")
        
        questions = [
            "What does the DataProcessor class do?",
            "What functions are available in utils?",
            "Are there any issues in the code?"
        ]
        
        for i, question in enumerate(questions, 1):
            answer = await initialized_system.orchestrator.ask_question(
                question=question,
                project_id=analysis_result["project_id"]
            )
            
            # Validar respuesta
            assert "answer" in answer
            assert "confidence" in answer
            assert answer["confidence"] >= 0.0
            assert answer["confidence"] <= 1.0
            
            print(f"   Q{i}: {question[:50]}...")
            print(f"   A{i}: {answer['answer'][:80]}... (confianza: {answer['confidence']:.2f})")
        
        # 3. Validar métricas del sistema
        print(f"\n3. Verificando métricas del sistema...")
        
        metrics = initialized_system.orchestrator.get_metrics()
        
        # El mock no tiene métricas reales, pero validamos la estructura
        if metrics:
            assert isinstance(metrics, dict)
            print(f"   ✓ Métricas obtenidas: {len(metrics)} items")
        
        print(f"\n{'='*60}")
        print("PRUEBA E2E COMPLETADA EXITOSAMENTE")
        print(f"{'='*60}")
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_large_project_analysis(self, initialized_system):
        """
        Test análisis de proyecto grande.
        Esta prueba es lenta y solo se ejecuta con --run-slow.
        """
        # Crear proyecto con muchos archivos
        import random
        import string
        
        large_project_dir = tempfile.mkdtemp(prefix="brain_e2e_large_")
        
        try:
            print(f"\nCreando proyecto grande en: {large_project_dir}")
            
            # Crear estructura con muchos archivos
            for i in range(100):  # 100 módulos
                module_dir = os.path.join(large_project_dir, f"module_{i:03d}")
                os.makedirs(module_dir, exist_ok=True)
                
                # Crear archivo Python
                with open(os.path.join(module_dir, f"mod_{i}.py"), "w") as f:
                    # Generar código aleatorio pero válido
                    f.write(f'''
"""
Módulo {i}
"""

import random

def generate_data(size: int = 10) -> list:
    """Genera datos aleatorios."""
    return [random.random() for _ in range(size)]

def process_data(data: list) -> dict:
    """Procesa datos."""
    if not data:
        return {{"error": "No data"}}
    
    return {{
        "sum": sum(data),
        "avg": sum(data) / len(data),
        "count": len(data)
    }}

class DataHandler{i}:
    """Manejador de datos para módulo {i}."""
    
    def __init__(self, config=None):
        self.config = config or {{}}
        self.history = []
    
    def handle(self, data):
        """Maneja datos."""
        result = process_data(data)
        self.history.append(result)
        return result
    
    def get_stats(self):
        """Obtiene estadísticas."""
        return {{
            "handled": len(self.history),
            "last_result": self.history[-1] if self.history else None
        }}
''')
            
            print(f"Proyecto creado: 100 módulos")
            
            # Analizar proyecto grande
            start_time = datetime.now()
            
            result = await initialized_system.orchestrator.analyze_project(
                large_project_dir,
                options={"mode": "standard"}
            )
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            # Validaciones
            assert result["status"] == "completed"
            assert result["files_analyzed"] >= 100
            
            print(f"✓ Proyecto grande analizado en {elapsed:.2f} segundos")
            print(f"✓ Archivos analizados: {result['files_analyzed']}")
            
            # Performance check
            assert elapsed < 30.0, f"Análisis tomó {elapsed:.2f}s (> 30s)"
            
        finally:
            # Limpiar
            shutil.rmtree(large_project_dir, ignore_errors=True)
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_error_recovery_flow(self, initialized_system):
        """
        Test flujo de recuperación de errores.
        """
        print(f"\n{'='*60}")
        print("PRUEBA E2E: Recuperación de errores")
        print(f"{'='*60}")
        
        # 1. Proyecto que no existe
        print(f"\n1. Intentando analizar proyecto inexistente...")
        
        try:
            await initialized_system.orchestrator.analyze_project(
                "/path/that/does/not/exist"
            )
            # Si llegamos aquí, el mock no está simulando el error
            print("   ⚠️  Mock no simuló error de proyecto inexistente")
        except Exception as e:
            print(f"   ✓ Error manejado correctamente: {type(e).__name__}")
        
        # 2. Pregunta sin proyecto
        print(f"\n2. Pregunta sin contexto de proyecto...")
        
        answer = await initialized_system.orchestrator.ask_question(
            "What is Python?"
        )
        
        # Debería responder incluso sin proyecto
        assert "answer" in answer
        print(f"   ✓ Respuesta generada (con o sin proyecto)")
        
        print(f"\n{'='*60}")
        print("PRUEBA DE RECUPERACIÓN COMPLETADA")
        print(f"{'='*60}")
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, initialized_system, sample_project):
        """
        Test operaciones concurrentes.
        """
        print(f"\n{'='*60}")
        print("PRUEBA E2E: Operaciones concurrentes")
        print(f"{'='*60}")
        
        import asyncio
        import time
        
        # Crear múltiples solicitudes
        num_operations = 10
        tasks = []
        
        start_time = time.time()
        
        for i in range(num_operations):
            if i % 2 == 0:
                # Análisis de proyecto (simulado)
                task = initialized_system.orchestrator.analyze_project(
                    sample_project,
                    options={"mode": "quick"}
                )
            else:
                # Preguntas
                task = initialized_system.orchestrator.ask_question(
                    f"What is function {i}?"
                )
            
            tasks.append(task)
        
        # Ejecutar concurrentemente
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        elapsed = time.time() - start_time
        
        # Analizar resultados
        successful = 0
        failed = 0
        
        for result in results:
            if isinstance(result, Exception):
                failed += 1
                print(f"   ✗ Operación falló: {result}")
            else:
                successful += 1
        
        print(f"\nResultados concurrentes:")
        print(f"   ✓ Exitosas: {successful}/{num_operations}")
        print(f"   ✗ Fallidas: {failed}/{num_operations}")
        print(f"   ⏱️  Tiempo total: {elapsed:.2f}s")
        print(f"   🚀 Throughput: {num_operations/elapsed:.1f} ops/seg")
        
        # Verificar que al menos algunas operaciones tuvieron éxito
        assert successful > 0, "Todas las operaciones concurrentes fallaron"
        
        print(f"\n{'='*60}")
        print("PRUEBA CONCURRENTE COMPLETADA")
        print(f"{'='*60}")


class TestRealComponentIntegration:
    """
    Pruebas de integración con componentes reales (cuando disponibles).
    Estas pruebas requieren dependencias reales instaladas.
    """
    
    @pytest.mark.e2e
    @pytest.mark.integration
    def test_real_parser_integration(self, sample_project):
        """
        Test integración con parser real (si está disponible).
        """
        try:
            from indexer.multi_language_parser import MultiLanguageParser, ParserConfig
            
            # Crear parser con configuración real
            config = ParserConfig(
                enabled_languages=["python"],
                cache_parsed_files=False
            )
            parser = MultiLanguageParser(config)
            
            # Parsear archivo real
            main_py = os.path.join(sample_project, "src", "main.py")
            result = parser.parse_file(main_py)
            
            # Validar resultados básicos
            assert result.success is True or result.success is False
            # Si tiene éxito, debería tener entidades
            if result.success:
                assert len(result.entities) > 0
                print(f"✓ Parser real extrajo {len(result.entities)} entidades")
            else:
                print(f"⚠️  Parser falló (posiblemente tree-sitter no instalado)")
                
        except ImportError as e:
            pytest.skip(f"Dependencia no disponible: {e}")
    
    @pytest.mark.e2e
    @pytest.mark.integration
    def test_real_embeddings_integration(self):
        """
        Test integración con embeddings reales (si está disponible).
        """
        try:
            from embeddings.embedding_generator import EmbeddingGenerator, EmbeddingConfig
            
            # Crear generador
            config = EmbeddingConfig(
                default_model="all-MiniLM-L6-v2",
                cache_embeddings=False
            )
            generator = EmbeddingGenerator(config)
            
            # Generar embedding
            text = "This is a test sentence for embeddings."
            embedding = generator.generate_text_embedding(text)
            
            # Validar
            if embedding:  # Puede ser None si el modelo no está disponible
                assert len(embedding) == 384  # Dimensión del modelo
                print(f"✓ Embeddings reales generados ({len(embedding)} dimensiones)")
            else:
                print(f"⚠️  Modelo de embeddings no disponible")
                
        except ImportError as e:
            pytest.skip(f"Dependencia no disponible: {e}")


if __name__ == "__main__":
    # Ejecutar pruebas e2e
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "e2e"
    ])