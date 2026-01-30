"""
Fixtures de proyectos de muestra para pruebas.
Contiene proyectos de ejemplo en diferentes lenguajes.
"""

import os
import tempfile
import shutil
from pathlib import Path


class SampleProject:
    """Clase base para proyectos de muestra."""
    
    def __init__(self, name):
        self.name = name
        self.temp_dir = None
        self.path = None
        
    def create(self):
        """Crear el proyecto en directorio temporal."""
        self.temp_dir = tempfile.mkdtemp(prefix=f"brain_sample_{self.name}_")
        self.path = self.temp_dir
        self._create_structure()
        return self.path
    
    def _create_structure(self):
        """Crear estructura del proyecto (debe implementarse en subclases)."""
        raise NotImplementedError
    
    def cleanup(self):
        """Limpiar directorio temporal."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def __enter__(self):
        self.create()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class PythonSimpleProject(SampleProject):
    """Proyecto Python simple."""
    
    def __init__(self):
        super().__init__("python_simple")
    
    def _create_structure(self):
        # main.py
        main_py = os.path.join(self.path, "main.py")
        with open(main_py, 'w') as f:
            f.write('''#!/usr/bin/env python3
"""
Proyecto Python simple para pruebas.
"""

import os
import sys
from typing import List, Dict, Optional


class DataProcessor:
    """Procesa datos de entrada."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.history = []
    
    def process(self, data: List[float]) -> Dict[str, float]:
        """Procesa lista de números."""
        if not data:
            return {"error": "Empty data"}
        
        result = {
            "sum": sum(data),
            "mean": sum(data) / len(data),
            "max": max(data),
            "min": min(data),
            "count": len(data)
        }
        
        self.history.append({
            "input": data,
            "output": result,
            "timestamp": "2024-01-01"
        })
        
        return result
    
    def get_stats(self) -> Dict:
        """Obtiene estadísticas del procesador."""
        return {
            "total_processed": len(self.history),
            "avg_input_size": (
                sum(len(h["input"]) for h in self.history) / len(self.history)
                if self.history else 0
            )
        }


def calculate_fibonacci(n: int) -> List[int]:
    """Calcula secuencia de Fibonacci."""
    if n <= 0:
        return []
    
    sequence = [0, 1]
    while len(sequence) < n:
        sequence.append(sequence[-1] + sequence[-2])
    
    return sequence[:n]


def main() -> None:
    """Función principal."""
    processor = DataProcessor({"debug": True})
    
    # Procesar algunos datos
    test_data = [1.5, 2.3, 3.7, 4.1, 5.9]
    result = processor.process(test_data)
    
    print(f"Datos procesados: {test_data}")
    print(f"Resultado: {result}")
    print(f"Estadísticas: {processor.get_stats()}")
    
    # Fibonacci
    fib = calculate_fibonacci(10)
    print(f"Fibonacci(10): {fib}")


if __name__ == "__main__":
    main()
''')
        
        # utils.py
        utils_py = os.path.join(self.path, "utils.py")
        with open(utils_py, 'w') as f:
            f.write('''"""
Utilidades para el proyecto.
"""

import re
from datetime import datetime
from typing import Any, Dict, List


def validate_email(email: str) -> bool:
    """Valida dirección de email."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def format_timestamp(timestamp: datetime, 
                    format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Formatea timestamp."""
    return timestamp.strftime(format_str)


def safe_divide(numerator: float, denominator: float) -> Optional[float]:
    """División segura (evita división por cero)."""
    if denominator == 0:
        return None
    return numerator / denominator


def merge_dicts(*dicts: Dict) -> Dict:
    """Fusiona múltiples diccionarios."""
    result = {}
    for d in dicts:
        result.update(d)
    return result


class Logger:
    """Logger simple."""
    
    def __init__(self, level: str = "INFO"):
        self.level = level
        self.messages = []
    
    def log(self, level: str, message: str, **kwargs) -> None:
        """Registra mensaje."""
        if self._should_log(level):
            entry = {
                "level": level,
                "message": message,
                "timestamp": datetime.now(),
                **kwargs
            }
            self.messages.append(entry)
            print(f"[{level}] {message}")
    
    def _should_log(self, level: str) -> bool:
        """Determina si debe registrar según nivel."""
        levels = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
        return levels.get(level.upper(), 1) >= levels.get(self.level.upper(), 1)
    
    def get_messages(self, level: Optional[str] = None) -> List[Dict]:
        """Obtiene mensajes registrados."""
        if level:
            return [m for m in self.messages if m["level"] == level.upper()]
        return self.messages
''')
        
        # requirements.txt
        req_txt = os.path.join(self.path, "requirements.txt")
        with open(req_txt, 'w') as f:
            f.write('''# Dependencias del proyecto
python>=3.8
pytest>=7.0.0
pytest-cov>=4.0.0
black>=22.0.0
flake8>=5.0.0
''')
        
        # README.md
        readme_md = os.path.join(self.path, "README.md")
        with open(readme_md, 'w') as f:
            f.write('''# Proyecto Python Simple

Proyecto de ejemplo para pruebas de Project Brain.

## Características

- Procesamiento de datos
- Utilidades varias
- Logging básico

## Uso

```bash
python main.py