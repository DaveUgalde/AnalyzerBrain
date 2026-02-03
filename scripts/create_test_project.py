#!/usr/bin/env python3
"""
Script para crear un proyecto de prueba para ANALYZERBRAIN.

Crea una estructura de proyecto de prueba con varios archivos
en diferentes lenguajes para probar el sistema.

Uso:
    python scripts/create_test_project.py [ruta_destino]
"""

import sys
from pathlib import Path

def create_test_project(destination: Path):
    """Crea un proyecto de prueba en la ruta especificada."""
    
    if destination.exists():
        print(f"❌ La ruta ya existe: {destination}")
        sys.exit(1)
    
    destination.mkdir(parents=True)
    
    # Estructura de directorios
    dirs = [
        "src",
        "tests",
        "docs",
        "config",
        "data"
    ]
    
    for dir_name in dirs:
        (destination / dir_name).mkdir()
    
    # Archivos Python
    python_files = {
        "src/__init__.py": "",
        "src/main.py": '''
"""
Archivo principal del proyecto de prueba.
"""

def main():
    print("¡Hola desde el proyecto de prueba!")
    return 0

if __name__ == "__main__":
    main()
''',
        "src/utils.py": '''
"""
Utilidades para el proyecto de prueba.
"""

import json
from typing import Any, Dict

def load_config(path: str) -> Dict[str, Any]:
    """Carga configuración desde un archivo JSON."""
    with open(path, 'r') as f:
        return json.load(f)

def save_data(data: Dict[str, Any], path: str) -> bool:
    """Guarda datos en un archivo JSON."""
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception:
        return False
''',
        "src/models.py": '''
"""
Modelos de datos para el proyecto de prueba.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class User:
    """Representa un usuario del sistema."""
    id: int
    name: str
    email: str
    active: bool = True
    
    def greet(self) -> str:
        """Retorna un saludo personalizado."""
        return f"Hola, {self.name}!"

@dataclass
class Product:
    """Representa un producto en el inventario."""
    id: int
    name: str
    price: float
    stock: int = 0
    
    def update_stock(self, quantity: int) -> None:
        """Actualiza el stock del producto."""
        self.stock += quantity
        if self.stock < 0:
            self.stock = 0
''',
        "tests/__init__.py": "",
        "tests/test_utils.py": '''
"""
Pruebas para el módulo utils.
"""

import pytest
from src.utils import load_config, save_data

def test_load_config(tmp_path):
    """Prueba la carga de configuración."""
    config_file = tmp_path / "config.json"
    config_file.write_text('{"debug": true, "max_items": 100}')
    
    config = load_config(str(config_file))
    assert config["debug"] == True
    assert config["max_items"] == 100

def test_save_data(tmp_path):
    """Prueba el guardado de datos."""
    data = {"test": 123, "hello": "world"}
    data_file = tmp_path / "data.json"
    
    result = save_data(data, str(data_file))
    assert result == True
    assert data_file.exists()
''',
        "tests/test_models.py": '''
"""
Pruebas para los modelos.
"""

from src.models import User, Product

def test_user_greet():
    """Prueba el saludo del usuario."""
    user = User(id=1, name="Juan", email="juan@test.com")
    assert user.greet() == "Hola, Juan!"

def test_product_update_stock():
    """Prueba la actualización de stock."""
    product = Product(id=1, name="Laptop", price=999.99, stock=10)
    product.update_stock(5)
    assert product.stock == 15
    
    product.update_stock(-20)
    assert product.stock == 0
''',
        "requirements.txt": '''
pytest>=7.0.0
pytest-cov>=4.0.0
''',
        "README.md": '''
# Proyecto de Prueba

Este es un proyecto de prueba para ANALYZERBRAIN.

## Estructura

- `src/` - Código fuente
- `tests/` - Pruebas unitarias
- `docs/` - Documentación
- `config/` - Configuraciones
- `data/` - Datos

## Uso

Para ejecutar las pruebas:

```bash
pytest tests/