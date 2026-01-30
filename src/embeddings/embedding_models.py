"""
EmbeddingModels - Gestión de modelos de embeddings.
Carga, descarga y gestiona modelos de embeddings para texto y código.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path
import warnings
from datetime import datetime
import hashlib
from pydantic import BaseModel, Field, validator
import torch
from transformers import AutoModel, AutoTokenizer

class ModelType(Enum):
    """Tipos de modelos de embeddings."""
    TEXT = "text"
    CODE = "code"
    MULTILINGUAL = "multilingual"
    HYBRID = "hybrid"

class ModelDevice(Enum):
    """Dispositivos para ejecutar modelos."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    AUTO = "auto"

@dataclass
class ModelInfo:
    """Información de un modelo de embeddings."""
    name: str
    type: ModelType
    dimensions: int
    max_tokens: int
    languages: List[str]
    description: str
    url: Optional[str] = None
    version: str = "1.0.0"
    loaded: bool = False
    loading_time: Optional[float] = None
    memory_usage_mb: Optional[float] = None

class ModelConfig(BaseModel):
    """Configuración para modelos de embeddings."""
    default_text_model: str = "all-MiniLM-L6-v2"
    default_code_model: str = "microsoft/codebert-base"
    default_multilingual_model: str = "intfloat/multilingual-e5-large"
    
    available_models: Dict[ModelType, List[str]] = Field(
        default_factory=lambda: {
            ModelType.TEXT: [
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2",
                "multi-qa-mpnet-base-dot-v1"
            ],
            ModelType.CODE: [
                "microsoft/codebert-base",
                "microsoft/graphcodebert-base",
                "codeparrot/codeparrot"
            ],
            ModelType.MULTILINGUAL: [
                "intfloat/multilingual-e5-large",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            ]
        }
    )
    
    cache_dir: str = "./models"
    device: ModelDevice = ModelDevice.AUTO
    max_memory_mb: int = 4096
    quantization: bool = False
    
    class Config:
        arbitrary_types_allowed = True

class EmbeddingModels:
    """
    Gestor de modelos de embeddings.
    
    Responsabilidades:
    1. Cargar y descargar modelos de embeddings
    2. Gestionar el ciclo de vida de los modelos
    3. Proporcionar información sobre modelos disponibles
    4. Optimizar el uso de memoria y GPU
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Inicializa el gestor de modelos.
        
        Args:
            config: Configuración de modelos (opcional)
        """
        self.config = config or ModelConfig()
        self._models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}
        self._model_info: Dict[str, ModelInfo] = {}
        
        # Crear directorio de caché si no existe
        os.makedirs(self.config.cache_dir, exist_ok=True)
        
        # Determinar dispositivo
        self.device = self._determine_device()
        
        # Precargar información de modelos disponibles
        self._load_model_info()
    
    def load_model(self, model_name: str, device: Optional[str] = None) -> bool:
        """
        Carga un modelo de embeddings.
        
        Args:
            model_name: Nombre del modelo
            device: Dispositivo a usar (opcional)
            
        Returns:
            bool: True si el modelo se cargó exitosamente
            
        Raises:
            ValueError: Si el modelo no está disponible
            RuntimeError: Si no hay suficiente memoria
        """
        if model_name in self._models:
            return True  # Ya está cargado
        
        # Verificar que el modelo está disponible
        if not self._is_model_available(model_name):
            raise ValueError(f"Model {model_name} is not available")
        
        device = device or str(self.device.value)
        start_time = datetime.now()
        
        try:
            # Cargar tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.config.cache_dir
            )
            
            # Cargar modelo con configuración optimizada
            model_args = {
                "pretrained_model_name_or_path": model_name,
                "cache_dir": self.config.cache_dir,
                "local_files_only": False
            }
            
            # Aplicar cuantización si está habilitada y es CPU
            if self.config.quantization and device == "cpu":
                model_args["load_in_8bit"] = True
            
            model = AutoModel.from_pretrained(**model_args)
            
            # Mover al dispositivo
            model.to(device)
            model.eval()  # Modo evaluación
            
            # Almacenar
            self._models[model_name] = model
            self._tokenizers[model_name] = tokenizer
            
            # Actualizar información
            info = self._model_info.get(model_name)
            if info:
                info.loaded = True
                info.loading_time = (datetime.now() - start_time).total_seconds()
                
                # Calcular uso de memoria
                if hasattr(model, "get_memory_footprint"):
                    info.memory_usage_mb = model.get_memory_footprint() / (1024 * 1024)
                else:
                    param_count = sum(p.numel() for p in model.parameters())
                    info.memory_usage_mb = (param_count * 4) / (1024 * 1024)  # 4 bytes por float32
            
            print(f"Model {model_name} loaded successfully on {device}")
            return True
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")
    
    def unload_model(self, model_name: str) -> bool:
        """
        Descarga un modelo de memoria.
        
        Args:
            model_name: Nombre del modelo a descargar
            
        Returns:
            bool: True si se descargó exitosamente
        """
        if model_name not in self._models:
            return False
        
        try:
            # Liberar memoria GPU si es necesario
            model = self._models[model_name]
            if hasattr(model, "cpu"):
                model.cpu()
            
            # Eliminar referencias
            del self._models[model_name]
            del self._tokenizers[model_name]
            
            # Actualizar información
            if model_name in self._model_info:
                self._model_info[model_name].loaded = False
                self._model_info[model_name].memory_usage_mb = None
            
            # Forzar garbage collection
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            warnings.warn(f"Error unloading model {model_name}: {str(e)}")
            return False
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        Lista todos los modelos disponibles.
        
        Returns:
            Lista de diccionarios con información de modelos
        """
        models = []
        
        for model_type, model_list in self.config.available_models.items():
            for model_name in model_list:
                info = self._model_info.get(model_name)
                if info:
                    models.append({
                        "name": model_name,
                        "type": model_type.value,
                        "dimensions": info.dimensions,
                        "loaded": info.loaded,
                        "languages": info.languages,
                        "description": info.description
                    })
        
        return models
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Obtiene información detallada de un modelo.
        
        Args:
            model_name: Nombre del modelo
            
        Returns:
            Dict con información del modelo
        """
        if model_name not in self._model_info:
            raise ValueError(f"Model {model_name} not found")
        
        info = self._model_info[model_name]
        return {
            "name": info.name,
            "type": info.type.value,
            "dimensions": info.dimensions,
            "max_tokens": info.max_tokens,
            "languages": info.languages,
            "description": info.description,
            "loaded": info.loaded,
            "loading_time_seconds": info.loading_time,
            "memory_usage_mb": info.memory_usage_mb,
            "url": info.url,
            "version": info.version
        }
    
    def validate_model_compatibility(self, model_name: str, task_type: str) -> bool:
        """
        Valida si un modelo es compatible con una tarea específica.
        
        Args:
            model_name: Nombre del modelo
            task_type: Tipo de tarea (text, code, retrieval, etc.)
            
        Returns:
            bool: True si el modelo es compatible
        """
        if model_name not in self._model_info:
            return False
        
        info = self._model_info[model_name]
        
        # Validaciones básicas
        if task_type == "code" and info.type != ModelType.CODE:
            return False
        
        if task_type == "multilingual" and info.type != ModelType.MULTILINGUAL:
            return False
        
        return True
    
    def get_model_dimensions(self, model_name: str) -> int:
        """
        Obtiene las dimensiones del embedding de un modelo.
        
        Args:
            model_name: Nombre del modelo
            
        Returns:
            int: Número de dimensiones
        """
        if model_name not in self._model_info:
            raise ValueError(f"Model {model_name} not found")
        
        return self._model_info[model_name].dimensions
    
    def warmup_model(self, model_name: str, num_iterations: int = 10) -> None:
        """
        Realiza inferencias de calentamiento para optimizar el modelo.
        
        Args:
            model_name: Nombre del modelo
            num_iterations: Número de iteraciones de calentamiento
        """
        if model_name not in self._models:
            raise ValueError(f"Model {model_name} is not loaded")
        
        model = self._models[model_name]
        tokenizer = self._tokenizers[model_name]
        
        # Texto de prueba
        test_text = "This is a warmup inference to optimize the model."
        
        with torch.no_grad():
            for _ in range(num_iterations):
                inputs = tokenizer(
                    test_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(model.device)
                
                outputs = model(**inputs)
                
                # Solo necesitamos ejecutar, no procesar resultados
                _ = outputs.last_hidden_state.mean(dim=1)
    
    def get_tokenizer(self, model_name: str):
        """
        Obtiene el tokenizer de un modelo.
        
        Args:
            model_name: Nombre del modelo
            
        Returns:
            Tokenizer del modelo
        """
        if model_name not in self._tokenizers:
            raise ValueError(f"Model {model_name} is not loaded")
        
        return self._tokenizers[model_name]
    
    def get_model(self, model_name: str):
        """
        Obtiene el modelo cargado.
        
        Args:
            model_name: Nombre del modelo
            
        Returns:
            Modelo cargado
        """
        if model_name not in self._models:
            raise ValueError(f"Model {model_name} is not loaded")
        
        return self._models[model_name]
    
    # Métodos privados
    
    def _determine_device(self) -> ModelDevice:
        """Determina el mejor dispositivo disponible."""
        if self.config.device != ModelDevice.AUTO:
            return self.config.device
        
        if torch.cuda.is_available():
            return ModelDevice.CUDA
        elif torch.backends.mps.is_available():
            return ModelDevice.MPS
        else:
            return ModelDevice.CPU
    
    def _is_model_available(self, model_name: str) -> bool:
        """Verifica si un modelo está disponible."""
        for model_list in self.config.available_models.values():
            if model_name in model_list:
                return True
        
        # También verificar modelos personalizados
        return os.path.exists(os.path.join(self.config.cache_dir, model_name))
    
    def _load_model_info(self) -> None:
        """Carga información de modelos disponibles."""
        # Información predefinida para modelos populares
        predefined_info = {
            "all-MiniLM-L6-v2": ModelInfo(
                name="all-MiniLM-L6-v2",
                type=ModelType.TEXT,
                dimensions=384,
                max_tokens=256,
                languages=["en"],
                description="Fast and efficient sentence transformer model"
            ),
            "all-mpnet-base-v2": ModelInfo(
                name="all-mpnet-base-v2",
                type=ModelType.TEXT,
                dimensions=768,
                max_tokens=512,
                languages=["en"],
                description="High-quality sentence transformer model"
            ),
            "microsoft/codebert-base": ModelInfo(
                name="microsoft/codebert-base",
                type=ModelType.CODE,
                dimensions=768,
                max_tokens=512,
                languages=["multilingual"],
                description="BERT model pre-trained on programming languages"
            ),
            "intfloat/multilingual-e5-large": ModelInfo(
                name="intfloat/multilingual-e5-large",
                type=ModelType.MULTILINGUAL,
                dimensions=1024,
                max_tokens=512,
                languages=["multilingual"],
                description="Multilingual embedding model supporting 100+ languages"
            )
        }
        
        self._model_info = predefined_info
        
        # Intentar cargar información adicional de archivos locales
        for model_file in Path(self.config.cache_dir).glob("*.info"):
            try:
                with open(model_file, 'r') as f:
                    import json
                    data = json.load(f)
                    
                    info = ModelInfo(
                        name=data.get("name", model_file.stem),
                        type=ModelType(data.get("type", "text")),
                        dimensions=data.get("dimensions", 384),
                        max_tokens=data.get("max_tokens", 256),
                        languages=data.get("languages", ["en"]),
                        description=data.get("description", "Custom model"),
                        url=data.get("url"),
                        version=data.get("version", "1.0.0")
                    )
                    
                    self._model_info[info.name] = info
                    
            except Exception as e:
                warnings.warn(f"Failed to load model info from {model_file}: {str(e)}")
    
    def _calculate_model_hash(self, model_name: str) -> str:
        """Calcula hash de un modelo para caché."""
        model_path = os.path.join(self.config.cache_dir, model_name)
        
        if not os.path.exists(model_path):
            return hashlib.md5(model_name.encode()).hexdigest()
        
        hasher = hashlib.sha256()
        for root, _, files in os.walk(model_path):
            for file in sorted(files):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b''):
                            hasher.update(chunk)
                except:
                    pass
        
        return hasher.hexdigest()

# Ejemplo de uso
if __name__ == "__main__":
    models = EmbeddingModels()
    
    # Listar modelos disponibles
    available = models.list_available_models()
    print(f"Available models: {len(available)}")
    
    # Cargar un modelo
    success = models.load_model("all-MiniLM-L6-v2")
    print(f"Model loaded: {success}")
    
    # Obtener información
    info = models.get_model_info("all-MiniLM-L6-v2")
    print(f"Model dimensions: {info['dimensions']}")
    
    # Calentamiento
    models.warmup_model("all-MiniLM-L6-v2")
    print("Model warmed up successfully")