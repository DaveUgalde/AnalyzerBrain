"""
Serialization - Utilidades para serialización y deserialización de datos.
Soporte para JSON, YAML, binario y formatos personalizados con validación.
"""

import json
import yaml
import pickle
import msgpack
import base64
import zlib
import gzip
from typing import Dict, List, Optional, Union, Any, TypeVar, Type
from pathlib import Path
from datetime import datetime, date, time
from decimal import Decimal
from enum import Enum
import dataclasses
from dataclasses import is_dataclass, asdict
import logging
from ..core.exceptions import BrainException, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar('T')

class SerializationFormat(str, Enum):
    """Formatos de serialización soportados."""
    JSON = "json"
    YAML = "yaml"
    BINARY = "binary"
    MSGPACK = "msgpack"
    XML = "xml"
    PROTOBUF = "protobuf"

class CustomJSONEncoder(json.JSONEncoder):
    """JSON encoder personalizado que maneja tipos complejos."""
    
    def default(self, obj):
        if isinstance(obj, (datetime, date, time)):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, Enum):
            return obj.value
        elif is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, 'dict'):
            return obj.dict()
        elif isinstance(obj, (set, frozenset)):
            return list(obj)
        elif isinstance(obj, bytes):
            return base64.b64encode(obj).decode('ascii')
        
        # Intentar serializar objetos complejos
        try:
            return super().default(obj)
        except TypeError:
            # Para objetos no serializables, retornar representación string
            return str(obj)

class Serialization:
    """
    Utilidades para serialización de datos con soporte múltiples formatos
    y manejo robusto de errores.
    """
    
    # Cache para serializadores cargados
    _serializers = {}
    
    @staticmethod
    def serialize_to_json(
        data: Any,
        pretty: bool = True,
        ensure_ascii: bool = False,
        sort_keys: bool = False,
        default_handler: Optional[Callable] = None,
        max_depth: int = 10
    ) -> str:
        """
        Serializa datos a formato JSON.
        
        Args:
            data: Datos a serializar
            pretty: Formatear con indentación
            ensure_ascii: Escapar caracteres no ASCII
            sort_keys: Ordenar claves de diccionarios
            default_handler: Handler personalizado para tipos no serializables
            max_depth: Profundidad máxima para serialización recursiva
            
        Returns:
            String JSON
        
        Raises:
            BrainException: Si hay error en la serialización
        """
        try:
            encoder = CustomJSONEncoder if default_handler is None else None
            
            if default_handler:
                # Usar handler personalizado
                def _default_handler(obj):
                    try:
                        return default_handler(obj)
                    except Exception:
                        return str(obj)
                
                kwargs = {
                    'default': _default_handler,
                    'ensure_ascii': ensure_ascii,
                    'sort_keys': sort_keys,
                    'indent': 2 if pretty else None,
                    'separators': (',', ': ') if pretty else (',', ':')
                }
            else:
                # Usar encoder personalizado
                kwargs = {
                    'cls': encoder,
                    'ensure_ascii': ensure_ascii,
                    'sort_keys': sort_keys,
                    'indent': 2 if pretty else None,
                    'separators': (',', ': ') if pretty else (',', ':')
                }
            
            # Serializar con control de profundidad
            result = json.dumps(
                Serialization._limit_depth(data, max_depth),
                **kwargs
            )
            
            return result
            
        except (TypeError, ValueError) as e:
            logger.error(f"JSON serialization failed: {e}")
            raise BrainException(f"JSON serialization failed: {e}")
    
    @staticmethod
    def deserialize_from_json(
        json_str: str,
        encoding: str = 'utf-8',
        object_hook: Optional[Callable] = None,
        parse_float: Optional[Callable] = None,
        parse_int: Optional[Callable] = None,
        parse_constant: Optional[Callable] = None,
        object_pairs_hook: Optional[Callable] = None
    ) -> Any:
        """
        Deserializa datos desde formato JSON.
        
        Args:
            json_str: String JSON a deserializar
            encoding: Encoding del string
            object_hook: Hook para objetos personalizados
            parse_float: Función para parsear floats
            parse_int: Función para parsear ints
            parse_constant: Función para parsear constantes
            object_pairs_hook: Hook para pares de objetos
            
        Returns:
            Datos deserializados
            
        Raises:
            BrainException: Si hay error en la deserialización
            ValidationError: Si el JSON es inválido
        """
        try:
            # Validar estructura básica
            if not json_str or not json_str.strip():
                raise ValidationError("Empty JSON string")
            
            kwargs = {}
            if object_hook:
                kwargs['object_hook'] = object_hook
            if parse_float:
                kwargs['parse_float'] = parse_float
            if parse_int:
                kwargs['parse_int'] = parse_int
            if parse_constant:
                kwargs['parse_constant'] = parse_constant
            if object_pairs_hook:
                kwargs['object_pairs_hook'] = object_pairs_hook
            
            result = json.loads(json_str, **kwargs)
            
            # Validar tipos básicos
            Serialization._validate_deserialized(result)
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            raise ValidationError(f"Invalid JSON: {e.msg} at line {e.lineno} column {e.colno}")
        except Exception as e:
            logger.error(f"JSON deserialization failed: {e}")
            raise BrainException(f"JSON deserialization failed: {e}")
    
    @staticmethod
    def serialize_to_yaml(
        data: Any,
        default_flow_style: bool = False,
        encoding: str = 'utf-8',
        allow_unicode: bool = True,
        sort_keys: bool = False,
        default_style: Optional[str] = None,
        default_tag: Optional[str] = None,
        canonical: bool = False,
        indent: Optional[int] = 2,
        width: Optional[int] = 80,
        line_break: Optional[str] = None
    ) -> str:
        """
        Serializa datos a formato YAML.
        
        Args:
            data: Datos a serializar
            default_flow_style: Usar estilo de flujo por defecto
            encoding: Encoding de salida
            allow_unicode: Permitir caracteres Unicode
            sort_keys: Ordenar claves de diccionarios
            default_style: Estilo por defecto para escalares
            default_tag: Tag por defecto
            canonical: Forma canónica
            indent: Nivel de indentación
            width: Ancho máximo de línea
            line_break: Carácter de salto de línea
            
        Returns:
            String YAML
            
        Raises:
            BrainException: Si hay error en la serialización
        """
        try:
            # Configurar Dumper
            class CustomYamlDumper(yaml.SafeDumper):
                pass
            
            # Agregar representantes para tipos especiales
            def datetime_representer(dumper, data):
                return dumper.represent_scalar(
                    'tag:yaml.org,2002:timestamp',
                    data.isoformat()
                )
            
            def date_representer(dumper, data):
                return dumper.represent_scalar(
                    'tag:yaml.org,2002:timestamp',
                    data.isoformat()
                )
            
            CustomYamlDumper.add_representer(datetime, datetime_representer)
            CustomYamlDumper.add_representer(date, date_representer)
            CustomYamlDumper.add_representer(
                Decimal,
                lambda dumper, data: dumper.represent_scalar(
                    'tag:yaml.org,2002:float',
                    str(float(data))
                )
            )
            
            kwargs = {
                'Dumper': CustomYamlDumper,
                'default_flow_style': default_flow_style,
                'allow_unicode': allow_unicode,
                'encoding': encoding,
                'sort_keys': sort_keys,
                'canonical': canonical,
                'indent': indent,
                'width': width,
                'line_break': line_break
            }
            
            if default_style:
                kwargs['default_style'] = default_style
            
            result = yaml.dump(data, **kwargs)
            return result
            
        except yaml.YAMLError as e:
            logger.error(f"YAML serialization failed: {e}")
            raise BrainException(f"YAML serialization failed: {e}")
        except Exception as e:
            logger.error(f"YAML serialization failed: {e}")
            raise BrainException(f"YAML serialization failed: {e}")
    
    @staticmethod
    def deserialize_from_yaml(
        yaml_str: str,
        encoding: str = 'utf-8',
        loader: Optional[yaml.Loader] = None,
        **kwargs
    ) -> Any:
        """
        Deserializa datos desde formato YAML.
        
        Args:
            yaml_str: String YAML a deserializar
            encoding: Encoding del string
            loader: Loader YAML personalizado
            **kwargs: Argumentos adicionales para yaml.safe_load
            
        Returns:
            Datos deserializados
            
        Raises:
            BrainException: Si hay error en la deserialización
            ValidationError: Si el YAML es inválido
        """
        try:
            if not yaml_str or not yaml_str.strip():
                raise ValidationError("Empty YAML string")
            
            # Usar SafeLoader por defecto para seguridad
            if loader is None:
                loader = yaml.SafeLoader
            
            result = yaml.load(yaml_str, Loader=loader, **kwargs)
            
            # Validar tipos básicos
            Serialization._validate_deserialized(result)
            
            return result
            
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML: {e}")
            raise ValidationError(f"Invalid YAML: {e}")
        except Exception as e:
            logger.error(f"YAML deserialization failed: {e}")
            raise BrainException(f"YAML deserialization failed: {e}")
    
    @staticmethod
    def serialize_to_binary(
        data: Any,
        protocol: int = pickle.HIGHEST_PROTOCOL,
        compress: bool = True,
        compression_level: int = 6
    ) -> bytes:
        """
        Serializa datos a formato binario (pickle).
        
        Args:
            data: Datos a serializar
            protocol: Protocolo de pickle
            compress: Comprimir datos serializados
            compression_level: Nivel de compresión (0-9)
            
        Returns:
            Bytes serializados
            
        Raises:
            BrainException: Si hay error en la serialización
        """
        try:
            # Serializar con pickle
            binary_data = pickle.dumps(data, protocol=protocol)
            
            # Comprimir si está habilitado
            if compress:
                binary_data = zlib.compress(binary_data, level=compression_level)
            
            return binary_data
            
        except pickle.PicklingError as e:
            logger.error(f"Binary serialization failed: {e}")
            raise BrainException(f"Binary serialization failed: {e}")
        except Exception as e:
            logger.error(f"Binary serialization failed: {e}")
            raise BrainException(f"Binary serialization failed: {e}")
    
    @staticmethod
    def deserialize_from_binary(
        binary_data: bytes,
        compressed: bool = True,
        encoding: str = 'latin1',
        fix_imports: bool = True,
        errors: str = 'strict'
    ) -> Any:
        """
        Deserializa datos desde formato binario (pickle).
        
        Args:
            binary_data: Bytes a deserializar
            compressed: Si los datos están comprimidos
            encoding: Encoding para strings
            fix_imports: Arreglar imports para Python 2
            errors: Manejo de errores de encoding
            
        Returns:
            Datos deserializados
            
        Raises:
            BrainException: Si hay error en la deserialización
            ValidationError: Si los datos binarios son inválidos
        """
        try:
            if not binary_data:
                raise ValidationError("Empty binary data")
            
            # Descomprimir si es necesario
            if compressed:
                try:
                    binary_data = zlib.decompress(binary_data)
                except zlib.error:
                    # Puede que no esté comprimido realmente
                    pass
            
            # Deserializar
            result = pickle.loads(
                binary_data,
                encoding=encoding,
                fix_imports=fix_imports,
                errors=errors
            )
            
            # Validar tipos básicos
            Serialization._validate_deserialized(result)
            
            return result
            
        except pickle.UnpicklingError as e:
            logger.error(f"Invalid binary data: {e}")
            raise ValidationError(f"Invalid binary data: {e}")
        except Exception as e:
            logger.error(f"Binary deserialization failed: {e}")
            raise BrainException(f"Binary deserialization failed: {e}")
    
    @staticmethod
    def serialize(
        data: Any,
        format: SerializationFormat = SerializationFormat.JSON,
        **kwargs
    ) -> Union[str, bytes]:
        """
        Serializa datos al formato especificado.
        
        Args:
            data: Datos a serializar
            format: Formato de serialización
            **kwargs: Argumentos específicos del formato
            
        Returns:
            Datos serializados (string o bytes)
            
        Raises:
            ValueError: Si el formato no es soportado
        """
        if format == SerializationFormat.JSON:
            return Serialization.serialize_to_json(data, **kwargs)
        elif format == SerializationFormat.YAML:
            return Serialization.serialize_to_yaml(data, **kwargs)
        elif format == SerializationFormat.BINARY:
            return Serialization.serialize_to_binary(data, **kwargs)
        elif format == SerializationFormat.MSGPACK:
            return Serialization._serialize_msgpack(data, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def deserialize(
        serialized_data: Union[str, bytes],
        format: SerializationFormat = SerializationFormat.JSON,
        **kwargs
    ) -> Any:
        """
        Deserializa datos desde el formato especificado.
        
        Args:
            serialized_data: Datos serializados
            format: Formato de serialización
            **kwargs: Argumentos específicos del formato
            
        Returns:
            Datos deserializados
            
        Raises:
            ValueError: Si el formato no es soportado
        """
        if format == SerializationFormat.JSON:
            if isinstance(serialized_data, bytes):
                serialized_data = serialized_data.decode('utf-8')
            return Serialization.deserialize_from_json(serialized_data, **kwargs)
        elif format == SerializationFormat.YAML:
            if isinstance(serialized_data, bytes):
                serialized_data = serialized_data.decode('utf-8')
            return Serialization.deserialize_from_yaml(serialized_data, **kwargs)
        elif format == SerializationFormat.BINARY:
            if isinstance(serialized_data, str):
                serialized_data = serialized_data.encode('latin1')
            return Serialization.deserialize_from_binary(serialized_data, **kwargs)
        elif format == SerializationFormat.MSGPACK:
            return Serialization._deserialize_msgpack(serialized_data, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def validate_serialization(
        data: Any,
        format: SerializationFormat = SerializationFormat.JSON,
        roundtrip: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Valida que los datos puedan ser serializados y deserializados correctamente.
        
        Args:
            data: Datos a validar
            format: Formato a validar
            roundtrip: Validar ida y vuelta (serializar -> deserializar)
            **kwargs: Argumentos para serialización/deserialización
            
        Returns:
            Diccionario con resultados de validación
        """
        results = {
            'format': format.value,
            'can_serialize': False,
            'can_deserialize': False,
            'roundtrip_success': False,
            'size_bytes': 0,
            'serialization_time': 0,
            'deserialization_time': 0,
            'errors': []
        }
        
        import time
        
        try:
            # Serializar
            start = time.time()
            serialized = Serialization.serialize(data, format, **kwargs)
            results['serialization_time'] = time.time() - start
            
            results['can_serialize'] = True
            if isinstance(serialized, str):
                results['size_bytes'] = len(serialized.encode('utf-8'))
            else:
                results['size_bytes'] = len(serialized)
            
            # Deserializar
            if roundtrip:
                start = time.time()
                deserialized = Serialization.deserialize(serialized, format, **kwargs)
                results['deserialization_time'] = time.time() - start
                
                results['can_deserialize'] = True
                
                # Verificar equivalencia (aproximada)
                results['roundtrip_success'] = Serialization._are_equivalent(data, deserialized)
                
        except Exception as e:
            results['errors'].append({
                'step': 'serialization' if not results['can_serialize'] else 'deserialization',
                'error': str(e),
                'type': type(e).__name__
            })
        
        return results
    
    # ========== MÉTODOS PRIVADOS ==========
    
    @staticmethod
    def _limit_depth(data: Any, max_depth: int, current_depth: int = 0) -> Any:
        """Limita la profundidad de datos recursivos."""
        if current_depth >= max_depth:
            if isinstance(data, (dict, list, set, tuple)):
                return f"[Max depth {max_depth} reached]"
            return data
        
        if isinstance(data, dict):
            return {
                key: Serialization._limit_depth(value, max_depth, current_depth + 1)
                for key, value in data.items()
            }
        elif isinstance(data, (list, set, tuple)):
            return [
                Serialization._limit_depth(item, max_depth, current_depth + 1)
                for item in data
            ]
        else:
            return data
    
    @staticmethod
    def _validate_deserialized(data: Any, max_depth: int = 50, current_depth: int = 0) -> None:
        """Valida datos deserializados para prevenir estructuras peligrosas."""
        if current_depth >= max_depth:
            raise ValidationError(f"Data structure exceeds maximum depth of {max_depth}")
        
        # Verificar tipos peligrosos
        if isinstance(data, type):
            raise ValidationError("Type objects are not allowed in deserialized data")
        
        # Verificar recursivamente
        if isinstance(data, dict):
            for key, value in data.items():
                if not isinstance(key, (str, int, float, bool, type(None))):
                    raise ValidationError(f"Invalid dictionary key type: {type(key)}")
                Serialization._validate_deserialized(value, max_depth, current_depth + 1)
        
        elif isinstance(data, (list, set, tuple)):
            for item in data:
                Serialization._validate_deserialized(item, max_depth, current_depth + 1)
    
    @staticmethod
    def _are_equivalent(obj1: Any, obj2: Any, tolerance: float = 1e-10) -> bool:
        """Compara si dos objetos son equivalentes (para validación de roundtrip)."""
        import math
        
        # Comparación por tipo
        if type(obj1) != type(obj2):
            # Permitir algunas conversiones numéricas
            if isinstance(obj1, (int, float)) and isinstance(obj2, (int, float)):
                return abs(float(obj1) - float(obj2)) <= tolerance
            return False
        
        # Comparación de valores primitivos
        if isinstance(obj1, (str, int, bool, type(None))):
            return obj1 == obj2
        
        # Comparación de floats con tolerancia
        if isinstance(obj1, float):
            return math.isclose(obj1, obj2, rel_tol=tolerance, abs_tol=tolerance)
        
        # Comparación de diccionarios
        if isinstance(obj1, dict):
            if set(obj1.keys()) != set(obj2.keys()):
                return False
            for key in obj1:
                if not Serialization._are_equivalent(obj1[key], obj2[key], tolerance):
                    return False
            return True
        
        # Comparación de secuencias
        if isinstance(obj1, (list, tuple, set)):
            if len(obj1) != len(obj2):
                return False
            
            # Para conjuntos, el orden no importa
            if isinstance(obj1, set):
                return all(
                    any(Serialization._are_equivalent(item1, item2, tolerance) 
                        for item2 in obj2)
                    for item1 in obj1
                )
            
            # Para listas/tuplas, el orden importa
            for item1, item2 in zip(obj1, obj2):
                if not Serialization._are_equivalent(item1, item2, tolerance):
                    return False
            return True
        
        # Para objetos complejos, comparar representación string
        return str(obj1) == str(obj2)
    
    @staticmethod
    def _serialize_msgpack(data: Any, **kwargs) -> bytes:
        """Serializa datos a MessagePack."""
        try:
            import msgpack
            
            # Configurar handlers para tipos especiales
            def default_handler(obj):
                if isinstance(obj, (datetime, date, time)):
                    return {'__datetime__': obj.isoformat()}
                elif isinstance(obj, Decimal):
                    return float(obj)
                elif isinstance(obj, Enum):
                    return obj.value
                elif is_dataclass(obj) and not isinstance(obj, type):
                    return asdict(obj)
                elif hasattr(obj, 'to_dict'):
                    return obj.to_dict()
                elif isinstance(obj, bytes):
                    return {'__bytes__': base64.b64encode(obj).decode('ascii')}
                raise TypeError(f"Object of type {type(obj)} is not msgpack serializable")
            
            return msgpack.packb(data, default=default_handler, **kwargs)
            
        except ImportError:
            raise ValueError("MessagePack not installed. Install with: pip install msgpack")
        except Exception as e:
            logger.error(f"MessagePack serialization failed: {e}")
            raise BrainException(f"MessagePack serialization failed: {e}")
    
    @staticmethod
    def _deserialize_msgpack(data: bytes, **kwargs) -> Any:
        """Deserializa datos desde MessagePack."""
        try:
            import msgpack
            
            # Configurar handlers para tipos especiales
            def object_hook(obj):
                if isinstance(obj, dict):
                    if '__datetime__' in obj:
                        from dateutil.parser import isoparse
                        return isoparse(obj['__datetime__'])
                    elif '__bytes__' in obj:
                        return base64.b64decode(obj['__bytes__'])
                return obj
            
            result = msgpack.unpackb(data, object_hook=object_hook, **kwargs)
            
            # Validar tipos básicos
            Serialization._validate_deserialized(result)
            
            return result
            
        except ImportError:
            raise ValueError("MessagePack not installed. Install with: pip install msgpack")
        except Exception as e:
            logger.error(f"MessagePack deserialization failed: {e}")
            raise BrainException(f"MessagePack deserialization failed: {e}")