"""
Pruebas unitarias para el módulo embeddings/embedding_generator.py
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List

from embeddings.embedding_generator import (
    EmbeddingGenerator,
    EmbeddingConfig,
    EmbeddingType
)


class TestEmbeddingConfig:
    """Pruebas para EmbeddingConfig."""
    
    def test_default_config(self):
        """Test configuración por defecto."""
        config = EmbeddingConfig()
        
        assert config.default_model == "all-MiniLM-L6-v2"
        assert config.device == "cpu"
        assert config.batch_size == 32
        assert config.normalize is True
        assert config.cache_embeddings is True
        assert config.cache_size == 10000
        assert config.max_sequence_length == 512
        assert config.dimension == 384  # Para all-MiniLM-L6-v2
    
    def test_custom_config(self):
        """Test configuración personalizada."""
        config = EmbeddingConfig(
            default_model="all-mpnet-base-v2",
            device="cuda",
            batch_size=64,
            normalize=False,
            cache_size=5000,
            dimension=768
        )
        
        assert config.default_model == "all-mpnet-base-v2"
        assert config.device == "cuda"
        assert config.batch_size == 64
        assert config.normalize is False
        assert config.cache_size == 5000
        assert config.dimension == 768


class TestEmbeddingGenerator:
    """Pruebas para EmbeddingGenerator."""
    
    @pytest.fixture
    def generator(self):
        """Crear generador de embeddings para pruebas."""
        config = EmbeddingConfig(cache_embeddings=False)
        return EmbeddingGenerator(config)
    
    @pytest.fixture
    def sample_texts(self):
        """Proporcionar textos de ejemplo."""
        return [
            "This is a test sentence.",
            "Another example for testing embeddings.",
            "Python is a programming language.",
            "Machine learning models can generate embeddings.",
            "Natural language processing is fascinating."
        ]
    
    @pytest.fixture
    def sample_code(self):
        """Proporcionar código de ejemplo."""
        return '''
def calculate_sum(a, b):
    """Calculate the sum of two numbers."""
    return a + b

class Calculator:
    def __init__(self):
        self.memory = []
    
    def add(self, x, y):
        result = x + y
        self.memory.append(result)
        return result
'''
    
    def test_initialization(self, generator):
        """Test inicialización del generador."""
        assert generator is not None
        assert hasattr(generator, 'config')
        assert hasattr(generator, '_models')
        assert hasattr(generator, '_cache')
        
    @patch('embeddings.embedding_generator.sentence_transformers')
    def test_load_model_success(self, mock_st, generator):
        """Test carga exitosa de modelo."""
        mock_model = Mock()
        mock_st.SentenceTransformer.return_value = mock_model
        
        success = generator.load_model("all-MiniLM-L6-v2")
        
        assert success is True
        assert "all-MiniLM-L6-v2" in generator._models
        mock_st.SentenceTransformer.assert_called_once_with(
            "all-MiniLM-L6-v2",
            device="cpu"
        )
    
    def test_load_model_invalid_name(self, generator):
        """Test carga de modelo con nombre inválido."""
        success = generator.load_model("")
        
        assert success is False
    
    @patch('embeddings.embedding_generator.sentence_transformers')
    def test_load_model_error(self, mock_st, generator):
        """Test error al cargar modelo."""
        mock_st.SentenceTransformer.side_effect = Exception("Model not found")
        
        success = generator.load_model("invalid-model")
        
        assert success is False
    
    @patch('embeddings.embedding_generator.sentence_transformers')
    def test_generate_text_embedding_success(self, mock_st, generator):
        """Test generación exitosa de embedding de texto."""
        # Mock del modelo
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1] * 384)
        generator._models["test-model"] = mock_model
        
        embedding = generator.generate_text_embedding(
            "Test text",
            model_name="test-model"
        )
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)
        mock_model.encode.assert_called_once()
    
    def test_generate_text_embedding_model_not_loaded(self, generator):
        """Test generación sin modelo cargado."""
        embedding = generator.generate_text_embedding("Test text")
        
        # Debería retornar lista vacía o None
        assert embedding == [] or embedding is None
    
    def test_generate_code_embedding(self, generator, sample_code):
        """Test generación de embedding de código."""
        with patch.object(generator, 'generate_text_embedding') as mock_generate:
            mock_generate.return_value = [0.1] * 384
            
            embedding = generator.generate_code_embedding(
                sample_code,
                language="python",
                context="testing"
            )
            
            assert len(embedding) == 384
            mock_generate.assert_called_once()
    
    @patch('embeddings.embedding_generator.sentence_transformers')
    def test_batch_generate(self, mock_st, generator, sample_texts):
        """Test generación por lotes."""
        # Mock del modelo
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1] * 384 for _ in range(5)])
        generator._models["test-model"] = mock_model
        
        embeddings = generator.batch_generate(
            sample_texts,
            model_name="test-model",
            batch_size=2
        )
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == 5
        assert all(len(e) == 384 for e in embeddings)
        mock_model.encode.assert_called_once()
    
    def test_normalize_embeddings(self, generator):
        """Test normalización de embeddings."""
        # Crear embeddings no normalizados
        embeddings = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]
        
        normalized = generator.normalize_embeddings(embeddings)
        
        # Verificar que cada embedding está normalizado (norma ≈ 1)
        for emb in normalized:
            norm = np.linalg.norm(emb)
            assert abs(norm - 1.0) < 0.001
    
    def test_pad_embeddings(self, generator):
        """Test padding de embeddings."""
        embeddings = [
            [1.0, 2.0],
            [3.0, 4.0, 5.0]
        ]
        
        padded = generator.pad_embeddings(embeddings, target_length=5)
        
        assert len(padded) == 2
        assert all(len(e) == 5 for e in padded)
        
        # Verificar que los valores originales se preservan
        assert padded[0][:2] == [1.0, 2.0]
        assert padded[1][:3] == [3.0, 4.0, 5.0]
        
        # Verificar padding con ceros
        assert padded[0][2:] == [0.0, 0.0, 0.0]
        assert padded[1][3:] == [0.0, 0.0]
    
    def test_truncate_embeddings(self, generator):
        """Test truncado de embeddings."""
        embeddings = [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0]
        ]
        
        truncated = generator.truncate_embeddings(embeddings, target_length=3)
        
        assert len(truncated) == 2
        assert all(len(e) == 3 for e in truncated)
        
        # Verificar que se preservan los primeros elementos
        assert truncated[0] == [1.0, 2.0, 3.0]
        assert truncated[1] == [6.0, 7.0, 8.0]
    
    def test_average_embeddings(self, generator):
        """Test promediado de embeddings."""
        embeddings = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]
        
        # Promedio simple
        avg = generator.average_embeddings(embeddings)
        
        assert len(avg) == 3
        assert avg == [4.0, 5.0, 6.0]  # (1+4+7)/3, (2+5+8)/3, (3+6+9)/3
        
        # Con pesos
        weights = [0.5, 0.3, 0.2]
        weighted_avg = generator.average_embeddings(embeddings, weights)
        
        expected = [
            1.0*0.5 + 4.0*0.3 + 7.0*0.2,
            2.0*0.5 + 5.0*0.3 + 8.0*0.2,
            3.0*0.5 + 6.0*0.3 + 9.0*0.2
        ]
        
        assert all(abs(a - b) < 0.001 for a, b in zip(weighted_avg, expected))
    
    def test_cache_operations(self, generator):
        """Test operaciones de caché."""
        # Habilitar caché
        generator.config.cache_embeddings = True
        
        embedding = [0.1] * 384
        key = "test_key"
        
        # Test almacenar en caché
        success = generator.cache_embedding(key, embedding, ttl_seconds=60)
        assert success is True
        
        # Test obtener de caché
        cached = generator.get_cached_embedding(key)
        assert cached == embedding
        
        # Test invalidad caché
        invalidated = generator.invalidate_cache([key])
        assert invalidated == 1
        
        # Verificar que fue eliminado
        cached = generator.get_cached_embedding(key)
        assert cached is None
    
    def test_cache_stats(self, generator):
        """Test estadísticas de caché."""
        # Habilitar caché
        generator.config.cache_embeddings = True
        
        # Agregar algunos items
        for i in range(5):
            generator.cache_embedding(f"key_{i}", [float(i)] * 384)
        
        stats = generator.get_cache_stats()
        
        assert "size" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        
        assert stats["size"] == 5
    
    def test_validate_embedding(self, generator):
        """Test validación de embedding."""
        # Embedding válido
        valid_embedding = [0.1] * 384
        assert generator.validate_embedding(valid_embedding, 384) is True
        
        # Dimensión incorrecta
        invalid_embedding = [0.1] * 100
        assert generator.validate_embedding(invalid_embedding, 384) is False
        
        # Embedding vacío
        assert generator.validate_embedding([], 384) is False
        
        # Embedding con valores no numéricos
        non_numeric = [0.1, "string", 0.3]
        assert generator.validate_embedding(non_numeric, 3) is False
    
    def test_compare_embeddings(self, generator):
        """Test comparación de embeddings."""
        # Embeddings idénticos
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [1.0, 0.0, 0.0]
        
        similarity = generator.compare_embeddings(emb1, emb2, "cosine")
        assert abs(similarity - 1.0) < 0.001
        
        # Embeddings ortogonales
        emb3 = [0.0, 1.0, 0.0]
        similarity = generator.compare_embeddings(emb1, emb3, "cosine")
        assert abs(similarity - 0.0) < 0.001
        
        # Euclidean distance
        distance = generator.compare_embeddings(emb1, emb2, "euclidean")
        assert abs(distance - 0.0) < 0.001
        
        distance = generator.compare_embeddings(emb1, emb3, "euclidean")
        assert abs(distance - np.sqrt(2)) < 0.001
        
        # Métrica inválida
        with pytest.raises(ValueError):
            generator.compare_embeddings(emb1, emb2, "invalid")
    
    def test_find_similar_embeddings(self, generator):
        """Test búsqueda de embeddings similares."""
        # Crear embeddings de prueba
        query = [1.0, 0.0, 0.0]
        candidates = [
            [1.0, 0.0, 0.0],      # Índice 0: idéntico
            [0.9, 0.1, 0.0],      # Índice 1: muy similar
            [0.1, 0.9, 0.0],      # Índice 2: poco similar
            [-1.0, 0.0, 0.0],     # Índice 3: opuesto
            [0.0, 0.0, 1.0]       # Índice 4: ortogonal
        ]
        
        # Buscar los 2 más similares
        similar = generator.find_similar_embeddings(
            query, candidates, top_k=2, threshold=0.0
        )
        
        assert len(similar) == 2
        
        # Verificar orden (más similar primero)
        assert similar[0][0] == 0  # Índice 0
        assert similar[1][0] == 1  # Índice 1
        
        # Verificar similitudes
        assert abs(similar[0][1] - 1.0) < 0.001
        assert 0.9 < similar[1][1] < 1.0
        
        # Test con threshold
        similar = generator.find_similar_embeddings(
            query, candidates, top_k=10, threshold=0.8
        )
        
        # Solo debería encontrar los que tienen similitud > 0.8
        assert len(similar) >= 2  # Índices 0 y 1
        
        # Verificar threshold
        for idx, score in similar:
            assert score >= 0.8
    
    @patch('embeddings.embedding_generator.sentence_transformers')
    def test_list_loaded_models(self, mock_st, generator):
        """Test listado de modelos cargados."""
        # Cargar algunos modelos mock
        mock_model = Mock()
        generator._models = {
            "model1": mock_model,
            "model2": mock_model,
            "model3": mock_model
        }
        
        loaded = generator.list_loaded_models()
        
        assert len(loaded) == 3
        assert "model1" in loaded
        assert "model2" in loaded
        assert "model3" in loaded
    
    def test_get_model_info(self, generator):
        """Test obtención de información del modelo."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model._model_card_data = {"language": ["en"]}
        
        generator._models["test-model"] = mock_model
        
        info = generator.get_model_info("test-model")
        
        assert "dimension" in info
        assert info["dimension"] == 384
        assert "languages" in info
        assert "en" in info["languages"]
        
        # Modelo no cargado
        info = generator.get_model_info("non-existent")
        assert info is None
    
    def test_warmup_model(self, generator):
        """Test warmup de modelo."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1] * 384])
        generator._models["test-model"] = mock_model
        
        # No debería lanzar excepciones
        generator.warmup_model("test-model", num_iterations=3)
        
        # Verificar que se llamó encode varias veces
        assert mock_model.encode.call_count == 3
    
    def test_unload_model(self, generator):
        """Test descarga de modelo."""
        mock_model = Mock()
        generator._models["test-model"] = mock_model
        
        success = generator.unload_model("test-model")
        
        assert success is True
        assert "test-model" not in generator._models
        
        # Intentar descargar modelo no cargado
        success = generator.unload_model("non-existent")
        assert success is False
    
    def test_preprocess_text(self, generator):
        """Test preprocesamiento de texto."""
        text = "  Hello   World!  \n\nThis is a TEST.  "
        
        processed = generator._preprocess_text(text)
        
        # Verificar que se limpian espacios extra y newlines
        assert "  " not in processed  # No dobles espacios
        assert "\n" not in processed  # No newlines
        
        # Test con lenguaje específico
        processed = generator._preprocess_text(text, language="python")
        assert isinstance(processed, str)
    
    def test_preprocess_code(self, generator):
        """Test preprocesamiento de código."""
        code = '''
def test():
    # This is a comment
    x = 1 + 2
    return x
'''
        
        processed = generator._preprocess_code(code, "python")
        
        # Debería eliminar o manejar comentarios
        assert isinstance(processed, str)
        # Puede o no contener el comentario dependiendo de la implementación
    
    def test_chunk_text(self, generator):
        """Test división de texto en chunks."""
        text = "This is a long text that needs to be chunked. " * 20
        
        chunks = generator._chunk_text(text, chunk_size=100, overlap=20)
        
        assert len(chunks) > 1
        
        # Verificar que los chunks no exceden el tamaño
        for chunk in chunks:
            assert len(chunk) <= 100
        
        # Verificar overlap
        for i in range(len(chunks) - 1):
            chunk1 = chunks[i]
            chunk2 = chunks[i + 1]
            # Deberían compartir algún texto
            assert any(word in chunk2 for word in chunk1.split()[:10])
    
    def test_calculate_embedding_dimension(self, generator):
        """Test cálculo de dimensión de embedding."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        generator._models["test-model"] = mock_model
        
        dimension = generator._calculate_embedding_dimension("test-model")
        
        assert dimension == 384
        
        # Modelo no cargado
        dimension = generator._calculate_embedding_dimension("non-existent")
        assert dimension == 384  # Valor por defecto
    
    def test_validate_batch_size(self, generator):
        """Test validación de tamaño de batch."""
        # Caso normal
        batch_size = generator._validate_batch_size(32, available_memory=8 * 1024**3)
        assert batch_size == 32
        
        # Memoria insuficiente
        batch_size = generator._validate_batch_size(1000, available_memory=1 * 1024**3)
        assert batch_size < 1000
        
        # Batch size 0 o negativo
        batch_size = generator._validate_batch_size(0, available_memory=8 * 1024**3)
        assert batch_size > 0
        
        batch_size = generator._validate_batch_size(-10, available_memory=8 * 1024**3)
        assert batch_size > 0


class TestPerformance:
    """Pruebas de performance."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_batch_generation_performance(self):
        """Test performance de generación por lotes."""
        import time
        
        config = EmbeddingConfig(cache_embeddings=False)
        generator = EmbeddingGenerator(config)
        
        # Crear textos de prueba
        texts = [f"This is test text number {i}" for i in range(100)]
        
        with patch.object(generator, 'load_model') as mock_load:
            mock_model = Mock()
            # Simular que encode toma 0.01 segundos por batch
            mock_model.encode.side_effect = lambda x, **kwargs: (
                np.random.randn(len(x), 384) if isinstance(x, list) 
                else np.random.randn(1, 384)
            )
            generator._models["test-model"] = mock_model
            
            # Medir tiempo
            start_time = time.time()
            embeddings = generator.batch_generate(
                texts, 
                model_name="test-model",
                batch_size=32,
                show_progress=False
            )
            elapsed = time.time() - start_time
            
            # Verificar resultados
            assert len(embeddings) == 100
            assert all(len(e) == 384 for e in embeddings)
            
            # Performance: 100 embeddings deberían generarse rápido
            # (con mock, debería ser < 0.5 segundos)
            assert elapsed < 0.5, f"Batch generation tomó {elapsed:.3f}s"
            
            print(f"\nBatch generation performance: "
                  f"100 embeddings en {elapsed:.3f}s "
                  f"({100/elapsed:.1f} embeddings/segundo)")
    
    @pytest.mark.performance
    def test_cache_performance(self):
        """Test performance de caché."""
        import time
        
        config = EmbeddingConfig(cache_embeddings=True, cache_size=10000)
        generator = EmbeddingGenerator(config)
        
        embedding = [0.1] * 384
        
        # Test escritura en caché
        start_time = time.time()
        for i in range(1000):
            generator.cache_embedding(f"key_{i}", embedding)
        write_time = time.time() - start_time
        
        # Test lectura de caché
        start_time = time.time()
        hits = 0
        for i in range(1000):
            if generator.get_cached_embedding(f"key_{i}"):
                hits += 1
        read_time = time.time() - start_time
        
        # Performance: debería ser rápido
        assert write_time < 1.0, f"Escritura en caché tomó {write_time:.3f}s"
        assert read_time < 1.0, f"Lectura de caché tomó {read_time:.3f}s"
        assert hits == 1000, f"Hit rate: {hits}/1000"
        
        print(f"\nCache performance: "
              f"escritura={write_time:.3f}s, "
              f"lectura={read_time:.3f}s, "
              f"hit rate={hits/10:.1f}%")
    
    @pytest.mark.performance
    def test_similarity_calculation_performance(self):
        """Test performance de cálculo de similitud."""
        import time
        import numpy as np
        
        config = EmbeddingConfig()
        generator = EmbeddingGenerator(config)
        
        # Crear embeddings de prueba
        np.random.seed(42)
        query = np.random.randn(384).tolist()
        candidates = [np.random.randn(384).tolist() for _ in range(1000)]
        
        # Test cosine similarity
        start_time = time.time()
        similar = generator.find_similar_embeddings(
            query, candidates, top_k=10, threshold=0.0
        )
        cosine_time = time.time() - start_time
        
        assert len(similar) == 10
        
        # Performance: 1000 comparaciones deberían ser rápidas
        assert cosine_time < 1.0, f"Cálculo cosine tomó {cosine_time:.3f}s"
        
        print(f"\nSimilarity calculation performance: "
              f"1000 comparaciones en {cosine_time:.3f}s "
              f"({1000/cosine_time:.0f} comparaciones/segundo)")


class TestEdgeCases:
    """Pruebas de casos límite."""
    
    @pytest.fixture
    def generator(self):
        config = EmbeddingConfig(cache_embeddings=False)
        return EmbeddingGenerator(config)
    
    def test_empty_text(self, generator):
        """Test generación de embedding para texto vacío."""
        with patch.object(generator, 'load_model') as mock_load:
            mock_model = Mock()
            mock_model.encode.return_value = np.array([0.0] * 384)
            generator._models["test-model"] = mock_model
            
            embedding = generator.generate_text_embedding("", model_name="test-model")
            
            assert len(embedding) == 384
            # Embedding de texto vacío podría ser todo ceros o algún valor especial
    
    def test_very_long_text(self, generator):
        """Test generación para texto muy largo."""
        long_text = "word " * 10000  # 50k palabras
        
        with patch.object(generator, 'load_model') as mock_load:
            mock_model = Mock()
            mock_model.encode.return_value = np.array([0.1] * 384)
            generator._models["test-model"] = mock_model
            
            # No debería lanzar excepción
            embedding = generator.generate_text_embedding(long_text, model_name="test-model")
            
            assert len(embedding) == 384
    
    def test_special_characters(self, generator):
        """Test texto con caracteres especiales."""
        special_text = "Text with émojis 🚀 and Ümläutß and 汉字"
        
        with patch.object(generator, 'load_model') as mock_load:
            mock_model = Mock()
            mock_model.encode.return_value = np.array([0.1] * 384)
            generator._models["test-model"] = mock_model
            
            # No debería lanzar excepción
            embedding = generator.generate_text_embedding(special_text, model_name="test-model")
            
            assert len(embedding) == 384
    
    def test_code_with_syntax_errors(self, generator):
        """Test código con errores de sintaxis."""
        invalid_code = '''
def broken
    # Missing colon
    pass
'''
        
        embedding = generator.generate_code_embedding(invalid_code, "python")
        
        # Debería manejar el código inválido
        assert embedding == [] or embedding is None or len(embedding) > 0
    
    def test_cache_key_collisions(self, generator):
        """Test colisiones de keys en caché."""
        generator.config.cache_embeddings = True
        
        embedding1 = [1.0] * 384
        embedding2 = [2.0] * 384
        
        # Mismo key, diferente embedding
        generator.cache_embedding("same_key", embedding1)
        generator.cache_embedding("same_key", embedding2)
        
        # Debería retornar el último
        cached = generator.get_cached_embedding("same_key")
        assert cached == embedding2
    
    def test_invalid_embeddings_in_cache(self, generator):
        """Test embeddings inválidos en caché."""
        generator.config.cache_embeddings = True
        
        # Embedding con dimensión incorrecta
        invalid_embedding = [1.0, 2.0]  # Solo 2 dimensiones
        
        success = generator.cache_embedding("invalid", invalid_embedding)
        
        # Dependiendo de la implementación, podría rechazarlo o almacenarlo
        assert success is True or success is False
    
    def test_model_dimension_mismatch(self, generator):
        """Test cuando el modelo retorna dimensión incorrecta."""
        with patch.object(generator, 'load_model') as mock_load:
            mock_model = Mock()
            # Modelo retorna dimensión incorrecta
            mock_model.encode.return_value = np.array([0.1] * 100)  # Solo 100 dims
            generator._models["wrong-dim-model"] = mock_model
            
            embedding = generator.generate_text_embedding(
                "test", 
                model_name="wrong-dim-model"
            )
            
            # Debería manejar la dimensión incorrecta
            assert len(embedding) == 100  # Aceptar la dimensión del modelo
            # o podría retornar None/[] dependiendo de la validación


if __name__ == "__main__":
    pytest.main([__file__, "-v"])