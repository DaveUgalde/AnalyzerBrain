"""
DimensionalityReducer - Reducción de dimensionalidad de embeddings.
Reduce dimensiones de embeddings preservando información importante.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field, validator
import warnings
from datetime import datetime

class ReductionMethod(Enum):
    """Métodos de reducción de dimensionalidad."""
    PCA = "pca"
    TSNE = "tsne"
    UMAP = "umap"
    TRUNCATED_SVD = "truncated_svd"
    AUTOENCODER = "autoencoder"

class ReductionResult(BaseModel):
    """Resultado de reducción de dimensionalidad."""
    method: ReductionMethod
    original_dimensions: int
    reduced_dimensions: int
    embeddings: List[List[float]]
    explained_variance: Optional[float] = None
    reconstruction_error: Optional[float] = None
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True

class ReductionConfig(BaseModel):
    """Configuración de reducción de dimensionalidad."""
    method: ReductionMethod = ReductionMethod.PCA
    target_dimensions: int = Field(128, ge=2, le=1024)
    random_state: int = 42
    preserve_variance: float = Field(0.95, ge=0.5, le=1.0)
    use_gpu: bool = False
    batch_size: Optional[int] = None
    
    # Método específico: PCA
    pca_whiten: bool = False
    
    # Método específico: t-SNE
    tsne_perplexity: float = 30.0
    tsne_early_exaggeration: float = 12.0
    tsne_learning_rate: float = 200.0
    
    # Método específico: UMAP
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    
    class Config:
        arbitrary_types_allowed = True

class DimensionalityReducer:
    """
    Reductor de dimensionalidad para embeddings.
    
    Características:
    1. Múltiples métodos de reducción (PCA, t-SNE, UMAP, etc.)
    2. Preservación de varianza configurable
    3. Soporte para GPU
    4. Evaluación de calidad de reducción
    5. Transformación inversa (cuando es posible)
    """
    
    def __init__(self, config: Optional[ReductionConfig] = None):
        """
        Inicializa el reductor de dimensionalidad.
        
        Args:
            config: Configuración (opcional)
        """
        self.config = config or ReductionConfig()
        self._models: Dict[ReductionMethod, Any] = {}
        self._fitted = False
        self._stats = {
            "total_reductions": 0,
            "method_usage": {method.value: 0 for method in ReductionMethod},
            "avg_processing_time_ms": 0.0,
            "avg_variance_preserved": 0.0
        }
    
    def reduce_dimensions(self,
                         embeddings: List[List[float]],
                         method: Optional[ReductionMethod] = None,
                         target_dimensions: Optional[int] = None) -> ReductionResult:
        """
        Reduce dimensiones de embeddings.
        
        Args:
            embeddings: Lista de embeddings a reducir
            method: Método a usar (None = config default)
            target_dimensions: Dimensiones objetivo (None = config default)
            
        Returns:
            ReductionResult con embeddings reducidos
        """
        if not embeddings:
            return ReductionResult(
                method=method or self.config.method,
                original_dimensions=0,
                reduced_dimensions=0,
                embeddings=[],
                processing_time_ms=0.0
            )
        
        start_time = datetime.now()
        method = method or self.config.method
        target_dim = target_dimensions or self.config.target_dimensions
        
        # Validar entrada
        original_dim = len(embeddings[0])
        for emb in embeddings:
            if len(emb) != original_dim:
                raise ValueError("All embeddings must have the same dimension")
        
        # Ajustar dimensiones objetivo si es necesario
        if target_dim >= original_dim:
            warnings.warn(f"Target dimensions ({target_dim}) >= original ({original_dim}), no reduction needed")
            return ReductionResult(
                method=method,
                original_dimensions=original_dim,
                reduced_dimensions=original_dim,
                embeddings=embeddings,
                explained_variance=1.0,
                processing_time_ms=0.0
            )
        
        # Convertir a numpy array
        X = np.array(embeddings, dtype=np.float32)
        
        try:
            # Aplicar reducción según método
            if method == ReductionMethod.PCA:
                result = self._apply_pca(X, target_dim)
            elif method == ReductionMethod.TSNE:
                result = self._apply_tsne(X, target_dim)
            elif method == ReductionMethod.UMAP:
                result = self._apply_umap(X, target_dim)
            elif method == ReductionMethod.TRUNCATED_SVD:
                result = self._apply_truncated_svd(X, target_dim)
            elif method == ReductionMethod.AUTOENCODER:
                result = self._apply_autoencoder(X, target_dim)
            else:
                raise ValueError(f"Unknown reduction method: {method}")
            
            # Actualizar estadísticas
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._stats["total_reductions"] += 1
            self._stats["method_usage"][method.value] += 1
            self._update_stats(processing_time, result.explained_variance)
            
            # Añadir tiempo de procesamiento al resultado
            result.processing_time_ms = processing_time
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            warnings.warn(f"Reduction failed: {str(e)}")
            
            # Fallback: devolver embeddings originales
            return ReductionResult(
                method=method,
                original_dimensions=original_dim,
                reduced_dimensions=original_dim,
                embeddings=embeddings,
                processing_time_ms=processing_time,
                metadata={"error": str(e)}
            )
    
    def apply_pca(self,
                 embeddings: List[List[float]],
                 target_dimensions: Optional[int] = None,
                 whiten: Optional[bool] = None) -> ReductionResult:
        """
        Aplica PCA (Análisis de Componentes Principales).
        
        Args:
            embeddings: Lista de embeddings
            target_dimensions: Dimensiones objetivo
            whiten: Si True, aplica whitening
            
        Returns:
            ReductionResult con PCA aplicado
        """
        # Guardar configuración temporal
        original_method = self.config.method
        original_whiten = self.config.pca_whiten
        
        try:
            self.config.method = ReductionMethod.PCA
            if whiten is not None:
                self.config.pca_whiten = whiten
            
            return self.reduce_dimensions(embeddings, target_dimensions=target_dimensions)
            
        finally:
            # Restaurar configuración
            self.config.method = original_method
            self.config.pca_whiten = original_whiten
    
    def apply_tsne(self,
                  embeddings: List[List[float]],
                  target_dimensions: Optional[int] = None,
                  perplexity: Optional[float] = None,
                  learning_rate: Optional[float] = None) -> ReductionResult:
        """
        Aplica t-SNE (t-Distributed Stochastic Neighbor Embedding).
        
        Args:
            embeddings: Lista de embeddings
            target_dimensions: Dimensiones objetivo (típicamente 2 o 3)
            perplexity: Perplejidad (ajusta vecindario efectivo)
            learning_rate: Tasa de aprendizaje
            
        Returns:
            ReductionResult con t-SNE aplicado
        """
        # Guardar configuración temporal
        original_method = self.config.method
        original_perplexity = self.config.tsne_perplexity
        original_learning_rate = self.config.tsne_learning_rate
        
        try:
            self.config.method = ReductionMethod.TSNE
            if perplexity is not None:
                self.config.tsne_perplexity = perplexity
            if learning_rate is not None:
                self.config.tsne_learning_rate = learning_rate
            
            return self.reduce_dimensions(embeddings, target_dimensions=target_dimensions)
            
        finally:
            # Restaurar configuración
            self.config.method = original_method
            self.config.tsne_perplexity = original_perplexity
            self.config.tsne_learning_rate = original_learning_rate
    
    def apply_umap(self,
                  embeddings: List[List[float]],
                  target_dimensions: Optional[int] = None,
                  n_neighbors: Optional[int] = None,
                  min_dist: Optional[float] = None) -> ReductionResult:
        """
        Aplica UMAP (Uniform Manifold Approximation and Projection).
        
        Args:
            embeddings: Lista de embeddings
            target_dimensions: Dimensiones objetivo
            n_neighbors: Número de vecinos
            min_dist: Distancia mínima entre puntos
            
        Returns:
            ReductionResult con UMAP aplicado
        """
        # Guardar configuración temporal
        original_method = self.config.method
        original_neighbors = self.config.umap_n_neighbors
        original_min_dist = self.config.umap_min_dist
        
        try:
            self.config.method = ReductionMethod.UMAP
            if n_neighbors is not None:
                self.config.umap_n_neighbors = n_neighbors
            if min_dist is not None:
                self.config.umap_min_dist = min_dist
            
            return self.reduce_dimensions(embeddings, target_dimensions=target_dimensions)
            
        finally:
            # Restaurar configuración
            self.config.method = original_method
            self.config.umap_n_neighbors = original_neighbors
            self.config.umap_min_dist = original_min_dist
    
    def optimize_reduction(self,
                          embeddings: List[List[float]],
                          min_dimensions: int = 2,
                          max_dimensions: int = 512,
                          step: int = 32) -> Dict[int, ReductionResult]:
        """
        Encuentra dimensiones óptimas probando diferentes valores.
        
        Args:
            embeddings: Lista de embeddings
            min_dimensions: Dimensiones mínimas a probar
            max_dimensions: Dimensiones máximas a probar
            step: Incremento entre pruebas
            
        Returns:
            Dict con resultados por número de dimensiones
        """
        results = {}
        
        for dim in range(min_dimensions, max_dimensions + 1, step):
            try:
                result = self.reduce_dimensions(embeddings, target_dimensions=dim)
                results[dim] = result
            except Exception as e:
                warnings.warn(f"Failed reduction to {dim} dimensions: {str(e)}")
        
        return results
    
    def evaluate_reduction(self,
                          original_embeddings: List[List[float]],
                          reduced_embeddings: List[List[float]]) -> Dict[str, float]:
        """
        Evalúa la calidad de la reducción.
        
        Args:
            original_embeddings: Embeddings originales
            reduced_embeddings: Embeddings reducidos
            
        Returns:
            Dict con métricas de evaluación
        """
        if not original_embeddings or not reduced_embeddings:
            return {"error": "Empty embeddings"}
        
        if len(original_embeddings) != len(reduced_embeddings):
            return {"error": "Mismatched number of embeddings"}
        
        X_orig = np.array(original_embeddings, dtype=np.float32)
        X_reduced = np.array(reduced_embeddings, dtype=np.float32)
        
        metrics = {}
        
        try:
            # 1. Varianza preservada (si es PCA)
            if hasattr(self, '_pca_model') and self._pca_model is not None:
                explained_variance = np.sum(self._pca_model.explained_variance_ratio_)
                metrics["explained_variance"] = float(explained_variance)
            
            # 2. Error de reconstrucción (MSE)
            if X_reduced.shape[1] < X_orig.shape[1]:
                # Reconstruir usando transformación inversa si está disponible
                X_reconstructed = self.inverse_transform(X_reduced.tolist())
                if X_reconstructed:
                    X_recon_array = np.array(X_reconstructed, dtype=np.float32)
                    mse = np.mean((X_orig - X_recon_array) ** 2)
                    metrics["reconstruction_mse"] = float(mse)
            
            # 3. Preservación de distancias
            # Calcular matriz de distancias original y reducida
            from scipy.spatial.distance import pdist, squareform
            dist_orig = pdist(X_orig, 'euclidean')
            dist_reduced = pdist(X_reduced, 'euclidean')
            
            # Correlación entre distancias
            correlation = np.corrcoef(dist_orig, dist_reduced)[0, 1]
            metrics["distance_correlation"] = float(correlation)
            
            # 4. Trustworthiness (preservación de vecindarios)
            metrics["trustworthiness"] = self._calculate_trustworthiness(X_orig, X_reduced)
            
            # 5. Reducción de dimensionalidad efectiva
            compression_ratio = X_reduced.shape[1] / X_orig.shape[1]
            metrics["compression_ratio"] = float(compression_ratio)
            
        except Exception as e:
            warnings.warn(f"Reduction evaluation failed: {str(e)}")
            metrics["error"] = str(e)
        
        return metrics
    
    def inverse_transform(self, reduced_embeddings: List[List[float]]) -> Optional[List[List[float]]]:
        """
        Transformación inversa (cuando es posible).
        
        Args:
            reduced_embeddings: Embeddings reducidos
            
        Returns:
            Embeddings reconstruidos o None si no es posible
        """
        if not reduced_embeddings:
            return None
        
        method = self.config.method
        
        try:
            X_reduced = np.array(reduced_embeddings, dtype=np.float32)
            
            if method == ReductionMethod.PCA and 'pca' in self._models:
                model = self._models['pca']
                if hasattr(model, 'inverse_transform'):
                    X_original = model.inverse_transform(X_reduced)
                    return X_original.tolist()
            
            elif method == ReductionMethod.TRUNCATED_SVD and 'svd' in self._models:
                model = self._models['svd']
                if hasattr(model, 'inverse_transform'):
                    X_original = model.inverse_transform(X_reduced)
                    return X_original.tolist()
            
            elif method == ReductionMethod.AUTOENCODER and 'autoencoder' in self._models:
                # Reconstrucción con autoencoder
                model = self._models['autoencoder']
                # Asumir que el modelo tiene método predict o similar
                if hasattr(model, 'predict'):
                    X_original = model.predict(X_reduced)
                    return X_original.tolist()
            
            # Métodos como t-SNE y UMAP no tienen transformación inversa directa
            warnings.warn(f"Inverse transform not available for {method.value}")
            return None
            
        except Exception as e:
            warnings.warn(f"Inverse transform failed: {str(e)}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del reductor.
        
        Returns:
            Dict con estadísticas
        """
        total = self._stats["total_reductions"]
        
        return {
            "total_reductions": total,
            "method_usage": self._stats["method_usage"],
            "avg_processing_time_ms": self._stats["avg_processing_time_ms"],
            "avg_variance_preserved": self._stats["avg_variance_preserved"],
            "most_used_method": max(self._stats["method_usage"].items(), key=lambda x: x[1])[0] if total > 0 else None
        }
    
    # Métodos privados de implementación
    
    def _apply_pca(self, X: np.ndarray, target_dim: int) -> ReductionResult:
        """Aplica PCA."""
        try:
            from sklearn.decomposition import PCA
            
            # Crear modelo PCA
            n_components = min(target_dim, X.shape[1] - 1)
            pca = PCA(
                n_components=n_components,
                whiten=self.config.pca_whiten,
                random_state=self.config.random_state
            )
            
            # Ajustar y transformar
            X_reduced = pca.fit_transform(X)
            
            # Guardar modelo
            self._models['pca'] = pca
            self._fitted = True
            
            # Calcular varianza explicada
            explained_variance = np.sum(pca.explained_variance_ratio_)
            
            return ReductionResult(
                method=ReductionMethod.PCA,
                original_dimensions=X.shape[1],
                reduced_dimensions=X_reduced.shape[1],
                embeddings=X_reduced.tolist(),
                explained_variance=float(explained_variance),
                metadata={
                    "components": pca.components_.tolist() if hasattr(pca, 'components_') else None,
                    "singular_values": pca.singular_values_.tolist() if hasattr(pca, 'singular_values_') else None
                }
            )
            
        except ImportError:
            raise RuntimeError("scikit-learn is required for PCA")
        except Exception as e:
            raise RuntimeError(f"PCA failed: {str(e)}")
    
    def _apply_tsne(self, X: np.ndarray, target_dim: int) -> ReductionResult:
        """Aplica t-SNE."""
        try:
            from sklearn.manifold import TSNE
            
            # t-SNE es computacionalmente costoso, limitar muestras si son muchas
            max_samples = 1000
            if X.shape[0] > max_samples:
                warnings.warn(f"t-SNE is computationally expensive, using first {max_samples} samples")
                X = X[:max_samples]
                indices_used = list(range(max_samples))
            else:
                indices_used = list(range(X.shape[0]))
            
            # Crear modelo t-SNE
            tsne = TSNE(
                n_components=min(target_dim, 3),  # t-SNE normalmente para 2D o 3D
                perplexity=self.config.tsne_perplexity,
                early_exaggeration=self.config.tsne_early_exaggeration,
                learning_rate=self.config.tsne_learning_rate,
                random_state=self.config.random_state,
                n_jobs=-1 if not self.config.use_gpu else 1
            )
            
            # Transformar
            X_reduced = tsne.fit_transform(X)
            
            # Guardar modelo
            self._models['tsne'] = tsne
            
            return ReductionResult(
                method=ReductionMethod.TSNE,
                original_dimensions=X.shape[1],
                reduced_dimensions=X_reduced.shape[1],
                embeddings=X_reduced.tolist(),
                metadata={
                    "indices_used": indices_used,
                    "kl_divergence": tsne.kl_divergence_ if hasattr(tsne, 'kl_divergence_') else None
                }
            )
            
        except ImportError:
            raise RuntimeError("scikit-learn is required for t-SNE")
        except Exception as e:
            raise RuntimeError(f"t-SNE failed: {str(e)}")
    
    def _apply_umap(self, X: np.ndarray, target_dim: int) -> ReductionResult:
        """Aplica UMAP."""
        try:
            import umap
            
            # Crear modelo UMAP
            reducer = umap.UMAP(
                n_components=min(target_dim, X.shape[1] - 1),
                n_neighbors=self.config.umap_n_neighbors,
                min_dist=self.config.umap_min_dist,
                random_state=self.config.random_state,
                metric='euclidean'
            )
            
            # Transformar
            X_reduced = reducer.fit_transform(X)
            
            # Guardar modelo
            self._models['umap'] = reducer
            
            return ReductionResult(
                method=ReductionMethod.UMAP,
                original_dimensions=X.shape[1],
                reduced_dimensions=X_reduced.shape[1],
                embeddings=X_reduced.tolist()
            )
            
        except ImportError:
            raise RuntimeError("UMAP is required. Install with: pip install umap-learn")
        except Exception as e:
            raise RuntimeError(f"UMAP failed: {str(e)}")
    
    def _apply_truncated_svd(self, X: np.ndarray, target_dim: int) -> ReductionResult:
        """Aplica Truncated SVD."""
        try:
            from sklearn.decomposition import TruncatedSVD
            
            # Crear modelo TruncatedSVD
            n_components = min(target_dim, X.shape[1] - 1)
            svd = TruncatedSVD(
                n_components=n_components,
                random_state=self.config.random_state
            )
            
            # Ajustar y transformar
            X_reduced = svd.fit_transform(X)
            
            # Guardar modelo
            self._models['svd'] = svd
            
            # Calcular varianza explicada
            explained_variance = np.sum(svd.explained_variance_ratio_)
            
            return ReductionResult(
                method=ReductionMethod.TRUNCATED_SVD,
                original_dimensions=X.shape[1],
                reduced_dimensions=X_reduced.shape[1],
                embeddings=X_reduced.tolist(),
                explained_variance=float(explained_variance)
            )
            
        except ImportError:
            raise RuntimeError("scikit-learn is required for Truncated SVD")
        except Exception as e:
            raise RuntimeError(f"Truncated SVD failed: {str(e)}")
    
    def _apply_autoencoder(self, X: np.ndarray, target_dim: int) -> ReductionResult:
        """Aplica Autoencoder (implementación simplificada)."""
        try:
            from sklearn.neural_network import MLPRegressor
            
            # Autoencoder simplificado usando MLP
            input_dim = X.shape[1]
            hidden_dim = max(target_dim * 2, 64)
            
            # Crear modelo MLP (autoencoder simplificado)
            autoencoder = MLPRegressor(
                hidden_layer_sizes=(hidden_dim, target_dim, hidden_dim),
                activation='relu',
                solver='adam',
                random_state=self.config.random_state,
                max_iter=100,
                batch_size=self.config.batch_size or min(32, X.shape[0])
            )
            
            # Entrenar autoencoder (reconstrucción)
            autoencoder.fit(X, X)
            
            # Obtener embeddings reducidos (capa intermedia)
            # En un autoencoder real, extraeríamos la representación de la capa intermedia
            # Para simplificar, usaremos PCA como aproximación
            
            warnings.warn("Autoencoder using PCA approximation. For full autoencoder, use dedicated neural network.")
            
            # Usar PCA como aproximación
            from sklearn.decomposition import PCA
            pca = PCA(n_components=target_dim, random_state=self.config.random_state)
            X_reduced = pca.fit_transform(X)
            
            # Calcular error de reconstrucción
            X_reconstructed = autoencoder.predict(X)
            reconstruction_error = np.mean((X - X_reconstructed) ** 2)
            
            # Guardar modelo
            self._models['autoencoder'] = autoencoder
            self._models['pca'] = pca  # También guardar PCA para consistencia
            
            return ReductionResult(
                method=ReductionMethod.AUTOENCODER,
                original_dimensions=input_dim,
                reduced_dimensions=target_dim,
                embeddings=X_reduced.tolist(),
                reconstruction_error=float(reconstruction_error),
                metadata={
                    "autoencoder_layers": autoencoder.hidden_layer_sizes,
                    "reconstruction_mse": float(reconstruction_error)
                }
            )
            
        except ImportError:
            raise RuntimeError("scikit-learn is required for autoencoder")
        except Exception as e:
            raise RuntimeError(f"Autoencoder failed: {str(e)}")
    
    def _calculate_trustworthiness(self, 
                                 X_orig: np.ndarray, 
                                 X_reduced: np.ndarray,
                                 k: int = 5) -> float:
        """Calcula trustworthiness (preservación de vecindarios)."""
        try:
            from sklearn.neighbors import NearestNeighbors
            
            n_samples = X_orig.shape[0]
            k = min(k, n_samples - 1)
            
            # Encontrar k-vecinos más cercanos en espacio original
            nbrs_orig = NearestNeighbors(n_neighbors=k+1).fit(X_orig)
            distances_orig, indices_orig = nbrs_orig.kneighbors(X_orig)
            
            # Encontrar k-vecinos más cercanos en espacio reducido
            nbrs_reduced = NearestNeighbors(n_neighbors=k+1).fit(X_reduced)
            distances_reduced, indices_reduced = nbrs_reduced.kneighbors(X_reduced)
            
            # Calcular trustworthiness
            trust = 0.0
            for i in range(n_samples):
                # Vecinos en espacio reducido que NO estaban en espacio original
                neighbors_reduced = set(indices_reduced[i, 1:k+1])  # Excluir el punto mismo
                neighbors_orig = set(indices_orig[i, 1:k+1])
                
                # Intrusos: vecinos en espacio reducido que no estaban en espacio original
                intruders = neighbors_reduced - neighbors_orig
                
                for intruder in intruders:
                    # Encontrar rango del intruso en espacio original
                    rank_in_orig = np.where(indices_orig[i] == intruder)[0]
                    if len(rank_in_orig) > 0:
                        rank = rank_in_orig[0]
                        trust += rank - k
            
            # Normalizar
            trust = 1.0 - (2.0 * trust) / (n_samples * k * (2 * n_samples - 3 * k - 1))
            
            return float(trust)
            
        except Exception as e:
            warnings.warn(f"Trustworthiness calculation failed: {str(e)}")
            return 0.0
    
    def _update_stats(self, processing_time: float, explained_variance: Optional[float]) -> None:
        """Actualiza estadísticas."""
        total = self._stats["total_reductions"]
        
        # Actualizar tiempo promedio
        current_avg = self._stats["avg_processing_time_ms"]
        self._stats["avg_processing_time_ms"] = (
            (current_avg * (total - 1) + processing_time) / total
        )
        
        # Actualizar varianza promedio
        if explained_variance is not None:
            current_var_avg = self._stats["avg_variance_preserved"]
            self._stats["avg_variance_preserved"] = (
                (current_var_avg * (total - 1) + explained_variance) / total
            )

# Ejemplo de uso
if __name__ == "__main__":
    # Crear reductor
    config = ReductionConfig(
        method=ReductionMethod.PCA,
        target_dimensions=2,
        preserve_variance=0.95
    )
    
    reducer = DimensionalityReducer(config)
    
    # Crear embeddings de ejemplo (100 embeddings de 50 dimensiones)
    np.random.seed(42)
    embeddings = np.random.randn(100, 50).tolist()
    
    # Reducir dimensiones
    result = reducer.reduce_dimensions(embeddings)
    
    print(f"Original dimensions: {result.original_dimensions}")
    print(f"Reduced dimensions: {result.reduced_dimensions}")
    print(f"Explained variance: {result.explained_variance:.4f}")
    print(f"Processing time: {result.processing_time_ms:.2f} ms")
    
    # Evaluar reducción
    metrics = reducer.evaluate_reduction(embeddings, result.embeddings)
    print(f"Evaluation metrics: {metrics}")
    
    # Transformación inversa (si es posible)
    reconstructed = reducer.inverse_transform(result.embeddings[:5])  # Solo primeros 5
    if reconstructed:
        print(f"Reconstructed {len(reconstructed)} embeddings")
    
    # Obtener estadísticas
    stats = reducer.get_stats()
    print(f"Total reductions: {stats['total_reductions']}")