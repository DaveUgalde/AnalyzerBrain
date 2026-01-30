"""
ParallelProcessing - Utilidades para procesamiento paralelo y distribuido.
Incluye paralelización de tareas, balanceo de carga y manejo de errores.
"""

import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Any, Callable, Optional, Union, Tuple, Iterator
from typing import TypeVar, Generic, Sequence
import multiprocessing
import queue
import threading
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from functools import partial
import traceback
from ..core.exceptions import BrainException, TimeoutError

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')

class ParallelizationStrategy(Enum):
    """Estrategias de paralelización."""
    THREADS = "threads"      # I/O bound tasks
    PROCESSES = "processes"  # CPU bound tasks
    ASYNC = "async"          # Coroutines
    HYBRID = "hybrid"        # Combinación óptima

@dataclass
class ParallelTask:
    """Representa una tarea para procesamiento paralelo."""
    id: str
    function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    timeout: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ParallelResult:
    """Resultado de una tarea paralela."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    processing_time: float = 0.0
    completed_at: Optional[datetime] = None

class ParallelProcessing:
    """
    Utilidades para procesamiento paralelo eficiente.
    
    Características:
    1. Paralelización automática basada en tipo de tarea
    2. Balanceo dinámico de carga
    3. Manejo robusto de errores y timeouts
    4. Monitoreo de performance
    5. Limpieza automática de recursos
    """
    
    # Pools globales compartidos
    _thread_pools: Dict[str, ThreadPoolExecutor] = {}
    _process_pools: Dict[str, ProcessPoolExecutor] = {}
    _active_tasks: Dict[str, Dict] = {}
    
    @staticmethod
    def parallel_map(
        items: List[T],
        func: Callable[[T], R],
        strategy: ParallelizationStrategy = None,
        max_workers: Optional[int] = None,
        chunk_size: Optional[int] = None,
        timeout_per_item: Optional[float] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[R]:
        """
        Aplica una función a cada elemento en paralelo (similar a map).
        
        Args:
            items: Lista de elementos a procesar
            func: Función a aplicar a cada elemento
            strategy: Estrategia de paralelización
            max_workers: Número máximo de workers
            chunk_size: Tamaño de chunk para procesamiento por lotes
            timeout_per_item: Timeout por elemento (segundos)
            progress_callback: Callback para reportar progreso
            
        Returns:
            Lista de resultados en el mismo orden que los elementos de entrada
            
        Raises:
            BrainException: Si hay errores en el procesamiento paralelo
            TimeoutError: Si algún elemento excede el timeout
        """
        if not items:
            return []
        
        # Determinar estrategia automáticamente si no se especifica
        if strategy is None:
            strategy = ParallelProcessing._determine_strategy(func, items)
        
        # Determinar número de workers
        if max_workers is None:
            max_workers = ParallelProcessing._get_optimal_workers(strategy)
        
        logger.debug(
            f"Parallel map: {len(items)} items, "
            f"strategy={strategy.value}, workers={max_workers}"
        )
        
        try:
            if strategy == ParallelizationStrategy.THREADS:
                return ParallelProcessing._thread_map(
                    items, func, max_workers, chunk_size,
                    timeout_per_item, progress_callback
                )
            elif strategy == ParallelizationStrategy.PROCESSES:
                return ParallelProcessing._process_map(
                    items, func, max_workers, chunk_size,
                    timeout_per_item, progress_callback
                )
            elif strategy == ParallelizationStrategy.ASYNC:
                return ParallelProcessing._async_map(
                    items, func, max_workers, chunk_size,
                    timeout_per_item, progress_callback
                )
            elif strategy == ParallelizationStrategy.HYBRID:
                return ParallelProcessing._hybrid_map(
                    items, func, max_workers, chunk_size,
                    timeout_per_item, progress_callback
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
                
        except Exception as e:
            logger.error(f"Parallel map failed: {e}")
            raise BrainException(f"Parallel processing failed: {e}")
    
    @staticmethod
    def distribute_workload(
        tasks: List[ParallelTask],
        strategy: ParallelizationStrategy = ParallelizationStrategy.THREADS,
        max_workers: Optional[int] = None,
        load_balancing: bool = True,
        fail_fast: bool = False
    ) -> Dict[str, ParallelResult]:
        """
        Distribuye y ejecuta tareas con balanceo de carga.
        
        Args:
            tasks: Lista de tareas a ejecutar
            strategy: Estrategia de paralelización
            max_workers: Número máximo de workers
            load_balancing: Habilitar balanceo dinámico de carga
            fail_fast: Detener al primer error
            
        Returns:
            Diccionario task_id -> ParallelResult
            
        Raises:
            BrainException: Si hay errores en la distribución
        """
        if not tasks:
            return {}
        
        # Ordenar por prioridad (mayor prioridad primero)
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)
        
        # Determinar número de workers
        if max_workers is None:
            max_workers = min(len(tasks), ParallelProcessing._get_optimal_workers(strategy))
        
        results = {}
        start_time = time.time()
        
        try:
            if strategy == ParallelizationStrategy.THREADS:
                pool = ParallelProcessing._get_thread_pool(max_workers)
                results = ParallelProcessing._execute_with_pool(
                    pool, sorted_tasks, load_balancing, fail_fast
                )
            elif strategy == ParallelizationStrategy.PROCESSES:
                pool = ParallelProcessing._get_process_pool(max_workers)
                results = ParallelProcessing._execute_with_pool(
                    pool, sorted_tasks, load_balancing, fail_fast
                )
            elif strategy == ParallelizationStrategy.ASYNC:
                results = asyncio.run(
                    ParallelProcessing._execute_async(sorted_tasks, load_balancing, fail_fast)
                )
            else:
                raise ValueError(f"Strategy {strategy} not supported for workload distribution")
            
            total_time = time.time() - start_time
            logger.info(
                f"Workload completed: {len(tasks)} tasks, "
                f"{sum(1 for r in results.values() if r.success)} succeeded, "
                f"time={total_time:.2f}s"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Workload distribution failed: {e}")
            raise BrainException(f"Workload distribution failed: {e}")
    
    @staticmethod
    def synchronize_tasks(
        tasks: List[Callable],
        dependencies: Dict[int, List[int]] = None,
        max_concurrent: Optional[int] = None,
        timeout: Optional[float] = None
    ) -> List[Any]:
        """
        Ejecuta tareas con dependencias y sincronización.
        
        Args:
            tasks: Lista de funciones a ejecutar
            dependencies: Diccionario task_index -> lista de índices de dependencias
            max_concurrent: Máximo de tareas concurrentes
            timeout: Timeout total para todas las tareas
            
        Returns:
            Lista de resultados en orden de tareas
            
        Raises:
            BrainException: Si hay ciclos en las dependencias o errores de ejecución
            TimeoutError: Si se excede el timeout total
        """
        if not tasks:
            return []
        
        # Validar dependencias
        if dependencies:
            ParallelProcessing._validate_dependencies(dependencies, len(tasks))
        
        # Crear grafo de dependencias
        dependency_graph = dependencies or {}
        
        # Ejecutar con orden topológico
        try:
            start_time = time.time()
            results = [None] * len(tasks)
            completed = set()
            in_progress = set()
            
            # Ejecutor para tareas concurrentes
            executor = ThreadPoolExecutor(
                max_workers=max_concurrent or ParallelProcessing._get_optimal_workers()
            )
            
            # Mapeo de futuros a índices de tareas
            future_to_index = {}
            
            while len(completed) < len(tasks):
                # Verificar timeout
                if timeout and (time.time() - start_time) > timeout:
                    raise TimeoutError(f"Total timeout of {timeout}s exceeded")
                
                # Encontrar tareas listas para ejecutar
                ready_tasks = []
                for i, task in enumerate(tasks):
                    if i in completed or i in in_progress:
                        continue
                    
                    # Verificar dependencias
                    deps = dependency_graph.get(i, [])
                    if all(dep in completed for dep in deps):
                        ready_tasks.append((i, task))
                
                # Ejecutar tareas listas
                for task_index, task_func in ready_tasks:
                    if max_concurrent and len(in_progress) >= max_concurrent:
                        break
                    
                    future = executor.submit(task_func)
                    future_to_index[future] = task_index
                    in_progress.add(task_index)
                
                # Esperar por la primera tarea que termine
                if future_to_index:
                    done, _ = concurrent.futures.wait(
                        list(future_to_index.keys()),
                        timeout=1.0,
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    
                    for future in done:
                        task_index = future_to_index.pop(future)
                        in_progress.remove(task_index)
                        
                        try:
                            results[task_index] = future.result(timeout=0)
                            completed.add(task_index)
                        except Exception as e:
                            # Si una tarea falla, marcar todas sus dependientes como falladas
                            logger.error(f"Task {task_index} failed: {e}")
                            # En modo fail_fast, propagaríamos el error
                            # Aquí continuamos marcando como completada pero con error
                            results[task_index] = e
                            completed.add(task_index)
            
            executor.shutdown(wait=True)
            return results
            
        except Exception as e:
            logger.error(f"Task synchronization failed: {e}")
            raise BrainException(f"Task synchronization failed: {e}")
    
    @staticmethod
    def handle_parallel_errors(
        results: List[Union[R, Exception]],
        error_threshold: float = 0.1,
        retry_func: Optional[Callable[[List[int]], List[R]]] = None,
        max_retries: int = 3
    ) -> Tuple[List[R], List[Dict[str, Any]]]:
        """
        Maneja errores en resultados de procesamiento paralelo.
        
        Args:
            results: Lista de resultados (pueden ser excepciones)
            error_threshold: Umbral de error para considerar el batch como fallido
            retry_func: Función para reintentar elementos fallidos
            max_retries: Máximo de reintentos
            
        Returns:
            Tuple de (resultados exitosos, información de errores)
        """
        errors = []
        successful = []
        error_indices = []
        
        # Clasificar resultados
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append({
                    'index': i,
                    'error': str(result),
                    'error_type': type(result).__name__,
                    'traceback': traceback.format_exc() if hasattr(result, '__traceback__') else None
                })
                error_indices.append(i)
            else:
                successful.append(result)
        
        # Verificar si excede umbral de error
        error_rate = len(errors) / len(results) if results else 0
        if error_rate > error_threshold:
            logger.warning(
                f"High error rate in parallel processing: "
                f"{error_rate:.1%} > {error_threshold:.1%}"
            )
        
        # Reintentar errores si se especifica una función de reintento
        if retry_func and error_indices and max_retries > 0:
            retry_results = ParallelProcessing._retry_errors(
                retry_func, error_indices, max_retries
            )
            
            # Combinar resultados exitosos de reintentos
            for i, retry_result in enumerate(retry_results):
                if not isinstance(retry_result, Exception):
                    successful.append(retry_result)
                    # Actualizar información de error
                    for error_info in errors:
                        if error_info['index'] == error_indices[i]:
                            error_info['retry_success'] = True
                            error_info['retry_result'] = retry_result
                            break
        
        return successful, errors
    
    @staticmethod
    def optimize_parallelization(
        func: Callable,
        sample_data: List[Any],
        strategies: List[ParallelizationStrategy] = None,
        iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Encuentra la configuración óptima de paralelización para una función.
        
        Args:
            func: Función a optimizar
            sample_data: Datos de muestra para benchmarking
            strategies: Estrategias a probar
            iterations: Número de iteraciones por prueba
            
        Returns:
            Diccionario con configuración óptima y métricas
        """
        if strategies is None:
            strategies = list(ParallelizationStrategy)
        
        results = {}
        
        for strategy in strategies:
            strategy_results = []
            
            # Probar diferentes números de workers
            max_cpus = multiprocessing.cpu_count()
            worker_options = [1, 2, 4, 8, max_cpus]
            
            for workers in worker_options:
                if workers > max_cpus and strategy == ParallelizationStrategy.PROCESSES:
                    continue
                
                times = []
                for _ in range(iterations):
                    start_time = time.time()
                    try:
                        ParallelProcessing.parallel_map(
                            sample_data[:10],  # Usar subset para pruebas
                            func,
                            strategy=strategy,
                            max_workers=workers
                        )
                        elapsed = time.time() - start_time
                        times.append(elapsed)
                    except Exception as e:
                        logger.warning(f"Strategy {strategy} with {workers} workers failed: {e}")
                        times.append(float('inf'))
                
                if times:
                    avg_time = sum(t for t in times if t != float('inf')) / len(times)
                    strategy_results.append({
                        'workers': workers,
                        'avg_time': avg_time,
                        'success_rate': sum(1 for t in times if t != float('inf')) / len(times)
                    })
            
            if strategy_results:
                # Encontrar la mejor configuración
                best_config = min(
                    [r for r in strategy_results if r['success_rate'] > 0.5],
                    key=lambda x: x['avg_time'],
                    default=None
                )
                
                if best_config:
                    results[strategy.value] = {
                        'best_workers': best_config['workers'],
                        'estimated_time': best_config['avg_time'],
                        'speedup': strategy_results[0]['avg_time'] / best_config['avg_time']
                        if strategy_results[0]['avg_time'] > 0 else 0,
                        'all_configs': strategy_results
                    }
        
        # Determinar la mejor estrategia general
        if results:
            best_strategy = min(
                results.items(),
                key=lambda x: x[1]['estimated_time']
            )
            results['recommended'] = {
                'strategy': best_strategy[0],
                'workers': best_strategy[1]['best_workers'],
                'estimated_time': best_strategy[1]['estimated_time']
            }
        
        return results
    
    @staticmethod
    def monitor_parallel_performance(
        task_ids: List[str] = None,
        time_window: int = 60
    ) -> Dict[str, Any]:
        """
        Monitorea el performance del procesamiento paralelo.
        
        Args:
            task_ids: IDs de tareas específicas a monitorear
            time_window: Ventana de tiempo en segundos
            
        Returns:
            Métricas de performance actuales
        """
        current_time = time.time()
        
        # Filtrar tareas activas por ventana de tiempo
        recent_tasks = {
            task_id: task_info
            for task_id, task_info in ParallelProcessing._active_tasks.items()
            if current_time - task_info['start_time'] <= time_window
        }
        
        if task_ids:
            recent_tasks = {
                task_id: task_info
                for task_id, task_info in recent_tasks.items()
                if task_id in task_ids
            }
        
        # Calcular métricas
        if not recent_tasks:
            return {
                'active_tasks': 0,
                'completed_tasks': 0,
                'avg_processing_time': 0,
                'throughput': 0,
                'error_rate': 0
            }
        
        completed_tasks = [
            t for t in recent_tasks.values() 
            if t.get('completed', False)
        ]
        
        active_tasks = [
            t for t in recent_tasks.values() 
            if not t.get('completed', False)
        ]
        
        processing_times = [
            t.get('processing_time', 0)
            for t in completed_tasks
            if t.get('processing_time', 0) > 0
        ]
        
        errors = [
            t for t in completed_tasks
            if t.get('error', False)
        ]
        
        return {
            'active_tasks': len(active_tasks),
            'completed_tasks': len(completed_tasks),
            'total_tasks': len(recent_tasks),
            'avg_processing_time': sum(processing_times) / len(processing_times) 
            if processing_times else 0,
            'throughput': len(completed_tasks) / time_window if time_window > 0 else 0,
            'error_rate': len(errors) / len(completed_tasks) if completed_tasks else 0,
            'recent_task_ids': list(recent_tasks.keys())[:10]
        }
    
    @staticmethod
    def cleanup_parallel_resources(
        pool_type: Optional[str] = None,
        force: bool = False
    ) -> Dict[str, int]:
        """
        Limpia recursos de procesamiento paralelo.
        
        Args:
            pool_type: 'threads', 'processes', o None para ambos
            force: Forzar cierre inmediato
            
        Returns:
            Diccionario con conteo de recursos liberados
        """
        cleaned = {
            'thread_pools': 0,
            'process_pools': 0,
            'active_tasks': len(ParallelProcessing._active_tasks)
        }
        
        # Limpiar pools de threads
        if pool_type in [None, 'threads']:
            for pool_name, pool in list(ParallelProcessing._thread_pools.items()):
                try:
                    if force:
                        pool.shutdown(wait=False)
                    else:
                        pool.shutdown(wait=True)
                    del ParallelProcessing._thread_pools[pool_name]
                    cleaned['thread_pools'] += 1
                except Exception as e:
                    logger.warning(f"Failed to cleanup thread pool {pool_name}: {e}")
        
        # Limpiar pools de procesos
        if pool_type in [None, 'processes']:
            for pool_name, pool in list(ParallelProcessing._process_pools.items()):
                try:
                    pool.shutdown()
                    del ParallelProcessing._process_pools[pool_name]
                    cleaned['process_pools'] += 1
                except Exception as e:
                    logger.warning(f"Failed to cleanup process pool {pool_name}: {e}")
        
        # Limpiar tareas activas
        if force:
            ParallelProcessing._active_tasks.clear()
        else:
            # Solo limpiar tareas antiguas
            current_time = time.time()
            old_tasks = [
                task_id for task_id, task_info in ParallelProcessing._active_tasks.items()
                if current_time - task_info.get('start_time', 0) > 3600  # 1 hora
            ]
            for task_id in old_tasks:
                ParallelProcessing._active_tasks.pop(task_id, None)
        
        logger.info(f"Cleaned parallel resources: {cleaned}")
        return cleaned
    
    # ========== MÉTODOS PRIVADOS ==========
    
    @staticmethod
    def _determine_strategy(func: Callable, sample_items: List) -> ParallelizationStrategy:
        """Determina la mejor estrategia de paralelización para una función."""
        # Análisis simple basado en tipo de función y datos
        func_name = func.__name__.lower()
        
        # Funciones I/O bound
        io_keywords = ['read', 'write', 'fetch', 'download', 'upload', 'network', 'api']
        if any(keyword in func_name for keyword in io_keywords):
            return ParallelizationStrategy.THREADS
        
        # Funciones CPU intensive
        cpu_keywords = ['calculate', 'compute', 'process', 'analyze', 'transform', 'encode']
        if any(keyword in func_name for keyword in cpu_keywords):
            return ParallelizationStrategy.PROCESSES
        
        # Por defecto, usar threads para seguridad
        return ParallelizationStrategy.THREADS
    
    @staticmethod
    def _get_optimal_workers(strategy: ParallelizationStrategy = None) -> int:
        """Obtiene número óptimo de workers para una estrategia."""
        cpu_count = multiprocessing.cpu_count()
        
        if strategy == ParallelizationStrategy.PROCESSES:
            return min(cpu_count, 8)  # Limitamos procesos para evitar sobrecarga
        elif strategy == ParallelizationStrategy.THREADS:
            return min(cpu_count * 4, 32)  # Más threads para I/O
        else:
            return min(cpu_count * 2, 16)  # Default balanced
    
    @staticmethod
    def _get_thread_pool(max_workers: int) -> ThreadPoolExecutor:
        """Obtiene o crea un pool de threads."""
        pool_key = f"threads_{max_workers}"
        
        if pool_key not in ParallelProcessing._thread_pools:
            ParallelProcessing._thread_pools[pool_key] = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix=f"BrainThreadPool_{max_workers}"
            )
        
        return ParallelProcessing._thread_pools[pool_key]
    
    @staticmethod
    def _get_process_pool(max_workers: int) -> ProcessPoolExecutor:
        """Obtiene o crea un pool de procesos."""
        pool_key = f"processes_{max_workers}"
        
        if pool_key not in ParallelProcessing._process_pools:
            ParallelProcessing._process_pools[pool_key] = ProcessPoolExecutor(
                max_workers=max_workers
            )
        
        return ParallelProcessing._process_pools[pool_key]
    
    @staticmethod
    def _thread_map(items, func, max_workers, chunk_size, timeout_per_item, progress_callback):
        """Implementación de map usando threads."""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Crear futures
            futures = []
            for i, item in enumerate(items):
                future = executor.submit(func, item)
                future.item_index = i
                futures.append(future)
            
            # Recolectar resultados
            results = [None] * len(items)
            completed = 0
            
            for future in concurrent.futures.as_completed(futures, timeout=timeout_per_item):
                try:
                    result = future.result(timeout=0)
                    results[future.item_index] = result
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(completed, len(items))
                        
                except concurrent.futures.TimeoutError:
                    results[future.item_index] = TimeoutError(
                        f"Item timeout after {timeout_per_item}s"
                    )
                    completed += 1
                except Exception as e:
                    results[future.item_index] = e
                    completed += 1
            
            return results
    
    @staticmethod
    def _process_map(items, func, max_workers, chunk_size, timeout_per_item, progress_callback):
        """Implementación de map usando procesos."""
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Usar map para preservar orden
            try:
                if chunk_size:
                    results = list(executor.map(
                        func, items, 
                        timeout=timeout_per_item,
                        chunksize=chunk_size
                    ))
                else:
                    results = list(executor.map(
                        func, items, 
                        timeout=timeout_per_item
                    ))
                
                # Callback de progreso
                if progress_callback:
                    progress_callback(len(items), len(items))
                
                return results
                
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"Process map timeout after {timeout_per_item}s")
    
    @staticmethod
    async def _async_map(items, func, max_workers, chunk_size, timeout_per_item, progress_callback):
        """Implementación de map usando async/await."""
        import asyncio
        
        # Convertir función síncrona a async si es necesario
        if not asyncio.iscoroutinefunction(func):
            # Ejecutar en thread pool para funciones síncronas
            loop = asyncio.get_event_loop()
            func_wrapper = lambda item: loop.run_in_executor(None, func, item)
        else:
            func_wrapper = func
        
        # Crear semáforo para limitar concurrencia
        semaphore = asyncio.Semaphore(max_workers)
        
        async def process_item(index, item):
            async with semaphore:
                try:
                    if timeout_per_item:
                        result = await asyncio.wait_for(
                            func_wrapper(item), 
                            timeout=timeout_per_item
                        )
                    else:
                        result = await func_wrapper(item)
                    
                    return index, result, None
                    
                except asyncio.TimeoutError:
                    return index, None, TimeoutError(f"Timeout after {timeout_per_item}s")
                except Exception as e:
                    return index, None, e
        
        # Ejecutar todas las tareas
        tasks = [process_item(i, item) for i, item in enumerate(items)]
        results = [None] * len(items)
        completed = 0
        
        for future in asyncio.as_completed(tasks):
            index, result, error = await future
            
            if error:
                results[index] = error
            else:
                results[index] = result
            
            completed += 1
            if progress_callback:
                # Ejecutar callback en thread si es síncrono
                if asyncio.iscoroutinefunction(progress_callback):
                    await progress_callback(completed, len(items))
                else:
                    loop = asyncio.get_event_loop()
                    loop.call_soon_threadsafe(
                        progress_callback, completed, len(items)
                    )
        
        return results
    
    @staticmethod
    def _hybrid_map(items, func, max_workers, chunk_size, timeout_per_item, progress_callback):
        """Implementación híbrida de map."""
        # Dividir trabajo entre CPU y I/O bound tasks
        # Implementación simplificada: usar threads por defecto
        return ParallelProcessing._thread_map(
            items, func, max_workers, chunk_size, 
            timeout_per_item, progress_callback
        )
    
    @staticmethod
    def _execute_with_pool(pool, tasks, load_balancing, fail_fast):
        """Ejecuta tareas con un pool de ejecutores."""
        from concurrent.futures import as_completed
        
        futures = {}
        results = {}
        
        # Enviar tareas iniciales
        initial_batch = min(len(tasks), pool._max_workers * 2)
        for task in tasks[:initial_batch]:
            future = pool.submit(
                ParallelProcessing._execute_task, task
            )
            futures[future] = task.id
        
        # Procesar mientras haya tareas pendientes
        task_iter = iter(tasks[initial_batch:])
        completed = 0
        
        while futures:
            # Esperar por la siguiente tarea completada
            done, _ = as_completed(
                list(futures.keys()), 
                timeout=1.0
            )
            
            for future in done:
                task_id = futures.pop(future)
                
                try:
                    result = future.result(timeout=0)
                    results[task_id] = result
                    completed += 1
                    
                    # Balanceo de carga: enviar nueva tarea si hay capacidad
                    if load_balancing and len(futures) < pool._max_workers:
                        try:
                            next_task = next(task_iter)
                            next_future = pool.submit(
                                ParallelProcessing._execute_task, next_task
                            )
                            futures[next_future] = next_task.id
                        except StopIteration:
                            pass
                    
                except Exception as e:
                    results[task_id] = ParallelResult(
                        task_id=task_id,
                        success=False,
                        error=str(e),
                        traceback=traceback.format_exc()
                    )
                    
                    if fail_fast:
                        # Cancelar todas las tareas pendientes
                        for f in list(futures.keys()):
                            f.cancel()
                        futures.clear()
                        break
        
        return results
    
    @staticmethod
    async def _execute_async(tasks, load_balancing, fail_fast):
        """Ejecuta tasks asincrónicamente."""
        import asyncio
        
        # Convertir tareas síncronas a async
        async def execute_async_task(task):
            try:
                # Registrar inicio
                ParallelProcessing._active_tasks[task.id] = {
                    'start_time': time.time(),
                    'task_type': 'async'
                }
                
                # Ejecutar
                start = time.time()
                
                if asyncio.iscoroutinefunction(task.function):
                    result = await task.function(*task.args, **task.kwargs)
                else:
                    # Ejecutar en thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, task.function, *task.args, **task.kwargs
                    )
                
                processing_time = time.time() - start
                
                # Actualizar estado
                ParallelProcessing._active_tasks[task.id].update({
                    'completed': True,
                    'processing_time': processing_time,
                    'end_time': time.time()
                })
                
                return ParallelResult(
                    task_id=task.id,
                    success=True,
                    result=result,
                    processing_time=processing_time,
                    completed_at=datetime.now()
                )
                
            except Exception as e:
                ParallelProcessing._active_tasks[task.id].update({
                    'completed': True,
                    'error': str(e),
                    'end_time': time.time()
                })
                
                return ParallelResult(
                    task_id=task.id,
                    success=False,
                    error=str(e),
                    traceback=traceback.format_exc(),
                    completed_at=datetime.now()
                )
        
        # Ejecutar con semáforo para limitar concurrencia
        semaphore = asyncio.Semaphore(50)  # Límite de concurrencia async
        
        async def bounded_execute(task):
            async with semaphore:
                return await execute_async_task(task)
        
        # Crear y ejecutar todas las tareas
        task_coros = [bounded_execute(task) for task in tasks]
        results_list = await asyncio.gather(*task_coros, return_exceptions=False)
        
        # Convertir a diccionario
        return {result.task_id: result for result in results_list}
    
    @staticmethod
    def _execute_task(task: ParallelTask) -> ParallelResult:
        """Función wrapper para ejecutar tareas y capturar resultados."""
        try:
            # Registrar inicio
            ParallelProcessing._active_tasks[task.id] = {
                'start_time': time.time(),
                'task_type': 'sync'
            }
            
            # Ejecutar
            start = time.time()
            
            if task.timeout:
                # Ejecutar con timeout
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(task.function, *task.args, **task.kwargs)
                    result = future.result(timeout=task.timeout)
            else:
                result = task.function(*task.args, **task.kwargs)
            
            processing_time = time.time() - start
            
            # Actualizar estado
            ParallelProcessing._active_tasks[task.id].update({
                'completed': True,
                'processing_time': processing_time,
                'end_time': time.time()
            })
            
            return ParallelResult(
                task_id=task.id,
                success=True,
                result=result,
                processing_time=processing_time,
                completed_at=datetime.now()
            )
            
        except concurrent.futures.TimeoutError:
            error_msg = f"Task timeout after {task.timeout}s"
            ParallelProcessing._active_tasks[task.id].update({
                'completed': True,
                'error': error_msg,
                'end_time': time.time()
            })
            
            return ParallelResult(
                task_id=task.id,
                success=False,
                error=error_msg,
                completed_at=datetime.now()
            )
            
        except Exception as e:
            ParallelProcessing._active_tasks[task.id].update({
                'completed': True,
                'error': str(e),
                'end_time': time.time()
            })
            
            return ParallelResult(
                task_id=task.id,
                success=False,
                error=str(e),
                traceback=traceback.format_exc(),
                completed_at=datetime.now()
            )
    
    @staticmethod
    def _validate_dependencies(dependencies: Dict[int, List[int]], task_count: int):
        """Valida que no haya ciclos en las dependencias."""
        # Verificar rangos
        for task_idx, deps in dependencies.items():
            if task_idx >= task_count:
                raise ValueError(f"Task index {task_idx} out of range (0-{task_count-1})")
            for dep in deps:
                if dep >= task_count:
                    raise ValueError(f"Dependency index {dep} out of range (0-{task_count-1})")
        
        # Verificar ciclos usando DFS
        visited = [False] * task_count
        rec_stack = [False] * task_count
        
        def has_cycle(v):
            visited[v] = True
            rec_stack[v] = True
            
            for dep in dependencies.get(v, []):
                if not visited[dep]:
                    if has_cycle(dep):
                        return True
                elif rec_stack[dep]:
                    return True
            
            rec_stack[v] = False
            return False
        
        for i in range(task_count):
            if not visited[i]:
                if has_cycle(i):
                    raise ValueError(f"Cyclic dependency detected in task graph")
    
    @staticmethod
    def _retry_errors(retry_func, error_indices, max_retries):
        """Reintenta elementos que fallaron."""
        results = []
        
        for attempt in range(max_retries):
            if not error_indices:
                break
            
            logger.info(f"Retry attempt {attempt + 1}/{max_retries} for {len(error_indices)} items")
            
            try:
                retry_results = retry_func(error_indices)
                
                # Actualizar resultados y preparar para siguiente reintento
                new_errors = []
                for i, (idx, result) in enumerate(zip(error_indices, retry_results)):
                    if isinstance(result, Exception):
                        new_errors.append(idx)
                        results.append(result)
                    else:
                        results.append(result)
                
                error_indices = new_errors
                
                if not error_indices:
                    break
                    
            except Exception as e:
                logger.warning(f"Retry attempt {attempt + 1} failed: {e}")
                # Si falla el reintento, agregar errores a resultados
                for idx in error_indices:
                    results.append(e)
                break
        
        return results