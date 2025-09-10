"""
Realtime explanation pipeline for cybersecurity anomaly detection.
Provides high-performance explanations suitable for production deployment and dashboard integration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Union
import logging
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import threading
from dataclasses import dataclass, asdict
import json

from .shap_explainer import ShapExplainer, ShapExplanation
from .lime_explainer import LimeExplainer, LimeExplanation
from .explanation_aggregator import ExplanationAggregator, AggregatedExplanation

class RealtimeExplanationRequest:
    """Request structure for realtime explanations"""
    def __init__(self, sample: np.ndarray, sample_id: str, timestamp: datetime,
                 priority: int = 1, methods: List[str] = None, callback: Optional[Callable] = None):
        self.sample = sample
        self.sample_id = sample_id
        self.timestamp = timestamp
        self.priority = priority
        self.methods = methods
        self.callback = callback
    
    def __lt__(self, other):
        """Support for priority queue comparison"""
        if not isinstance(other, RealtimeExplanationRequest):
            return NotImplemented
        # Higher priority (lower number) comes first, then by timestamp
        return (self.priority, self.timestamp) < (other.priority, other.timestamp)

@dataclass
class RealtimeExplanationResult:
    """Result structure for realtime explanations"""
    sample_id: str
    success: bool
    explanation: Optional[AggregatedExplanation] = None
    error: Optional[str] = None
    processing_time_ms: Optional[float] = None
    methods_used: Optional[List[str]] = None

class RealtimeExplainer:
    """High-performance realtime explanation pipeline for cybersecurity detection"""
    
    def __init__(self, model, training_data: np.ndarray, feature_names: List[str],
                 max_workers: int = 4, queue_size: int = 1000):
        """
        Initialize realtime explainer
        
        Args:
            model: Trained ML model
            training_data: Training data for explainer initialization
            feature_names: List of feature names
            max_workers: Maximum number of worker threads
            queue_size: Maximum size of request queue
        """
        self.model = model
        self.feature_names = feature_names
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        self.aggregator = ExplanationAggregator(feature_names)
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'successful_explanations': 0,
            'failed_explanations': 0,
            'avg_processing_time_ms': 0.0,
            'queue_size': 0,
            'active_workers': 0
        }
        
        # Request queue and processing
        self.request_queue = queue.PriorityQueue(maxsize=queue_size)
        self.result_callbacks = {}
        self.is_running = False
        self.workers = []
        
        # Configuration for performance optimization
        self.config = {
            'enable_shap': True,
            'enable_lime': False,  # Disabled by default for performance
            'shap_background_size': 100,
            'lime_num_samples': 1000,
            'fast_mode': True,  # Use approximations for speed
            'cache_explanations': False,  # Disabled to ensure fresh explanations each time
            'max_processing_time_ms': 5000  # Timeout for individual requests
        }
        
        # Initialize explainers in background
        self._initialize_explainers_async(training_data)
        
        # Explanation cache for similar samples
        self.explanation_cache = {}
        self.cache_similarity_threshold = 0.95
        
    def _initialize_explainers_async(self, training_data: np.ndarray):
        """Initialize explainers asynchronously to avoid blocking"""
        def init_explainers():
            try:
                # Initialize SHAP explainer
                if self.config['enable_shap']:
                    background_size = min(self.config['shap_background_size'], len(training_data))
                    background_sample = training_data[:background_size]
                    self.shap_explainer = ShapExplainer(
                        self.model, 
                        background_sample, 
                        self.feature_names
                    )
                    self.logger.info("SHAP explainer initialized")
                
                # Initialize LIME explainer
                if self.config['enable_lime']:
                    categorical_features = self._detect_categorical_features(training_data)
                    self.lime_explainer = LimeExplainer(
                        training_data, 
                        self.feature_names,
                        categorical_features
                    )
                    self.logger.info("LIME explainer initialized")
                    
            except Exception as e:
                self.logger.error(f"Failed to initialize explainers: {e}")
        
        # Run initialization in separate thread
        init_thread = threading.Thread(target=init_explainers, daemon=True)
        init_thread.start()
    
    def _detect_categorical_features(self, data: np.ndarray) -> List[int]:
        """Detect categorical features for LIME"""
        categorical_indices = []
        for i in range(data.shape[1]):
            unique_values = len(np.unique(data[:, i]))
            # Consider feature categorical if it has few unique values
            if unique_values <= 20 and unique_values < len(data) * 0.1:
                categorical_indices.append(i)
        return categorical_indices
    
    def start_processing(self):
        """Start the realtime processing workers"""
        if self.is_running:
            return
        
        self.is_running = True
        self.workers = []
        
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop, 
                name=f"ExplainerWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        self.logger.info(f"Started {self.max_workers} explanation workers")
    
    def stop_processing(self):
        """Stop the realtime processing workers"""
        self.is_running = False
        
        # Wake up workers
        for _ in range(self.max_workers):
            try:
                # Create shutdown request
                shutdown_request = RealtimeExplanationRequest(
                    sample=np.array([]),
                    sample_id="shutdown",
                    timestamp=datetime.now(),
                    priority=0
                )
                self.request_queue.put(shutdown_request, timeout=1)
            except queue.Full:
                pass
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)
        
        self.logger.info("Stopped explanation workers")
    
    def _worker_loop(self):
        """Main worker loop for processing explanation requests"""
        while self.is_running:
            try:
                # Get request from queue (with timeout)
                request = self.request_queue.get(timeout=1)
                
                if request.sample_id == "shutdown":  # Shutdown signal
                    break
                
                self.stats['active_workers'] += 1
                
                # Process the request
                result = self._process_request(request)
                
                # Execute callback if provided
                if request.callback:
                    try:
                        request.callback(result)
                    except Exception as e:
                        self.logger.error(f"Callback failed for {request.sample_id}: {e}")
                
                self.stats['active_workers'] -= 1
                self.request_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker error: {e}")
                self.stats['active_workers'] -= 1
    
    def _process_request(self, request: RealtimeExplanationRequest) -> RealtimeExplanationResult:
        """Process a single explanation request"""
        start_time = time.time()
        
        try:
            # Update stats
            self.stats['total_requests'] += 1
            
            # Check cache first
            if self.config['cache_explanations']:
                cached_result = self._check_cache(request.sample)
                if cached_result:
                    processing_time = (time.time() - start_time) * 1000
                    return RealtimeExplanationResult(
                        sample_id=request.sample_id,
                        success=True,
                        explanation=cached_result,
                        processing_time_ms=processing_time,
                        methods_used=["cached"]
                    )
            
            # Generate explanations
            methods_used = []
            shap_explanation = None
            lime_explanation = None
            
            # SHAP explanation
            if self.config['enable_shap'] and self.shap_explainer and \
               (not request.methods or "shap" in request.methods):
                try:
                    shap_explanation = self.shap_explainer.explain_prediction(request.sample)
                    methods_used.append("shap")
                except Exception as e:
                    self.logger.warning(f"SHAP explanation failed for {request.sample_id}: {e}")
            
            # LIME explanation (only if enabled and requested)
            if self.config['enable_lime'] and self.lime_explainer and \
               (request.methods and "lime" in request.methods):
                try:
                    lime_explanation = self.lime_explainer.explain_prediction(
                        self.model, 
                        request.sample,
                        num_samples=self.config['lime_num_samples']
                    )
                    methods_used.append("lime")
                except Exception as e:
                    self.logger.warning(f"LIME explanation failed for {request.sample_id}: {e}")
            
            # Aggregate explanations
            if shap_explanation or lime_explanation:
                aggregated = self.aggregator.aggregate_explanations(
                    shap_explanation, 
                    lime_explanation,
                    request.sample_id
                )
                
                # Cache the result
                if self.config['cache_explanations']:
                    self._cache_result(request.sample, aggregated)
                
                processing_time = (time.time() - start_time) * 1000
                self.stats['successful_explanations'] += 1
                self._update_avg_processing_time(processing_time)
                
                return RealtimeExplanationResult(
                    sample_id=request.sample_id,
                    success=True,
                    explanation=aggregated,
                    processing_time_ms=processing_time,
                    methods_used=methods_used
                )
            else:
                raise Exception("No explanation methods succeeded")
                
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.stats['failed_explanations'] += 1
            self.logger.error(f"Failed to process request {request.sample_id}: {e}")
            
            return RealtimeExplanationResult(
                sample_id=request.sample_id,
                success=False,
                error=str(e),
                processing_time_ms=processing_time
            )
    
    def explain_sample_async(self, sample: np.ndarray, sample_id: str = None,
                           priority: int = 1, methods: List[str] = None,
                           callback: Callable = None) -> bool:
        """
        Submit sample for asynchronous explanation
        
        Args:
            sample: Network traffic sample to explain
            sample_id: Unique identifier for the sample
            priority: Request priority (1=normal, 2=high, 3=critical)
            methods: List of explanation methods to use
            callback: Function to call when explanation is ready
            
        Returns:
            True if request was queued successfully
        """
        if not self.is_running:
            self.start_processing()
        
        if sample_id is None:
            sample_id = f"sample_{int(time.time()*1000000)}"
        
        request = RealtimeExplanationRequest(
            sample=sample,
            sample_id=sample_id,
            timestamp=datetime.now(),
            priority=priority,
            methods=methods or ["shap"],
            callback=callback
        )
        
        try:
            # Put request directly (priority handling is done by __lt__ method)
            self.request_queue.put(request, timeout=0.1)
            self.stats['queue_size'] = self.request_queue.qsize()
            return True
        except queue.Full:
            self.logger.warning(f"Request queue full, dropping request {sample_id}")
            return False
    
    def explain_sample_sync(self, sample: np.ndarray, sample_id: str = None,
                          methods: List[str] = None, timeout: float = 10.0) -> RealtimeExplanationResult:
        """
        Get synchronous explanation for a sample
        
        Args:
            sample: Network traffic sample to explain
            sample_id: Unique identifier for the sample
            methods: List of explanation methods to use
            timeout: Maximum time to wait for result
            
        Returns:
            RealtimeExplanationResult with explanation or error
        """
        if sample_id is None:
            sample_id = f"sync_{int(time.time()*1000000)}"
        
        result = None
        result_ready = threading.Event()
        
        def callback(res):
            nonlocal result
            result = res
            result_ready.set()
        
        # Submit async request
        success = self.explain_sample_async(
            sample, sample_id, priority=3, methods=methods, callback=callback
        )
        
        if not success:
            return RealtimeExplanationResult(
                sample_id=sample_id,
                success=False,
                error="Failed to queue request"
            )
        
        # Wait for result
        if result_ready.wait(timeout=timeout):
            return result
        else:
            return RealtimeExplanationResult(
                sample_id=sample_id,
                success=False,
                error=f"Timeout after {timeout} seconds"
            )
    
    def explain_batch_realtime(self, samples: np.ndarray, 
                             batch_id: str = None) -> List[RealtimeExplanationResult]:
        """
        Process a batch of samples for realtime explanation
        
        Args:
            samples: Batch of network traffic samples
            batch_id: Identifier for the batch
            
        Returns:
            List of explanation results
        """
        if batch_id is None:
            batch_id = f"batch_{int(time.time())}"
        
        results = []
        result_lock = threading.Lock()
        
        def callback(result):
            with result_lock:
                results.append(result)
        
        # Submit all samples
        for i, sample in enumerate(samples):
            sample_id = f"{batch_id}_sample_{i}"
            self.explain_sample_async(sample, sample_id, callback=callback)
        
        # Wait for all results (with timeout)
        timeout = 30.0
        start_time = time.time()
        
        while len(results) < len(samples) and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        # Sort results by sample order
        results.sort(key=lambda x: int(x.sample_id.split('_')[-1]))
        return results
    
    def _check_cache(self, sample: np.ndarray) -> Optional[AggregatedExplanation]:
        """Check if a similar sample explanation is cached"""
        if not self.explanation_cache:
            return None
        
        # Simple similarity check based on feature values
        sample_key = self._compute_sample_key(sample)
        
        for cached_key, cached_explanation in self.explanation_cache.items():
            similarity = self._compute_similarity(sample_key, cached_key)
            if similarity >= self.cache_similarity_threshold:
                return cached_explanation
        
        return None
    
    def _cache_result(self, sample: np.ndarray, explanation: AggregatedExplanation):
        """Cache explanation result"""
        if len(self.explanation_cache) > 1000:  # Limit cache size
            # Remove oldest entries
            oldest_key = next(iter(self.explanation_cache))
            del self.explanation_cache[oldest_key]
        
        sample_key = self._compute_sample_key(sample)
        self.explanation_cache[sample_key] = explanation
    
    def _compute_sample_key(self, sample: np.ndarray) -> tuple:
        """Compute a key for sample similarity comparison"""
        # Use quantized values for approximate matching
        return tuple(np.round(sample.flatten(), decimals=2))
    
    def _compute_similarity(self, key1: tuple, key2: tuple) -> float:
        """Compute similarity between two sample keys"""
        if len(key1) != len(key2):
            return 0.0
        
        # Compute cosine similarity
        vec1, vec2 = np.array(key1), np.array(key2)
        dot_product = np.dot(vec1, vec2)
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _update_avg_processing_time(self, processing_time: float):
        """Update average processing time statistics"""
        n = self.stats['successful_explanations']
        current_avg = self.stats['avg_processing_time_ms']
        self.stats['avg_processing_time_ms'] = ((n - 1) * current_avg + processing_time) / n
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return {
            **self.stats,
            'cache_size': len(self.explanation_cache),
            'explainers_ready': {
                'shap': self.shap_explainer is not None,
                'lime': self.lime_explainer is not None
            },
            'configuration': self.config
        }
    
    def update_configuration(self, config_updates: Dict[str, Any]):
        """Update configuration settings"""
        self.config.update(config_updates)
        self.logger.info(f"Configuration updated: {config_updates}")
    
    def clear_cache(self):
        """Clear the explanation cache"""
        self.explanation_cache.clear()
        self.logger.info("Explanation cache cleared")
    
    def format_for_dashboard_streaming(self, result: RealtimeExplanationResult) -> Dict[str, Any]:
        """Format result for dashboard streaming/websocket"""
        if not result.success:
            return {
                "sample_id": result.sample_id,
                "status": "error",
                "error": result.error,
                "processing_time_ms": result.processing_time_ms,
                "timestamp": datetime.now().isoformat()
            }
        
        dashboard_data = self.aggregator.format_for_dashboard(result.explanation)
        
        return {
            "sample_id": result.sample_id,
            "status": "success",
            "processing_time_ms": result.processing_time_ms,
            "methods_used": result.methods_used,
            "timestamp": datetime.now().isoformat(),
            **dashboard_data
        }