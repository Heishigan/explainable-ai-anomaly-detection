#!/usr/bin/env python3
"""
Web Dashboard Server for Explainable AI Cybersecurity System

This FastAPI-based web server provides a real-time dashboard interface
for monitoring cybersecurity anomaly detection with explainable AI insights.

Features:
- Real-time WebSocket streaming of predictions and explanations
- RESTful API endpoints for dashboard data
- Interactive web interface with live visualizations
- Integration with existing DashboardDataManager and RealtimeExplainer

Usage:
    python web_dashboard.py --host 0.0.0.0 --port 8000
    
    Then visit: http://localhost:8000
"""

import sys
import os
import asyncio
import json
import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# FastAPI and WebSocket
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import Request
import uvicorn

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules
from src.config.config import get_config
from src.data.preprocessor import NetworkDataPreprocessor
from src.models.model_trainer import ModelTrainer
from src.explainability.realtime_explainer import (
    RealtimeExplainer, RealtimeExplanationResult
)
from src.explainability.dashboard_interface import (
    DashboardDataManager, DashboardAlert, AlertLevel
)
import pandas as pd
import numpy as np
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WebDashboardServer:
    """Main web dashboard server class"""
    
    def __init__(self, config_env: str = 'development'):
        """Initialize the web dashboard server"""
        self.config = get_config(config_env)
        
        # Initialize FastAPI
        self.app = FastAPI(
            title="Explainable AI Cybersecurity Dashboard",
            description="Real-time monitoring and explanation dashboard for network intrusion detection",
            version="1.0.0"
        )
        
        # System components
        self.model = None
        self.preprocessor = None
        self.feature_engineer = None
        self.realtime_explainer = None
        self.dashboard_manager = None
        
        # WebSocket connections
        self.websocket_connections: List[WebSocket] = []
        
        # Demo simulation
        self.demo_running = False
        self.demo_thread = None
        self.test_data = None
        self.original_test_data = None
        
        # Stratified sampling for demo
        self.attack_samples = None
        self.normal_samples = None
        self.attack_indices = None
        self.normal_indices = None
        self.demo_alternating = True  # Alternate between attack and normal
        self.demo_stats = {'attacks_shown': 0, 'normal_shown': 0, 'total_shown': 0}
        
        # Setup routes and static files
        self._setup_routes()
        self._setup_static_files()
        
    def _setup_static_files(self):
        """Setup static file serving and templates"""
        # Create web directory structure if it doesn't exist
        web_dir = Path("web")
        web_dir.mkdir(exist_ok=True)
        (web_dir / "static").mkdir(exist_ok=True)
        (web_dir / "templates").mkdir(exist_ok=True)
        
        # Mount static files
        self.app.mount("/static", StaticFiles(directory="web/static"), name="static")
        
        # Setup templates
        self.templates = Jinja2Templates(directory="web/templates")
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            """Serve the main dashboard page"""
            return self.templates.TemplateResponse(
                "dashboard.html", 
                {"request": request, "title": "Cybersecurity AI Dashboard"}
            )
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "system_ready": self.realtime_explainer is not None
            }
        
        @self.app.get("/api/dashboard/data")
        async def get_dashboard_data():
            """Get complete dashboard data snapshot"""
            if not self.dashboard_manager:
                raise HTTPException(status_code=503, detail="Dashboard not initialized")
            
            try:
                data = self.dashboard_manager.get_dashboard_data()
                return JSONResponse(content=data)
            except Exception as e:
                logger.error(f"Error getting dashboard data: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/dashboard/metrics")
        async def get_metrics():
            """Get current system metrics"""
            if not self.dashboard_manager:
                raise HTTPException(status_code=503, detail="Dashboard not initialized")
            
            try:
                data = self.dashboard_manager.get_dashboard_data()
                return JSONResponse(content=data["metrics"])
            except Exception as e:
                logger.error(f"Error getting metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/dashboard/detections")
        async def get_recent_detections(limit: int = 50):
            """Get recent detection results"""
            if not self.dashboard_manager:
                raise HTTPException(status_code=503, detail="Dashboard not initialized")
            
            try:
                data = self.dashboard_manager.get_detections_paginated(page=1, per_page=limit)
                return JSONResponse(content=data)
            except Exception as e:
                logger.error(f"Error getting detections: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/dashboard/alerts")
        async def get_active_alerts():
            """Get active alerts"""
            if not self.dashboard_manager:
                raise HTTPException(status_code=503, detail="Dashboard not initialized")
            
            try:
                data = self.dashboard_manager.get_dashboard_data()
                return JSONResponse(content={"alerts": data["active_alerts"]})
            except Exception as e:
                logger.error(f"Error getting alerts: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/dashboard/analytics")
        async def get_attack_analytics():
            """Get attack analytics and trends"""
            if not self.dashboard_manager:
                raise HTTPException(status_code=503, detail="Dashboard not initialized")
            
            try:
                analytics = self.dashboard_manager.get_attack_analytics()
                return JSONResponse(content=analytics)
            except Exception as e:
                logger.error(f"Error getting analytics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/dashboard/timeseries")
        async def get_timeseries_data(hours: int = 24):
            """Get time-series data for charts"""
            if not self.dashboard_manager:
                raise HTTPException(status_code=503, detail="Dashboard not initialized")
            
            try:
                data = self.dashboard_manager.get_time_series_data(hours=hours)
                return JSONResponse(content=data)
            except Exception as e:
                logger.error(f"Error getting timeseries data: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/demo/start")
        async def start_demo():
            """Start the demo simulation"""
            if self.demo_running:
                return {"status": "already_running", "message": "Demo is already running"}
            
            if not self.realtime_explainer or self.test_data is None or self.test_data.empty:
                raise HTTPException(status_code=503, detail="System not ready for demo")
            
            self._start_demo_simulation()
            return {"status": "started", "message": "Demo simulation started"}
        
        @self.app.post("/api/demo/stop")
        async def stop_demo():
            """Stop the demo simulation"""
            self._stop_demo_simulation()
            return {"status": "stopped", "message": "Demo simulation stopped"}
        
        @self.app.get("/api/demo/status")
        async def get_demo_status():
            """Get demo status"""
            return {
                "running": self.demo_running,
                "system_ready": self.realtime_explainer is not None,
                "data_loaded": self.test_data is not None and not self.test_data.empty,
                "demo_stats": self.demo_stats,
                "alternating_mode": self.demo_alternating
            }
        
        @self.app.get("/api/demo/stats")
        async def get_demo_stats():
            """Get demo statistics"""
            if self.attack_samples is None or self.normal_samples is None:
                return {"error": "Demo data not loaded"}
            
            total_attacks = len(self.attack_samples)
            total_normal = len(self.normal_samples)
            total_samples = total_attacks + total_normal
            
            return {
                "dataset_stats": {
                    "total_samples": total_samples,
                    "attack_samples": total_attacks,
                    "normal_samples": total_normal,
                    "attack_percentage": round(total_attacks / total_samples * 100, 1),
                    "normal_percentage": round(total_normal / total_samples * 100, 1)
                },
                "demo_stats": self.demo_stats.copy(),
                "demo_mode": {
                    "alternating": self.demo_alternating,
                    "pattern": "2 attacks : 1 normal" if self.demo_alternating else "random"
                }
            }
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await websocket.accept()
            self.websocket_connections.append(websocket)
            logger.info(f"WebSocket connection established. Total connections: {len(self.websocket_connections)}")
            
            try:
                while True:
                    # Keep connection alive
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)
                logger.info(f"WebSocket connection closed. Total connections: {len(self.websocket_connections)}")
    
    async def initialize_system(self) -> bool:
        """Initialize all system components"""
        logger.info("Initializing explainable AI system...")
        
        try:
            # Load models and preprocessors
            if not await self._load_trained_components():
                return False
            
            # Load test data
            if not await self._load_test_data():
                return False
            
            # Initialize explainer
            if not await self._initialize_explainer():
                return False
            
            # Initialize dashboard
            await self._initialize_dashboard()
            
            logger.info("System initialization complete!")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    async def _load_trained_components(self) -> bool:
        """Load trained models and preprocessors from existing results"""
        logger.info("Loading trained components...")
        
        try:
            # Load preprocessor from existing results
            preprocessor_path = Path('results/preprocessing/preprocessor.joblib')
            if preprocessor_path.exists():
                self.preprocessor = NetworkDataPreprocessor.load(str(preprocessor_path))
                logger.info("Loaded preprocessor")
            else:
                logger.error("Preprocessor not found. Please run training first:")
                logger.error("  python train_models.py")
                return False
            
            # Load feature engineer - check if it exists in preprocessing results
            from src.data.feature_engineering import NetworkFeatureEngineer
            feature_engineer_path = Path('results/preprocessing/feature_engineer.joblib')
            if feature_engineer_path.exists():
                self.feature_engineer = NetworkFeatureEngineer.load(str(feature_engineer_path))
                logger.info("Loaded feature engineer")
            else:
                # If not found, create a new one and initialize it to match the existing preprocessor
                logger.warning("Feature engineer not found, creating new one...")
                self.feature_engineer = NetworkFeatureEngineer()
                # Load some sample data to initialize feature engineer
                if os.path.exists(self.config.TRAIN_DATA_PATH):
                    sample_data = pd.read_csv(self.config.TRAIN_DATA_PATH).head(1000)
                    processed_sample = self.preprocessor.transform(sample_data)
                    
                    # Apply feature engineering and selection to match training pipeline
                    enhanced_sample = self.feature_engineer.create_derived_features(processed_sample)
                    if 'label' in sample_data.columns:
                        # Perform feature selection with labels
                        selected_features = self.feature_engineer.select_features(
                            enhanced_sample, 
                            sample_data['label'].head(1000),
                            method='f_score',  # Faster method
                            k_features=min(50, enhanced_sample.shape[1])
                        )
                        logger.info(f"Initialized feature engineer with {len(self.feature_engineer.selected_features)} features")
                    else:
                        # If no labels, just create derived features
                        self.feature_engineer.selected_features = enhanced_sample.columns.tolist()
                        logger.info("Initialized feature engineer without feature selection (no labels)")
                    
                    # Save for future use
                    feature_engineer_path.parent.mkdir(exist_ok=True, parents=True)
                    self.feature_engineer.save(str(feature_engineer_path))
                    logger.info("Created and saved new feature engineer")
                else:
                    logger.error("Cannot create feature engineer without training data")
                    return False
            
            # Load model from existing results
            from src.models.ensemble_models import XGBoostDetector
            model_path = Path('results/models/xgboost_model.joblib')
            if model_path.exists():
                self.model = XGBoostDetector.load(str(model_path))
                logger.info("Loaded XGBoost model")
            else:
                logger.error("XGBoost model not found. Please run training first:")
                logger.error("  python train_models.py")
                return False
            
            # Validate feature counts
            model_features = self.model.model.n_features_in_
            expected_features = len(self.feature_engineer.selected_features)
            
            logger.info(f"Feature validation:")
            logger.info(f"  Model expects: {model_features} features")
            logger.info(f"  Feature engineer provides: {expected_features} features")
            
            if model_features != expected_features:
                logger.warning(f"Feature count mismatch: model expects {model_features}, feature engineer provides {expected_features}")
                logger.warning("This may cause issues. Consider retraining with consistent pipeline.")
                # Don't fail - let it proceed and see if it works
            
            logger.info(f"Components loaded successfully: {model_features} model features")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load trained components: {e}")
            return False
    
    def _load_fallback_attack_samples(self) -> bool:
        """
        Fallback method to load attack samples directly from test CSV when 
        regular data pipeline fails to find attack samples.
        """
        try:
            logger.info("[FALLBACK] Loading attack samples directly from test CSV...")
            
            # Load raw test data directly
            import pandas as pd
            test_csv_path = "data/raw/UNSW_NB15_testing-set.csv"
            
            if not os.path.exists(test_csv_path):
                logger.error(f"[FALLBACK] Test CSV not found: {test_csv_path}")
                return False
            
            raw_test_df = pd.read_csv(test_csv_path)
            logger.info(f"[FALLBACK] Loaded raw test data: {raw_test_df.shape}")
            
            # Check label distribution
            if 'label' not in raw_test_df.columns:
                logger.error("[FALLBACK] No 'label' column in raw test CSV")
                return False
                
            label_counts = raw_test_df['label'].value_counts()
            logger.info(f"[FALLBACK] Raw label distribution: {label_counts.to_dict()}")
            
            # Extract attack samples
            attack_samples_raw = raw_test_df[raw_test_df['label'] == 1].head(1000)  # Limit to 1000 for performance
            
            if len(attack_samples_raw) == 0:
                logger.error("[FALLBACK] No attack samples found in raw test CSV")
                return False
                
            logger.info(f"[FALLBACK] Found {len(attack_samples_raw)} attack samples in raw CSV")
            
            # Process attack samples through the full pipeline
            logger.info("[FALLBACK] Processing attack samples through pipeline...")
            step1_preprocessed = self.preprocessor.transform(attack_samples_raw)
            step2_engineered = self.feature_engineer.transform_new_data(step1_preprocessed)
            
            # Store processed attack samples
            self.attack_samples = step2_engineered
            self.attack_indices = np.arange(len(step2_engineered))  # Create new indices
            
            # Update original data reference for these samples
            fallback_original_data = attack_samples_raw.reset_index(drop=True)
            
            logger.info(f"[FALLBACK] Successfully processed {len(self.attack_samples)} attack samples")
            logger.info(f"[FALLBACK] Attack sample shape: {self.attack_samples.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"[FALLBACK] Failed to load fallback attack samples: {e}")
            return False
    
    async def _load_test_data(self) -> bool:
        """Load test data for demo with stratified sampling using complete pipeline"""
        logger.info("Loading test data...")
        
        try:
            if not os.path.exists(self.config.TEST_DATA_PATH):
                logger.error(f"Test data not found: {self.config.TEST_DATA_PATH}")
                return False
            
            self.original_test_data = pd.read_csv(self.config.TEST_DATA_PATH)
            
            # Apply complete pipeline: preprocessor -> feature engineer
            step1_preprocessed = self.preprocessor.transform(self.original_test_data)
            step2_engineered = self.feature_engineer.transform_new_data(step1_preprocessed)
            
            self.test_data = step2_engineered
            
            # Separate attack and normal samples for demo
            logger.info("Analyzing original test data for attack/normal separation...")
            logger.info(f"  - Original data shape: {self.original_test_data.shape}")
            logger.info(f"  - Original data columns: {list(self.original_test_data.columns)}")
            
            # Check label distribution in original data
            if 'label' in self.original_test_data.columns:
                original_label_counts = self.original_test_data['label'].value_counts()
                logger.info(f"  - Original label distribution: {original_label_counts.to_dict()}")
            else:
                logger.error("  - ERROR: 'label' column not found in original test data!")
                logger.info(f"  - Available columns: {list(self.original_test_data.columns)}")
            
            attack_mask = self.original_test_data['label'] == 1
            normal_mask = self.original_test_data['label'] == 0
            
            logger.info(f"  - Attack mask: {attack_mask.sum()} True values out of {len(attack_mask)}")
            logger.info(f"  - Normal mask: {normal_mask.sum()} True values out of {len(normal_mask)}")
            
            # Get indices for both processed and original data
            self.attack_indices = np.where(attack_mask)[0]
            self.normal_indices = np.where(normal_mask)[0]
            
            logger.info(f"  - Attack indices: {len(self.attack_indices)} samples")
            logger.info(f"  - Normal indices: {len(self.normal_indices)} samples")
            
            # Store separated samples (using engineered features)
            self.attack_samples = self.test_data[attack_mask]
            self.normal_samples = self.test_data[normal_mask]
            
            attack_count = len(self.attack_samples)
            normal_count = len(self.normal_samples)
            total_count = len(self.test_data)
            
            logger.info(f"Demo sample separation completed:")
            logger.info(f"  - Total processed samples: {total_count:,}")
            logger.info(f"  - Attack samples available: {attack_count:,} ({attack_count/total_count*100:.1f}%)")
            logger.info(f"  - Normal samples available: {normal_count:,} ({normal_count/total_count*100:.1f}%)")
            
            # Validation checks and fallback loading
            if attack_count == 0:
                logger.error("[CRITICAL] No attack samples found! Attempting fallback loading...")
                logger.error("  - Check if test data contains samples with label=1")
                logger.error("  - Verify data loading and preprocessing pipeline")
                
                # Fallback: Load attack samples directly from test CSV
                if self._load_fallback_attack_samples():
                    attack_count = len(self.attack_samples)
                    logger.info(f"[FALLBACK SUCCESS] Loaded {attack_count} attack samples directly from test CSV")
                else:
                    logger.error("[FALLBACK FAILED] Could not load attack samples - dashboard will only show normal traffic")
                    
            elif attack_count < 100:
                logger.warning(f"[WARNING] Only {attack_count} attack samples found. Demo may not be representative.")
            
            if normal_count == 0:
                logger.error("[CRITICAL] No normal samples found!")
            
            logger.info(f"Complete pipeline: Raw ({len(self.original_test_data.columns)} cols) -> Preprocessor ({step1_preprocessed.shape[1]} cols) -> Feature Engineer ({self.test_data.shape[1]} cols)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            return False
    
    async def _initialize_explainer(self) -> bool:
        """Initialize the real-time explainer with loaded components"""
        logger.info("Initializing real-time explainer...")
        
        try:
            # Get background data (already processed through complete pipeline)
            background_sample = self.test_data[:100]
            background_data = background_sample.values
            
            # Use feature engineer's selected feature names
            feature_names = self.feature_engineer.selected_features
            
            # Validate feature consistency
            model_features = self.model.model.n_features_in_
            data_features = background_data.shape[1]
            
            logger.info(f"Pipeline validation:")
            logger.info(f"  Model expects: {model_features} features")
            logger.info(f"  Background data has: {data_features} features")
            logger.info(f"  Feature names: {len(feature_names)} features")
            
            if model_features != data_features:
                logger.warning(f"Feature count mismatch: model={model_features}, data={data_features}")
                # Try to proceed with available features
                if data_features < model_features:
                    logger.error("Background data has fewer features than model expects!")
                    return False
                else:
                    # Trim background data to match model
                    background_data = background_data[:, :model_features]
                    logger.info(f"Trimmed background data to {model_features} features")
            
            # Initialize explainer with available data
            self.realtime_explainer = RealtimeExplainer(
                model=self.model,
                training_data=background_data,
                feature_names=feature_names[:model_features],  # Match feature names to model
                max_workers=2,
                queue_size=500
            )
            
            # Configure for web dashboard
            self.realtime_explainer.update_configuration({
                'enable_shap': True,
                'enable_lime': False,
                'fast_mode': True,
                'cache_explanations': True,
                'shap_background_size': min(50, len(background_data))
            })
            
            # Start processing
            self.realtime_explainer.start_processing()
            
            # Wait for initialization
            await asyncio.sleep(2)
            
            logger.info("Real-time explainer ready")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize explainer: {e}")
            return False
    
    async def _initialize_dashboard(self):
        """Initialize dashboard manager"""
        logger.info("Initializing dashboard manager...")
        
        self.dashboard_manager = DashboardDataManager(
            max_detections=1000,
            max_alerts=100,
            metrics_window_minutes=30
        )
        
        # Subscribe to real-time updates
        self.dashboard_manager.subscribe_to_updates(self._handle_dashboard_update)
        self.dashboard_manager.subscribe_to_alerts(self._handle_alert)
        
        logger.info("Dashboard manager ready")
    
    def _handle_dashboard_update(self, event_type: str, data: Any):
        """Handle dashboard updates for WebSocket broadcasting"""
        if self.websocket_connections:
            asyncio.create_task(self._broadcast_update(event_type, data))
    
    def _handle_alert(self, alert):
        """Handle new alerts"""
        if self.websocket_connections:
            alert_data = {
                "type": "alert",
                "data": {
                    "id": alert.id,
                    "level": alert.level.value,
                    "title": alert.title,
                    "message": alert.message,
                    "timestamp": alert.timestamp
                }
            }
            asyncio.create_task(self._broadcast_to_websockets(alert_data))
    
    async def _broadcast_update(self, event_type: str, data: Any):
        """Broadcast dashboard updates to WebSocket clients"""
        if not self.websocket_connections:
            return
        
        message = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        await self._broadcast_to_websockets(message)
    
    async def _broadcast_to_websockets(self, message: Dict):
        """Send message to all connected WebSocket clients"""
        if not self.websocket_connections:
            return
        
        disconnected = []
        message_str = json.dumps(message, default=str)
        
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(message_str)
            except Exception as e:
                logger.warning(f"Failed to send WebSocket message: {e}")
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected:
            try:
                self.websocket_connections.remove(websocket)
            except ValueError:
                pass
    
    def _start_demo_simulation(self):
        """Start demo simulation in background thread"""
        if self.demo_running:
            return
        
        # Reset demo statistics
        self.demo_stats = {'attacks_shown': 0, 'normal_shown': 0, 'total_shown': 0}
        
        # Pre-flight checks for demo simulation
        attack_available = len(self.attack_indices) if hasattr(self, 'attack_indices') else 0
        normal_available = len(self.normal_indices) if hasattr(self, 'normal_indices') else 0
        
        logger.info(f"Starting demo simulation with {attack_available} attack samples and {normal_available} normal samples")
        
        if attack_available == 0:
            logger.error("[DEMO ERROR] Cannot start demo - no attack samples available!")
            logger.error("  - Demo will show only normal traffic or fail")
            logger.error("  - Check data loading pipeline and sample separation logic")
        
        self.demo_running = True
        self.demo_thread = threading.Thread(target=self._demo_simulation_loop, daemon=True)
        self.demo_thread.start()
        logger.info(f"Demo simulation started with stratified sampling (2 attacks : 1 normal pattern)")
        logger.info(f"  - Target pattern: Show {attack_available} attacks, {normal_available} normal samples")
    
    def _stop_demo_simulation(self):
        """Stop demo simulation"""
        self.demo_running = False
        if self.demo_thread:
            self.demo_thread.join(timeout=5)
        logger.info("Demo simulation stopped")
    
    def _demo_simulation_loop(self):
        """Main demo simulation loop with stratified sampling"""
        logger.info("Demo simulation loop started")
        
        while self.demo_running:
            try:
                # Stratified sampling: alternate between attack and normal traffic
                if self.demo_alternating:
                    # Alternate pattern: show attacks more frequently for demo purposes
                    cycle_position = self.demo_stats['total_shown'] % 3
                    should_show_attack = cycle_position == 0 or cycle_position == 1
                    
                    if should_show_attack:
                        # Show attack (2 out of every 3 samples)
                        if len(self.attack_indices) == 0:
                            logger.warning(f"[DEMO] No attack samples available at cycle {self.demo_stats['total_shown']} - skipping attack, showing normal instead")
                            # Fallback to normal sample
                            if len(self.normal_indices) > 0:
                                sample_idx = np.random.choice(self.normal_indices)
                                is_attack = False
                            else:
                                logger.error("[DEMO] No samples available at all - stopping demo")
                                break
                        else:
                            sample_idx = np.random.choice(self.attack_indices)
                            is_attack = True
                            if self.demo_stats['total_shown'] % 10 == 0:  # Log every 10th sample
                                logger.info(f"[DEMO] Showing attack sample {sample_idx} (cycle {cycle_position})")
                    else:
                        # Show normal (1 out of every 3 samples)
                        if len(self.normal_indices) == 0:
                            logger.warning(f"[DEMO] No normal samples available at cycle {self.demo_stats['total_shown']} - skipping normal, showing attack instead")
                            # Fallback to attack sample
                            if len(self.attack_indices) > 0:
                                sample_idx = np.random.choice(self.attack_indices)
                                is_attack = True
                            else:
                                logger.error("[DEMO] No samples available at all - stopping demo")
                                break
                        else:
                            sample_idx = np.random.choice(self.normal_indices)
                            is_attack = False
                            if self.demo_stats['total_shown'] % 10 == 0:  # Log every 10th sample
                                logger.info(f"[DEMO] Showing normal sample {sample_idx} (cycle {cycle_position})")
                else:
                    # Fallback to random sampling
                    sample_idx = np.random.choice(len(self.test_data))
                    is_attack = self.original_test_data.iloc[sample_idx]['label'] == 1
                
                # Get the already processed sample (through complete pipeline)
                sample = self.test_data.iloc[sample_idx].values
                original_sample = self.original_test_data.iloc[sample_idx]
                
                # Update demo statistics
                self.demo_stats['total_shown'] += 1
                if is_attack:
                    self.demo_stats['attacks_shown'] += 1
                else:
                    self.demo_stats['normal_shown'] += 1
                
                # Get explanation
                result = self.realtime_explainer.explain_sample_sync(
                    sample=sample,
                    sample_id=f"demo_{int(time.time()*1000)}",
                    methods=["shap"],
                    timeout=10.0  # Increased timeout for web dashboard
                )
                
                if result.success and result.explanation:
                    # Add to dashboard
                    self.dashboard_manager.add_detection(result)
                    
                    # Log demo progress periodically
                    if self.demo_stats['total_shown'] % 10 == 0:
                        attacks = self.demo_stats['attacks_shown']
                        normal = self.demo_stats['normal_shown'] 
                        total = self.demo_stats['total_shown']
                        logger.info(f"Demo progress: {total} samples shown ({attacks} attacks, {normal} normal)")
                    
                    # Broadcast real-time update
                    try:
                        dashboard_data = self.realtime_explainer.format_for_dashboard_streaming(result)
                        # Add ground truth information for demo
                        dashboard_data['ground_truth'] = {
                            'actual_label': int(original_sample['label']),
                            'is_attack': is_attack,
                            'attack_category': original_sample.get('attack_cat', 'Normal') if is_attack else 'Normal'
                        }
                        dashboard_data['demo_stats'] = self.demo_stats.copy()
                        
                        asyncio.run(self._broadcast_to_websockets({
                            "type": "detection",
                            "data": dashboard_data
                        }))
                    except Exception as broadcast_error:
                        logger.warning(f"Failed to broadcast update: {broadcast_error}")
                else:
                    logger.warning(f"Failed to get explanation for sample {sample_idx}: {result.error_message if hasattr(result, 'error_message') else 'Unknown error'}")
                
                # Wait before next sample
                time.sleep(2)  # 2 seconds between samples
                
            except Exception as e:
                logger.error(f"Demo simulation error: {e}")
                time.sleep(5)  # Wait longer on error
        
        logger.info("Demo simulation loop ended")


# Global server instance
dashboard_server = WebDashboardServer()

# FastAPI app instance (for uvicorn)
app = dashboard_server.app

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    success = await dashboard_server.initialize_system()
    if not success:
        logger.error("Failed to initialize system")
    else:
        logger.info("Web dashboard server ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    dashboard_server._stop_demo_simulation()
    if dashboard_server.realtime_explainer:
        dashboard_server.realtime_explainer.stop_processing()
    logger.info("Web dashboard server shutdown complete")

def main():
    """Main function to run the server"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Explainable AI Web Dashboard Server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--env', choices=['development', 'production', 'testing'],
                       default='development', help='Environment configuration')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    
    args = parser.parse_args()
    
    # Use the global server instance with the specified environment
    if args.env != 'development':
        logger.warning(f"Environment {args.env} specified, but using development config (global instance)")
        logger.warning("For production deployment, modify the global server creation")
    
    logger.info(f"Starting web dashboard server on {args.host}:{args.port}")
    logger.info(f"Dashboard will be available at: http://{args.host}:{args.port}")
    
    uvicorn.run(
        "web_dashboard:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()