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
        self.realtime_explainer = None
        self.dashboard_manager = None
        
        # WebSocket connections
        self.websocket_connections: List[WebSocket] = []
        
        # Demo simulation
        self.demo_running = False
        self.demo_thread = None
        self.test_data = None
        self.original_test_data = None
        
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
                "data_loaded": self.test_data is not None and not self.test_data.empty
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
        """Load trained models and preprocessors"""
        logger.info("Loading trained models...")
        
        try:
            # Load preprocessor
            preprocessor_path = Path(self.config.PREPROCESSING_DIR) / 'preprocessor.joblib'
            if preprocessor_path.exists():
                self.preprocessor = NetworkDataPreprocessor.load(str(preprocessor_path))
                logger.info("Loaded preprocessor")
            else:
                logger.warning("Preprocessor not found")
                return False
            
            # Load best model
            trainer = ModelTrainer(results_dir=self.config.MODELS_DIR)
            try:
                best_model_name, self.model = trainer.get_best_model()
                if self.model:
                    logger.info(f"Loaded best model: {best_model_name}")
                else:
                    logger.error("No trained models found")
                    return False
            except Exception as e:
                logger.warning(f"Could not load best model: {e}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load trained components: {e}")
            return False
    
    async def _load_test_data(self) -> bool:
        """Load test data for demo"""
        logger.info("Loading test data...")
        
        try:
            if not os.path.exists(self.config.TEST_DATA_PATH):
                logger.error(f"Test data not found: {self.config.TEST_DATA_PATH}")
                return False
            
            self.original_test_data = pd.read_csv(self.config.TEST_DATA_PATH)
            self.test_data = self.preprocessor.transform(self.original_test_data)
            
            logger.info(f"Loaded {len(self.test_data):,} test samples")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            return False
    
    async def _initialize_explainer(self) -> bool:
        """Initialize the real-time explainer"""
        logger.info("Initializing real-time explainer...")
        
        try:
            # Get model feature requirements
            model_n_features = None
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'n_features_in_'):
                model_n_features = self.model.model.n_features_in_
            elif hasattr(self.model, 'n_features_in_'):
                model_n_features = self.model.n_features_in_
            
            prep_features = self.preprocessor.get_feature_names()
            prep_n_features = len(prep_features)
            
            logger.info(f"Model expects {model_n_features} features, preprocessor provides {prep_n_features}")
            
            if model_n_features and model_n_features != prep_n_features:
                logger.warning(f"Feature mismatch detected: model expects {model_n_features}, preprocessor provides {prep_n_features}")
                logger.info("Creating feature alignment wrapper for demo purposes")
                
                # Create a wrapper that handles feature mismatch
                from sklearn.base import BaseEstimator, TransformerMixin
                
                class FeatureAlignmentWrapper(BaseEstimator, TransformerMixin):
                    def __init__(self, model, expected_features, provided_feature_names):
                        self.model = model
                        self.expected_n_features = expected_features
                        self.provided_feature_names = provided_feature_names
                        
                    def predict(self, X):
                        X_aligned = self._align_features(X)
                        return self.model.predict(X_aligned)
                        
                    def predict_proba(self, X):
                        X_aligned = self._align_features(X)
                        return self.model.predict_proba(X_aligned)
                        
                    def _align_features(self, X):
                        import numpy as np
                        
                        if isinstance(X, np.ndarray):
                            current_n_features = X.shape[1]
                        else:
                            current_n_features = len(X.columns) if hasattr(X, 'columns') else X.shape[1]
                        
                        # If we have more features than expected, truncate
                        if current_n_features > self.expected_n_features:
                            if isinstance(X, np.ndarray):
                                return X[:, :self.expected_n_features]
                            else:
                                return X.iloc[:, :self.expected_n_features].values
                        
                        # If we have fewer features than expected, pad with zeros
                        elif current_n_features < self.expected_n_features:
                            if isinstance(X, np.ndarray):
                                padding = np.zeros((X.shape[0], self.expected_n_features - current_n_features))
                                return np.hstack([X, padding])
                            else:
                                X_array = X.values if hasattr(X, 'values') else X
                                padding = np.zeros((X_array.shape[0], self.expected_n_features - current_n_features))
                                return np.hstack([X_array, padding])
                        
                        # Perfect match
                        return X.values if hasattr(X, 'values') else X
                
                # Wrap the model
                wrapped_model = FeatureAlignmentWrapper(self.model, model_n_features, prep_features)
                
                # Use original background data for SHAP (before alignment)
                # The RealtimeExplainer will handle alignment separately for predictions
                background_sample = self.test_data[:100]
                background_data = background_sample.values  # Keep original 42 features for SHAP
                
                # Use only the actual preprocessor features for SHAP explanations
                # The model wrapper will handle the feature alignment, but SHAP should only
                # try to explain features that actually exist in the data
                feature_names = prep_features
                
                self.realtime_explainer = RealtimeExplainer(
                    model=wrapped_model,
                    training_data=background_data,
                    feature_names=feature_names,
                    max_workers=2,
                    queue_size=500
                )
            else:
                # No mismatch, use directly
                background_data = self.test_data[:100].values
                feature_names = prep_features
                
                self.realtime_explainer = RealtimeExplainer(
                    model=self.model,
                    training_data=background_data,
                    feature_names=feature_names,
                    max_workers=2,
                    queue_size=500
                )
            
            # Configure for web dashboard - use actual background data size
            background_size = min(50, len(background_data))
            self.realtime_explainer.update_configuration({
                'enable_shap': True,
                'enable_lime': False,
                'fast_mode': True,
                'cache_explanations': True,
                'shap_background_size': background_size
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
        
        self.demo_running = True
        self.demo_thread = threading.Thread(target=self._demo_simulation_loop, daemon=True)
        self.demo_thread.start()
        logger.info("Demo simulation started")
    
    def _stop_demo_simulation(self):
        """Stop demo simulation"""
        self.demo_running = False
        if self.demo_thread:
            self.demo_thread.join(timeout=5)
        logger.info("Demo simulation stopped")
    
    def _demo_simulation_loop(self):
        """Main demo simulation loop"""
        logger.info("Demo simulation loop started")
        
        while self.demo_running:
            try:
                # Select random sample
                sample_idx = np.random.choice(len(self.test_data))
                sample = self.test_data.iloc[sample_idx].values
                original_sample = self.original_test_data.iloc[sample_idx]
                
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
                    
                    # Broadcast real-time update
                    try:
                        dashboard_data = self.realtime_explainer.format_for_dashboard_streaming(result)
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
    
    # Update server config
    global dashboard_server
    dashboard_server = WebDashboardServer(args.env)
    app = dashboard_server.app
    
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