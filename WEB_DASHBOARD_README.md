# Web Dashboard - Phase 4: Interactive Web Interface

This document describes the comprehensive web dashboard implementation for the Explainable AI Cybersecurity system.

## Overview

Phase 4 adds a modern, real-time web interface that provides:
- **Live monitoring** of network traffic analysis
- **Interactive visualizations** with Chart.js
- **Real-time explanations** via WebSocket
- **Alert management** with multi-level severity
- **Attack analytics** and trend analysis
- **Responsive design** for desktop and mobile

## Architecture

### Backend: FastAPI + WebSocket
- **FastAPI Server**: High-performance async web server
- **WebSocket Integration**: Real-time data streaming
- **RESTful API**: Complete dashboard data endpoints  
- **System Integration**: Leverages existing DashboardDataManager
- **Performance Optimized**: Async processing, caching, background tasks

### Frontend: Modern Web Interface
- **Pure HTML5/CSS3/JavaScript**: No complex frameworks
- **Chart.js**: Interactive charts and visualizations
- **WebSocket Client**: Real-time updates and live feed
- **Responsive Design**: Mobile-friendly adaptive layout
- **Dark Theme**: Cybersecurity-appropriate visual design

## Installation

### 1. Install Web Dependencies
```bash
# Install required packages
pip install fastapi uvicorn[standard] websockets jinja2

# Or install all requirements
pip install -r requirements.txt
```

### 2. Verify System Requirements
```bash
# Test the web dashboard components
python test_web_dashboard.py

# Quick system readiness check
python run_web_dashboard.py --help
```

## Usage

### Quick Start
```bash
# Start the web dashboard (recommended)
python run_web_dashboard.py

# Or start directly
python web_dashboard.py

# Custom host/port
python run_web_dashboard.py --host 0.0.0.0 --port 8080
```

### Access Dashboard
1. **Open browser**: http://localhost:8000
2. **Start Demo**: Click "Start Demo" button
3. **Monitor**: Watch real-time predictions and explanations
4. **Interact**: Use controls to pause, clear, adjust time ranges

## Dashboard Features

### 1. Real-Time Metrics
- **Samples Processed**: Total network traffic analyzed
- **Attacks Detected**: Real-time attack count and detection rate
- **Processing Speed**: Average processing time and throughput
- **System Status**: Health and operational status

### 2. Interactive Charts
- **Detection Timeline**: Time-series chart of attacks vs normal traffic
- **Attack Type Distribution**: Pie chart of attack categories
- **Confidence Trends**: Analysis of prediction confidence levels
- **Time Range Controls**: 1 hour, 6 hours, 24 hours views

### 3. Live Detection Feed
- **Real-time Stream**: Live feed of predictions as they happen
- **Color Coded**: Red (attacks) vs Green (normal traffic)
- **Detailed Info**: Confidence scores, processing times, sample IDs
- **Feed Controls**: Pause, resume, clear functionality

### 4. Feature Explanations
- **Latest Explanation**: SHAP feature importance for recent predictions
- **Top Features**: Most influential features with importance scores  
- **Visual Indicators**: Positive/negative feature contributions
- **Method Information**: Shows explanation methods used (SHAP/LIME)

### 5. Alert System
- **Multi-Level Alerts**: Critical, High, Medium, Low, Info
- **Auto-Dismiss**: Configurable alert timeout
- **Real-time Notifications**: Instant alerts for high-risk detections
- **Alert History**: Track and manage alert notifications

## API Endpoints

### Dashboard Data
- `GET /` - Main dashboard HTML page
- `GET /health` - System health check
- `GET /api/dashboard/data` - Complete dashboard data snapshot
- `GET /api/dashboard/metrics` - Current system metrics
- `GET /api/dashboard/detections?limit=50` - Recent detection results
- `GET /api/dashboard/alerts` - Active alert notifications
- `GET /api/dashboard/analytics` - Attack analytics and trends
- `GET /api/dashboard/timeseries?hours=24` - Time-series data for charts

### Demo Control
- `POST /api/demo/start` - Start demonstration mode
- `POST /api/demo/stop` - Stop demonstration mode  
- `GET /api/demo/status` - Get demo running status

### Real-Time Updates
- `WebSocket /ws` - Real-time data streaming connection

## Configuration Options

### Command Line Arguments
```bash
# Host and port configuration
python web_dashboard.py --host 0.0.0.0 --port 8000

# Development mode with auto-reload
python web_dashboard.py --reload

# Environment configuration
python web_dashboard.py --env production
```

### Environment Variables
- `HOST`: Server host (default: 127.0.0.1)
- `PORT`: Server port (default: 8000)  
- `ENV`: Environment config (development/production/testing)

## File Structure
```
web/
├── templates/
│   └── dashboard.html          # Main dashboard HTML template
├── static/
│   ├── css/
│   │   └── dashboard.css       # Dashboard styles and theming  
│   └── js/
│       └── dashboard.js        # Real-time functionality and charts
web_dashboard.py                # FastAPI server implementation
run_web_dashboard.py            # Easy startup script
test_web_dashboard.py           # Web dashboard test suite
```

## Technical Details

### WebSocket Protocol
The dashboard uses WebSocket for real-time updates with message types:
- `detection`: New prediction with explanation
- `alert`: System alert notification  
- `metrics_update`: Updated system metrics

### Chart.js Integration
Real-time charts using Chart.js with:
- **Automatic updates**: Charts update as new data arrives
- **Smooth animations**: Transition effects for data changes
- **Interactive tooltips**: Hover details for data points
- **Responsive scaling**: Adapts to screen size changes

### Performance Features
- **Async Processing**: Non-blocking request handling
- **Connection Management**: Automatic WebSocket reconnection  
- **Data Caching**: Client-side caching for smooth updates
- **Memory Management**: Limited history to prevent memory leaks

## Integration with Existing System

The web dashboard seamlessly integrates with existing components:

### DashboardDataManager Integration
- Uses existing `DashboardDataManager` for data management
- Subscribes to real-time updates via callback system
- Leverages existing alert generation and metrics calculation

### RealtimeExplainer Integration  
- Connects to existing `RealtimeExplainer` for live predictions
- Uses `format_for_dashboard_streaming()` for WebSocket data
- Maintains compatibility with existing explanation pipeline

### Configuration System
- Uses existing `Config` classes for environment management
- Integrates with existing model loading and data preprocessing
- Maintains consistency with console demo and training systems

## Customization

### Styling
Edit `web/static/css/dashboard.css` to customize:
- Color schemes and themes
- Layout and spacing
- Chart styling and animations  
- Responsive breakpoints

### Functionality  
Modify `web/static/js/dashboard.js` to add:
- New chart types or visualizations
- Additional real-time features
- Custom alert handling
- Enhanced interaction controls

### Backend Features
Extend `web_dashboard.py` to add:
- New API endpoints
- Additional data sources
- Custom WebSocket message types
- Advanced analytics features

## Troubleshooting

### Common Issues

1. **FastAPI not installed**
   ```bash
   pip install fastapi uvicorn[standard] websockets jinja2
   ```

2. **Port already in use**
   ```bash
   python run_web_dashboard.py --port 8080
   ```

3. **WebSocket connection issues**
   - Check firewall settings
   - Verify no proxy blocking WebSocket connections
   - Try different browser or incognito mode

4. **System not ready**
   ```bash
   # Ensure models are trained
   python train_models.py
   
   # Verify test data is available  
   ls data/raw/UNSW_NB15_testing-set.csv
   ```

### Performance Optimization
- **Reduce background data size**: Edit SHAP background size in config
- **Limit detection history**: Adjust maxlen in detection storage
- **Increase processing interval**: Modify demo simulation timing
- **Optimize chart updates**: Reduce chart data points for smoother updates

## Security Considerations

### Network Security
- **Default binding**: Binds to localhost (127.0.0.1) by default
- **External access**: Use `--host 0.0.0.0` only in secure networks
- **No authentication**: Current version has no built-in auth (suitable for demo/internal use)

### Data Privacy
- **In-memory only**: No persistent storage of sensitive data
- **Local processing**: All analysis happens locally
- **No external dependencies**: Charts and UI assets can be served locally

## Future Enhancements

### Phase 5 Possibilities
- **Authentication**: User login and role-based access
- **Database integration**: Persistent storage for historical data
- **Advanced analytics**: ML-powered trend analysis
- **Multi-model support**: Switch between different trained models
- **Export capabilities**: PDF reports, CSV data export
- **Mobile app**: React Native or PWA mobile interface

### Production Deployment
- **Docker containerization**: Ready for container deployment
- **Load balancing**: Multiple server instances with shared state
- **SSL/HTTPS**: Secure connections for production use
- **Monitoring integration**: Prometheus/Grafana integration
- **Logging**: Structured logging for production debugging

## Conclusion

Phase 4 delivers a comprehensive, production-ready web interface that showcases the sophisticated explainable AI capabilities built in previous phases. The dashboard provides an intuitive, real-time view into network security analysis with professional-grade visualizations and interactivity.

The implementation leverages modern web technologies while maintaining simplicity and performance, making it suitable for both demonstration purposes and real-world deployment scenarios.