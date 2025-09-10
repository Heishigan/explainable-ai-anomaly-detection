/**
 * Real-Time Explainable AI Dashboard JavaScript
 * Provides real-time updates, WebSocket connection, and interactive visualizations
 */

class ExplainableAIDashboard {
    constructor() {
        this.websocket = null;
        this.charts = {};
        this.isConnected = false;
        this.isPaused = false;
        this.detectionCount = 0;
        
        // Data storage
        this.detectionHistory = [];
        this.metricsData = {
            totalSamples: 0,
            attacksDetected: 0,
            normalDetected: 0,
            avgProcessingTime: 0,
            samplesPerMinute: 0
        };
        
        // Time-based data management
        this.timeBuckets = new Map(); // Map of timestamp -> {attacks: count, normal: count}
        this.bucketIntervalMs = 30000; // 30 seconds per bucket
        this.maxBuckets = 20; // Keep last 20 buckets
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.initializeCharts();
        this.connectWebSocket();
        this.startDataRefresh();
        this.startTimelineRefresh();
        
        console.log('🚀 Explainable AI Dashboard initialized');
    }
    
    setupEventListeners() {
        // Demo control buttons
        document.getElementById('demoStartBtn').addEventListener('click', () => this.startDemo());
        document.getElementById('demoStopBtn').addEventListener('click', () => this.stopDemo());
        
        // Feed controls
        document.getElementById('pauseBtn').addEventListener('click', () => this.togglePause());
        document.getElementById('clearBtn').addEventListener('click', () => this.clearFeed());
        
        // Auto-scroll management
        this.feedAutoScroll = true;
        this.setupFeedScrollManagement();
        
        // Live updates state
        this.liveUpdates = true;
        
        // Alert close
        document.getElementById('alertClose').addEventListener('click', () => this.hideAlert());
        
        // Time range selector
        document.getElementById('timeRange').addEventListener('change', (e) => {
            const hours = parseInt(e.target.value);
            this.updateTimelineChart(hours);
        });
        
        // Chart controls
        document.getElementById('refreshChart').addEventListener('click', () => {
            const hours = parseInt(document.getElementById('timeRange').value);
            this.updateTimelineChart(hours);
        });
        
        document.getElementById('toggleLive').addEventListener('click', () => {
            this.toggleLiveUpdates();
        });
        
        // Initialize timeline with appropriate interval
        this.initializeTimelineBuckets();
    }
    
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        console.log(`Connecting to WebSocket: ${wsUrl}`);
        
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onopen = (event) => {
            console.log('✅ WebSocket connected');
            this.updateConnectionStatus('connected');
            this.isConnected = true;
        };
        
        this.websocket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };
        
        this.websocket.onclose = (event) => {
            console.log('❌ WebSocket disconnected');
            this.updateConnectionStatus('disconnected');
            this.isConnected = false;
            
            // Attempt to reconnect after 5 seconds
            setTimeout(() => {
                if (!this.isConnected) {
                    console.log('🔄 Attempting to reconnect...');
                    this.connectWebSocket();
                }
            }, 5000);
        };
        
        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateConnectionStatus('disconnected');
        };
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'detection':
                this.handleNewDetection(data.data);
                break;
            case 'alert':
                this.showAlert(data.data);
                break;
            case 'metrics_update':
                this.updateMetrics(data.data);
                break;
            default:
                console.log('Unknown WebSocket message type:', data.type);
        }
    }
    
    handleNewDetection(detection) {
        if (this.isPaused) return;
        
        // Add to history
        this.detectionHistory.unshift(detection);
        if (this.detectionHistory.length > 100) {
            this.detectionHistory = this.detectionHistory.slice(0, 100);
        }
        
        // Update metrics
        this.updateMetricsFromDetection(detection);
        
        // Add to feed
        this.addDetectionToFeed(detection);
        
        // Update latest explanation
        this.updateLatestExplanation(detection);
        
        // Update charts
        this.updateChartsWithDetection(detection);
    }
    
    addDetectionToFeed(detection) {
        const feed = document.getElementById('detectionFeed');
        
        // Remove placeholder if it exists
        const placeholder = feed.querySelector('.feed-placeholder');
        if (placeholder) {
            placeholder.remove();
        }
        
        const detectionElement = this.createDetectionElement(detection);
        
        // Add smooth insertion animation
        detectionElement.style.opacity = '0';
        detectionElement.style.transform = 'translateX(30px)';
        
        feed.insertBefore(detectionElement, feed.firstChild);
        
        // Animate in
        setTimeout(() => {
            detectionElement.style.transition = 'all 0.3s ease';
            detectionElement.style.opacity = '1';
            detectionElement.style.transform = 'translateX(0)';
        }, 10);
        
        // Limit to 25 visible detections and fade out old ones
        const items = feed.querySelectorAll('.detection-item');
        if (items.length > 25) {
            const oldItems = Array.from(items).slice(25);
            oldItems.forEach(item => {
                item.style.transition = 'all 0.3s ease';
                item.style.opacity = '0';
                item.style.transform = 'translateX(-30px)';
                setTimeout(() => item.remove(), 300);
            });
        }
        
        // Auto-scroll to top if enabled
        if (this.feedAutoScroll && !this.isPaused) {
            feed.scrollTop = 0;
        }
    }
    
    createDetectionElement(detection) {
        const isAttack = detection.status === 'success' && detection.prediction === 1;
        const confidence = detection.confidence || 0;
        const processingTime = detection.processing_time_ms || 0;
        const timestamp = new Date();
        
        const element = document.createElement('div');
        element.className = `detection-item ${isAttack ? 'attack' : 'normal'}`;
        
        // Add click handler for expanded details
        element.onclick = () => this.toggleDetectionDetails(element, detection);
        element.style.cursor = 'pointer';
        
        element.innerHTML = `
            <div class="detection-header">
                <div class="detection-status">
                    <span class="detection-type ${isAttack ? 'attack' : 'normal'}">
                        ${isAttack ? '🚨 ATTACK DETECTED' : '✅ NORMAL TRAFFIC'}
                    </span>
                    <span class="detection-risk-badge ${this.getRiskClass(detection)}">
                        ${this.formatRiskLevel(detection.risk_level || 'unknown')}
                    </span>
                </div>
                <span class="detection-time" title="${timestamp.toLocaleString()}">
                    ${timestamp.toLocaleTimeString()}
                </span>
            </div>
            <div class="detection-details">
                <span class="detail-item">Confidence: ${(confidence * 100).toFixed(1)}%</span>
                <span class="detail-item">Processing: ${processingTime.toFixed(1)}ms</span>
                <span class="detail-item">Sample: ${(detection.sample_id || 'N/A').substring(0, 8)}...</span>
                ${detection.attack_type ? `<span class="detail-item attack-type">Type: ${detection.attack_type}</span>` : ''}
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill ${this.getConfidenceClass(confidence)}" 
                     style="width: ${confidence * 100}%"></div>
            </div>
            <div class="detection-expanded" style="display: none;">
                <!-- Expanded details will be inserted here -->
            </div>
        `;
        
        return element;
    }
    
    getConfidenceClass(confidence) {
        if (confidence >= 0.8) return 'high';
        if (confidence >= 0.6) return 'medium';
        return 'low';
    }
    
    getRiskClass(detection) {
        const risk = detection.risk_level || 'unknown';
        return risk.toLowerCase().replace(' ', '-');
    }
    
    formatRiskLevel(risk) {
        return risk.toUpperCase().replace('_', ' ');
    }
    
    toggleDetectionDetails(element, detection) {
        const expanded = element.querySelector('.detection-expanded');
        const isVisible = expanded.style.display !== 'none';
        
        if (isVisible) {
            expanded.style.display = 'none';
            element.classList.remove('expanded');
        } else {
            // Populate expanded details
            expanded.innerHTML = this.createExpandedDetails(detection);
            expanded.style.display = 'block';
            element.classList.add('expanded');
        }
    }
    
    createExpandedDetails(detection) {
        const features = detection.top_features || [];
        const methods = detection.explanation_methods || [];
        
        return `
            <div class="expanded-content">
                ${methods.length > 0 ? `
                    <div class="expanded-section">
                        <h5>Explanation Methods:</h5>
                        <p>${methods.join(', ')}</p>
                    </div>
                ` : ''}
                ${features.length > 0 ? `
                    <div class="expanded-section">
                        <h5>Key Features:</h5>
                        <div class="mini-features">
                            ${features.slice(0, 3).map(f => `
                                <div class="mini-feature">
                                    <span class="feature-name">${f.name || f.feature}</span>
                                    <span class="feature-value">${(f.importance || 0).toFixed(3)}</span>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}
                <div class="expanded-actions">
                    <button class="detail-btn" onclick="navigator.clipboard.writeText('${detection.id || ''}')">
                        📋 Copy ID
                    </button>
                </div>
            </div>
        `;
    }
    
    setupFeedScrollManagement() {
        const feed = document.getElementById('detectionFeed');
        let scrollTimeout;
        
        feed.addEventListener('scroll', () => {
            clearTimeout(scrollTimeout);
            
            // Disable auto-scroll if user manually scrolls
            if (feed.scrollTop > 50) {
                this.feedAutoScroll = false;
            }
            
            // Re-enable auto-scroll after 3 seconds of no scrolling
            scrollTimeout = setTimeout(() => {
                if (feed.scrollTop <= 50) {
                    this.feedAutoScroll = true;
                }
            }, 3000);
        });
    }
    
    updateLatestExplanation(detection) {
        const metaElement = document.getElementById('explanationMeta');
        const contentElement = document.getElementById('explanationContent');
        
        if (detection.status !== 'success' || !detection.top_features) {
            return;
        }
        
        // Update meta information
        const timestamp = new Date().toLocaleTimeString();
        const methods = detection.methods_used || ['SHAP'];
        metaElement.textContent = `${timestamp} | Methods: ${methods.join(', ')}`;
        
        // Update feature importance
        const features = detection.top_features || [];
        if (features.length > 0) {
            contentElement.innerHTML = `
                <div class="feature-importance">
                    <h4 style="color: #00d4ff; margin-bottom: 1rem;">Top Contributing Features</h4>
                    ${features.slice(0, 5).map(feature => `
                        <div class="feature-item">
                            <span class="feature-name">${feature.name || feature.feature}</span>
                            <div class="feature-value">
                                <div class="feature-bar">
                                    <div class="feature-bar-fill ${feature.importance > 0 ? 'positive' : 'negative'}"
                                         style="width: ${Math.abs(feature.importance) * 100}%"></div>
                                </div>
                                <span class="feature-score">${(feature.importance || 0).toFixed(3)}</span>
                            </div>
                        </div>
                    `).join('')}
                </div>
            `;
        }
    }
    
    updateMetricsFromDetection(detection) {
        this.metricsData.totalSamples++;
        
        if (detection.status === 'success' && detection.prediction === 1) {
            this.metricsData.attacksDetected++;
        } else if (detection.status === 'success') {
            this.metricsData.normalDetected++;
        }
        
        // Update processing time average
        if (detection.processing_time_ms) {
            const currentAvg = this.metricsData.avgProcessingTime;
            const newTime = detection.processing_time_ms;
            this.metricsData.avgProcessingTime = 
                (currentAvg * (this.metricsData.totalSamples - 1) + newTime) / this.metricsData.totalSamples;
        }
        
        this.updateMetricsDisplay();
    }
    
    updateMetricsDisplay() {
        document.getElementById('totalSamples').textContent = this.metricsData.totalSamples.toLocaleString();
        document.getElementById('attacksDetected').textContent = this.metricsData.attacksDetected.toLocaleString();
        
        const detectionRate = this.metricsData.totalSamples > 0 
            ? (this.metricsData.attacksDetected / this.metricsData.totalSamples * 100).toFixed(1)
            : 0;
        document.getElementById('detectionRate').textContent = `${detectionRate}% detection rate`;
        
        const avgTime = this.metricsData.avgProcessingTime.toFixed(1);
        document.getElementById('processingSpeed').textContent = `${avgTime}ms`;
        
        // Calculate samples per minute (rough estimate)
        const samplesPerMin = Math.round(this.metricsData.totalSamples / Math.max(1, Date.now() / 60000 - this.startTime / 60000));
        document.getElementById('throughput').textContent = `${samplesPerMin} samples/min`;
        
        // Update last update time
        document.getElementById('lastUpdate').textContent = `Last updated: ${new Date().toLocaleTimeString()}`;
    }
    
    initializeCharts() {
        // Timeline Chart
        const timelineCtx = document.getElementById('timelineChart').getContext('2d');
        this.charts.timeline = new Chart(timelineCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Attacks',
                    data: [],
                    borderColor: '#ff4444',
                    backgroundColor: 'rgba(255, 68, 68, 0.1)',
                    tension: 0.4,
                    fill: true
                }, {
                    label: 'Normal Traffic',
                    data: [],
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0, 255, 136, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    legend: {
                        labels: { color: '#ffffff' }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(26, 26, 62, 0.95)',
                        titleColor: '#00d4ff',
                        bodyColor: '#ffffff',
                        borderColor: '#333366',
                        borderWidth: 1
                    }
                },
                scales: {
                    x: {
                        type: 'category',
                        ticks: { 
                            color: '#cccccc',
                            maxRotation: 45
                        },
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        title: {
                            display: true,
                            text: 'Time',
                            color: '#cccccc'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        ticks: { 
                            color: '#cccccc',
                            stepSize: 1,
                            callback: function(value) {
                                return Number.isInteger(value) ? value : '';
                            }
                        },
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        title: {
                            display: true,
                            text: 'Detections',
                            color: '#cccccc'
                        }
                    }
                },
                animation: {
                    duration: 300
                }
            }
        });
        
        // Attack Types Chart
        const attackTypesCtx = document.getElementById('attackTypesChart').getContext('2d');
        this.charts.attackTypes = new Chart(attackTypesCtx, {
            type: 'doughnut',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    backgroundColor: [
                        '#ff4444', '#ff6666', '#ff8888', '#ffaaaa',
                        '#00ff88', '#22ffaa', '#44ffcc', '#66ffee'
                    ],
                    borderColor: '#1a1a3e',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { 
                            color: '#ffffff',
                            padding: 15,
                            usePointStyle: true
                        }
                    }
                }
            }
        });
    }
    
    updateChartsWithDetection(detection) {
        // Update time-based buckets
        const now = new Date();
        const bucketTime = this.getBucketTime(now);
        
        // Initialize bucket if it doesn't exist
        if (!this.timeBuckets.has(bucketTime)) {
            this.timeBuckets.set(bucketTime, { attacks: 0, normal: 0 });
        }
        
        // Update bucket counts
        const bucket = this.timeBuckets.get(bucketTime);
        if (detection.status === 'success' && detection.prediction === 1) {
            bucket.attacks++;
        } else if (detection.status === 'success') {
            bucket.normal++;
        }
        
        // Clean old buckets and update chart
        this.cleanOldBuckets();
        this.updateTimelineChart();
    }
    
    getBucketTime(timestamp) {
        // Round timestamp down to nearest bucket interval
        const time = new Date(timestamp);
        const bucketMs = Math.floor(time.getTime() / this.bucketIntervalMs) * this.bucketIntervalMs;
        return bucketMs;
    }
    
    cleanOldBuckets() {
        // Remove buckets older than maxBuckets
        const sortedTimes = Array.from(this.timeBuckets.keys()).sort((a, b) => a - b);
        const cutoff = sortedTimes.length - this.maxBuckets;
        
        if (cutoff > 0) {
            for (let i = 0; i < cutoff; i++) {
                this.timeBuckets.delete(sortedTimes[i]);
            }
        }
    }
    
    updateTimelineChart() {
        const timeline = this.charts.timeline;
        
        // Get sorted time buckets
        const sortedTimes = Array.from(this.timeBuckets.keys()).sort((a, b) => a - b);
        
        // Ensure we have continuous time intervals
        const now = Date.now();
        const startTime = now - (this.maxBuckets - 1) * this.bucketIntervalMs;
        
        const labels = [];
        const attackData = [];
        const normalData = [];
        
        for (let i = 0; i < this.maxBuckets; i++) {
            const bucketTime = startTime + i * this.bucketIntervalMs;
            const bucket = this.timeBuckets.get(bucketTime) || { attacks: 0, normal: 0 };
            
            // Format time label
            const timeLabel = new Date(bucketTime).toLocaleTimeString().slice(0, 5);
            labels.push(timeLabel);
            attackData.push(bucket.attacks);
            normalData.push(bucket.normal);
            
            // Ensure bucket exists for continuous timeline
            if (!this.timeBuckets.has(bucketTime)) {
                this.timeBuckets.set(bucketTime, { attacks: 0, normal: 0 });
            }
        }
        
        timeline.data.labels = labels;
        timeline.data.datasets[0].data = attackData;
        timeline.data.datasets[1].data = normalData;
        
        timeline.update('none');
    }
    
    updateConnectionStatus(status) {
        const statusElement = document.querySelector('.status-dot');
        const statusText = document.querySelector('.status-text');
        
        statusElement.className = `status-dot ${status}`;
        
        switch (status) {
            case 'connected':
                statusText.textContent = 'Connected';
                break;
            case 'connecting':
                statusText.textContent = 'Connecting...';
                break;
            case 'disconnected':
                statusText.textContent = 'Disconnected';
                break;
        }
    }
    
    showAlert(alertData) {
        const alertBar = document.getElementById('alertBar');
        const alertTitle = document.getElementById('alertTitle');
        const alertMessage = document.getElementById('alertMessage');
        
        alertTitle.textContent = alertData.title;
        alertMessage.textContent = alertData.message;
        
        alertBar.style.display = 'block';
        
        // Auto-hide after 10 seconds
        setTimeout(() => {
            this.hideAlert();
        }, 10000);
    }
    
    hideAlert() {
        document.getElementById('alertBar').style.display = 'none';
    }
    
    async startDemo() {
        try {
            const response = await fetch('/api/demo/start', { method: 'POST' });
            const result = await response.json();
            
            if (result.status === 'started') {
                document.getElementById('demoStartBtn').disabled = true;
                document.getElementById('demoStopBtn').disabled = false;
                this.startTime = Date.now();
                console.log('✅ Demo started');
            } else {
                console.log('ℹ️ Demo status:', result.message);
            }
        } catch (error) {
            console.error('Error starting demo:', error);
        }
    }
    
    async stopDemo() {
        try {
            const response = await fetch('/api/demo/stop', { method: 'POST' });
            const result = await response.json();
            
            if (result.status === 'stopped') {
                document.getElementById('demoStartBtn').disabled = false;
                document.getElementById('demoStopBtn').disabled = true;
                console.log('⏹️ Demo stopped');
            }
        } catch (error) {
            console.error('Error stopping demo:', error);
        }
    }
    
    togglePause() {
        this.isPaused = !this.isPaused;
        const pauseBtn = document.getElementById('pauseBtn');
        pauseBtn.textContent = this.isPaused ? '▶️ Resume' : '⏸️ Pause';
        
        // Update feed visual state
        const feed = document.getElementById('detectionFeed');
        if (this.isPaused) {
            feed.classList.add('paused');
        } else {
            feed.classList.remove('paused');
            this.feedAutoScroll = true; // Re-enable auto-scroll on resume
        }
        
        console.log(this.isPaused ? '⏸️ Feed paused' : '▶️ Feed resumed');
    }
    
    clearFeed() {
        const feed = document.getElementById('detectionFeed');
        feed.innerHTML = `
            <div class="feed-placeholder">
                <p>🔄 Feed cleared</p>
                <p class="placeholder-subtitle">New detections will appear here</p>
            </div>
        `;
        console.log('🗑️ Detection feed cleared');
    }
    
    startDataRefresh() {
        // Refresh dashboard data every 30 seconds
        setInterval(async () => {
            try {
                const response = await fetch('/api/dashboard/metrics');
                if (response.ok) {
                    const metrics = await response.json();
                    // Update any missing metrics from server
                }
            } catch (error) {
                console.error('Error refreshing data:', error);
            }
        }, 30000);
    }
    
    startTimelineRefresh() {
        // Refresh timeline chart every 30 seconds to maintain continuous time axis
        setInterval(() => {
            if (!this.isPaused && this.liveUpdates) {
                this.updateTimelineChart();
            }
        }, 30000);
    }
    
    toggleLiveUpdates() {
        this.liveUpdates = !this.liveUpdates;
        const toggleBtn = document.getElementById('toggleLive');
        toggleBtn.textContent = this.liveUpdates ? '⏸️' : '▶️';
        toggleBtn.title = this.liveUpdates ? 'Pause Live Updates' : 'Resume Live Updates';
        
        console.log(this.liveUpdates ? '▶️ Live updates enabled' : '⏸️ Live updates paused');
    }
    
    initializeTimelineBuckets() {
        // Initialize empty buckets for the current time range
        const now = Date.now();
        const startTime = now - (this.maxBuckets - 1) * this.bucketIntervalMs;
        
        for (let i = 0; i < this.maxBuckets; i++) {
            const bucketTime = startTime + i * this.bucketIntervalMs;
            this.timeBuckets.set(bucketTime, { attacks: 0, normal: 0 });
        }
        
        this.updateTimelineChart();
    }
    
    async updateTimelineChart(hours = 24) {
        try {
            const response = await fetch(`/api/dashboard/timeseries?hours=${hours}`);
            if (response.ok) {
                const data = await response.json();
                this.updateTimelineChartWithServerData(data, hours);
            }
        } catch (error) {
            console.error('Error updating timeline chart:', error);
        }
    }
    
    updateTimelineChartWithServerData(data, hours) {
        const timeline = this.charts.timeline;
        const timelineData = data.detection_timeline || [];
        
        // Adjust bucket interval based on time range
        if (hours === 1) {
            this.bucketIntervalMs = 30000; // 30 seconds
            this.maxBuckets = 20;
        } else if (hours === 6) {
            this.bucketIntervalMs = 180000; // 3 minutes
            this.maxBuckets = 20;
        } else {
            this.bucketIntervalMs = 600000; // 10 minutes
            this.maxBuckets = 24;
        }
        
        // Clear existing buckets and populate with server data
        this.timeBuckets.clear();
        
        timelineData.forEach(item => {
            const timestamp = new Date(item.timestamp).getTime();
            const bucketTime = this.getBucketTime(timestamp);
            this.timeBuckets.set(bucketTime, {
                attacks: item.attacks || 0,
                normal: item.normal || 0
            });
        });
        
        // Update chart with new data
        this.updateTimelineChart();
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new ExplainableAIDashboard();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        console.log('📱 Page hidden - maintaining connection');
    } else {
        console.log('👀 Page visible - refreshing data');
        if (window.dashboard && !window.dashboard.isConnected) {
            window.dashboard.connectWebSocket();
        }
    }
});