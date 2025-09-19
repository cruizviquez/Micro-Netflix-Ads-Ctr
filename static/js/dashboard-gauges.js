// Dashboard with speedometer gauges
let ctrGauge, engagementGauge, revenueGauge;
let realtimeChart;

// Initialize gauges
function initGauges() {
    // CTR Gauge
    const ctrTarget = document.getElementById('ctrGauge');
    const ctrOpts = {
        angle: -0.2,
        lineWidth: 0.2,
        radiusScale: 0.75,
        pointer: {
            length: 0.6,
            strokeWidth: 0.045,
            color: '#fff'
        },
        limitMax: false,
        limitMin: false,
        colorStart: '#e50914',
        colorStop: '#f40612',
        strokeColor: '#333',
        generateGradient: true,
        highDpiSupport: true,
        staticZones: [
            {strokeStyle: "#f03e3e", min: 0, max: 2},
            {strokeStyle: "#ffdd00", min: 2, max: 5},
            {strokeStyle: "#30b32d", min: 5, max: 10}
        ],
        staticLabels: {
            font: "10px sans-serif",
            labels: [0, 2, 4, 6, 8, 10],
            color: "#fff",
            fractionDigits: 0
        },
    };
    
    ctrGauge = new Gauge(ctrTarget).setOptions(ctrOpts);
    ctrGauge.maxValue = 10;
    ctrGauge.setMinValue(0);
    ctrGauge.animationSpeed = 32;
    
    // Engagement Gauge
    const engagementTarget = document.getElementById('engagementGauge');
    const engagementOpts = {
        angle: -0.2,
        lineWidth: 0.2,
        radiusScale: 0.75,
        pointer: {
            length: 0.6,
            strokeWidth: 0.045,
            color: '#fff'
        },
        limitMax: false,
        limitMin: false,
        colorStart: '#6FADCF',
        colorStop: '#8FC0DA',
        strokeColor: '#333',
        generateGradient: true,
        highDpiSupport: true,
    };
    
    engagementGauge = new Gauge(engagementTarget).setOptions(engagementOpts);
    engagementGauge.maxValue = 100;
    engagementGauge.setMinValue(0);
    engagementGauge.animationSpeed = 32;
    
    // Revenue Gauge
    const revenueTarget = document.getElementById('revenueGauge');
    const revenueOpts = {
        angle: -0.2,
        lineWidth: 0.2,
        radiusScale: 0.75,
        pointer: {
            length: 0.6,
            strokeWidth: 0.045,
            color: '#fff'
        },
        limitMax: false,
        limitMin: false,
        colorStart: '#30b32d',
        colorStop: '#4BC0C8',
        strokeColor: '#333',
        generateGradient: true,
        highDpiSupport: true,
    };
    
    revenueGauge = new Gauge(revenueTarget).setOptions(revenueOpts);
    revenueGauge.maxValue = 1000;
    revenueGauge.setMinValue(0);
    revenueGauge.animationSpeed = 32;
    
    // Set initial values
    updateGauges();
}

// Update gauge values
function updateGauges() {
    // Simulate real-time data
    const ctrValue = (Math.random() * 5 + 2).toFixed(2);
    const engagementValue = Math.floor(Math.random() * 40 + 40);
    const revenueValue = Math.floor(Math.random() * 600 + 200);
    
    ctrGauge.set(ctrValue);
    document.getElementById('ctrValue').textContent = ctrValue + '%';
    
    engagementGauge.set(engagementValue);
    document.getElementById('engagementValue').textContent = engagementValue;
    
    revenueGauge.set(revenueValue);
    document.getElementById('revenueValue').textContent = '$' + revenueValue;
    
    // Update other metrics with animation
    updateMetricTile('impressionsCount', Math.floor(Math.random() * 50000 + 10000));
    updateMetricTile('clicksCount', Math.floor(Math.random() * 2500 + 500));
    updateMetricTile('conversionRate', (Math.random() * 3 + 1).toFixed(1) + '%');
    updateMetricTile('activeUsers', Math.floor(Math.random() * 2000 + 500));
}

// Update metric tiles with animation
function updateMetricTile(id, value) {
    const element = document.getElementById(id);
    const tile = element.closest('.metric-tile');
    
    tile.classList.add('updating');
    element.textContent = typeof value === 'number' ? value.toLocaleString() : value;
    
    setTimeout(() => {
        tile.classList.remove('updating');
    }, 500);
}

// Initialize real-time chart
function initRealtimeChart() {
    const ctx = document.getElementById('realtimeChart').getContext('2d');
    
    realtimeChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: generateTimeLabels(),
            datasets: [{
                label: 'CTR %',
                data: generateInitialData(),
                borderColor: '#e50914',
                backgroundColor: 'rgba(229, 9, 20, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true
            }, {
                label: 'Engagement',
                data: generateInitialData(50),
                borderColor: '#6FADCF',
                backgroundColor: 'rgba(111, 173, 207, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            scales: {
                x: {
                    grid: {
                        color: '#333'
                    },
                    ticks: {
                        color: '#888'
                    }
                },
                y: {
                    grid: {
                        color: '#333'
                    },
                    ticks: {
                        color: '#888'
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#fff'
                    }
                }
            }
        }
    });
}

// Generate time labels
function generateTimeLabels() {
    const labels = [];
    const now = new Date();
    
    for (let i = 29; i >= 0; i--) {
        const time = new Date(now.getTime() - i * 60000);
        labels.push(time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }));
    }
    
    return labels;
}

// Generate initial data
function generateInitialData(base = 3) {
    return Array.from({ length: 30 }, () => Math.random() * 2 + base);
}

// Update chart with new data
function updateChart() {
    // Remove oldest data point
    realtimeChart.data.labels.shift();
    realtimeChart.data.datasets.forEach(dataset => {
        dataset.data.shift();
    });
    
    // Add new data point
    const now = new Date();
    realtimeChart.data.labels.push(now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }));
    
    realtimeChart.data.datasets[0].data.push(Math.random() * 2 + 3);
    realtimeChart.data.datasets[1].data.push(Math.random() * 20 + 40);
    
    realtimeChart.update('none');
}

// Initialize everything
document.addEventListener('DOMContentLoaded', function() {
    initGauges();
    initRealtimeChart();
    
    // Update gauges every 3 seconds
    setInterval(updateGauges, 3000);
    
    // Update chart every 2 seconds
    setInterval(updateChart, 2000);
});