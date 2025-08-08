const { useState, useEffect, useRef } = React;

function Dashboard() {
    const [chartData, setChartData] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [files, setFiles] = useState([]);
    const [layoutConfig, setLayoutConfig] = useState({ charts_per_row: 4 });
    const [monitoringStatus, setMonitoringStatus] = useState(null);
    const [lastUpdate, setLastUpdate] = useState(null);
    const [autoRefresh, setAutoRefresh] = useState(true);
    const [refreshInterval, setRefreshInterval] = useState(null);
    
    // Animation and scaling state
    const [chartScales, setChartScales] = useState({});
    const [animationData, setAnimationData] = useState({});
    const animationRefs = useRef({});
    const previousData = useRef({});

    useEffect(() => {
        fetchInitialData();
        fetchMonitoringStatus();
        
        // Set up auto-refresh if enabled
        if (autoRefresh) {
            const interval = setInterval(() => {
                fetchChartDataOnly(); // Only fetch chart data, not full page
                fetchMonitoringStatus();
            }, 1000); // Default 1 second
            setRefreshInterval(interval);
        }

        // Add window resize listener for responsive chart dimensions
        const handleResize = () => {
            // Force re-render of charts when window is resized
            setChartData(prevData => [...prevData]);
        };

        window.addEventListener('resize', handleResize);

        return () => {
            if (refreshInterval) {
                clearInterval(refreshInterval);
            }
            window.removeEventListener('resize', handleResize);
        };
    }, [autoRefresh]);

    useEffect(() => {
        // Update CSS custom property for charts per row
        const chartsGrid = document.querySelector('.charts-grid');
        if (chartsGrid && layoutConfig.charts_per_row) {
            chartsGrid.style.setProperty('--charts-per-row', layoutConfig.charts_per_row);
        }
    }, [layoutConfig]);

    const fetchInitialData = async () => {
        try {
            setLoading(true);
            const [chartResponse, filesResponse, layoutResponse] = await Promise.all([
                fetch('/api/chart-data'),
                fetch('/api/files'),
                fetch('/api/layout-config')
            ]);

            if (!chartResponse.ok || !filesResponse.ok || !layoutResponse.ok) {
                throw new Error('Failed to fetch data');
            }

            const chartDataResult = await chartResponse.json();
            const filesResult = await filesResponse.json();
            const layoutResult = await layoutResponse.json();

            // Handle case where no data is available yet
            if (!chartDataResult || chartDataResult.length === 0) {
                setLoading(false);
                setError('No chart data available yet. Waiting for JSON files to be processed...');
                return;
            }

            // Initialize scales and animation data
            const initialScales = {};
            const initialAnimationData = {};
            
            console.log('Initial chart data:', chartDataResult);
            
            chartDataResult.forEach(chart => {
                const chartId = chart.chart_title;
                const maxValue = Math.max(...chart.datasets.map(d => d.value || 0));
                
                console.log(`Setting up chart: ${chartId}, maxValue: ${maxValue}`);
                
                // Set initial scale based on chart type
                if (chart.chart_title.includes('Percent') || chart.chart_title.includes('Percentage')) {
                    initialScales[chartId] = 100; // Percentage charts: 0-100
                } else {
                    // For other charts, start with 2x the max value or a reasonable default
                    initialScales[chartId] = Math.max(maxValue * 2, 1000);
                }
                
                console.log(`Scale for ${chartId}: ${initialScales[chartId]}`);
                
                // Initialize animation data with 0 values
                initialAnimationData[chartId] = chart.datasets.map(d => ({
                    ...d,
                    animatedValue: 0,
                    targetValue: d.value || 0
                }));
                
                console.log(`Animation data for ${chartId}:`, initialAnimationData[chartId]);
            });

            setChartScales(initialScales);
            setAnimationData(initialAnimationData);
            previousData.current = chartDataResult;
            
            // Sort and set data
            const sortedFiles = sortFiles(filesResult);
            const sortedChartData = sortChartData(chartDataResult);
            
            setChartData(sortedChartData);
            setFiles(sortedFiles);
            setLayoutConfig(layoutResult);
            setError(null);
            setLoading(false);
            
            // Trigger initial animation after a short delay to ensure state is set
            setTimeout(() => {
                console.log('Triggering initial animation...');
                animateCharts();
            }, 100);
            
        } catch (err) {
            console.error('Error fetching initial data:', err);
            setError('Failed to load dashboard data. Please check if the server is running and try again.');
            setLoading(false);
        }
    };

    const fetchChartDataOnly = async () => {
        try {
            const response = await fetch('/api/chart-data');
            if (!response.ok) {
                console.warn('Failed to fetch chart data, keeping existing data');
                return;
            }

            const newChartData = await response.json();
            
            // If no new data is available, keep existing data
            if (!newChartData || newChartData.length === 0) {
                console.log('No new chart data available, keeping existing data');
                return;
            }
            
            const sortedChartData = sortChartData(newChartData);
            
            // Update animation data with new target values
            const updatedAnimationData = { ...animationData };
            const updatedScales = { ...chartScales };
            
            sortedChartData.forEach(chart => {
                const chartId = chart.chart_title;
                const currentAnimationData = updatedAnimationData[chartId] || [];
                
                // Update target values and check for scale adjustments
                const newMaxValue = Math.max(...chart.datasets.map(d => d.value || 0));
                const currentScale = updatedScales[chartId];
                
                // Check if we need to increase scale (if tallest bar reaches 75% of current scale)
                if (newMaxValue > currentScale * 0.75 && !chart.chart_title.includes('Percent')) {
                    updatedScales[chartId] = currentScale * 2;
                }
                
                // Update animation targets
                updatedAnimationData[chartId] = chart.datasets.map((d, index) => ({
                    ...d,
                    animatedValue: currentAnimationData[index]?.animatedValue || 0,
                    targetValue: d.value || 0
                }));
            });
            
            setChartScales(updatedScales);
            setAnimationData(updatedAnimationData);
            setChartData(sortedChartData);
            setLastUpdate(new Date().toLocaleTimeString());
            
            // Animate to new values
            animateCharts();
            
        } catch (err) {
            console.error('Error fetching chart data:', err);
            // Don't update state on error, keep existing data
        }
    };

    const animateCharts = () => {
        console.log('Starting chart animations...');
        console.log('Current animation data:', animationData);
        
        Object.keys(animationData).forEach(chartId => {
            const chartAnimationData = animationData[chartId];
            if (!chartAnimationData) return;
            
            console.log(`Animating chart: ${chartId}`, chartAnimationData);
            
            chartAnimationData.forEach((dataset, index) => {
                const currentValue = dataset.animatedValue;
                const targetValue = dataset.targetValue;
                
                console.log(`Dataset ${index}: current=${currentValue}, target=${targetValue}`);
                
                // Start animation if values are different (not just if difference > 0.1)
                if (Math.abs(currentValue - targetValue) > 0.01) {
                    console.log(`Starting animation for ${chartId} dataset ${index}: ${currentValue} -> ${targetValue}`);
                    animateValue(chartId, index, currentValue, targetValue);
                } else {
                    console.log(`No animation needed for ${chartId} dataset ${index}: values are the same`);
                }
            });
        });
    };

    const animateValue = (chartId, datasetIndex, startValue, endValue) => {
        console.log(`animateValue called: ${chartId}[${datasetIndex}] ${startValue} -> ${endValue}`);
        
        const duration = 1000; // 1 second animation
        const startTime = performance.now();
        
        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Easing function for smooth animation
            const easeOutQuart = 1 - Math.pow(1 - progress, 4);
            const currentValue = startValue + (endValue - startValue) * easeOutQuart;
            
            console.log(`Animation progress: ${progress.toFixed(2)}, currentValue: ${currentValue.toFixed(2)}`);
            
            // Update animation data
            setAnimationData(prev => ({
                ...prev,
                [chartId]: prev[chartId].map((dataset, index) => 
                    index === datasetIndex 
                        ? { ...dataset, animatedValue: currentValue }
                        : dataset
                )
            }));
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                console.log(`Animation completed for ${chartId}[${datasetIndex}]`);
            }
        };
        
        requestAnimationFrame(animate);
    };

    const sortFiles = (files) => {
        return files.sort((a, b) => {
            const providerComparison = a.llm_provider.localeCompare(b.llm_provider);
            if (providerComparison !== 0) {
                return providerComparison;
            }
            return a.llm_model.localeCompare(b.llm_model);
        });
    };

    const sortChartData = (chartData) => {
        return chartData.map(chart => ({
            ...chart,
            datasets: chart.datasets.sort((a, b) => {
                const aParts = a.label.split('_');
                const bParts = b.label.split('_');
                
                if (aParts.length >= 2 && bParts.length >= 2) {
                    const aProvider = aParts[0];
                    const bProvider = bParts[0];
                    const aModel = aParts[1];
                    const bModel = bParts[1];
                    
                    const providerComparison = aProvider.localeCompare(bProvider);
                    if (providerComparison !== 0) {
                        return providerComparison;
                    }
                    return aModel.localeCompare(bModel);
                }
                
                return a.label.localeCompare(b.label);
            })
        }));
    };

    const fetchMonitoringStatus = async () => {
        try {
            const response = await fetch('/api/monitoring-status');
            if (response.ok) {
                const status = await response.json();
                setMonitoringStatus(status);
            }
        } catch (err) {
            console.error('Failed to fetch monitoring status:', err);
        }
    };

    const toggleAutoRefresh = () => {
        setAutoRefresh(!autoRefresh);
        if (refreshInterval) {
            clearInterval(refreshInterval);
            setRefreshInterval(null);
        }
    };

    const forceRefresh = async () => {
        try {
            await fetch('/api/force-refresh');
            await fetchInitialData();
            setLastUpdate(new Date().toLocaleTimeString());
        } catch (err) {
            console.error('Failed to force refresh:', err);
        }
    };

    const generateDistinctColors = (count) => {
        const colors = [];
        
        // Moderately soft, sophisticated color palette
        const professionalColors = [
            '#4A5F7A', // Medium Blue-Gray
            '#8E44AD', // Medium Purple
            '#E67E22', // Medium Orange
            '#27AE60', // Medium Green
            '#C0392B', // Medium Red
            '#2980B9', // Medium Blue
            '#16A085', // Medium Teal
            '#F39C12', // Medium Yellow-Orange
            '#7F8C8D', // Medium Gray
            '#A0522D', // Medium Brown
            '#2E8B57', // Medium Mint
            '#FF6B6B', // Soft Red
            '#4ECDC4', // Soft Teal
            '#45B7D1', // Soft Blue
            '#96CEB4', // Soft Green
            '#FFEAA7', // Soft Yellow
            '#DDA0DD', // Soft Purple
            '#98D8C8', // Soft Mint
            '#F7DC6F', // Soft Gold
            '#BB8FCE'  // Soft Lavender
        ];
        
        for (let i = 0; i < count; i++) {
            colors.push(professionalColors[i % professionalColors.length]);
        }
        
        return colors;
    };

    const getColorForIndex = (index, totalCount) => {
        const colors = generateDistinctColors(totalCount);
        return colors[index % colors.length];
    };

    // Helper function to format values based on chart configuration
    const formatValue = (value, fieldName, chartInfo) => {
        // Get decimal places from chart configuration
        const decimalPlaces = chartInfo?.decimal_places;
        
        if (decimalPlaces === 0) {
            return Math.round(value).toString();
        } else if (decimalPlaces > 0) {
            return value.toFixed(decimalPlaces);
        } else {
            // Default to 1 decimal place if not specified
            return value.toFixed(1);
        }
    };

    // Helper function to calculate responsive chart dimensions
    const getChartDimensions = () => {
        const screenWidth = window.innerWidth;
        
        if (screenWidth <= 480) {
            return { width: 100, height: 75 }; // 4:3 ratio for mobile
        } else if (screenWidth <= 768) {
            return { width: 110, height: 82.5 }; // 4:3 ratio for tablet
        } else if (screenWidth <= 1200) {
            return { width: 115, height: 86.25 }; // 4:3 ratio for small desktop
        } else {
            return { width: 120, height: 90 }; // 4:3 ratio for large desktop
        }
    };

    const renderAnimatedBarChart = (chartInfo) => {
        const chartId = chartInfo.chart_title;
        const animationDataForChart = animationData[chartId] || chartInfo.datasets;
        const scale = chartScales[chartId] || 100;
        
        // Debug logging
        console.log(`Rendering chart: ${chartId}`, {
            animationDataForChart,
            scale,
            chartScales: chartScales
        });
        
        const margin = { top: 20, right: 20, bottom: 40, left: 40 };
        const { width: chartWidth, height: chartHeight } = getChartDimensions();
        const barPadding = 2;
        
        const colors = generateDistinctColors(animationDataForChart.length);
        
        return (
            <div key={chartId} className="chart-container">
                <div className="chart-title">{chartInfo.chart_title}</div>
                <div className="chart-wrapper">
                    <svg width="100%" height="100%" style={{ overflow: 'visible' }} viewBox={`0 0 ${chartWidth} ${chartHeight}`} preserveAspectRatio="xMidYMid meet">
                        {/* Test: Add a background rectangle to see the SVG area */}
                        <rect x="0" y="0" width={chartWidth} height={chartHeight} fill="rgba(255,0,0,0.1)" stroke="red" strokeWidth="0.5"/>
                        
                        {animationDataForChart.map((dataset, index) => {
                            // TEMPORARY: Use actual value directly instead of animatedValue for debugging
                            const value = dataset.value || 0;
                            
                            // For debugging, use a simple scale calculation
                            const maxValue = Math.max(...animationDataForChart.map(d => d.value || 0));
                            const simpleScale = maxValue > 0 ? maxValue * 1.2 : 100;
                            let barHeight = (value / simpleScale) * (chartHeight - margin.top - margin.bottom);
                            
                            // Ensure minimum bar height for visibility
                            if (barHeight < 10) {
                                barHeight = 10;
                            }
                            
                            const barX = margin.left + index * ((chartWidth - margin.left - margin.right - (animationDataForChart.length - 1) * barPadding) / animationDataForChart.length + barPadding);
                            const barY = margin.top + (chartHeight - margin.top - margin.bottom) - barHeight;
                            const barWidth = (chartWidth - margin.left - margin.right - (animationDataForChart.length - 1) * barPadding) / animationDataForChart.length;
                            
                            // Debug logging for each bar
                            console.log(`Bar ${index} for ${chartId}:`, {
                                value,
                                barHeight,
                                barX,
                                barY,
                                barWidth,
                                simpleScale,
                                maxValue,
                                dataset: dataset
                            });
                            
                            return (
                                <g key={index}>
                                    <rect
                                        x={barX}
                                        y={barY}
                                        width={barWidth}
                                        height={barHeight}
                                        fill={colors[index % colors.length]}
                                        rx="2"
                                        ry="2"
                                        style={{ 
                                            cursor: 'pointer',
                                            transition: 'opacity 0.2s ease'
                                        }}
                                        onMouseEnter={(e) => {
                                            e.target.style.opacity = '0.8';
                                        }}
                                        onMouseLeave={(e) => {
                                            e.target.style.opacity = '1';
                                        }}
                                    />
                                    <text
                                        x={barX + barWidth / 2}
                                        y={barY - 5}
                                        textAnchor="middle"
                                        fontSize="4"
                                        fill="#333"
                                        fontWeight="bold"
                                    >
                                        {formatValue(value, chartInfo.field_name, chartInfo)}
                                    </text>
                                    <text
                                        x={barX + barWidth / 2}
                                        y={margin.top + chartHeight + 5}
                                        textAnchor="middle"
                                        fontSize="3"
                                        fill="#666"
                                    >
                                        {dataset.label.split('_').slice(-1)[0]}
                                    </text>
                                </g>
                            );
                        })}
                    </svg>
                </div>
            </div>
        );
    };

    if (loading) {
        return (
            <div className="container">
                <div className="loading">
                    <h2>Loading dashboard data...</h2>
                    <p>Please wait while we fetch the comparison data.</p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="container">
                <div className="error">
                    <h3>Error loading data</h3>
                    <p>{error}</p>
                    <button onClick={fetchInitialData} style={{
                        marginTop: '10px',
                        padding: '10px 20px',
                        backgroundColor: '#667eea',
                        color: 'white',
                        border: 'none',
                        borderRadius: '5px',
                        cursor: 'pointer'
                    }}>
                        Retry
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="container">
            <div className="header">
                <h1>LLM Performance Comparison Dashboard</h1>
                <p>Comparing results across different LLM providers and models</p>
                
                {/* Real-time Monitoring Status */}
                <div className="monitoring-controls" style={{
                    marginTop: '20px',
                    padding: '15px'
                }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                                <div style={{
                                    width: '10px',
                                    height: '10px',
                                    borderRadius: '50%',
                                    backgroundColor: monitoringStatus?.monitoring_active ? '#28a745' : '#dc3545'
                                }}></div>
                                <span style={{ fontWeight: 'bold', color: 'white' }}>
                                    {monitoringStatus?.monitoring_active ? '🟢 Live Monitoring' : '🔴 Monitoring Off'}
                                </span>
                            </div>
                            
                            {monitoringStatus && (
                                <div style={{ fontSize: '14px', color: 'rgba(255,255,255,0.8)' }}>
                                    📁 {monitoringStatus.json_files_count} files | 
                                    ⏱️ {monitoringStatus.update_frequency_seconds}s updates |
                                    🕐 Last: {lastUpdate || 'Never'}
                                </div>
                            )}
                        </div>
                        
                        <div style={{ display: 'flex', gap: '10px' }}>
                            <button 
                                onClick={toggleAutoRefresh}
                                style={{
                                    padding: '8px 16px',
                                    backgroundColor: autoRefresh ? '#28a745' : '#6c757d',
                                    color: 'white',
                                    border: 'none',
                                    borderRadius: '5px',
                                    cursor: 'pointer',
                                    fontSize: '14px'
                                }}
                            >
                                {autoRefresh ? '🔄 Auto-Refresh ON' : '⏸️ Auto-Refresh OFF'}
                            </button>
                            
                            <button 
                                onClick={forceRefresh}
                                style={{
                                    padding: '8px 16px',
                                    backgroundColor: '#007bff',
                                    color: 'white',
                                    border: 'none',
                                    borderRadius: '5px',
                                    cursor: 'pointer',
                                    fontSize: '14px'
                                }}
                            >
                                🔄 Force Refresh
                            </button>
                        </div>
                    </div>
                    
                    {/* Status Message */}
                    {monitoringStatus && monitoringStatus.status_message && (
                        <div style={{ 
                            marginTop: '10px', 
                            padding: '8px 12px',
                            borderRadius: '5px',
                            fontSize: '14px',
                            fontWeight: '500',
                            backgroundColor: monitoringStatus.status_type === 'warning' ? 'rgba(255, 193, 7, 0.2)' : 
                                           monitoringStatus.status_type === 'info' ? 'rgba(13, 202, 240, 0.2)' : 
                                           'rgba(40, 167, 69, 0.2)',
                            border: `1px solid ${
                                monitoringStatus.status_type === 'warning' ? '#ffc107' : 
                                monitoringStatus.status_type === 'info' ? '#0dcaf0' : 
                                '#28a745'
                            }`,
                            color: monitoringStatus.status_type === 'warning' ? '#ffc107' : 
                                   monitoringStatus.status_type === 'info' ? '#0dcaf0' : 
                                   '#28a745'
                        }}>
                            {monitoringStatus.status_message}
                        </div>
                    )}
                    
                    {monitoringStatus && (
                        <div style={{ marginTop: '10px', fontSize: '12px', color: 'rgba(255,255,255,0.7)' }}>
                            📂 Directory: {monitoringStatus.json_directory} | 
                            {monitoringStatus.files_changed ? ' 🔄 Files changed' : ' ✅ No changes'} |
                            {monitoringStatus.has_cached_data ? ' 💾 Using cached data' : ' 📭 No cached data'}
                        </div>
                    )}
                </div>
            </div>

            {files.length > 0 && (
                <div className="stats-summary">
                    <div className="stats-title">Available Data Files</div>
                    <div className="stats-grid">
                        {files.map((file, index) => (
                            <div 
                                key={index} 
                                className="stat-item"
                                style={{
                                    backgroundColor: getColorForIndex(index, files.length),
                                    color: 'white',
                                    borderLeft: `2px solid ${getColorForIndex(index, files.length)}`,
                                    padding: '7px'
                                }}
                            >
                                <div className="stat-label" style={{color: 'white', fontSize: '0.64rem', marginBottom: '2px'}}>LLM Provider</div>
                                <div className="stat-value" style={{color: 'white', fontSize: '0.83rem'}}>{file.llm_provider}</div>
                                <div className="stat-label" style={{color: 'white', fontSize: '0.64rem', marginBottom: '2px'}}>Model</div>
                                <div className="stat-value" style={{color: 'white', fontSize: '0.83rem'}}>{file.llm_model}</div>
                                <div className="stat-label" style={{color: 'white', fontSize: '0.64rem', marginBottom: '2px'}}>Strategy</div>
                                <div className="stat-value" style={{color: 'white', fontSize: '0.83rem'}}>{file.strategy}</div>
                                <div className="stat-label" style={{color: 'white', fontSize: '0.64rem', marginBottom: '2px'}}>Mode</div>
                                <div className="stat-value" style={{color: 'white', fontSize: '0.83rem'}}>{file.mode}</div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            <div className="charts-grid">
                {chartData.map((chartInfo, index) => renderAnimatedBarChart(chartInfo))}
            </div>
        </div>
    );
}

// Render the dashboard
ReactDOM.render(<Dashboard />, document.getElementById('root')); 