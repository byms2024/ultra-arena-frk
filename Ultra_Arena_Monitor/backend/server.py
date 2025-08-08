from flask import Flask, render_template, jsonify, send_from_directory, request
import json
import os
import sys
import time
import threading
from pathlib import Path
from datetime import datetime
import logging

# Add the parent directory to the path to import chart_config
sys.path.append(str(Path(__file__).parent.parent))
from config.chart_config import chart_config_all, CHARTS_PER_ROW, JSON_DATA_DIR, REAL_TIME_MONITORING, UPDATE_FREQUENCY_SECONDS, FILE_WATCH_ENABLED

app = Flask(__name__, static_folder='../frontend/static')

# Global variables for caching and monitoring
json_cache = {}
last_modified_times = {}
cache_lock = threading.Lock()
monitoring_active = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_json_directory():
    """Get the JSON data directory path"""
    # JSON_DATA_DIR is relative to the repository root monitor package parent
    return (Path(__file__).parent.parent / JSON_DATA_DIR).resolve()

def get_file_modification_time(file_path):
    """Get the modification time of a file"""
    try:
        return os.path.getmtime(file_path)
    except OSError:
        return 0

def debug_file_times():
    """Debug function to show current file modification times"""
    json_dir = get_json_directory()
    if not json_dir.exists():
        logger.info("📁 JSON directory does not exist")
        return
    
    logger.info("🔍 Current file modification times:")
    with cache_lock:
        for json_file in json_dir.glob("*.json"):
            current_mtime = get_file_modification_time(json_file)
            last_mtime = last_modified_times.get(str(json_file), 0)
            logger.info(f"   {json_file.name}: current={current_mtime}, last={last_mtime}")

def has_files_changed():
    """Check if any JSON files have changed since last check"""
    json_dir = get_json_directory()
    if not json_dir.exists():
        return False
    
    files_changed = False
    with cache_lock:
        for json_file in json_dir.glob("*.json"):
            current_mtime = get_file_modification_time(json_file)
            last_mtime = last_modified_times.get(str(json_file), 0)
            
            if current_mtime > last_mtime:
                # Update the last modification time to prevent repeated false positives
                last_modified_times[str(json_file)] = current_mtime
                logger.info(f"📁 File changed: {json_file.name}")
                files_changed = True
    
    return files_changed

def load_json_files():
    """Load all JSON files from the configured directory with caching.
    Supports either a directory of JSONs or a single consolidated modular_results.json.
    """
    global json_cache, last_modified_times
    
    json_dir = get_json_directory()
    json_files = {}
    
    if not json_dir.exists():
        logger.warning(f"📁 JSON directory does not exist: {json_dir}")
        return json_files
    
    with cache_lock:
        # If a consolidated file exists, load only that
        consolidated = json_dir / "modular_results.json"
        files_to_read = [consolidated] if consolidated.exists() else list(json_dir.glob("*.json"))

        for json_file in files_to_read:
            try:
                # Check if file has changed
                current_mtime = get_file_modification_time(json_file)
                last_mtime = last_modified_times.get(str(json_file), 0)
                
                if current_mtime > last_mtime or str(json_file) not in json_cache:
                    # File has changed or not in cache, reload it
                    with open(json_file, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)
                    
                    json_files[json_file.stem] = file_data
                    json_cache[str(json_file)] = file_data
                    last_modified_times[str(json_file)] = current_mtime
                    
                    logger.info(f"📄 Loaded/Updated: {json_file.name}")
                else:
                    # Use cached data
                    json_files[json_file.stem] = json_cache[str(json_file)]
                    
            except Exception as e:
                logger.error(f"❌ Error loading {json_file}: {e}")
    
    return json_files

def extract_leaf_nodes(config_dict, parent_path=""):
    """Recursively extract all leaf nodes with their paths and chart titles"""
    leaf_nodes = []
    
    for key, value in config_dict.items():
        current_path = f"{parent_path}.{key}" if parent_path else key
        
        if isinstance(value, dict):
            if "chart_title" in value:
                # This is a leaf node with chart_title
                # Extract the actual data path by removing the chart_config wrapper
                data_path = current_path.replace("chart_config_1.comparing_fields.", "")
                leaf_node = {
                    "path": data_path,
                    "chart_title": value["chart_title"],
                    "field_name": key
                }
                
                # Add decimal configuration if present
                if "decimal_point" in value:
                    leaf_node["decimal_places"] = value["decimal_point"]
                
                leaf_nodes.append(leaf_node)
            else:
                # Continue traversing
                leaf_nodes.extend(extract_leaf_nodes(value, current_path))
    
    return leaf_nodes

def get_value_from_path(data_dict, path):
    """Get value from nested dictionary using dot notation path"""
    keys = path.split('.')
    current = data_dict
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    
    return current

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('../frontend/static', 'index.html')

@app.route('/dashboard.js')
def dashboard_js():
    """Serve the dashboard JavaScript file"""
    return send_from_directory('../frontend/static', 'dashboard.js')

@app.route('/api/chart-data')
def get_chart_data():
    """API endpoint to get chart data for all leaf nodes"""
    json_files = load_json_files()
    leaf_nodes = extract_leaf_nodes(chart_config_all)
    
    chart_data = []
    
    for leaf_node in leaf_nodes:
        chart_info = {
            "chart_title": leaf_node["chart_title"],
            "field_name": leaf_node["field_name"],
            "datasets": []
        }
        
        # Add decimal places configuration if present
        if "decimal_places" in leaf_node:
            chart_info["decimal_places"] = leaf_node["decimal_places"]
        
        for file_name, file_data in json_files.items():
            # Extract LLM info for the label
            llm_provider = file_data.get("run_settings", {}).get("llm_provider", "unknown")
            llm_model = file_data.get("run_settings", {}).get("llm_model", "unknown")
            label = f"{llm_provider} - {llm_model}"
            
            # Get the value for this specific field
            value = get_value_from_path(file_data, leaf_node["path"])
            
            chart_info["datasets"].append({
                "label": label,
                "value": value
            })
        
        chart_data.append(chart_info)
    
    return jsonify(chart_data)

@app.route('/api/files')
def get_files():
    """API endpoint to get list of available files"""
    json_files = load_json_files()
    file_info = []
    
    for file_name, file_data in json_files.items():
        llm_provider = file_data.get("run_settings", {}).get("llm_provider", "unknown")
        llm_model = file_data.get("run_settings", {}).get("llm_model", "unknown")
        strategy = file_data.get("run_settings", {}).get("strategy", "unknown")
        mode = file_data.get("run_settings", {}).get("mode", "unknown")
        
        file_info.append({
            "file_name": file_name,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "strategy": strategy,
            "mode": mode
        })
    
    return jsonify(file_info)

@app.route('/api/layout-config')
def get_layout_config():
    """API endpoint to get layout configuration"""
    return jsonify({
        "charts_per_row": CHARTS_PER_ROW
    })

@app.route('/api/monitoring-status')
def get_monitoring_status():
    """API endpoint to get real-time monitoring status"""
    json_dir = get_json_directory()
    directory_exists = json_dir.exists()
    json_files = list(json_dir.glob("*.json")) if directory_exists else []
    
    # Check if we have any cached data even if directory doesn't exist
    has_cached_data = len(json_cache) > 0
    
    # Determine status message
    if not directory_exists:
        status_message = "📁 Directory not found - waiting for files to appear"
        status_type = "warning"
    elif len(json_files) == 0:
        status_message = "📁 Directory empty - waiting for JSON files"
        status_type = "info"
    else:
        status_message = "✅ Monitoring active"
        status_type = "success"
    
    return jsonify({
        "monitoring_enabled": REAL_TIME_MONITORING,
        "update_frequency_seconds": UPDATE_FREQUENCY_SECONDS,
        "file_watch_enabled": FILE_WATCH_ENABLED,
        "monitoring_active": monitoring_active,
        "json_directory": str(json_dir),
        "directory_exists": directory_exists,
        "json_files_count": len(json_files),
        "has_cached_data": has_cached_data,
        "last_check": datetime.now().isoformat(),
        "files_changed": has_files_changed(),
        "status_message": status_message,
        "status_type": status_type
    })

@app.route('/api/monitoring-config', methods=['GET', 'POST'])
def monitoring_config():
    """API endpoint to get or update monitoring configuration"""
    if request.method == 'POST':
        # Update configuration (for future implementation)
        data = request.get_json()
        logger.info(f"📝 Monitoring config update requested: {data}")
        return jsonify({"status": "Configuration update not yet implemented"})
    
    return jsonify({
        "real_time_monitoring": REAL_TIME_MONITORING,
        "update_frequency_seconds": UPDATE_FREQUENCY_SECONDS,
        "file_watch_enabled": FILE_WATCH_ENABLED
    })

@app.route('/api/debug-file-times')
def debug_file_times_endpoint():
    """Debug endpoint to show file modification times"""
    debug_file_times()
    return jsonify({"status": "Debug info logged to console"})

@app.route('/api/force-refresh')
def force_refresh():
    """API endpoint to force refresh of all JSON data"""
    global json_cache, last_modified_times
    
    with cache_lock:
        json_cache.clear()
        last_modified_times.clear()
    
    logger.info("🔄 Forced refresh of JSON data cache")
    return jsonify({"status": "Cache cleared, data will be reloaded on next request"})

def start_file_monitoring():
    """Start background file monitoring thread"""
    global monitoring_active
    
    if not REAL_TIME_MONITORING or monitoring_active:
        return
    
    monitoring_active = True
    
    def monitor_files():
        logger.info("🔍 Starting file monitoring...")
        while monitoring_active:
            try:
                if has_files_changed():
                    logger.info("🔄 Files changed, cache will be updated on next request")
                # Don't log when no changes detected to reduce noise
                time.sleep(UPDATE_FREQUENCY_SECONDS)
            except Exception as e:
                logger.error(f"❌ Error in file monitoring: {e}")
                time.sleep(UPDATE_FREQUENCY_SECONDS)
    
    monitor_thread = threading.Thread(target=monitor_files, daemon=True)
    monitor_thread.start()
    logger.info(f"🚀 File monitoring started with {UPDATE_FREQUENCY_SECONDS}s frequency")

if __name__ == '__main__':
    start_file_monitoring() # Start monitoring when the server starts
    app.run(debug=True, host='0.0.0.0', port=8000) 