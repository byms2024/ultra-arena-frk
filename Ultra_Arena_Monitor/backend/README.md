# LLM Performance Dashboard - Backend

A Flask-based backend server for the LLM Performance Comparison Dashboard.

## Features

- **Flask API**: RESTful API endpoints for data access
- **JSON Processing**: Loads and processes LLM performance data
- **Chart Configuration**: Dynamic chart generation based on configuration
- **Data Extraction**: Intelligent path-based data extraction from nested JSON
- **Static File Serving**: Serves frontend static files

## File Structure

```
backend/
├── server.py              # Main Flask application
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Technologies Used

- **Flask**: Python web framework
- **JSON**: Data format for LLM performance metrics
- **Pathlib**: File system operations
- **Sys**: Module path management

## Getting Started

### Prerequisites

- Python 3.7+
- pip (Python package manager)

### Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the server:**
   ```bash
   python server.py
   ```

3. **Access the application:**
   - Open your browser and go to `http://localhost:8000`
   - The server will serve both the API and frontend

## API Endpoints

### `GET /`
- **Description**: Serves the main dashboard HTML page
- **Response**: HTML page with React frontend

### `GET /api/chart-data`
- **Description**: Returns chart data for all configured metrics
- **Response**: JSON array of chart configurations with data
- **Example Response**:
  ```json
  [
    {
      "chart_title": "Percent of Files Retried",
      "field_name": "percentage_files_had_retry",
      "datasets": [
        {
          "label": "google - gemini-2.5-flash",
          "value": 0.0
        },
        {
          "label": "deepseek - deepseek-chat",
          "value": 0.0
        }
      ]
    }
  ]
  ```

### `GET /api/files`
- **Description**: Returns information about available data files
- **Response**: JSON array of file information
- **Example Response**:
  ```json
  [
    {
      "file_name": "direct_file_parallel_google_gemini_2.5_flash_08-01-15-42-26",
      "llm_provider": "google",
      "llm_model": "gemini-2.5-flash",
      "strategy": "direct_file",
      "mode": "parallel"
    }
  ]
  ```

### `GET /api/layout-config`
- **Description**: Returns layout configuration for the dashboard
- **Response**: JSON object with layout settings
- **Example Response**:
  ```json
  {
    "charts_per_row": 3
  }
  ```

## Data Sources

The backend automatically loads JSON files from the directory specified in the configuration:
```
../{JSON_DATA_DIR}/*.json
```

Default location: `../inputs/json/combo1/*.json`

Each JSON file should contain LLM performance metrics with the following structure:
- `run_settings`: LLM configuration information
- `retry_stats`: Retry-related metrics
- `overall_stats`: Overall performance statistics
- `benchmark_errors`: Error metrics

## Chart Configuration

Charts are dynamically generated based on the configuration in:
```
../config/chart_config.py
```

The configuration defines:
- Which metrics to display
- Chart titles
- Data field paths

## Development

### Adding New Metrics

1. **Update chart configuration** in `../config/chart_config.py`
2. **Ensure JSON files** contain the new metric data
3. **Restart the server** to see changes

### Changing Data Source

1. **Update `JSON_DATA_DIR`** in `../config/chart_config.py`
2. **Ensure the new directory** contains JSON files with the expected structure
3. **Restart the server** to load from the new location

### Adding New API Endpoints

1. **Add new route** in `server.py`
2. **Implement data processing** logic
3. **Return JSON response**

### Error Handling

The server includes error handling for:
- Missing JSON files
- Invalid data paths
- Server errors

## Configuration

### Port Configuration
- Default port: 8000
- Change in `server.py` line: `app.run(debug=True, host='0.0.0.0', port=8000)`

### Static Files
- Frontend files served from: `../frontend/static/`
- Configured in: `app = Flask(__name__, static_folder='../frontend/static')`

## Troubleshooting

### Common Issues

1. **Port already in use**: Change port in `server.py`
2. **Missing JSON files**: Ensure files exist in `../inputs/json/combo1/`
3. **Import errors**: Check Python path and dependencies

### Debug Mode

The server runs in debug mode by default:
- Auto-reloads on file changes
- Detailed error messages
- Development server warnings

## Production Deployment

For production deployment:
1. Disable debug mode
2. Use a production WSGI server (Gunicorn, uWSGI)
3. Configure proper static file serving
4. Set up environment variables for configuration 