"""
Real-time monitoring configuration for My_Ult_Monitor
"""

# Real-time monitoring settings
REAL_TIME_MONITORING = True
UPDATE_FREQUENCY_SECONDS = 1  # How often to check for updates (default: 1 second)
FILE_WATCH_ENABLED = True  # Enable file system watching for immediate updates

# Performance settings
CACHE_ENABLED = True  # Enable caching for better performance
MAX_CACHE_SIZE = 100  # Maximum number of files to cache
CACHE_TTL_SECONDS = 300  # Cache time-to-live in seconds

# Logging settings
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FILE_CHANGES = True  # Log when files are detected as changed
LOG_PERFORMANCE = True  # Log performance metrics

# UI settings
SHOW_MONITORING_STATUS = True  # Show monitoring status in UI
SHOW_LAST_UPDATE_TIME = True  # Show last update time
SHOW_FILE_COUNT = True  # Show number of JSON files being monitored
SHOW_DIRECTORY_PATH = True  # Show the directory being monitored

# Advanced settings
ENABLE_WEBSOCKET = False  # Enable WebSocket for real-time updates (future feature)
ENABLE_FILE_WATCHER = True  # Enable file system watcher for immediate updates
RETRY_ON_ERROR = True  # Retry failed operations
MAX_RETRIES = 3  # Maximum number of retries
RETRY_DELAY_SECONDS = 5  # Delay between retries 