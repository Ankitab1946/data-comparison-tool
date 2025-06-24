# """Configuration settings for the Data Comparison Tool."""

# # File processing settings
# LARGE_FILE_THRESHOLD = 3 * 1024 * 1024 * 1024  # 3GB
# CHUNK_SIZE = 10 ** 6  # 1 million rows per chunk
# TEMP_DIR = "temp"
# MAX_PREVIEW_ROWS = 1000

# # Database settings
# DEFAULT_TIMEOUT = 300  # seconds
# MAX_RETRIES = 3
# # RETRY_DELAY = 5  # seconds

# # Report settings
# REPORTS_DIR = "reports"
# REPORT_FORMATS = {
#     'excel': '.xlsx',
#     'csv': '.csv',
#     'html': '.html'
# }

# # Data type mappings
# TYPE_MAPPING = {
#     'int': 'int32',
#     'int64': 'int64',
#     'numeric': 'int64',
#     'bigint': 'int64',
#     'smallint': 'int64',
#     'varchar': 'string',
#     'nvarchar': 'string',
#     'char': 'string',
#     'date': 'datetime64[ns]',
#     'datetime': 'datetime64[ns]',
#     'decimal': 'float',
#     'float': 'float',
#     'bit': 'bool',
#     'nchar': 'char',
#     'boolean': 'bool'
# }

# # Supported data sources
# SUPPORTED_SOURCES = [
#     'CSV file',
#     'DAT file',
#     'SQL Server',
#     'Stored Procs',
#     'Teradata',
#     'API',
#     'Parquet file',
#     'Flat files inside zipped folder'
# ]

# # Default delimiters
# DEFAULT_DELIMITERS = {
#     'CSV file': ',',
#     'DAT file': '|',
#     'TXT file': '\t'
# }

# # Comparison settings
# NUMERIC_TOLERANCE = 0.0001  # For floating point comparisons
# DATE_FORMAT = '%Y-%m-%d'
# DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'

# # Logging settings
# LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# LOG_LEVEL = 'INFO'
# LOG_FILE = 'comparison_tool.log'

# # Profile report settings
# PROFILE_SETTINGS = {
#     'title': 'Data Comparison Profile',
#     'explorative': True,
#     'minimal': False
# }

# # Error thresholds
# MAX_ERROR_ROWS = 1000  # Maximum number of error rows to store
# ERROR_SAMPLE_SIZE = 100  # Number of error samples to show in reports

"""Configuration settings for the Data Comparison Tool."""

# Database settings
DEFAULT_TIMEOUT = 300  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# Data type mappings
TYPE_MAPPING = {
    'int': 'int32',
    'int64': 'int64',
    'numeric': 'int64',
    'bigint': 'int64',
    'smallint': 'int64',
    'varchar': 'string',
    'nvarchar': 'string',
    'char': 'string',
    'date': 'datetime64[ns]',
    'datetime': 'datetime64[ns]',
    'decimal': 'float',
    'float': 'float',
    'bit': 'bool',
    'nchar': 'char',
    'boolean': 'bool'
}

# Supported data sources
SUPPORTED_SOURCES = [
    'SQL Server',
    'Teradata'
]

# Logging settings
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'
LOG_FILE = 'comparison_tool.log'
