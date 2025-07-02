
Built by https://www.blackbox.ai

---

# Data Comparison Tool

## Project Overview
The **Data Comparison Tool** is a web-based application built using Streamlit that enables users to compare data across multiple sources effortlessly. Whether you're dealing with CSV files, databases, or APIs, this tool facilitates detailed analysis and reporting, making data validation straightforward and efficient.

## Installation

To run this application locally, you will need to install the required dependencies and run the Streamlit server.

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/data-comparison-tool.git
   ```
2. Navigate to the project directory:
   ```bash
   cd data-comparison-tool
   ```
3. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```
4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
5. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Upon running the application, you will see tools to load sample or custom datasets.
2. Select your source and target data types (CSV, API, SQL Server, etc.).
3. Upload your datasets or use built-in sample data for comparison.
4. Configure the column mappings to specify how the columns from both datasets correspond.
5. Click on the "Compare" button to see the results, which will provide metrics on how the datasets match.
6. You can download detailed reports after the comparison.

## Features
- Support for various data sources including CSV files, APIs, SQL databases, and more.
- Automatic mapping of columns between datasets.
- Summary statistics and detailed reports on matching rows and columns.
- User-friendly interface for uploading data and configuring comparisons.
- Downloadable reports in various formats.

## Dependencies

The project requires the following Python packages, listed in `requirements.txt`:
- `pandas` - for data manipulation and analysis.
- `streamlit` - for building the web application.
- Additional dependencies related to database connections and data handling might be included based on your specific implementation.

## Project Structure
```
data-comparison-tool/
│
├── app.py                       # Main application file
├── requirements.txt             # Python dependencies
├── utils/                       # Directory for utility functions
│   ├── data_loader.py           # Module for data loading functions
│   ├── comparison_engine.py      # Module for handling comparison logic
│
├── reports/                     # Directory for report generation
│   ├── report_generator.py       # Module for generating reports
│
└── sample_data/                 # Sample datasets for testing
    ├── source.csv               # Sample source data file
    └── target.csv               # Sample target data file
```

Feel free to contribute to this project or create issues if you encounter problems. Your feedback is valuable for improving the application!

## I have created a script to host the Streamlit application for multiple users. Here's how to use it:

```
I've created run_app.sh that:

Automatically kills any existing process on port 8501
Starts Streamlit with network access using --server.address 0.0.0.0
Makes the application accessible from other machines
To host the application:

Run the script: ./run_app.sh
The application will be accessible at:
Local machine: http://localhost:8501
Other machines: http://YOUR_IP_ADDRESS:8501 (Replace YOUR_IP_ADDRESS with your machine's IP address)
Your team members can access the application by:

Opening their web browser
Entering http://YOUR_IP_ADDRESS:8501 in the address bar
Where YOUR_IP_ADDRESS is your machine's IP address on the network
Note: Make sure your firewall allows incoming connections on port 8501 for other team members to access the application.


```

## Installation & Running

### Windows Users:
1. Install Python 3.10 or later from [python.org](https://www.python.org/downloads/)
2. Double-click `run_comparison.bat`
3. The tool will open in your default web browser at http://localhost:8000

### Linux/Mac Users:
1. Install Python 3.10 or later
2. Open terminal in the project directory
3. Make the script executable:
   ```bash
   chmod +x run_comparison.sh
   ```
4. Run the script:
   ```bash
   ./run_comparison.sh
   ```

### SQL Server Authentication

For SQL Server connectivity, create a `.env` file in the project directory with your credentials:

```env
SQL_SERVER=your_server_name
SQL_DATABASE=your_database_name
SQL_USERNAME=your_username
SQL_PASSWORD=your_password
```

Or use Windows Authentication by setting:
```env
SQL_USE_WINDOWS_AUTH=true
```

## Features

- Compare data from CSV files or SQL Server databases
- Generate detailed comparison reports:
  - Summary worksheet with record counts and status
  - Diff_Report worksheet showing side-by-side differences
  - Highlighted differences (yellow for updates)
  - Status indicators (Left Only, Right Only, Both - Update)
- Export reports in Excel format
- Support for SSO and Windows Authentication

## Usage

1. Launch the application using the appropriate script for your OS
2. Choose your data source:
   - Upload CSV files directly
   - Use SQL Server connection (requires .env configuration)
3. Select join columns for comparison
4. Click "Generate Report" to create the comparison
5. Download the generated report ZIP file

## Report Types

The tool generates several types of reports:

