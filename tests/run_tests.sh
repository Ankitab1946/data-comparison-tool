#!/bin/bash

# Test Execution Script for Selenium+Python+Pytest+Allure Framework
# This script provides easy commands to run tests in different modes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to display usage
usage() {
    cat << EOF
Usage: ./run_tests.sh [OPTION]

Test Execution Options:
  all                Run all tests
  smoke              Run smoke tests only
  regression         Run regression tests
  critical           Run critical tests
  parallel           Run tests in parallel
  parallel-smoke     Run smoke tests in parallel
  headless           Run tests in headless mode
  repeat             Run tests 3 times
  database           Run database tests only
  
Report Options:
  allure             Generate and serve Allure report
  allure-generate    Generate Allure report (static)
  html               Open HTML report
  logs               View test logs
  
Utility Options:
  setup              Install dependencies
  clean              Clean reports and cache
  help               Display this help message

Examples:
  ./run_tests.sh smoke
  ./run_tests.sh parallel
  ./run_tests.sh allure

EOF
}

# Function to check if pytest is installed
check_dependencies() {
    if ! command -v pytest &> /dev/null; then
        print_error "pytest is not installed. Run './run_tests.sh setup' first."
        exit 1
    fi
}

# Function to setup environment
setup() {
    print_info "Setting up test environment..."
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    print_info "Activating virtual environment..."
    source venv/bin/activate
    
    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip
    
    # Install dependencies
    print_info "Installing dependencies..."
    pip install -r requirements.txt
    
    # Create .env file if it doesn't exist
    if [ ! -f ".env" ]; then
        print_info "Creating .env file from template..."
        cp .env.example .env
        print_warning "Please update .env file with your configuration"
    fi
    
    print_success "Setup completed successfully!"
}

# Function to clean reports and cache
clean() {
    print_info "Cleaning reports and cache..."
    
    rm -rf reports/html/*
    rm -rf reports/logs/*
    rm -rf reports/screenshots/*
    rm -rf reports/allure-results/*
    rm -rf reports/allure-report/*
    rm -rf .pytest_cache
    rm -rf __pycache__
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    
    print_success "Cleanup completed!"
}

# Function to run all tests
run_all() {
    print_info "Running all tests..."
    check_dependencies
    pytest tests/ -v
    print_success "Test execution completed!"
}

# Function to run smoke tests
run_smoke() {
    print_info "Running smoke tests..."
    check_dependencies
    pytest tests/ -m smoke -v
    print_success "Smoke tests completed!"
}

# Function to run regression tests
run_regression() {
    print_info "Running regression tests..."
    check_dependencies
    pytest tests/ -m regression -v
    print_success "Regression tests completed!"
}

# Function to run critical tests
run_critical() {
    print_info "Running critical tests..."
    check_dependencies
    pytest tests/ -m critical -v
    print_success "Critical tests completed!"
}

# Function to run tests in parallel
run_parallel() {
    print_info "Running tests in parallel..."
    check_dependencies
    pytest tests/ -n auto -v
    print_success "Parallel test execution completed!"
}

# Function to run smoke tests in parallel
run_parallel_smoke() {
    print_info "Running smoke tests in parallel..."
    check_dependencies
    pytest tests/ -m smoke -n 4 -v
    print_success "Parallel smoke tests completed!"
}

# Function to run tests in headless mode
run_headless() {
    print_info "Running tests in headless mode..."
    check_dependencies
    HEADLESS=true pytest tests/ -v
    print_success "Headless test execution completed!"
}

# Function to run tests multiple times
run_repeat() {
    print_info "Running tests 3 times..."
    check_dependencies
    pytest tests/ --count=3 -v
    print_success "Repeated test execution completed!"
}

# Function to run database tests
run_database() {
    print_info "Running database tests..."
    check_dependencies
    pytest tests/ -m database -v
    print_success "Database tests completed!"
}

# Function to generate and serve Allure report
generate_allure() {
    print_info "Generating Allure report..."
    
    if ! command -v allure &> /dev/null; then
        print_error "Allure is not installed. Please install Allure first."
        print_info "Visit: https://docs.qameta.io/allure/#_installing_a_commandline"
        exit 1
    fi
    
    if [ ! -d "reports/allure-results" ] || [ -z "$(ls -A reports/allure-results)" ]; then
        print_warning "No Allure results found. Running tests first..."
        pytest tests/ --alluredir=reports/allure-results
    fi
    
    print_info "Serving Allure report..."
    allure serve reports/allure-results
}

# Function to generate static Allure report
generate_allure_static() {
    print_info "Generating static Allure report..."
    
    if ! command -v allure &> /dev/null; then
        print_error "Allure is not installed. Please install Allure first."
        exit 1
    fi
    
    if [ ! -d "reports/allure-results" ] || [ -z "$(ls -A reports/allure-results)" ]; then
        print_warning "No Allure results found. Running tests first..."
        pytest tests/ --alluredir=reports/allure-results
    fi
    
    allure generate reports/allure-results -o reports/allure-report --clean
    print_success "Static Allure report generated at: reports/allure-report/index.html"
}

# Function to open HTML report
open_html() {
    print_info "Opening HTML report..."
    
    if [ ! -f "reports/html/report.html" ]; then
        print_warning "HTML report not found. Running tests first..."
        pytest tests/
    fi
    
    # Detect OS and open accordingly
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open reports/html/report.html
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        xdg-open reports/html/report.html
    else
        print_info "Please open: reports/html/report.html"
    fi
}

# Function to view logs
view_logs() {
    print_info "Viewing test logs..."
    
    if [ -f "reports/logs/test_execution.log" ]; then
        tail -f reports/logs/test_execution.log
    else
        print_warning "No logs found. Run tests first."
    fi
}

# Main script logic
case "${1:-help}" in
    all)
        run_all
        ;;
    smoke)
        run_smoke
        ;;
    regression)
        run_regression
        ;;
    critical)
        run_critical
        ;;
    parallel)
        run_parallel
        ;;
    parallel-smoke)
        run_parallel_smoke
        ;;
    headless)
        run_headless
        ;;
    repeat)
        run_repeat
        ;;
    database)
        run_database
        ;;
    allure)
        generate_allure
        ;;
    allure-generate)
        generate_allure_static
        ;;
    html)
        open_html
        ;;
    logs)
        view_logs
        ;;
    setup)
        setup
        ;;
    clean)
        clean
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        print_error "Unknown option: $1"
        usage
        exit 1
        ;;
esac
