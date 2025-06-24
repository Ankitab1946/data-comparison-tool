#!/bin/bash

# Kill any process using port 8501
echo "Checking for existing processes on port 8501..."
lsof -ti:8501 | xargs kill -9 2>/dev/null

# Run streamlit with network access
echo "Starting Streamlit application..."
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
