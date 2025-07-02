#!/bin/bash
echo "Starting Side by Side Comparison Tool..."
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.port 8000
