@echo off
echo Starting Side by Side Comparison Tool...
python -m venv venv
call venv\Scripts\activate.bat
pip install -r requirements.txt
streamlit run app.py --server.port 8000
pause
