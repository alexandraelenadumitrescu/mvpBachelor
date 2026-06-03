@echo off
cd /d %~dp0
echo.
echo  PhotoMatch Server
echo  -----------------
echo  URL: http://0.0.0.0:8000
echo  Docs: http://localhost:8000/docs
echo  Press Ctrl+C to stop.
echo.
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
pause
