@echo off
setlocal
python -m src.web.server --host 0.0.0.0 --port 8000
endlocal
