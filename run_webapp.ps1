$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
.\.venv\Scripts\python.exe -m flask --app webapp.app init-db
.\.venv\Scripts\python.exe -m waitress --host=127.0.0.1 --port=5000 webapp.app:app
