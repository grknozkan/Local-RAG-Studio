@echo off
title Ollama Local RAG Server
echo ===================================================
echo  Foundry Local RAG - Ollama Servisi Baslatiliyor...
echo ===================================================
set OLLAMA_ORIGINS=*
if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
    "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
) else (
    ollama serve
)
pause
