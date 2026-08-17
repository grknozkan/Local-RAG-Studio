@echo off
title Ollama Local RAG Server
echo ===================================================
echo  Foundry Local RAG - Ollama Servisi Baslatiliyor...
echo ===================================================
set OLLAMA_ORIGINS=*
"C:\Users\Gürkan\AppData\Local\Programs\Ollama\ollama.exe" serve
pause
