@echo off
chcp 65001 >nul
title Enviar Projeto para o GitHub

echo ==========================================================
echo ENVIANDO CRYPTO MONITOR PARA O GITHUB
echo ==========================================================
echo.

:: Verifica se o Git está acessível
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERRO] O comando git nao esta reconhecido pelo sistema.
    echo.
    echo Como voce instalou o Git agora, por favor:
    echo 1. Feche todas as janelas do Prompt de Comando e do VS Code.
    echo 2. Reinicie o computador para aplicar as configuracoes do Windows.
    echo 3. Execute o upload.bat novamente.
    echo.
    pause
    exit /b
)

echo [1/5] Inicializando repositorio Git local...
git init
echo.

echo [2/5] Adicionando arquivos do projeto...
git add .
echo.

echo [3/5] Criando o primeiro commit local...
git commit -m "feat: first commit of crypto volatility monitor"
echo.

echo [4/5] Definindo a branch principal como main...
git branch -M main
echo.

echo [5/5] Vinculando ao seu repositorio do GitHub...
git remote remove origin >nul 2>&1
git remote add origin https://github.com/danilo-g-rosa/crypto-volatility-monitor.git
echo URL vinculada: https://github.com/danilo-g-rosa/crypto-volatility-monitor.git
echo.

echo [6/5] Enviando arquivos para o GitHub...
echo O Windows abrira seu navegador para autenticacao.
echo.
git push -u origin main

if %errorlevel% equ 0 (
    echo.
    echo ==========================================================
    echo SUCESSO! Seu projeto foi publicado no GitHub!
    echo Acesse: https://github.com/danilo-g-rosa/crypto-volatility-monitor
    echo ==========================================================
) else (
    echo.
    echo [ERRO] Ocorreu uma falha no envio.
    echo Certifique-se de criar o repositorio no site do GitHub primeiro.
)
echo.
pause
