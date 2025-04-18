@echo off
echo ========================================
echo 🔧 Création de l'environnement Python...
echo ========================================
cd /d %USERPROFILE%\Downloads
python -m venv env_detection

echo.
echo ========================================
echo 🚀 Activation de l'environnement...
echo ========================================
call env_detection\Scripts\activate.bat

echo.
echo ========================================
echo 📦 Installation des packages...
echo ========================================
pip install --upgrade pip
pip install streamlit opencv-python numpy tensorflow scikit-learn

echo.
echo ✅ Tout est prêt ! Tu peux lancer Streamlit.
echo ✨ Tape : streamlit run ton_script.py
pause
