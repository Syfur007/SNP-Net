@echo off
REM Activate environment and run test
cd /d "C:\Users\Shadman Sakib\Downloads\SNP-Net"
call env\Scripts\activate.bat
python test_mayocardial_autoencoder.py
pause
