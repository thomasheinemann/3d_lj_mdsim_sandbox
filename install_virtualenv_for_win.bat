rem for many python versions, replace "python" below with its full path
mkdir data
python -m venv .
Scripts\activate.bat & python -m pip install -r requirements.txt & deactivate

