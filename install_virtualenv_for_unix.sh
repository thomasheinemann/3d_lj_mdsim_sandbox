#for many python versions, replace "python" below with its full path
mkdir data
/usr/bin/python3.6 -m venv .
source bin/activate 
/usr/bin/python3.6 -m pip install -r requirements.txt
deactivate

