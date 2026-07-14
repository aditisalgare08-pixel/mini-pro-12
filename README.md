# Motion Detection System - Fixed ZIP

This is the corrected version. It avoids the `matplotlib ft2font` startup error by making Matplotlib optional.

## Run in VS Code

```powershell
cd Motion_Detection_System_FIXED
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Optional packages

Only install this if your Python supports them correctly:

```powershell
pip install -r optional_requirements.txt
```

## If Python 3.14 gives issues

Use Python 3.12:

```powershell
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Features

- Python single-file app: `app.py`
- HTML, CSS, JavaScript inside `app.py`
- Flask web dashboard
- OpenCV motion detection
- NumPy frame processing
- SQLite database logs
- Optional Pandas analytics
- Optional Matplotlib chart route
- Optional Tkinter launcher:

```powershell
python app.py --gui
```
