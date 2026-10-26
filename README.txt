# Retail Customer Segmentation — Machine Learning Lab Project 2026

## Included
- `app.py` — interactive Streamlit website
- `requirements.txt` — Python dependencies

## Requirements
- Python 3.10 or newer recommended
- Internet access for installing packages the first time

## Run on Windows
1. Put `app.py`, `requirements.txt`, and `OnlineRetail.csv` in the same folder.
2. Open Command Prompt or the VS Code terminal in that folder.
3. Create a virtual environment (recommended):
   `py -m venv .venv`
4. Activate it:
   `.venv\Scripts\activate`
5. Install dependencies:
   `python -m pip install -r requirements.txt`
6. Start the website:
   `python -m streamlit run app.py`
7. The browser should open automatically. If it does not, open the Local URL shown in the terminal, usually `http://localhost:8501`.
8. Upload `OnlineRetail.csv` using the upload control.

## Run on macOS/Linux
1. Put `app.py`, `requirements.txt`, and `OnlineRetail.csv` in one folder.
2. In a terminal in that folder:
   `python3 -m venv .venv`
   `source .venv/bin/activate`
   `python -m pip install -r requirements.txt`
   `python -m streamlit run app.py`
3. Upload `OnlineRetail.csv` in the app.

## Dataset
The app expects columns such as `InvoiceNo`, `Quantity`, `InvoiceDate`, `UnitPrice`, and `CustomerID`. It also supports the usual Online Retail columns `StockCode`, `Description`, and `Country`.

## How the algorithm works
The app cleans transaction rows, calculates sales, aggregates customer-level Recency/Frequency/Monetary (RFM) features, log-transforms and standardises them, and applies K-Means. Choose K from the sidebar (2–8). Customer results can be downloaded as a CSV.

## Troubleshooting
- If `py` is not recognised, install Python from https://www.python.org/downloads/ and select “Add Python to PATH” during setup.
- If `streamlit` is not recognised, run `python -m pip install -r requirements.txt` and then `python -m streamlit run app.py`.
- Keep the CSV in the same folder for easy access, then upload it in the browser.
