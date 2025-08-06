# Backend - Loan Default Prediction API

## How to Run the Backend

This backend is a FastAPI application for loan default prediction.

### Prerequisites

- Python 3.11 or higher installed
- Virtual environment setup (recommended)

### Steps to Run

1. Open a terminal and navigate to the root project directory:
   ```
   cd "c:/Users/Munashe Muchinako/OneDrive/Desktop/data science/kraddle proj/Loan Dashboard"
   ```

2. (Optional) Activate the virtual environment if available:
   - The backend directory contains a `.venv` folder for the virtual environment.
   - On Windows PowerShell, try:
     ```
     .\.venv\Scripts\Activate.ps1
     ```
     If this command fails, you can run the backend without activating the virtual environment, but it is recommended to set up and activate one for dependency management.

3. Run the FastAPI backend server using uvicorn from the root directory:
   ```
   uvicorn backend.app:app --reload
   ```

4. Access the API at:
   ```
   http://127.0.0.1:8000
   ```

5. Access the interactive API documentation at:
   ```
   http://127.0.0.1:8000/docs
   ```

### Notes

- The backend uses SQLite database located at `backend/loan_db.sqlite3`.
- If you want to run uvicorn from inside the backend directory, you may need to add an empty `__init__.py` file in the backend directory to make it a package.
- If you encounter module import errors, ensure you run uvicorn from the root directory as shown above.

### Stopping the Server

- Press `CTRL+C` in the terminal running uvicorn to stop the backend server.

---

## Backend API Endpoint Testing Documentation

The following backend API endpoints were tested to verify correct functionality. All tests were performed by sending HTTP GET requests and verifying successful responses (status code 200 OK) and expected data structures.

### Tested Endpoints

- `/kpi-metrics`
- `/feature-importance`
- `/average-income-by-default`
- `/default-rate-by-employment`
- `/loan-risk-data`

### Testing Summary

- Each endpoint returned a 200 OK status.
- The response content matched the expected JSON format.
- No errors or exceptions were encountered during testing.
- The backend server remained stable and responsive throughout testing.

This testing confirms that the primary backend API endpoints are functioning as intended.
