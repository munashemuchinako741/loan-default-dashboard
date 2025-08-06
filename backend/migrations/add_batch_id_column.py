import sqlite3

def add_batch_id_column(db_path: str = "backend/loan_db.sqlite3"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if batch_id column already exists
    cursor.execute("PRAGMA table_info(prediction_history);")
    columns = [info[1] for info in cursor.fetchall()]
    if "batch_id" in columns:
        print("batch_id column already exists in prediction_history table.")
        conn.close()
        return

    # Add batch_id column
    try:
        cursor.execute("ALTER TABLE prediction_history ADD COLUMN batch_id TEXT NOT NULL DEFAULT '';")
        print("batch_id column added successfully.")
    except sqlite3.OperationalError as e:
        print(f"Error adding batch_id column: {e}")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    add_batch_id_column()
