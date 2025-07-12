import sqlite3
import os

DB_FILE = "sample_company.db"

def create_sample_db():
    # Remove old database file if it exists
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Create departments table
    cursor.execute("""
    CREATE TABLE departments (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        location TEXT
    )
    """)

    # Create employees table
    cursor.execute("""
    CREATE TABLE employees (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        role TEXT,
        department_id INTEGER,
        salary INTEGER,
        hire_date TEXT,
        FOREIGN KEY (department_id) REFERENCES departments (id)
    )
    """)

    # Insert sample data into departments
    departments_data = [
        (1, "Engineering", "Building A"),
        (2, "Human Resources", "Building B"),
        (3, "Sales", "Building C"),
        (4, "Marketing", "Building A"),
    ]
    cursor.executemany("INSERT INTO departments VALUES (?,?,?)", departments_data)

    # Insert sample data into employees
    employees_data = [
        (1, "Alice Smith", "Software Engineer", 1, 90000, "2020-01-15"),
        (2, "Bob Johnson", "HR Manager", 2, 85000, "2019-03-01"),
        (3, "Charlie Brown", "Sales Representative", 3, 75000, "2021-06-10"),
        (4, "Diana Prince", "Marketing Specialist", 4, 70000, "2022-08-20"),
        (5, "Edward Norton", "Senior Software Engineer", 1, 120000, "2018-07-01"),
        (6, "Fiona Gallagher", "HR Assistant", 2, 50000, "2023-01-10"),
        (7, "George Harrison", "Sales Lead", 3, 95000, "2019-11-05"),
        (8, "Helen Troy", "Content Creator", 4, 65000, "2023-03-15"),
        (9, "Ian McKellen", "DevOps Engineer", 1, 110000, "2021-09-01"),
        (10, "Julia Roberts", "Recruiter", 2, 72000, "2022-05-01"),
    ]
    cursor.executemany("INSERT INTO employees VALUES (?,?,?,?,?,?)", employees_data)

    conn.commit()
    conn.close()
    print(f"Database '{DB_FILE}' created and populated successfully.")

if __name__ == "__main__":
    create_sample_db()
