import sqlite3
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

DB_PATH = os.getenv("SQLITE_DB_PATH", "./rag.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # DDL
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        product TEXT NOT NULL,
        amount REAL NOT NULL,
        order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status TEXT DEFAULT 'pending',
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)

    # Clear existing data
    cursor.execute("DELETE FROM orders")
    cursor.execute("DELETE FROM users")

    # Insert 100 users
    users = [(f"User {i}", f"user{i}@example.com") for i in range(1, 101)]
    cursor.executemany("INSERT INTO users (name, email) VALUES (?, ?)", users)
    print(f"Inserted {len(users)} users")

    # Insert 5 orders per user (total 500 orders)
    orders = []
    products = ["Laptop", "Smartphone", "Tablet", "Headphones", "Smartwatch", "Charger", "Case", "Screen Protector"]
    statuses = ["pending", "shipped", "delivered", "cancelled"]
    for user_id in range(1, 101):
        for order_num in range(5):
            product = products[(user_id + order_num) % len(products)]
            amount = 99.99 + (user_id * 10) + order_num
            status = statuses[(user_id + order_num) % len(statuses)]
            orders.append((user_id, product, amount, status))
    
    cursor.executemany("INSERT INTO orders (user_id, product, amount, status) VALUES (?, ?, ?, ?)", orders)
    print(f"Inserted {len(orders)} orders")

    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")

if __name__ == "__main__":
    init_db()
