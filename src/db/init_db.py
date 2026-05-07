import os
from dotenv import load_dotenv
from datetime import datetime
import psycopg2
from psycopg2.extras import DictCursor

load_dotenv()

DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag")

def init_db():
    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()
    
    # DDL - Drop tables first to avoid sequence issues
    cursor.execute("DROP TABLE IF EXISTS orders CASCADE")
    cursor.execute("DROP TABLE IF EXISTS users CASCADE")
    
    cursor.execute("""
    CREATE TABLE users (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        email VARCHAR(255) UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    cursor.execute("""
    CREATE TABLE orders (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL,
        product VARCHAR(255) NOT NULL,
        amount DECIMAL(10,2) NOT NULL,
        order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status VARCHAR(50) DEFAULT 'pending',
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)
    
    conn.commit()
    
    # Insert 100 users
    users = [(f"User {i}", f"user{i}@example.com") for i in range(1, 101)]
    cursor.executemany("INSERT INTO users (name, email) VALUES (%s, %s)", users)
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
    
    cursor.executemany("INSERT INTO orders (user_id, product, amount, status) VALUES (%s, %s, %s, %s)", orders)
    print(f"Inserted {len(orders)} orders")
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_URL}")

if __name__ == "__main__":
    init_db()
