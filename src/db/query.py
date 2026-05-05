import sqlite3
import os
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("SQLITE_DB_PATH", "./rag.db")

def query_users():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    results = cursor.fetchall()
    conn.close()
    return results

def query_orders(order_by="id"):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM orders ORDER BY {order_by}")
    results = cursor.fetchall()
    conn.close()
    return results

def print_query_results(label, results, columns=None):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for row in results[:10]:  # Show first 10 rows
        print(row)
    if len(results) > 10:
        print(f"... and {len(results) - 10} more rows")
    print(f"Total: {len(results)} rows\n")

if __name__ == "__main__":
    import sys
    query_type = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    if query_type in ["users", "all"]:
        results = query_users()
        print_query_results("USERS (SELECT * FROM users)", results)
    
    if query_type in ["orders", "all"]:
        results = query_orders()
        print_query_results("ORDERS (SELECT * FROM orders ORDER BY id)", results)
