import subprocess
import sys
import uuid
import secrets
import bcrypt

DB_URL = "postgresql://postgres:postgres@localhost:5432/rag"
ENV_PATH = ".env"

def get_project_id():
    res = subprocess.run(
        ["docker", "compose", "exec", "langfuse-db", "psql", "-U", "postgres", "-d", "rag", "-t", "-A",
         "-c", "SELECT id FROM projects LIMIT 1"],
        capture_output=True, text=True, timeout=10
    )
    pid = res.stdout.strip()
    if not pid:
        print("No project found in Langfuse DB. Have you started the containers?")
        sys.exit(1)
    return pid

def key_exists(project_id):
    res = subprocess.run(
        ["docker", "compose", "exec", "langfuse-db", "psql", "-U", "postgres", "-d", "rag", "-t", "-A",
         "-c", f"SELECT count(*) FROM api_keys WHERE project_id='{project_id}'"],
        capture_output=True, text=True, timeout=10
    )
    return int(res.stdout.strip()) > 0

def create_key(project_id):
    public_key = "pk-" + secrets.token_hex(16)
    secret_key = "sk-" + secrets.token_hex(32)
    display_secret_key = secret_key[:12] + "..."
    hashed = bcrypt.hashpw(secret_key.encode(), bcrypt.gensalt()).decode()
    key_id = uuid.uuid4().hex

    sql = f"""INSERT INTO api_keys (id, public_key, hashed_secret_key, display_secret_key, note, project_id, created_at)
VALUES ('{key_id}', '{public_key}', '{hashed}', '{display_secret_key}', 'auto-generated', '{project_id}', NOW());"""
    subprocess.run(
        ["docker", "compose", "exec", "-T", "langfuse-db", "psql", "-U", "postgres", "-d", "rag", "-c", sql],
        capture_output=True, text=True, timeout=10, check=True
    )
    return public_key, secret_key

def update_env(public_key, secret_key):
    import re
    try:
        with open(ENV_PATH) as f:
            content = f.read()
    except FileNotFoundError:
        print(f"{ENV_PATH} not found")
        return

    content = re.sub(r'^LANGFUSE_PUBLIC_KEY=.*', f'LANGFUSE_PUBLIC_KEY={public_key}', content, flags=re.MULTILINE)
    content = re.sub(r'^LANGFUSE_SECRET_KEY=.*', f'LANGFUSE_SECRET_KEY={secret_key}', content, flags=re.MULTILINE)

    with open(ENV_PATH, 'w') as f:
        f.write(content)
    print(f"Updated {ENV_PATH}")

if __name__ == "__main__":
    project_id = get_project_id()
    if key_exists(project_id):
        print("API key already exists, skipping.")
        sys.exit(0)

    public_key, secret_key = create_key(project_id)
    update_env(public_key, secret_key)
    print(f"Created API key: {public_key}")
