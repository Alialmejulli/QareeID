import sqlite3 
from pathlib import Path 
DB_PATH = Path('data/quranid.db') 
print('DB exists:', DB_PATH.exists()) 
conn = sqlite3.connect(str(DB_PATH)) 
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() 
print('Tables:', tables) 
conn.close() 
