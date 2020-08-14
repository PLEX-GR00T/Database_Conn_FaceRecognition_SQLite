import sqlite3

conn = sqlite3.connect('personinfo.db')
c = conn.cursor()

def create_table():
    c.execute ('CREATE TABLE IF NOT EXISTS personalinfo (unix REAL, datestamps TEXT,keyword TEXT, value REAL)')

def data_entry():
    c.execute("INSERT INTO personalinfo VALUES(145665,'2019-05-12','python',8)")
    conn.commit()
    c.close()
    conn.close()

create_table()
data_entry()