import sqlite3
import time
import datetime 
import random 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates 
from matplotlib import style
style.use ('fivethirtyeight')

conn = sqlite3.connect('personinfo.db')
c = conn.cursor()

def create_table():
    c.execute ('CREATE TABLE IF NOT EXISTS personalinfo (unix REAL, datestamps TEXT,keyword TEXT, value REAL)')

def data_entry():
    c.execute("INSERT INTO personalinfo VALUES(145665,'2019-05-12','python',8)")
    conn.commit()
    c.close()
    conn.close()

def dynamic_data_entry():
    unix = time.time()
    date =str(datetime.datetime.fromtimestamp(unix).strftime('%d-%m-%Y  %H:%M:%S'))
    keyword = 'python'
    value = random.randrange(0,45)
    c.execute("INSERT INTO personalinfo (unix, datestamps,keyword,value) VALUES (?,?,?,?)",
    (unix,date,keyword,value))
    conn.commit()

def read_from_db():
    c.execute("SELECT * FROM personalinfo ")
    # date = c.fetchall()
    # print(date)
    for row in c.fetchall():
        print(row)

def graph_date():
    c.execute('SELECT unix, value FROM personalinfo ')
    dates = []
    values =[]
    for row in c.fetchall():
        # pritn(row[0])
        # print(datetime.datetime.fronttimestamp(row[0]))
        dates.append(datetime.datetime.fromtimestamp(row[0]))
        values.append(row[1])
    plt.plot_date(dates,values,'-')
    plt.show()

graph_date()
# create_table()
# data_entry()
# for i in range(10):
#     dynamic_data_entry()
#     time.sleep(1)
# print('your data set is created')
#read_from_db()

c.close()
conn.close()