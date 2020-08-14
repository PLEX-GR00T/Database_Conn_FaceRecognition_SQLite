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

def getinfo():
    #print("this is fun where Name value :" + Name)
    #conn = sqlite3.connect('personinfo.db')
    #cmd = "SELECT * FROM People WHERE name ="+str(Name)
    cmd = c.execute('SELECT ID FROM People WHERE collage = "Bcom" ')
    #c.execute('SELECT * FROM People WHERE name ="' + str(Name) + '";')
    #cursor = conn.execute(cmd)
    #c.execute("SELECT * FROM People WHERE name ="+str(Name))
    print("this is lol "+str(cmd))
    profile = None
    #for row in c.fetchall():
    for row in cmd:
        profile = row
        print(row)
        print(profile[0])
    #conn.close()
    #print("this is fun where profile value :" + profile[0])
    return profile

#profile = getinfo()

# if profile != None:
#     print("it has somethig")
# else:
#     print("empty : " + str(profile))

def practice():
    unix = time.time()
    date =str(datetime.datetime.fromtimestamp(unix).strftime('%d-%m-%Y  %H:%M:%S'))

def doCount(Name):
    c.execute('SELECT count(*) FROM Attendance WHERE IDs ="' + str(Name) + '";')
    count = None
    for row in c.fetchall():
        count = row
        print(count[0])
    return count[0]

# def isDataExists(onlydate):
#     c.execute('SELECT IDs FROM Attendance WHERE Date ="' + str(onlydate) + '";')
#     atID = None
#     for row in c.fetchall():
#         atID = row
#     return atID

def doCount(IDs):
    c.execute('SELECT count(*) FROM Attendance WHERE IDs ="' + str(IDs) + '";')
    count = None
    for row in c.fetchall():
        count = row
        print(count[0])
    return count[0]

# atID = isDataExists('15-06-2019')
# lol = atID.split()
# if(4 == lol ): 
#     print(lol)
#     print('in which ID = '+ ID)
# else:
#     print('No ID in LOL')

def isDataExists(onlydate):
    c.execute('SELECT P.ID FROM People AS P, Attendance AS A WHERE P.ID = A.IDs AND A.Date ="' + str(onlydate) + '";')
    atID = None
    for row in c.fetchall():
        atID = row
        for i in range(len(atID)):
            if(3 == i ):
                print('Maro id = '+ID) 
                print(i)
                print('Madigayooooooo ID = '+ ID)
            else:
                print('No ID in LOL')
        print(atID)
    return atID

# for i in range(len(atID)):
#     if(3 == i ):
#         print('Maro id = '+ID) 
#         print(i)
#         print('Madigayooooooo ID = '+ ID)
#     else:
#         print('No ID in LOL')

atID = isDataExists('15-06-2019')
print(atID)