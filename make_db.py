import pandas as pd
from bs4 import BeautifulSoup as bs
import sqlite3 as sql
from stlist import stList
import os
import gevent
from gevent import monkey
import datetime

stl = list(stList.keys())
db_path = "./db/%s.db"
target_dir = [i for i in os.listdir("./data") if os.path.isdir("./data/"+i)]

def make_db(directory):
    for _html in sorted(os.listdir("./data/"+directory), key=lambda x: datetime.datetime.strptime(x[:6] if x[6]=="_" else x[:7], '%Y-%m')):
        year = _html[:4]
        date = _html[:6] if _html[6]=='_' else _html[:7]
        with open("./data/%s/%s" % (directory, _html), "r+") as html:
            sp = bs(html, "lxml")
        sp.tbody.name = "table"
        table = sp.find_all("table")[1]
        table.table.tr.replace_with("")
        Df = pd.read_html(str(table.table), flavor="bs4", header=0)[0]
        data = Df[["觀測時間(LST)ObsTime","測站氣壓(hPa)StnPres","氣溫(℃)Temperature","最高氣溫(℃)T Max","最低氣溫(℃)T Min","相對溼度(%)RH","風速(m/s)WS","降水量(mm)Precp"]]
        data["觀測時間(LST)ObsTime"] = data["觀測時間(LST)ObsTime"].apply(lambda x: date+"-"+str(x))
        db = sql.connect(db_path % directory)
        data.to_sql(year, db, if_exists="append", index=False)
        db.close()


    
monkey.patch_all()
jobs = [gevent.spawn(make_db, directory) for directory in target_dir]
gevent.joinall(jobs)






