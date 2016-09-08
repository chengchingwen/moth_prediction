import sqlite3 as sql
import pandas as pd
import datetime as d
import make_db as m

date = "%d-%d-%d"
datef = "%Y-%m-%d"
year_end = "%Y-12-31"
year_start = "%Y-01-01"
query = "select * from `%d` where Time between '%s' and '%s'"
def get_ft(year, month, day, place ,delta=10 ):
    today = d.datetime(year, month ,day-1)
    start_date = today - d.timedelta(delta-1)
    db = sql.connect(place)
    if today.year == start_date.year:
        table = pd.read_sql(query % (year, start_date.strftime(datef),today.strftime(datef)),db)
    else:
        table1 = pd.read_sql(query % (start_date.year, start_date.strftime(datef), start_date.strftime(year_end)), db)
        table2 = pd.read_sql(query % (today.year, today.strftime(year_start),today.strftime(datef)),db)
        table = pd.concat([table1, table2])
        table.index = range(delta)
    return table
