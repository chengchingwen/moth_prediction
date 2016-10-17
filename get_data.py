#from stlist import stList
import requests as rq
from bs4 import BeautifulSoup as bs
import gevent
from gevent import monkey
import time as t
import random
import os
import json
http = "http://e-service.cwb.gov.tw/HistoryDataQuery/MonthDataController.do?command=viewMain&station=%s&stname=%s&datepicker=%d-%d"

proxies = {
    'https': 'http://220.141.162.136:8080',
}

#z = 0
json_data = open("./stlist.py","r+").read()
stl = json.loads(json_data)#list(stList.keys())
def get_data(ID):
    z = stl.index(ID)
    StN_C = stList[ID][0]
    StN = stList[ID][1]
    County = stList[ID][2]
    directory = "./data/%d_%s_%s_%s/" % (z,ID,StN_C, County)
    if not os.path.exists(directory):
        os.makedirs(directory)
    for year in [2016]:#[2013, 2014, 2015, 2016]:
        for month in range(1,13):
            if not (year == 2016 and month in range(11,13)):
                f = directory+"%d-%d_%s_%s.html" % (year, month, ID, StN)
                if not os.path.exists(f):
                    t.sleep(3)
                    html = rq.get(http % (ID, StN, year, month), proxies=proxies)
                    t.sleep(random.randint(0,5))
                    with open(f , "w+") as fw:
                        fw.write(html.text)
                    #end write 
                #end file    
    #end
    #z+=1

if __name__ == '__main__':   
    print(stl)
    #monkey.patch_all()
    #jobs = [gevent.spawn(get_data, ID) for ID in stList.keys()]
    #gevent.joinall(jobs)






