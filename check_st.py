from  stlist import stList
import requests as rq
from bs4 import BeautifulSoup as bs
import os
st_http = "http://e-service.cwb.gov.tw/HistoryDataQuery/QueryDataController.do?command=viewMain&_=1473065277608"

soup = bs(rq.get(st_http).text, "html.parser")

list_text = soup.find("script",type="text/javascript").text
head = list_text.find("var stList =") + 13
end = list_text[head:].find("}")
var = bs(list_text[head:head+end+1],"html.parser")
newList = eval(var.text)

for key in newList.keys():
    for i in range(len(newList[key])):
        newList[key][i] = newList[key][i].strip()

#print(newList)
if newList != stList:
    os.rename("./stlist.py", "./stlist_old.py")
    with open("./stlist.py", "w+") as fw:
        fw.write("stList=")
        fw.write(str(newList))
        



