import csv
import requests
from bs4 import BeautifulSoup
import time
import re
import ast
import astor
with open('11.py') as t:
    s = t.read()

root = ast.parse(s)
for node in ast.iter_child_nodes(root):
    print(astor.to_source(node),node.lineno)
exit(0)
with open('urls.csv', 'r') as f:
    text = csv.reader(f)
    for line in text:
        try:
            time.sleep(2)
            response = requests.get(line[0])
            #print(line[0])
            Soup = BeautifulSoup(response.text, "lxml")

            text = Soup.find_all('meta')
            res = re.match('.*Using data from(.*)" name=.*', str(text[2]))
            name = res.group(1)
            with open('url_competition.csv', 'a+') as w:
                writer = csv.writer(w)
                writer.writerow([line[0], name])
        except Exception as e:
            print(e)
            print(line[0])