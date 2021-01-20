import os
import csv
import ast
import astor
import numpy as np
import  pandas as pd

path = r'D:\experiment\project'
csv.field_size_limit(500 * 1024 * 1024)
c = set() # 执行的文件
e = set() # 执行有误的
a = 0
error_dict = {}

z = ['AttributeError', 'KeyError', 'ValueError', 'AssertionError', 'TypeError', 'NameError']
for root, dirs, files in os.walk(path, topdown=False):
    for name in files:
        if name.endswith('executed.csv'):
            with open(os.path.join(root, name), 'r') as f1:
                csv_read = csv.reader(f1)
                for line in csv_read:
                    if len(line):
                        c.add(line[0])

        elif name.endswith('error_type.csv'):
            continue
            with open(os.path.join(root, name), 'r') as f2:
                csv_read = csv.reader(f2)
                for line in csv_read:
                    if len(line):

                        error = line[3].split(':')[0]
                        error = error.replace('b"', '').replace('b\'', '')
                        if len(error) and (error in z):
                            if 'r2 =' in line[2]:
                                pass
                            else:
                                if error == 'AssertionError':
                                    pass
                                else:
                                    with open(line[1], 'r') as f3:
                                        text = f3.read()
                                        print(text)
                                    print(line[1],line[2], line[3])
                                    exit(0)

                            with open(line[1], 'r') as f3:
                                text = f3.read()
                                print(text)
                                print(line[1], line[3])
                            a = a + 1
                            continue
                            root1 = ast.parse(text)
                            for node in ast.walk(root1):
                                if isinstance(node, ast.Call):
                                    if isinstance(node.func, ast.Attribute) and node.func.attr == 'writerow':
                                        if isinstance(node.func.value, ast.Name) and node.func.value.id == 'writer':
                                            stmt = astor.to_source(node)
                                            with open('1_15_error.csv', 'a+') as f4:
                                                writer = csv.writer(f4)
                                                writer.writerow([line[1], line[3], stmt, error])



print(len(c))

zi = set()
with open('op.csv', 'r',encoding='UTF-8-sig') as f3:
    csv_read = csv.reader(f3)
    for line in csv_read:
        if len(line):
            try:
                with open(os.path.join(line[0], 'link.txt'), 'r')as r1:
                    url = r1.read()

                zi.add(url)
            except Exception as e:
                print(e)
count =0
for item in c:
    j = item.split('\code')[0]


    with open(os.path.join(j, 'link.txt'), 'r')as r2:
        k = r2.read()
    with open(os.path.join(j, 'competition.txt'), 'r')as r7:
        competition = r7.read()
    #print(k)
    if k in zi:
        #print(item)
        with open(item,'r') as r3:
            text = r3.read()
        root1 = ast.parse(text)
        for node in ast.walk(root1):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'writerow':
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == 'writer':
                        stmt = astor.to_source(node)
                        r = ast.parse(stmt)
                        #print(stmt)
                        for node in ast.walk(r):
                            if isinstance(node, ast.Call) and node.func.attr == 'writerow':
                                s = astor.to_source(node.args[0])
                                break
                            # print(s)
                        url = ''
                        t1 = ''
                        t2 = ''
                        try:
                            t = eval(s)

                            with open('1_16_15.10.csv', 'a+') as f3:
                                writer = csv.writer(f3)
                                writer.writerow([ competition,j,t[1], t[2], t[3], t[6], t[7]])

                        except Exception as e:
                            print(e)
        count =count+1

print(count)
print(zi)