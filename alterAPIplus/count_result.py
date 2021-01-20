import csv
import os

dict1 = {}

with open('optimizetion.csv', 'r', encoding='utf-8') as f2:
    reader = csv.reader(f2)
    for line in reader:
        if len(line):
            if line[0]in dict1.keys():
                dict1[line[0]] = dict1[line[0]]+1
            else:
                dict1[line[0]] = 1

print(len(dict1))

with open('num.csv', 'r', encoding='utf-8') as f3:
    reader = csv.reader(f3)
    for line in reader:
        if len(line):
            with open('tex.csv','a+') as f4:
                writer = csv.writer(f4)

                if line[0]in dict1.keys():
                    writer.writerow([line[0], line[1], line[2], dict1[line[0]]])
                    print(line[0])
                else:
                    writer.writerow([line[0], line[1], line[2], 0])
