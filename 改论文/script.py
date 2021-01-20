import csv
count = 0
s =dict()
with open('kaggle_result.csv', 'r') as f1:
    line = csv.reader(f1)
    for l in line:
        if len(l):
            if l[1]+l[2]+l[3] in s.keys():
                pass
            else:
                s[l[1]+l[2]+l[3]] = l[0]



with open('new_data.csv', 'r') as f2:
    line = csv.reader(f2)
    for l in line:
        if len(l):
            if l[0] == '1':
                l[0] = s[l[2]+l[3]+l[4]]
                with open('last_date.csv','a+') as f3:
                    w = csv.writer(f3)
                    w.writerow(l)
            else:
                with open('last_date.csv','a+') as f3:
                    w = csv.writer(f3)
                    w.writerow(l)
print(count)