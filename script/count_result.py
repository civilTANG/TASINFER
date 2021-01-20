import os
import csv
import ast
import ast
a = 0
pass_path = []
with open('finally_12_27.csv', 'r') as f1:
    csv_read1 = csv.reader(f1)
    for line in csv_read1:
        if len(line):
            if line[-1] == '1':
                pass_path.append(line[0])

_1 = 1

with open('kaggle_exec.csv', 'r') as f4:
    csv_read1 = csv.reader(f4)
    for line in csv_read1:
        if len(line):
                pass_path.append(line[2])
b = 0
with open('temp_12_29.csv', 'r') as f5:
    csv_read1 = csv.reader(f5)
    for line in csv_read1:
        if len(line):
            if line[-1] == '2':
                pass_path.append(line[2])
            elif line[-1] == '1':
                with open('no_type_candidate.csv', 'r') as f2:
                    csv_read2 = csv.reader(f2)
                    for line1 in csv_read2:
                        if len(line1):
                            if line1[2] == line[2]:
                                a = a + 1
                                # print(line)
                                with open('type_candidate.csv', 'a+') as f3:
                                    writer = csv.writer(f3)
                                    writer.writerow([line1[0], line1[1], line1[2], line1[3], line1[4], line1[5], line1[6], line1[7]])


#print(len(pass_path))


print(a)
print(b)



