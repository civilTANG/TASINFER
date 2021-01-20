import csv
import ast
import astor
import os
import numpy as np
import  pandas as pd
competition_dict ={}
implenmetion =set()
count = 0
str = "df.loc['c']"
r = ast.parse(str)
print(ast.dump(r))


with open('result.csv','r') as r:
      c = csv.reader(r)
      for j in c:
          if len(j) :
            alternative = 0
            set_candidate = set()
            set_program = set()
            set_recommand = list()
            set_right_candid =set()
            set_right = list()
            with open('last_date_3.csv','r',  encoding='utf-8-sig',) as f:
                line = csv.reader(f)
                for l in line:
                    if len(l) and l[0]in ['0','1','2','3']:
                        if l[1] == j[0]:
                            set_recommand.append(l)
                            set_program.add(l[2])
                            set_candidate.add(l[2]+l[3])
                            if l[0] == '0':
                               count =count+1
                               set_right_candid.add(l[2]+l[3])
                               set_right.append(l)
                            #print(l)
                            if l[0] == '0' or l[0] == '1':
                                alternative =  alternative+1

                        continue
                        if key in implenmetion:
                            pass
                        else:
                            count = count + 1
                            with open('data1.csv', 'a+') as f2:
                                w = csv.writer(f2)
                                w.writerow(l)
                            print(l)
                            implenmetion.add(key)
                            #print(key)
                        continue
                        if l[1] in competition_dict.keys():
                                print(l[2])
                                competition_dict[l[1]].add(l[2])
                                #print(l[1])
                        else:
                            #print(l[2])
                            a = set()
                            a.add(l[2])
                            competition_dict[l[1]] = a
            with open('result_page5.csv','a+') as w2:
                writer = csv.writer(w2)
                writer.writerow([j[0],len(set_program), len(set_candidate), len(set_right_candid), len(set_recommand), len(set_right),alternative])
print(count)
exit(0)
def craet_template(code1,code2):
    try:
        r1 = ast.parse(code1)
        r2 = ast.parse(code2)
        dict_map = {}
        count = 0
        for node in ast.walk(r2):
            if isinstance(node, ast.Name):
                if node.id == 'np' and node.id == 'pd':
                    continue
                if node.id not in dict_map.keys():
                    var = 'v' + str(count)
                    dict_map[node.id] = var
                    node.id = var
                    count = count+1
                else:
                    node.id = dict_map[node.id]
        for node in ast.walk(r1):
            if isinstance(node, ast.Name):
                if node.id in dict_map.keys():
                    node.id = dict_map[node.id]

        print(astor.to_source(r1), astor.to_source(r2))
    except Exception as e:
        print(e)
with open('validated_alternative_implementations.csv','r') as f:
     reader = csv.reader(f)
     for line in reader:
         # print(line[6],line[7])
         craet_template(line[6],line[7])

