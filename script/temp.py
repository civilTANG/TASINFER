import csv
import  os
import ast
import astor
path = r'D:\crw_data'
with open('error_1_15_21.csv') as f2:
    csv_read = csv.reader(f2)
    for line in csv_read:
        if len(line):
            with open('jiayou.csv', 'a+') as f3:
                writer = csv.writer(f3)
                writer.writerow([line[0], line[1].split('\code')[0], line[3], line[4], line[5], line[6]])


exit(0)
with open('kaggle_result.csv') as f2:
    csv_read = csv.reader(f2)
    for line in csv_read:
        if len(line):
            flag = 0
            for root, dirs, files in os.walk(path, topdown=False):
                for name in files:
                     if name.endswith('link.txt'):
                         with open(os.path.join(root,'link.txt'), 'r') as r1:
                             link = r1.read()
                         if line[0] == link:
                             flag =1
                             with open(os.path.join(root, 'competition.txt'), 'r') as r2:
                                 competition_name = r2.read()
                             with open('1_15_opm.csv', 'a+') as f3:
                                 writer = csv.writer(f3)
                                 writer.writerow([competition_name, root, line[1], line[2], line[5], line[6]])
                             break
                if flag ==1:
                    break












# 处理数据
with open('1_15_error.csv', 'r') as f2:
    csv_read = csv.reader(f2)
    for line in csv_read:
        if len(line):
            c_path = os.path.join(line[0].split('\code')[0],'competition.txt')
            with open(c_path, 'r') as f:
                competition_name = f.read()
            print(competition_name)
            temp_str = line[2].replace('\n',' ')
            r = ast.parse(temp_str)
            for node in ast.walk(r):
                if isinstance(node, ast.Call) and node.func.attr == 'writerow':
                    s = astor.to_source(node.args[0])
                    break
            #print(s)
            url = ''
            t1 = ''
            t2 = ''
            try:
                t = eval(s)
                with open('1_15_13.01.csv', 'a+') as f3:
                    writer = csv.writer(f3)
                    writer.writerow([competition_name, line[0],  t[1], t[2], t[3], t[6], t[7]])

            except Exception as e:
                print(e)