import os
import csv
import ast
import astor
import numpy as np
import  pandas as pd

a = pd.Series(np.arange(16))
b = pd.DataFrame(np.arange(16).reshape(2, 8))
c = np.arange(16)
d = np.ones(16)
print(np.hstack((c, d)))
print(np.append(c, d, axis=0))

#print(d)
print(b.shape)
print(dir(a))
print(dir(b))
print(dir(c))
print(type(c))
#print(help(a.apply))
exit(0)
path = r'D:\experiment\project'
csv.field_size_limit(500 * 1024 * 1024)
c = set() # 执行的文件
e = set() # 执行有误的
a = 0
error_dict = {}

z = ['NotImplementedError', 'AttributeError', 'KeyError', 'ValueError', 'AssertionError', 'TypeError', 'NameError', 'MemoryError']
for root, dirs, files in os.walk(path, topdown=False):
    for name in files:
        if name.endswith('executed.csv'):
            with open(os.path.join(root, name), 'r') as f1:
                csv_read = csv.reader(f1)
                for line in csv_read:
                    if len(line):
                        c.add(line[0])

                        print(line)
                        exit(0)
        elif name.endswith('error_type.csv'):
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
                                    continue
                            if error in error_dict.keys():
                                error_dict[error] = error_dict[error]+1
                            else:
                                error_dict[error] = 1
                            with open(line[1], 'r') as f3:
                                text = f3.read()
                                print(text)
                                print(line[1], line[3])
                            root1 = ast.parse(text)
                            for node in ast.walk(root1):
                                if isinstance(node, ast.Call):
                                    if isinstance(node.func, ast.Attribute) and node.func.attr == 'writerow':
                                        if isinstance(node.func.value,ast.Name) and node.func.value.id == 'writer':
                                            stmt = astor.to_source(node)
                                            with open('12_24_error.csv', 'a+') as f4:
                                                writer = csv.writer(f4)
                                                writer.writerow([line[1], line[3], stmt, error])
                            continue
                            print('请输入0，1，2')
                            case = input()
                            case = int(case)
                            while case not in [0,1,2]:
                                    print('请输入0，1，2')
                                    case = input()
                                    case = int(case)
                            if case == 0:
                                with open('new_replace.csv', 'a+') as f4:
                                    writer = csv.writer(f4)
                                    writer.writerow([line[1], error, case])
                            elif case == 1:
                                with open('new_replace.csv','a+') as f4:
                                    writer = csv.writer(f4)
                                    writer.writerow([line[1], error, case])
                            elif case == 2:
                                print('请输入代码:')
                                origin = input()
                                print('请输入答案:')
                                answer = input()
                                print('请输入错误答案')
                                wrong = input()
                                with open('new_replace.csv', 'a+') as f4:
                                    writer = csv.writer(f4)
                                    writer.writerow([line[1], error, case, answer, origin, wrong])


                            e.add(line[1])

                            #print(code_path)
                            #print(line[1])
                            a = a + 1



    for name in dirs:
        pass
        # print(os.path.join(root, name))
print(len(c))
print(len(e))
for key, value in error_dict.items():
    pass
    print(key, value)

result = {}
for i in c:
    code_path = os.path.join(i.split('\\')[0], i.split('\\')[1], i.split('\\')[2])
    if code_path in result.keys():
        result[code_path]['yes'] = result[code_path]['yes']+1
    else:
        result[code_path]={'yes': 1, 'no': 0}
    #print(code_path)

for i in e:
    code_path = os.path.join(i.split('\\')[0], i.split('\\')[1], i.split('\\')[2])
    if code_path in result.keys():
        result[code_path]['no'] = result[code_path]['no']+1
    else:
        result[code_path] = {'yes': 0, 'no': 1}
    #print(code_path)

for key, value in result.items():
    pass
    #print(key, value)

# print(len(result))
#print(len(e))