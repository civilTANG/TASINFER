import os
import csv
import ast
data_path = r'D:\crw_data'
dicts = {}
dict2s = {}
new_replace_dict = {}
finally_list = []
with open('finally_12_27.csv', 'r') as f7:
    reader = csv.reader(f7)
    for line in reader:
        if len(line):
            finally_list.append(line[0])
print(finally_list)

with open('new_replace.csv', 'r') as f4:
    reader = csv.reader(f4)
    for line in reader:
        if len(line):
            if line[0] in new_replace_dict.keys():
                new_replace_dict[line[0]] = line[2]
            else:
                new_replace_dict[line[0]] = line[2]

#print(new_replace_dict)
#exit(0)

# 统计每个链接生成正确的可替代API的个数
with open('optimization_fliter.csv', 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        if line[0] == 'url':
            continue
        if len(line):
            if line[0] in dicts.keys():
                dicts[line[0]][0] = dicts[line[0]][0]+1
            else:
                dicts[line[0]] = [1]

# 统计每个链接生成错误的可替代API的个数
acc = 0
with open('1_.csv', 'r') as f2:
    reader = csv.reader(f2)
    for line in reader:
        if len(line):
            # print(line[0])
            if line[0] in new_replace_dict.keys() and line[0] not in finally_list:
                if new_replace_dict[line[0]] == '2':
                    with open(line[0], 'r') as f5:
                        text = f5.read()
                        print(text)
                        print(line[0], line[1])
                    print('请输入0，1，2')
                    case = input()
                    case = int(case)
                    while case not in [0, 1, 2]:
                        print('请输入0，1，2')
                        case = input()
                        case = int(case)
                    with open('finally_12_27.csv', 'a+') as f6:
                        writer = csv.writer(f6)
                        writer.writerow([line[0], line[1], case])
                pass
                # print(new_replace_dict[line[0]])
            else:
                continue
                acc = acc + 1
                with open(line[0], 'r') as f5:
                    text = f5.read()
                    print(text)
                error = line[1].split(':')[0]
                error = error.replace('b"', '').replace('b\'', '')
                print(line[0], line[1])
                root1 = ast.parse(text)
                print('请输入0，1，2')
                case = input()
                case = int(case)
                while case not in [0, 1, 2]:
                    print('请输入0，1，2')
                    case = input()
                    case = int(case)

                with open('new_replace.csv', 'a+') as f6:
                    writer = csv.writer(f6)

                    writer.writerow([line[0], error, case])
            continue

            temp_path = os.path.join(line[0].split('code')[0][:-1], 'link.txt')
            #print(temp_path)
            with open(temp_path, 'r') as f3:
                url = f3.read()
            if url in dict2s.keys():
                dict2s[url][0] = dict2s[url][0] + 1
            else:
                dict2s[url] = [1]

print(acc)
exit(0)
print(dict2s)
print(len(dict2s))
print(dicts)
print(len(dicts))

exit(0)
for root, dirs, files in os.walk(data_path, topdown=False):
    for name in files:
        if name.endswith('link.txt'):
            with open(os.path.join(root, 'link.txt'), 'r') as f1:
                link = f1.read()
                if link in dicts.keys():
                        dicts[link].append(root)



