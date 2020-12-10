import requests
from bs4 import BeautifulSoup
import re
from lxml import etree
import json
import os
url ='https://numpy.org/doc/stable/reference/routines.html'
root = 'https://numpy.org/doc/stable/reference/'
response = requests.get(url)
Soup = BeautifulSoup(response.text, "lxml")
text = Soup.find_all('li', attrs={'class': "toctree-l1"})
for a in text:
    a = str(a)
    res = re.match('.*href="(.*)">.*', a)
    url = res.group(1)
    url = url.split('">')[0]
    new_url = root + url
    res_2 = requests.get(new_url)
    Soup_2 = BeautifulSoup(res_2.text, "lxml")
    text_2 = Soup_2.find_all('p')
    for i in text_2:
        try:
            if 'reference internal' in str(i) and 'title' in str(i):
                res = re.match('.*href="(.*)">.*', str(i))
                ur2 = res.group(1)
                ur2 = ur2.split('title')[0]
                ur2 = ur2.replace('"', ' ').strip()
                target_url = root + ur2
                json_data = {}
                print(target_url)
                try:
                    res_3 = requests.get(target_url)
                    Soup_3 = BeautifulSoup(res_3.text, "lxml")
                    text_3 = Soup_3.find_all('dl', attrs={'class': 'function'})
                    dict_information = dict()
                    dict_information['api_name'] = text_3[0].find_next('code', attrs={'class': "sig-name descname"}).string
                    dict_information['description'] = text_3[0].find_next('p').string
                    print(dict_information)
                    json_data["overall"] = dict_information
                    info = text_3[0].find_next('dl', attrs={'class': ["field-list simple", 'field-list']})


                    try:
                        p = info.find_all('dd', attrs={'class': 'field-odd'})[0].find_all('dt')
                        p_d = info.find_all('dd', attrs={'class': 'field-odd'})[0].find_all('dd')
                        ret = info.find_all('dd', attrs={'class': 'field-even'})[0].find_all('dt')
                        ret_d = info.find_all('dd', attrs={'class': 'field-even'})[0].find_all('dd')
                    except Exception as e:
                        ret = []
                        ret_d = []
                        p = []
                        p = p_d
                        print(e)
                    p_dict = dict()
                    p_d_list = []
                    ret_dict = dict()
                    ret_d_list = []
                    for dsenten in p_d:
                        temp = ''
                        flag = True
                        for char in str(dsenten):
                            if char == '<':
                                flag = False
                                continue
                            elif char == '>':
                                flag = True
                            elif flag:
                                temp = temp + char
                        p_d_list.append(temp.replace('\n', ' '))

                    for dsenten in ret_d:
                        temp = ''
                        flag = True
                        for char in str(dsenten):
                            if char == '<':
                                flag = False
                                continue
                            elif char == '>':
                                flag = True
                            elif flag:
                                temp = temp + char
                        ret_d_list.append(temp.replace('\n', ' '))

                    print(p)
                    for i in range(0, len(p)):
                        p_info = {'type': p[i].find_next('span', attrs={'class': "classifier"}).string, 'description': p_d_list[i]}
                        p_dict[p[i].find_next('strong').string] = p_info
                    print(p_dict)

                    for i in range(0, len(ret)):
                        ret_info = {'type':ret[i].find_next('span',attrs={'class':"classifier"}).string, 'description': ret_d_list[i]}
                        ret_dict[ret[i].find_next('strong').string] = ret_info
                    print(ret_dict)
                    json_data['parameter'] = p_dict
                    json_data['return'] = ret_dict

                    try:
                        html = etree.HTML(res_3.text)
                        x_name = 'numpy-' + dict_information['api_name']
                        x_name = x_name.replace('_', '-').replace('.', '-')
                        path = '//*[@id="{}"]/dl/dd/p[3]'.format(x_name)
                        note = html.xpath(path)
                        temp = ''
                        n_flag = True
                        for char in str(etree.tostring(note[0])):
                            if char == '<':
                                n_flag = False
                                continue
                            elif char == '>':
                                n_flag = True
                            elif n_flag:
                                temp = temp + char

                        note_string = temp.replace('\\n', ' ').replace("b'",'')
                        json_data['note'] = note_string
                    except Exception as e:
                        print(dict_information)
                        print(e)
                    json_data = json.dumps(json_data, indent=4)
                    save_path = os.path.join('..', 'np_information', x_name+'.json')
                    with open(save_path, 'w+') as f:
                        f.write(json_data)
                    f.close()
                except Exception as e:
                    with open('api.txt', 'a+') as f:
                        f.write(target_url)
                        f.write('\n')
                #exit(0)
        except Exception as e:
            with open('api.txt', 'a+') as f:
                f.write(str(i))
                f.write('\n')


