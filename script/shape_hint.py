import os
import json
import nltk
information_path = '../np_information'
#nltk.download()
keyword = ['dimension', 'dimensions', 'shape',  'size', '1d', '2d', '2-d', '1-d', 'dimensional', 'one-dimensional',
           'dimensionality']


def is_contain_keyword(str):
    for word in keyword:
        if word in str:
            return True
    return False


def product_shape_hint_return(string):
    text = nltk.word_tokenize(string)
    pos_tag = nltk.pos_tag(text, tagset='universal')
    print(pos_tag)
    for i in range(len(pos_tag)):
        if pos_tag[i][0] in keyword:
            flag = i
    left_flag = flag
    right_flag = flag
    while left_flag > 0:
        left_flag = left_flag-1
        if pos_tag[left_flag][1] == '.':
            break
    while right_flag < len(pos_tag):
        right_flag = right_flag+1
        if pos_tag[right_flag][1] == '.':
            break
    sentence = ''
    for i in range(left_flag+1, right_flag):
        sentence = sentence + ' '+pos_tag[i][0]
    print(sentence)
    print('.......')


for api in os.listdir(information_path):
    if api.endswith('json'):
        api_path = os.path.join(information_path, api)
        print(api_path)
        with open(api_path, 'r') as f:
            api_information = f.read()
        f.close()
        api_dict = json.loads(api_information)
        template = """
        def {}({},return_ts = None):\n
            {}\n
            return None\n
        """
        for key, value in api_dict.items():
            # print(key)
            if isinstance(value, dict):
                for k, v in value.items():
                    template_parameter = ''
                    if isinstance(v, dict):
                        if isinstance(v['description'], str):
                            description_str = v['description'].lower()
                            if is_contain_keyword(description_str):
                                if key == 'return':
                                    print(key, description_str)
                                    product_shape_hint_return(description_str)
                                elif key == 'parameter':
                                    print(key, description_str)

                    else:
                        if k == 'description':
                            if isinstance(v, str):
                                description_str = v.lower()
                                if is_contain_keyword(description_str):
                                    print(key, description_str)
            else:
                if isinstance(value, dict):
                    if isinstance(value['description'], str):
                        description_str = value['description'].lower()
                        if is_contain_keyword(description_str):
                            print(key, description_str)
                else:
                    if isinstance(value, str):
                        description_str = value.lower()
                        if is_contain_keyword(description_str):
                            print(key, description_str)
