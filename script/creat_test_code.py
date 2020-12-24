import ast
import os
import csv
import random
import astor
import shutil
kaggle_path = os.path.join('..', 'kaggle_code')
template = """ 
if '{}'not in TANGSHAN:   \n
    import csv\n
    if isinstance({}, np.ndarray) or isinstance({},pd.DataFrame) or isinstance({},pd.Series):\n
        shape_size ={}.shape\n    
    
    
    else:
        shape_size = 0\n  
    
    check_type = type({})\n
    with open("tas.csv", "a+") as f:\n
        TANGSHAN.append('{}')
        writer = csv.writer(f)\n
        writer.writerow(['{}', {},check_type ,shape_size])\n
"""

class CallParser(ast.NodeVisitor):
    def __init__(self, var):
        self.var = var
        self.attrs = []
        self.name = ""


    def generic_visit(self, node):
        rlist = list(ast.iter_fields(node))
        for field, value in reversed(rlist):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)

    def visit_Name(self, node):
        self.generic_visit(node)
        if node.id == self.var:
                self.attrs.append(node)


# insert new code
class CodeInstrumentator(ast.NodeTransformer):
    def __init__(self, lineno, newnode):
        self.line = lineno
        self.newnode = newnode

    def generic_visit(self, node):
        rlist = list(ast.iter_fields(node))
        for field, value in reversed(rlist):
            if isinstance(value, list):
                for item in value:
                    if hasattr(item, 'lineno') and item.lineno == self.line:
                        index = value.index(item)
                        value.insert(index + 1, self.newnode)
                        return node
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)


competitions = []
with open('competition_name.csv', 'r', encoding='utf-8') as r:
        text = csv.reader(r)
        for name in text:
            competitions.append(name[1])
competitions.append('20-newsgroups-ciphertext-challenge')
competitions.append('2018-hse-ml-competion-02')
dirs = [i for i in os.listdir(kaggle_path)]
for dir in dirs:
    if dir not in competitions:
        shutil.rmtree(os.path.join(kaggle_path, dir))
    else:
        print(dir)
exit(0)

for competition in competitions:
    code_path = os.path.join(kaggle_path, competition, 'code.py')
    items_of_file = os.listdir(os.path.join(kaggle_path, competition))
    if 'test_code.py' in items_of_file or 'v.csv' not in items_of_file:
        continue
    print(code_path)

    with open(code_path, "r", encoding='utf-8') as f:
          code_text = f.read()
    f.close()
    root = ast.parse(code_text)
    for node in ast.walk(root):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == 'print':
                new_node = ast.NameConstant()
                new_node.value = True
                node.value = new_node

    vars_path = os.path.join(kaggle_path, competition, 'v.csv')
    with open(vars_path, 'r', encoding='utf-8') as r:
        text = csv.reader(r)
        for item in text:
            if item[1] == '0':
                continue
            var = item[1]
            if var == 'len' or var == 'type' or var == 'isinstance' or var == 'open' or var == 'min' or var == 'list':
                continue
            v = CallParser(var)
            v.visit(root)
            candidates = v.attrs
            # print(var)
            # print(len(candidates))
            num = random.randint(0, len(candidates)-1)
            node = candidates[num]
            to_add = template.format(var, var, var, var, var, var, var, var, node.lineno)
            to_insert = ast.parse(to_add)

            # insert the new node
            CodeInstrumentator(node.lineno, to_insert).visit(root)
            #print(var)
        instru_source = astor.to_source(root)
        with open(os.path.join(kaggle_path, competition, 'test_code.py'), 'w+') as wf:
            wf.write(instru_source)


