import ast
import os
import csv
import astunparse
import random
import astor
kaggle_path = os.path.join('..', 'kaggle_code')
template = """ 
if '{}'not in TANGSHAN:   \n
    import csv\n
    if isinstance({}, np.ndarray) or isinstance({},pd.DataFrame) or isinstance({},pd.Series):\n
        shape_size ={}.shape\n    
    
    elif isinstance({},list):\n
        shape_size = len({})\n
    
    else:
        shape_size = 'any'\n  
    
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


for competition in os.listdir(kaggle_path):
    code_path = os.path.join(kaggle_path, 'digit-recognizer', 'code.py')
    with open(code_path, "r", encoding='utf-8') as f:
          code_text = f.read()
    f.close()
    root = ast.parse(code_text)
    flag = 'TANGSHAN = []'
    to_insert = ast.parse(flag)
    CodeInstrumentator(1, to_insert).visit(root)
    vars_path = os.path.join(kaggle_path, 'digit-recognizer', 'v.csv')
    with open(vars_path, 'r', encoding='utf-8') as r:
        text = csv.reader(r)
        for i in text:
            if i[1] == '0':
                continue
            var = i[1]
            if var == 'len' or var == 'type' or var == 'isinstance' or var == 'open':
                continue
            v = CallParser(var)
            v.visit(root)
            candidates = v.attrs
            num = random.randint(0, len(candidates)-1)
            node = candidates[num]
            to_add = template.format(var, var, var, var, var, var, var, var, var,var,node.lineno)
            #print(to_add)
            #print(to_add)
            to_insert = ast.parse(to_add)

            # insert the new node
            CodeInstrumentator(node.lineno, to_insert).visit(root)
            print(var)
        instru_source = astor.to_source(root)
        with open(os.path.join(kaggle_path, 'digit-recognizer', 'test_code.py'), 'w+') as wf:
            wf.write(instru_source)

    exit(0)

