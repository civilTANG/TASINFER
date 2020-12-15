import os
import ast
import  numpy as np
import pandas as pd
save_path = "../kaggle_code"

for folder in os.listdir(save_path):
    code_path = os.path.join(save_path,folder,'code.py')
    code_path = os.path.join(save_path, 'digit-recognizer', 'code.py')
    with open(code_path, 'r', encoding='utf-8') as f:
        code_text = f.read()

    try:
        root = ast.parse(code_text)
        module_identifier = []
        use_definition_identifier = []
        global_variable = []
        variables = set()
        for node in ast.walk(root):
             if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                 for temp_node in ast.walk(node):
                     if isinstance(temp_node, ast.alias):
                         if temp_node.asname:
                             module_identifier.append(temp_node.asname)
                         else:
                             module_identifier.append(temp_node.name)
             elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef):
                    use_definition_identifier.append(node.name)
                    new_node = ast.Pass()
                    node.body = [new_node]
                # print(astor.to_source(node))
             elif isinstance(node, ast.Global):
                 if isinstance(node.names, list):
                     for name in node.names:
                         global_variable.append(node.names)
                 else:
                     global_variable.append(node.names)
                 # print(astor.to_source(node))

        for node in ast.walk(root):
            if isinstance(node, ast.Name):
                if node.id not in use_definition_identifier and node.id not in module_identifier:
                    variables.add(node.id)

        for variable in variables:
            if variable in module_identifier:
                print('#' + variable + '# is module type')
                continue
            elif variable in use_definition_identifier:
                print('#' + variable + '# is functype')
                continue

        df = pd.DataFrame(data=variables)
        save_csv = code_path.replace('code.py', 'v.csv')
        print(save_csv)
        df.to_csv(save_csv)
        print(use_definition_identifier)
        print(module_identifier)
        print(global_variable)
        exit(0)
    except Exception as e:
        print(e)
