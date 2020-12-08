import ast
import typed_ast
with open('linalg.pyi', 'r') as f:
    text = f.read()
r = ast.parse(text)
print(ast.dump(r))
