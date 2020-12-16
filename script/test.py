import ast
import astor


class DataFlow(object):
    def __init__(self, code_path, identifier):
        with open(code_path, 'r', encoding='utf-8') as f:
            code = f.read()
        self.code_path = code_path
        self.code = code
        self.identifier = identifier
        self.data_flow = []
        self.module_identifier = []
        self.use_definition_identifier = []
        self.result = []

    def pre_processing(self):
        root = ast.parse(self.code)
        for node in ast.walk(root):
            # 收集导入的模块名称，类
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                for temp_node in ast.walk(node):
                    if isinstance(temp_node, ast.alias):
                        if temp_node.asname:
                            self.module_identifier.append(temp_node.asname)
                        else:
                            self.module_identifier.append(temp_node.name)
            # 收集用户自定义的方法和类
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef):
                self.use_definition_identifier.append(node.name)

    def __deal_for__(self, node):
        if isinstance(node.target, ast.Name):
            if node.target.id == self.identifier:
                self.data_flow.append(self.identifier + 'in' + astor.to_source(node.iter))
        elif isinstance(node.target, ast.Tuple):
            for elt in node.target.elts:
                if isinstance(elt, ast.Name):
                    if elt.id == self.identifier:
                        self.data_flow.append(self.identifier + 'in' + astor.to_source(node.iter))

        for for_node in node.body:
            # print(astor.to_source(for_node))
            if isinstance(for_node, ast.For):
                self.__deal_for__(for_node)
            elif isinstance(for_node, ast.If):
                self.__deal_if__(for_node)
            else:
                for find_point in ast.walk(for_node):
                    if isinstance(find_point, ast.Name):
                        if find_point.id == self.identifier:
                            self.data_flow.append([for_node, find_point.ctx])
                            # print(astor.to_source(sub_node))
                            # print(sub_node.lineno)
                            break

    def __deal_if__(self, node):
        for p_cond in ast.walk(node.test):
            if isinstance(p_cond, ast.Name):
                if p_cond.id == self.identifier:
                    self.data_flow.append([p_cond, p_cond.ctx])

        for stmt_node in node.body:
            # print(astor.to_source(stmt_node))
            if isinstance(stmt_node, ast.For):
                self.__deal_for__(stmt_node)
            elif isinstance(stmt_node, ast.If):
                self.__deal_if__(stmt_node)
            else:
                for find_point in ast.walk(stmt_node):
                    if isinstance(find_point, ast.Name):
                        if find_point.id == self.identifier:
                            self.data_flow.append([stmt_node, find_point.ctx])
                            break

        for stmt_node in node.orelse:
            if isinstance(stmt_node, ast.For):
                self.__deal_for__(stmt_node)
            elif isinstance(stmt_node, ast.If):
                self.__deal_if__(stmt_node)
            else:
                for find_point in ast.walk(stmt_node):
                    if isinstance(find_point, ast.Name):
                        if find_point.id == self.identifier:
                            self.data_flow.append([stmt_node, find_point.ctx])
                            break

    def collect_chain(self):
        root = ast.parse(self.code)
        for node in ast.iter_child_nodes(root):
            for find_point in ast.walk(node):
                if isinstance(find_point, ast.Name):
                    if find_point.id == self.identifier:
                        if isinstance(node, ast.For):
                            self.__deal_for__(node)
                        elif isinstance(node, ast.If):
                            self.__deal_if__(node)
                        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef):
                            pass
                        else:
                            self.data_flow.append([node, find_point.ctx])
                        break

    def collect(self):
        self.pre_processing()
        self.collect_chain()
        for i in range(0, len(self.data_flow)):
            self.result.append((astor.to_source(self.data_flow[i][0]).strip(),
                                self.data_flow[i][0].lineno, self.data_flow[i][1]))


def use_constrain_pattern(statement, var):
    all_node = []
    root = ast.parse(statement)
    #print(ast.dump(root))
    for node in ast.walk(root):
        if isinstance(node, ast.Call):
            for temp_node in ast.walk(node):
                if isinstance(temp_node, ast.Name) and temp_node.id == var:
                    all_node.append([node, astor.to_source(node).strip()])
                    break
        if isinstance(node, ast.BinOp) or isinstance(node, ast.Compare):
            for temp_node in ast.walk(node):
                if isinstance(temp_node, ast.Name) and temp_node.id == var:
                    all_node.append([node, astor.to_source(node).strip()])
                    break
    min_len = len(statement)
    tag = 0
    for j in range(len(all_node)):
        temp_len = len(all_node[j][1])
        if temp_len < min_len:
            tag = j
            min_len = temp_len
    if isinstance(all_node[tag][0], ast.Call):
        return [1, all_node[tag][1]]  # 函数调用
    elif isinstance(all_node[tag][0],ast.BinOp) or isinstance(all_node[tag][0],ast.Compare):
        return [2, all_node[tag][1]] # 二目操作
    else:
        return None


var = 'y_hat'
x = DataFlow('code.py', var)
x.collect()
line = 427
rely_var = []

# print(x.result)

for i in range(len(x.result)):
    if x.result[i][1] == line:
        tag = i
left = tag
right = tag+1
print(x.result)
while left >= 0:
    if isinstance(x.result[left][2], ast.Store):
        break
    else:
        seed = use_constrain_pattern(x.result[left][0], var)
        if seed:
            print(seed)
        else:
            pass

        left = left - 1

while right < len(x.result):
    if isinstance(x.result[right][2], ast.Store):
        break
    else:
        right = right + 1

if left < 0:
    left = 0

