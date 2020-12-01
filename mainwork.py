import ast
import astor
import constraintGenerator
code_path = 'code.py'
with open(code_path, 'r', encoding='utf-8') as f:
    code_text = f.read()

root = ast.parse(code_text)
module_identifier = []
use_definition_identifier = []
global_variable = []
local_variable = set()


# preprocess data
for node in ast.walk(root):
    if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
        for temp_node in ast.walk(node):
            if isinstance(temp_node, ast.alias):
                if temp_node.asname:
                    module_identifier.append(temp_node.asname)
                else:
                    module_identifier.append(temp_node.name)
        # print(astor.to_source(node))
        # print(node.lineno)

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

    elif isinstance(node, ast.If):
        pass
        # print(astor.to_source(node))
        # print(node.lineno)

    elif isinstance(node, ast.For):
        pass

    elif isinstance(node, ast.Expr):
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name) and node.value.func.id == 'print':
                new_node = ast.NameConstant()
                new_node.value = True
                node.value = new_node

for node in ast.walk(root):
    if isinstance(node, ast.Name):
        if node.id in module_identifier or node.id in use_definition_identifier or node.id in global_variable:
            pass
        else:
            local_variable.add(node.id)


# Data flow extraction and control flow extraction

def deal_for(node, variable, data_flow_sqe):
    if isinstance(node.target, ast.Name):
        if node.target.id == variable:
            data_flow_sqe.append(variable + 'in' + astor.to_source(node.iter))
    elif isinstance(node.target, ast.Tuple):
        for elt in node.target.elts:
            if isinstance(elt, ast.Name):
                if elt.id == variable:
                    data_flow_sqe.append(variable + 'in' + astor.to_source(node.iter))

    for for_node in node.body:
        # print(astor.to_source(for_node))
        if isinstance(for_node, ast.For):
            deal_for(for_node, variable, data_flow_sqe)
        elif isinstance(for_node, ast.If):
            deal_if(for_node, variable, data_flow_sqe)
        else:
            for find_point in ast.walk(for_node):
                if isinstance(find_point, ast.Name):
                    if find_point.id == variable:
                        data_flow_sqe.append(astor.to_source(for_node))
                        # print(astor.to_source(sub_node))
                        # print(sub_node.lineno)
                        break


def deal_if(node, variable, data_flow_sqe):
    for p_cond in ast.walk(node.test):
        if isinstance(p_cond, ast.Name):
            if p_cond.id == variable:
                data_flow_sqe.append(astor.to_source(p_cond))

    for stmt_node in node.body:
        # print(astor.to_source(stmt_node))
        if isinstance(stmt_node, ast.For):
            deal_for(stmt_node, variable, data_flow_sqe)
        elif isinstance(stmt_node, ast.If):
            deal_if(stmt_node, variable, data_flow_sqe)
        else:
            for find_point in ast.walk(stmt_node):
                if isinstance(find_point, ast.Name):
                    if find_point.id == variable:
                        data_flow_sqe.append(astor.to_source(stmt_node))
                        break

    for stmt_node in node.orelse:
        if isinstance(stmt_node, ast.For):
            deal_for(stmt_node, variable, data_flow_sqe)
        elif isinstance(stmt_node, ast.If):
            deal_if(stmt_node, variable, data_flow_sqe)
        else:
            for find_point in ast.walk(stmt_node):
                if isinstance(find_point, ast.Name):
                    if find_point.id == variable:
                        data_flow_sqe.append(astor.to_source(stmt_node))
                        break


# Constraint solver

def constraint(stmt):
    generator = constraintGenerator.ConstrainGenerator()
    stmt = 'a[b]'
    t = ast.parse(stmt)
    if isinstance(t.body[0], ast.Assign):
        # 变量被定义
        for target in t.body[0].targets:
            if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store) and 1:
                # 变量被函数调用定义
                if isinstance(t.body[0].value, ast.Call):
                    if isinstance(t.body[0].value.func, ast.Attribute):
                        m_name = t.body[0].value.func.value.id
                        f_name = t.body[0].value.func.attr
                        a_name = t.body[0].value.args
                        for i in range(0, len(a_name)):
                            a_name[i] = astor.to_source(a_name[i]).strip()
                        if m_name in module_identifier:
                            constrain = generator.constrain_function_definition(m_name, f_name, a_name)
                        else:
                            constrain = generator.constrain_function_definition(m_name, f_name, a_name)
                    else:
                        f_name = t.body[0].value.func.id
                        if f_name in use_definition_identifier:
                            pass
                        else:
                            a_name = t.body[0].value.args
                            for i in range(0, len(a_name)):
                                a_name[i] = astor.to_source(a_name[i]).strip()
                            print(f_name, a_name)
                            constrain = generator.constrain_function_definition('python', f_name, a_name)
                # 被属性定义
                elif isinstance(t.body[0].value, ast.Attribute):
                    v_name = t.body[0].value.value.id
                    attr_name = t.body[0].value.attr
                    print(v_name, attr_name)
                    constrain = generator.constrain_attr_definition(v_name, attr_name)

                # 被二目操作定义
                elif isinstance(t.body[0].value, ast.BinOp):
                    pass

                elif isinstance(t.body[0].value, ast.Subscript):
                    pass

        # 变量被使用

        # 变量作为函数调用被使用
        if isinstance(t.body[0].value, ast.Call):

            # 变量作为函数调用本体在使用
            if isinstance(t.body[0].value.func, ast.Attribute):
                if t.body[0].value.func.value.id == 1:
                    f_name = t.body[0].value.attr
                else:

                    # 变量作为函数调用参数在使用

                    m_name = t.body[0].value.func.value.id
                    f_name = t.body[0].value.attr
                    a_name = t.body[0].value.args
                    for i in range(0, len(a_name)):
                        a_name[i] = astor.to_source(a_name[i]).strip()
            else:
                a_name = t.body[0].value.args
                for i in range(0, len(a_name)):
                    a_name[i] = astor.to_source(a_name[i]).strip()

        # 变量属性被使用
        elif isinstance(t.body[0].value, ast.Attribute):
            if t.body[0].value.value.id == 1:
                attr_name = t.body[0].value.attr

        # 变量作为切片被使用
        elif isinstance(t.body[0].value, ast.Subscript):
            if t.body[0].value.value.id == 1:
                pass
            else:
                pass

    # 变量被使用
    if isinstance(t.body[0], ast.Expr):
        # 变量作为函数调用被使用
        if isinstance(t.body[0].value, ast.Call):

            # 变量作为函数调用本体在使用
            if isinstance(t.body[0].value.func, ast.Attribute):
                if t.body[0].value.func.value.id == 1:
                    f_name = t.body[0].value.attr
                else:

                    # 变量作为函数调用参数在使用

                    m_name = t.body[0].value.func.value.id
                    f_name = t.body[0].value.attr
                    a_name = t.body[0].value.args
                    for i in range(0, len(a_name)):
                        a_name[i] = astor.to_source(a_name[i]).strip()
            else:
                a_name = t.body[0].value.args
                for i in range(0, len(a_name)):
                    a_name[i] = astor.to_source(a_name[i]).strip()

        # 变量属性被使用
        elif isinstance(t.body[0].value, ast.Attribute):
            if t.body[0].value.value.id == 1:
                attr_name = t.body[0].value.attr

        # 变量作为切片被使用
        elif isinstance(t.body[0].value, ast.Subscript):
            if t.body[0].value.value.id == 1:
                pass
            else:
                pass

    print(ast.dump(t.body[0]))

    exit(0)


for variable in local_variable:
    print(variable)
    data_flow_sqe = []
    for node in ast.iter_child_nodes(root):
            for find_point in ast.walk(node):
                if isinstance(find_point, ast.Name):
                    if find_point.id == variable:
                        if isinstance(node, ast.For):
                            deal_for(node, variable, data_flow_sqe)
                        elif isinstance(node, ast.If):
                            deal_if(node, variable, data_flow_sqe)
                        else:
                            data_flow_sqe.append(astor.to_source(node))
                        break



# print(module_identifier)
# print(use_definition_identifier)
# print(astor.to_source(root))