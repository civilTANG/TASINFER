import numpy as np
attr = ['flags', 'shape', 'strides', 'ndim', 'data', 'size', 'itemsize', 'nbytes', 'base',  # 内存布局
        'dtype',    # 数组元素的数据类型
        'T'	, 'real', 'imag', 'flat', 'ctypes',  # 其他属性
        '__array_interface__', '__array_struct__',# 数组接口
        'ctypes'  ] #一个简化数组与ctypes模块交互的对象

method_numpy = ['all', 'any', 'argmax', 'argmin', 'argpartition',
                'argsort', 'choose', 'clip', 'compress', 'copy',
                'cumprod', 'cumsum', 'diagonal', 'imag', 'max',
                'mean', 'min', 'nonzero', 'partition', 'prod',
                'ptp', 'put', 'ravel', 'real', 'repeat', 'reshape', 'round', 'searchsorted',
                'sort', 'squeeze', 'std', 'sum', 'swapaxes', 'take', 'trace', 'transpose', 'var']# numpy 里面也有的

array_transpose = ['item', 'tolist', 'itemset', 'tostring', 'tobytes', 'tofile', 'dump', 'dumps',
                'astype', 'byteswap', 'copy', 'view', 'getfield', 'setflags', 'fill']  # 数组转化

array_shape = ['reshape', 'resize', 'transpose', 'swapaxes', 'flatten', 'ravel', 'squeeze']  # 形状操作

array_select = ['take', 'put', 'repeat', 'choose', 'sort', 'argsort', 'partition', 'argpartition',
                'searchsorted', 'nonzero', 'compress', 'diagonal']  # 数组选择

array_calculation = ['max', 'argmax', 'min', 'argmin', 'ptp', 'clip', 'conj', 'round',
                     'trace', 'sum', 'cumsum', 'mean', 'var', 'std', 'prod', 'cumprod', 'all', 'any']  # 计算

matrix_attr = ['T', 'H', 'I', 'A']
matrix_fun = ['mat', 'bmat','empty','zeros','ones','eye', 'identity', 'repmat' , 'rand', 'randn']


creat_array = ['empty', 'empty_like', 'eye', 'identity', 'ones', 'ones_like', 'zeros',
               'zeros_like', 'full', 'full_like',# 创建数组 从现有数据创建
               'array', 'asarray', 'asanyarray', 'ascontiguousarray', 'asmatrix', 'copy',
               'frombuffer', 'fromfile', 'fromfunction', 'fromiter', 'fromstring', 'loadtxt'  # 创建数组 从原有的数据
]

creat_matrix = ['diag','diagflat', 'tri', 'tril', 'triu', 'vander','mat', 'bmat'] #  矩阵类创建

shape_numpy = ['copyto', 'reshape', 'ravel', 'moveaxis', 'rollaxis', 'swapaxes', 'transpose',
'concatenate', 'stack', 'column_stack', 'dstack', 'hstack', 'vstack', 'block',
'split', 'array_split', 'dsplit', 'hsplit', 'vsplit', 'tile', 'repeat','delete',
'insert', 'append', 'resize', 'trim_zeross', 'unique']  # 更改维度数



