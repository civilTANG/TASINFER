
def array(object=None, dtype=None, copy=None, order=None, subok=None, ndmin=None, like=None):
    shape = None
    dem = None
    if object:
        shape = object + '.shape'
        dem = object + '.dem'
        return [shape, dem]
    return [shape, dem]


def linspace(start = None,stop = None,num = None,endpoint = None,retstep = None,dtype = None ):
    shape = None
    dem = None
    if num:
        shape = num
        dem = 1
    return [shape, dem]

def sqrt(x = None):
    shape = None
    dem = None
    if x:
        shape = x + '.shape'
        dem = x + '.shape'
    return [shape, dem]

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


print(linspace(0.00125, 0.00125, 'oinst'))