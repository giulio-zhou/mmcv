import inspect
import os.path
import shutil
import sys
from importlib import import_module
from tempfile import mkdtemp

def merge_modules(left, right, output):
    left_module_name = os.path.basename(left[:-3])
    right_module_name = os.path.basename(right[:-3])

    tempdir = mkdtemp()
    shutil.copyfile(left, '%s/left_module.py' % tempdir)
    shutil.copyfile(right, '%s/right_module.py' % tempdir)
    sys.path.insert(0, tempdir)
    left_module = import_module('left_module')
    right_module = import_module('right_module')
    sys.path.pop(0)

    # Left module is the default one.
    for field in dir(left_module):
        if '__' not in field:
            if field in dir(right_module):
                print("Replace field %s: %s with %s" % (field, getattr(right_module, field), getattr(left_module, field)))
            setattr(right_module, field, getattr(left_module, field))
    with open(output, 'w') as outfile:
        for field in dir(right_module):
            if '__' not in field:
                out_elem = getattr(right_module, field)
                if type(out_elem) == str:
                    out_elem = "\'%s\'" % out_elem
                outfile.write('%s = %s\n' % (field, out_elem))

    shutil.rmtree(tempdir)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'merge_modules':
        left = sys.argv[2]
        right = sys.argv[3]
        output = sys.argv[4]
        merge_modules(left, right, output)
