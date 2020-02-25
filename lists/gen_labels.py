import os
import sys

def main(argv):
    if(len(argv) < 1):
        print("usage: python gen_labels.py [flist] optional:[decode]")
        exit(-1)
    try:
        flist = open(argv[0], "r")
    except IOError as e:
        print("[%s]: %s" % (sys.argv[0], e))
        exit(-1)
    files = flist.read().split()
    flist.close()
    labels = open(os.path.splitext(argv[0])[0] + ".ref", "w")
    for fname in files:
        if len(argv) == 1:
            base_dir = os.path.dirname(fname)
            last_dir = base_dir.rfind('/')
            labels.write(base_dir[last_dir+1:] + "\n")
        else:
            name_file = os.path.basename(fname)
            tokenized = name_file.split("_test")
            labels.write(tokenized[0] + "\n")
    labels.close()
    return True

if __name__ == '__main__':
    main(sys.argv[1:])
