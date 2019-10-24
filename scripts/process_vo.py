import sys
import json

def process_vo(lines):
    vo_dict = {}
    for line in lines:
        splits = line.split(" ")
        first = splits[0]
        second = splits[2]
        if second not in vo_dict:
            vo_dict[second] = [(first, 1.0)] #also include dummy score for data consitancy
        elif (first,1.0) not in vo_dict[second]:
            vo_dict[second].append((first, 1.0))

    return vo_dict


if __name__ == "__main__":
    vo_file = sys.argv[1]
    outfile = sys.argv[2]
    with open(vo_file, 'r') as fi:
        lines = fi.readlines()

    vo_dict = process_vo(lines)

    with open(outfile, 'w') as out:
        json.dump(vo_dict, out)
    
