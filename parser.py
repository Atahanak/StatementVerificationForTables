import sys
import untangle
import xmltodict
from pprint import pprint
import json

def usage():
    print("file")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()
        exit()
    obj = untangle.parse(sys.argv[1])
#    print(obj)

    with open (sys.argv[1]) as f:
        doc = xmltodict.parse(f.read())
#    pprint(doc)
    jdata = json.dumps(doc, indent=4, sort_keys=True)
    print(jdata)