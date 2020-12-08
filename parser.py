"""
parsers xml formated tables and statements preceding
"""
import sys
import json
#import untangle
import xmltodict
#from pprint import pprint

def usage():
    ''' prints usage '''
    print("file")

def xml2dict(xml_file_name):
    ''' dump xml to dictionary '''
    with open(xml_file_name) as file:
        doc = xmltodict.parse(file.read())
    return json.dumps(doc, indent=4, sort_keys=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()
        sys.exit()

    file_name = sys.argv[1]
    data = xml2dict(file_name)
    print(data)
