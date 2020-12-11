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
    #return json.dumps(doc, indent=4, sort_keys=True)
    return doc

def getTableElements(table):
    print("************************************")
    caption = table['caption']['@text']
    print("Caption:")
    print(caption)
    print()

    print("Rows:")
    rows = table['row']
    for r in rows:
        row =  ""
        for c in r['cell']:
            row += c['@text'] + " "
        print(row)

    legend = table['legend']['@text']
    print("Legend:")
    print(legend)
    
    statements = table['statements']['statement']
    print()
    print("Statements:")
    for s in statements:
        print(s['@text'])
    print("************************************")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()
        sys.exit()

    file_name = sys.argv[1]
    data = xml2dict(file_name)
    #print(json.dumps(data, indent=4, sort_keys=True))
    table = data['document']['table']
    if type(table) == dict:
        getTableElements(table)        
    elif type(table) == list:
        for t in table:
            getTableElements(t)
    else:
        print("Unknown table type.")
    #print(data)
