"""
Represents raw data in structured format with utilities attached
"""
class TableForVerification():
    def __init__(self, obj):
        self.label_map = {
            "refuted": 0,
            "entailed": 1
        }
        self.legend = ""
        self.caption = ""
        self.column_names = []
        self.rows = []
        self.statements = []
        self.parse_table(obj)

    def parse_table(self, table):
        if 'caption' in table:
            self.caption = table['caption']['@text']

        for idx, r in enumerate(table['row']):
            row = [] 
            if type(r['cell']) == list:
                for c in r['cell']:
                    for t in range (int(c['@col-end']) - int(c['@col-start'])+1):
                        row.append(c['@text'])
            else:
                for t in range (int(r['cell']['@col-end']) - int(r['cell']['@col-start'])+1):
                    row.append(r['cell']['@text'])

            if idx == 0:
                self.column_names = row  
            else:
                self.rows.append(row)

        if 'legend' in table:
            self.legend = table['legend']['@text']

        if 'statements' in table: 
            if 'statement' in table['statements']:
                if type(table['statements']['statement']) == list:
                    for s in table['statements']['statement']:
                        #print(s)
                        self.statements.append((s['@text'], self.label_map[s['@type']]))
                else:
                    self.statements.append((table['statements']['statement']['@text'], self.label_map[table['statements']['statement']['@type']]))

    def generate_sequence_from_table(self):
        result = ""
        for row in self.rows:
            for idx, col in enumerate(row):
                result += f'{self.column_names[idx]} {col}. '
        return result

    def get_samples_and_labels(self):
        samples = []
        labels = []
        tab_seq = self.generate_sequence_from_table()
        for st in self.statements:
            samples.append((tab_seq, st[0]))
            labels.append(st[1])
        return {"samples": samples, "labels": labels}