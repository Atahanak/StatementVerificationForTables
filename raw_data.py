"""
Represents raw data in structured format with utilities attached
"""
import pandas as pd
import random
class TableForVerification():
    def __init__(self, obj, dtype = None):
        if dtype == None:
            self.label_map = {
                "refuted": 0,
                "entailed": 1,
                "unknown": 2
            }
        elif dtype == "un":
            self.label_map = {
                "refuted": 1,
                "entailed": 1,
                "unknown": 0
            }
        elif dtype == "enre":
            self.label_map = {
                "refuted": 0,
                "entailed": 1,
            }

        self.dtype = dtype
        self.legend = ""
        self.caption = ""
        self.column_names = []
        self.rows = []
        self.statements = []
        self.parse_table(obj)
        #print("****************")
        #print("LEGEND: ", self.legend)
        #print("CAPTION: ", self.caption)
        #print("COLUMN NAMES: ", self.column_names)
        #print("ROWS: ", self.rows)
        #print("STATEMENTS: ", self.statements)
        #print("****************")

    def parse_table(self, table):
        if 'caption' in table:
            self.caption = table['caption']['@text'].strip(' ')

        for idx, r in enumerate(table['row']):
            row = [] 
            if type(r['cell']) == list:
                for c in r['cell']:
                    for t in range (int(c['@col-end']) - int(c['@col-start'])+1):
                        row.append(c['@text'].strip(' '))
            else:
                for t in range (int(r['cell']['@col-end']) - int(r['cell']['@col-start'])+1):
                    row.append(r['cell']['@text'].strip(' '))

            if idx == 0:
                self.column_names = row  
            else:
                self.rows.append(row)

        if 'legend' in table:
            self.legend = table['legend']['@text'].strip(' ')

        if 'statements' in table: 
            if 'statement' in table['statements']:
                if type(table['statements']['statement']) == list:
                    for s in table['statements']['statement']:
                        self.statements.append((s['@text'].strip(' '), self.label_map[s['@type']]))
                else:
                    self.statements.append((table['statements']['statement']['@text'].strip(' '), self.label_map[table['statements']['statement']['@type']]))

    def generate_sequence_from_table(self):
        result = ""
        for row in self.rows:
            for idx, col in enumerate(row):
                #result += f'{self.column_names[idx]} {col}. '
                result += f'{col} '
            result += '\n'
        #print(result)
        return result

    def generate_df_from_table(self):
        df = pd.DataFrame(self.rows, columns=self.column_names)
        #print(df)
        return df.astype(str)

    def lol(self):
        #print(self.statements[0][0])
        return self.statements[0][0]
    def lal(self):
        #print(self.statements[0][0])
        return [self.statements[0][1]]

    def get_samples_and_labels(self, tt=None):
        samples = []
        labels = []
        tab_seq = self.generate_sequence_from_table()
        for st in self.statements:
            samples.append((tab_seq, st[0]))
            labels.append(st[1])
        if tt != None and len(tt) > 0:
            for x in range(0, int(len(self.statements)/3)):
                samples.append((tab_seq, random.choice(tt)[1]))
                labels.append(2)
        return {"samples": samples, "labels": labels}
    
    def populate_tables_statements_labels(self, tables, statements, labels):
        table = self.generate_df_from_table()
        for st in self.statements:
            tables.append(table)
            statements.append(st[0])
            labels.append(st[1])

    def populate_tables_statements_labels_tt(self, tables, statements, labels):
        pp = statements
        table = self.generate_df_from_table()
        for st in self.statements:
            tables.append(table)
            statements.append(st[0])
            labels.append(st[1])
        if self.dtype != "enre":
            for x in range(0, int(len(self.statements)/3)):
                tables.append(table)
                labels.append(self.label_map["unknown"])
                statements.append(random.choice(pp))