with open('conll2003/english/eng.train.bioes.conll','r') as rp:
    chars_idx = dict()
    for lines in rp:
        if lines!='\n':
            for c in lines.split()[1]:
                if c.isalpha() or c.isdigit():
                    c = "\""+c+"\""
                    if c not in chars_idx:
                        chars_idx[c]=len(chars_idx)
    for c in chars_idx:
        print c+ ','
        #print c+':'+str(chars_idx[c])+ ','
    print len(chars_idx)



"""
1 In - - B-ARGM-LOC
2 Jerusalem - - I-ARGM-LOC
3 Saul - - B-ARG0
4 was - - O
5 still - - B-ARGM-TMP
6 trying - - B-V
7 to - - B-ARG1
8 scare - - I-ARG1
9 the - - I-ARG1
10 followers - - I-ARG1
"""
