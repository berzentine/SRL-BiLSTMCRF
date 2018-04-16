with open('conll2003/english/eng.train.bioes.conll','r') as rp:
    words_idx = dict()
    words_idx["_PAD"]=0
    for lines in rp:
        if lines!='\n':
            word = lines.split()[1]
            word =  "\""+word+"\""
            if word not in words_idx:
                words_idx[word]=len(words_idx)
    for w in words_idx:
        #print str(words_idx[w])+ ','
        print w+','
    print len(words_idx)



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
