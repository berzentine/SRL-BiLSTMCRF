with open('conll2003/english/eng.train.bioes.conll','r') as rp:
    tags = dict()
    for lines in rp:
        if lines!='\n':
            if lines.split()[-1] not in tags:
                tags[lines.split()[-1]] = 1
    print(tags, len(tags))
    for t in tags:
        print(t)
