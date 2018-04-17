#from __future__ import print_function
__author__ = 'Nidhi'
"""
Post-process wrapper for evaluating SRL
"""
import os
import subprocess
from subprocess import Popen, PIPE, STDOUT

def eval(gold_file, pred_file):
    #os.system('pwd')

    #print('./examples/eval/srl-eval.pl '+gold_file+' '+pred_file)
    #import os.path
    score_file = 'temp_result.txt'
    #command  = './examples/eval/srl-eval.pl '+gold_file+' '+pred_file
    os.system("./examples/eval/srl-eval.pl  %s  %s > %s" % (gold_file, pred_file, score_file))
    #os.system("examples/eval/conll03eval.v2 < %s > %s" % (output_file, score_file))
    #ps = subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    #output = ps.communicate()[1]
    #print str(output).split()
    #exit(0)
    #path = os.path.join(BaseDirectory, 'subdir1', 'subdir2')
    #os.system('./examples/eval/srl-eval.pl '+gold_file+' '+pred_file+' | grep \'Overall\' > temp_result')
    #os.system("pwd")
    result = []
    #with open("temp_result", "w") as output:
    #subprocess.Popen(command,shell=True,stdout=output,stderr=subprocess.STDOUT)
    with open(score_file, 'r') as lp:
        i  = 0
        print lp.readlines()
        for l in lp:
            print l
            if i==6:
                line = l.split()
                result = l.split()[4:7]
                break
            i+=1
    #exit(0)
    return result[0], result[1], result[2]
    # ../bin/srl-eval.pl ../../476526_test181_pred.tsv ../../476526_test181_gold.tsv -P | grep 'Overall'
    # evaluate the output files


# Take a input file and generate output files X 2
def makedict(in_file):
    sent = dict() # holds the sentence -> mapping to all its target and predictions
    sent_verbs = dict() # holds the verbs of the sentence in the order of occurance
    #in_file = '476526_dev187'
    words_sent = ""
    verbs, tgts, preds = [], [], []
    # 97 sick - 0 I-ARG2 I-ARG2 = ... tgt pred
    with open(in_file,'r') as rp:
        for lines in rp:
            if len(lines.split())==0:
                # add the sentence read into the verb dictionary for its verbs
                if words_sent not in sent_verbs:
                    sent_verbs[words_sent] = []
                sent_verbs[words_sent]+=verbs
                # add the sentence read into the tgts predictios in order
                if words_sent not in sent:
                    sent[words_sent] = []
                sent[words_sent].append((tgts, preds))
                verbs, tgts, preds = [], [], []
                words_sent = ""
                continue
            words_sent+=(lines.split()[1])+" "
            tgt = lines.split()[4]
            pred = lines.split()[5]
            tgts.append(tgt)
            preds.append(pred)
            if tgt=="B-V" or tgt=="I-V":
                verbs.append(lines.split()[1])
    gold_file, pred_file = generate(sent, sent_verbs, in_file)
    return eval(gold_file, pred_file)


    #return sent, sent_verbs


def generate(sent, sent_verbs, in_file):
    gold_file = in_file+"_gold.tsv"
    pred_file = in_file+"_pred.tsv"
    with open(gold_file,'w') as gf:
        with open(pred_file,'w') as pf:
            assert (len(sent)==len(sent_verbs))
            for s in sent: # for each sentence
                words = s.split()
                #bracket_is_open = False
                for w in range(len(words)): # for each word in sentence
                    if words[w] in sent_verbs[s]: # write the verb or a -
                        gf.write(words[w]+'\t')
                        pf.write(words[w]+'\t')
                    else:
                        gf.write('-'+'\t')
                        pf.write('-'+'\t')

                    for ann in sent[s]: # write all annotations for the word in sentence
                        if ann[0][w]=='O':
                            gf.write('*'+'\t')
                        elif ann[0][w][0]=='B' and w!=(len(words)-1) and ann[0][w+1][0]=='I':
                            gf.write('('+ann[0][w][2:]+'*'+'\t')
                        elif ann[0][w][0]=='B' and (w==(len(words)-1) or ann[0][w+1][0]!='I'):
                            gf.write('('+ann[0][w][2:]+'*)'+'\t')
                        elif ann[0][w][0]=='I' and w!=(len(words)-1) and ann[0][w+1][0]=='I':
                            gf.write('*'+'\t')
                        elif ann[0][w][0]=='I' and (w==(len(words)-1) or ann[0][w+1][0]!='I'):
                            gf.write('*)'+'\t')


                        if ann[1][w]=='O':
                            pf.write('*'+'\t')
                        elif ann[1][w][0]=='B' and w!=(len(words)-1) and ann[1][w+1][0]=='I':
                            pf.write('('+ann[1][w][2:]+'*'+'\t')
                            #bracket_is_open = True
                        elif ann[1][w][0]=='B' and (w==(len(words)-1) or ann[1][w+1][0]!='I'):
                            pf.write('('+ann[1][w][2:]+'*)'+'\t')
                        elif ann[1][w][0]=='I' and w!=(len(words)-1) and ann[1][w+1][0]=='I':
                            pf.write('*'+'\t')
                        elif ann[1][w][0]=='I' and (w==(len(words)-1) or ann[1][w+1][0]!='I'):
                            # w==0 then?
                            if w==0 or ann[1][w-1][1:]!=ann[1][w][1:]:
                                pf.write('*'+'\t')
                            else:
                                pf.write('*)'+'\t')



                            #if bracket_is_open and w==(len(words)-1):
                            #pf.write('*)'+'\t')
                            #bracket_is_open = False
                            #elif !bracket_is_open and w==(len(words)-1):
                            #    pf.write('*'+'\t')

                            # if bracket open is set only then clos it else jut put *

                        #gf.write(ann[0][w]+'\t')
                        #pf.write(ann[1][w]+'\t')
                    # add new line here after each word is added
                    gf.write('\n')
                    pf.write('\n')
                # add new line here afer each sentence is added
                gf.write('\n')
                pf.write('\n')

    return gold_file, pred_file
"""

if __name__ == '__main__':
    prec, recal, f1 = makedict()
    print prec, recal, f1
"""
