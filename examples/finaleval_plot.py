from __future__ import print_function
import matplotlib
matplotlib.use('agg')
import seaborn as sns
from examples.eval import makedict
import pandas as pd

__author__ = 'Nidhi'
"""
Function to plot and save the graph for accuracy on dev set.
"""

# plotting results across different epochs on the dev set
def plot(uid_name, results_file):
    """ Plots the F1 curves """
    print("Plotting P-R-F1 curve")
    precision, recall, F1_measure  = [], [], []
    with open(results_file, 'r') as rp:
        for lines in rp:
            if lines.split()[0]!="Epoch:":
                continue
            prec, rec, F1 = float(lines.split()[3]), float(lines.split()[4]), \
                            float(lines.split()[5])
            precision.append(prec)
            recall.append(rec)
            F1_measure.append(F1)

    df = pd.DataFrame({"Precision":precision, "Recall":recall, "F1":F1_measure},
                      columns=["Precision","Recall", "F1"])
    df["Epochs"] = df.index

    var_name = "Metric"
    value_name = "Precentage Value"
    df = pd.melt(df, id_vars=["Epochs"], value_vars=["Precision","Recall", "F1"],
                 var_name=var_name, value_name=value_name)
    sns.tsplot(df, time="Epochs", unit=var_name, condition=var_name,
               value=value_name)
    matplotlib.pyplot.savefig("tmp/"+uid_name+'_plot.png', bbox_inches="tight")


# final evaluation using best model called after completion of training
def finaleval(output_file):
    """
    output_file: The final test file to evaluate using official the SRL script
    """
    print('Running CoNLL SRL script to evaluate file: ', output_file)
    prec, recal, f1 = makedict(output_file)
    return 0, float(prec), float(recal), float(f1)




#plot('efghid_delete','../tmp/results_test_12.txt')
