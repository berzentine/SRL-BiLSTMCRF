__author__ = 'Nidhi'
"""
 CoNLL SRL metric are based on exact span matching. This metric implements span
 based precision and recall metrics for a BIO tagging scheme. It produces P, R, F1
 per tag, and is adapted from allenAI2 implementation. It is a close proxy of the
 actual script and can be helpful to evaluate the model during training.
"""
true_positive = dict()
false_positive = dict()
false_negative = dict()


def bio_tags_to_spans(tag_seq, class_ignore):
    """
    Parameters
    ----------
    tag_seq = List(str) is the List of string tags
    class_ignore = List(str) is the class to ignore while doing metric calc


    Returns
    ---------
    spans: List(String spans) of format (label, (span_start, span_end))
    """

    spans = set()
    span_start, span_end = 0,0
    active_conll_tag = None
    for index, string_tag in enumerate(tag_seq):
        bio_tag = string_tag[0] # Should be B, I, O
        conll_tag = string_tag[2:] # Should be the actual tag Arg0 etc
        if bio_tag == 'O' or conll_tag in class_ignore:
            # The span we are looking at has ended. So handle the active tag
            if active_conll_tag:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = None
            continue
        elif bio_tag == 'B':
            # We are looking at a new span and entering here, reset indices and active span
            if active_conll_tag:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = conll_tag
            span_start = index
            span_end = index
        elif bio_tag == 'I' and conll_tag == active_conll_tag:
            # We are inside the span of what we are looking at, so only update index
            span_end+=1
        else:
            # handle ill formed spans like I tag without really having a B
            # or I is from a different coll annotation
            if active_conll_tag:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = conll_tag
            span_start = index
            span_end = index

    # handle any last tokens that would have been part of any spans
    if active_conll_tag:
        spans.add((active_conll_tag, (span_start, span_end)))
    return list(spans)

def handle_continued_span(spans):
    """
    Method to handle the C-tag for CoNLL 2012 data. Adopted from AllenAI
    Parameters
    ---------
    spans: List(String spans) of format (label, (span_start, span_end))
    Return
    ---------
    spans: List(String spans) of format (label, (span_start, span_end)) where C
    arguments are replaced with a single span
    """
    span_set  = set(spans)
    continued_labels = [label[2:] for (label, span) in span_set if label.startswith("C-")]
    for label in continued_labels:
        continued_spans = {span for span in span_set if label in span[0]}

        span_start = min(span[1][0] for span in continued_spans)
        span_end = max(span[1][1] for span in continued_spans)
        replacement_span = (label, (span_start, span_end))
        span_set.difference_update(continued_spans)
        span_set.add(replacement_span)

    return list(span_set)

def compute_metrics(true_positives, false_positives, false_negatives):
    precision = float(true_positives) / float(true_positives + false_positives + 1e-13)
    recall = float(true_positives) / float(true_positives + false_negatives + 1e-13)
    f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
    return precision, recall, f1_measure

def get_metric():
    """
    Returns
    -------
    A Dict per label containing following the span based metrics:
    precision : float
    recall : float
    f1-measure : float
    Additionally, an ``overall`` key is included, which provides the precision,
    recall and f1-measure for all spans.
    """
    all_tags = set()
    all_tags.update(true_positive.keys())
    all_tags.update(false_positive.keys())
    all_tags.update(false_negative.keys())
    all_metrics = {}
    for tag in all_tags:
        precision, recall, f1_measure = compute_metrics(true_positive[tag],
                                                              false_positive[tag],
                                                              false_negative[tag])
        precision_key = "precision" + "-" + tag
        recall_key = "recall" + "-" + tag
        f1_key = "f1-measure" + "-" + tag
        all_metrics[precision_key] = precision
        all_metrics[recall_key] = recall
        all_metrics[f1_key] = f1_measure

    # Compute the precision, recall and f1 for all spans jointly.
    precision, recall, f1_measure = compute_metrics(sum(true_positive.values()),
                                                          sum(false_positive.values()),
                                                          sum(false_negative.values()))
    all_metrics["precision-overall"] = precision
    all_metrics["recall-overall"] = recall
    all_metrics["f1-measure-overall"] = f1_measure


    return all_metrics

def proxy_eval(pred_labels, gold_labels, pred_labels_string, gold_labels_string, ignore_class, all_labels):
    """
    Parameters
    ----------
    pred_labels = List(seq_length X num_clases)
    gold_labels = List(seq_length)
    pred_labels_string = List() each element is a list of predictions of the ith sample
    gold_labels_string = List() each element is a list of predictions of the ith sample
    ignore_class = List() each element is the string of the tag you wish to ignore in
                    metric calculation
    Give in the entire dev set data to this function
    """
    # TODO: the dictionaries above should be 0 intialized with all labels
    #num_classes = pred_labels[0].size(1)
    for i in all_labels:
        true_positive[i]=0
        false_positive[i]=0
        false_negative[i]=0

    samples = len(pred_labels_string)
    for i in range(samples):
        #pred_sample = pred_labels[i]
        #gold_sample = gold_labels[i]
        #length = pred_labels[i].size(0)
        pred_string_sample, gold_string_sample = pred_labels_string[i], gold_labels_string[i]

        predicted_spans = bio_tags_to_spans(pred_string_sample, ignore_class)
        gold_spans = bio_tags_to_spans(gold_string_sample, ignore_class)

        # handle continued spans for CoNLL 2012 format
        predicted_spans = handle_continued_span(predicted_spans)
        gold_spans = handle_continued_span(gold_spans)

        for span in predicted_spans:
            if span in gold_spans:
                true_positive[span[0]]+=1
                gold_spans.remove(span)
            else:
                false_positive[span[0]]+=1
        for span in gold_spans:
            false_negative[span[0]]+=1

    # Handle metric calculation
    return get_metric()
