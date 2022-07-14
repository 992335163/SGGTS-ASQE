import torch 

def get_aspects(tags, length, ignore_index=-1):
    spans = []
    start = -1
    if tags[0][0] == 1 or tags[0][0] >= 3:
        spans.append([0, 0])
    for i in range(1, length):
        if tags[i][i] == ignore_index: continue
        elif tags[i][i] == 1:
            if start == -1:
                start = i
        elif tags[i][i] != 1:
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length-1])
    return spans

def get_opinions(tags, length, ignore_index=-1):
    spans = []
    start = -1
    if tags[0][0] >= 2:
        spans.append([0, 0])
    for i in range(length):
        if tags[i][i] == ignore_index: continue
        elif tags[i][i] == 2:
            if start == -1:
                start = i
        elif tags[i][i] != 2:
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length-1])
    return spans

def find_quadruplet(class_category, tags, tags_c, aspect_spans, opinion_spans):
    quadruplet, ee_find_quadruplet, ei_find_quadruplet, ie_find_quadruplet, ii_find_quadruplet = [], [], [], [], []

    for al, ar in aspect_spans:
        for pl, pr in opinion_spans:
            tag_sentiment = [0]*(6+1)
            tag_category = [0]*(class_category+1)
            for i in range(al, ar+1):
                for j in range(pl, pr+1):
                    if al < pl:
                        tag_sentiment[int(tags[i][j])] += 1
                        tag_category[int(tags_c[i][j])] += 1
                    else:
                        tag_sentiment[int(tags[j][i])] += 1
                        tag_category[int(tags_c[j][i])] += 1
            tag_sentiment, tag_category = torch.Tensor(tag_sentiment)[4:], torch.Tensor(tag_category)[1:]
            if sum(tag_sentiment) == 0 or sum(tag_category) == 0: continue
            sentiment = (torch.argmax(tag_sentiment)+1).item()
            category = (torch.argmax(tag_category)+1).item()
            quadruplet.append([al, ar, pl, pr, sentiment, category])

            if al == 0 and ar == 0 and pl == 0 and pr == 0:
                ii_find_quadruplet.append([al, ar, pl, pr, sentiment, category])
            elif al == 0 and ar == 0:
                ie_find_quadruplet.append([al, ar, pl, pr, sentiment, category])
            elif pl == 0 and pr == 0:
                ei_find_quadruplet.append([al, ar, pl, pr, sentiment, category])
            else:
                ee_find_quadruplet.append([al, ar, pl, pr, sentiment, category])
    return quadruplet, ee_find_quadruplet, ei_find_quadruplet, ie_find_quadruplet, ii_find_quadruplet


def score_uniontags(args, predicted, predicted_c, golden, golden_c, lengths, ignore_index=-1):
    assert len(predicted) == len(golden)
    assert len(predicted) == len(predicted_c)
    assert len(golden) == len(golden_c)

    golden_set, golden_eeset, golden_eiset, golden_ieset, golden_iiset = set(), set(), set(), set(), set()
    predicted_set, predicted_eeset, predicted_eiset, predicted_ieset, predicted_iiset = set(), set(), set(), set(), set()
    for i in range(len(golden)):
        golden_aspect_spans = get_aspects(golden[i], lengths[i]+1, ignore_index)
        golden_opinion_spans = get_opinions(golden[i], lengths[i]+1, ignore_index)
        golden_quadruplet, golden_EAEO, golden_EAIO, golden_IAEO, golden_IAIO = find_quadruplet(args.class_category, golden[i], golden_c[i], golden_aspect_spans, golden_opinion_spans)
        for quadruplet in golden_quadruplet:
            golden_set.add(str(i) + '-'+ '-'.join(map(str, quadruplet)))
        for quadruplet in golden_EAEO:
            golden_eeset.add(str(i) + '-'+ '-'.join(map(str, quadruplet)))
        for quadruplet in golden_EAIO:
            golden_eiset.add(str(i) + '-'+ '-'.join(map(str, quadruplet)))
        for quadruplet in golden_IAEO:
            golden_ieset.add(str(i) + '-'+ '-'.join(map(str, quadruplet)))
        for quadruplet in golden_IAIO:
            golden_iiset.add(str(i) + '-'+ '-'.join(map(str, quadruplet)))

        predicted_aspect_spans = get_aspects(predicted[i], lengths[i]+1, ignore_index)
        predicted_opinion_spans = get_opinions(predicted[i], lengths[i]+1, ignore_index)
        predicted_quadruplet, predicted_EAEO, predicted_EAIO, predicted_IAEO, predicted_IAIO = find_quadruplet(args.class_category, predicted[i], predicted_c[i], predicted_aspect_spans, predicted_opinion_spans)
        for quadruplet in predicted_quadruplet:
            predicted_set.add(str(i) + '-'+ '-'.join(map(str, quadruplet)))
        for quadruplet in predicted_EAEO:
            predicted_eeset.add(str(i) + '-'+ '-'.join(map(str, quadruplet)))
        for quadruplet in predicted_EAIO:
            predicted_eiset.add(str(i) + '-'+ '-'.join(map(str, quadruplet)))
        for quadruplet in predicted_IAEO:
            predicted_ieset.add(str(i) + '-'+ '-'.join(map(str, quadruplet)))
        for quadruplet in predicted_IAIO:
            predicted_iiset.add(str(i) + '-'+ '-'.join(map(str, quadruplet)))

    correct_num = len(golden_set & predicted_set)
    precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
    recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    ee_correct_num = len(golden_eeset & predicted_eeset)
    ee_precision = ee_correct_num / len(predicted_eeset) if len(predicted_eeset) > 0 else 0
    ee_recall = ee_correct_num / len(golden_eeset) if len(golden_eeset) > 0 else 0
    ee_f1 = 2 * ee_precision * ee_recall / (ee_precision + ee_recall) if (ee_precision + ee_recall) > 0 else 0

    ei_correct_num = len(golden_eiset & predicted_eiset)
    ei_precision = ei_correct_num / len(predicted_eiset) if len(predicted_eiset) > 0 else 0
    ei_recall = ei_correct_num / len(golden_eiset) if len(golden_eiset) > 0 else 0
    ei_f1 = 2 * ei_precision * ei_recall / (ei_precision + ei_recall) if (ei_precision + ei_recall) > 0 else 0

    ie_correct_num = len(golden_ieset & predicted_ieset)
    ie_precision = ie_correct_num / len(predicted_ieset) if len(predicted_ieset) > 0 else 0
    ie_recall = ie_correct_num / len(golden_ieset) if len(golden_ieset) > 0 else 0
    ie_f1 = 2 * ie_precision * ie_recall / (ie_precision + ie_recall) if (ie_precision + ie_recall) > 0 else 0

    ii_correct_num = len(golden_iiset & predicted_iiset)
    ii_precision = ii_correct_num / len(predicted_iiset) if len(predicted_iiset) > 0 else 0
    ii_recall = ii_correct_num / len(golden_iiset) if len(golden_iiset) > 0 else 0
    ii_f1 = 2 * ii_precision * ii_recall / (ii_precision + ii_recall) if (ii_precision + ii_recall) > 0 else 0

    return [precision, recall, f1], [ee_precision, ee_recall, ee_f1], [ei_precision, ei_recall, ei_f1], [ie_precision, ie_recall, ie_f1], [ii_precision, ii_recall, ii_f1]
