import numpy as np
import os
import torch

def PowerSetsRecursive2(items):
    # the power set of the empty set has one element, the empty set
    result = [[]]
    for x in items:
        result.extend([subset + [x] for subset in result])
    return result


def get_label(label_path):


    fr = open(label_path)
    labels = fr.readlines()
    results = []
    for obj in labels:
        flag = 0
        obj = obj.strip().split()
        confidence = float(obj[-1])
        x1, y1, x2, y2 = [int(float(t)) for t in obj[:-1]]


        result = [x1, y1, x2, y2, confidence]
        results.append(result)
    # if len(results) >0:
    #     results = np.vstack(results)
    # else:
    #     results = np.array(results)
    fr.close()
    return results


def nms(boxes, scores, overlap=0.3, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count



def main(result_path, output_first):

    result_names = []
    for obj in result_path :
        last_name = obj.strip().split('/')[-3:-1]
        last_name = '_'.join(last_name)
        # print last_name
        result_names.append(last_name)
    # last2_name  = result2_path.strip().split('/')[-2]


    output_path = os.path.join(output_first, '-'.join(result_names)+"_2")

    if os.path.exists(output_path) == False:
        os.makedirs(output_path)

    # label1_path = os.path.join(result_path[1])

    for i in range(0, 100):
        img_name = str(i) + '.txt'
        label_path = []
        results = []

        for index in range(len(result_path)):
            label_path.append(os.path.join(result_path[index], img_name))
            # print label_path[index]
            results.append(get_label(label_path[index]))
            # print i
            # print result_path[index]
            # print len(results[index])
            results[index] = np.vstack(results[index])
        output_label = os.path.join(output_path, img_name)
        concat_result = np.vstack(results)
        # print concat_result.shape
        box = concat_result[:, :4]
        box = torch.from_numpy(box)
        score = concat_result[:, 4]
        score = torch.from_numpy(score)
        output = torch.zeros(750, 5)
        ids, count = nms(box, score, overlap=0.5, top_k=750)

        output[:count] = torch.cat((score[ids[:count]].unsqueeze(1),
                                    box[ids[:count]]), 1)
        # print count
        # print box[ids[:count]].shape
        # print score[ids[:count]].unsqueeze(1).shape
        # print output.shape
        fp = open(output_label, 'w+')

        for j in range(len(output)):
            item = []

            item.append(str(output[j,1]))  #x1
            item.append(str(output[j,2]))  #y1
            item.append(str(output[j,3]))  #x2
            item.append(str(output[j,4]))  #y2
            item.append(str(output[j,0]))  # confidence
            fp.write(' '.join(item)+'\n')
        fp.close()



if __name__ == "__main__":
    mix_model_file = "/home/minivision/AI/minivision/learning/cv/ug2/output/mix_model.txt"
    output_first = "/home/minivision/AI/minivision/Data/ug2/mix_output/2019-05-27/1152*1792/"
    if os.path.exists(output_first) == False:
        os.makedirs(output_first)

    fr = open(mix_model_file)
    model_list =[]
    for obj in fr.readlines():
        obj = obj.strip()
        model_list.append(obj)
    print len(model_list)
    result = PowerSetsRecursive2(model_list)
    for obj in result:
        if len(obj) >= 2:
            print obj
            main(obj, output_first)
