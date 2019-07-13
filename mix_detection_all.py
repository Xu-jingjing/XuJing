# -*- encoding:utf-8 -*-
import numpy as np
from nms import py_nms
import os
import time


def IOU(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2): #计算IOU
    sa = abs((ax2-ax1)*(ay2-ay1))
    sb = abs((bx2-bx1)*(by2-by1))
    x1,y1 = max(ax1,bx1),max(ay1,by1)
    x2,y2 = min(ax2,bx2),min(ay2,by2)
    w  = x2 - x1
    h  = y2 - y1
    if w<0 or h<0:
        return 0.0
    else:
        return 1.0*w*h/(sa+sb-w*h)

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
        x1, y1, x2, y2 = [int(t) for t in obj[:-1]]
        result = [x1, y1, x2, y2, confidence, flag]
        if confidence >= 0.1:
            results.append(result)
    # if len(results) >0:
    #     results = np.vstack(results)
    # else:
    #     results = np.array(results)
    return results



def main(result_path, output_first):

    result_names = []
    num_model = len(result_path)
    print num_model
    for obj in result_path :
        last_name = obj.strip().split('/')[-1]
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
            if len(results[index]) == 0:
                results[index] = np.array(results[index])
            else:
                results[index] = np.vstack(results[index])
        output_label = os.path.join(output_path, img_name)
        # result1 = get_label(label1_path)
        # result2 = get_label(label2_path)
        fp = open(output_label, 'w+')
        # result1 = np.vstack(result1)
        # result2 = np.vstack(result2)
        result1 = results[0]
        for m in range(1, len(results)):
            result2 = results[m]
            for j in range(len(result1)):
                ax1, ay1, ax2, ay2 = result1[j, :4]



                for k in range(len(result2)):
                    if result2[k, 5] >0 :
                        continue
                    bx1 = result2[k, 0]
                    by1 = result2[k, 1]
                    bx2 = result2[k, 2]
                    by2 = result2[k, 3]
                    iou_value = IOU(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)
                    if iou_value > 0.5:
                        # if result1[j, -1] + result2[k, -1] > 1:
                        #     result1[j, -1] = 1
                        # else:

                        if result1[j, 4] < result2[k, 4]:

                            result1[j,:4] = result2[k, :4]


                        result1[j, 4] = (result1[j, 4] + result2[k, 4])
                        result1[j, 5] += 1
                        result2[k, 5] += 1
                        break

            if len(result1) == 0:
                print '1'
                result1 = result2.copy()
            for k in range(len(result2)):
                if result2[k, 5] ==0 and result2.shape[0] == 6:

                    result1= np.vstack((result1,result2[k]))


        for j in range(len(result1)):
            item = []
            # item.append("person")
            # if result1[j, 5] != 0:
            #     item.append(str(result1[j, 4] / result1[j, 5]))
            # else:

            item.append(str(result1[j,0]))  #x1
            item.append(str(result1[j,1]))  #y1
            item.append(str(result1[j,2]))  #x2
            item.append(str(result1[j,3]))  #y2
            item.append(str(result1[j, 4] / num_model))  # confidence
            fp.write(' '.join(item)+'\n')
        fp.close()
        # break



if __name__ == "__main__":
    mix_model_file = "/home/minivision/AI/minivision/learning/cv/ug2/output/mix_model.txt"
    output_first = "/home/minivision/AI/minivision/Data/ug2/mix_output/2019-05-15/"
    fr = open(mix_model_file)
    model_list =[]
    for obj in fr.readlines():
        obj = obj.strip()
        model_list.append(obj)
    # print model_list
    result = PowerSetsRecursive2(model_list)
    for obj in result:
        if len(obj) >= 2:
            print obj
            print time.ctime()
            main(obj, output_first)
            # break
            print time.ctime()
