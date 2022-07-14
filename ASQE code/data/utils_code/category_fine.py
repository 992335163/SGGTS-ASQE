import csv
import json 

countC = 1
countF = 1
LapC2index = {}
LapF2index = {}

def processFiles(filename):
    global countC, countF
    with open(filename+".tsv",'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            instance = ""
            for piece in row:
                instance += piece + ", " 
            instance = instance[:-1].split("\t")
            sentence = instance[0]
            quads = instance[1:]
            # print(sentence, quads)
            temp = []
            for quad in quads:
                quad = quad.split()
                if quad[0][-1] == ",":
                    quad[0] = quad[0][:-1]
                if quad[-1][-1] == ",":
                    quad[-1] = quad[-1][:-1]
                if quad[-2][-1] == ",":
                    quad[-2] = quad[-2][:-1]
                assert(int(quad[1]) <= len(sentence.split()))
                assert(int(quad[-1]) <= len(sentence.split()))

                cate = quad[2].split("#")
                
                if cate[0] not in LapC2index:
                    LapC2index[cate[0]] = countC
                    countC += 1

                if cate[1] not in LapF2index:
                    LapF2index[cate[1]] = countF
                    countF += 1

if __name__ == "__main__":
    processFiles("Laptop-ACOS/laptop_quad_train")
    processFiles("Laptop-ACOS/laptop_quad_dev")
    processFiles("Laptop-ACOS/laptop_quad_test")

    file = open("LapC2index.json",'w')
    file.write(json.dumps(LapC2index))
    file.close()

    file = open("LapF2index.json",'w')
    file.write(json.dumps(LapF2index))
    file.close()