import csv
import json 

count = 1
category2index = {}

def processFiles(filename):
    global count
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

                cate = quad[2]
                if cate not in category2index:
                    category2index[cate] = count
                    count += 1

if __name__ == "__main__":
    # processFiles("Laptop-ACOS/laptop_quad_train")
    # processFiles("Laptop-ACOS/laptop_quad_dev")
    # processFiles("Laptop-ACOS/laptop_quad_test")

    processFiles("Restaurant-ACOS/rest16_quad_train")
    processFiles("Restaurant-ACOS/rest16_quad_dev")
    processFiles("Restaurant-ACOS/rest16_quad_test")
    
    file = open("cateRes_index.json",'w')
    file.write(json.dumps(category2index))
    file.close()