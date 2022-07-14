import csv
import json 

def processFiles(filename):
    count, instances = 0, []
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
                quad[0], quad[1], quad[3], quad[4], quad[5] = int(quad[0]), int(quad[1]), int(quad[3]), int(quad[4]), int(quad[5])
                temp.append(quad)
            # print(sentence, temp)
            instance = {                  
                'id': count,                               
                'sentence': sentence,                    
                'quads': temp,                     
            }
            count += 1
            instances.append(instance)
    
    file = open(filename+".json",'w')
    file.write(json.dumps(instances))
    file.close()

if __name__ == "__main__":
    processFiles("Laptop-ACOS/laptop_quad_train")
    processFiles("Laptop-ACOS/laptop_quad_dev")
    processFiles("Laptop-ACOS/laptop_quad_test")

    processFiles("Restaurant-ACOS/rest16_quad_train")
    processFiles("Restaurant-ACOS/rest16_quad_dev")
    processFiles("Restaurant-ACOS/rest16_quad_test")
   