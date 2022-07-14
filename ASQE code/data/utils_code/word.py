import csv
import json 

count = 2
word2index = {}

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
            words = sentence.split()
            for word in words:
                if word not in word2index:
                    word2index[word] = count
                    count += 1
    

if __name__ == "__main__":
    processFiles("Laptop-ACOS/laptop_quad_train")
    processFiles("Laptop-ACOS/laptop_quad_dev")
    processFiles("Laptop-ACOS/laptop_quad_test")

    processFiles("Restaurant-ACOS/rest16_quad_train")
    processFiles("Restaurant-ACOS/rest16_quad_dev")
    processFiles("Restaurant-ACOS/rest16_quad_test")

    file = open("word_index.json",'w')
    file.write(json.dumps(word2index))
    file.close()
   