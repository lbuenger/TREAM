import os


def main():

    src_path = "./results/complete_trees/normal_split/"
    #des_path = "./results/clean/trials/"
    out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    counter = 0

    for filename in os.listdir(src_path):
        if(filename.find("pkl") == -1):
            counter = counter + 1
            #print(filename)
            with open(src_path + filename, 'r') as f:
                #with open(des_path + filename, 'w+') as w:
                for idx, line in enumerate(f.readlines()):
                    print(line)
                    if(idx != 0):
                        out[idx-1] += float(line.split(",")[1][:-2])
                    #else:
                        #out = line.split()
                        #out = line.split(" ")[1] + line.split(" ")[3]
                    #w.write(out + "\n")
    for idx, sum in enumerate(out):
        out[idx] = sum / counter
    print(out)














if __name__ == '__main__':
    main()