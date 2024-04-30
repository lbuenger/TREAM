import os


def main():

    #src_path = "./results/raw/rsdt/resilience_split/"
    #des_path = "./results/clean/rsdt/resilience_split/"

    #src_path = "./results/raw/rsdt/fval_half/"
    #des_path = "./results/clean/rsdt/fval_half/"

    src_path = "./results/raw/rsdt/fidx_half/"
    des_path = "./results/clean/rsdt/fidx_half/"

    #src_path = "./results/raw/rsdt/chidx_half/"
    #des_path = "./results/clean/rsdt/chidx_half/"
    out = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    counter = 0

    for filename in os.listdir(src_path):
        if(filename.find("pkl") == -1 and filename.find("output") == -1 and filename.find("results") == -1):
            counter = counter + 1
            #print(filename)
            with open(src_path + filename, 'r') as f:
                #with open(des_path + filename, 'w+') as w:
                for idx, line in enumerate(f.readlines()):
                    #print(line)
                    if(idx != 0):
                        for idxy, num in enumerate(line.split(",")):
                            out[idx-1][idxy] = out[idx-1][idxy] + float(num)



                        #out[idx-1] += float(line.split(",")[1][:-2])
                    #else:
                        #out = line.split()
                        #out = line.split(" ")[1] + line.split(" ")[3]
                    #w.write(out + "\n")
    for idx, line in enumerate(out):
        for idxy, sum in enumerate(line):
            out[idx][idxy] = round(sum / counter, 4)


    with open(des_path + "sum.txt", 'w+') as w:
        w.write("BERs,  rsdt0, rsdt5, rsdt10,rsdt15,rsdt20,rsdt25,rsdt30,rsdt35,rsdt40,rsdt45,rsdt50\n")
        for line in out:
            for idx, num in enumerate(line):
                w.write("{:.4f}".format(num))
                if(idx < len(line) - 1):
                    w.write(",")
            w.write("\n")

    print(out)


#print("BER: {:.4f}, Accuracy: {:.4f} ({:.4f},{:.4f})".format(ber, acc_mean, acc_mean - acc_min, acc_max - acc_mean))










if __name__ == '__main__':
    main()