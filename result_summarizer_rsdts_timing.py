import os


def main():

    src_path = "./results/output.txt"
    des_path = "./results/timing_sum_half.txt"

    out_building = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    out_evaluation = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    counter = 0
    file_counter = 0


    #print(filename)
    with open(src_path, 'r') as f:
        #with open(des_path + filename, 'w+') as w:
        for line in enumerate(f.readlines()):
            word = str(line).split()[1]
            #print(word[1])
            if word[2]=="u":
                print(line)
                out_building[counter] += round(float(str(line).split()[3][:-4]),4)
            elif word[2]=="v":
                out_evaluation[counter] += round(float(str(line).split()[3][:-4]),4)
                counter += 1
                #if counter == 11:
                #    counter = 0
            elif word[1] == "/":
                file_counter += 1
                counter = 0

                #out[idx-1] += float(line.split(",")[1][:-2])
            #else:
                #out = line.split()
                #out = line.split(" ")[1] + line.split(" ")[3]
            #w.write(out + "\n")
    print(counter)
    print(file_counter)
    for idx, sum in enumerate(out_building):
        out_building[idx] = round(out_building[idx] / file_counter, 4)
        out_evaluation[idx] = round(out_evaluation[idx] / file_counter, 4)



    with open(des_path, 'w+') as w:
        #w.write("rsdt  ,b_time,e_time\n0.0000,\n0.0001,\n0.0010,\n0.0100,\n0.1000,\n0.2000,\n0.4000,\n0.6000,\n0.8000,\n1.0000,")
        w.write("rsdt  ,b_time,e_time\n")
        for idx, num in enumerate(out_building):
            w.write("{:.4f}".format(out_building[idx]) + "," + "{:.4f}".format(out_evaluation[idx]) + "\n")

    #print(out)


#print("BER: {:.4f}, Accuracy: {:.4f} ({:.4f},{:.4f})".format(ber, acc_mean, acc_mean - acc_min, acc_max - acc_mean))










if __name__ == '__main__':
    main()