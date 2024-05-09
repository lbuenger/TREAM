import os


def main():

    #src_path = "./results/raw/nt/split/output.txt"
    #des_path = "./results/clean/nt/timing_sum.txt"

    #src_path = "./results/complete_trees/c_fidx/output.txt"
    #des_path = "./results/clean/ct/timing_sum.txt"

    src_path = "./results/raw/crt/split/output.txt"
    des_path = "./results/clean/crt/timing_sum.txt"

    out_building = 0
    out_evaluation = 0

    counter = 0
    file_counter = 0


    #print(filename)
    with open(src_path, 'r') as f:
        #with open(des_path + filename, 'w+') as w:
        for line in enumerate(f.readlines()):
            if(str(line).find("Building") != -1):
                out_building += round(float(str(line).split()[3][:-4]),4)
            elif(str(line).find("Evaluation") != -1):
                out_evaluation += round(float(str(line).split()[3][:-4]),4)
                file_counter += 1

    #print(counter)
    #print(file_counter)
    out_building = round(out_building / file_counter, 4)
    out_evaluation = round(out_evaluation / file_counter, 4)



    with open(des_path, 'w+') as w:
        #w.write("rsdt  ,b_time,e_time\n0.0000,\n0.0001,\n0.0010,\n0.0100,\n0.1000,\n0.2000,\n0.4000,\n0.6000,\n0.8000,\n1.0000,")
        w.write("b_time,e_time\n{:.4f}".format(out_building) + "," + "{:.4f}".format(out_evaluation))

    print(out_building)


#print("BER: {:.4f}, Accuracy: {:.4f} ({:.4f},{:.4f})".format(ber, acc_mean, acc_mean - acc_min, acc_max - acc_mean))










if __name__ == '__main__':
    main()