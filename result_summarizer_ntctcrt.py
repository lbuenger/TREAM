import os


def main():

    #src_path = "./results/raw/nt/n_split/"
    #des_path = "./results/clean/nt/res_n_split.txt"

    #src_path = "./results/raw/nt/n_fval/"
    #des_path = "./results/clean/nt/res_n_fval.txt"

    #src_path = "./results/raw/nt/n_fidx/"
    #des_path = "./results/clean/nt/res_n_fidx.txt"

    #src_path = "./results/raw/nt/n_chidx/"
    #des_path = "./results/clean/nt/res_n_chidx.txt"


    #src_path = "./results/raw/ct/c_split/"
    #des_path = "./results/clean/ct/res_ct_split.txt"

    #src_path = "./results/raw/ct/c_fval/"
    #des_path = "./results/clean/ct/res_ct_fval.txt"

    #src_path = "./results/raw/ct/c_fidx/"
    #des_path = "./results/clean/ct/res_ct_fidx.txt"

    #src_path = "./results/raw/ct/n_chidx/"
    #des_path = "./results/clean/ct/res_ct_chidx.txt"


    #src_path = "./results/raw/crt/cr_split/"
    #des_path = "./results/clean/crt/res_crt_split.txt"

    #src_path = "./results/raw/crt/cr_fval/"
    #des_path = "./results/clean/crt/res_crt_fval.txt"

    src_path = "./results/raw/crt/cr_fidx/"
    des_path = "./results/clean/crt/res_crt_fidx.txt"

    #src_path = "./results/raw/nt/n_chidx/"
    #des_path = "./results/clean/nt/res_n_chidx.txt"


    out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    BERs = ["0.00001,", "0.0001,", "0.0010,", "0.0100,", "0.1000,", "0.2000,", "0.4000,", "0.6000,", "0.8000,", "1.0000,"]
    counter = 0

    for filename in os.listdir(src_path):
        if(filename.find("pkl") == -1 and filename.find("output") == -1 and filename.find("result") == -1):
            counter = counter + 1
            #print(filename)
            with open(src_path + filename, 'r') as f:
                #with open(des_path + filename, 'w+') as w:
                for idx, line in enumerate(f.readlines()):
                    #print(line)
                    if(idx != 0):
                        #print(line)
                        out[idx-1] += float(line.split(",")[1])
                    #else:
                        #out = line.split()
                        #out = line.split(" ")[1] + line.split(" ")[3]
                    #w.write(out + "\n")
    for idx, sum in enumerate(out):
        out[idx] = sum / counter
    #print(out)

    with open(des_path, 'w+') as w:
        for idx, sum in enumerate(out):
            w.write(BERs[idx] + "{:.4f}".format(sum) + "\n")














if __name__ == '__main__':
    main()