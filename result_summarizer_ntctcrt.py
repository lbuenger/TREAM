import os


def main():

    #src_path = "./results/raw/nt/split/"
    #des_path = "./results/clean/nt/res_n_split.txt"

    #src_path = "./results/raw/nt/fval/"
    #des_path = "./results/clean/nt/res_n_fval.txt"

    #src_path = "./results/raw/nt/fidx/"
    #des_path = "./results/clean/nt/res_n_fidx.txt"

    #src_path = "./results/raw/nt/chidx/"
    #des_path = "./results/clean/nt/res_n_chidx.txt"


    #src_path = "./results/raw/ct/split/"
    #des_path = "./results/clean/ct/res_ct_split.txt"

    #src_path = "./results/raw/ct/fval/"
    #des_path = "./results/clean/ct/res_ct_fval.txt"

    #src_path = "./results/raw/ct/fidx/"
    #des_path = "./results/clean/ct/res_ct_fidx.txt"

    #src_path = "./results/raw/ct/chidx/"
    #des_path = "./results/clean/ct/res_ct_chidx.txt"


    #src_path = "./results/raw/crt/split/"
    #des_path = "./results/clean/crt/res_crt_split.txt"

    #src_path = "./results/raw/crt/fval/"
    #des_path = "./results/clean/crt/res_crt_fval.txt"

    #src_path = "./results/raw/crt/fidx/"
    #des_path = "./results/clean/crt/res_crt_fidx.txt"

    #src_path = "./results/raw/nt/chidx/"
    #des_path = "./results/clean/nt/res_n_chidx.txt"



    #DT
    #src_path = "./results/raw/crt/split/"
    #des_path = "./results/clean/crt/res_crt_split_DT.txt"

    #src_path = "./results/raw/crt/fval/"
    #des_path = "./results/clean/crt/res_crt_fval_DT.txt"

    #src_path = "./results/raw/crt/fidx/"
    #des_path = "./results/clean/crt/res_crt_fidx_DT.txt"


    #src_path = "./results/raw/nt/split/"
    #des_path = "./results/clean/crt/res_nt_split_DT.txt"

    #src_path = "./results/raw/nt/fval/"
    #des_path = "./results/clean/crt/res_nt_fval_DT.txt"

    #src_path = "./results/raw/nt/fidx/"
    #des_path = "./results/clean/crt/res_nt_fidx_DT.txt"

    #src_path = "./results/raw/nt/chidx/"
    #des_path = "./results/clean/crt/res_nt_chidx_DT.txt"



    #RF
    #src_path = "./results/raw/crt/split/"
    #des_path = "./results/clean/crt/res_crt_split_RF.txt"

    #src_path = "./results/raw/crt/fval/"
    #des_path = "./results/clean/crt/res_crt_fval_RF.txt"

    #src_path = "./results/raw/crt/fidx/"
    #des_path = "./results/clean/crt/res_crt_fidx_RF.txt"


    #src_path = "./results/raw/nt/split/"
    #des_path = "./results/clean/crt/res_nt_split_RF.txt"

    #src_path = "./results/raw/nt/fval/"
    #des_path = "./results/clean/crt/res_nt_fval_RF.txt"

    #src_path = "./results/raw/nt/fidx/"
    #des_path = "./results/clean/crt/res_nt_fidx_RF.txt"

    src_path = "./results/raw/nt/chidx/"
    des_path = "./results/clean/crt/res_nt_chidx_RF.txt"


    out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    BERs = ["0.00001,", "0.0001,", "0.0010,", "0.0100,", "0.1000,", "0.2000,", "0.4000,", "0.6000,", "0.8000,", "1.0000,"]
    counter = 0

    for filename in os.listdir(src_path):
        if(filename.find("pkl") == -1 and filename.find("output") == -1 and filename.find("result") == -1
                #and (filename.find("_T5") == -1 and filename.find("_T10") == -1)
                and (filename.find("_T5") != -1 or filename.find("_T10") != -1)
        ):
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
    print(counter)

    with open(des_path, 'w+') as w:
        for idx, sum in enumerate(out):
            w.write(BERs[idx] + "{:.4f}".format(sum) + "\n")














if __name__ == '__main__':
    main()