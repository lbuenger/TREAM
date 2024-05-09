import os


def main():

    #src_path = "./results/raw/nt/chidx_full/"
    #des_path = "./results/raw/nt/chidx_full_clean/"

    #src_path = "./results/raw/rsdt/fidx_full/"
    #des_path = "./results/raw/nt/fidx/"

    #src_path = "./results/raw/rsdt/fval_full/"
    #des_path = "./results/raw/nt/fval/"

    src_path = "./results/raw/rsdt/split_full/"
    des_path = "./results/raw/nt/split/"

    for filename in os.listdir(src_path):
        with open(src_path + filename, 'r') as f:
            if (filename.find("pkl") == -1 and filename.find("output") == -1 and filename.find("result") == -1):
                with open(des_path + filename, 'w+') as w:
                    for line in f.readlines():
                        w.write(line.split(",")[0] + ", " + line.split(",")[1] + "\n")













if __name__ == '__main__':
    main()