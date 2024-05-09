import os


def main():

    src_path = "./results/raw/trials/"
    des_path = "./results/clean/trials/"
    out = ""

    for filename in os.listdir(src_path):
        #print(filename)
        with open(src_path + filename, 'r') as f:
            with open(des_path + filename, 'w+') as w:
                for idx, line in enumerate(f.readlines()):
                    if(idx == 0):
                        out = "BERs,Accuracy"
                    else:
                        #out = line.split()
                        out = line.split(" ")[1] + line.split(" ")[3]
                    w.write(out + "\n")













if __name__ == '__main__':
    main()