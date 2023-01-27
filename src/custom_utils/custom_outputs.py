import os
from collections import OrderedDict


def output_save_results(output_dir, fname, annotations, influences1, influences2, influences3, claim):
    # SORT RESULTS PROPERLY
    sorted_fastif = OrderedDict({k: v for k, v in sorted(influences1.items(), key=lambda item_fastif: item_fastif[1])})
    res_fastif = []
    for i in sorted_fastif.items():
        res_fastif.append(i)
    res_fastif.reverse()

    sorted_tracin = OrderedDict({k: v for k, v in sorted(influences1.items(), key=lambda item_tracin: item_tracin[1])})
    res_tracin = []
    for i in sorted_tracin.items():

        res_tracin.append(i)

    # PRINT SUMMARY RESULTS
    print("=====FASTIF=====")
    print(res_fastif[0:4], sep='\n')
    print("=====TRACIN=====")
    print(res_tracin[0:4], sep='\n')
    print("=====KNN=====")
    print(influences2, sep='\n')

    # SAVE OUTPUT TO FILE
    with open(os.path.join(output_dir, fname + ".txt"), 'w') as fp:
        fp.write("INCLUDED ANNOTATIONS \n")
        fp.write("[")
        for item in annotations:
            fp.write("%s ," % item)

        fp.write("]")
        fp.write("FASTIF \n")
        if len(res_fastif) > 100:
            for i in res_fastif[0:99]:
                fp.write("(" + str(i[0]) + ", " + str(i[1]) + ")")
        else:
            for i in res_fastif:
                fp.write("(" + str(i[0]) + ", " + str(i[1]) + ")")
        fp.write("\n")
        fp.write("TRACIN \n")
        if len(res_tracin) > 100:
            for i in res_tracin[0:99]:
                fp.write("(" + str(i[0]) + ", " + str(i[1]) + ")")
        else:
            for i in res_tracin:
                fp.write("(" + str(i[0]) + ", " + str(i[1]) + ")")
        fp.write("\n")
        fp.write("KNN \n")
        for t in influences2:
            fp.write("(" + str(t[0]) + ", " + str(t[1]) + ")")
        fp.close()
