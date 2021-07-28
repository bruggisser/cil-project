from numpy import genfromtxt
import os


def calculate_validation_metrics():
    SUFFIX = "_validation_set.csv"
    validation_files = [f for f in os.listdir("./data/") if SUFFIX in f]
    model_performance_file = open("./data/model_performance.csv", "w")
    model_performance_file.write("model,accuracy,precision,recall,f1")

    for validation_file in validation_files:
        df = genfromtxt("./data/" + validation_file, delimiter=",", skip_header=True)
        model = validation_file[:-len(SUFFIX)]
        TP, FP, TN, FN = 4 * (0.,)
        
        for row in df:
            if(row[1] == 1 and row[2] >= 0.5):
                TP += 1
                continue
            elif(row[1] == 1 and row[2] < 0.5):
                FP += 1
                continue
            elif(row[1] == 0 and row[2] < 0.5):
                TN += 1
                continue
            elif(row[1] == 0 and row[2] >= 0.5):
                FN += 1
                continue
            else:
                raise Exception("Impossible!")

        accuracy = (TP + TN) / (TP + FP + FN + TN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (recall * precision) / (recall + precision)

        print(50 * "-")
        print(f"Statistics for {model} model")
        print(50 * "-")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1}")
        print(50 * "-")
        
        row = f"{model},{accuracy},{precision},{recall},{f1}\n"
        model_performance_file.write(row)

    model_performance_file.close()

if __name__ == "__main__":
    calculate_validation_metrics()