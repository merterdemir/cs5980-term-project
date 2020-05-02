import pandas as pd

def readFile():
    fileName = input("Enter the file name + extension to open: ")
    col_list = ["prediction0","prediction1","prediction2"]
    df = pd.read_csv(fileName,usecols = col_list)
    
    for i in range(0,3):
        p = df["prediction"+str(i)]
        output = open("prediction"+str(i)+".txt","w")
        for r in p:
            output.write(str(r)+"\n")
        output.close()
        
def main():
    readFile()
    
if __name__ == '__main__':
    main()
    
