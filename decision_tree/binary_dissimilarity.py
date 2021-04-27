import pandas as pd
import numpy as np
import os

class Binary_Dissimilarity:
    def __init__(self):
        self.df = None
        self.__dice__ = {
            'm11':None,
            'm10': None,
            'm01': None,
            'm00': None,
            "index":None
        }
        pass

    def open_excel(self, dir, name, sheet_name):
        p = os.path.join(dir,name)
        df = pd.read_excel(p, sheet_name)
        self.df = df

        return self


    def dropColumns(self, column_array):
        self.df = self.df.drop(column_array, axis=1)
        return self

    def dice_similarity(self, df, col1, col2):
        '''implementation of the dice similarity metric for given col names'''

        l1 = df[col1].tolist()
        l2 = df[col2].tolist()
        #now iteratre and calculate the statistics
        m11,m10,m01,m00 = 0,0,0,0
        m11 = sum([1 for index in range(len(l1)) if l1[index] == l2[index] and l1[index] == 1])
        m10 = sum([1 for index in range(len(l1)) if l1[index] ==1 and  l2[index] == 0])
        m01 = sum([1 for index in range(len(l1)) if l1[index] == 0 and l2[index] == 1])
        m00 = sum([1 for index in range(len(l1)) if l1[index] == l2[index] and l1[index] == 0])

        self.__dice__['m11'] = m11
        self.__dice__['m10'] = m10
        self.__dice__['m01'] = m01
        self.__dice__['m00'] = m00

        try:
            self.__dice__['index'] = 2*m11/(2*m11+m01+m10)
        except:
            self.__dice__['index'] = np.inf

        return self

    def dice_similarity_matrix(self, between_self=True, label='right'):
        if(between_self):
            temp_df = self.df[self.df['direction'] == label].drop(['direction'], axis=1)
        else:
            temp_df = self.df.drop(['direction'], axis=1)
        colNames = temp_df.columns

        mat = np.zeros((len(colNames), len(colNames)))
        for col1 in range(len(colNames)):
            for col2 in range(len(colNames)):
                if(col1 == col2):
                    mat[col1, col2] = 0
                    continue
                else:
                    c1,c2 = colNames[col1], colNames[col2]
                    self.dice_similarity(temp_df, c1,c2)
                    mat[col1, col2] = self.__dice__['index']
        self.matprint(mat)

    def entropy(self, data):
        pass

    def matprint(self, mat, fmt="g"):
        col_maxes = [max([len(("{:" + fmt + "}").format(x)) for x in col]) for col in mat.T]
        for x in mat:
            for i, y in enumerate(x):
                print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
            print("")

if __name__ == "__main__":
    dissim = Binary_Dissimilarity()
    dir = r"C:\Users\paulk\OneDrive - University of Toronto\engineering_masters\thesis work\code"

    dissim.open_excel(dir, "decision_data_new.xlsx", "sys2").dropColumns(["Gradient_1", "Gradient_2"])
    dissim.dice_similarity_matrix(between_self=False)


