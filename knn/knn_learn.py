import numpy as np
import pandas as pd
import random
import seaborn as sns


class knn_learn:
    def __init__(self):
        pass

    def create_test_data(self, df, test_size):
        temp_df = df
        df_cols = df.columns.tolist()
        test_df = pd.DataFrame([], columns=df_cols)

        for i in range(test_size):
            r = random.randint(0, len(temp_df.iloc[:,0])-1)
            try:
                test_df.loc[len(test_df)] = temp_df.iloc[r, :]

                temp_df = temp_df.drop(r)
                temp_df.reset_index(drop=True, inplace=True)
            except:
                print(r)
                print(len(temp_df.iloc[:,0]))

        return temp_df, test_df

    def euclidian_distance(self, test_pt, neighbour_pt, dim=2):
        '''
        calculate the euclidian distance
        '''

        #data pts should be passed in as an array of pts
        distance = 0.0
        for i in range(0, dim):
            distance += (test_pt[i] - neighbour_pt[i])**2
        distance = distance**(1/2)
        return distance

    def get_neighbour(self, train, test_row, num_neighbours, dim=2):
        '''
        this function creates an array of values for all datapoints to the input pts
        :param train:
        :param test_row:
        :param num_neighbours:
        :return:
        '''

        distances = list()
        rows = len(train.iloc[:,0])
        for i in range(rows):
            #now calculate the distance between pts
            dist = self.euclidian_distance(test_row, list(train.iloc[i, 0:dim]), dim=dim)
            train_row = train.iloc[i, 0:dim]
            distances.append((list(train_row)+[i], dist))

        distances.sort(key=lambda tup : tup[1])
        neighbours = list()
        for i in range(num_neighbours):
            neighbours.append(distances[i][0])
        return neighbours

    def decision(self, train, neighbours, actual, label):
        '''grab the nearest neighbours and determine the majority class'''
        classes = dict()
        for n in neighbours:
            if(train.loc[n[-1]][label] not in classes.keys()):
                classes[train.loc[n[-1]][label]] = 0
            classes[train.loc[n[-1]][label]] += 1/len(neighbours)

        m_value = max(classes.values())
        m_keys = [k for k, v in classes.items() if v == m_value]

        return m_keys[0], m_value, actual

class test_knn:
    def __init__(self, input_data):
        self.data = input_data
        self.knn = knn_learn()

    def test_distance_method(self, data, method='euclidean'):
        if(method == 'euclidean'):
            e_distance = []
            base = data[0]
            for element in data:
                e_distance.append(self.knn.euclidian_distance(base, element, dim=6))
            self.print_test_output(method, data, e_distance)


    def print_test_output(self, method, input_data, method_return):
        if(method == 'euclidean'):
            print("Euclidean distance results: ")
            print("Pts")
            for pts in range(len(input_data)):
                print("Pts: {0}, {1} | Given distance: {2}".format(input_data[pts][0], input_data[pts][1], method_return[pts]))



if __name__=="__main__":
    """test_pts = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

    #tester = test_knn(input_data=test_pts)
    #tester.test_distance_method(test_pts)

    knn = knn_learn()

    #test neighbours
    neighbours = knn.get_neighbour(test_pts, test_pts[0], 5)
    print(neighbours[-1])"""


    file = "../new_datas_sys2.csv"
    data = pd.DataFrame(pd.read_csv(file))

    #can select according to a list of headings
    #df = data[["first_peak","second_peak","second_last_peak", "last_peak", "first_peak_2",	"second_peak_2", "second_last_peak_2", "last_peak_2",
    #           "Gradient_1", "Gradient_2","first_peak_differential","last_peak_differential", "direction"]]
    df = data[["Gradient_1", "Gradient_2","first_peak_differential","last_peak_differential", "direction"]]
    df.reset_index(drop=True, inplace=True)
    #df = df.iloc[69:, :]

    kmeans = [1,3,5,7,9,11]
    res = dict()
    for km in kmeans:
        accuracy, tot = 0.0, 0.0
        for _iter_ in range(1,101):
            if(_iter_%10 == 0):
                print(f"{_iter_}%")
            knn = knn_learn()
            train,test = knn.create_test_data(df, 20)
            for index in range(len(test.iloc[:,0])):
                neighbours = knn.get_neighbour(train, test.loc[index], km)
                classes, probability, actual = knn.decision(train, neighbours, test['direction'][index], 'direction')
                if(classes == actual):
                    accuracy += 1
                tot +=1

        res[km] = accuracy/tot * 100.0

    print(f"Tested with accuracy: {accuracy/tot * 100.0}")
    print(res)


    #with 4 features
    first = {1: 92.35, 3: 92.0, 5: 93.10, 7: 94.15, 9: 94.05, 11: 94.1}
    second = {1: 92.4, 3: 92.6, 5: 93.70, 7: 93.95, 9: 94.8, 11: 93.7}
    third = {1: 91.85, 3: 92.30, 5: 92.80, 7: 93.55, 9: 93.60, 11: 94.15}

    #with 5 features


