import numpy as np

class knn_learn:
    def __init__(self):
        pass

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

    def get_neighbour(self, train, test_row, num_neighbours):
        '''
        this function creates an array of values for all datapoints to the input pts
        :param train:
        :param test_row:
        :param num_neighbours:
        :return:
        '''

        distances = list()
        for train_row in train:
            #now calculate the distance between pts
            if(train_row == test_row):
                continue
            dist = self.euclidian_distance(test_row, train_row, dim=2)
            print(dist)
            distances.append((train_row, dist))

        distances.sort(key=lambda tup : tup[1])
        neighbours = list()
        for i in range(num_neighbours):
            neighbours.append(distances[i][0])
        return neighbours

class test_knn:
    def __init__(self, input_data):
        self.data = input_data
        self.knn = knn_learn()

    def test_distance_method(self, data, method='euclidean'):
        if(method == 'euclidean'):
            e_distance = []
            base = data[0]
            for element in data:
                e_distance.append(self.knn.euclidian_distance(base, element, dim=2))
            self.print_test_output(method, data, e_distance)


    def print_test_output(self, method, input_data, method_return):
        if(method == 'euclidean'):
            print("Euclidean distance results: ")
            print("Pts")
            for pts in range(len(input_data)):
                print("Pts: {0}, {1} | Given distance: {2}".format(input_data[pts][0], input_data[pts][1], method_return[pts]))



if __name__=="__main__":
    test_pts = [[2.7810836,2.550537003,0],
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
    print(neighbours[-1])