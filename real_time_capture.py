import serial
import datetime
import struct
import time
import statistics
import csv
import scipy.stats
import pandas as pd
import os

import mcp_3008_driver as mcp

import matplotlib.pyplot as plt
from decision_tree import impurity

class serialCapture:
    def __init__(self, port, baudrate, timeout=4, data_name = None, low_delay=15,
                 peak_method='slope', thresh_holds = [2.25, 2.75]):
        
        

        self.data_name = data_name
        self.rawData = bytearray(4)
        self.csvData = [[], [], []]

        self.filter_length = 21
        self.buffer_length = 16
        self.low_delay = low_delay
        self.thresh_holds = thresh_holds

        self.peak_method = peak_method

        self.csv_write = []

        try:
            if(os.name == 'nt'):
                self.handle = serial.Serial(port, baudrate, timeout=timeout)
                print("Connected")

                df = pd.read_excel("./decision_data.xlsx", "data_no_higher_order")
                # df = df.drop(columns=['figure'])

                df_mat = df.values.tolist()
                """self.tree = impurity.build_tree(df_mat)
                impurity.print_tree(self.tree)"""

            elif(os.name == 'posix'):
                print("connecting to ADC")
                self.mcp = mcp.mcp_external()
                df = pd.read_excel("./decision_data.xlsx", "schema_2")
                # df = df.drop(columns=['figure'])

                df_mat = df.values.tolist()
                self.tree = impurity.build_tree(df_mat)
                impurity.print_tree(self.tree)
                print("starting capture")
            
        except Exception as e:
            print(e)

    def printCollectedData(self, save_data, save_plot, sample_points = 400):
        print('sleeping for test')
        time.sleep(5)
        if(os.name == "nt"):
            self.handle.reset_input_buffer()

        upper = self.thresh_holds[1]
        lower = self.thresh_holds[0]

        var_limit = 0.00750

        tik = time.time()
        
        avg_buffer = []
        data_buffer = []
        trig_tracker = []
        trig_time = []
        slope_pts = []
        slope_time = []

        total_slope = []
        total_slope_time = []

        variance_pt = []
        variance_time_pt = []

        #these are the time points
        max_points = []
        min_points = []
        max_points_2 = []
        min_points_2 = []

        max_time = []
        min_time = []
        max_time_2 = []
        min_time_2 = []

        tiktik = time.time()

        _event = False
        window = []
        time_stamps_windows = []

        first_peak = None
        last_peak = None
        second_peak = None
        first_peak_2 = None
        last_peak_2 = None
        second_peak_2 = None

        start_index = None
        stop_index = None
        start_index_2 = None
        stop_index_2 = None

        tracked_signal_marks = []

        decisions = []

        slope_1 = None

        between_peak_time = []
        prev_peak = None

        _min_peak = 2.5
        _max_peak = 2.5

        counter = 5
        print("start")
        while(True):
            time.sleep(0.035)
            if(os.name == "nt"):
                self.handle.readinto(self.rawData)
            #else:
                #print("{}".format(self.mcp.read_IO(0)/65355*5.2))
                #self.csvData[1].append(float(self.mcp.read_IO(0)/65355*5.2))
                #self.csvData[2].append(float(self.mcp.read_IO(1)/65355*5.2))
            #print(self.rawData)

            #implement the moving average filter
            if(len(avg_buffer) < self.filter_length):
                t=time.time() - tik
                self.csvData[0].append(t)
                if(os.name == 'nt'):
                    self.csvData[1].append(struct.unpack('f', self.rawData)[0])
                elif(os.name == 'posix'):
                    self.csvData[1].append(float(self.mcp.read_IO(0)/65355*5))
                    self.csvData[2].append(float(self.mcp.read_IO(1)/65355*5))
            else:
                self.csvData[0].append(time.time() - tik)
                if(os.name == 'nt'):
                    data = struct.unpack('f', self.rawData)[0]/self.filter_length
                elif(os.name == 'posix'):
                    data = float(self.mcp.read_IO(0)/65355*5)
                    data1 = float(self.mcp.read_IO(1)/65355*5)
                    #print("DATA: {}".format(data/65355*5.2))
                #data = struct.unpack('f', self.rawData)[0]/self.filter_length
                for i in range(self.filter_length-1):
                    data += self.csvData[1][-(i+1)]
                    data1 += self.csvData[2][-(i+1)]
                self.csvData[1].append(data)
                self.csvData[2].append(data1)


            if(len(data_buffer) < self.buffer_length):
                data_buffer.append(self.csvData[1][-1])
            else:
                data_buffer = data_buffer[1:] + [self.csvData[1][-1]]

                var_mean = data_buffer[0]
                Sk = 0
                for k in range(1,len(data_buffer)):
                    var_mean_1 = var_mean + (1/self.buffer_length)*(data_buffer[k] - var_mean)
                    Sk_1 = Sk + (data_buffer[k] - var_mean_1)*(data_buffer[k] - var_mean)

                    #update recursive variables
                    var_mean = var_mean_1
                    Sk = Sk_1

                #set a flag if its triggered - variance window
                variance_pt.append(Sk/self.buffer_length)
                variance_time_pt.append(t)
                if(Sk/self.buffer_length < var_limit):
                    counter += 1
                    if(counter >= self.low_delay):
                        _event = False

                        trig_tracker.append(0)
                        #slope = (self.csvData[1][-1] - self.csvData[1][-2]) / (
                        #        self.csvData[0][-1] - self.csvData[0][-2])
                        total_slope.append(1)
                        total_slope_time.append(t)
                        slope_1 = 1



                        if(stop_index != None and start_index != None and first_peak != None and last_peak != None):
                            if(stop_index - start_index > 1.25):
                                '''peaks - 0, below 2.5. 1, above 2.5'''
                                '''gradient - 0, negative'''

                                temp = [0, 0, 0, 0, 0, 0]
                                try:
                                    gradient = ((last_peak - 2.5) - (first_peak - 2.5))/(stop_index - start_index)
                                    gradient_2 = ((last_peak_2 - 2.5) - (first_peak_2 - 2.5))/(stop_index_2 - start_index_2)
                                except TypeError:
                                    fig, axs = plt.subplots(1)

                                    fig.suptitle("Recorded data")
                                    print(len(self.csvData[0]), len(self.csvData[1]))
                                    axs.scatter(self.csvData[0], self.csvData[1])
                                    if(self.peak_method == 'slope'):
                                        axs.scatter(slope_time, slope_pts)
                                    elif(self.peak_method == 'peak_detection'):
                                        axs.scatter(max_time, max_points)
                                        axs.scatter(min_time, min_points)
                                    axs.axhline(y=lower, c='red')
                                    axs.axhline(y=upper, c='red')
                                    axs.set_xlim(self.csvData[0][0], self.csvData[0][-1])
                                    axs.set_xlabel("time [s]")
                                    axs.set_ylabel("Voltage [V]")
                                    
                                    return
                                except ZeroDivisionError:
                                    fig, axs = plt.subplots(1)

                                    fig.suptitle("Recorded data")
                                    print(len(self.csvData[0]), len(self.csvData[1]))
                                    print(len(self.csvData[0]), len(self.csvData[1]))
                                    axs.scatter(self.csvData[0], self.csvData[1])
                                    if(self.peak_method == 'slope'):
                                        axs.scatter(slope_time, slope_pts)
                                    elif(self.peak_method == 'peak_detection'):
                                        axs.scatter(max_time, max_points)
                                        axs.scatter(min_time, min_points)
                                    axs.axhline(y=lower, c='red')
                                    axs.axhline(y=upper, c='red')
                                    axs.set_xlim(self.csvData[0][0], self.csvData[0][-1])
                                    axs.set_xlabel("time [s]")
                                    axs.set_ylabel("Voltage [V]")
                                    
                                    return
                                except Exception:
                                    fig, axs = plt.subplots(1)

                                    fig.suptitle("Recorded data")
                                    print(len(self.csvData[0]), len(self.csvData[1]))
                                    print(len(self.csvData[0]), len(self.csvData[1]))
                                    axs.scatter(self.csvData[0], self.csvData[1])
                                    if(self.peak_method == 'slope'):
                                        axs.scatter(slope_time, slope_pts)
                                    elif(self.peak_method == 'peak_detection'):
                                        axs.scatter(max_time, max_points)
                                        axs.scatter(min_time, min_points)
                                    axs.axhline(y=lower, c='red')
                                    axs.axhline(y=upper, c='red')
                                    axs.set_xlim(self.csvData[0][0], self.csvData[0][-1])
                                    axs.set_xlabel("time [s]")
                                    axs.set_ylabel("Voltage [V]")
                                    
                                    return
                                

                                temp[0] = 1 if first_peak >= 2.5 else 0
                                temp[1] = 1 if last_peak >= 2.5 else 0
                                #temp[2] = 1 if gradient >= 0 else 0
                                temp[2] = 1 if first_peak_2 >= 2.5 else 0
                                temp[3] = 1 if last_peak_2 >= 2.5 else 0
                                temp[4] = 1 if second_peak >= 2.5 else 0
                                temp[5] = 1 if second_peak_2 >= 2.5 else 0

                                #time_period = end_time - start_time
                                """s_index = between_peak_time.index(start_time)
                                e_index = between_peak_time.index(end_time)
                                mean_pk2pk = statistics.mean(between_peak_time[s_index:e_index+1])"""

                                s_index = self.csvData[0].index(start_index)
                                e_index = self.csvData[0].index(stop_index)

                                _mean = statistics.mean(self.csvData[1])
                                variance = statistics.variance(self.csvData[1])
                                skewness = scipy.stats.skew(self.csvData[1])
                                
                                variance_2 = statistics.variance(self.csvData[2])
                                skewness_2 = scipy.stats.skew(self.csvData[2])

                                print("#########################Getting features#######################################")
                                print("")
                                print("")

                                #self.csv_write.append(start_index, stop_index, first_peak, last_peak, gradient, _max_peak, _min_peak, start_time, end_time, time_period)
                                tracked_signal_marks.append(temp)
                                #self.csv_write.append([temp[0], temp[1], gradient, temp[2], temp[3], gradient_2, _mean, variance, skewness, variance_2, skewness_2])

                                #_classify = impurity.classify(
                                #    [temp[0], temp[1], gradient, temp[2], temp[3], gradient_2, temp[4], temp[5], variance, skewness, variance_2, skewness_2], self.tree)
                                '''_classify = impurity.classify(
                                    [first_peak, last_peak, first_peak_2, last_peak_2, second_peak, second_peak_2, gradient, gradient_2], self.tree)'''
                                _classify = impurity.classify(
                                    [temp[0], temp[1], gradient, temp[2], temp[3], gradient_2, temp[4], temp[5]], self.tree)
                                max_guess = 0
                                max_class = None
                                for _class_ in _classify:
                                    if (_classify[_class_] > max_guess):
                                        max_class, max_guess = _class_, _classify[_class_]
                                print("Predicted: {}".format(max_class))
                                self.csv_write.append([first_peak, last_peak, first_peak_2, last_peak_2, second_peak, second_peak_2, temp[0], temp[1], gradient, temp[2], temp[3], gradient_2, temp[4], temp[5], variance, skewness, variance_2, skewness_2, max_class])
                                
                        start_index, stop_index, first_peak, first_peak_2, second_peak, second_peak_2, last_peak, last_peak_2, end_time, _max_peak, _min_peak = \
                            None, None, None, None, None, None, None, None, None, 2.5, 2.5


                    else:
                        trig_tracker.append(1)

                        if(self.peak_method == 'slope'):
                            slope = (self.csvData[1][-1] - self.csvData[1][-2]) / (
                                        self.csvData[0][-1] - self.csvData[0][-2])

                            total_slope.append(slope)
                            total_slope_time.append(t)

                            slope_1 = slope
                        elif(self.peak_method == 'peak_detection'):
                            # simple implementation of the peak detection algorithm

                            # find max
                            if(self.csvData[1][-2] >= upper):
                                if (self.csvData[1][-1] < self.csvData[1][-2] and self.csvData[1][-3] < self.csvData[1][-2]):
                                    max_points.append(self.csvData[1][-2])
                                    max_time.append(self.csvData[0][-2])

                                    if(len(between_peak_time) == 0):
                                        between_peak_time.append(self.csvData[0][-2])
                                    elif(prev_peak != 1):
                                        between_peak_time.append(self.csvData[0][-2])
                                    prev_peak = 1

                                    if(self.csvData[1][-2] > _max_peak):
                                        _max_peak = self.csvData[1][-2]


                                elif (self.csvData[1][-1] <= self.csvData[1][-2] and self.csvData[1][-3] < self.csvData[1][-2]):
                                    max_points.append(self.csvData[1][-2])
                                    max_time.append(self.csvData[0][-2])
                                    if (len(between_peak_time) == 0):
                                        between_peak_time.append(self.csvData[0][-2])
                                    elif (prev_peak != 1):
                                        between_peak_time.append(self.csvData[0][-2])

                                    if (self.csvData[1][-2] > _max_peak):
                                        _max_peak = self.csvData[1][-2]
                                    prev_peak = 1
                                elif (self.csvData[1][-1] < self.csvData[1][-2] and self.csvData[1][-3] <= self.csvData[1][-2]):
                                    max_points.append(self.csvData[1][-2])
                                    max_time.append(self.csvData[0][-2])

                                    if (len(between_peak_time) == 0):
                                        between_peak_time.append(self.csvData[0][-2])
                                    elif (prev_peak != 1):
                                        between_peak_time.append(self.csvData[0][-2])
                                    if (self.csvData[1][-2] > _max_peak):
                                        _max_peak = self.csvData[1][-2]
                                    prev_peak = 1

                            if (os.name == "posix"):
                                if (self.csvData[2][-2] >= upper):
                                    if (self.csvData[2][-1] < self.csvData[2][-2] and self.csvData[2][-3] < self.csvData[2][
                                        -2]):
                                        max_points_2.append(self.csvData[2][-2])
                                        max_time_2.append(self.csvData[0][-2])

                                        prev_peak = 1
                                    elif (self.csvData[2][-1] <= self.csvData[2][-2] and self.csvData[2][-3] <
                                          self.csvData[2][
                                              -2]):
                                        max_points_2.append(self.csvData[2][-2])
                                        max_time_2.append(self.csvData[0][-2])

                                        prev_peak = 1
                                    elif (self.csvData[2][-1] < self.csvData[2][-2] and self.csvData[2][-3] <=
                                          self.csvData[2][
                                              -2]):
                                        max_points_2.append(self.csvData[1][-2])
                                        max_time_2.append(self.csvData[0][-2])

                                        prev_peak = 1

                            # now for the mins
                            if(self.csvData[1][-2] <= lower):
                                if (self.csvData[1][-1] > self.csvData[1][-2] and self.csvData[1][-3] > self.csvData[1][-2]):
                                    min_points.append(self.csvData[1][-2])
                                    min_time.append(self.csvData[0][-2])

                                    if (len(between_peak_time) == 0):
                                        between_peak_time.append(self.csvData[0][-2])
                                    elif (prev_peak != 0):
                                        between_peak_time.append(self.csvData[0][-2])

                                    if(self.csvData[1][-2] < _min_peak):
                                        _min_peak = self.csvData[1][-2]
                                    prev_peak = 0


                                elif (self.csvData[1][-1] >= self.csvData[1][-2] and self.csvData[1][-3] > self.csvData[1][-2]):
                                    min_points.append(self.csvData[1][-2])
                                    min_time.append(self.csvData[0][-2])

                                    if (len(between_peak_time) == 0):
                                        between_peak_time.append(self.csvData[0][-2])
                                    elif (prev_peak != 0):
                                        between_peak_time.append(self.csvData[0][-2])

                                    if (self.csvData[1][-2] < _min_peak):
                                        _min_peak = self.csvData[1][-2]
                                    prev_peak = 0


                                elif (self.csvData[1][-1] > self.csvData[1][-2] and self.csvData[1][-3] >= self.csvData[1][-2]):
                                    min_points.append(self.csvData[1][-2])
                                    min_time.append(self.csvData[0][-2])
                                    if (len(between_peak_time) == 0):
                                        between_peak_time.append(self.csvData[0][-2])
                                    elif (prev_peak != 0):
                                        between_peak_time.append(self.csvData[0][-2])

                                    if (self.csvData[1][-2] < _min_peak):
                                        _min_peak = self.csvData[1][-2]
                                    prev_peak = 0

                            if(os.name == "posix"):
                                # min peaks 2
                                if (self.csvData[2][-2] <= lower):
                                    if (self.csvData[2][-1] > self.csvData[2][-2] and self.csvData[2][-3] >
                                            self.csvData[2][
                                                -2]):
                                        min_points_2.append(self.csvData[2][-2])
                                        min_time_2.append(self.csvData[0][-2])

                                        prev_peak = 0
                                    elif (self.csvData[2][-1] >= self.csvData[2][-2] and self.csvData[2][-3] >
                                          self.csvData[2][
                                              -2]):
                                        min_points_2.append(self.csvData[2][-2])
                                        min_time_2.append(self.csvData[0][-2])

                                        prev_peak = 0
                                    elif (self.csvData[2][-1] > self.csvData[2][-2] and self.csvData[2][-3] >=
                                          self.csvData[2][
                                              -2]):
                                        min_points_2.append(self.csvData[2][-2])
                                        min_time_2.append(self.csvData[0][-2])

                                        prev_peak = 0
                else:
                    _event = True
                    window.append(t)
                    trig_tracker.append(1)
                    counter = 0

                    if(self.peak_method == 'slope'):
                        #flow for slope
                        slope = (self.csvData[1][-1] - self.csvData[1][-2]) / (self.csvData[0][-1] - self.csvData[0][-2])

                        total_slope.append(slope)
                        total_slope_time.append(t)

                        if(slope_1 != None):
                            ##check slopes
                            if(slope <= 0 and slope_1 >= 0):
                                '''transition to downward slope'''
                                if(self.csvData[1][-2] >= upper or self.csvData[1][-2] <= lower):
                                    if (len(slope_pts) == 0):
                                        slope_pts.append(self.csvData[1][-2])
                                        slope_time.append(self.csvData[0][-2])
                                    elif (abs(self.csvData[1][-2] - slope_pts[-1]) > 0.5):
                                        slope_pts.append(self.csvData[1][-2])
                                        slope_time.append(self.csvData[0][-2])
                            elif(slope >= 0 and slope_1 <= 0):
                                if((self.csvData[1][-2] >= upper or self.csvData[1][-2] <= lower)):
                                    if(len(slope_pts) == 0):
                                        slope_pts.append(self.csvData[1][-2])
                                        slope_time.append(self.csvData[0][-2])
                                    elif(abs(self.csvData[1][-2] - slope_pts[-1]) > 0.5):
                                        slope_pts.append(self.csvData[1][-2])
                                        slope_time.append(self.csvData[0][-2])
                            slope_1 = slope
                            last_peak = self.csvData[1][-2]
                            stop_index = t
                        else:
                            slope_1 = slope
                            if(first_peak == None):
                                first_peak = self.csvData[1][-2]
                                start_index = t
                                start_time = self.csvData[0][-2]

                            last_peak = self.csvData[1][-2]
                            stop_index = t
                            end_time = self.csvData[0][-2]


                    elif(self.peak_method == 'peak_detection'):
                        #simple implementation of the peak detection algorithm

                        #find max _peak_1
                        if(self.csvData[1][-2] >= upper):
                            if(self.csvData[1][-1] < self.csvData[1][-2] and self.csvData[1][-3] < self.csvData[1][-2]):
                                max_points.append(self.csvData[1][-2])
                                max_time.append(self.csvData[0][-2])
                                if(first_peak != None and second_peak == None):
                                    second_peak = self.csvData[1][-2]
                                if (first_peak == None):
                                    first_peak = self.csvData[1][-2]
                                    start_index = t
                                    start_time = self.csvData[0][-2]

                                last_peak = self.csvData[1][-2]
                                stop_index = t
                                end_time = self.csvData[0][-2]
                                if (len(between_peak_time) == 0):
                                    between_peak_time.append(self.csvData[0][-2])
                                elif (prev_peak != 1):
                                    between_peak_time.append(self.csvData[0][-2]- between_peak_time[-1])
                                if (self.csvData[1][-2] > _max_peak):
                                    _max_peak = self.csvData[1][-2]
                                prev_peak = 1
                            elif(self.csvData[1][-1] <= self.csvData[1][-2] and self.csvData[1][-3] < self.csvData[1][-2]):
                                max_points.append(self.csvData[1][-2])
                                max_time.append(self.csvData[0][-2])
                                if (first_peak != None and second_peak == None):
                                    second_peak = self.csvData[1][-2]
                                if (first_peak == None):
                                    first_peak = self.csvData[1][-2]
                                    start_index = t
                                    start_time = self.csvData[0][-2]

                                last_peak = self.csvData[1][-2]
                                stop_index = t
                                end_time = self.csvData[0][-2]
                                if (len(between_peak_time) == 0):
                                    between_peak_time.append(self.csvData[0][-2])
                                elif (prev_peak != 1):
                                    between_peak_time.append(self.csvData[0][-2]- between_peak_time[-1])
                                if (self.csvData[1][-2] > _max_peak):
                                    _max_peak = self.csvData[1][-2]
                                prev_peak = 1
                            elif (self.csvData[1][-1] < self.csvData[1][-2] and self.csvData[1][-3] <= self.csvData[1][-2]):
                                max_points.append(self.csvData[1][-2])
                                max_time.append(self.csvData[0][-2])
                                if (first_peak != None and second_peak == None):
                                    second_peak = self.csvData[1][-2]
                                if (first_peak == None):
                                    first_peak = self.csvData[1][-2]
                                    start_index = t
                                    start_time = self.csvData[0][-2]

                                last_peak = self.csvData[1][-2]
                                stop_index = t
                                end_time = self.csvData[0][-2]
                                if (len(between_peak_time) == 0):
                                    between_peak_time.append(self.csvData[0][-2])
                                elif (prev_peak != 1):
                                    between_peak_time.append(self.csvData[0][-2] - between_peak_time[-1])
                                if (self.csvData[1][-2] > _max_peak):
                                    _max_peak = self.csvData[1][-2]
                                prev_peak = 1


                            
                        if (os.name == "posix"):
                            #peaks 2
                            if (self.csvData[2][-2] >= upper):
                                if (self.csvData[2][-1] < self.csvData[2][-2] and self.csvData[2][-3] < self.csvData[2][
                                    -2]):
                                    max_points_2.append(self.csvData[2][-2])
                                    max_time_2.append(self.csvData[0][-2])
                                    if (first_peak_2 != None and second_peak_2 == None):
                                        second_peak_2 = self.csvData[2][-2]
                                    if (first_peak_2 == None):
                                        first_peak_2 = self.csvData[2][-2]
                                        start_index_2 = t
                                        start_time = self.csvData[0][-2]

                                    last_peak_2 = self.csvData[2][-2]
                                    stop_index_2 = t
                                    prev_peak = 1
                                elif (self.csvData[2][-1] <= self.csvData[2][-2] and self.csvData[2][-3] < self.csvData[2][
                                    -2]):
                                    max_points_2.append(self.csvData[2][-2])
                                    max_time_2.append(self.csvData[0][-2])
                                    if (first_peak_2 != None and second_peak_2 == None):
                                        second_peak_2 = self.csvData[2][-2]
                                    if (first_peak_2 == None):
                                        first_peak_2 = self.csvData[2][-2]
                                        start_index_2 = t
                                        start_time = self.csvData[0][-2]

                                    last_peak_2 = self.csvData[2][-2]
                                    stop_index_2 = t
                                    prev_peak = 1
                                elif (self.csvData[2][-1] < self.csvData[2][-2] and self.csvData[2][-3] <= self.csvData[2][
                                    -2]):
                                    max_points_2.append(self.csvData[2][-2])
                                    max_time_2.append(self.csvData[0][-2])
                                    if (first_peak_2 != None and second_peak_2 == None):
                                        second_peak_2 = self.csvData[2][-2]
                                    if (first_peak_2 == None):
                                        first_peak_2 = self.csvData[2][-2]
                                        start_index_2 = t

                                    last_peak_2 = self.csvData[2][-2]
                                    stop_index_2 = t
                                    
                                    prev_peak = 1

                        #now for the mins
                        if(self.csvData[1][-2] <= lower):
                            if (self.csvData[1][-1] > self.csvData[1][-2] and self.csvData[1][-3] > self.csvData[1][-2]):
                                min_points.append(self.csvData[1][-2])
                                min_time.append(self.csvData[0][-2])
                                if (first_peak != None and second_peak == None):
                                    second_peak = self.csvData[1][-2]
                                if (first_peak == None):
                                    first_peak = self.csvData[1][-2]
                                    if (start_index == None):
                                        start_index = t

                                last_peak = self.csvData[1][-2]
                                stop_index = t
                                end_time = self.csvData[0][-2]
                                if (len(between_peak_time) == 0):
                                    between_peak_time.append(self.csvData[0][-2])
                                elif (prev_peak != 0):
                                    between_peak_time.append(self.csvData[0][-2]- between_peak_time[-1])
                                if (self.csvData[1][-2] < _min_peak):
                                    _min_peak = self.csvData[1][-2]
                                prev_peak = 0
                            elif (self.csvData[1][-1] >= self.csvData[1][-2] and self.csvData[1][-3] > self.csvData[1][-2]):
                                min_points.append(self.csvData[1][-2])
                                min_time.append(self.csvData[0][-2])
                                if (first_peak != None and second_peak == None):
                                    second_peak = self.csvData[1][-2]
                                if (first_peak == None):
                                    first_peak = self.csvData[1][-2]
                                    if (start_index == None):
                                        start_index = t

                                last_peak = self.csvData[1][-2]
                                stop_index = t
                                end_time = self.csvData[0][-2]
                                if (len(between_peak_time) == 0):
                                    between_peak_time.append(self.csvData[0][-2])
                                elif (prev_peak != 0):
                                    between_peak_time.append(self.csvData[0][-2]- between_peak_time[-1])
                                if (self.csvData[1][-2] < _min_peak):
                                    _min_peak = self.csvData[1][-2]
                                prev_peak = 0
                            elif (self.csvData[1][-1] > self.csvData[1][-2] and self.csvData[1][-3] >= self.csvData[1][-2]):
                                min_points.append(self.csvData[1][-2])
                                min_time.append(self.csvData[0][-2])
                                if (first_peak != None and second_peak == None):
                                    second_peak = self.csvData[1][-2]
                                if (first_peak == None):
                                    first_peak = self.csvData[1][-2]
                                    if (start_index == None):
                                        start_index = t

                                last_peak = self.csvData[1][-2]
                                stop_index = t
                                end_time = self.csvData[0][-2]
                                if (len(between_peak_time) == 0):
                                    between_peak_time.append(self.csvData[0][-2])
                                elif (prev_peak != 0):
                                    between_peak_time.append(self.csvData[0][-2]- between_peak_time[-1])
                                if (self.csvData[1][-2] < _min_peak):
                                    _min_peak = self.csvData[1][-2]
                                prev_peak = 0

                        if (os.name == "posix"):
                            #min peaks 2
                            if (self.csvData[2][-2] <= lower):
                                if (self.csvData[2][-1] > self.csvData[2][-2] and self.csvData[2][-3] > self.csvData[2][
                                    -2]):
                                    min_points_2.append(self.csvData[2][-2])
                                    min_time_2.append(self.csvData[0][-2])
                                    if (first_peak_2 != None and second_peak_2 == None):
                                        second_peak_2 = self.csvData[2][-2]
                                    if (first_peak_2 == None):
                                        first_peak_2 = self.csvData[2][-2]
                                        start_index_2 = t
                                        start_time = self.csvData[0][-2]

                                    last_peak_2 = self.csvData[2][-2]
                                    stop_index_2 = t
                                    end_time = self.csvData[0][-2]
                                    prev_peak = 0
                                elif (self.csvData[2][-1] >= self.csvData[2][-2] and self.csvData[2][-3] > self.csvData[2][
                                    -2]):
                                    min_points_2.append(self.csvData[2][-2])
                                    min_time_2.append(self.csvData[0][-2])
                                    if (first_peak_2 != None and second_peak_2 == None):
                                        second_peak_2 = self.csvData[2][-2]
                                    if (first_peak_2 == None):
                                        first_peak_2 = self.csvData[2][-2]
                                        start_index_2 = t
                                        start_time = self.csvData[0][-2]

                                    last_peak_2 = self.csvData[2][-2]
                                    stop_index_2 = t
                                    end_time = self.csvData[0][-2]
                                    prev_peak = 0
                                elif (self.csvData[2][-1] > self.csvData[2][-2] and self.csvData[2][-3] >= self.csvData[2][
                                    -2]):
                                    min_points_2.append(self.csvData[2][-2])
                                    min_time_2.append(self.csvData[0][-2])
                                    if (first_peak_2 != None and second_peak_2 == None):
                                        second_peak_2 = self.csvData[2][-2]
                                    if (first_peak_2 == None):
                                        first_peak_2 = self.csvData[2][-2]
                                        start_index_2 = t
                                        start_time = self.csvData[0][-2]

                                    last_peak_2 = self.csvData[2][-2]
                                    stop_index_2 = t
                                    end_time = self.csvData[0][-2]
                                    prev_peak = 0


                #trig_tracker.append(0) if Sk/self.buffer_length < 0.0075 else trig_tracker.append(1)
                trig_time.append(t)



            if(len(self.csvData[0]) >= sample_points):
                tok = time.time()
                break

        print(f"elapsed time {tok-tik}")
        print(f"sample rate: {len(self.csvData[1])/(tok-tik)}")

        print(f"inpeak time: {between_peak_time}")
        print(f"max peaks: {max_points}")
        print(f"min peaks: {min_points}")

        with open('decision_data.csv', 'a') as fd:
            #fd.write(self.csv_write)
            writer = csv.writer(fd)
            for row in self.csv_write:
                writer.writerow(row)

        fig, axs = plt.subplots(4)

        fig.suptitle("Recorded data")
        #print(len(trig_tracker), len(trig_time))
        print(len(self.csvData[0]), len(self.csvData[1]))
        axs[0].scatter(self.csvData[0], self.csvData[1])
        if(self.peak_method == 'slope'):
            axs[0].scatter(slope_time, slope_pts)
        elif(self.peak_method == 'peak_detection'):
            axs[0].scatter(max_time, max_points)
            axs[0].scatter(min_time, min_points)
        axs[0].axhline(y=lower, c='red')
        axs[0].axhline(y=upper, c='red')
        axs[0].set_xlim(self.csvData[0][0], self.csvData[0][-1])
        axs[0].set_xlabel("time [s]")
        axs[0].set_ylabel("Voltage [V]")
        plt.grid(True)
        axs[1].scatter(self.csvData[0], self.csvData[2])
        axs[1].scatter(max_time_2, max_points_2)
        axs[1].scatter(min_time_2, min_points_2)
        #axs[1].axhline(y=0, c='red')
        axs[1].set_xlim(self.csvData[0][0],self.csvData[0][-1])
        axs[1].set_xlabel("time [s]")
        axs[1].set_ylabel("dV/dt [V]")
        plt.grid(True)

        axs[2].plot(trig_time, trig_tracker)
        axs[2].set_xlim(self.csvData[0][0], self.csvData[0][-1])
        axs[2].set_xlabel("time [s]")
        axs[2].set_ylabel("Voltage [V]")
        plt.grid(True)

        axs[3].scatter(variance_time_pt, variance_pt)
        axs[3].set_xlim(self.csvData[0][0], self.csvData[0][-1])
        axs[3].set_xlabel("time [s]")
        axs[3].set_ylabel("Voltage [V]")
        plt.grid(True)
        plt.show()

        if(save_plot):
            """save the plot"""

        if(save_data):

            '''write the data to a csv'''

        print("Here are the time windows")
        print(tracked_signal_marks)
        print(decisions)

if __name__ == "__main__":
    port = "COM3"
    baudrate=115200

    date = datetime.date.today().strftime("%m-%d-%Y-%H-%M-%S")
    csv_name = f"data_capture_{date}"

    handle = serialCapture(port, baudrate, timeout=4, data_name=csv_name, peak_method='peak_detection', thresh_holds=[2.20, 2.80])
    handle.printCollectedData(None, None, sample_points=1000)