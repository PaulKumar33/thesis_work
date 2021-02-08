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


class system_main(object):
    def __init(self, low_delay, peak_method="peaks", window_thresholds=[2.25, 2.75],
               variance_threshold=0.085):

        self.filter_length = 21
        self.buffer_length = 16 #length of the variance data buffer
        self.low_delay = low_delay
        self.thresh_holds = thresh_holds
        self.variance_limit = variance_threshold

        self.peak_method = peak_method

        self.csv_write = []

        self.HW_FLAG = False #this should be set to true during HW event
        self.RECORD_FLAG = False
        self.HW_EVENT = 0
        self.HW_COUNT = 0

        try:
            if (os.name == 'nt'):
                self.handle = serial.Serial(port, baudrate, timeout=timeout)
                print("Connected")

                df = pd.read_excel("./decision_data.xlsx", "data_no_higher_order")
                # df = df.drop(columns=['figure'])

                df_mat = df.values.tolist()
                """self.tree = impurity.build_tree(df_mat)
                impurity.print_tree(self.tree)"""

            elif (os.name == 'posix'):
                print("connecting to ADC")
                self.mcp = mcp.mcp_external()
                df = pd.read_excel("./decision_data.xlsx", "scheme_1")
                # df = df.drop(columns=['figure'])

                df_mat = df.values.tolist()
                self.tree = impurity.build_tree(df_mat)
                impurity.print_tree(self.tree)
                print("starting capture")

        except Exception as e:
            print(e)
        pass

    def runCollection(self):
        print('sleeping for test')
        time.sleep(5)

        self.upper = self.thresh_holds[1]
        self.lower = self.thresh_holds[0]

        self.var_limit = 0.00850

        tik = time.time()

        avg_buffer = []
        data_buffer = []
        trig_tracker = []
        self.trig_time = []
        self.slope_pts = []
        self.slope_time = []

        total_slope = []
        total_slope_time = []

        variance_pt = []
        variance_time_pt = []

        # these are the time points
        self.max_points = []
        self.min_points = []
        self.max_points_2 = []
        self.min_points_2 = []

        self.max_time = []
        self.min_time = []
        self.max_time_2 = []
        self.min_time_2 = []

        _event = False
        window = []

        self.first_peak = None
        self.last_peak = None
        self.second_peak = None
        self.second_last_peak = None
        self.first_peak_2 = None
        self.last_peak_2 = None
        self.second_peak_2 = None
        self.second_last_peak_2 = None

        self.start_index = None
        self.stop_index = None
        self.start_index_2 = None
        self.stop_index_2 = None

        self.tracked_signal_marks = []


        self._min_peak = 2.5
        self._max_peak = 2.5

        self.DATA_STOP = True
        self.STOP_TIME = 0

        counter = 5
        print("start")
        while(True):

            if(self.HW_FLAG):
                #need to check if we recorded the wash
                #set the flag low after its completed
                self.HW_COUNT += 1
                self.HW_EVENT += 1
                self.HW_FLAG = False

                #set the stop collection flag
                self.DATA_STOP = True
                self.STOP_TIME = time.time()

                #we want to continue onto the next iteration. Also stop tracking for a few seconds and refresh the buffers
                continue

            if(time.time() - self.STOP_TIME > 3):
                self.DATA_STOP = False


            # implement the moving average filter - before the buffer fills
            if (len(avg_buffer) < self.filter_length):
                t = time.time() - tik
                self.csvData[0].append(t)
                self.csvData[1].append(float(self.mcp.read_IO(0) / 65355 * 5))
                self.csvData[2].append(float(self.mcp.read_IO(1) / 65355 * 5))

            #after the buffer fills
            else:
                data = float(self.mcp.read_IO(0) / 65355 * 5)
                data1 = float(self.mcp.read_IO(1) / 65355 * 5)
                # print("DATA: {}".format(data/65355*5.2))
                # data = struct.unpack('f', self.rawData)[0]/self.filter_length

                #implement the filter
                for i in range(self.filter_length - 1):
                    data += self.csvData[1][-(i + 1)]
                    data1 += self.csvData[2][-(i + 1)]
                self.csvData[1].append(data)
                self.csvData[2].append(data1)

            #now lets check the data
            #implement moving variance
            if (len(data_buffer) < self.buffer_length):
                data_buffer.append(self.csvData[1][-1])
            else:
                data_buffer = data_buffer[1:] + [self.csvData[1][-1]]

                var_mean = data_buffer[0]
                Sk = 0

                #moving variance
                for k in range(1, len(data_buffer)):
                    var_mean_1 = var_mean + (1 / self.buffer_length) * (data_buffer[k] - var_mean)
                    Sk_1 = Sk + (data_buffer[k] - var_mean_1) * (data_buffer[k] - var_mean)

                    # update recursive variables
                    var_mean = var_mean_1
                    Sk = Sk_1

                variance_pt.append(Sk / self.buffer_length)
                variance_time_pt.append(t)
                if (Sk / self.buffer_length < var_limit and self.DATA_STOP == True):
                    counter += 1
                    if (counter >= self.low_delay):
                        _event = False

                        trig_tracker.append(0)
                        # slope = (self.csvData[1][-1] - self.csvData[1][-2]) / (
                        #        self.csvData[0][-1] - self.csvData[0][-2])
                        total_slope.append(1)
                        total_slope_time.append(t)
                        slope_1 = 1

                        if (stop_index != None and start_index != None and first_peak != None and last_peak != None):
                            if (stop_index - start_index > 1.5):
                                self.HW_EVENT += 1

                                #compute the classification
                                temp = [0, 0, 0, 0, 0, 0, 0, 0]
                                try:
                                    '''gradient = ((last_peak - 2.5) - (first_peak - 2.5))/(stop_index - start_index)
                                    gradient_2 = ((last_peak_2 - 2.5) - (first_peak_2 - 2.5))/(stop_index_2 - start_index_2)'''
                                    print("Here are the peaks: {0}, {1}, {2}, {3}".format(self.first_peak, self.second_peak,
                                                                                          self.second_last_peak, self.last_peak))
                                    print("Here are the peaks: {0}, {1}, {2}, {3}".format(self.first_peak_2, self.second_peak_2,
                                                                                          self.second_last_peak_2,
                                                                                          self.last_peak_2))
                                    gradient = ((self.last_peak + self.second_last_peak) / 2 - (self.first_peak + self.second_peak) / 2) / (
                                                self.stop_index - self.start_index)
                                    gradient_2 = ((self.last_peak_2 + self.second_last_peak_2) / 2 - (
                                                self.first_peak_2 + self.second_peak_2) / 2) / (self.stop_index_2 - self.start_index_2)
                                    print("here me gradients: {0}, {1}".format(gradient, gradient_2))
                                except Exception as e:
                                    print(e)

                                temp[0] = 1 if self.first_peak >= 2.5 else 0
                                temp[1] = 1 if self.last_peak >= 2.5 else 0
                                # temp[2] = 1 if gradient >= 0 else 0
                                temp[2] = 1 if self.first_peak_2 >= 2.5 else 0
                                temp[3] = 1 if self.last_peak_2 >= 2.5 else 0
                                temp[4] = 1 if self.second_peak >= 2.5 else 0
                                temp[5] = 1 if self.second_peak_2 >= 2.5 else 0
                                temp[6] = 1 if self.second_last_peak >= 2.5 else 0
                                temp[7] = 1 if self.second_last_peak_2 >= 2.5 else 0

                                # time_period = end_time - start_time
                                """s_index = between_peak_time.index(start_time)
                                e_index = between_peak_time.index(end_time)
                                mean_pk2pk = statistics.mean(between_peak_time[s_index:e_index+1])"""

                                s_index = self.csvData[0].index(self.start_index)
                                e_index = self.csvData[0].index(self.stop_index)

                                _mean = statistics.mean(self.csvData[1])
                                variance = statistics.variance(self.csvData[1])
                                skewness = scipy.stats.skew(self.csvData[1])

                                variance_2 = statistics.variance(self.csvData[2])
                                skewness_2 = scipy.stats.skew(self.csvData[2])

                                print(
                                    "#########################Getting features#######################################")
                                print("")
                                print("")

                                # self.csv_write.append(start_index, stop_index, first_peak, last_peak, gradient, _max_peak, _min_peak, start_time, end_time, time_period)
                                self.tracked_signal_marks.append(temp)
                                # self.csv_write.append([temp[0], temp[1], gradient, temp[2], temp[3], gradient_2, _mean, variance, skewness, variance_2, skewness_2])

                                # _classify = impurity.classify(
                                #    [temp[0], temp[1], gradient, temp[2], temp[3], gradient_2, temp[4], temp[5], variance, skewness, variance_2, skewness_2], self.tree)
                                '''_classify = impurity.classify(
                                    [first_peak, last_peak, first_peak_2, last_peak_2, second_peak, second_peak_2, gradient, gradient_2], self.tree)'''
                                _classify = impurity.classify(
                                    [temp[0], temp[1], gradient, temp[2], temp[3], gradient_2, temp[4], temp[5],
                                     temp[6], temp[7]], self.tree)
                                max_guess = 0
                                max_class = None
                                for _class_ in _classify:
                                    if (_classify[_class_] > max_guess):
                                        max_class, max_guess = _class_, _classify[_class_]
                                print("Predicted: {}".format(max_class))
                                self.csv_write.append(
                                    [self.first_peak, self.second_peak, self.second_last_peak, self.last_peak, self.first_peak_2, self.second_peak_2,
                                     self.second_last_peak_2, self.last_peak_2, self.second_peak, temp[0], temp[1], gradient, temp[2],
                                     temp[3], gradient_2, temp[4], temp[5], temp[6], temp[7]])

                            self.start_index, self.stop_index, self.first_peak, self.first_peak_2, self.second_peak, self.second_peak_2, self.last_peak, self.last_peak_2, self.end_time, self._max_peak, self._min_peak = \
                                None, None, None, None, None, None, None, None, None, 2.5, 2.5
                            self.second_last_peak, self.second_last_peak_2 = None, None
                        else:
                            trig_tracker.append(1)
                            if (self.peak_method == 'peak_detection'):
                                # simple implementation of the peak detection algorithm

                                # find max
                                self.runPeakDetection()
                else:
                    _event = True
                    window.append(t)
                    trig_tracker.append(1)
                    counter = 0

                    if(self.peak_method == "peak_detection"):
                        self.runPeakDetection()

                trig_tracker.append(t)

            if (len(self.csvData[0]) >= sample_points):
                tok = time.time()
                break

        print(f"elapsed time {tok - tik}")
        print(f"sample rate: {len(self.csvData[1]) / (tok - tik)}")

        print(f"inpeak time: {self.between_peak_time}")
        print(f"max peaks: {self.max_points}")
        print(f"min peaks: {self.min_points}")

        with open('decision_data.csv', 'a') as fd:
            # fd.write(self.csv_write)
            writer = csv.writer(fd)
            for row in self.csv_write:
                writer.writerow(row)

        fig, axs = plt.subplots(4)

        fig.suptitle("Recorded data")
        # print(len(trig_tracker), len(trig_time))
        print(len(self.csvData[0]), len(self.csvData[1]))
        axs[0].scatter(self.csvData[0], self.csvData[1])
        if (self.peak_method == 'peak_detection'):
            axs[0].scatter(self.max_time, self.max_points)
            axs[0].scatter(self.min_time, self.min_points)
        axs[0].axhline(y=self.lower, c='red')
        axs[0].axhline(y=self.upper, c='red')
        axs[0].set_xlim(self.csvData[0][0], self.csvData[0][-1])
        axs[0].set_xlabel("time [s]")
        axs[0].set_ylabel("Voltage [V]")
        plt.grid(True)
        axs[1].scatter(self.csvData[0], self.csvData[2])
        axs[1].scatter(self.max_time_2, self.max_points_2)
        axs[1].scatter(self.min_time_2, self.min_points_2)
        # axs[1].axhline(y=0, c='red')
        axs[1].set_xlim(self.csvData[0][0], self.csvData[0][-1])
        axs[1].set_xlabel("time [s]")
        axs[1].set_ylabel("dV/dt [V]")
        plt.grid(True)

        axs[2].plot(self.trig_time, trig_tracker)
        axs[2].set_xlim(self.csvData[0][0], self.csvData[0][-1])
        axs[2].set_xlabel("time [s]")
        axs[2].set_ylabel("Voltage [V]")
        plt.grid(True)

        axs[3].scatter(self.variance_time_pt, self.variance_pt)
        axs[3].set_xlim(self.csvData[0][0], self.csvData[0][-1])
        axs[3].set_xlabel("time [s]")
        axs[3].set_ylabel("Voltage [V]")
        plt.grid(True)
        plt.show()


    def runPeakDetection(self, t):
        # simple implementation of the peak detection algorithm

        # find max _peak_1
        if (self.csvData[1][-2] >= self.upper):
            if (self.csvData[1][-1] < self.csvData[1][-2] and self.csvData[1][-3] < self.csvData[1][-2]):
                self.max_points.append(self.csvData[1][-2])
                self.max_time.append(self.csvData[0][-2])
                if (self.first_peak != None and self.second_peak == None and (self.first_peak - 2.5) / (
                        self.csvData[1][-2] - 2.5) < 0):
                    self.second_peak = self.csvData[1][-2]
                if (self.first_peak == None):
                    self.first_peak = self.csvData[1][-2]
                    self.start_index = t

                if (self.last_peak != None and (self.last_peak - 2.5) / (self.csvData[1][-2] - 2.5) < 0):
                    self.second_last_peak = self.last_peak

                self.last_peak = self.csvData[1][-2]
                self.stop_index = t

                if (self.csvData[1][-2] > self._max_peak):
                    self._max_peak = self.csvData[1][-2]
            elif (self.csvData[1][-1] <= self.csvData[1][-2] and self.csvData[1][-3] < self.csvData[1][-2]):
                self.max_points.append(self.csvData[1][-2])
                self.max_time.append(self.csvData[0][-2])
                if (self.first_peak != None and self.second_peak == None and (self.first_peak - 2.5) / (
                        self.csvData[1][-2] - 2.5) < 0):
                    self.second_peak = self.csvData[1][-2]
                if (self.first_peak == None):
                    self.first_peak = self.csvData[1][-2]
                    self.start_index = t

                if (self.last_peak != None and (self.last_peak - 2.5) / (self.csvData[1][-2] - 2.5) < 0):
                    self.second_last_peak = self.last_peak

                self.last_peak = self.csvData[1][-2]
                self.stop_index = t
                if (self.csvData[1][-2] > self._max_peak):
                    _max_peak = self.csvData[1][-2]
                prev_peak = 1
            elif (self.csvData[1][-1] < self.csvData[1][-2] and self.csvData[1][-3] <= self.csvData[1][-2]):
                self.max_points.append(self.csvData[1][-2])
                self.max_time.append(self.csvData[0][-2])
                if (self.first_peak != None and self.second_peak == None and (self.first_peak - 2.5) / (
                        self.csvData[1][-2] - 2.5) < 0):
                    self.second_peak = self.csvData[1][-2]
                if (self.first_peak == None):
                    self.first_peak = self.csvData[1][-2]
                    self.start_index = t

                if (self.last_peak != None and (self.last_peak - 2.5) / (self.csvData[1][-2] - 2.5) < 0):
                    self.second_last_peak = self.last_peak

                self.last_peak = self.csvData[1][-2]
                self.stop_index = t
                if (self.csvData[1][-2] > self._max_peak):
                    self._max_peak = self.csvData[1][-2]

        if (os.name == "posix"):
            # peaks 2
            if (self.csvData[2][-2] >= self.upper):
                if (self.csvData[2][-1] < self.csvData[2][-2] and self.csvData[2][-3] < self.csvData[2][
                    -2]):
                    self.max_points_2.append(self.csvData[2][-2])
                    self.max_time_2.append(self.csvData[0][-2])
                    if (self.first_peak_2 != None and self.second_peak_2 == None and (self.first_peak_2 - 2.5) / (
                            self.csvData[2][-2] - 2.5) < 0):
                        self.second_peak_2 = self.csvData[2][-2]
                    if (self.first_peak_2 == None):
                        self.first_peak_2 = self.csvData[2][-2]
                        self.start_index_2 = t

                    if (self.last_peak_2 != None and (self.last_peak_2 - 2.5) / (self.csvData[2][-2] - 2.5) < 0):
                        self.second_last_peak_2 = self.last_peak_2

                    self.last_peak_2 = self.csvData[2][-2]
                    self.stop_index_2 = t
                elif (self.csvData[2][-1] <= self.csvData[2][-2] and self.csvData[2][-3] < self.csvData[2][
                    -2]):
                    self.max_points_2.append(self.csvData[2][-2])
                    self.max_time_2.append(self.csvData[0][-2])
                    if (self.first_peak_2 != None and self.second_peak_2 == None and (self.first_peak_2 - 2.5) / (
                            self.csvData[2][-2] - 2.5) < 0):
                        self.second_peak_2 = self.csvData[2][-2]
                    if (self.first_peak_2 == None):
                        self.first_peak_2 = self.csvData[2][-2]
                        self.start_index_2 = t

                    if (self.last_peak_2 != None and (self.last_peak_2 - 2.5) / (self.csvData[2][-2] - 2.5) < 0):
                        self.second_last_peak_2 = self.last_peak_2

                    self.last_peak_2 = self.csvData[2][-2]
                    self.stop_index_2 = t
                elif (self.csvData[2][-1] < self.csvData[2][-2] and self.csvData[2][-3] <= self.csvData[2][
                    -2]):
                    self.max_points_2.append(self.csvData[2][-2])
                    self.max_time_2.append(self.csvData[0][-2])
                    if (self.first_peak_2 != None and self.second_peak_2 == None and (self.first_peak_2 - 2.5) / (
                            self.csvData[2][-2] - 2.5) < 0):
                        self.second_peak_2 = self.csvData[2][-2]
                    if (self.first_peak_2 == None):
                        self.first_peak_2 = self.csvData[2][-2]
                        self.start_index_2 = t

                    if (self.last_peak_2 != None and (self.last_peak_2 - 2.5) / (self.csvData[1][-2] - 2.5) < 0):
                        self.second_last_peak_2 = self.last_peak_2

                    self.last_peak_2 = self.csvData[2][-2]
                    self.stop_index_2 = t

        # now for the mins
        if (self.csvData[1][-2] <= self.lower):
            if (self.csvData[1][-1] > self.csvData[1][-2] and self.csvData[1][-3] > self.csvData[1][-2]):
                self.min_points.append(self.csvData[1][-2])
                self.min_time.append(self.csvData[0][-2])
                if (self.first_peak != None and self.second_peak == None and (self.first_peak - 2.5) / (
                        self.csvData[1][-2] - 2.5) < 0):
                    self.second_peak = self.csvData[1][-2]
                if (self.first_peak == None):
                    self.first_peak = self.csvData[1][-2]
                    self.start_index = t

                if (self.last_peak != None and (self.last_peak - 2.5) / (self.csvData[1][-2] - 2.5) < 0):
                    self.second_last_peak = self.last_peak

                self.last_peak = self.csvData[1][-2]
                self.stop_index = t
                self.end_time = self.csvData[0][-2]
                if (self.csvData[1][-2] < self._min_peak):
                    self._min_peak = self.csvData[1][-2]

            elif (self.csvData[1][-1] >= self.csvData[1][-2] and self.csvData[1][-3] > self.csvData[1][-2]):
                self.min_points.append(self.csvData[1][-2])
                self.min_time.append(self.csvData[0][-2])
                if (self.first_peak != None and self.second_peak == None and (self.first_peak - 2.5) / (
                        self.csvData[1][-2] - 2.5) < 0):
                    self.second_peak = self.csvData[1][-2]
                if (self.first_peak == None):
                    self.first_peak = self.csvData[1][-2]
                    self.start_index = t

                if (self.last_peak != None and (self.last_peak - 2.5) / (self.csvData[1][-2] - 2.5) < 0):
                    self.second_last_peak = self.last_peak

                self.last_peak = self.csvData[1][-2]
                self.stop_index = t
                self.end_time = self.csvData[0][-2]
                if (self.csvData[1][-2] < self._min_peak):
                    self._min_peak = self.csvData[1][-2]

            elif (self.csvData[1][-1] > self.csvData[1][-2] and self.csvData[1][-3] >= self.csvData[1][-2]):
                self.min_points.append(self.csvData[1][-2])
                self.min_time.append(self.csvData[0][-2])
                if (self.first_peak != None and self.second_peak == None and (self.first_peak - 2.5) / (
                        self.csvData[1][-2] - 2.5) < 0):
                    self.second_peak = self.csvData[1][-2]
                if (self.first_peak == None):
                    self.first_peak = self.csvData[1][-2]
                    self.start_index = t

                if (self.last_peak != None and (self.last_peak - 2.5) / (self.csvData[1][-2] - 2.5) < 0):
                    self.second_last_peak = self.last_peak

                self.last_peak = self.csvData[1][-2]
                self.stop_index = t
                self.end_time = self.csvData[0][-2]
                if (self.csvData[1][-2] < self._min_peak):
                    self._min_peak = self.csvData[1][-2]

        if (os.name == "posix"):
            # min peaks 2
            if (self.csvData[2][-2] <= lower):
                if (self.csvData[2][-1] > self.csvData[2][-2] and self.csvData[2][-3] > self.csvData[2][
                    -2]):
                    self.min_points_2.append(self.csvData[2][-2])
                    self.min_time_2.append(self.csvData[0][-2])
                    if (self.first_peak_2 != None and self.second_peak_2 == None and (self.first_peak_2 - 2.5) / (
                            self.csvData[2][-2] - 2.5) < 0):
                        self.second_peak_2 = self.csvData[2][-2]
                    if (self.first_peak_2 == None):
                        self.first_peak_2 = self.csvData[2][-2]
                        self.start_index_2 = t

                    if (self.last_peak_2 != None and (self.last_peak_2 - 2.5) / (self.csvData[2][-2] - 2.5) < 0):
                        self.second_last_peak_2 = self.last_peak_2

                    self.last_peak_2 = self.csvData[2][-2]
                    self.stop_index_2 = t
                    self.end_time = self.csvData[0][-2]

                elif (self.csvData[2][-1] >= self.csvData[2][-2] and self.csvData[2][-3] > self.csvData[2][
                    -2]):
                    self.min_points_2.append(self.csvData[2][-2])
                    self.min_time_2.append(self.csvData[0][-2])
                    if (self.first_peak_2 != None and self.second_peak_2 == None and (self.first_peak_2 - 2.5) / (
                            self.csvData[2][-2] - 2.5) < 0):
                        self.second_peak_2 = self.csvData[2][-2]
                    if (self.first_peak_2 == None):
                        self.first_peak_2 = self.csvData[2][-2]
                        self.start_index_2 = t

                    if (self.last_peak_2 != None and (self.last_peak_2 - 2.5) / (self.csvData[2][-2] - 2.5) < 0):
                        self.second_last_peak_2 = self.last_peak_2

                    self.last_peak_2 = self.csvData[2][-2]
                    self.stop_index_2 = t
                    self.end_time = self.csvData[0][-2]

                elif (self.csvData[2][-1] > self.csvData[2][-2] and self.csvData[2][-3] >= self.csvData[2][
                    -2]):
                    self.min_points_2.append(self.csvData[2][-2])
                    self.min_time_2.append(self.csvData[0][-2])
                    if (self.first_peak_2 != None and self.second_peak_2 == None and (self.first_peak_2 - 2.5) / (
                            self.csvData[2][-2] - 2.5) < 0):
                        self.second_peak_2 = self.csvData[2][-2]
                    if (self.first_peak_2 == None):
                        self.first_peak_2 = self.csvData[2][-2]
                        self.start_index_2 = t

                    if (self.last_peak_2 != None and (self.last_peak_2 - 2.5) / (self.csvData[2][-2] - 2.5) < 0):
                        self.second_last_peak_2 = self.last_peak_2

                    self.last_peak_2 = self.csvData[2][-2]
                    self.stop_index_2 = t
                    self.end_time = self.csvData[0][-2]


    def refreshFeatures(self):
        self.start_index, self.stop_index, self.first_peak, self.first_peak_2, self.second_peak, self.second_peak_2, self.last_peak, self.last_peak_2, self.end_time, self._max_peak, self._min_peak = \
            None, None, None, None, None, None, None, None, None, 2.5, 2.5
        self.second_last_peak, self.second_last_peak_2 = None, None
