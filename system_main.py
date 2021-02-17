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
import RPi.GPIO as GPIO

import matplotlib.pyplot as plt
from decision_tree import impurity


def softwareBtnInterrupt(channel):
        FLAGS["HW_FLAG"] = True

#HW_FLAG = False #this should be set to true during HW event
FLAGS={
    "HW_FLAG": False
    }
class system_main:   
    
    def __init__(self, low_delay, wait_time, peak_method="peak_detection", window_thresholds=[2.25, 2.75],
               variance_threshold=0.085):

        self.filter_length = 3
        self.buffer_length = 16 #length of the variance data buffer
        self.low_delay = low_delay
        self.thresh_holds = window_thresholds
        self.variance_limit = variance_threshold

        self.peak_method = peak_method

        self.csv_write = []
        self.csvData = [[],[],[]]
        
        self.RECORD_FLAG = False
        self.HW_EVENT = 0
        self.HW_COUNT = 0
        self.wait_time = wait_time
        
        #setu0p the GPIOs for interrupts
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(24, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        
        GPIO.add_event_detect(24, GPIO.RISING, callback=softwareBtnInterrupt, bouncetime=300)
        GPIO.setup(16, GPIO.OUT)

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
                df = pd.read_excel("./decision_data_new.xlsx", "adjusted_lens")
                # df = df.drop(columns=['figure'])

                df_mat = df.values.tolist()
                self.df_mat = df_mat
                self.tree = impurity.build_tree(df_mat)
                impurity.print_tree(self.tree)
                print("starting capture")

        except Exception as e:
            print(e)

    def runCollection(self, sample_points=100):
        
        DIRECTION_FLAG = 0

        print(self.tree)
        print('sleeping for test')
        print(FLAGS)
        time.sleep(self.wait_time)

        self.upper = self.thresh_holds[1]
        self.lower = self.thresh_holds[0]

        self.var_limit = 0.00850

        tik = time.time()

        avg_buffer = []
        data_buffer = []
        data_buffer_2 = []
        trig_tracker = []
        trig_time = []
        self.slope_pts = []
        self.slope_time = []

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

        self._min_peak = 2.5
        self._max_peak = 2.5

        self.DATA_STOP = False
        self.STOP_TIME = 0

        counter = 5
        print("start")
        while(True):
            #for the sample rate
            time.sleep(0.035)

            if(FLAGS["HW_FLAG"]):
                #need to check if we recorded the wash
                #set the flag low after its completed
                self.HW_COUNT += 1
                self.HW_EVENT += 1
                FLAGS["HW_FLAG"] = False

                #set the stop collection flag
                self.DATA_STOP = True
                self.STOP_TIME = time.time()
                print(f"Event recorded: {t}")

                #we want to continue onto the next iteration. Also stop tracking for a few seconds and refresh the buffers
                continue

            if(time.time() - self.STOP_TIME > 4.5 and self.DATA_STOP == True):
                print("recordin starting again")
                self.DATA_STOP = False


            # implement the moving average filter - before the buffer fills
            #if (len(self.csvData[1]) < self.filter_length):
            if (True):
                t = time.time() - tik
                self.csvData[0].append(t)
                self.csvData[1].append(float(self.mcp.read_IO(0) / 65355 * 5))
                self.csvData[2].append(float(self.mcp.read_IO(1) / 65355 * 5))

            #after the buffer fills
            else:
                data = float(self.mcp.read_IO(0) / 65355 * 5)/(self.filter_length)
                data1 = float(self.mcp.read_IO(1) / 65355 * 5)/(self.filter_length)
                # print("DATA: {}".format(data/65355*5.2))
                # data = struct.unpack('f', self.rawData)[0]/self.filter_length

                #implement the filter
                for i in range(self.filter_length - 1):
                    data += self.csvData[1][-(i + 1)]/(self.filter_length)
                    data1 += self.csvData[2][-(i + 1)]/(self.filter_length)
                t = time.time() - tik
                self.csvData[0].append(t)
                self.csvData[1].append(data)
                self.csvData[2].append(data1)

            #now lets check the data
            #implement moving variance
            if (len(data_buffer) < self.buffer_length):
                data_buffer.append(self.csvData[1][-1])
                data_buffer_2.append(self.csvData[2][-1])
            else:
                data_buffer = data_buffer[1:] + [self.csvData[1][-1]]
                data_buffer_2 = data_buffer_2[1:] + [self.csvData[2][-1]]

                var_mean = data_buffer[0]
                var_mean_2 = data_buffer_2[0]
                Sk_2 = 0
                Sk = 0

                #moving variance
                for k in range(1, len(data_buffer)):

                    #variance
                    var_mean_1 = var_mean + (1 / self.buffer_length) * (data_buffer[k] - var_mean)
                    var_mean_21 = var_mean_2 + (1 / self.buffer_length) * (data_buffer_2[k] - var_mean_2)
                    Sk_1 = Sk + (data_buffer[k] - var_mean_1) * (data_buffer[k] - var_mean)
                    Sk_21 = Sk_2 + (data_buffer_2[k] - var_mean_2)*(data_buffer_2[k] - var_mean_2)

                    # update recursive variables
                    var_mean = var_mean_1
                    var_mean_2 = var_mean_21
                    Sk = Sk_1
                    Sk_2 = Sk_21
                
                if(self.DATA_STOP == True):
                    #force the variance signal low
                    Sk = 0
                    Sk_2 = 0

                variance_pt.append(Sk / self.buffer_length)
                variance_time_pt.append(t)
                if ((Sk / self.buffer_length < self.var_limit and self.DATA_STOP == False) and
                        (Sk_2/self.buffer_length < self.var_limit and self.DATA_STOP == False)):
                    #if data stop high, we do not care about the direction classification, we record the HW event
                    counter += 1
                    if (counter >= self.low_delay):
                        _event = False
                        #counter = 0
                        self.gpioLOW(16)
                        trig_tracker.append(0)

                        if (self.stop_index != None and self.start_index != None and self.first_peak != None and self.last_peak != None):
                            if (self.stop_index - self.start_index > 1.25):

                                #compute the classification
                                try:
                                    self.printPeaksTest()
                                    gradient = self.getGradients(self.first_peak, self.second_peak, self.last_peak, self.second_last_peak, self.stop_index, self.start_index)
                                    gradient_2 = self.getGradients(self.first_peak_2, self.second_peak_2, self.last_peak_2, self.second_peak_2, self.stop_index, self.start_index)
                                    print("here me gradients: {0}, {1}".format(gradient, gradient_2))
                                except Exception as e:
                                    print(e)

                                temp = self.oneHotPeaks([self.first_peak, self.last_peak, self.first_peak_2, self.last_peak_2, self.second_peak, self.second_peak_2, self.second_last_peak, self.second_last_peak_2, self.second_last_peak_2])

                                print(
                                    "#########################Getting features#######################################")
                                print("")
                                print("")

                                #gradient_2=-1
                                '''
                                temp0 - first_peak
                                1 - last_peak                                
                                2 - first_peak_2
                                3 - last_peak_2
                                4 - second_peak
                                5 - second_peak_2
                                6 - second_last_peak
                                7 - second_last_peak_2
                                '''

                                _classify = impurity.classify([temp[0], temp[1], gradient, temp[4], temp[6]], self.tree)
                                max_guess = 0
                                max_class = None

                                for _class_ in _classify:
                                    if (_classify[_class_] > max_guess):
                                        max_class, max_guess = _class_, _classify[_class_]
                                print("Predicted: {}".format(max_class))

                                self.directionIndication(max_class, DIRECTION_FLAG)
                                self.csv_write.append(
                                    [self.first_peak, self.second_peak, self.second_last_peak, self.last_peak, self.first_peak_2, self.second_peak_2,
                                     self.second_last_peak_2, self.last_peak_2, self.second_peak, temp[0], temp[1], gradient, temp[2],
                                     temp[3], gradient_2, temp[4], temp[5], temp[6], temp[7], _classify])

                            self.start_index, self.stop_index, self.first_peak, self.first_peak_2, self.second_peak, self.second_peak_2, self.last_peak, self.last_peak_2, self.end_time, self._max_peak, self._min_peak = \
                                None, None, None, None, None, None, None, None, None, 2.5, 2.5
                            self.second_last_peak, self.second_last_peak_2 = None, None

                    else:

                        trig_tracker.append(1)
                        self.gpioHIGH(16)
                        if (self.peak_method == 'peak_detection'):
                            # simple implementation of the peak detection algorithm
                            # find max
                            self.runPeakDetection(t)                    

                elif((Sk/self.buffer_length > self.var_limit and self.DATA_STOP == False) or
                     (Sk_2/self.buffer_length > self.var_limit and self.DATA_STOP == False)):
                    #we have detected movement and no HW event yet
                    _event = True
                    self.gpioHIGH(16)
                    window.append(t)
                    trig_tracker.append(1)
                    counter = 0

                    if(self.peak_method == "peak_detection"):
                        self.runPeakDetection(t)
                elif(self.DATA_STOP == True):
                    self.gpioLOW(16)
                    trig_tracker.append(0)
                    Sk = 0

                trig_time.append(t)

            if (len(self.csvData[0]) >= sample_points):
                tok = time.time()
                break

        print(f"elapsed time {tok - tik}")
        print(f"sample rate: {len(self.csvData[1]) / (tok - tik)}\n")
        
        
        print(f"total HW events: {self.HW_EVENT}")
        print(f"total HW completed: {self.HW_COUNT}")

        with open('new_datas.csv', 'a') as fd:
            # fd.write(self.csv_write)
            writer = csv.writer(fd)
            for row in self.csv_write:
                writer.writerow(row)
        
        with open("recorded_data.csv", "a") as fl:
            writer = csv.writer(fl)
            for row in self.csvData:
                writer.writerow(row)
        self.gpioLOW(16)

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

    def directionIndication(self, max_class, DIRECTION_FLAG):
        '''
        left  - 0
        right - 1
        '''
        if (max_class == 'right' and DIRECTION_FLAG == 0):
            self.gpioLOW(16)
        elif (max_class == 'right' and DIRECTION_FLAG == 1):
            self.HW_EVENT += 1
            self.gpioHIGH(16)
        elif (max_class == 'left' and DIRECTION_FLAG == 0):
            self.gpioHIGH(16)
            self.HW_EVENT += 1
        elif (max_class == 'left' and DIRECTION_FLAG == 1):
            self.gpioLOW(16)


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
            if (self.csvData[2][-2] <= self.lower):
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


    def getGradients(self, first_peak, second_peak, last_peak, second_last_peak, stop_index, start_index):

        gradient = ((last_peak + second_last_peak) / 2 - (first_peak + second_peak) / 2) / (
                stop_index - start_index)

        return gradient


    def printPeaksTest(self):
        print("Here are the peaks: {0}, {1}, {2}, {3}".format(self.first_peak, self.second_peak,
                                                              self.second_last_peak, self.last_peak))
        print("Here are the peaks: {0}, {1}, {2}, {3}".format(self.first_peak_2, self.second_peak_2,
                                                              self.second_last_peak_2,
                                                              self.last_peak_2))

    def oneHotPeaks(self, peaks):
        for i in range(len(peaks)):
            peaks[i] = 1 if peaks[i] >= 2.5 else 0

        return peaks

    def refreshFeatures(self):
        self.start_index, self.stop_index, self.first_peak, self.first_peak_2, self.second_peak, self.second_peak_2, self.last_peak, self.last_peak_2, self.end_time, self._max_peak, self._min_peak = \
            None, None, None, None, None, None, None, None, None, 2.5, 2.5
        self.second_last_peak, self.second_last_peak_2 = None, None
    
    def softwareBtnInterrupt(self, channel):
        self.HW_FLAG = True
        print("flag set")
    
    def testBtnInterrupt(self):
        print("starting test")
        tik = time.time()
        while(True):
            if(time.time() - tik > 10):
                print("test done")
                return
    
    def gpioHIGH(self,gpio):
        GPIO.output(gpio, 1)
    
    def gpioLOW(self,gpio):
        GPIO.output(gpio, 0)

if __name__=="__main__":
    HW_FLAG = False
    HW_system = system_main(15, 3, window_thresholds=[2.10,2.90])
    #HW_system.testBtnInterrupt()
    HW_system.runCollection(900)
    
    
