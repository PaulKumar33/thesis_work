import serial
import datetime
import struct
import time
import statistics
import csv
#import scipy.stats
import pandas as pd
import os
import signal
import atexit


import mcp_3008_driver as mcp
import RPi.GPIO as GPIO

import matplotlib.pyplot as plt
from decision_tree import impurity


def softwareBtnInterrupt(channel):
    FLAGS["HW_FLAG"] = True
        
def softwareBuzzInteruppt():
    FLAGS["BUZZ"] = True

#HW_FLAG = False #this should be set to true during HW event
FLAGS={
    "HW_FLAG": False
    }
class system_main:

    def handleExit(self, signum, frame):
        self.gpioLOW(2)
        self.gpioLOW(16)
    
    def __init__(self, low_delay, wait_time, peak_method="peak_detection", window_thresholds=[2.25, 2.75],
               variance_threshold=0.095):
        atexit.register(self.handleExit, None, None)
        signal.signal(signal.SIGTERM, self.handleExit)
        signal.signal(signal.SIGINT, self.handleExit)
        self.filter_length = 3
        self.buffer_length = 16 #length of the variance data buffer
        self.low_delay = low_delay
        self.thresh_holds = window_thresholds
        self.variance_limit = variance_threshold

        self.peak_method = peak_method

        self.csv_write = []
        self.csvData = [[],[],[]]
        
        self.RECORD_FLAG = False
        self.RECENT_HW_FLAG = False
        self.HW_EVENT = 0
        self.HW_COUNT = 0
        self.wait_time = wait_time
        
        #setu0p the GPIOs for interrupts
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(24, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.add_event_detect(24, GPIO.RISING, callback=softwareBtnInterrupt, bouncetime=1500)
        
        GPIO.setup(16, GPIO.OUT)
        GPIO.setup(2, GPIO.OUT)
        self.gpioLOW(2)
        
        self.gpioHIGH(2)
        self.gpioHIGH(16)
        time.sleep(1)
        
        self.gpioLOW(2)
        self.gpioLOW(16)
        

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
                df = pd.read_excel("./decision_data_new.xlsx", "sys_wall")
                # df = df.drop(columns=['figure'])
                df_mat = df.values.tolist()
                
                self.tree = impurity.build_tree(df_mat)
                impurity.print_tree(self.tree)
                print("starting capture")

        except Exception as e:
            print(e)

    def runCollection(self, sample_points=100):
        
        DIRECTION_FLAG = 0

        impurity.print_tree(self.tree)
        print('sleeping for test')
        print(FLAGS)
        time.sleep(self.wait_time)

        self.upper = self.thresh_holds[1]
        self.lower = self.thresh_holds[0]

        self.var_limit = 0.025

        tik = time.time()

        #0 first peak, 1 last peak. 2 first peak_2 3 last peak_2
        self.peak_points = [None, None, None, None]

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
        _evt_time = 0

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

        self.DATA_STOP = False                          #RAISED WHEN THE DATA STOP EVENT OCCURS
        self.BUZZ_FLAG = False                          #RAISED WHEN THE BUZZER BEEPS
        self.LAST_DIRECTION = 0                         #RAISED AT ANY DIRECTION EVENT
        self.STOP_TIME = 0                              #ACCUMULATES FOR WHEN THE STOP EVENT IS RAISED
        self.LAST_TRIGGER_TIME = -1                     #TRACKS THE TIME OF THE LAST CLASSIFICATION
        self.LAST_HW_TIME = -1                          #TRACKS THE TIME OF THE LAST HW
        self.HW_DELAY_TIME = 2.5*60                     #HW DELAY TIME DEFAULT
        
        self.CUE_FLAG = True
        
        self.CURRENT_STATUS = 'NO_CUE'
        self.curr_time_track = datetime.datetime.today().strftime("%M")
        

        '''
        
        THE LAST DIRECTION FLAG SHOULD BE FALSE IF ITS NOT THE HW DIRECTION
        
        '''


        '''
        
        A FEW NOTES ON THE FLOW:
        
        1) IF A HW EVENT IS CLASSIFIED, WE SET THE STOP TIME TO A DEFAULT AMOUNT OF TIME - USUALLY DEFAULTS 5 MIN
        
        2) IF THERE IS A RETURN EVENT (IE GIVEN LEFT IS HW, A NO-LEFT OR RIGHT OCCURS AND THEN WITH 5 MIN A LEFT OCCURS), 
        DO NOT CLASSIFY THE EVENT AS A HW EVENT. IF A HW OCCURS, RECORD THE EVENT
        
        3) IF A TYPICAL EVENT OCCURS, RECORD THE EVENT
        
        4) IF A HW OCCURS, DISABLE THE TRACKING FOR 5 MIN OR THE DEFAULT TIME 
        
        '''


        counter = 5
        print("start")
        cnt2 = 0
        buzz_tik = 0
        while(True):
            #for the sample rate
            time.sleep(0.03)
            if(int(datetime.datetime.today().strftime("%M"))-int(self.curr_time_track)>= 1):
                with open('hw_tracking.csv', 'a') as fd:
                    # fd.write(self.csv_write)
                    print("writitng")
                    writer = csv.writer(fd)                    
                    writer.writerow([self.HW_EVENT,self.HW_COUNT,self.CURRENT_STATUS, datetime.datetime.today().strftime("%Y-%m-%d:%H-%M")])
                    self.curr_time_track = datetime.datetime.today().strftime("%M")
                
            
            if(datetime.datetime.today().strftime("%Y-%m-%d:%H-%M") == "2021-04-04:15-30"):
                self.CUE_FLAG = True
                self.CURRENT_STATUS = "CUES"
            
            if(self.BUZZ_FLAG == False):
                #print("setting low")
                self.gpioLOW(2)
            
            if(self.BUZZ_FLAG == True and time.time() - buzz_tik >= 0.25):
                print("chimcken")
                self.gpioLOW(2)
                self.BUZZ_FLAG = False

            if(len(self.csvData[0]) > 2048):
                self.csvData[0] = self.csvData[0][-128:]
                self.csvData[1] = self.csvData[1][-128:]
                self.csvData[2] = self.csvData[2][-128:]
                print('truncating')

            if(FLAGS["HW_FLAG"]):
                print("PRINTING FLAGS")
                self.printFlags()

                if(time.time() - self.LAST_TRIGGER_TIME < 5*60 and self.LAST_DIRECTION == 1):
                    self.HW_COUNT += 1
                else:
                    self.HW_COUNT += 1  # need to check if we recorded the wash. Set the flag low after its completed
                    self.HW_EVENT += 1

                self.LAST_DIRECTION = 0
                self.LAST_HW_TIME = time.time()

                FLAGS["HW_FLAG"] = False

                self.DATA_STOP = True                                   #set the stop collection flag
                self.STOP_TIME = time.time()
                print(f"Event recorded: ")
                continue

            if(time.time() - self.STOP_TIME > 6.5 and self.DATA_STOP == True ):                
                print("recordin starting again")
                self.DATA_STOP = False
                self.DATA_STOP = False
                
                print("FLAGS")
                self.printFlags()
                

            if (True):
                t = time.time() - tik                                   #implement the moving average filter - before the buffer fills. if (len(self.csvData[1]) < self.filter_length):
                self.csvData[0].append(t)
                self.csvData[1].append(float(self.mcp.read_IO(0) / 65355 * 5))
                self.csvData[2].append(float(self.mcp.read_IO(1) / 65355 * 5))

            #after the buffer fills
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

                '''
                ##############################################################################
                #BEGING THE TRACKING AT THIS POINT. WHEN VARIANCE EXCEEDS A CERTAIN THRESHOLD#
                ##############################################################################
                '''


                if ((Sk / self.buffer_length < self.var_limit and self.DATA_STOP == False) and
                        (Sk_2/self.buffer_length < self.var_limit and self.DATA_STOP == False)):
                    _evt_time = 0
                                        

                    #if data stop high, we do not care about the direction classification, we record the HW event
                    counter += 1
                    if (counter >= self.low_delay):

                        '''
                        ###########################################
                        #NO MORE SIGINIFICANT MOVEMENT IS DETECTED#
                        ###########################################
                        '''
                        
                        _event = False
                        #counter = 0
                        self.gpioLOW(16)
                        trig_tracker.append(0)
                        cond_test = self.stop_index != None and self.start_index != None and self.first_peak != None and self.last_peak != None or \
                            (self.peak_points[2] != None and self.peak_points[3] != None)
                        
                        if (self.stop_index != None and self.start_index != None and self.first_peak != None and self.last_peak != None or
                            (self.peak_points[2] != None and self.peak_points[3] != None)):

                            '''
                            #####################################
                            #CHECK IF THE WINDOW IS LARGE ENOGUH#
                            #####################################
                            '''
                            if(self.stop_index != None and self.start_index != None):
                                interval = self.stop_index - self.start_index
                            elif(self.peak_points[2] != None and self.peak_points[3] != None):
                                interval = self.peak_points[3] - self.peak_points[2]

                            
                            if (interval > 1.25 ):
                                if(self.LAST_HW_TIME != -1 and time.time() - self.LAST_HW_TIME < self.HW_DELAY_TIME):
                                    """
                                    ###########################################################################
                                    #CHECK IF THE TIME HAS PASSED. IF THIS IS THE FIRST EVENT IT WILL NOT SKIP#
                                    ###########################################################################
                                    """
                                    continue

                                
                                '''
                                ##########################
                                #BEGIN THE CLASSIFICATION#
                                ##########################
                                '''
                                #compute the classification
                                try:
                                    self.printPeaksTest()
                                    if(self.peak_points[0] != None and self.peak_points[1] != None):
                                        gradient = self.getGradients(self.first_peak, self.second_peak, self.last_peak, self.second_last_peak, self.stop_index, self.start_index)
                                    else:
                                        gradient = 9999

                                    if(self.peak_points[2] != None and self.peak_points[3] != None):
                                        gradient_2 = self.getGradients(self.first_peak_2, self.second_peak_2, self.last_peak_2, self.second_peak_2, self.peak_points[3], self.peak_points[2])
                                    else:
                                        gradient_2 = 9999

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
                                self.peak_points = self.nullNonePeaks(self.peak_points)
                                
                                #get first peak differential
                                first_peak_differential = 1 if self.peak_points[0] < self.peak_points[2] else 0
                                last_peak_differential = 1 if self.peak_points[1] < self.peak_points[3] else 0

                                print(type(temp[0]), type(temp[1]), type(temp[4]), type(temp[6]), type(gradient))
                                #_classify = impurity.classify([temp[0], temp[1], gradient, temp[2], temp[3],
                                #                               gradient_2, temp[4], temp[5], temp[6],
                                #                               temp[7],first_peak_differential,
                                #                               last_peak_differential], self.tree)
                                #classify for the new datas
                                _classify = impurity.classify([gradient, gradient_2, first_peak_differential, last_peak_differential], self.tree)
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
                                     temp[3], gradient_2, temp[4], temp[5], temp[6], temp[7],first_peak_differential, last_peak_differential, _classify])

                            self.start_index, self.stop_index, self.first_peak, self.first_peak_2, self.second_peak, self.second_peak_2, self.last_peak, self.last_peak_2, self.end_time, self._max_peak, self._min_peak = \
                                None, None, None, None, None, None, None, None, None, 2.5, 2.5
                            self.second_last_peak, self.second_last_peak_2 = None, None
                            self.peak_points = [None, None, None, None]
                            

                    else:
                        trig_tracker.append(1)
                        _evt_time = 0
                        if(self.CUE_FLAG == True):
                            self.gpioHIGH(16)
                        
                        if (self.peak_method == 'peak_detection'):
                            # simple implementation of the peak detection algorithm
                            # find max
                            self.runPeakDetection(t)                    

                elif((Sk/self.buffer_length > self.var_limit and self.DATA_STOP == False) or
                     (Sk_2/self.buffer_length > self.var_limit and self.DATA_STOP == False)):
                    #we have detected movement and no HW event yet
                    '''
                    very easy, if both variance are high and first_peak_time_1 < first_peak_time_2 -> right
                    elif(opposite) -> lef
                    '''

                    b1 = self.peak_points[0] if self.peak_points[0] != None else -1
                    b2 = self.peak_points[2] if self.peak_points[2] != None else -1
                    if(b1<b2 and (b1!=-1 and b2 != -1) and DIRECTION_FLAG == 0):
                        if (self.BUZZ_FLAG == False and time.time() - buzz_tik >= 10 and self.CUE_FLAG and _evt_time >= 10):
                            print("THIS IS DIRECTIONF: ")
                            print(DIRECTION_FLAG)
                            self.gpioHIGH(2)
                            self.BUZZ_FLAG = True
                            buzz_tik = time.time()
                    elif(b1>b2 and (b1!=-1 and b2 != -1) and DIRECTION_FLAG == 1):
                        if(self.BUZZ_FLAG == False and time.time() - buzz_tik >= 10 and self.CUE_FLAG and _evt_time >= 10):
                            self.gpioHIGH(2)
                            self.BUZZ_FLAG = True
                            buzz_tik = time.time()
                    
                    if(self.BUZZ_FLAG == True and time.time() - buzz_tik >= 0.25):
                        self.gpioLOW(2)
                        
                    _event = True
                    if(self.CUE_FLAG == True):
                        self.gpioHIGH(16)
                    window.append(t)
                    trig_tracker.append(1)
                    counter = 0
                    _evt_time += 1

                    if(self.peak_method == "peak_detection"):
                        self.runPeakDetection(t)

                elif(self.DATA_STOP == True):
                    self.BUZZ_FLAG = False
                    self.gpioLOW(2)
                    _evt_time = 0
                    print("CONDITION 3")
                    self.gpioLOW(16)
                    trig_tracker.append(0)
                    Sk = 0
                    Sk_2 = 0

                trig_time.append(t)

            if (time.time() - tik > 7.5*60):
                tok = time.time()
                break

        print(f"elapsed time {tok - tik}")
        
        print(f"total HW events: {self.HW_EVENT}")
        print(f"total HW completed: {self.HW_COUNT}")

        with open('new_datas_sys2.csv', 'a') as fd:
            # fd.write(self.csv_write)
            writer = csv.writer(fd)
            for row in self.csv_write:
                writer.writerow(row)
                
                
                
    def printFlags(self):
        print(self.DATA_STOP, self.LAST_DIRECTION,
        self.STOP_TIME, self.LAST_TRIGGER_TIME,
        self.LAST_HW_TIME, self.HW_DELAY_TIME)


    def directionIndication(self, max_class, DIRECTION_FLAG):
        '''
        positive direction = 1
        negative direction = 0

        DIRECTION_FLAG - 0 IF LEFT 1 IF RIGHT
        '''
        if(max_class != 'right' and max_class != "left"):
            #if it is a no cross event
            self.gpioLOW(16)
            self.LAST_DIRECTION = 0
            print("LAST_DIRECTION SET: {}".format(self.LAST_DIRECTION))
            return

        elif (max_class != 'left' and DIRECTION_FLAG == 0):
            self.gpioLOW(16)
            self.LAST_DIRECTION = 0
            self.LAST_TRIGGER_TIME = time.time()
            print("Direction of no interest")
        elif (max_class == 'right' and DIRECTION_FLAG == 1):
            #TRIGGER THE EVENT IF A RECENT LEAVE HASNT OCCURRED# ----> LAST_DIRECTION == 0 IS NEGATIVE DIRECTION
            if (time.time() - self.LAST_TRIGGER_TIME > self.HW_DELAY_TIME
                or self.LAST_TRIGGER_TIME == -1):
                #IF LAST WAS LEFT AND COME BACK WITHIN 5
                self.HW_EVENT += 1
                if(self.CUE_FLAG == True):
                    self.gpioHIGH(16)
                self.LAST_DIRECTION = 1
                print("Event captured")
            elif(self.LAST_DIRECTION == 0):
                if(self.CUE_FLAG == True):
                    self.gpioHIGH(16)
                self.LAST_DIRECTION = 1

            self.LAST_DIRECTION = 1
            self.LAST_TRIGGER_TIME = time.time()
            return


        elif (max_class == 'left' and DIRECTION_FLAG == 0):
            if(time.time() - self.LAST_TRIGGER_TIME > self.HW_DELAY_TIME or
               self.LAST_TRIGGER_TIME == -1):
                self.HW_EVENT += 1
                if(self.CUE_FLAG == True):
                    self.gpioHIGH(16)
                self.LAST_DIRECTION = 1
                print("Event capture")

            self.LAST_DIRECTION = 1
            self.LAST_TRIGGER_TIME = time.time()
            return
        
        elif (max_class != 'right' and DIRECTION_FLAG == 1):
            self.gpioLOW(16)
            self.LAST_DIRECTION = 0
            self.LAST_TRIGGER_TIME = time.time()
            print("Direction of no interest")


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
                    self.peak_points[0] = t

                if (self.last_peak != None and (self.last_peak - 2.5) / (self.csvData[1][-2] - 2.5) < 0):
                    self.second_last_peak = self.last_peak

                self.last_peak = self.csvData[1][-2]
                self.stop_index = t
                self.peak_points[1] = t

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
                    self.peak_points[0] = t

                if (self.last_peak != None and (self.last_peak - 2.5) / (self.csvData[1][-2] - 2.5) < 0):
                    self.second_last_peak = self.last_peak

                self.last_peak = self.csvData[1][-2]
                self.stop_index = t
                self.peak_points[1] = t

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
                    self.peak_points[0] = t

                if (self.last_peak != None and (self.last_peak - 2.5) / (self.csvData[1][-2] - 2.5) < 0):
                    self.second_last_peak = self.last_peak

                self.last_peak = self.csvData[1][-2]
                self.stop_index = t
                self.peak_points[1] = t
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
                        self.peak_points[2] = t

                    if (self.last_peak_2 != None and (self.last_peak_2 - 2.5) / (self.csvData[2][-2] - 2.5) < 0):
                        self.second_last_peak_2 = self.last_peak_2

                    self.last_peak_2 = self.csvData[2][-2]
                    self.stop_index_2 = t
                    self.peak_points[3] = t
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
                        self.peak_points[2] = t

                    if (self.last_peak_2 != None and (self.last_peak_2 - 2.5) / (self.csvData[2][-2] - 2.5) < 0):
                        self.second_last_peak_2 = self.last_peak_2

                    self.last_peak_2 = self.csvData[2][-2]
                    self.stop_index_2 = t
                    self.peak_points[3] = t

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
                        self.peak_points[2] = t

                    if (self.last_peak_2 != None and (self.last_peak_2 - 2.5) / (self.csvData[1][-2] - 2.5) < 0):
                        self.second_last_peak_2 = self.last_peak_2

                    self.last_peak_2 = self.csvData[2][-2]
                    self.stop_index_2 = t
                    self.peak_points[3] = t

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
                    self.peak_points[0] = t

                if (self.last_peak != None and (self.last_peak - 2.5) / (self.csvData[1][-2] - 2.5) < 0):
                    self.second_last_peak = self.last_peak

                self.last_peak = self.csvData[1][-2]
                self.stop_index = t
                self.peak_points[1] = t
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
                    self.peak_points[0] = t

                if (self.last_peak != None and (self.last_peak - 2.5) / (self.csvData[1][-2] - 2.5) < 0):
                    self.second_last_peak = self.last_peak

                self.last_peak = self.csvData[1][-2]
                self.stop_index = t
                self.peak_points[1] = t
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
                    self.peak_points[0] = t

                if (self.last_peak != None and (self.last_peak - 2.5) / (self.csvData[1][-2] - 2.5) < 0):
                    self.second_last_peak = self.last_peak

                self.last_peak = self.csvData[1][-2]
                self.stop_index = t
                self.peak_points[1] = t
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
                        self.peak_points[2] = t

                    if (self.last_peak_2 != None and (self.last_peak_2 - 2.5) / (self.csvData[2][-2] - 2.5) < 0):
                        self.second_last_peak_2 = self.last_peak_2

                    self.last_peak_2 = self.csvData[2][-2]
                    self.stop_index_2 = t
                    self.peak_points[3] = t
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
                        self.peak_points[2] = t

                    if (self.last_peak_2 != None and (self.last_peak_2 - 2.5) / (self.csvData[2][-2] - 2.5) < 0):
                        self.second_last_peak_2 = self.last_peak_2

                    self.last_peak_2 = self.csvData[2][-2]
                    self.stop_index_2 = t
                    self.peak_points[3] = t
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
                        self.peak_points[2] = t

                    if (self.last_peak_2 != None and (self.last_peak_2 - 2.5) / (self.csvData[2][-2] - 2.5) < 0):
                        self.second_last_peak_2 = self.last_peak_2

                    self.last_peak_2 = self.csvData[2][-2]
                    self.stop_index_2 = t
                    self.peak_points[3] = t
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
            if(peaks[i] != None):
                peaks[i] = 1 if peaks[i] >= 2.5 else 0
            else:
                peaks[i] = -1

        return peaks

    def nullNonePeaks(self, peaks):
        ret_peaks = [-1,-1,-1,-1]
        for index in range(len(peaks)):
            ret_peaks[index] = peaks[index] if peaks[index] != None else -1
        
        return ret_peaks


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
    HW_system = system_main(20, 3, window_thresholds=[2.10,2.90])
    #HW_system.testBtnInterrupt()
    HW_system.runCollection(600)
    
    
