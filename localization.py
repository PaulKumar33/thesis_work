import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
from PyQt5 import QtWidgets
import sys
from pyqtgraph.Qt import QtGui, QtCore
from random import randint
import os
import time
import csv
import pandas as pd
import impurity
import json

import mcp_3008_driver as mcp
import RPi.GPIO as GPIO

class Plot2D(QtWidgets.QMainWindow):
    def setup_gpios(self):
        GPIO.setmode(GPIO.BCM)
        with open('./config.json') as f:
            data = json.load(f)
            for item in data.items():
                io = GPIO.OUT if item[1][0] == "out" else GPIO.IN
                if(item[1][0] == "in"):
                    logic = GPIO.PUD_DOWN if item[1][0] == "down" else GPIO.PUD_UP
                    GPIO.setup(int(item[0]), io, logic)
                if(item[1][0] == "out"):
                    GPIO.setup(item[0], io)

    def setGPIOLow(self):
        '''called on setup. forces IOs low'''
        for io in self.outputGPIO:
            GPIO.output(int(io), 0)
    
    def clear_features(self):
        self.trigger_cnt = 0
        self.p2_peaks, self.p1_peaks = [], []
        self.first_trigger, self.last_trigger = None, None
        s1_p1, s1_p2, s2_p1, s2_p2 = None, None, None, None
        s1_gr, s2_gr = None, None 


    def hw_isr(self, e):
        '''pause execution and edit response'''
        print("HW recorded")
        if(self.flags["TRIG"]):
            #if the device was triggered, set
            print("Recording for recent event")
            self.globals["COMPLETED_HW"] += 1
        else:
            #if the device is not triggered, count new event as well
            print("recording as new record")
            self.globals["COMPLETED_HW"] += 1
            self.globals["HW_EVENTS"] += 1
        self.globals["SUCCESS_TRIG"] = time.time()
        self.flags["DATA"] = False
        self.flags["TRIG"] = False

        #set the timers to neg times
        self.globals["TRIG_TIME"] = -3*60
        
        cnt = 0
        self.lower_buzz()
        while(cnt <= 2):
            self.LED_indicator(1)
            time.sleep(0.65)
            self.LED_indicator(1)
            time.sleep(0.65)
            cnt += 1
            print(">>>attempt")
        self.clear_features()
            

    
    def __init__(self, *args, **kwargs):
        super(Plot2D, self).__init__(*args, **kwargs)

        '''
        direction of interest 0 - left, 1 - right
        '''
        self.flags={
            "HW_TIMER":True,
            "DIRECTION":1,
            "DATA":True,
            "TRIG": False,
            "BUZZ": False
        }

        '''
        trigger time is set for either a right or left direction. if it is not a direction of interest,
        we set a timer for 5 mins. If an individual returns within 5 minutes the event is not recorded
        
        hw trig time tracks the time from the last HW event
        
        threshes set times for when a HW is valid or when a non interesting direction gets cleard
        '''
        self.default_trig, self.default_hw_trig, self.default_success = -3*60, -2.5*60, -0.5*60
        self.globals = {
            "COMPLETED_HW": 0,
            "HW_EVENTS": 0,
            "BUZZER_TIME": -0.55,
            "TRIG_TIME": -3*60,
            "HW_TRIG_TIME": -2.5*60,
            "SUCCESS_TRIG": -.5*60,
            "TRIG_THRESH": 3*60,
            "HW_TIMER_THRESH": 2*60,
            "BUZZER_THRESH": 0.55,
            "SUCCESS_TIMER_THRESH": 0.5*60,
            "LAST_DIR": None,
        }
        print(">>> Starting up")
        print(">>> Flags: ")
        
        for key in self.flags.keys():
            print("{0}: {1}".format(key, self.flags[key]))
            time.sleep(0.1)
        
        time.sleep(0.5)
        print(">>> Globals")
        for key in self.globals.keys():
            print("{0}: {1}".format(key, self.globals[key]))
            time.sleep(0.1)

        self.e_buffer_len = 10
        self.e_buffer_1 = [0 for i in range(self.e_buffer_len)]
        self.e_buffer_2 = [0 for i in range(self.e_buffer_len)]

        print(">>> Setting up GPIO")
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(16, GPIO.OUT)
        GPIO.setup(2, GPIO.OUT)
        GPIO.setup(24, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.add_event_detect(24, GPIO.RISING, bouncetime=3000, callback=self.hw_isr)

        self.outputGPIO = [2,16]
        self.setGPIOLow()
        time.sleep(0.5)
        print("Done")
        

        """self.plot = pg.PlotWidget()
        self.setCentralWidget(self.plot)"""

        w = pg.GraphicsLayoutWidget(size=(1000, 750))

        #for signal 1
        self.plot = w.addPlot(row=0, col=0)

        #for signal 2
        self.plot2 = w.addPlot(row=0, col=1)

        #for the std1
        self.plot3 = w.addPlot(row=1, col=0)
        #for std2
        self.plot4 = w.addPlot(row=1, col=1)
        #for acculated
        self.plot5 = w.addPlot(row=2, col=0)
        self.plot6 = w.addPlot(row=2, col=1)
        self.plot7 = w.addPlot(row=3, col=0)

        self.setCentralWidget(w)
        self.x = np.arange(0,5.01, 0.01)
        self.y = [randint(-10,10) for _ in range(len(self.x))]

        if(os.name == 'posix'):
            self.plot_bool = False
            print("Running ADC")
            self.mcp = mcp.mcp_external()

            self.buffer = 8
            self.N = 3
            self.N2 = 3
            self.M=39
            self.hd = self.build_fir_square(self.M, np.pi/3)

            self.calcSS()

            self.var_1 = [0 for i in range(128)]
            self.var_2 = [0 for i in range(128)]
            
            #lets fill the arrays first
            self.t = []
            self.x1 = []
            self.x2 = []
            self.y1 = []
            self.y2 = []
            self.p1 = [0 for i in range(128)]
            self.p2 = [0 for i in range(128)]
            self.trigger = [0 for i in range(128)]
            self.p1_peaks = []
            self.p2_peaks = []
            self.p1_t = []
            self.p2_t = []
            self.first_trigger = None
            self.last_trigger = None


            #set tracking variables for csv
            self.tt  = []
            self.ty1 = []
            self.ty2 = []
            self.te1 = []
            self.te2 = []
            self.tp1 = []
            self.tp2 = []
            self.cnt = 0
            self.trigger_cnt = 0
            self.flash_cnt = 0
            self.flash = 1

            df = pd.read_csv("./localization.csv")
            # df = df.drop(columns=['figure'])
            df_mat = df.values.tolist()

            df_direction = pd.read_csv("./direction_data.csv")
            mf_mat_direction = df_direction.values.tolist()

            self.tree = impurity.build_tree(df_mat)
            self.tree_dir = impurity.build_tree(mf_mat_direction)
            impurity.print_tree(self.tree)
            impurity.print_tree(self.tree_dir)
            print("starting capture")


            self.e1 = [0 for i in range(128)]
            self.e2 = [0 for i in range(128)]
            
            self.schmit_trig = 0
            
            for i in range(128):
                self.t.append(i)
                x1 = float(self.mcp.read_IO(0)/65355*5)
                x2 = float(self.mcp.read_IO(1) / 65355 * 5)
                if(len(self.x1) > self.M-1):
                    self.x1 = np.concatenate((self.x1, [x1]), axis=None)
                    self.x2 = np.concatenate((self.x2, [x2]), axis=None)
                    y1 = self.filt(self.hd, self.x1[-self.M:])
                    self.y1 = np.concatenate((self.y1, [y1]), axis=None)
                    self.y2 = self.x2
                    
                else:
                    self.x1.append(x1)
                    self.x2.append(x2)
                    self.y1 = self.x1
                    self.y2 = self.x2
            
            '''self.d = self.plot.plot(self.t, self.x1)
            self.d1 = self.plot2.plot(self.t, self.x2)
            self.pl_var_1 = self.plot3.plot(self.t, self.var_1)
            self.pl_var_2 = self.plot4.plot(self.t, self.var_2)
            self.pl_p1 = self.plot5.plot(self.t, self.p1)
            self.pl_p2 = self.plot6.plot(self.t, self.p2)
            self.t_plot = self.plot7.plot(self.t, self.trigger)'''
            self.runCapture()

        # plot data: x, y values
        else:
            self.d = self.plot.plot(self.x, self.y)

            ##init the timer for real time plotting
            self.timer = QtCore.QTimer()
            self.timer.setInterval(10) #50ms
            self.timer.timeout.connect(self.update_real_time)
            self.timer.start()

    def runCapture(self):
        self.timer = QtCore.QTimer()
        self.timer.setInterval(50)  # 50ms
        self.timer.timeout.connect(self.update_adc_measurement)
        self.timer.start()

    def update_adc_measurement(self):
        self.lower_buzz()

        x1 = float(self.mcp.read_IO(0)/65355*5)
        x2 = float(self.mcp.read_IO(1) / 65355 * 5)
        ttemp = self.t[1:] if len(self.t[1:]) <= 128 else self.t[1:128]
        x1temp = self.x1[1:] if len(self.x1[1:]) <= 128 else self.x1[1:128]
        x2temp = self.x2[1:] if len(self.x2[1:]) <= 128 else self.x2[1:128]
        #x1 = self.update_array_movag(x1, x1temp[-1-self.N2+1:-1], self.N2)
        #x2 = self.update_array_movag(x2, x2temp[-1-self.N2+1:-1], self.N2)
        self.x1 = np.concatenate((x1temp, [x1]), axis=None)
        self.x2 = np.concatenate((x2temp, [x2]), axis=None)
        
        y1 = self.filt(self.hd, self.x1[-self.M:])
        y1temp = self.y1[1:] if len(self.y1[1:]) <= 128 else self.y1[1:128]
        self.y1=np.concatenate((y1temp, [y1]), axis=None)
        
        y2 = self.filt(self.hd, self.x2[-self.M:])
        y2temp = self.y2[1:] if len(self.y2[1:]) <= 128 else self.y2[1:128]
        self.y2=np.concatenate((y2temp, [y2]), axis=None)
        
        self.t = np.concatenate((ttemp, [ttemp[-1]+1]), axis=None)

        adj = 127-self.buffer
        vmu_1, vmu_2 = self.y1[adj], self.y2[adj]
        sigma1, sigma2 = 0,0
        e1, e2 =0, 0
        for k in range(1, self.buffer):
            var1 = vmu_1+(1/self.buffer)*(self.y1[adj+k]-vmu_1)
            var2 = vmu_2+(1/self.buffer)*(self.y2[adj+k]-vmu_2)
            v_partial_1 = sigma1+(self.y1[adj+k]-var1)*(self.y1[adj+k]-var1)
            v_partial_2 = sigma1+(self.y2[adj+k]-var2)*(self.y2[adj+k]-var2)
            e1 += np.power(np.abs(self.y1[adj+k] - self.ss1),2)
            e2 += np.power(np.abs(self.y2[adj + k] - self.ss2), 2)

        self.var_1 = self.var_1[1:] if len(self.var_1[1:]) <= 128 else self.var_1[1:128]
        #v_partial_1 = self.update_array_movag(v_partial_1, self.var_1[-1-self.N2+1:-1], self.N2)
        self.var_1.append(v_partial_1)
        
        self.var_2 = self.var_2[1:] if len(self.var_2[1:]) <= 128 else self.var_2[1:128]
        #v_partial_2 = self.update_array_movag(v_partial_2, self.var_2[-1-self.N2+1:-1], self.N2)
        self.var_2.append(v_partial_2)
        
        e1temp = self.e1[1:] if len(self.e1[1:]) <= 128 else self.e1[1:128]
        self.e1=np.concatenate((e1temp, [e1]), axis=None)
        
        e2temp = self.e2[1:] if len(self.e2[1:]) <= 128 else self.e2[1:128]
        self.e2=np.concatenate((e2temp, [e2]), axis=None)
        
        total = e1+e2
        
        p1 = e1/total
        p2 = e2/total

        if(self.flags["DATA"] == False and np.abs(time.time() - self.globals["SUCCESS_TRIG"])>=self.globals["SUCCESS_TIMER_THRESH"]):
            #start collecting again
            print(">>> Running collection")
            self.flags["DATA"] = True

        if(self.flags["TRIG"] and np.abs(time.time()-self.globals["HW_TRIG_TIME"])>=self.globals["HW_TIMER_THRESH"]):
            #set trig flag to low if outside of the window
            print(">>> Lowering trigger")
            self.flags["TRIG"] = False
            self.flash_cnt = 0

        #implement the schmitt trigger
        if(e1 >= 0.6 or e2 >= 0.6):
            self.LED_indicator(1)
            # get the first sensor high
            self.trigger_cnt += 1
            if(self.first_trigger == None and p1 >= p2):
                self.first_trigger = 1
            elif(self.first_trigger == None and p1 < p2):
                self.first_trigger = 0
                
            if(p1 >= p2):
                self.last_trigger = 1
            else:
                self.last_trigger = 0
            
            if(e1 >= 0.6 and e2 >= 0.6):
                if(self.first_trigger == 0 and self.flags["DIRECTION"] == 1 and self.flags["BUZZ"]==False):
                    self.buzzer_indicator(1)
                    self.globals["BUZZER_TIME"] = time.time()
                    self.flags["BUZZ"] = True
                elif(self.first_trigger == 1 and self.flags["DIRECTION"] == 0 and self.flags["BUZZ"]==False):
                    self.buzzer_indicator(1)
                    self.globals["BUZZER_TIME"] = time.time()
                    self.flags["BUZZ"] = True
                    

            #update the energy buffers
            self.e_buffer_1 = np.concatenate((self.e_buffer_1[1:-1], e1), axis=None)
            self.e_buffer_2 = np.concatenate((self.e_buffer_2[1:-1], e2), axis=None)

            self.schmit_trig = 1
            ttrigger = self.trigger[1:] if len(self.trigger[1:]) <= 128 else self.trigger[1:128]
            self.trigger = np.concatenate((ttrigger, [1]),axis=None)

            #this is for the location
            '''_classify = impurity.classify([p1,p2,e1,e2],
                                          self.tree)
            max_guess = 0
            max_class = None

            for _class_ in _classify:
                if (_classify[_class_] > max_guess):
                    max_class, max_guess = _class_, _classify[_class_]
            #print("Location: {}".format(max_class))'''
                
            if(np.abs(self.y1[-1] - self.ss1) >= 0.3):
                self.update_s1_peak()
            if(np.abs(self.y2[-1] - self.ss2) >= 0.3):
                self.update_s2_peak()
        elif(e1 <= 0.065 and e2 <= 0.065 and self.schmit_trig == 1):
            self.LED_indicator(1)
            self.schmit_trig = 0
            ttrigger = self.trigger[1:] if len(self.trigger[1:]) <= 128 else self.trigger[1:128]
            self.trigger = np.concatenate((ttrigger, [0]),axis=None)
            print("THIS IS A TEST FOR PEAKS")
            print(self.p1_peaks)
            print(self.p2_peaks)
            
            #lets get the features here
            if(self.trigger_cnt >= 15):
                if(len(self.p1_peaks) > 0):
                    s1_p1 = self.p1_peaks[0]
                    s1_p2 = self.p1_peaks[-1]
                    s1_gr = s1_p2 - s1_p1
                else:
                    s1_p1 = None
                    s1_p2 = None
                    s1_gr = 9999
                    
                if(len(self.p2_peaks) > 0):
                    s2_p1 = self.p2_peaks[0]
                    s2_p2 = self.p2_peaks[-1]
                    s2_gr = s2_p2 - s2_p1
                else:
                    s2_p1 = None
                    s2_p2 = None
                    s2_gr = 9999
                
                print("FEATURE SET")
                print("p11: {0} \n P12: {1} \n P21: {2} \n P22: {3}".format(s1_p1, s1_p2, s2_p1, s2_p2))
                print("g1: {0} \n g2: {1} \n t1: {2} \n t2: {3}".format(s1_gr, s2_gr, self.first_trigger, self.last_trigger))

                _classify = impurity.classify([self.first_trigger, self.last_trigger, s1_gr, s2_gr, s1_p1, s1_p2, s2_p1, s2_p2],
                                              self.tree_dir)
                max_guess = 0
                max_class = None

                for _class_ in _classify:
                    if (_classify[_class_] > max_guess):
                        max_class, max_guess = _class_, _classify[_class_]
                #print("Predicted: {}".format(max_class))
                if (self.flags["DATA"]):
                    print("Predicted: {}".format(max_class))
                    self.direction_classification(max_class)
                else:
                    print("Data low. Here is direction: {}".format(max_class))

                '''with open("direction_data.csv", "a") as dd:
                    writer = csv.writer(dd)
                    writer.writerow([self.first_trigger, self.last_trigger, s1_gr, s2_gr, s1_p1, s1_p2, s2_p1, s2_p2])'''
            self.clear_features()              
            '''self.d1.setData(self.t, self.y2)
            self.d.setData(self.t, self.y1)'''
            
        elif(self.schmit_trig == 0):
            if(self.flags["TRIG"]):
                if(self.flash_cnt % 25 == 0):
                    print("we are here")
                    self.LED_indicator(not self.flash)            
                self.flash_cnt += 1
            else:
                self.LED_indicator(0)
                self.lower_buzz()
                self.flags["BUZZ"] = False
                
            self.schmit_trig = 0
            self.trigger_ct = 0
            ttrigger = self.trigger[1:] if len(self.trigger[1:]) <= 128 else self.trigger[1:128]
            self.trigger = np.concatenate((ttrigger, [0]),axis=None)
        
        #simple direction estimation
        
        
        self.p1 = self.p1[1:] if len(self.p1[1:]) <= 128 else self.p1[1:128]
        p1 = self.update_array_movag(p1, self.p1[-1-self.N+1: -1], self.N)
        self.p1 = np.concatenate((self.p1, [p1]), axis=None)
        self.p2 = self.p2[1:] if len(self.p2[1:]) <= 128 else self.p2[1:128]
        p2 = self.update_array_movag(p2, self.p2[-1 - self.N + 1: -1], self.N)
        self.p2 = np.concatenate((self.p2, [p2]), axis=None)


        self.tt = np.concatenate((self.tt, [self.t[-1]]), axis=None)
        self.ty1 = np.concatenate((self.ty1, [self.y1[-1]]),axis=None)
        self.ty2 = np.concatenate((self.ty2, [self.y2[-1]]), axis=None)
        self.te1 = np.concatenate((self.te1, [self.e1[-1]]), axis=None)
        self.te2 = np.concatenate((self.te2, [self.e2[-1]]), axis=None)
        self.tp1 = np.concatenate((self.tp1, [self.p1[-1]]), axis=None)
        self.tp2 = np.concatenate((self.tp2, [self.p2[-1]]), axis=None)
        #movavg filter
        
        if(self.plot_bool):
            self.d.setData(self.t, self.y1)
            self.d1.setData(self.t, self.y2)
            self.pl_var_1.setData(self.t, self.e1)
            self.pl_var_2.setData(self.t, self.e2)
            self.pl_p1.setData(self.t, self.p1)
            self.pl_p2.setData(self.t, self.p2)
            self.t_plot.setData(self.t, self.trigger)

        self.cnt +=1
    
    def update_s1_peak(self):
        if (self.y1[-3] < self.y1[-2] and self.y1[-1] <= self.y1[-2] and np.abs(self.y1[-2] - self.ss1) >= 0.3):
            if (len(self.p1_peaks) > 0):
                if (abs(self.p1_peaks[-1] - self.y1[-2]) >= 0.5):
                    self.p1_peaks.append(self.y1[-2])
                    self.p1_t.append(self.t[-2])
            else:
                self.p1_peaks.append(self.y1[-2])
                self.p1_t.append(self.t[-2])
        elif (self.y1[-3] <= self.y1[-2] and self.y1[-1] < self.y1[-2] and np.abs(self.y1[-2] - self.ss1) >= 0.3):
            if (len(self.p1_peaks) > 0):
                if (abs(self.p1_peaks[-1] - self.y1[-2]) >= 0.5):
                    self.p1_peaks.append(self.y1[-2])
                    self.p1_t.append(self.t[-2])
            else:
                self.p1_peaks.append(self.y1[-2])
                self.p1_t.append(self.t[-2])

        elif (self.y1[-3] >= self.y1[-2] and self.y1[-1] > self.y1[-2] and np.abs(self.y1[-2] - self.ss1) >= 0.3):
            if (len(self.p1_peaks) > 0):
                if (abs(self.p1_peaks[-1] - self.y1[-2]) >= 0.5):
                    self.p1_peaks.append(self.y1[-2])
                    self.p1_t.append(self.t[-2])
            else:
                self.p1_peaks.append(self.y1[-2])
                self.p1_t.append(self.t[-2])

        elif (self.y1[-3] > self.y1[-2] and self.y1[-1] >= self.y1[-2] and np.abs(self.y1[-2] - self.ss1) >= 0.3):
            if (len(self.p1_peaks) > 0):
                if (abs(self.p1_peaks[-1] - self.y1[-2]) >= 0.5):
                    self.p1_peaks.append(self.y1[-2])
                    self.p1_t.append(self.t[-2])
            else:
                self.p1_peaks.append(self.y1[-2])
                self.p1_t.append(self.t[-2])
        
            
    def update_s2_peak(self):
        if(self.y2[-3] < self.y2[-2] and self.y2[-1] <= self.y2[-2] and np.abs(self.y2[-2] - self.ss2) >= 0.3):
            if(len(self.p2_peaks) > 0):
                if(abs(self.p2_peaks[-1] - self.y2[-2]) >= 0.5):
                    self.p2_peaks.append(self.y2[-2])
                    self.p2_t.append(self.t[-2])
            else:
                self.p2_peaks.append(self.y2[-2])
                self.p2_t.append(self.t[-2])
        elif(self.y2[-3] <= self.y2[-2] and self.y2[-1] < self.y2[-2] and np.abs(self.y2[-2] - self.ss2) >= 0.3):
            if(len(self.p2_peaks) > 0):
                if(abs(self.p2_peaks[-1] - self.y2[-2]) >= 0.5):
                    self.p2_peaks.append(self.y2[-2])
                    self.p2_t.append(self.t[-2])
            else:
                self.p2_peaks.append(self.y2[-2])
                self.p2_t.append(self.t[-2])
            
        elif(self.y2[-3] >= self.y2[-2] and self.y2[-1] > self.y2[-2] and np.abs(self.y2[-2] - self.ss2) >= 0.3):
            if(len(self.p2_peaks) > 0):
                if(abs(self.p2_peaks[-1] - self.y2[-2]) >= 0.5):
                    self.p2_peaks.append(self.y2[-2])
                    self.p2_t.append(self.t[-2])
            else:
                self.p2_peaks.append(self.y2[-2])
                self.p2_t.append(self.t[-2])
                
        elif(self.y2[-3] > self.y2[-2] and self.y2[-1] >= self.y2[-2] and np.abs(self.y2[-2] - self.ss2) >= 0.3):
            if(len(self.p2_peaks) > 0):
                if(abs(self.p2_peaks[-1] - self.y2[-2]) >= 0.5):
                    self.p2_peaks.append(self.y2[-2])
                    self.p2_t.append(self.t[-2])
            else:
                self.p2_peaks.append(self.y2[-2])
                self.p2_t.append(self.t[-2])


    def LED_indicator(self, io):
        GPIO.output(16, io)
        
    def lower_buzz(self):
        if(np.abs(time.time() - self.globals["BUZZER_TIME"])):
            self.buzzer_indicator(0)
            #self.flags["BUZZ"] = False

    def buzzer_indicator(self, io):
        GPIO.output(2, io)

    def direction_classification(self, dir):
        if(dir != "right" and dir != "left"):
            print("Quadrent movement")
            self.globals["LAST_DIR"] = -1
        else:

            if(dir == 'right' and self.flags['DIRECTION'] == 1 or dir == "left" and self.flags["DIRECTION"] == 0):
                # if there is not a current active event
                if (np.abs(time.time() - self.globals["TRIG_TIME"]) >= self.globals["TRIG_THRESH"]
                        and np.abs(time.time() - self.globals["HW_TRIG_TIME"]) >= self.globals["HW_TIMER_THRESH"]):
                    print("Event Captured")
                    self.buzzer_indicator(1)
                    self.globals["BUZZER_TIME"] = time.time()
                    #set the trig time
                    self.globals["HW_TRIG_TIME"] = time.time()
                    self.globals['HW_EVENTS'] += 1
                    self.flags["TRIG"] = True
                    self.flags["DATA"] = False
                else:
                    print("One threshold was not exceeded")
                    print("Timer Flag: {}".format(np.abs(time.time() - self.globals["TRIG_TIME"]) >= self.globals["TRIG_THRESH"]))
                    print("HW Timer Flag: {}".format(
                        np.abs(time.time() - self.globals["HW_TRIG_TIME"]) >= self.globals["HW_TIMER_THRESH"]))
            else:
                print("Direction of non interest")
                print("Setting timer")
                if(self.flags["TRIG"] == False):
                    #self.flags["DATA"] = False
                    self.globals["TRIG_TIME"] = time.time()
                self.globals["LAST_DIR"] = 0

        #self.globals["LAST_DIR"] = dir


    def update_array_movag(self, pt, arr, N):
        return 1/N*(sum(arr)+pt)
    
    def filt(self, h, x):
        return np.dot(h, np.flip(x))
    
    def build_fir_square(self, width, wc):
        '''builds the fir window square filter'''
        M = width
        n = np.arange(0,M,1)
        
        inner = wc*(n-(M-1)/2)
        hd = np.sin(inner)/(np.pi*(n-(M-1)/2))
        hd[int((M-1)/2)] = wc/np.pi
        
        return hd

    def calcSS(self):
        mk0 = 0
        mk1 = 0
        cnt = 1
        while(cnt <= 1000):
            if(cnt%100 == 0):
                print("{}% done".format(cnt/1000.0 * 100))
            x1 = float(self.mcp.read_IO(0) / 65355 * 5)
            x2 = float(self.mcp.read_IO(1) / 65355 * 5)

            mk0 = mk0*(cnt-1.0)/cnt + x1/cnt
            mk1 = mk1 * (cnt - 1.0) / cnt + x2 / cnt
            
            cnt+=1

        print(">>>Steady state calculated: {0}, {1}".format(mk0, mk1))
        self.ss1 = mk0
        self.ss2 = mk1


    def update_real_time(self):
        self.x=np.concatenate((self.x[1:], [self.x[-1]+0.01]), axis=None)
        self.y=np.concatenate((self.y[1:], [randint(-10,10)]), axis=None)

        self.d.setData(self.x, self.y)

    def setTitle(self, str):
        self.plot.setTitle(str)

    def setXAxis(self, str, style):
        #styles = {'font-size': '20px'}
        self.plot.setLabel('bottom', str)

    def setYAxis(self, str, style={}, loc='left'):
        self.plot.setLabel(loc, str)

    def grid(self, x=True, y=True):
        self.plot.showGrid(x,y)

    def yLimits(self, lims, e=0):
        self.plot.setYRange(lims[0], lims[1], padding=e)

    def limits(self, lims):
        try:
            if(isinstance(lims, list) == False):
                raise("TypeError: input should be list")
            else:
                lims = [lims[0], lims[1], 0, lims[2]] if(len(lims) == 3) else lims
                lims = [lims[0], lims[1], lims[0], lims[1]] if len(lims) == 2 else lims
                lims = [lims[0], lims[0], lims[0], lims[0]] if len(lims) == 1 else lims
                self.plot.setXRange(lims[0], lims[1], padding=0)
                self.plot.setYRange(lims[2], lims[3], padding=0)
        except Exception as e:
            print(e)

    def clearPlot(self):
        self.plot.clear()


    def start(self):
        if((sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION")):
            QtGui.QGuiApplication.instance().exec_()

    def runADC(self):
        pass


def main():
    app = QtWidgets.QApplication(sys.argv)
    main = Plot2D()
    main.setTitle("Dummy figure")
    main.setXAxis("time [h]", {'color':'r', 'font-size':'20px'})
    main.setYAxis("Temperature ")
    #main.yLimits([-10, 10], 2)
    #main.limits([0, 3.05, -10, 10])
    main.grid()
    main.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()