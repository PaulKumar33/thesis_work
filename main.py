import machine
from machine import ADC
#import sdcard
import uos
import utime
import time
import math

_CMD_TIMEOUT = const(100)

_R1_IDLE_STATE = const(1 << 0)
# R1_ERASE_RESET = const(1 << 1)
_R1_ILLEGAL_COMMAND = const(1 << 2)
# R1_COM_CRC_ERROR = const(1 << 3)
# R1_ERASE_SEQUENCE_ERROR = const(1 << 4)
# R1_ADDRESS_ERROR = const(1 << 5)
# R1_PARAMETER_ERROR = const(1 << 6)
_TOKEN_CMD25 = const(0xFC)
_TOKEN_STOP_TRAN = const(0xFD)
_TOKEN_DATA = const(0xFE)


class SDCard:
    def __init__(self, spi, cs):
        self.spi = spi
        self.cs = cs

        self.cmdbuf = bytearray(6)
        self.dummybuf = bytearray(512)
        self.tokenbuf = bytearray(1)
        for i in range(512):
            self.dummybuf[i] = 0xFF
        self.dummybuf_memoryview = memoryview(self.dummybuf)

        # initialise the card
        self.init_card()

    def init_spi(self, baudrate):
        try:
            master = self.spi.MASTER
        except AttributeError:
            # on ESP8266
            self.spi.init(baudrate=baudrate, phase=0, polarity=0)
        else:
            # on pyboard
            self.spi.init(master, baudrate=baudrate, phase=0, polarity=0)

    def init_card(self):
        # init CS pin
        self.cs.init(self.cs.OUT, value=1)

        # init SPI bus; use low data rate for initialisation
        self.init_spi(100000)

        # clock card at least 100 cycles with cs high
        for i in range(16):
            self.spi.write(b"\xff")

        # CMD0: init card; should return _R1_IDLE_STATE (allow 5 attempts)
        for _ in range(5):
            if self.cmd(0, 0, 0x95) == _R1_IDLE_STATE:
                break
        else:
            raise OSError("no SD card")

        # CMD8: determine card version
        r = self.cmd(8, 0x01AA, 0x87, 4)
        if r == _R1_IDLE_STATE:
            self.init_card_v2()
        elif r == (_R1_IDLE_STATE | _R1_ILLEGAL_COMMAND):
            self.init_card_v1()
        else:
            raise OSError("couldn't determine SD card version")

        # get the number of sectors
        # CMD9: response R2 (R1 byte + 16-byte block read)
        if self.cmd(9, 0, 0, 0, False) != 0:
            raise OSError("no response from SD card")
        csd = bytearray(16)
        self.readinto(csd)
        if csd[0] & 0xC0 == 0x40:  # CSD version 2.0
            self.sectors = ((csd[8] << 8 | csd[9]) + 1) * 1024
        elif csd[0] & 0xC0 == 0x00:  # CSD version 1.0 (old, <=2GB)
            c_size = csd[6] & 0b11 | csd[7] << 2 | (csd[8] & 0b11000000) << 4
            c_size_mult = ((csd[9] & 0b11) << 1) | csd[10] >> 7
            self.sectors = (c_size + 1) * (2 ** (c_size_mult + 2))
        else:
            raise OSError("SD card CSD format not supported")
        # print('sectors', self.sectors)

        # CMD16: set block length to 512 bytes
        if self.cmd(16, 512, 0) != 0:
            raise OSError("can't set 512 block size")

        # set to high data rate now that it's initialised
        self.init_spi(1320000)

    def init_card_v1(self):
        for i in range(_CMD_TIMEOUT):
            self.cmd(55, 0, 0)
            if self.cmd(41, 0, 0) == 0:
                self.cdv = 512
                # print("[SDCard] v1 card")
                return
        raise OSError("timeout waiting for v1 card")

    def init_card_v2(self):
        for i in range(_CMD_TIMEOUT):
            time.sleep_ms(50)
            self.cmd(58, 0, 0, 4)
            self.cmd(55, 0, 0)
            if self.cmd(41, 0x40000000, 0) == 0:
                self.cmd(58, 0, 0, 4)
                self.cdv = 1
                # print("[SDCard] v2 card")
                return
        raise OSError("timeout waiting for v2 card")

    def cmd(self, cmd, arg, crc, final=0, release=True, skip1=False):
        self.cs(0)

        # create and send the command
        buf = self.cmdbuf
        buf[0] = 0x40 | cmd
        buf[1] = arg >> 24
        buf[2] = arg >> 16
        buf[3] = arg >> 8
        buf[4] = arg
        buf[5] = crc
        self.spi.write(buf)

        if skip1:
            self.spi.readinto(self.tokenbuf, 0xFF)

        # wait for the response (response[7] == 0)
        for i in range(_CMD_TIMEOUT):
            self.spi.readinto(self.tokenbuf, 0xFF)
            response = self.tokenbuf[0]
            if not (response & 0x80):
                # this could be a big-endian integer that we are getting here
                for j in range(final):
                    self.spi.write(b"\xff")
                if release:
                    self.cs(1)
                    self.spi.write(b"\xff")
                return response

        # timeout
        self.cs(1)
        self.spi.write(b"\xff")
        return -1

    def readinto(self, buf):
        self.cs(0)

        # read until start byte (0xff)
        for i in range(_CMD_TIMEOUT):
            self.spi.readinto(self.tokenbuf, 0xFF)
            if self.tokenbuf[0] == _TOKEN_DATA:
                break
            time.sleep_ms(1)
        else:
            self.cs(1)
            raise OSError("timeout waiting for response")

        # read data
        mv = self.dummybuf_memoryview
        if len(buf) != len(mv):
            mv = mv[: len(buf)]
        self.spi.write_readinto(mv, buf)

        # read checksum
        self.spi.write(b"\xff")
        self.spi.write(b"\xff")

        self.cs(1)
        self.spi.write(b"\xff")

    def write(self, token, buf):
        self.cs(0)

        # send: start of block, data, checksum
        self.spi.read(1, token)
        self.spi.write(buf)
        self.spi.write(b"\xff")
        self.spi.write(b"\xff")

        # check the response
        if (self.spi.read(1, 0xFF)[0] & 0x1F) != 0x05:
            self.cs(1)
            self.spi.write(b"\xff")
            return

        # wait for write to finish
        while self.spi.read(1, 0xFF)[0] == 0:
            pass

        self.cs(1)
        self.spi.write(b"\xff")

    def write_token(self, token):
        self.cs(0)
        self.spi.read(1, token)
        self.spi.write(b"\xff")
        # wait for write to finish
        while self.spi.read(1, 0xFF)[0] == 0x00:
            pass

        self.cs(1)
        self.spi.write(b"\xff")

    def readblocks(self, block_num, buf):
        nblocks = len(buf) // 512
        assert nblocks and not len(buf) % 512, "Buffer length is invalid"
        if nblocks == 1:
            # CMD17: set read address for single block
            if self.cmd(17, block_num * self.cdv, 0, release=False) != 0:
                # release the card
                self.cs(1)
                raise OSError(5)  # EIO
            # receive the data and release card
            self.readinto(buf)
        else:
            # CMD18: set read address for multiple blocks
            if self.cmd(18, block_num * self.cdv, 0, release=False) != 0:
                # release the card
                self.cs(1)
                raise OSError(5)  # EIO
            offset = 0
            mv = memoryview(buf)
            while nblocks:
                # receive the data and release card
                self.readinto(mv[offset : offset + 512])
                offset += 512
                nblocks -= 1
            if self.cmd(12, 0, 0xFF, skip1=True):
                raise OSError(5)  # EIO

    def writeblocks(self, block_num, buf):
        nblocks, err = divmod(len(buf), 512)
        assert nblocks and not err, "Buffer length is invalid"
        if nblocks == 1:
            # CMD24: set write address for single block
            if self.cmd(24, block_num * self.cdv, 0) != 0:
                raise OSError(5)  # EIO

            # send the data
            self.write(_TOKEN_DATA, buf)
        else:
            # CMD25: set write address for first block
            if self.cmd(25, block_num * self.cdv, 0) != 0:
                raise OSError(5)  # EIO
            # send the data
            offset = 0
            mv = memoryview(buf)
            while nblocks:
                self.write(_TOKEN_CMD25, mv[offset : offset + 512])
                offset += 512
                nblocks -= 1
            self.write_token(_TOKEN_STOP_TRAN)

    def ioctl(self, op, arg):
        if op == 4:  # get number of blocks
            return self.sectors


#reduce clock speed
machine.freq(25000000)

def format_time(tup):
    global initial_time
    return (utime.mktime(tup) - initial_time)/60

def write_to_device(message):
    with open("/sd/data_collection_test01.txt", "a+") as f:
        f.write("{}\r\n".format(message))
        f.close()
    #print(file_sys.read())
    return


#make an initial write

l = machine.Pin(25, machine.Pin.OUT)

LED_PIN  = 4
IR_PIN   = 3
BUZZ_PIN = 12
IR_IN    = 19
COMP_IN  = 18
WAKE_UP  = 15

ADC_0 = 26
ADC_1 = 27

vref = 5

##setup the device and the ios

#can use the third argument as PULL UP or PULL DOWN
led        = machine.Pin(LED_PIN, machine.Pin.OUT)
ir_out     = machine.Pin(IR_PIN, machine.Pin.OUT)
ir_in      = machine.Pin(IR_IN, machine.Pin.IN, machine.Pin.PULL_DOWN)
buzzer       = machine.Pin(BUZZ_PIN, machine.Pin.OUT)
comparator = machine.Pin(COMP_IN, machine.Pin.IN, machine.Pin.PULL_DOWN)
wake_up    = machine.Pin(WAKE_UP, machine.Pin.IN, machine.Pin.PULL_DOWN)
buzzer.value(0)

adc_0      = ADC(machine.Pin(ADC_0))
adc_1      = ADC(machine.Pin(ADC_1))

def flash_LED(num):
    cnt = 0
    while(cnt<num):
        led.value(1)
        utime.sleep(0.5)
        led.value(0)
        utime.sleep(0.5)
        cnt += 1

def build_fir_square(width, wc):
    '''builds the fir window square filter'''
    M, hd = width, [0 for i in range(width)]
    #n = np.arange(0, M, 1)
    #n = [i for i in range(0,M)]
    for i in range(0,M):
        print(i)
        inner = wc*(i-(M-1)/2)
        if(i-(M-1)/2 == 0):
            hd[int((M - 1) / 2)] = wc / math.pi
            continue
        hd[i] = math.sin(inner)/(math.pi*(i-(M-1)/2))
    '''inner = wc * (n - (M - 1) / 2)
    hd = math.sin(inner) / (math.pi * (n - (M - 1) / 2))
    hd[int((M - 1) / 2)] = wc / math.pi'''

    return hd

def calculateSS():
    mk0, mk1, cnt = 0, 0, 1
    global adc_0, adc_1, vref, ss1, ss2
    while(cnt <= 1000):
        if(cnt%100 == 0):
            print("{}% Done".format(cnt/1000.0 * 100))
        x1 = float(adc_0.read_u16()/65355*vref)
        x2 = float(adc_1.read_u16()/65355*vref)

        #recursive mean
        mk0 = mk0*(cnt-1)/cnt + x1/cnt
        mk1 = mk1*(cnt-1)/cnt + x2/cnt

        cnt +=1

    print("Steady state calculated: {0}, {1}".format(mk0, mk1))
    ss1 = mk0
    ss2 = mk1


#0b00000000001 - HW TIMER
#0b00000000010 - last direction
#0b00000000100 - data
#0b00000001000 - trig -> TRIGGERED DEVICE. IT HAS A DETECTED A DOI
#0b00000010000 - buzz
#0b00000100000 - cues -> use cues
#0b00001000000 - collect	-> SHOULD WE BE COLLECTING
#0b00010000000 - direction -> doi
#0b00100000000 - transition -> if a transition just occurred
#0b01000000000 - just left -> whether someone just left
#b100000000000 - just hw
hw_timer, last_dir, data, trig, buzz, cues, collect, direction, transition, just_left, just_hw = False,False,False,False,False,False,False,False,False,False,False
collect, direction = True, True
writing = False
FLAGS = 0b00000000000
HW_GLOBALS = {"COMPLETED_HW":0, "HW_EVENTS":0}
timers_irq = {"ir":-5*1000}
thresh_irq = {"ir": 5*1000}
timers     = {"BUZZ_TIME":-1500, "TRIG_TIME":10*1000,
              "HW_TRIG_TIME":-2.5*60*1000, "SUCCESS_TRIG":-30*1000,
              "TRANSITION_TIME": 0, 'COLLECTION_LEAVE_DELAY':-2*60*1000, 'COLLECTION_HW_DELAY': -2*60*1000}
thresh     = {"SUCCESS_THRESH":30*1000, "BUZZ_THRESH": 1000,
              "TRIG_THRESH": 1*60*1000, "HW_TIMER_THRESH":2*60*1000,
              "TRANS_BUZZ_THRESH": 750, 'COLLECTION_LEAVE_THRESH': 2*60*1000,
              "COLLECTION_HW_THRESH": 2*60*1000}

print("Defaults set")
flash_LED(2)

#convention - for peripherals we will use one _
#convention - for interrupt methods we will use __ 2

print("Peripherals Set")
flash_LED(3)

#lets setup the energy buffer
e_buffer_len = 10
e_buffer_1 = [0 for i in range(e_buffer_len)]
e_buffer_2 = [0 for i in range(e_buffer_len)]

#let the light go for 3 flashes again
flash_LED(3)

#lets set up the globals now
buffer = 8
N = 3
N2 = 3
M = 19
hd = build_fir_square(M, math.pi/3)
buzzer.value(1)
utime.sleep(0.25)
buzzer.value(0)
utime.sleep(0.25)
buzzer.value(1)
utime.sleep(0.25)
buzzer.value(0)
utime.sleep(0.25)

utime.sleep(30)

buzzer.value(1)
utime.sleep(0.25)
buzzer.value(0)
utime.sleep(0.25)
buzzer.value(1)
utime.sleep(0.25)
buzzer.value(0)
utime.sleep(0.25)
cs = machine.Pin(9, machine.Pin.OUT)
spi = machine.SPI(1, baudrate=1000000, polarity=0,phase=0,bits=8,firstbit=machine.SPI.MSB,sck=machine.Pin(10),mosi=machine.Pin(11),miso=machine.Pin(8))
sd = SDCard(spi,cs)
vfs = uos.VfsFat(sd)
uos.mount(vfs, "/sd")

filename = "data_collection_test01"

#set the local time to jan first
rtc = machine.RTC()
rtc.datetime((2021, 1, 1, 0, 0, 0, 0, 1))
initial_time = utime.mktime(utime.localtime())
with open("/sd/data_collection_test01.txt", "a+") as f:
    f.write("time, time_dif, event, indicators, hw, t_after, total_events, total_hw\r\n")
    f.close()
    
print("MAKING INITIAL WRITE")
utime.sleep(2)
#write_to_device("time, time_dif, event, indicators, hw, t_after, total_events, total_hw\r\n")
with open("/sd/data_collection_test01.txt", "r") as f:
    print(f.readlines())

utime.sleep(5)

ss1, ss2 = 0, 0
calculateSS()

#calculate the variances
var_1 = [0 for i in range(64)]
var_2 = [0 for i in range(64)]

#lets fill the arrays first
t = []
x1 = [0 for i in range(64)]
x2 = [0 for i in range(64)]
y1 = []
y2 = []
p1 = [0 for i in range(64)]
p2 = [0 for i in range(64)]
trigger = [0 for i in range(64)]
p1_peaks = []
p2_peaks = []
p1_t = []
p2_t = []
first_trigger = None
last_trigger = None
down_time = None

recent_buzz = 0

#set tracking variables for csv
cnt = 0
trigger_cnt = 0
flash_cnt = 0
flash = 1
last_trigger_time = 0
last_buzz = 0

#set cue and collect flags
#FLAGS = FLAGS | 0b00001100000
cues, collect = True, True


def _led_(io):
    '''this method turns the LED on or off'''
    led.value(io)
    
def _buzz_():
    global recent_buzz
    if(utime.ticks_ms() - recent_buzz > 750):
        recent_buzz = utime.ticks_ms()
        buzzer.value(not buzz.value())
    
def _ir_(io):
    ir.value(io)

def __ir__(pin):
    '''interrupts execution'''
    global  ir_in, FLAGS, timers_irq, just_left, collect, buzz, just_left, just_hw, hw_timer, trig, direction, first_trigger, writing
    ir_in.irq(handler=None)
    if(not hw_timer):
        hw_timer = True
        if(not trig):
            HW_GLOBALS["HW_EVENTS"] += 1
        HW_GLOBALS["COMPLETED_HW"] += 1
        
    used_cues = 1 if trig else 0
    entering = "entering" if (not direction and first_trigger == 0) or (direction and first_trigger == 1) or trig else "IR"
    if(writing == False):
        writing = True
        write_to_device("{0}, {1}, {2}, {3}, 1, {4}, {5}, {6}".format(str(utime.localtime()).replace("(", "").replace(")","").replace(", ", "-"), format_time(utime.localtime()), entering, used_cues, str(abs((last_trigger_time - utime.ticks_ms())/1000)) if abs(last_trigger_time - utime.ticks_ms()) <= 3*60*1000 else ">3min",
                                                                 HW_GLOBALS["COMPLETED_HW"], HW_GLOBALS["HW_EVENTS"]))
    
    collect, buzz, just_left, trig = False, False, False, False
    just_hw = True
    buzzer.value(0)
    timers_irq["ir"] = utime.ticks_ms()
    timers['COLLECTION_HW_DELAY'] = utime.ticks_ms()
    print("Detected")
    print("IR IRQ disabled")

def irq_sleep(pin):
    print("Waking up!")
    
def ir_down():
    '''this method makes sure the MOSFET controlling the ir sensor is down'''
    ir_out.value(0)
    
def ir_on():
    ir_out.value(1)
    
####the decision tree####
def decision_tree(t1, t2):
    if(t1 <= 0):
        if(t2 <= 0):
            return "left"
        else:
            return "no-right"
    elif(t1 >= 1):
        if(t2 >= 1):
            return "right"
        else:
            return "no-left"

def record(_direction):
    global last_trigger_time, trig, collect, direction, just_left
    if(collect and not trig):
        print(direction)
        if((direction == False and _direction == 'left') or (direction == True and _direction == 'right')):
            print("Recording event. Writing entering")

            #FOR BUZZ
            last_trigger_time, timers["BUZZ_TIME"] = utime.ticks_ms(), utime.ticks_ms()
            # FOR BUZZ

            collect = False
            #FLAGS &= 0b11110111111  # set the collection flag low, set timer
            timers["COLLECTION_LEAVE_DELAY"] = utime.ticks_ms()
            timers["TRIG_TIME"] = utime.ticks_ms()
            HW_GLOBALS["HW_EVENTS"] += 1
            trig, buzz = True, True
            buzzer.value(1)
        elif(_direction == 'right' or _direction == 'left'):
            print("Leaving. lowering collection for a few mins")
            print("writing leaving")
            write_to_device("{0}, {1}, leaving, 0, 0, 0, {2}, {3}".format(str(utime.localtime()).replace("(", "").replace(")","").replace(", ", "-"), format_time(utime.localtime()), HW_GLOBALS["COMPLETED_HW"], HW_GLOBALS["HW_EVENTS"]))
            timers["COLLECTION_LEAVE_DELAY"] = utime.ticks_ms()
            collect, just_left = False, True
            #FLAGS &= 0b11110111111  # set the collection flag low, set timer
            #FLAGS |= 0b01000000000  # set just left flag
    else:
        print("Not collecting")


#HARDWARE INTERRUPTS
ir_in.irq(handler=__ir__, trigger=machine.Pin.IRQ_RISING)
wake_up.irq(irq_sleep,
        machine.Pin.IRQ_RISING)


#run setup
print("power on...")
downtime = utime.localtime()[4]

l = machine.Pin(25, machine.Pin.OUT)
l.value(0)
time.sleep(1)
l.value(1)
time.sleep(1)
l.value(0)
ir_down()

#now lets setup the buffers
e1, e2 = [0 for i in range(64)], [0 for i in range(64)]
x1, x2 = [0 for i in range(64)], [0 for i in range(64)]
SCHMITT_TRIG = 0

print("setting up sensor buffers")

transition = False
tik = utime.ticks_ms()
led.value(0)

print("Setting collection flag at start")
HW_GLOBALS["HW_EVENTS"], HW_GLOBALS["COMPLETED_HW"] = 0, 0

#FLAGS |= 0b001000000
#FLAGS |= 0b010000000
while(True):
    if(utime.ticks_ms() - tik < 30):
        utime.sleep_ms(30 - (utime.ticks_ms() - tik))
    tik = utime.ticks_ms()
    

    #If trigger is set and buzz on
    if((trig and buzz) and abs(timers["BUZZ_TIME"] - utime.ticks_ms())>thresh["BUZZ_THRESH"]):
        #lower buzz - keep trigger high
        buzz, trig = False, True
        #FLAGS &= 0b00000001000
        print("LOWERED BUZZ")
        last_buzz = utime.ticks_ms()
        buzzer.value(0)
        
    elif(buzz and not trig and abs(timers["BUZZ_TIME"] - utime.ticks_ms())>thresh["BUZZ_THRESH"]):
        print("BUZZ DOWN")
        buzz = False
        buzzer.value(0)

    if(trig and not buzz and abs(last_buzz-utime.ticks_ms()) > 10*1000):
        #buzz should be low
        trig, buzz = True, True
        #FLAGS |= 0b00000011000
        timers["BUZZ_TIME"] = utime.ticks_ms()
        print("BUZZING")
        buzzer.value(1)
       
    #IF THE DEVICE IS TRIGGERED, LOWER THE TRIGGER
    if(trig and abs(timers['TRIG_TIME'] - utime.ticks_ms()) > thresh["TRIG_THRESH"]):
        print("lowering trig. Lowering buzz")
        trig, buzz, collect = False, False, True
        write_to_device("{0}, {1}, entering, {2}, 0, {3}, {4}, {5}".format(str(utime.localtime()).replace("(", "").replace(")","").replace(", ", "-"), format_time(utime.localtime()), 1, str(abs((last_trigger_time - utime.ticks_ms())/1000)) if abs(last_trigger_time - utime.ticks_ms()) <= 3*60*1000 else ">3min",
                                                             HW_GLOBALS["COMPLETED_HW"], HW_GLOBALS["HW_EVENTS"]))
        #FLAGS &= 0b11111100111
        #FLAGS |= 0b00001000000
        buzzer.value(0)

    #FOR COLLECTION AFTER LEAVE
    if(not collect and just_left and abs(timers["COLLECTION_LEAVE_DELAY"] - utime.ticks_ms())>thresh["COLLECTION_LEAVE_THRESH"]):
        print('raising collection AFTER LEAVE')
        collect, just_left = True, False
        #FLAGS |= 0b00001000000
        #FLAGS ^= 0b01000000000
        
    #FOR COLLECTION AFTER HW
    elif(not collect and just_hw and abs(timers["COLLECTION_HW_DELAY"] - utime.ticks_ms())>thresh["COLLECTION_HW_THRESH"]):
        print('raising collection AFTER HW')
        collect, just_hw = True, False
        #FLAGS |= 0b00001000000
        #FLAGS ^= 0b10000000000
        
    '''elif(abs(timers["COLLECTION_HW_DELAY"] - utime.ticks_ms()) > 40*1000 or abs(timers["COLLECTION_LEAVE_DELAY"] - utime.ticks_ms())>40*1000 and collect == False and (just_left or just_hw)):
        print("RAISING HERE")
        collect, just_left, just_hw = True, False, False'''
                

    #TRANSITION BUZZER - lower transition after movement. for now, just 1
    if(transition and buzz and abs(utime.ticks_ms() - timers["TRANSITION_TIME"]) > thresh["TRANS_BUZZ_THRESH"]):
        print("Lower trans buzz")
        buzz, transition = False, True
        buzzer.value(0)
        
    if(buzzer.value() == 1 and utime.ticks_ms() - timers["BUZZ_TIME"] >= 2000):
        print("MAKING SURE WE ARE LOWER")
        buzzer.value(0)
        buzz = False
    
    if((hw_timer) and (abs(utime.ticks_ms() - timers_irq["ir"]) > thresh_irq["ir"])):
        print("resetting irq")
        hw_timer = False
        writing = False
        utime.sleep(2)
        ir_in.irq(handler=__ir__, trigger=machine.Pin.IRQ_RISING)

    #set for data collecting after success
    '''if(FLAGS & 0b00000100 != 0b00000100 and abs(utime.ticks_ms() - timers["SUCCESS_TRIG"]) > thresh["SUCCESS_THRESH"]):
        print("running collection")
        FLAGS ^= 0b00000100'''

    x1 = [float(adc_0.read_u16()/65355 * vref)] + x1[:-1]
    x2 = [float(adc_1.read_u16() / 65355 * vref)] + x2[:-1]

    adj = 63-buffer
    sigma1, sigma2, vmu_1, vmu_2 = 0, 0, x1[adj], x2[adj]
    e1, e2 = 0, 0
    for k in range(1, buffer):
        var1 = vmu_1 + (1 / buffer) * (x1[adj + k] - vmu_1)
        var2 = vmu_2 + (1 / buffer) * (x2[adj + k] - vmu_2)
        v_partial_1 = sigma1 + (x1[adj + k] - var1) * (x1[adj + k] - var1)
        v_partial_2 = sigma1 + (x2[adj + k] - var2) * (x2[adj + k] - var2)
        e1 += pow(abs(x1[adj + k] - ss1), 2)
        e2 += pow(abs(x2[adj + k] - ss2), 2)

    total = e1+e2
    p1, p2 = e1/total, e2/total
       
    
    if(e1 >= 3 or e2 >= 3):
        if(machine.freq() != 25*1E+6):
            print("speeding up")
            machine.freq(25000000)
        cnt+=1
        SCHMITT_TRIG = 1
        led.value(1)
        ir_on()
        
        #grab which side triggered first
        if(first_trigger == None and p1 >= p2):
            first_trigger = 0
        elif(first_trigger == None and p1<p2):
            first_trigger = 1
        
        #now get the current higest energy proportion
        last_trigger = 0 if p2>=p1 else 1
        
        #now if we see a tranisition in energy
        if(not direction and not transition and (first_trigger == 0 and p2 > 0.55)):
            print("BUZZ 1")
            timers["TRANSITION_TIME"] = utime.ticks_ms()
            timers["BUZZ_TIME"] = utime.ticks_ms()
            buzzer.value(1)
            transition, buzz = True, True
        elif(not transition and direction and (first_trigger == 1 and p1 > 0.55)):
            print("BUZZ 2")
            timers["TRANSITION_TIME"] = utime.ticks_ms()
            timers["BUZZ_TIME"] = utime.ticks_ms()
            buzzer.value(1)
            buzz, transition = True, True
        
    elif(e1 <= 0.15 and e2 <= 0.15 and SCHMITT_TRIG == 1):
        SCHMITT_TRIG = 0
        #FLAGS |= 0b00000010000
        led.value(0)
        ir_down()
        #now we check if it met at least 10 samples where it was high
        if(cnt >= 10):
            #make a decision on the direction
            _direction = decision_tree(first_trigger, last_trigger)
            print("Direction of Movement: {}".format(_direction))
            record(_direction)
        first_trigger, last_trigger = None, None
        transition = False
        cnt = 0
        downtime = utime.localtime()[4]
    

    
    

        

    