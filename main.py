import machine
from machine import ADC
import utime
import time
import math



'''##############################################
REG_IO_BANK0_BASE = 0x40014000
REG_IO_BANK0_INTR0 = 0x0f0
REG_IO_BANK0_DORMANT_WAKE_INTE0 = 0x160

IO_BANK0_DORMANT_WAKE_INTE0_GPIO0_EDGE_HIGH_BITS = 0x00000008
IO_BANK0_DORMANT_WAKE_INTE0_GPIO0_EDGE_LOW_BITS = 0x00000004
IO_BANK0_DORMANT_WAKE_INTE0_GPIO0_LEVEL_HIGH_BITS = 0x00000002
IO_BANK0_DORMANT_WAKE_INTE0_GPIO0_LEVEL_LOW_BITS = 0x00000001

REG_XOSC_BASE = 0x40024000
REG_XOSC_DORMANT = 0x08
REG_XOSC_STATUS = 0x04

XOSC_DORMANT_VALUE_DORMANT = 0x636f6d61
XOSC_STATUS_STABLE_BITS = 0x80000000

@micropython.asm_thumb
def _read_bits(r0):
    ldr(r0, [r0, 0])

@micropython.asm_thumb
def _write_bits(r0, r1):
    str(r1, [r0, 0])

def gpio_acknowledge_irq(gpio, events):
    _write_bits(REG_IO_BANK0_BASE + REG_IO_BANK0_INTR0 + int(gpio / 8) * 4,
                events << 4 * (gpio % 8))

def dormant_until_pins(gpio_pins, edge=True, high=True):
    low = not high
    level = not edge

    if level and low:
        event = IO_BANK0_DORMANT_WAKE_INTE0_GPIO0_LEVEL_LOW_BITS
    if level and high:
        event = IO_BANK0_DORMANT_WAKE_INTE0_GPIO0_LEVEL_HIGH_BITS
    if edge and low:
        event = IO_BANK0_DORMANT_WAKE_INTE0_GPIO0_EDGE_LOW_BITS
    if edge and high:
        event = IO_BANK0_DORMANT_WAKE_INTE0_GPIO0_EDGE_HIGH_BITS

    events = 0
    for gpio_pin in gpio_pins:
        gpio_acknowledge_irq(gpio_pin, event)

        # Enable Wake-up from GPIO IRQ
        gpio_acknowledge_irq(gpio_pin, event)
        en_reg = REG_IO_BANK0_BASE + REG_IO_BANK0_DORMANT_WAKE_INTE0 + int(gpio_pin / 8) * 4
        events += event << 4 * (gpio_pin % 8)
    _write_bits(en_reg, events)

    # Go dormant
    _write_bits(REG_XOSC_BASE + REG_XOSC_DORMANT,
                XOSC_DORMANT_VALUE_DORMANT)

    while not _read_bits(REG_XOSC_BASE + REG_XOSC_STATUS) & XOSC_STATUS_STABLE_BITS:
        pass

    for gpio_pin in gpio_pins:
        gpio_acknowledge_irq(gpio_pin, event)

def dormant_until_pin(gpio_pin, edge=True, high=True):
    dormant_until_pins([gpio_pin], edge, high)

@micropython.asm_thumb
def lightsleep():
    wfi()
##############################################'''

#reduce clock speed
machine.freq(13000000)
l = machine.Pin(25, machine.Pin.OUT)

def flash_LED(num):
    cnt = 0
    while(cnt<num):
        l.value(1)
        utime.sleep(0.5)
        l.value(0)
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


#0b0000001 - HW TIMER
#0b0000010 - last direction
#0b0000100 - data
#0b0001000 - trig
#0b0010000 - buzz
#0b0100000 - cues
#0b1000000 - collect
FLAGS = 0b0000000
HW_GLOBALS = {"COMPLETED_HW":0, "HW_EVENTS":0}
timers_irq = {"ir":-5*1000}
thresh_irq = {"ir": 5*1000}
timers     = {"BUZZ_TIME":-750, "TRIG_TIME":-3*60*1000,
              "HW_TRIG_TIME":-2.5*60*1000, "SUCCESS_TRIG":-30*1000}
thresh     = {"SUCCESS_THRESH":30*1000, "BUZZ_THRESH": 750,
              "TRIG_THRESH": 3*60*1000, "HW_TIMER_THRESH":2*60*1000}

print("Defaults set")
flash_LED(2)

#convention - for peripherals we will use one _
#convention - for interrupt methods we will use __ 2

LED_PIN  = 10
IR_PIN   = 11
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
buzz       = machine.Pin(BUZZ_PIN, machine.Pin.OUT)
comparator = machine.Pin(COMP_IN, machine.Pin.IN, machine.Pin.PULL_DOWN)
wake_up    = machine.Pin(WAKE_UP, machine.Pin.IN, machine.Pin.PULL_DOWN)

adc_0      = ADC(machine.Pin(ADC_0))
adc_1      = ADC(machine.Pin(ADC_1))

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

ss1, ss2 = 0, 0
calculateSS()

#calculate the variances
var_1 = [0 for i in range(128)]
var_2 = [0 for i in range(128)]

#lets fill the arrays first
t = []
x1 = []
x2 = []
y1 = []
y2 = []
p1 = [0 for i in range(128)]
p2 = [0 for i in range(128)]
trigger = [0 for i in range(128)]
p1_peaks = []
p2_peaks = []
p1_t = []
p2_t = []
first_trigger = None
last_trigger = None

recent_buzz = 0

#set tracking variables for csv
tt  = []
ty1 = []
ty2 = []
te1 = []
te2 = []
tp1 = []
tp2 = []
cnt = 0
trigger_cnt = 0
flash_cnt = 0
flash = 1

#set cue and collect flags
FLAGS = FLAGS | 0b1100000


def _led_(io):
    '''this method turns the LED on or off'''
    led.value(io)
    
def _buzz_():
    global recent_buzz
    if(utime.ticks_ms() - recent_buzz > 750):
        recent_buzz = utime.ticks_ms()
        buzz.value(not buzz.value())
    
def _ir_(io):
    ir.value(io)

def __ir__(pin):
    '''interrupts execution'''
    global  ir_in, FLAGS, timers_irq
    timers_irq["ir"] = utime.ticks_ms()
    print("Detected")
    FLAGS = FLAGS | 0b0000001
    ir_in.irq(handler=None)
    
    print("IR IRQ disabled")
def irq_sleep(pin):
    print("Waking up!")
    

#HARDWARE INTERRUPTS
ir_in.irq(handler=__ir__, trigger=machine.Pin.IRQ_RISING)
wake_up.irq(irq_sleep,
        machine.Pin.IRQ_RISING)


#run setup
print("power on...")

cnt = 0
l = machine.Pin(25, machine.Pin.OUT)
l.value(0)
time.sleep(1)
l.value(1)
time.sleep(1)
l.value(0)

#now lets setup the buffers
e1, e2 = [0 for i in range(128)], [0 for i in range(128)]
x1, x2 = [0 for i in range(128)], [0 for i in range(128)]
SCHMITT_TRIG = 0

print("setting up sensor buffers")
for i in range(128):
    x1 = [float(adc_0.read_u16()/65355*vref)]

def runCapture():
    x1 = [float(adc_0.read_u16()/65355 * vref)] + x1[:-1]
    x2 = [float(adc_1.read_u16() / 65355 * vref)] + x2[:-1]

    adj = 127-buffer
    sigma1, sigma2, vmu1, vmu2 = 0, 0, x1[adj], x2[adj]
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
    print(p1, p2)


while(True):
    #CUES
    if(FLAGS & 0b0100000 == 0b0100000):
        FLAGS |= 0b0100000

    #BUZZER
    if(FLAGS & 0b0010000 == 0b0010000 and abs(utime.ticks_ms()-timers["BUZZ_TIME"]) > thresh["BUZZ_THRESH"]):
        #lower the buzzer
        FLAGS ^= 0b0010000

    #BUZZER
    if(FLAGS & 0b0010000 == 0b001000):
        pass
    
    if((FLAGS & 0b0000001) and (abs(utime.ticks_ms() - timers_irq["ir"]) > thresh_irq["ir"])):
        print("resetting irq")
        FLAGS ^= 0b0000001
        utime.sleep(2)
        ir_in.irq(handler=__ir__, trigger=machine.Pin.IRQ_RISING)

    #set for data collecting after success
    if(FLAGS & 0b0000100 != 0b0000100 and abs(utime.ticks_ms() - timers["SUCCESS_TRIG"]) > thresh["SUCCESS_THRESH"]):
        print("running collection")
        FLAGS ^= 0b0000100
    if(FLAGS & 0b0001000 and abs(utime.ticks_ms() - timers["HW_TRIG_TIME"]) >= thresh["HW_TIMER_THRESH"]):
        FLAGS ^= 0b00100

    runCapture()
    
    

        

    