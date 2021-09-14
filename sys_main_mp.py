import machine
from machine import ADC
import utime
import time



##############################################
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
##############################################

FLAGS = 0b0000000
timers_irq ={"ir":-5*1000}
thresh_irq = {"ir": 5*1000}


#convention - for peripherals we will use one _
#convention - for interrupt methods we will use __ 2

LED_PIN  = 10
IR_PIN   = 11
BUZZ_PIN = 12
IR_IN    = 19
COMP_IN  = 18
WAKE_UP  = 15


##setup the device and the ios

#can use the third argument as PULL UP or PULL DOWN
led        = machine.Pin(LED_PIN, machine.Pin.OUT)
ir_out     = machine.Pin(IR_PIN, machine.Pin.OUT)
ir_in      = machine.Pin(IR_IN, machine.Pin.IN, machine.Pin.PULL_DOWN)
buzz       = machine.Pin(BUZZ_PIN, machine.Pin.OUT)
comparator = machine.Pin(COMP_IN, machine.Pin.IN, machine.Pin.PULL_DOWN)
wake_up    = machine.Pin(WAKE_UP, machine.Pin.IN, machine.Pin.PULL_DOWN)

def _led_(io):
    '''this method turns the LED on or off'''
    led.value(io)
    
def _buzz_(io):
    buzz.value(io)
    
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

#setup adc


def calculateSS(adc_num):
    return


#run setup
print("power on...")

cnt = 0
l = machine.Pin(25, machine.Pin.OUT)
l.value(0)
time.sleep(1)
l.value(1)
while(True):
    '''if((FLAGS & 0b0000001) and (abs(utime.ticks_ms() - timers_irq["ir"]) > thresh_irq["ir"])):
        print("resetting irq")
        FLAGS = FLAGS & 0b1111110
        utime.sleep(2)
        ir_in.irq(handler=__ir__, trigger=machine.Pin.IRQ_RISING)'''
    
    machine.deepsleep()
    
        
        

    