import RPi.GPIO as GPIO
import time
ON_FLAG = False
t = 0


GPIO.setmode(GPIO.BCM)
GPIO.cleanup()
GPIO.setup(16, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(14, GPIO.OUT)

def gpioOn(gpio):
    GPIO.output(gpio, 1)
    
def gpioOff(gpio):
    GPIO.output(gpio, 0)
    global t
    t = time.time()


def cb(channel):
    print("interrupt detected")
    
def dPIRInterrupt(channel):
    print("Triggered")
    global t
    t=time.time()
    gpioOn(14)
    global ON_FLAG
    ON_FLAG = True

GPIO.add_event_detect(16, GPIO.RISING, callback = dPIRInterrupt, bouncetime=1000*3)

t = time.time()
print('starting')
while(True):
    if(ON_FLAG and time.time()-t > 2):
        print('turning off')
        ON_FLAG = False
        gpioOff(14)
        t = time.time()
    
    #print("waiting for edge")
    #GPIO.wait_for_edge(16, GPIO.RISING)
    #gpioOn(14)
    #time.sleep(2)
    #gpioOff(14)
    #print("Rising edge detected")