import os
import time
import busio
import digitalio
import board
import adafruit_mcp3xxx.mcp3008 as MCP
from adafruit_mcp3xxx.analog_in import AnalogIn


class mcp_external(object):
    def __init__(self):
        # create the spi bus
        self.spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)
         
        # create the cs (chip select)
        cs = digitalio.DigitalInOut(board.D22)
         
        # create the mcp object
        self.mcp = MCP.MCP3008(self.spi, cs)
         
        # create an analog input channel on pin 0
        chan0 = AnalogIn(self.mcp, MCP.P0)
        chan1 = AnalogIn(self.mcp, MCP.P1)
        
        self.channels = {
            0:MCP.P0,
            1:MCP.P1,
            2:MCP.P2,
            3:MCP.P3            
            }

        print('Raw ADC Value: ', chan0.value)
        print('ADC Voltage: ' + str(chan0.value/65355*5.2) + 'V')


        print('Raw ADC Value: ', chan1.value)
        print('ADC Voltage: ' + str(chan1.value/65355*5.2) + 'V')
    
    def read_IO(self, channel):
        """
        returns the analog reading in 16-bit value
        """
        return AnalogIn(self.mcp, self.channels[channel]).value
    
if __name__ == "__main__":
    mcp = mcp_external()
    print("{}".format(mcp.read_IO(0)))