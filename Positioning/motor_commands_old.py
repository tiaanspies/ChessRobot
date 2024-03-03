import RPi.GPIO as GPIO
from time import sleep

class MotorCommands:
    def __init__(self, thetas):
        # Pin numbers
        self.BASE = 1
        self.SHOULDER = 2
        self.ELBOW = 3
        self.GRIP = 4

        # set numbering system to BOARD (as opposed to BCM)
        GPIO.setmode(GPIO.BOARD)

        # set the channels
        GPIO.setup([self.BASE, self.SHOULDER, self.ELBOW, self.GRIP], GPIO.OUT)

        # PWM instances for each motor at 50Hz
        self.base = GPIO.PWM(self.BASE, 50)
        self.shoulder = GPIO.PWM(self.SHOULDER, 50)
        self.elbow = GPIO.PWM(self.ELBOW, 50)
        self.grip = GPIO.PWM(self.GRIP, 50)

        # load the list of commands
        self.thetas = thetas

    def go_to(self, theta):
        """moves directly to provided theta configuration"""
        duty = self.thetas2duty(theta)
        self.base.start(duty[0])
        self.shoulder.start(duty[1])
        self.elbow.start(duty[2])
        self.grip.start(duty[3])

    def run(self):
        """runs the full set of theta commands"""
        try:
            for theta in self.thetas.T:
                duty = self.thetas2duty(theta)
                self.base.start(duty[0])
                self.shoulder.start(duty[1])
                self.elbow.start(duty[2])
                self.grip.start(duty[3])
                sleep(1) # will need to decrease eventually
        except KeyboardInterrupt:
            self.base.stop()
            self.shoulder.stop()
            self.elbow.stop()
            self.grip.stop() # TODO: make sure this means open

    def finish(self):
        self.base.stop()
        self.shoulder.stop()
        self.elbow.stop()
        self.grip.stop() # TODO: make sure this means open
        GPIO.cleanup()

    @staticmethod
    def thetas2duty(theta):
        return theta / 18 + 2 # will need to double check. 18 bc 10% window, 2 because that was the starting duty percentage