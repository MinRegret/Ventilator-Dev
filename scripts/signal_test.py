import os
import time
import signal

print(f"My pid is: {os.getpid()}")

def receiveSignal(signalNumber, frame):
    print('Received:', signalNumber)
    time.sleep(5)
    raise SystemExit(1)
    return

if __name__ == '__main__':
    # register the signals to be caught
    signal.signal(signal.SIGINT, receiveSignal)
    signal.signal(signal.SIGQUIT, receiveSignal)
    signal.signal(signal.SIGTERM, receiveSignal)
    
    while True:
        print("Waiting...")
        time.sleep(30)

