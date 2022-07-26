"""Example program to show how to read a multi-channel time series from LSL."""

import pylsl
import time
from pylsl import StreamInlet, resolve_stream


def main():
    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_stream()
    # create a new inlet to read from the stream
    inlet1 = StreamInlet(streams[0])
    inlet2 = StreamInlet(streams[1])

    while True:
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)

        sample, timestamp = inlet1.pull_sample()
        time_s = time.time()
        # print(inlet1.pull_sample(), inlet2.pull_sample())
        print(time_s, sample)


if __name__ == '__main__':
    main()