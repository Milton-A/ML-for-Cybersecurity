import pyshark

capture = pyshark.LiveCapture(interface='Wi-Fi', output_file='output.pcap')
#capture = pyshark.RemoteCapture('192.168.0.1', 'eth0') #Reading from a live remote interface:
for packet in capture.sniff_continuously(packet_count=5):
    print('Just arrived:', packet)
