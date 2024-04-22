from modules.cameras import DRIFTX3


if __name__ == '__main__':
    vc = DRIFTX3("rtsp://192.168.8.100/live")
    for frame in vc:
        print(1111111111111)
        vc.beep()
