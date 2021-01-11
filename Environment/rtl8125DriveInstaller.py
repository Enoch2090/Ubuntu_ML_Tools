# auto install rtl drive
# Put at the same level of r8125 dir.
import os
import time

global IP
IP = '192.168.0.201'
r8125Dir = "r8125-9.004.01"
os.system("cp /etc/netplan netplan_bkp -r")


def probe():
    global IP
    ip = IP
    backinfo = os.system('ping %s -t 1' % ip)
    return backinfo == 0


while True:
    if not probe():
        print("Can not reach %s at %s, running auto-install." %
              (IP, time.asctime(time.localtime(time.time()))))
        os.system("%s/.autorun.sh")
        os.system("cp netplan_bkp /etc/netplan -r")
        os.system("netplan apply")
    else:
        print("Reached to %s at %s." %
              (IP, time.asctime(time.localtime(time.time()))))
    time.sleep(1000)
