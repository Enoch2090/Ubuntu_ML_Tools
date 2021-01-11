# rtl8125DriveInstaller.py

**NOT TESTED YET! USE AT YOUR OWN RISK!**

MSI B550M is a relatively new motherboard, it has many compatibility issues with Ubuntu. One problem is that after installing `nvidia-smi`, the NIC driver sometimes will be disabled.

This script theoretically pings another server every 1,000 seconds. If the ping failed, it means the NIC driver is down. It will install the r8125-9.004.0 driver automatically.

Unzip `r8125-9.004.01.tar` first.