#!/bin/bash
# 1. Kill any existing locks
sudo killall -9 python3 v4l2-ctl 2>/dev/null

# 2. Reset the Media Graph
sudo media-ctl -d /dev/media0 -r

# 3. Create the Bridge (Cam2 Path)
# PHY2 to Decoder
sudo media-ctl -d /dev/media0 -l '"msm_csiphy2":1 -> "msm_csid0":0[1]'
# Decoder to Processor
sudo media-ctl -d /dev/media0 -l '"msm_csid0":1 -> "msm_vfe0_rdi0":0[1]'
# CRITICAL FIX: Link the Processor output pad to the actual video buffer output engine
sudo media-ctl -d /dev/media0 -l '"msm_vfe0_rdi0":1 -> "msm_vfe0_video0":0[1]'

# 4. Set Formats to 8-bit Bayer (RGGB) across the whole pipeline
sudo media-ctl -d /dev/media0 -V '"imx219 18-0010":0 [fmt:SRGGB8_1X8/1920x1080]'
sudo media-ctl -d /dev/media0 -V '"msm_csiphy2":0 [fmt:SRGGB8_1X8/1920x1080]'
sudo media-ctl -d /dev/media0 -V '"msm_csid0":0 [fmt:SRGGB8_1X8/1920x1080]'
# FIX: Force the RDI node to switch out of UYVY mode into 8-bit Bayer mode
sudo media-ctl -d /dev/media0 -V '"msm_vfe0_rdi0":0 [fmt:SRGGB8_1X8/1920x1080]'

# 5. Boost Brightness (Manual Exposure/Gain)
v4l2-ctl -d /dev/v4l-subdev27 --set-ctrl=exposure=1250
v4l2-ctl -d /dev/v4l-subdev27 --set-ctrl=analogue_gain=400

echo "Cam2 Pipeline Secured at 8-bit. Pipeline fully routed."
