[app]
title = V2X Hazard Alert
package.name = v2xhazardalert
package.domain = org.vwimobilothon

source.dir = .
source.include_exts = py,png,jpg,kv,atlas

version = 1.0

requirements = python3,kivy==2.3.0,pyjnius,plyer,android

orientation = portrait
fullscreen = 0

# Android permissions
android.permissions = INTERNET,ACCESS_WIFI_STATE,CHANGE_WIFI_STATE,CHANGE_NETWORK_STATE,ACCESS_NETWORK_STATE,ACCESS_FINE_LOCATION,ACCESS_COARSE_LOCATION,NEARBY_WIFI_DEVICES,ACCESS_BACKGROUND_LOCATION

# Target Android API
android.api = 33
android.minapi = 21
android.ndk = 25b
android.accept_sdk_license = True

# Add location permission features
# android.features = android.hardware.location,android.hardware.location.gps,android.hardware.wifi.direct

[buildozer]
log_level = 2
warn_on_root = 1
