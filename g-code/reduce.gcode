;FLAVOR:Marlin
;TIME:123
;Filament used:0.123m
;Layer height:0.2
;MINX:0
;MINY:0
;MINZ:0
;MAXX:100
;MAXY:50
;MAXZ:1

; Start G-code
G21 ; set units to millimeters
G90 ; use absolute coordinates
M82 ; set extruder to absolute mode
G28 ; home all axes
G92 E0 ; reset extruder position

; Set temperatures
M104 S200 ; set extruder temperature to 200°C
M140 S60 ; set bed temperature to 60°C
M109 S200 ; wait for extruder to reach 200°C
M190 S60 ; wait for bed to reach 60°C

; Move to start position
G1 X0 Y0 Z0.2 F1500.0 ; move to starting point

; Draw square perimeter
G1 F1800 E0 ; start extrusion
G1 X100 Y0 E5 ; draw bottom edge
G1 X100 Y50 E10 ; draw right edge
G1 X0 Y50 E15 ; draw top edge
G1 X0 Y0 E20 ; draw left edge

; Fill inside area
G92 E0 ; reset extruder position
G1 F1500 ; set feedrate for infill

; Infill pattern
; Lines from Y=0.5 to Y=49.5 every 0.5mm
G1 X0 Y0.5 F1800 E0.2
G1 X100 Y0.5 E5.2
G1 X100 Y1.0 E5.4
G1 X0 Y1.0 E10.4
G1 X0 Y1.5 E10.6
G1 X100 Y1.5 E15.6
G1 X100 Y2.0 E15.8
G1 X0 Y2.0 E20.8
G1 X0 Y2.5 E21.0
G1 X100 Y2.5 E26.0
G1 X100 Y3.0 E26.2
G1 X0 Y3.0 E31.2
G1 X0 Y3.5 E31.4
G1 X100 Y3.5 E36.4
G1 X100 Y4.0 E36.6
G1 X0 Y4.0 E41.6
G1 X0 Y4.5 E41.8
G1 X100 Y4.5 E46.8
G1 X100 Y5.0 E47.0
G1 X0 Y5.0 E52.0
; คุณสามารถเพิ่มคำสั่งในลักษณะเดียวกันจนถึง Y=49.5

; End G-code
G1 E-1 F1800 ; retract filament to prevent stringing
G1 Z10 F3000 ; raise nozzle
M104 S0 ; turn off extruder heater
M140 S0 ; turn off bed heater
M84 ; disable motors
