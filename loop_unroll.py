#!/usr/bin/python

import sys
import math
phi = 0.0;
i = 0;
while (phi < 2*math.pi):
    phi += math.pi/120.0
    print("dx = rad * " + format(math.cos(phi), '.10f') + "; \n\
dy = rad * " + format(math.sin(phi), '.10f') + "; \n\n\
if (tex2D(abs_tex, x + dx, y + dy)  > 20) {\n\
\tfloat grad = PI + tex2D(phi_tex, x + dx, y + dy) - " + repr(phi) + ";\n\
\tif (-PI / 12.0 < grad && grad < PI / 12.0)\n\
\t\tlocal_acc++;\n}")
    print("\n\n");
