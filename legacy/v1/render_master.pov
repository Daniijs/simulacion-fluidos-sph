#version 3.7;
// Archivo maestro para cargar los frames generados por el script de Python

#declare FrameNum = frame_number;
#declare FileName = concat("POV_Frames/render_", str(FrameNum, -4, 0), ".pov")

#debug concat("Renderizando: ", FileName, "\n")

#include FileName
