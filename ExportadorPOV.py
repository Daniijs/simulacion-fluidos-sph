import os

class ExportadorPOV:
    @staticmethod
    def exportar(directorio, n_frame, sistema, radio=0.012, limites_escena=None, obstaculo=None):
        if not os.path.exists(directorio):
            os.makedirs(directorio)
            
        filename = os.path.join(directorio, f"render_{n_frame:04d}.pov")
        
        num = sistema.num_activas
        pos = sistema.pos 
        
        with open(filename, 'w') as f:
            # Cabeceras e inclusiones
            f.write("#include \"colors.inc\"\n")
            f.write("#include \"textures.inc\"\n")
            f.write("#include \"glass.inc\"\n")
            
            # Configuración Global (FOTONES)
            f.write("global_settings {\n")
            f.write("  assumed_gamma 1.0\n")
            f.write("  max_trace_level 15\n")
            f.write("  photons { count 50000 autostop 0 jitter 0.4 }\n")
            f.write("}\n")
            
            # Cámara (Adaptada al nuevo espacio alargado y alejada para abarcar el canal completo)
            f.write("camera { location <3.2, 2.3, -5.2> look_at <1.0, 0.5, 0> right x*image_width/image_height }\n")
            
            # Entorno (Fondo de estudio gris neutro limpio, estándar de simulación física)
            f.write("sky_sphere {\n")
            f.write("  pigment {\n")
            f.write("    gradient y\n")
            f.write("    color_map {\n")
            f.write("      [0.0 color rgb <0.75, 0.77, 0.80>]\n")
            f.write("      [1.0 color rgb <0.88, 0.90, 0.93>]\n")
            f.write("    }\n")
            f.write("  }\n")
            f.write("}\n")
            
            # Suelo de Hormigón Gris Claro Mate (Captura sombras perfectas sin reflejos distractores)
            f.write("plane { y, 0 \n")
            f.write("  pigment { color rgb <0.75, 0.75, 0.75> } \n")
            f.write("  finish { ambient 0.2 diffuse 0.8 specular 0.1 roughness 0.1 } \n")
            f.write("  photons { collect on } \n")
            f.write("}\n")
            
            # Iluminación de Estudio Profesional Nítida (Una única sombra principal)
            f.write("light_source { <4, 12, -8> color rgb <0.9, 0.9, 0.9> \n")
            f.write("  area_light <3,0,0>, <0,0,3>, 4, 4 adaptive 1 jitter \n")
            f.write("  photons { reflection on refraction on } \n")
            f.write("}\n")
            f.write("light_source { <-5, 8, -4> color rgb <0.4, 0.4, 0.45> shadowless }\n")
            f.write("light_source { <2, 10, 6> color rgb <0.3, 0.3, 0.3> shadowless }\n")
            
            # Caja Contenedora (CRISTAL ULTRA-SUTIL Y TRANSPARENTE CON IOR DE 1.1)
            # Se aprecia que hay caja por reflejos sutiles pero sin oscurecer ni distorsionar la física.
            if limites_escena:
                min_v = limites_escena['min']
                max_v = limites_escena['max']
                f.write(f"box {{ <{min_v.x}, {min_v.y}, {min_v.z}>, <{max_v.x}, {max_v.y}, {max_v.z}> \n")
                f.write("      pigment { color rgbf <0.98, 0.98, 1.0, 0.98> } \n")
                f.write("      finish { specular 0.3 roughness 0.005 reflection { 0.05 fresnel on } } \n")
                f.write("      interior { ior 1.1 } \n")
                f.write("      hollow \n")
                f.write("      photons { target reflection on refraction on }\n")
                f.write("    }\n")
            
            # Obstáculo (Esfera de Carmesí Brillante de Alto Contraste)
            if obstaculo:
                o_pos = obstaculo.pos
                o_rad = obstaculo.radius
                f.write(f"sphere {{ <{o_pos.x}, {o_pos.y}, {o_pos.z}>, {o_rad} \n")
                f.write("      pigment { color rgb <0.9, 0.15, 0.15> } \n")
                f.write("      finish { ambient 0.2 diffuse 0.7 specular 0.6 roughness 0.02 } \n")
                f.write("      photons { target reflection on refraction on }\n")
                f.write("    }\n")
            
            # FLUIDO (EL BLOB)
            if num > 0:
                f.write("blob {\n")
                f.write("  threshold 0.25\n")
                
                # Generar todas las esferas de golpe para velocidad
                spheres = [f"  sphere {{ <{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}>, {radio*3.8:.4f}, 1.0 }}\n" for p in pos[:num]]
                f.write("".join(spheres))

                
                f.write("  pigment { color rgbf <0.05, 0.55, 0.95, 0.65> }\n")
                f.write("  finish { ambient 0.1 diffuse 0.4 specular 0.9 roughness 0.004 reflection { 0.15, 0.9 fresnel on } conserve_energy }\n")
                f.write("  interior { ior 1.333 fade_distance 0.35 fade_color <0.0, 0.3, 0.6> }\n")
                f.write("  photons { target reflection on refraction on }\n")
                f.write("}\n")
