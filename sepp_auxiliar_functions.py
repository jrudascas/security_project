__name__ = "Utilities SEPP"

#%matplotlib inline
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import gmaps
import gmaps.datasets
import random 
import timeit
import datetime
from shapely import wkt
from pyproj import Proj
from shapely.geometry import Point, LineString, Polygon
from shapely.geometry import MultiPoint, MultiLineString, MultiPolygon, box
import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster
from pyproj import Proj, transform
from pyproj import Transformer
from collections import namedtuple



def cells_on_map(b1, h1, setting, num):
    """
    Calculate the cells (rectangle) of the grid on any localidad or upz. 
    b1: base lenght
    h1: height lenght
    setting = 1: loc
    setting = 2: upz
    num: setting (1 or 2) number
    """
    # Externals Bogota map points
    point1 = Point(-74.2235814249999, 4.836779094841296)
    point2 = Point(-73.98653427799991, 4.836779094841296)
    point3 = Point(-73.98653427799991, 4.269664096859796)
    point4 = Point(-74.2235814249999, 4.269664096859796)

    # EPSG to which we are going to project (32718 also works)
    nys = Proj(init='EPSG:3857') 

    # Points projected in the new EPSG
    p1_proj = nys(point1.x, point1.y)
    p2_proj = nys(point2.x, point2.y)
    p3_proj = nys(point3.x, point3.y)
    p4_proj = nys(point4.x, point4.y)

    # Length of the base and the height of the grid
    longitud_base = Point(p1_proj).distance(Point(p2_proj))
    longitud_altura = Point(p2_proj).distance(Point(p3_proj))

    # Base and height size of each cell
    #b1 = 500
    #h1 = 500
    
    # External grid where we locate the extreme points of the Bogota map
    topLeft = p1_proj
    topRight = Point(Point(p1_proj).x + math.ceil(longitud_base/b1)*b1, Point(p2_proj).y)
    bottomRight = Point(topRight.x, Point(p2_proj).y - math.ceil(longitud_altura/h1)*h1)
    bottomLeft = Point(Point(p1_proj).x, bottomRight.y) 

    # Convertion the grid to polygon
    poligono_mayor = Polygon([topLeft, topRight, bottomRight, bottomLeft])

    # Columns and rows of the grid
    cols = np.linspace(np.array(bottomLeft)[0], np.array(bottomRight)[0], math.ceil(longitud_base/b1) + 1) 
    rows = np.linspace(np.array(topLeft)[1], np.array(bottomLeft)[1], math.ceil(longitud_altura/h1) + 1)     

    # Polygons of the cells that make up the grid
    poligonos = [Polygon([Point(cols[i], rows[j]), Point(cols[i+1], rows[j]), Point(cols[i+1], rows[j+1]), 
                          Point(cols[i], rows[j+1]), Point(cols[i], rows[j])]) for i in range(len(cols)-1) 
                 for j in range(len(rows)-1)]

    poligonos_series = gpd.GeoSeries(poligonos)

    poligonos_df = gpd.GeoDataFrame({'geometry': poligonos})
    poligonos_df['cells'] = poligonos_df.index
    grid = MultiPolygon(poligonos)

    # Mapa de las localidad de Bogotá (excluimos Sumapaz)
    polig_loc_df = gpd.read_file('poligonos-localidades.geojson')
    polig_loc_df = polig_loc_df.drop([2]).reset_index()
    polig_loc_df.drop(columns='index',inplace=True)
    polig_loc_df = polig_loc_df.to_crs("EPSG:3857")
    polig_loc_df['Identificador unico de la localidad'] = polig_loc_df['Identificador unico de la localidad'].astype(int)
    polig_loc_df = polig_loc_df.set_index('Identificador unico de la localidad')

    
    # Mapa de las localidad de Bogotá (excluimos Sumapaz)
    #polig_upz_df =  gpd.read_file('polig_upz_df_transf.geojson')
    polig_upz_df =  gpd.read_file('polig_upz_df.geojson')

    
                    ##################################################################
                    ###### Exclusión de los poligonos que no están sobre Bogotá ######
                    ##################################################################
    
    if setting == 1:
        array_index = np.array([])            # Arreglo en el que almacenamos los poligonos que estan sobre bogota
        for i in range(0,len(poligonos_df)):
            if poligonos_df.geometry.loc[i].intersects(polig_loc_df['geometry'].loc[num]) == True:
                poligonos_df.geometry.loc[i] = poligonos_df.geometry.loc[i].intersection(polig_loc_df['geometry'].loc[num])
                array_index = np.append(array_index,i)
            #if poligonos_df.geometry.loc[i].intersection(polig_loc_df['geometry'].loc[num])/poligonos_df.geometry.loc[i] > 0.15:
    if setting == 2:
        array_index = np.array([])            # Arreglo en el que almacenamos los poligonos que estan sobre bogota
        for i in range(0,len(poligonos_df)):
            if poligonos_df.geometry.loc[i].intersects(polig_upz_df['geometry'].loc[polig_upz_df[polig_upz_df.UPlCodigo==num].index[0]]) == True:
                poligonos_df.geometry.loc[i] = poligonos_df.geometry.loc[i].intersection(polig_upz_df['geometry'].loc[polig_upz_df[polig_upz_df.UPlCodigo==num].index[0]])
                array_index = np.append(array_index,i)
        
    
    # Elegimos solo los poligonos sobre bogota           
    poligonos_df = poligonos_df.loc[np.unique(array_index),'geometry']
    # Convertimos en series para poder graficarlos sobre el mapa de bogota
    poligonos_series = gpd.GeoSeries(poligonos_df)
    # Creamos el geodataframe
    poligonos_df = gpd.GeoDataFrame({'geometry': poligonos_df})
    # Reseteamos el índice que quedo desconfigurado al elegir ciertas celdas
    poligonos_df = poligonos_df.reset_index()
    poligonos_df.drop(columns='index',inplace=True)
    
    
                                    #######################################
                                    ###### Covariados de las celdas  ######
                                    #######################################
    
    #########################  DATAFRAME DE COMANDO DE ATENCIÓN INMEDIATA 2020 ######################## 
    # Unidad policial con recursos humanos y materiales asignados a una jurisdicción, que en forma organizada con la comunidad 
    # y a través de la instrucción permanente, busca la solución de problemas de seguridad, con el objetivo de fraternizar 
    # y unir la policía con la comunidad.
    com_at_df =  gpd.read_file('com_at_df.geojson')
    #com_at_df = com_at_df.to_crs("EPSG:3857")
    poligonos_df['Num de Comando de Atencion Inmediata 2020'] = ''
    
    ################## DATAFRAME DE LOS RESTAURANTES Y BARES EN BOGOTÁ 2019 #################
    # Se entiende por establecimientos gastronómicos, bares y similares aquellos establecimientos comerciales en cabeza de 
    # las personas naturales o jurídicas, cuya actividad económica esté relacionada con la producción, servicio y venta de
    # alimentos y/o bebidas para consumo. Además, podrán prestar otros servicios complementarios.
    # NO FUNCIONA # NO FUNCIONA : MENSAJE DE ERROR  HTTPError: HTTP Error 404: Not Found
    #rest_bar_df = gpd.read_file("https://datosabiertos.bogota.gov.co/dataset/b0c66a77-3230-4d0c-a119-dead7f9b8b8e/resource/9c3829e3-6b4b-4aac-a3e5-297fe0127b67/download/egba.geojson")
    #rest_bar_df = rest_bar_df.to_crs("EPSG:3857")
    #poligonos_df['Rest y Bares'] = ''
    
    ########################## DataFrame de Manzanas-Estratificación 2019 #########################
    #Son unidades geográficas tipo manzana a las cuales se les asocia la variable de estrato socioeconómico, siendo esta, 
    #la clasificación de los inmuebles residenciales que deben recibir servicios públicos. Se realiza principalmente para 
    #cobrar de manera diferencial por estratos los servicios públicos domiciliarios, permitiendo asignar subsidios y cobrar 
    #contribuciones. (# 3982 # 24462 # 43310 # 43819 # 43971  PRESENTAN PROBLEMAS EN LA TOPOLOGÍA)
    estrat_df = gpd.read_file('estrat_df.geojson')
    #estrat_df = estrat_df.to_crs("EPSG:3857")
    poligonos_df['Promedio Estrato 2019'] = ''

    ################## DATAFRAME DE LAS ESTACIONES DE POLICIA EN BOGOTÁ 2020 #################
    # Es la unidad básica de la organización policial cuya jurisdicción corresponde a cada municipio, en el que se divide 
    # el territorio nacional, sin perjuicio que en un Municipio funcionen varias estaciones. En Bogotá las estaciones 
    # corresponden a las localidades que integran el Distrito Capital.
    est_pol_df = gpd.read_file('est_pol_df.geojson')
    #est_pol_df = est_pol_df.to_crs("EPSG:3857")
    poligonos_df['Num de Estaciones de Policia 2020'] = ''

    #########################  DATAFRAME DE CUADRANTES DE POLICÍA 2020 ######################## 
    # Es un sector geográfico fijo, que a partir de sus características sociales, demográficas y geográficas, recibe distintos tipos
    # de atención de servicio policial, entre los cuales se cuentan la prevención, la disuasión, control de delitos y contravenciones 
    # y la educación ciudadana en seguridad y convivencia.
    cuadr_pol_df = gpd.read_file('cuadr_pol_df.geojson')
    #cuadr_pol_df = cuadr_pol_df.to_crs("EPSG:3857") 
    poligonos_df['Area de Cuadrantes de Policia 2020'] = ''

    ######################### DATAFRAME DE INSPECCIONES DE POLICÍA 2020 ######################## 
    # Institución adscrita a la Secretaría de Gobierno del Distrito que cumplen una función vital en la promoción de la convivencia 
    # pacífica de la ciudad, se encargan de prevenir, conciliar, resolver los conflictos que surgen de las relaciones entre vecinos 
    # y todos aquellos que afecten la tranquilidad, seguridad, salubridad, movilidad y el espacio público de los ciudadanos y sancionar 
    # las conductas violatorias al Código de Policía de Bogotá.
    insp_pol_df = gpd.read_file('insp_pol_df.geojson') 
    #insp_pol_df =  insp_pol_df.to_crs("EPSG:3857")
    poligonos_df['Num de Inspecciones de Policia 2020'] = ''

    ######################## DATAFRAME DE LOS CENTROS COMERCIALES 2020 ######################## 
    # Son espacios comerciales administrados por el IPES y ubicados estratégicamente en la Ciudad, en donde los vendedores 
    # informales que ingresan a este servicio, realizan sus actividades comerciales en módulos, locales, restaurantes, 
    # cafeterías y espacios; permitiendo así, la generación de ingresos, fortalecimiento económico y productivo del 
    # ciudadano para hacer viable su ejercicio comercial y la inserción en el mercado formal de la ciudad. Su extensión 
    # geográfica es el Distrito Capital de Bogotá en el área urbana.
    #cc_df = pd.read_csv('https://datosabiertos.bogota.gov.co/dataset/a690b981-4246-42c9-afa0-3125726ae9f2/resource/8efe71cb-fb12-4d63-a449-6fc06fbde27b/download/centro-comercial.csv', sep = ';', encoding='latin-1')
    #cc_df['coord_x'] = (cc_df['coord_x'].replace(',','.', regex=True).astype(float))
    #cc_df['coord_y'] = (cc_df['coord_y'].replace(',','.', regex=True).astype(float))
    #cc_df = gpd.GeoDataFrame(cc_df, geometry=gpd.points_from_xy(cc_df.coord_x, cc_df.coord_y),crs="EPSG:3857")
    #for i in range(0,len(cc_df)):
    #    cc_df['geometry'][i] = Point(nys(cc_df.geometry.iloc[i].x, cc_df.geometry.iloc[i].y))

    #cc_df = cc_df.drop([0], axis=0).reset_index()
    cc_df = gpd.read_file('cc_df.geojson') 
    poligonos_df['Num de Centros Comerciales 2020'] = ''

    ######################### DATAFRAME DE LAS FERIAS INSTITUCIONALES 2020 ######################## 
    # Son alternativas comerciales realizadas de forma permanente, creadas para re ubicar a vendedores informales que 
    # ocupan el espacio público de la ciudad, mitigando el impacto que estas personas generan en la movilidad de la ciudad. Las ferias cuentan con servicio de baño, vigilancia privada y mobiliario compuesto por carpas para la venta de: ropa, calzado, antigüedades y artículos de segunda mano, entre otros. Su extensión geográfica es el Distrito Capital de Bogotá en el área urbana.
    # NOTA: TIENE 3 ELEMENTOS
    #fer_inst_df = pd.read_csv('feria-institucional.csv', sep = ';', encoding='latin-1')#pd.read_csv('https://datosabiertos.bogota.gov.co/dataset/36409df7-da26-4dd3-8f65-352918263dab/resource/a7bd1f96-702b-4f77-a636-444f72ccceef/download/feria-institucional.csv', sep = ';', encoding='latin-1')
    #fer_inst_df['coord_x'] = (fer_inst_df['coord_x'].replace(',','.', regex=True).astype(float))
    #fer_inst_df['coord_y'] = (fer_inst_df['coord_y'].replace(',','.', regex=True).astype(float))
    #fer_inst_df = gpd.GeoDataFrame(fer_inst_df, geometry=gpd.points_from_xy(fer_inst_df.coord_x, fer_inst_df.coord_y))
    #for i in range(0,len(fer_inst_df)):
    #    fer_inst_df['geometry'][i] = Point(nys(fer_inst_df.geometry.iloc[i].x, fer_inst_df.geometry.iloc[i].y))
    fer_inst_df = gpd.read_file('fer_inst_df.geojson') 
    poligonos_df['Num de Ferias Institucionales 2020'] = ''

    ######################### DATAFRAME DE LAS PLAZAS DE MERCADOS 2020 ######################## 
    # Las Plazas de Mercado encuentran su origen en pueblos o ciudades intermedias, donde eran el centro de comercio, donde 
    # familias y cultivadores del campo llegaban con sus cosechas a ofertarlas y adquirir sus productos para la semana o el 
    # mes. Esta comercialización se convirtió en el motor de la principal economía de muchas regiones del país, incluida 
    # Bogotá. Actualmente la ciudad cuenta con 19 Plazas públicas a cargo del IPES, tres de ellas son patrimonio arquitectónico
    # de Bogotá (Perseverancia, Concordia y Cruces y fueron creadas durante el primer tercio del siglo XX); otras están en 
    # trance de ser declaradas como tales, en razón de sus espléndidos diseños realizados por parte de reconocidos 
    # arquitectos como Dicken Castro (cubierta Veinte de Julio). El IPES ha construido en esta administración un nuevo 
    # modelo de gestión de plazas de mercado, orientada hacia la gobernanza, teniendo en cuenta los siguientes componentes: 
    # infraestructura, mercadeo y comercialización, gestión ambiental y fomento a la participación. Lo anterior, con el 
    # propósito de posicionarlas y recuperarlas para la ciudad como destinos turísticos, gastronómicos y culturales. 
    # Su extensión geográfica es el Distrito Capital de Bogotá en el área urbana.
    #pl_merc_df = pd.read_csv('https://datosabiertos.bogota.gov.co/dataset/fd2a6046-5cc3-4acd-81ac-03fe5ef7d549/resource/910e4895-21c3-43b9-84c4-ed40f7e5409e/download/plazas-de-mercado.csv', sep = ';', encoding='latin-1')
    #pl_merc_df['coord_x'] = (pl_merc_df['coord_x'].replace(',','.', regex=True).astype(float))
    #pl_merc_df['coord_y'] = (pl_merc_df['coord_y'].replace(',','.', regex=True).astype(float))
    #pl_merc_df = gpd.GeoDataFrame(pl_merc_df, geometry=gpd.points_from_xy(pl_merc_df.coord_x, pl_merc_df.coord_y))
    #for i in range(0,len(pl_merc_df)):
    #    pl_merc_df['geometry'][i] = Point(nys(pl_merc_df.geometry.iloc[i].x, pl_merc_df.geometry.iloc[i].y))
    pl_merc_df = gpd.read_file('pl_merc_df.geojson')
    poligonos_df['Num de Plazas de Mercado 2020'] = ''

    #########################  DATAFRAME DE ESTABLECIMIENTOS DE ALOJAMIENTO Y HOSPEDAJE 2019 ######################## 
    # Es el conjunto de bienes destinados por la persona natural o jurídica a prestar el servicio de alojamiento no permanente 
    # inferior a 30 días, con o sin alimentación y servicios básicos y/o complementarios o accesorios de alojamiento, mediante }
    # contrato de hospedaje.
    # NO FUNCIONA. ERROR DE HTTP
    #aloj_df = gpd.read_file('https://datosabiertos.bogota.gov.co/dataset/2e8e8046-7033-4d4b-854c-59dd5ecd9d86/resource/318834c0-5d5f-44b5-8f36-ea3a5a8503c1/download/establecimiento_de_alojamiento_y_hospedaje_bogota_2018.geojson')
    #aloj_df = aloj_df.to_crs("EPSG:3857")
    #poligonos_df['Num de Establecimientos de Alojamiento y Hospedaje 2019'] = ''

    #########################  DATAFRAME DE LA TASA DE DESEMPLEO PARA UPZ 2020 ######################## 
    # Muestra la relación porcentual entre el número de personas en búsqueda de empleo y la población económicamente activa 
    # en el distrito Capital.
    desempl_df = gpd.read_file('desempl_df.geojson')
    #desempl_df = desempl_df.to_crs("EPSG:3857")
    poligonos_df['Promedio de Tasa Global de Participacion 2011 (Tasa de Desempleo)'] = ''
    poligonos_df['Promedio de Tasa Global de Participacion 2014 (Tasa de Desempleo)'] = ''
    poligonos_df['Promedio de Tasa Global de Participacion 2017 (Tasa de Desempleo)'] = ''

    ################ DATAFRAME DE TASA DE DESERCIÓN ESCOLAR EN COLEGIOS NO OFICIALES 2019 ################
    # La tasa de deserción escolar no oficial se define como el cociente entre el número de alumnos que desertaron el curso en el que 
    # estaban matriculados en el año t sobre el total de matriculados (aprobados+ reprobados + desertores) en este mismo periodo de 
    # tiempo para entidades no oficiales, por UPZ. Este indicador varía entre 0 y 100. 
    desesc_df = gpd.read_file('desesc_df.geojson')
    #desesc_df = desesc_df.to_crs("EPSG:3857")
    poligonos_df['Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES TRANSICION) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios no oficiales (MUJERES TRANSICION) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES PRIMERO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios no oficiales (MUJERES PRIMERO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES SEGUNDO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios no oficiales (MUJERES SEGUNDO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES TERCERO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios no oficiales (MUJERES TERCERO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES CUARTO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios no oficiales (MUJERES CUARTO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES QUINTO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios no oficiales (MUJERES QUINTO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES SEXTO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios no oficiales (MUJERES SEXTO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES SEPTIMO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios no oficiales (MUJERES SEPTIMO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES OCTAVO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios no oficiales (MUJERES OCTAVO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES NOVENO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios no oficiales (MUJERES NOVENO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES DECIMO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios no oficiales (MUJERES DECIMO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES ONCE) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios no oficiales (MUJERES ONCE) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios no oficiales (TOTAL HOMBRES) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios no oficiales (TOTAL MUJERES) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES DOCE) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios no oficiales (MUJERES DOCE) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES TRECE) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios no oficiales (MUJERES TRECE) 2019'] = ''
    
    ################ DATAFRAME DE TASA DE DESERCIÓN ESCOLAR EN COLEGIOS OFICIALES 2019 ################
    # La tasa de deserción escolar oficial se define como el cociente entre el número de alumnos que desertaron el curso en el que 
    # estaban matriculados en el año t sobre el total de matriculados (aprobados+ reprobados + desertores) en este mismo periodo de 
    # tiempo para entidades no oficiales, por UPZ. Este indicador varía entre 0 y 100.
    desescof_df = gpd.read_file('desescof_df.geojson')
    #desescof_df = desescof_df.to_crs("EPSG:3857")
    poligonos_df['Porcentaje de desercion escolar-Colegios oficiales (HOMBRES TRANSICION) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios oficiales (MUJERES TRANSICION) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios oficiales (HOMBRES PRIMERO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios oficiales (MUJERES PRIMERO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios oficiales (HOMBRES SEGUNDO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios oficiales (MUJERES SEGUNDO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios oficiales (HOMBRES TERCERO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios oficiales (MUJERES TERCERO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios oficiales (HOMBRES CUARTO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios oficiales (MUJERES CUARTO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios oficiales (HOMBRES QUINTO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios oficiales (MUJERES QUINTO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios oficiales (HOMBRES SEXTO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios oficiales (MUJERES SEXTO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios oficiales (HOMBRES SEPTIMO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios oficiales (MUJERES SEPTIMO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios oficiales (HOMBRES OCTAVO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios oficiales (MUJERES OCTAVO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios oficiales (HOMBRES NOVENO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios oficiales (MUJERES NOVENO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios oficiales (HOMBRES DECIMO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios oficiales (MUJERES DECIMO) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios oficiales (HOMBRES ONCE) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios oficiales (MUJERES ONCE) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios oficiales (TOTAL HOMBRES) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios oficiales (TOTAL MUJERES) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios oficiales (HOMBRES ACELERACION) 2019'] = ''
    poligonos_df['Porcentaje de desercion escolar-Colegios oficiales (MUJERES ACELERACION) 2019'] = ''
    
    #########################  DATAFRAME DE LA INFORMALIDAD LABORAL FUERTE 2020 ######################## 
    # Muestra la proporción de individuos que en el mercado laboral tienen una ocupación con un ingreso, pero no están inscritos en 
    # los sistemas de salud en el régimen contributivo y tampoco no cotizan en un fondo de pensiones en el distrito Capital.
    inf_df = gpd.read_file('inf_df.geojson')
    #inf_df = inf_df.to_crs("EPSG:3857")
    poligonos_df['Promedio de la informalidad laboral fuerte 2011'] = ''
    poligonos_df['Promedio de la informalidad laboral fuerte 2014'] = ''
    poligonos_df['Promedio de la informalidad laboral fuerte 2017'] = ''
 
    ######################### DATAFRAME DE TASA DE ASALARIADOS 2020 ######################## 
    #Muestra la relación del total de individuos asalariados sobre el número total de personas ocupadas de la ciudad en el distrito
    #Capital
    tasal_df = gpd.read_file('tasal_df.geojson')
    #tasal_df = tasal_df.to_crs("EPSG:3857")
    poligonos_df['Tasa Promedio de Asalariados 2011'] = ''
    poligonos_df['Tasa Promedio de Asalariados 2014'] = ''
    poligonos_df['Tasa Promedio de Asalariados 2017'] = ''

    ######################### DATAFRAME DE COMANDO DE ATENCION INMEDIATA ######################## 
    comandos_atencion_inmediata_gdf = gpd.read_file('comandos_atencion_inmediata.geojson')
    poligonos_df['Comando Atencion Inmediata'] = ''

    ##### CENTRO DE ATENCION DE VICTIMAS DE DELITOS SEXUALES Y VIOLENCIA INTRAFAMILIAR #####
    centro_atencion_victimas_delito_sexual_y_violencia_intrafamiliar_gdf = gpd.read_file('centro_atencion_victimas_delito_sexual_y_violencia_intrafamiliar.geojson')
    poligonos_df['Centro de Atencion a Victimas de Delito Sexual y Violencia IntraFamiliar'] = ''

    
    
    
                            ###############################################################
                            ###### Copiado de  Covariados de las celdas al DataFrame ######
                            ###############################################################
    
    for i in range(0, len(poligonos_df)):
    
        ############################### COMANDOS DE ATENCIÓN INMEDIATA 2020 ##################################          
        array1 = np.array([])
        #array2 = np.array([])
        for j in range (0, len(com_at_df)):
    
            if poligonos_df['geometry'].loc[i].contains(com_at_df['geometry'].loc[j]) == True:     
                array1 = np.append(array1, int(j))
        
            #if poligonos_df['Coord del Centroide del Poligono'][i].distance(com_at_df['geometry'][j]) < distance:
            #    array2 = np.append(array2, int(j))
            
        poligonos_df.loc[i,'Num de Comando de Atencion Inmediata 2020'] = len(array1)
    
        ############################### BARES Y RESTAURANTES DE BOGOTÁ ################################## 
        #array1 = np.array([])
        #array2 = np.array([])
        #for j in range (0, len(rest_bar_df)):
        
        #    if poligonos_df['geometry'].loc[i].contains(rest_bar_df.geometry.loc[j]) == True:
        #        array1 = np.append(array1, int(j))
            
            #if poligonos_df['Coord del Centroide del Poligono'][i].distance(rest_bar_df['geometry'][j]) < distance:
            #    array2 = np.append(array2, int(j))
    
        #poligonos_df.loc[i,'Rest y Bares'] = len(array1)
        ############################### PROMEDIO DE ESTRATO 2019 ################################## 
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(estrat_df)):
        
            if  poligonos_df['geometry'][i].intersects(estrat_df['geometry'][j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(estrat_df.geometry[j]).area*estrat_df['ESTRATO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(estrat_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Promedio Estrato 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Promedio Estrato 2019'] = 0
    
        ############################### ESTACIONES DE POLICIA 2020 ##################################
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(est_pol_df)):
            
        
            if poligonos_df['geometry'][i].contains(est_pol_df.geometry[j]) == True:
                array1 = np.append(array1, int(j))
            
        #    if poligonos_df['Círculo con radio R'][i].distance(est_pol_df['geometry'][j]) < distance:
        #        array2 = np.append(array2, int(j))
    
        
        poligonos_df.loc[i,'Num de Estaciones de Policia 2020'] = len(array2)
    
        ############################### CUDRADANTES DE POLICIA 2020 ##################################          
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(cuadr_pol_df)):
        
            if poligonos_df['geometry'][i].intersects(cuadr_pol_df['geometry'][j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(cuadr_pol_df['geometry'][j]).area)
        
        #    if poligonos_df['Círculo con radio R'][i].intersects(cuadr_pol_df['geometry'][j]) == True:
        #        array2 = np.append(array2, poligonos_df['Círculo con radio R'][i].intersection(cuadr_pol_df['geometry'][j]).area)
        poligonos_df.loc[i,'Area de Cuadrantes de Policia 2020'] = sum(array1)
        #poligonos_df.loc[i,'Área de Cuadrantes de Policía en un radio R 2020'] = sum(array2)

        ############################### INSPECCIONES DE POLICÍA 2020 ##################################          
        array1 = np.array([])
        #array2 = np.array([])
        for j in range (0, len(insp_pol_df)):
        
            if poligonos_df['geometry'][i].contains(insp_pol_df['geometry'][j]) == True:
                array1 = np.append(array1, int(j))
            
        #    if poligonos_df['Coord del Centroide del Poligono'][i].distance(insp_pol_df['geometry'][j]) < distance:
        #        array2 = np.append(array2, int(j))
            
        poligonos_df.loc[i,'Num de Inspecciones de Policia 2020'] = len(array1)
        #poligonos_df.loc[i,'Núm de Inspecciones de Policía en un radio R 2020'] = len(array2)
    
        ############################### CENTROS COMERCIALES 2020 ##################################
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(cc_df)):
        
            if poligonos_df['geometry'][i].contains(cc_df['geometry'][j]) == True:
                array1 = np.append(array1, int(j))
        
        #    if poligonos_df['Coord del Centroide del Poligono'][i].distance(cc_df['geometry'][j]) < distance:
        #        array2 = np.append(array2, int(j))
        
        poligonos_df.loc[i,'Num de Centros Comerciales 2020'] = len(array1)
        #poligonos_df.loc[i,'Núm de Centros Comerciales en un radio R 2020'] = len(array2)

        ############################### FERIAS INSTITUCIONALES 2020 ##################################
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(fer_inst_df)):
    
            if poligonos_df['geometry'][i].contains(fer_inst_df['geometry'][j]) == True:
                array1 = np.append(array1, int(j))
            
        #    if poligonos_df['Coord del Centroide del Poligono'][i].distance(fer_inst_df['geometry'][j]) < distance:
        #        array2 = np.append(array2, int(j))
    
        poligonos_df.loc[i,'Num de Ferias Institucionales 2020'] = len(array1)
        #poligonos_df.loc[i,'Núm de Ferias Institucionales en un radio R 2020'] = len(array2)
    
        ############################### PLAZAS DE MERCADO 2020 ##################################
        array1 = np.array([])
        #array2 = np.array([])
        for j in range (0, len(pl_merc_df)):
       
            if poligonos_df['geometry'][i].contains(pl_merc_df.geometry[j]) == True:
                array1 = np.append(array1, int(j))
            
            #if poligonos_df['Coord del Centroide del Poligono'][i].distance(pl_merc_df.geometry[j]) < distance:
            #    array2 = np.append(array2, int(j))
            
        poligonos_df.loc[i,'Num de Plazas de Mercado 2020'] = len(array1)
        #poligonos_df.loc[i,'Núm de Plazas de Mercado en un radio R 2020'] = len(array2)
        
        ############################### SITIOS DE HOSPEDAJE Y ALOJAMIENTO 2019 ##################################
        #array1 = np.array([])
        #array2 = np.array([])
        #for j in range (0, len(aloj_df)):
        
        #    if poligonos_df['geometry'][i].contains(aloj_df['geometry'][j]) == True:
        #    array1 = np.append(array1, int(j))
            
        #    if poligonos_df['Coord del Centroide del Poligono'][i].distance(aloj_df['geometry'][j]) < distance:
        #        array2 = np.append(array2,int(j))
    
        #poligonos_df.loc[i,'Num de Establecimientos de Alojamiento y Hospedaje 2019'] = len(array1)
        #poligonos_df.loc[i,'Núm de Establecimientos de Alojamiento y Hospedaje en un radio R 2019'] = len(array2)
    
        #################### PROMEDIO DE TASA GLOBAL DE PARTICIPACION (Tasa de Desempleo) 2020 #####################
             
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desempl_df)):
        
            if  poligonos_df['geometry'][i].intersects(desempl_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desempl_df.geometry[j]).area*desempl_df['tpg2011'].iloc[j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desempl_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Promedio de Tasa Global de Participacion 2011 (Tasa de Desempleo)'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Promedio de Tasa Global de Participacion 2011 (Tasa de Desempleo)'] = 0
    
        ###########
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desempl_df)):
        
            if  poligonos_df['geometry'][i].intersects(desempl_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desempl_df.geometry[j]).area*desempl_df['tpg2014'].iloc[j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desempl_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Promedio de Tasa Global de Participación 2014 (Tasa de Desempleo)'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Promedio de Tasa Global de Participación 2014 (Tasa de Desempleo)'] = 0
        
        ###########
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desempl_df)):
        
            if  poligonos_df['geometry'][i].intersects(desempl_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desempl_df.geometry[j]).area*desempl_df['tpg2017'].iloc[j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desempl_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Promedio de Tasa Global de Participación 2017 (Tasa de Desempleo)'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Promedio de Tasa Global de Participación 2017 (Tasa de Desempleo)'] = 0
    
        ############################### PORCENTAJE DE DESERCIÓN ESCOLAR EN COLEGIOS NO OFICIALES 2020 ##################################
        #La tasa de deserción escolar no oficial se define como el cociente entre el número de alumnos que desertaron el 
        #curso en el que estaban matriculados en el año t sobre el total de matriculados (aprobados+ reprobados + 
        #desertores) en este mismo periodo de tiempo para entidades no oficiales, por UPZ. Este indicador varía entre 0 y 100. 
        #La información primaria tiene como fuente el Censo C600 del año 2019, los cálculos por género en cada grado escolar 
        #fue desarrollado por la Oficina Asesora de Planeación - Grupo de Estadística, 2020.
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desesc_df)):
            if  poligonos_df['geometry'][i].intersects(desesc_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area*desesc_df['H_TRANS'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES TRANSICION) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES TRANSICION) 2019'] = 0
        
        ##############
    
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desesc_df)):
        
            if  poligonos_df['geometry'][i].intersects(desesc_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area*desesc_df['M_TRANS'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (MUJERES TRANSICION) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (MUJERES TRANSICION) 2019'] = 0
    
        ##############
    
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desesc_df)):
        
            if  poligonos_df['geometry'][i].intersects(desesc_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area*desesc_df['H_PRIMERO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES PRIMERO) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES PRIMERO) 2019'] = 0
            
        ##############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desesc_df)):
        
            if  poligonos_df['geometry'][i].intersects(desesc_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area*desesc_df['M_PRIMERO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (MUJERES PRIMERO) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de deserción escolar-Colegios no oficiales (MUJERES PRIMERO) 2019'] = 0
        
        ##############
    
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desesc_df)):
        
            if  poligonos_df['geometry'][i].intersects(desesc_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area*desesc_df['H_SEGUNDO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES SEGUNDO) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES SEGUNDO) 2019'] = 0
        
        ##############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desesc_df)):
        
            if  poligonos_df['geometry'][i].intersects(desesc_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area*desesc_df['M_SEGUNDO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (MUJERES SEGUNDO) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (MUJERES SEGUNDO) 2019'] = 0
        
        ##############
    
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desesc_df)):
        
            if  poligonos_df['geometry'][i].intersects(desesc_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area*desesc_df['H_TERCERO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES TERCERO) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES TERCERO) 2019'] = 0
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desesc_df)):
            
            if  poligonos_df['geometry'][i].intersects(desesc_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area*desesc_df['M_TERCERO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (MUJERES TERCERO) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (MUJERES TERCERO) 2019'] = 0
        
        ##############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desesc_df)):
        
            if  poligonos_df['geometry'][i].intersects(desesc_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area*desesc_df['H_CUARTO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES CUARTO) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES CUARTO) 2019'] = 0
    
        ##############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desesc_df)):
        
            if  poligonos_df['geometry'][i].intersects(desesc_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area*desesc_df['M_CUARTO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (MUJERES CUARTO) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (MUJERES CUARTO) 2019'] = 0
        
        ##############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desesc_df)):
        
            if  poligonos_df['geometry'][i].intersects(desesc_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area*desesc_df['H_QUINTO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES QUINTO) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES QUINTO) 2019'] = 0
        
        ##############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desesc_df)):
        
            if  poligonos_df['geometry'][i].intersects(desesc_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area*desesc_df['M_QUINTO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (MUJERES QUINTO) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de deserción escolar-Colegios no oficiales (MUJERES QUINTO) 2019'] = 0   
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desesc_df)):
        
            if  poligonos_df['geometry'][i].intersects(desesc_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area*desesc_df['H_SEXTO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES SEXTO) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES SEXTO) 2019'] = 0
        
        ##############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desesc_df)):
        
            if  poligonos_df['geometry'][i].intersects(desesc_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area*desesc_df['M_SEXTO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (MUJERES SEXTO) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (MUJERES SEXTO) 2019'] = 0
        
        ##############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desesc_df)):
        
            if  poligonos_df['geometry'][i].intersects(desesc_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area*desesc_df['H_SEPTIMO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES SEPTIMO) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES SEPTIMO) 2019'] = 0
            
        ##############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desesc_df)):
        
            if  poligonos_df['geometry'][i].intersects(desesc_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area*desesc_df['M_SEPTIMO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (MUJERES SEPTIMO) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (MUJERES SEPTIMO) 2019'] = 0
        
        ##############
    
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desesc_df)):
        
            if  poligonos_df['geometry'][i].intersects(desesc_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area*desesc_df['H_OCTAVO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES OCTAVO) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES OCTAVO) 2019'] = 0
     
        ##############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desesc_df)):
        
            if  poligonos_df['geometry'][i].intersects(desesc_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area*desesc_df['M_OCTAVO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (MUJERES OCTAVO) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (MUJERES OCTAVO) 2019'] = 0
        
        ##############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desesc_df)):
        
            if  poligonos_df['geometry'][i].intersects(desesc_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area*desesc_df['H_NOVENO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES NOVENO) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES NOVENO) 2019'] = 0
        
        ##############
    
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desesc_df)):
        
            if  poligonos_df['geometry'][i].intersects(desesc_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area*desesc_df['M_NOVENO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (MUJERES NOVENO) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (MUJERES NOVENO) 2019'] = 0
        
        ##############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desesc_df)):
        
            if  poligonos_df['geometry'][i].intersects(desesc_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area*desesc_df['H_DECIMO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES DECIMO) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES DECIMO) 2019'] = 0
        
        ##############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desesc_df)):
      
            if  poligonos_df['geometry'][i].intersects(desesc_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area*desesc_df['M_DECIMO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (MUJERES DECIMO) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (MUJERES DECIMO) 2019'] = 0
        
        ##############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desesc_df)):
        
            if  poligonos_df['geometry'][i].intersects(desesc_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area*desesc_df['H_ONCE'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES ONCE) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES ONCE) 2019'] = 0
        
        ##############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desesc_df)):
        
            if  poligonos_df['geometry'][i].intersects(desesc_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area*desesc_df['M_ONCE'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (MUJERES ONCE) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (MUJERES ONCE) 2019'] = 0
        
        ##############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desesc_df)):
        
            if  poligonos_df['geometry'][i].intersects(desesc_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area*desesc_df['THOMBRE_UP'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (TOTAL HOMBRES) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (TOTAL HOMBRES) 2019'] = 0
        
        ##############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desesc_df)):
        
            if  poligonos_df['geometry'][i].intersects(desesc_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area*desesc_df['TMUJER_UPZ'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (TOTAL MUJERES) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (TOTAL MUJERES) 2019'] = 0
   
        ##############
    
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desesc_df)):
        
            if  poligonos_df['geometry'][i].intersects(desesc_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area*desesc_df['H_DOCE'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES DOCE) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES DOCE) 2019'] = 0
        
        ##############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desesc_df)):
        
            if  poligonos_df['geometry'][i].intersects(desesc_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area*desesc_df['M_DOCE'].iloc[j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (MUJERES DOCE) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (MUJERES DOCE) 2019'] = 0
   
        ##############
    
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desesc_df)):
        
            if  poligonos_df['geometry'][i].intersects(desesc_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area*desesc_df['H_TRECE'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES TRECE) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (HOMBRES TRECE) 2019'] = 0
        
        ##############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desesc_df)):
        
            if  poligonos_df['geometry'][i].intersects(desesc_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area*desesc_df['M_TRECE'].iloc[j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desesc_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (MUJERES TRECE) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios no oficiales (MUJERES TRECE) 2019'] = 0
        
        ####################### PORCENTAJE DE DESERCIÓN ESCOLAR EN COLEGIOS OFICIALES 2020 ####################

        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desescof_df)):
        
            if  poligonos_df['geometry'][i].intersects(desescof_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area*desescof_df['H_TRANS'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (HOMBRES TRANSICION) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (HOMBRES TRANSICION) 2019'] = 0
        
        #############
    
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desescof_df)):
        
            if  poligonos_df['geometry'][i].intersects(desescof_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area*desescof_df['M_TRANS'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (MUJERES TRANSICION) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (MUJERES TRANSICION) 2019'] = 0
    
        #############
    
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desescof_df)):
        
            if  poligonos_df['geometry'][i].intersects(desescof_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area*desescof_df['H_PRIMERO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (HOMBRES PRIMERO) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (HOMBRES PRIMERO) 2019'] = 0
        
        #############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desescof_df)):
      
            if  poligonos_df['geometry'][i].intersects(desescof_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area*desescof_df['M_PRIMERO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (MUJERES PRIMERO) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (MUJERES PRIMERO) 2019'] = 0
        
        #############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desescof_df)):
        
            if  poligonos_df['geometry'][i].intersects(desescof_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area*desescof_df['H_SEGUNDO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (HOMBRES SEGUNDO) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (HOMBRES SEGUNDO) 2019'] = 0
        
        #############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desescof_df)):
        
            if  poligonos_df['geometry'][i].intersects(desescof_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area*desescof_df['M_SEGUNDO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (MUJERES SEGUNDO) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (MUJERES SEGUNDO) 2019'] = 0
        
        #############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desescof_df)):
        
            if  poligonos_df['geometry'][i].intersects(desescof_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area*desescof_df['H_TERCERO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (HOMBRES TERCERO) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (HOMBRES TERCERO) 2019'] = 0
        
        #############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desesc_df)):
        
            if  poligonos_df['geometry'][i].intersects(desescof_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area*desescof_df['M_TERCERO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (MUJERES TERCERO) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (MUJERES TERCERO) 2019'] = 0
        
        #############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desescof_df)):
        
            if  poligonos_df['geometry'][i].intersects(desescof_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area*desescof_df['H_CUARTO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (HOMBRES CUARTO) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (HOMBRES CUARTO) 2019'] = 0
    
        #############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desescof_df)):
        
            if  poligonos_df['geometry'][i].intersects(desescof_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area*desescof_df['M_CUARTO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (MUJERES CUARTO) 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (MUJERES CUARTO) 2019'] = 0

        #############
    
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desescof_df)):
        
            if  poligonos_df['geometry'][i].intersects(desescof_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area*desescof_df['H_QUINTO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (HOMBRES QUINTO) 2019'] = sum(array1)/sum(array2)

        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (HOMBRES QUINTO) 2019'] = 0
       
        #############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desescof_df)):

            if  poligonos_df['geometry'][i].intersects(desescof_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area*desescof_df['M_QUINTO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area)

        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (MUJERES QUINTO) 2019'] = sum(array1)/sum(array2)

        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (MUJERES QUINTO) 2019'] = 0   
        
        #############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desescof_df)):

            if  poligonos_df['geometry'][i].intersects(desescof_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area*desescof_df['H_SEXTO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area)

        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (HOMBRES SEXTO) 2019'] = sum(array1)/sum(array2)

        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (HOMBRES SEXTO) 2019'] = 0

        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desescof_df)):

            if  poligonos_df['geometry'][i].intersects(desescof_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area*desescof_df['M_SEXTO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area)

        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (MUJERES SEXTO) 2019'] = sum(array1)/sum(array2)

        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (MUJERES SEXTO) 2019'] = 0
        
        #############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desescof_df)):

            if  poligonos_df['geometry'][i].intersects(desescof_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area*desescof_df['H_SEPTIMO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area)

        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (HOMBRES SEPTIMO) 2019'] = sum(array1)/sum(array2)

        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (HOMBRES SEPTIMO) 2019'] = 0
        
        #############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desescof_df)):

            if  poligonos_df['geometry'][i].intersects(desescof_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area*desescof_df['M_SEPTIMO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area)

        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (MUJERES SEPTIMO) 2019'] = sum(array1)/sum(array2)

        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (MUJERES SEPTIMO) 2019'] = 0

        #############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desescof_df)):

            if  poligonos_df['geometry'][i].intersects(desescof_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area*desescof_df['H_OCTAVO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area)

        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (HOMBRES OCTAVO) 2019'] = sum(array1)/sum(array2)

        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (HOMBRES OCTAVO) 2019'] = 0
        
        #############

        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desescof_df)):

            if  poligonos_df['geometry'][i].intersects(desescof_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area*desescof_df['M_OCTAVO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area)

        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (MUJERES OCTAVO) 2019'] = sum(array1)/sum(array2)

        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (MUJERES OCTAVO) 2019'] = 0
        
        #############

        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desescof_df)):

            if  poligonos_df['geometry'][i].intersects(desescof_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area*desescof_df['H_NOVENO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area)

        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (HOMBRES NOVENO) 2019'] = sum(array1)/sum(array2)

        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (HOMBRES NOVENO) 2019'] = 0
        
        #############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desescof_df)):

            if  poligonos_df['geometry'][i].intersects(desescof_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area*desescof_df['M_NOVENO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area)

        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (MUJERES NOVENO) 2019'] = sum(array1)/sum(array2)

        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (MUJERES NOVENO) 2019'] = 0

        #############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desescof_df)):

            if  poligonos_df['geometry'][i].intersects(desescof_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area*desescof_df['H_DECIMO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area)

        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (HOMBRES DECIMO) 2019'] = sum(array1)/sum(array2)

        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (HOMBRES DECIMO) 2019'] = 0
        
        #############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desescof_df)):

            if  poligonos_df['geometry'][i].intersects(desescof_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area*desescof_df['M_DECIMO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area)

        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (MUJERES DECIMO) 2019'] = sum(array1)/sum(array2)

        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (MUJERES DECIMO) 2019'] = 0

        #############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desescof_df)):

            if  poligonos_df['geometry'][i].intersects(desescof_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area*desescof_df['H_ONCE'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area)

        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (HOMBRES ONCE) 2019'] = sum(array1)/sum(array2)

        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (HOMBRES ONCE) 2019'] = 0
        
        #############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desescof_df)):

            if  poligonos_df['geometry'][i].intersects(desescof_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area*desescof_df['M_ONCE'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area)

        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (MUJERES ONCE) 2019'] = sum(array1)/sum(array2)

        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (MUJERES ONCE) 2019'] = 0
        
        #############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desescof_df)):

            if  poligonos_df['geometry'][i].intersects(desescof_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area*desescof_df['THOMBRE_UP'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area)

        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (TOTAL HOMBRES) 2019'] = sum(array1)/sum(array2)

        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (TOTAL HOMBRES) 2019'] = 0
        
        #############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desescof_df)):

            if  poligonos_df['geometry'][i].intersects(desescof_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area*desescof_df['TMUJER_UPZ'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area)

        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (TOTAL MUJERES) 2019'] = sum(array1)/sum(array2)

        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (TOTAL MUJERES) 2019'] = 0

        #############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desescof_df)):

            if  poligonos_df['geometry'][i].intersects(desescof_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area*desescof_df['H_ACELERAC'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area)

        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (HOMBRES ACELERACION) 2019'] = sum(array1)/sum(array2)

        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (HOMBRES ACELERACION) 2019'] = 0

        #############
        
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(desescof_df)):

            if  poligonos_df['geometry'][i].intersects(desescof_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area*desescof_df['M_ACELERAC'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(desescof_df.geometry[j]).area)

        if len(array2) != 0:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (MUJERES ACELERACION) 2019'] = sum(array1)/sum(array2)

        else:
            poligonos_df.loc[i,'Porcentaje de desercion escolar-Colegios oficiales (MUJERES ACELERACION) 2019'] = 0
        
        
        #########################  DATAFRAME DE LA INFORMALIDAD LABORAL FUERTE 2020 ######################## 
        # Muestra la proporción de individuos que en el mercado laboral tienen una ocupación con un ingreso, pero no están inscritos en 
        # los sistemas de salud en el régimen contributivo y tampoco no cotizan en un fondo de pensiones en el distrito Capital.   
                    
                         
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(inf_df)):
        
            if  poligonos_df['geometry'][i].intersects(inf_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(inf_df.geometry[j]).area*inf_df['ilf2011'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(inf_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Promedio de la informalidad laboral fuerte 2011'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Promedio de la informalidad laboral fuerte 2011'] = 0
        
        #############

        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(inf_df)):

            if  poligonos_df['geometry'][i].intersects(inf_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(inf_df.geometry[j]).area*inf_df['ilf2014'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(inf_df.geometry[j]).area)

        if len(array2) != 0:
            poligonos_df.loc[i,'Promedio de la informalidad laboral fuerte 2014'] = sum(array1)/sum(array2)

        else:
            poligonos_df.loc[i,'Promedio de la informalidad laboral fuerte 2014'] = 0

        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(inf_df)):

            if  poligonos_df['geometry'][i].intersects(inf_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(inf_df.geometry[j]).area*inf_df['ilf2017'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(inf_df.geometry[j]).area)

        if len(array2) != 0:
            poligonos_df.loc[i,'Promedio de la informalidad laboral fuerte 2017'] = sum(array1)/sum(array2)

        else:
            poligonos_df.loc[i,'Promedio de la informalidad laboral fuerte 2011'] = 0

        ######################### TASA DE ASALARIADOS  2020 ######################## 
    

        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(tasal_df)):

            if  poligonos_df['geometry'][i].intersects(tasal_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(tasal_df.geometry[j]).area*tasal_df['ta2011'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(tasal_df.geometry[j]).area)

        if len(array2) != 0:
            poligonos_df.loc[i,'Tasa Promedio de Asalariados 2011'] = sum(array1)/sum(array2)

        else:
            poligonos_df.loc[i,'Tasa Promedio de Asalariados 2011'] = 0

        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(tasal_df)):

            if  poligonos_df['geometry'][i].intersects(tasal_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(tasal_df.geometry[j]).area*tasal_df['ta2014'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(tasal_df.geometry[j]).area)

        if len(array2) != 0:
            poligonos_df.loc[i,'Tasa Promedio de Asalariados 2014'] = sum(array1)/sum(array2)

        else:
            poligonos_df.loc[i,'Tasa Promedio de Asalariados 2014'] = 0

        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(tasal_df)):

            if  poligonos_df['geometry'][i].intersects(tasal_df.geometry[j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(tasal_df.geometry[j]).area*tasal_df['ta2017'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(tasal_df.geometry[j]).area)

        if len(array2) != 0:
            poligonos_df.loc[i,'Tasa Promedio de Asalariados 2017'] = sum(array1)/sum(array2)

        else:            
            poligonos_df.loc[i,'Tasa Promedio de Asalariados 2017'] = 0
            
        ############################### COMANDOS DE ATENCIÓN INMEDIATA 2020 ##################################          
        array1 = np.array([])
        #array2 = np.array([])
        for j in range (0, len(comandos_atencion_inmediata_gdf)):
    
            if poligonos_df['geometry'].loc[i].contains(comandos_atencion_inmediata_gdf['geometry'].loc[j]) == True:     
                array1 = np.append(array1, int(j))
        
            #if poligonos_df['Coord del Centroide del Poligono'][i].distance(com_at_df['geometry'][j]) < distance:
            #    array2 = np.append(array2, int(j))
            
        poligonos_df.loc[i,'Comando Atencion Inmediata'] = len(array1)
        
        ##### Centro de Atencion a Victimas de Delito Sexual y Violencia IntraFamiliar #####
        array1 = np.array([])
        #array2 = np.array([])
        for j in range (0, len(centro_atencion_victimas_delito_sexual_y_violencia_intrafamiliar_gdf)):
    
            if poligonos_df['geometry'].loc[i].contains(centro_atencion_victimas_delito_sexual_y_violencia_intrafamiliar_gdf['geometry'].loc[j]) == True:     
                array1 = np.append(array1, int(j))
        
            #if poligonos_df['Coord del Centroide del Poligono'][i].distance(com_at_df['geometry'][j]) < distance:
            #    array2 = np.append(array2, int(j))
            
        poligonos_df.loc[i,'Centro de Atencion a Victimas de Delito Sexual y Violencia IntraFamiliar'] = len(array1)
        
    poligonos_df['cells'] = poligonos_df.index
    #poligonos_df[poligonos_df.columns[10]] = poligonos_df[poligonos_df.columns[10]].astype(float)
    #poligonos_df[poligonos_df.columns[11]] = poligonos_df[poligonos_df.columns[10]].astype(float)
    poligonos_df.to_file("covariates_on_map.geojson", driver='GeoJSON')
    poligonos_df.to_csv('covariates_on_map.csv')
    
                                        #############################
                                        #####    Events data    ##### 
                                        #############################
    
    events_df = pd.read_csv('merged_nuse1.csv')
    events_df = events_df[(events_df['FECHA'] >= '2018-01-01 00:00:00') & (events_df['FECHA'] <= '2018-05-06 23:59:59')].sort_values(by=["FECHA"])
    events_df['FECHA'] = events_df['FECHA'].astype(str)
    def dejar_solo_cifras(txt):
        return "".join(c for c in txt if c.isdigit())
    events_df['COD_UPZ'] = events_df['COD_UPZ'].map(dejar_solo_cifras)
    #events_df['COD_UPZ'] = events_df['COD_UPZ'].astype(int)
    events_df['COD_UPZ'] = pd.to_numeric(events_df['COD_UPZ'])
    #events_df['COD_UPZ'].value_counts().to_numpy() #.astype({'COD_UPZ': 'int64'})
    if setting == 1:
        events_df = events_df[events_df['COD_LOCALIDAD'] == num]
    if setting == 2:
        events_df = events_df[events_df['COD_UPZ'] == float(num)]

    events_df = events_df.reset_index()
    events_df = events_df.drop("index", axis=1)
    
    events_gdf = gpd.GeoDataFrame(events_df, geometry=gpd.points_from_xy(events_df.LONGITUD, events_df.LATITUD),crs="EPSG:3857")
    nys = Proj(init='EPSG:3857') 
    for i in range(0, len(events_gdf)):
        events_gdf['geometry'].iloc[i] = Point(nys(events_gdf['geometry'].iloc[i].x, events_gdf['geometry'].iloc[i].y)) 
    
    events_gdf['TimeStamp'] = ' '
    for i in range(0, len(events_gdf)):
        events_gdf.loc[i,'TimeStamp'] = datetime.datetime.fromisoformat(str(events_gdf.loc[i, 'FECHA'])).timestamp()- datetime.datetime.fromisoformat('2018-01-01 00:00:10').timestamp()
    
    
    polig_upz_df =  gpd.read_file('polig_upz_df_transf.geojson')
    
    if setting == 1:
        array_index = np.array([])            # Arreglo en el que almacenamos los poligonos que estan sobre bogota
        for i in range(0,len(poligonos_df)):
            if poligonos_df.geometry.loc[i].intersects(polig_loc_df['geometry'].loc[num]) == True:
                array_index = np.append(array_index,i)
    if setting == 2:
        array_index = np.array([])            # Arreglo en el que almacenamos los poligonos que estan sobre bogota
        for i in range(0,len(poligonos_df)):
            if poligonos_df.geometry.loc[i].intersects(polig_upz_df['geometry'].loc[polig_upz_df[polig_upz_df.UPlCodigo==num].index[0]]) == True:
                array_index = np.append(array_index,i)
                
    if setting == 1:
        ax = polig_loc_df.loc[[num],'geometry'].plot(figsize=(10,10), color='none', edgecolor='black', zorder=3, alpha=0.5)
        pl1 = poligonos_df.geometry.plot(figsize=(10,10), color='none', edgecolor='black', ax=ax)
        ev1 = events_gdf.geometry.plot(figsize=(10,10), color='black', edgecolor='black', ax=ax)
        
    if setting == 2:
        ax = polig_upz_df.loc[[polig_upz_df[polig_upz_df.UPlCodigo==num].index[0]],'geometry'].plot(figsize=(10,10), color='none', edgecolor='black', zorder=3, alpha=0.5)
        pl1 = poligonos_df.geometry.plot(figsize=(10,10), color='none', edgecolor='black', ax=ax)
        ev1 = events_gdf.geometry.plot(figsize=(10,10), color='black', edgecolor='black', ax=ax)
    
    events_gdf = events_gdf.loc[:,['FECHA','TimeStamp','geometry']]
    poligonos_df = gpd.GeoDataFrame(poligonos_df, crs="EPSG:3857", geometry=poligonos_df.geometry)
    
    return events_gdf, poligonos_df



def training_data(init_data, final_data, gpd_events):
    gpd_events_train = gpd_events[(gpd_events.FECHA>=init_data) & (gpd_events.FECHA<=final_data)]
    gpd_events_train = gpd_events_train.reset_index().drop(columns=['FECHA','index'])
    return gpd_events_train

def test_data(init_data, final_data, gpd_events):
    gpd_events_test = gpd_events[(gpd_events.FECHA>=init_data) & (gpd_events.FECHA<=final_data)]
    gpd_events_test = gpd_events_test.reset_index().drop(columns=['FECHA','index'])
    return gpd_events_test


#######################################################
# Join the covariates of the polygons with the events #
#######################################################

def cov_join_events(gpd_events, poligonos_df_cov):
    """
    Join dataframe of the events with the polygons with its covariates
    
    Input
    par gpd_events: dataframe of the events (train_data for example)
    par poligonos_df_cov: dataframe with the geometry of each polygons
                      with the covariates associated to it.
    Output
    cov_norm_cell_m: array of covariates of each cell of grid
    cov_norm_eventos_m: array of covariates of each cell of grid where occured the event
    poligonos_df_cov: polygons with the covariates
    gpd_events: dataframe with the events with the column "cells" and covariate of the cells
    """
    gpd_events['cells'] = ' ' 
    for i in range(0, len(poligonos_df_cov)):
        for j in range(0, len(gpd_events)):
            if poligonos_df_cov.loc[i,'geometry'].contains(gpd_events.loc[j,'geometry']) == True:
                gpd_events.cells.iloc[j] = int(i)
    gpd_events['X'] = gpd_events.geometry.x
    gpd_events['Y'] = gpd_events.geometry.y
    gpd_events.cells = gpd_events.cells.astype(int)
    
    gpd_events = pd.merge(gpd_events, poligonos_df_cov, on='cells').sort_values(["TimeStamp"]).reset_index().drop(columns=['index'])
    gpd_events = gpd_events.drop(columns='geometry_y')
    gpd_events = gpd_events.rename(columns={'geometry_x':'geometry'})
    #Eliminamos tasalaboral 2014 ya que tiene problemas en la normalizacion del cov
    gpd_events = gpd_events.drop(['Promedio de Tasa Global de Participacion 2014 (Tasa de Desempleo)','Promedio de Tasa Global de Participacion 2017 (Tasa de Desempleo)'],axis=1)
    
    # Covariate events normalization
    for i in range(5, len(gpd_events.columns)):
        if gpd_events[gpd_events.columns[i]].astype(float).max()-gpd_events[gpd_events.columns[i]].astype(float).min() == 0:
            pass
        else:
            gpd_events[gpd_events.columns[i]] = (gpd_events[gpd_events.columns[i]].astype(float)-gpd_events[gpd_events.columns[i]].astype(float).min())/(gpd_events[gpd_events.columns[i]].astype(float).max()-gpd_events[gpd_events.columns[i]].astype(float).min())
    
    poligonos_df_cov = poligonos_df_cov.drop(['Promedio de Tasa Global de Participacion 2014 (Tasa de Desempleo)','Promedio de Tasa Global de Participacion 2017 (Tasa de Desempleo)'],axis=1)
    cov_cells = poligonos_df_cov.iloc[:,1:]
    for i in range(0, len(cov_cells.columns)-1):
        if cov_cells[cov_cells.columns[i]].astype(float).max() - cov_cells[cov_cells.columns[i]].astype(float).min() == 0:
            pass
        else: 
            cov_cells[cov_cells.columns[i]] = (cov_cells[cov_cells.columns[i]].astype(float) - cov_cells[cov_cells.columns[i]].astype(float).min())/(cov_cells[cov_cells.columns[i]].astype(float).max() - cov_cells[cov_cells.columns[i]].astype(float).min())                 
    cov_cells['Int'] = 1
    # we choose the covariates
    cov_cells =  cov_cells.iloc[:,[1,-1]]
    cov_norm_cell_m = cov_cells.to_numpy()
    
    cov_eventos = gpd_events.iloc[:,5:]
    cov_eventos['Int'] = 1
    
    # we choose the covariates
    cov_eventos = cov_eventos.iloc[:,[1,-1]]
    cov_norm_eventos_m = cov_eventos.to_numpy().astype(float)
    
    x_eventos = gpd_events.X.to_numpy()
    y_eventos = gpd_events.Y.to_numpy()
    tiempo_eventos = gpd_events.TimeStamp.to_numpy()
    
    return cov_norm_cell_m, cov_norm_eventos_m, poligonos_df_cov, gpd_events

    
######################################
# Data for training and test process #
######################################

def training_data(init_data, final_data, gpd_events):
    '''
    Gets the data for the training process

    :param init_data: start date of the training process
    :param final_data: end date of the training process
    :param gpd_events: siedo dataframe with all events in the locality or upz chosen
    :return: data for the training process 
    '''
    gpd_events_train = gpd_events[(gpd_events.FECHA>=init_data) & (gpd_events.FECHA<=final_data)]
    gpd_events_train = gpd_events_train.reset_index().drop(columns=['FECHA','index'])
    return gpd_events_train

def test_data(init_data, final_data, gpd_events):
    '''
    Gets the data for the test process
        
    :param init_data: start date of the test process
    :param final_data: end date of the test process
    :param gpd_events: siedo dataframe with all events in the locality or upz chosen
    :return: data for the test process 
    '''
    gpd_events_test = gpd_events[(gpd_events.FECHA>=init_data) & (gpd_events.FECHA<=final_data)]
    gpd_events_test = gpd_events_test.reset_index().drop(columns=['FECHA','index'])
    return gpd_events_test

#######################################################
# Join the covariates of the polygons with the events #
#######################################################

def cov_join_events(gpd_events, poligonos_df_cov):
    """
    Join dataframe of the events with the polygons with its covariates
    
    Input
    par gpd_events: dataframe of the events (train_data for example)
    par poligonos_df_cov: dataframe with the geometry of each polygons
                      with the covariates associated to it.
    Output
    cov_norm_cell_m: array of covariates of each cell of grid
    cov_norm_eventos_m: array of covariates of each cell of grid where occured the event
    poligonos_df_cov: polygons with the covariates
    gpd_events: dataframe with the events with the column "cells" and covariate of the cells
    """
    gpd_events['cells'] = ' ' 
    for i in range(0, len(poligonos_df_cov)):
        for j in range(0, len(gpd_events)):
            if poligonos_df_cov.loc[i,'geometry'].contains(gpd_events.loc[j,'geometry']) == True:
                gpd_events.cells.iloc[j] = int(i)
    gpd_events['X'] = gpd_events.geometry.x
    gpd_events['Y'] = gpd_events.geometry.y
    gpd_events.cells = gpd_events.cells.astype(int)
    
    gpd_events = pd.merge(gpd_events, poligonos_df_cov, on='cells').sort_values(["TimeStamp"]).reset_index().drop(columns=['index'])
    gpd_events = gpd_events.drop(columns='geometry_y')
    gpd_events = gpd_events.rename(columns={'geometry_x':'geometry'})
    #Eliminamos tasalaboral 2014 ya que tiene problemas en la normalizacion del cov
    gpd_events = gpd_events.drop(['Promedio de Tasa Global de Participacion 2014 (Tasa de Desempleo)','Promedio de Tasa Global de Participacion 2017 (Tasa de Desempleo)'],axis=1)
    
    # Covariate events normalization
    for i in range(5, len(gpd_events.columns)):
        if gpd_events[gpd_events.columns[i]].astype(float).max()-gpd_events[gpd_events.columns[i]].astype(float).min() == 0:
            pass
        else:
            gpd_events[gpd_events.columns[i]] = (gpd_events[gpd_events.columns[i]].astype(float)-gpd_events[gpd_events.columns[i]].astype(float).min())/(gpd_events[gpd_events.columns[i]].astype(float).max()-gpd_events[gpd_events.columns[i]].astype(float).min())
    
    poligonos_df_cov = poligonos_df_cov.drop(['Promedio de Tasa Global de Participacion 2014 (Tasa de Desempleo)','Promedio de Tasa Global de Participacion 2017 (Tasa de Desempleo)'],axis=1)
    cov_cells = poligonos_df_cov.iloc[:,1:]
    for i in range(0, len(cov_cells.columns)-1):
        if cov_cells[cov_cells.columns[i]].astype(float).max() - cov_cells[cov_cells.columns[i]].astype(float).min() == 0:
            pass
        else: 
            cov_cells[cov_cells.columns[i]] = (cov_cells[cov_cells.columns[i]].astype(float) - cov_cells[cov_cells.columns[i]].astype(float).min())/(cov_cells[cov_cells.columns[i]].astype(float).max() - cov_cells[cov_cells.columns[i]].astype(float).min())                 
    cov_cells['Int'] = 1
    # we choose the covariates
    cov_cells =  cov_cells.iloc[:,[1,-1]]
    cov_norm_cell_m = cov_cells.to_numpy()
    
    cov_eventos = gpd_events.iloc[:,5:]
    cov_eventos['Int'] = 1
    
    # we choose the covariates
    cov_eventos = cov_eventos.iloc[:,[1,-1]]
    cov_norm_eventos_m = cov_eventos.to_numpy().astype(float)
    
    x_eventos = gpd_events.X.to_numpy()
    y_eventos = gpd_events.Y.to_numpy()
    tiempo_eventos = gpd_events.TimeStamp.to_numpy()
    
    return cov_norm_cell_m, cov_norm_eventos_m, poligonos_df_cov, gpd_events

#########################################################################
# Array of number of events and number of cell where the events ocurred #
#########################################################################

def arr_cells_events_data(array_data_events, array_cells_events_sim):
    """
    Give an array with the number of events in each cell and the number of cell
    for array of events of dataset on polygons, with siedco events, can be train or test events
    
    :param array_data_events: array of the test data events with the cell number 
    :param array_cells_events_sim: array of the simulated data events with the cell number 
    :return array_cells_events_data: array with the number of test events for cell and number of the cell 
    """
    array_cells_events_data_prev1 = []
    array_cells_events_data_prev2 = []
    array_cells_events_data_prev1 = array_data_events.cells.value_counts().sort_values().rename_axis('cell').reset_index(name='events').to_numpy()
    array_cells_events_data_prev1 = array_cells_events_data_prev1[array_cells_events_data_prev1[:,0].astype(int).argsort()]
    for i in range(0, len(array_cells_events_data_prev1)):
        array_cells_events_data_prev2.append([array_cells_events_data_prev1[:,0][i],array_cells_events_data_prev1[:,1][i]])
      
    list1 = [i[0] for i in array_cells_events_sim]
    list2 = [i[0] for i in array_cells_events_data_prev2]
    set_difference = set(list1) - set(list2)
    list_difference = list(set_difference)

    array_cells_events_data_prev3 = []
    for i in range(0, len(list_difference)):
        array_cells_events_data_prev3.append([list_difference[i], 0])
    
    merged_list = array_cells_events_data_prev2 + array_cells_events_data_prev3
    array_cells_events_data = sorted(merged_list, key=lambda x: x[0])
        
    return array_cells_events_data

###################################################################
# Filtering of data: we take only the events in the hotspot cells #
###################################################################

def filtering_data(porcentage_area,array_cells_events_tst_data_1_cells, puntos_gdf, poligonos_df, gpd_ev):
    # array of tst_data_1_cells sorted
    array_cells_events_tst_data_1_cells_sorted = sorted(array_cells_events_tst_data_1_cells, key=lambda x: x[1], reverse=True)
    # number of cells according to the chosen percentage
    length = math.ceil(len(array_cells_events_tst_data_1_cells)*porcentage_area/100)
    # we select the percentage of cells that have more events with their events
    array_cells_hotspots_tsts_data_1 = array_cells_events_tst_data_1_cells_sorted[:length]
    # array with the number of the cell of the hotspot
    array_cells_hotspots_tst_data_1_number_cell = list(np.array(array_cells_hotspots_tsts_data_1)[:,0])
    # dataset of the simulated events for testdata1
    puntos_gdf_cells = cov_join_events(puntos_gdf, poligonos_df)[3]

    array_filtered_sim_tst_data_1 = np.array([])
    for i in range(0, len(puntos_gdf)):
        for j in range(0, len(array_cells_hotspots_tst_data_1_number_cell)):
            if puntos_gdf_cells.cells.iloc[i] == array_cells_hotspots_tst_data_1_number_cell[j]:
                array_filtered_sim_tst_data_1 = np.append(array_filtered_sim_tst_data_1, i)
            
    puntos_gdf_filtered = puntos_gdf_cells.iloc[array_filtered_sim_tst_data_1]
            
    array_filtered_tst_data_1 = np.array([])
    for i in range(0, len(gpd_ev)):
        for j in range(0, len(array_cells_hotspots_tst_data_1_number_cell)):
            if gpd_ev.cells.iloc[i] == array_cells_hotspots_tst_data_1_number_cell[j]:
                array_filtered_tst_data_1 = np.append(array_filtered_tst_data_1, i)

    tst_data_1_cells_filtered = gpd_ev.iloc[array_filtered_tst_data_1] 
    
    return puntos_gdf_filtered, tst_data_1_cells_filtered    