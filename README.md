<a id="top"></a><h1 style="background:#C9FF35; border:0; color:black; padding-top:15px; padding-bottom:15px; text-align: center; font: normal 1rem/1.5rem 'bernino-regular',Verdana,Arial,Geneva,sans-serif !important; font-weight: 700 !important; font-size: 1.75rem !important; line-height: 2.25rem !important; letter-spacing: -0.00625rem !important; vertical-align: baseline !important;">Proyecto de Machine Learining y Deep Learning con datos de Idealista</h1>

![logo](logo_project.jpg)

<a id="proyecto"></a><h2 style='background:#C9FF35; border:0; color:black; padding-top:10px; padding-bottom:10px; padding-left:20px'>Breve descripción del proyecto</h2>

Este repositorio contiene el proyecto Data Science que se ha desarrollado aplicando diferentes técnicas aprendidas en el **Data Science Bootcamp de The Bridge**.

El proyecto está centrado en el análisis y predicción del precio de viviendas en venta de Madrid, estudiando diversas variables relacionadas con la venta de bienes inmuebles extraídas de datos obtenidos del portal inmobiliario Idealista.

El conjunto de datos utilizado contiene una coordenada geográfica por cada vivienda, lo que ha permitido combinar herramientas de análisis **GIS (sistema de información geográfica)** con las herramientas de Data Science.

El siguiente [Story Map](https://storymaps.arcgis.com/stories/6cc01a1558154305aa16e974cd669584) contiene descripción más detallada del proyecto.

<a id="proyecto"></a><h2 style='background:#C9FF35; border:0; color:black; padding-top:10px; padding-bottom:10px; padding-left:20px'>Estructura del proyecto</h2>

El proyecto tiene la siguiente estructura de carpetas:
- **src**: carpeta con todo el código
- **src/data**: carpeta con todos los archivos de datos utilizados en el analisis (datasets y objetos python serializados)
- **src/images**: carpeta con imágenes utilizadas en el Story Map que describe el proyecto
- **src/models**: fichreos con el mejor modelo de Machine Learning y el modelo Deep Learning generados en el proyecto
- **src/utils**: contiene las clases python implementadas en el proyecto
- **src/main.ipynb**: notebook principal del proyecto
- **src/GetDataIdealista.ipynb**: notebook que contiene el código para descargar información de la base de datos del portal inmobiliaro Idealista a través de su API
