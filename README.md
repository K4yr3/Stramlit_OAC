# Predicción de pasajeros por ruta aérea

Proyecto de Machine Learning desarrollado siguiendo la metodología **CRISP-DM**, cuyo objetivo es predecir la cantidad de pasajeros de una operación de transporte aéreo en Colombia a partir de sus características operativas.

La solución incluye el procesamiento de datos, entrenamiento y evaluación de diferentes modelos de regresión y una aplicación web desarrollada con **Streamlit** para realizar predicciones con nuevos datos.

---

## 🎯 Objetivo

Predecir el **número de pasajeros** de una ruta aérea en Colombia considerando:

* Año y mes de la operación.
* Aerolínea.
* Aeropuerto de origen.
* Aeropuerto de destino.
* Tipo de tráfico.
* Tipo de vuelo.
* Clasificación del vuelo.
* Cantidad de carga transportada.

La aplicación permite introducir estas características y obtener una estimación del número de pasajeros.

---

## 📊 Dataset

El proyecto utiliza datos públicos del Gobierno de Colombia:

**Operaciones aéreas por ruta y aerolínea - Colombia**

Fuente: [Datos Abiertos Colombia](https://www.datos.gov.co/Transporte/Operaciones-a-reas-por-ruta-y-aerol-nea-Colombia/w9ei-uhm3/about_data)

El dataset utilizado contiene aproximadamente **610.000 registros y 10 variables** después de la selección de características.

### Variables

| Variable                 | Descripción                      | Tipo                  |
| ------------------------ | -------------------------------- | --------------------- |
| `anio`                   | Año de la operación              | Numérica              |
| `mes`                    | Mes de la operación              | Numérica              |
| `sigla_aerolinea`        | Código de la aerolínea           | Categórica            |
| `sigla_origen`           | Código del aeropuerto de origen  | Categórica            |
| `sigla_destino`          | Código del aeropuerto de destino | Categórica            |
| `trafico`                | Tipo de tráfico                  | Categórica            |
| `tipo_vuelo`             | Tipo de vuelo                    | Categórica            |
| `pasajeros`              | Número de pasajeros              | **Variable objetivo** |
| `carga_kg`               | Carga transportada en kilogramos | Numérica              |
| `tipo_vuelo_clasificado` | Clasificación del vuelo          | Categórica            |

---

## ⚙️ Preprocesamiento

Las variables numéricas utilizadas como características son:

```text
anio
mes
carga_kg
```

Las variables categóricas son:

```text
sigla_aerolinea
sigla_origen
sigla_destino
trafico
tipo_vuelo
tipo_vuelo_clasificado
```

### Transformaciones

* Las variables numéricas se escalan mediante `StandardScaler`.
* Las variables categóricas se transforman mediante `OneHotEncoder`.
* `handle_unknown="ignore"` permite procesar categorías que no hayan sido observadas durante el entrenamiento.
* Se mantiene una representación **sparse** para evitar convertir la matriz de características a una estructura densa de gran tamaño.

La división de los datos se realizó utilizando:

```text
70% entrenamiento
30% prueba
random_state = 42
```

---

## 🤖 Modelos evaluados

Se evaluaron cinco modelos clásicos de regresión:

1. Ridge
2. Red neuronal (`MLPRegressor`)
3. Árbol de decisión (`DecisionTreeRegressor`)
4. KNN
5. SVM con kernel RBF

También se evaluaron modelos de ensamble:

* Voting Regressor
* Random Forest Regressor
* AdaBoost Regressor

KNN presentó problemas durante la validación cruzada y SVM RBF resultó computacionalmente impráctico para el tamaño y dimensionalidad del dataset.

---

## 🏆 Modelo final

El modelo seleccionado fue un:

**`DecisionTreeRegressor`**

con los siguientes hiperparámetros:

```python
{
    "max_depth": None,
    "min_samples_split": 10,
    "min_samples_leaf": 1
}
```

### Resultados

| Modelo                |        MAE |       RMSE |         R² |
| --------------------- | ---------: | ---------: | ---------: |
| **Árbol de decisión** | **151.41** | **874.96** | **0.9120** |
| Voting                |     390.57 |    1195.18 |     0.8359 |
| Random Forest         |     375.32 |    1223.70 |     0.8279 |
| Red neuronal          |     357.25 |    1229.39 |     0.8263 |
| Ridge                 |     852.65 |    2236.32 |     0.4254 |
| AdaBoost              |    1174.50 |    2270.66 |     0.4076 |

El árbol de decisión obtuvo el **menor RMSE y el mayor R² en el conjunto de prueba**, superando a los modelos de ensamble evaluados.

Además, presentó un tiempo de entrenamiento de aproximadamente **129 segundos** y un tiempo de predicción de aproximadamente **0.17 segundos**, por lo que ofrece un buen equilibrio entre desempeño y costo computacional.

---

## 🖥️ Aplicación Streamlit

La aplicación permite seleccionar los parámetros de una operación aérea y obtener una predicción de pasajeros.

### Entradas

* Año
* Mes
* Aeropuerto de origen
* Aeropuerto de destino
* Aerolínea
* Tipo de tráfico
* Tipo de vuelo
* Clasificación del vuelo
* Carga transportada en kg

### Salida

La aplicación muestra el **número estimado de pasajeros** para la operación seleccionada.

---

## 📁 Estructura del proyecto

```text
air_traffic_prediction/
│
├── app.py
├── requirements.txt
│
└── saved_models/
    ├── decision_tree_final.pkl
    ├── preprocessor.pkl
    ├── metadata.pkl
    ├── origin_airports.pkl
    └── destination_airports.pkl
```

### Archivos principales

| Archivo                    | Descripción                                       |
| -------------------------- | ------------------------------------------------- |
| `app.py`                   | Aplicación web desarrollada con Streamlit         |
| `requirements.txt`         | Dependencias necesarias para ejecutar el proyecto |
| `decision_tree_final.pkl`  | Modelo de regresión seleccionado                  |
| `preprocessor.pkl`         | Pipeline de transformación de las características |
| `metadata.pkl`             | Información de las variables utilizadas           |
| `origin_airports.pkl`      | Lista de aeropuertos disponibles como origen      |
| `destination_airports.pkl` | Lista de aeropuertos disponibles como destino     |

---

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd air_traffic_prediction
```

### 2. Crear un entorno virtual

```bash
python -m venv .venv
```

Activar el entorno:

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## ▶️ Ejecutar la aplicación

Desde la carpeta principal del proyecto:

```bash
streamlit run app.py
```

Streamlit proporcionará una dirección local para acceder a la aplicación desde el navegador.

---

## 📦 Dependencias

Las principales librerías utilizadas son:

```text
streamlit
pandas
numpy
scikit-learn
joblib
```

---

## 🔬 Metodología

El proyecto sigue las etapas principales de **CRISP-DM**:

1. **Business Understanding**
   Definición del problema de predicción de pasajeros en operaciones aéreas.

2. **Data Understanding**
   Exploración y análisis del dataset público de operaciones aéreas de Colombia.

3. **Data Preparation**
   Selección de variables, tratamiento de variables categóricas y numéricas y preparación de los datos para Machine Learning.

4. **Modeling**
   Entrenamiento y ajuste de diferentes modelos de regresión.

5. **Evaluation**
   Comparación mediante MAE, RMSE y R², además de validación cruzada para los modelos ajustados.

6. **Deployment**
   Implementación del modelo seleccionado mediante una aplicación web desarrollada con Streamlit.

---

## ⚠️ Consideraciones

Las predicciones son **estimaciones realizadas por un modelo de Machine Learning** y no deben interpretarse como valores oficiales o garantizados de tráfico aéreo.

Además, las predicciones para años futuros pueden representar una extrapolación limitada. En particular, los árboles de decisión no realizan extrapolación numérica tradicional fuera del rango de los datos utilizados durante el entrenamiento.

---

## 👤 Autor William A, Pabon

Proyecto académico de Machine Learning y despliegue utilizando Python y Streamlit.
