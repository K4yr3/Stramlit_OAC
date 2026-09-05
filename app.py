import streamlit as st
import pandas as pd
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Predicción de Pasajeros por Ruta",
    page_icon="✈️",
    layout="centered"
)


MODEL_PATH = "saved_models/decision_tree_final.pkl"
PREPROCESSOR_PATH = "saved_models/preprocessor.pkl"
METADATA_PATH = "saved_models/metadata.pkl"

ORIGIN_AIRPORTS_PATH = "saved_models/origin_airports.pkl"
DESTINATION_AIRPORTS_PATH = "saved_models/destination_airports.pkl"


# ============================================================
# LOAD MODEL AND ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    metadata = joblib.load(METADATA_PATH)

    origin_airports = joblib.load(ORIGIN_AIRPORTS_PATH)
    destination_airports = joblib.load(DESTINATION_AIRPORTS_PATH)

    return (
        model,
        preprocessor,
        metadata,
        origin_airports,
        destination_airports
    )


(
    model,
    preprocessor,
    metadata,
    origin_airports,
    destination_airports
) = load_artifacts()


# ============================================================
# METADATA
# ============================================================

numeric_features = metadata["numeric_features"]
categorical_features = metadata["categorical_features"]
target = metadata["target"]


# ============================================================
# GET CATEGORICAL OPTIONS FROM TRAINED ENCODER
# ============================================================

encoder = preprocessor.named_transformers_["cat"]

encoder_categories = dict(
    zip(
        categorical_features,
        encoder.categories_
    )
)


airline_options = sorted(
    [str(x) for x in encoder_categories["sigla_aerolinea"]
     if pd.notna(x)]
)

traffic_options = sorted(
    [str(x) for x in encoder_categories["trafico"]
     if pd.notna(x)]
)

flight_type_options = sorted(
    [str(x) for x in encoder_categories["tipo_vuelo"]
     if pd.notna(x)]
)

flight_class_options = sorted(
    [str(x) for x in encoder_categories["tipo_vuelo_clasificado"]
     if pd.notna(x)]
)


# ============================================================
# TITLE
# ============================================================

st.title("✈️ Predicción de pasajeros por ruta aérea")

st.write(
    "Estima el número de pasajeros asociados a una ruta aérea "
    "en Colombia a partir del período y las características "
    "de la operación."
)


# ============================================================
# PERIOD
# ============================================================

st.subheader("📅 Período")

col1, col2 = st.columns(2)

with col1:

    year = st.number_input(
        "Año",
        min_value=2000,
        max_value=2100,
        value=2025,
        step=1
    )

with col2:

    month = st.selectbox(
        "Mes",
        options=list(range(1, 13)),
        format_func=lambda x: [
            "Enero",
            "Febrero",
            "Marzo",
            "Abril",
            "Mayo",
            "Junio",
            "Julio",
            "Agosto",
            "Septiembre",
            "Octubre",
            "Noviembre",
            "Diciembre"
        ][x - 1]
    )


# ============================================================
# ROUTE
# ============================================================

st.subheader("🛫 Ruta")

origin = st.selectbox(
    "Aeropuerto de origen",
    options=origin_airports
)

destination = st.selectbox(
    "Aeropuerto de destino",
    options=destination_airports
)


# ============================================================
# OPERATION CHARACTERISTICS
# ============================================================

st.subheader("✈️ Características de la operación")

airline = st.selectbox(
    "Aerolínea",
    options=airline_options
)

traffic = st.selectbox(
    "Tipo de tráfico",
    options=traffic_options
)

flight_type = st.selectbox(
    "Tipo de vuelo",
    options=flight_type_options
)

flight_class = st.selectbox(
    "Clasificación del vuelo",
    options=flight_class_options
)


# ============================================================
# CARGO
# ============================================================

st.subheader("📦 Carga")

cargo = st.number_input(
    "Carga transportada (kg)",
    min_value=0.0,
    value=0.0,
    step=100.0
)


# ============================================================
# PREDICTION
# ============================================================

st.divider()

if st.button(
    "🔮 Predecir pasajeros",
    use_container_width=True
):

    # Prevent same origin and destination
    if origin == destination:

        st.error(
            "El aeropuerto de origen y destino deben ser diferentes."
        )

    else:

        # Create input DataFrame with EXACT same
        # columns used during model training

        input_data = pd.DataFrame({
            "anio": [year],
            "mes": [month],
            "sigla_aerolinea": [airline],
            "sigla_origen": [origin],
            "sigla_destino": [destination],
            "trafico": [traffic],
            "tipo_vuelo": [flight_type],
            "carga_kg": [cargo],
            "tipo_vuelo_clasificado": [flight_class]
        })


        # Apply the same preprocessing used during training

        X_input = preprocessor.transform(input_data)


        # Make prediction

        prediction = model.predict(X_input)[0]


        # Avoid displaying negative passenger values
        prediction = max(0, prediction)


        # ====================================================
        # RESULT
        # ====================================================

        st.success("Predicción realizada correctamente.")

        st.metric(
            label="Pasajeros estimados",
            value=f"{prediction:,.0f}"
        )

        st.info(
            f"Ruta: **{origin} → {destination}**  \n"
            f"Período: **{month}/{int(year)}**  \n"
            f"Aerolínea: **{airline}**"
        )