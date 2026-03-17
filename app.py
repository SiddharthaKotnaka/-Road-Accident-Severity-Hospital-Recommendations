import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_javascript import st_javascript
import requests
import math
from streamlit_folium import st_folium
import folium

# ================= PAGE =================
st.set_page_config(page_title="Smart Accident Detection", layout="wide")
st.title("🚑 Smart Accident Detection & Hospital Recommendation")

# ================= GET GPS =================
coords = st_javascript("""
navigator.geolocation.getCurrentPosition(
    (position) => {
        const lat = position.coords.latitude;
        const lon = position.coords.longitude;
        return [lat, lon];
    }
)
""")

if isinstance(coords, list) and len(coords) == 2:
    latitude, longitude = coords
    st.success(f"📍 Location Detected: {latitude:.4f}, {longitude:.4f}")
else:
    st.warning("⚠️ Please allow location access. Using default location.")
    latitude, longitude = 17.3850, 78.4867  # Hyderabad fallback

# ================= LOAD MODEL =================
model = load_model("cnn_model.h5")
classes = ['Hand', 'Head', 'Leg']

# ================= DISTANCE FUNCTION =================
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

# ================= FETCH REAL HOSPITALS =================
def get_nearby_hospitals(lat, lon):
    query = f"""
    [out:json];
    (
      node["amenity"="hospital"](around:3000,{lat},{lon});
    );
    out;
    """
    url = "http://overpass-api.de/api/interpreter"
    response = requests.get(url, params={'data': query})

    hospitals = []
    if response.status_code == 200:
        data = response.json()
        for item in data['elements']:
            name = item['tags'].get('name', 'Hospital')
            h_lat = item['lat']
            h_lon = item['lon']
            dist = calculate_distance(lat, lon, h_lat, h_lon)
            hospitals.append((name, h_lat, h_lon, dist))

    hospitals.sort(key=lambda x: x[3])
    return hospitals[:5]

# ================= IMAGE UPLOAD =================
uploaded_file = st.file_uploader("Upload Accident Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=400)

    # ================= PREPROCESS =================
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (64, 64))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # ================= PREDICTION =================
    prediction = model.predict(img)
    result = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Predicted Injury: {result}")
    st.write(f"Confidence: {confidence:.2f}%")
    st.error("🚨 Emergency! Immediate care required!")

    # ================= FETCH REAL HOSPITAL =================
    hospitals = get_nearby_hospitals(latitude, longitude)

    if hospitals:
        nearest = hospitals[0]

        st.subheader("🏥 Nearest Hospital")
        st.write(f"🏥 Name: {nearest[0]}")
        st.write(f"📍 Distance: {nearest[3]:.2f} km")

        # ================= MAP =================
        m = folium.Map(location=[latitude, longitude], zoom_start=14)

        # user marker
        folium.Marker(
            [latitude, longitude],
            tooltip="Your Location",
            icon=folium.Icon(color="blue")
        ).add_to(m)

        # hospital markers
        for h in hospitals:
            folium.Marker(
                [h[1], h[2]],
                tooltip=f"{h[0]} ({h[3]:.2f} km)",
                icon=folium.Icon(color="red")
            ).add_to(m)

        st_folium(m, width=700, height=400)

        # ================= GOOGLE MAP =================
        st.markdown(f"[🌍 Navigate to Hospital](https://www.google.com/maps/search/?api=1&query={nearest[1]},{nearest[2]})")

    # ================= AMBULANCE =================
    st.subheader("🚑 Emergency Support")
    st.write("📞 National Ambulance: 108")

    if st.button("🚑 Call Ambulance"):
        st.write("📲 Call 108 immediately!")
