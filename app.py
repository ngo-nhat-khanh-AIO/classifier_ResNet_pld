import streamlit as st
import requests
from PIL import Image
import plotly.express as px

# URL của API
API_URL = "http://127.0.0.1:8000/predict/"

# Tiêu đề ứng dụng
st.title("Nhận diện bệnh khoai tây 🍠")

# Tải ảnh lên
uploaded_file = st.file_uploader("Tải ảnh khoai tây lên và nhận diện bệnh: Healthy, Early Blight, hoặc Late Blight.", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Hiển thị ảnh người dùng đã tải lên
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh đã tải lên", use_column_width=True)

    # Gửi ảnh đến API khi bấm nút
    if st.button("Dự đoán"):
        with st.spinner("Đang dự đoán..."):
            # Gửi request đến API
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                # Xử lý kết quả trả về từ API
                result = response.json()
                predicted_class = result["predicted_class"]
                diagnosis = result["diagnosis"]
                probabilities = result.get("probabilities", [0.0, 0.0, 0.0])  # Lấy xác suất các lớp
                
                # Hiển thị kết quả
                st.success(f"Chẩn đoán: {diagnosis}")
                st.info(f"Chi tiết: {result['label']} (Class {predicted_class})")
                
                # Hiển thị biểu đồ phân phối xác suất
                st.subheader("Phân phối xác suất các lớp")
                classes = ["Early Blight", "Late Blight", "Healthy"]

                # Vẽ biểu đồ tương tác bằng Plotly
                fig = px.bar(
                    x=classes,
                    y=probabilities,
                    labels={'x': "Lớp", 'y': "Xác suất"},
                    title="Xác suất dự đoán cho từng lớp",
                    color=classes,
                    color_discrete_sequence=["red", "orange", "green"]  # Màu cho từng lớp
                )
                fig.update_layout(yaxis=dict(range=[0, 1]))  # Giới hạn trục Y từ 0 đến 1
                st.plotly_chart(fig)
            else:
                st.error("Lỗi: Không thể kết nối tới API!")
