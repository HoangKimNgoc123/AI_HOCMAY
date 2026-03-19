import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="Dự đoán giá nhà", page_icon="🏡", layout="centered")
# --- 2. LOAD MÔ HÌNH VÀ DỮ LIỆU MẪU ---
@st.cache_resource
def load_model_and_template():
    
    model = joblib.load('pipeline.pkl') 
    
    df_template = pd.read_csv('train.csv', index_col='Id').iloc[[0]].copy()
    df_template["TotalFlrSF"] = df_template["1stFlrSF"] + df_template["2ndFlrSF"]
    df_template["TotalBath"] = df_template["FullBath"] + 0.5 * df_template["HalfBath"]
    
    return model, df_template
model, df_template = load_model_and_template()

# --- 3. GIAO DIỆN NGƯỜI DÙNG (UI) ---
st.title("🏡 Hệ thống Dự đoán Giá nhà Thông minh")
st.markdown("Nhập các thông số cơ bản của ngôi nhà, trí tuệ nhân tạo (XGBoost) sẽ ước tính giá trị giúp bạn!")
st.divider()

# Chia làm 2 cột cho đẹp mắt
col1, col2 = st.columns(2)

with col1:
    st.subheader("🛠️ Thông tin kiến trúc")
    overall_qual = st.slider("Chất lượng tổng thể (1 - 10)", min_value=1, max_value=10, value=6)
    year_built = st.number_input("Năm xây dựng", min_value=1800, max_value=2026, value=2000)
    gr_liv_area = st.number_input("Diện tích sàn sinh hoạt (sqft)", min_value=500, max_value=5000, value=1500)

with col2:
    st.subheader("🛁 Tiện ích đi kèm")
    total_bath = st.number_input("Tổng số phòng tắm", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
    garage_cars = st.slider("Sức chứa Gara (số lượng xe)", min_value=0, max_value=4, value=2)
    neighborhood = st.selectbox("Khu vực (Neighborhood)", 
                                ['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst', 
                                 'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes', 'Timber'])

st.divider()

# --- 4. XỬ LÝ NÚT BẤM DỰ ĐOÁN ---
if st.button("🚀 XEM GIÁ DỰ ĐOÁN", use_container_width=True):
    with st.spinner("Hệ thống đang tính toán..."):
        # Copy khung dữ liệu mẫu và cập nhật 6 giá trị người dùng vừa nhập
        input_data = df_template.copy()
        input_data['OverallQual'] = overall_qual
        input_data['YearBuilt'] = year_built
        input_data['GrLivArea'] = gr_liv_area
        input_data['TotalBath'] = total_bath
        input_data['GarageCars'] = garage_cars
        input_data['Neighborhood'] = neighborhood

        # Thực hiện dự đoán
        predicted_log_price = model.predict(input_data)
        
        # Biến đổi ngược từ log1p về giá gốc bằng expm1
        final_price = np.expm1(predicted_log_price)[0]

        # In kết quả ra màn hình
        st.success("🎉 Dự đoán thành công!")
        st.markdown(f"<h2 style='text-align: center; color: #4CAF50;'>Giá trị ước tính: ${final_price:,.0f}</h2>", unsafe_allow_html=True)