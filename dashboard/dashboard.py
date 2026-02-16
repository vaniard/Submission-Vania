import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Bike Sharing Dashboard - Vania Rachmawati Dewi",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    df_day = pd.read_csv('clean_bike_rental_day.csv')
    df_day['dateday'] = pd.to_datetime(df_day['dateday'])
    
    # Hitung Recency (tanggal terbaru dalam dataset)
    current_date = df_day['dateday'].max()
    df_day['Recency'] = (current_date - df_day['dateday']).dt.days
    
    # Buat R_Score (5 = paling baru, 1 = paling lama)
    df_day['R_Score'] = pd.qcut(df_day['Recency'], 5, labels=False, duplicates='drop')
    df_day['R_Score'] = 5 - df_day['R_Score']
    
    # Buat F_Score dan M_Score (berdasarkan count)
    df_day['F_Score'] = pd.qcut(df_day['count'], 5, labels=False, duplicates='drop') + 1
    df_day['M_Score'] = pd.qcut(df_day['count'], 5, labels=False, duplicates='drop') + 1
    
    # Buat RFM_Score gabungan
    df_day['RFM_Score'] = df_day['R_Score'].astype(str) + df_day['F_Score'].astype(str) + df_day['M_Score'].astype(str)
    
    # Fungsi untuk menentukan segment
    def rfm_segment(row):
        r_score = row['R_Score']
        f_score = row['F_Score']
        m_score = row['M_Score']
        
        if r_score >= 4 and f_score >= 4 and m_score >= 4:
            return 'Best Days'
        elif r_score >= 3 and f_score >= 3 and m_score >= 3:
            return 'Good Days'
        elif r_score >= 2 and f_score >= 2 and m_score >= 2:
            return 'Regular Days'
        elif r_score <= 2 and f_score >= 3 and m_score >= 3:
            return 'Needs Attention'
        else:
            return 'Lost Days'
    
    df_day['Segment'] = df_day.apply(rfm_segment, axis=1)
    
    # Kategori suhu
    df_day['temp_category'] = pd.cut(
        df_day['temperature'],
        bins=[0, 0.2, 0.5, 0.7, 1.0],
        labels=['Cold', 'Mild', 'Warm', 'Hot'],
        include_lowest=True
    )
    
    # Kategori kelembaban
    df_day['hum_category'] = pd.cut(
        df_day['humadity'],
        bins=[0, 0.33, 0.66, 1.0],
        labels=['Low Humidity', 'Medium Humidity', 'High Humidity'],
        include_lowest=True
    )
    
    # Kategori volume penyewaan
    rental_bins = df_day['count'].quantile([0, 0.25, 0.5, 0.75, 1]).tolist()
    rental_labels = ['Low Rentals', 'Medium Rentals', 'High Rentals', 'Very High Rentals']
    
    df_day['rental_volume_category'] = pd.cut(
        df_day['count'],
        bins=rental_bins,
        labels=rental_labels,
        include_lowest=True
    )
    
    return df_day

df = load_data()

# Sidebar - Profil dan Filter
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2972/2972185.png", width=150)
    st.markdown("# 🚲 Bike Sharing Dashboard")
    st.markdown("---")
    
    st.markdown("### 👤 Profil")
    st.markdown("**Nama:** Vania Rachmawati Dewi")
    st.markdown("**Email:** vaniardewi@gmail.com")
    st.markdown("**ID Dicoding:** vaniard")
    
    st.markdown("---")
    st.markdown("### 🎯 Filter Data")
    
    # Filter Tahun
    years = df['year'].unique()
    selected_years = st.multiselect(
        "Tahun",
        options=sorted(years),
        default=sorted(years)
    )
    
    # Filter Musim
    seasons = df['season'].unique()
    season_names = {
        'spring': 'Spring', 
        'summer': 'Summer', 
        'fall': 'Fall', 
        'winter': 'Winter'
    }
    selected_seasons = st.multiselect(
        "Musim",
        options=seasons,
        format_func=lambda x: season_names.get(x, x),
        default=seasons
    )
    
    # Filter Cuaca
    weather = df['weather_condition'].unique()
    weather_names = {
        'clear': 'Clear',
        'mist': 'Mist',
        'light rain': 'Light Rain'
    }
    selected_weather = st.multiselect(
        "Kondisi Cuaca",
        options=weather,
        format_func=lambda x: weather_names.get(x, x),
        default=weather
    )
    
    # Filter Hari
    day_type = st.radio(
        "Tipe Hari",
        options=['Semua', 'Weekday', 'Weekend']
    )
    
    st.markdown("---")
    st.markdown("### 📊 Tentang Dataset")
    st.markdown(f"""
    - **Total Data:** {len(df)} hari
    - **Periode:** {df['dateday'].min().strftime('%d %b %Y')} - {df['dateday'].max().strftime('%d %b %Y')}
    - **Rata-rata Penyewaan:** {df['count'].mean():.0f}/hari
    """)

# Apply filter
filtered_df = df[
    (df['year'].isin(selected_years)) &
    (df['season'].isin(selected_seasons)) &
    (df['weather_condition'].isin(selected_weather))
]

if day_type == 'Weekday':
    filtered_df = filtered_df[filtered_df['day_type'] == 'weekday']
elif day_type == 'Weekend':
    filtered_df = filtered_df[filtered_df['day_type'] == 'weekend']

# Header
st.title("🚴‍♂️ Proyek Analisis Data: Bike Sharing")
st.markdown("---")

# Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📅 Total Hari", f"{len(filtered_df):,}")
with col2:
    st.metric("🚲 Total Penyewaan", f"{filtered_df['count'].sum():,}")
with col3:
    st.metric("📊 Rata-rata/Hari", f"{filtered_df['count'].mean():.0f}")
with col4:
    st.metric("🏆 Penyewaan Tertinggi", f"{filtered_df['count'].max():,}")

st.markdown("---")

# ============================================================================
# VISUALISASI 1. Pengaruh musim terhadap jumlah penyewaan sepeda
# ============================================================================
st.header("🍂 Pengaruh Musim terhadap Jumlah Penyewaan Sepeda")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Rata-rata Penyewaan per Musim")
    
    # Persiapan data - SESUAI NOTEBOOK
    seasonal_avg_rentals = filtered_df.groupby('season')['count'].mean().reset_index()
    
    season_order = ['spring', 'summer', 'fall', 'winter']
    season_names_id = {'spring': 'Spring', 'summer': 'Summer', 'fall': 'Fall', 'winter': 'Winter'}
    
    seasonal_avg_rentals['season'] = pd.Categorical(
        seasonal_avg_rentals['season'], 
        categories=season_order, 
        ordered=True
    )
    seasonal_avg_rentals = seasonal_avg_rentals.sort_values('season')
    seasonal_avg_rentals['season_display'] = seasonal_avg_rentals['season'].map(season_names_id)
    
    # Membuat barplot - SESUAI NOTEBOOK
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 4))
    bars = ax.bar(seasonal_avg_rentals['season_display'], seasonal_avg_rentals['count'], color=colors)
    
    ax.set_title('Rata-rata Jumlah Penyewaan Sepeda Berdasarkan Musim', fontsize=14, pad=20)
    ax.set_xlabel('Musim', fontsize=12)
    ax.set_ylabel('Rata-rata Jumlah Penyewaan', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Tambahkan nilai di atas bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("📦 Distribusi Penyewaan per Musim")
    
    # Boxplot - SESUAI NOTEBOOK
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Siapkan data untuk boxplot
    plot_data = filtered_df.copy()
    plot_data['season_display'] = pd.Categorical(
        plot_data['season'], 
        categories=season_order, 
        ordered=True
    )
    plot_data['season_display'] = plot_data['season_display'].map(season_names_id)
    
    sns.boxplot(x='season_display', y='count', data=plot_data, 
                palette='viridis', ax=ax)
    
    ax.set_title('Distribusi Jumlah Penyewaan Sepeda Berdasarkan Musim', fontsize=14, pad=20)
    ax.set_xlabel('Musim', fontsize=12)
    ax.set_ylabel('Jumlah Penyewaan', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Statistik per musim
st.subheader("📋 Statistik Penyewaan per Musim")

season_stats = filtered_df.groupby('season').agg({
    'casual': 'mean',
    'registered': 'mean',
    'count': ['max', 'min', 'mean']
}).round(2)

season_stats.columns = ['Casual (Rata-rata)', 'Registered (Rata-rata)', 'Max', 'Min', 'Rata-rata']
season_stats = season_stats.reindex(season_order)
season_stats.index = [season_names_id[s] for s in season_order]

st.dataframe(season_stats.style.background_gradient(cmap='viridis', subset=['Rata-rata']),
            use_container_width=True)

st.info("""
**Insight :**
- **Musim Gugur (Fall)** memiliki rata-rata penyewaan tertinggi (5.644) karena kondisi cuaca yang paling ideal
- **Musim Panas (Summer)** menempati posisi kedua dengan rata-rata 4.992
- **Musim Semi (Spring)** memiliki rata-rata penyewaan terendah (2.604) karena suhu rendah dan cuaca kurang stabil
- **Musim Dingin (Winter)** menunjukkan variasi terbesar, dengan nilai minimum sangat rendah (22) dan maksimum tinggi (8.555)
""")

st.markdown("---")

# ============================================================================
# VISUALISASI 2. Pengaruh cuaca terhadap pengguna berdasarkan tipe
# ============================================================================
st.header("☁️ Pengaruh Cuaca terhadap Pengguna Sepeda (Casual vs Registered)")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🌤️ Rata-rata Penyewaan per Kondisi Cuaca")
    
    # Barplot cuaca - SESUAI NOTEBOOK
    weather_avg_rentals = filtered_df.groupby('weather_condition')['count'].mean().reset_index()
    
    weather_order = ['clear', 'mist', 'light rain']
    weather_names_id = {'clear': 'Clear', 'mist': 'Mist', 'light rain': 'Light Rain'}
    
    weather_avg_rentals['weather_condition'] = pd.Categorical(
        weather_avg_rentals['weather_condition'], 
        categories=weather_order, 
        ordered=True
    )
    weather_avg_rentals = weather_avg_rentals.sort_values('weather_condition')
    weather_avg_rentals['weather_display'] = weather_avg_rentals['weather_condition'].map(weather_names_id)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(weather_avg_rentals)))
    bars = ax.bar(weather_avg_rentals['weather_display'], weather_avg_rentals['count'], color=colors)
    
    ax.set_title('Rata-rata Jumlah Penyewaan Sepeda Berdasarkan Kondisi Cuaca', 
                fontsize=14, pad=20)
    ax.set_xlabel('Kondisi Cuaca', fontsize=12)
    ax.set_ylabel('Rata-rata Jumlah Penyewaan', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("👥 Casual vs Registered per Kondisi Cuaca")
    
    # Barplot casual vs registered - SESUAI NOTEBOOK
    weather_user = filtered_df.groupby('weather_condition')[['casual', 'registered']].mean().reset_index()
    weather_user['weather_condition'] = pd.Categorical(
        weather_user['weather_condition'], 
        categories=weather_order, 
        ordered=True
    )
    weather_user = weather_user.sort_values('weather_condition')
    weather_user['weather_display'] = weather_user['weather_condition'].map(weather_names_id)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(weather_user))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, weather_user['casual'], width, 
                   label='Casual', color='skyblue', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, weather_user['registered'], width, 
                   label='Registered', color='teal', edgecolor='black', linewidth=0.5)
    
    ax.set_title('Rata-rata Jumlah Penyewaan Sepeda (Casual vs Registered) Berdasarkan Kondisi Cuaca',
                fontsize=14, pad=20)
    ax.set_xlabel('Kondisi Cuaca', fontsize=12)
    ax.set_ylabel('Rata-rata Jumlah Penyewaan', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(weather_user['weather_display'])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Tambahkan nilai di atas bar
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 20,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Tabel statistik cuaca
st.subheader("📋 Statistik Penyewaan per Kondisi Cuaca")

weather_stats = filtered_df.groupby(by='weather_condition').agg({
    'count': ['max', 'min', 'mean', 'sum'],
    'casual': 'mean',
    'registered': 'mean'
}).round(2)

weather_stats.columns = ['Max', 'Min', 'Rata-rata', 'Total', 'Rata-rata Casual', 'Rata-rata Registered']
weather_stats = weather_stats.reindex(weather_order)
weather_stats.index = [weather_names_id[w] for w in weather_order]

st.dataframe(weather_stats.style.background_gradient(cmap='YlOrRd', subset=['Rata-rata', 'Total']),
            use_container_width=True)

st.info("""
**Insight :**
- **Cuaca Cerah (Clear)** adalah kondisi paling ideal dengan rata-rata penyewaan tertinggi (4.877)
- **Cuaca Berkabut (Mist)** masih menarik banyak penyewa dengan rata-rata 4.036
- **Hujan Ringan (Light Rain)** sangat mengurangi minat penyewaan, rata-rata hanya 1.803
- **Pengguna Registered** jauh lebih stabil dan loyal, bahkan di cuaca buruk mereka tetap menyewa
- **Pengguna Casual** sangat sensitif terhadap cuaca - mereka hanya muncul dalam jumlah besar saat cuaca cerah
""")

st.markdown("---")

# ============================================================================
# VISUALISASI TAMBAHAN: Weekday vs Weekend
# ============================================================================
st.header("📅 Analisis Hari Kerja vs Akhir Pekan")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Weekday vs Weekend")
    
    # Barplot weekday vs weekend - SESUAI NOTEBOOK
    day_type_avg = filtered_df.groupby('day_type')['count'].mean().reset_index()
    day_type_names = {'weekday': 'Weekday', 'weekend': 'Weekend'}
    day_type_avg['day_display'] = day_type_avg['day_type'].map(day_type_names)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#FF6B6B', '#4ECDC4']
    bars = ax.bar(day_type_avg['day_display'], day_type_avg['count'], color=colors, 
                  edgecolor='black', linewidth=0.5)
    
    ax.set_title('Rata-rata Jumlah Penyewaan Sepeda: Hari Kerja vs Akhir Pekan',
                fontsize=14, pad=20)
    ax.set_xlabel('Tipe Hari', fontsize=12)
    ax.set_ylabel('Rata-rata Jumlah Penyewaan', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 20,
                f'{int(height)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("📆 Rata-rata Penyewaan per Hari")
    
    # Barplot per hari
    day_order = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    day_names_id = {
        'monday': 'Monday', 'tuesday': 'Tuesday', 'wednesday': 'Wednesday', 
        'thursday': 'Thursday', 'friday': 'Friday', 'saturday': 'Saturday', 'sunday': 'Sunday'
    }
    
    weekday_avg = filtered_df.groupby('weekday')['count'].mean().reset_index()
    weekday_avg['weekday'] = pd.Categorical(weekday_avg['weekday'], categories=day_order, ordered=True)
    weekday_avg = weekday_avg.sort_values('weekday')
    weekday_avg['day_display'] = weekday_avg['weekday'].map(day_names_id)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Paired(np.linspace(0.1, 0.9, 7))
    bars = ax.bar(weekday_avg['day_display'], weekday_avg['count'], color=colors,
                  edgecolor='black', linewidth=0.5)
    
    ax.set_title('Rata-rata Jumlah Penyewaan Sepeda per Hari', fontsize=14, pad=20)
    ax.set_xlabel('Hari', fontsize=12)
    ax.set_ylabel('Rata-rata Jumlah Penyewaan', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.info("""
**Insight Weekday vs Weekend:**
- **Akhir pekan (weekend)** menunjukkan rata-rata penyewaan yang lebih tinggi dibanding hari kerja
- Penggunaan sepeda lebih banyak untuk tujuan rekreasi selama akhir pekan
- Hari Jumat memiliki rata-rata penyewaan tertinggi di antara hari kerja
- Hari Minggu memiliki rata-rata terendah secara keseluruhan
""")

st.markdown("---")

# ============================================================================
# RFM Analysis (Opsional)
# ============================================================================
st.header("🎯 RFM Analysis - Segmentasi Hari")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Distribusi Segmen RFM")
    
    # Barplot RFM segments
    segment_counts = filtered_df['Segment'].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Jumlah']
    
    segment_order = ['Best Days', 'Good Days', 'Regular Days', 'Needs Attention', 'Lost Days']
    segment_counts['Segment'] = pd.Categorical(segment_counts['Segment'], 
                                              categories=segment_order, ordered=True)
    segment_counts = segment_counts.sort_values('Segment')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(segment_counts)))
    bars = ax.bar(segment_counts['Segment'], segment_counts['Jumlah'], color=colors,
                  edgecolor='black', linewidth=0.5)
    
    ax.set_title('Distribusi Hari di Seluruh Segmen RFM', fontsize=14, pad=20)
    ax.set_xlabel('Segmen RFM', fontsize=12)
    ax.set_ylabel('Jumlah Hari', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("📋 Detail Segmen RFM")
    
    rfm_summary = filtered_df.groupby('Segment').agg({
        'count': ['mean', 'min', 'max'],
        'Recency': 'mean',
        'R_Score': 'mean',
        'F_Score': 'mean',
        'M_Score': 'mean'
    }).round(2)
    
    rfm_summary.columns = ['Rata-rata', 'Min', 'Max', 'Recency', 'R', 'F', 'M']
    rfm_summary = rfm_summary.reindex(segment_order)
    
    st.dataframe(
        rfm_summary.style.background_gradient(cmap='Blues', subset=['Rata-rata', 'Recency']),
        use_container_width=True
    )

st.info("""
**Insight RFM Analysis:**
- **Best Days (249 hari)**: Hari-hari terbaik dengan penyewaan tertinggi dan recency terbaru
- **Regular Days (231 hari)**: Hari-hari dengan performa rata-rata
- **Lost Days (172 hari)**: Hari-hari dengan penyewaan rendah dan sudah lama berlalu
- **Good Days (53 hari)**: Hari-hari baik namun tidak sebaik Best Days
- **Needs Attention (26 hari)**: Hari-hari yang perlu perhatian khusus
""")

st.markdown("---")

# ============================================================================
# CLUSTERING & KATEGORISASI (Sesuai dengan notebook)
# ============================================================================
st.header("📈 Clustering & Kategorisasi")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🌡️ Kategori Suhu")
    
    # Countplot kategori suhu - SESUAI NOTEBOOK
    temp_counts = filtered_df['temp_category'].value_counts().reset_index()
    temp_counts.columns = ['Kategori', 'Jumlah']
    
    temp_order = ['Cold', 'Mild', 'Warm', 'Hot']
    
    temp_counts['Kategori'] = pd.Categorical(temp_counts['Kategori'], categories=temp_order, ordered=True)
    temp_counts = temp_counts.sort_values('Kategori')
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#ADD8E6', '#90EE90', '#FFD700', '#FFA07A']
    bars = ax.bar(temp_counts['Kategori'], temp_counts['Jumlah'], color=colors,
                  edgecolor='black', linewidth=0.5)
    
    ax.set_title('Distribusi Hari Berdasarkan Kategori Suhu', fontsize=12, pad=15)
    ax.set_xlabel('Kategori Suhu', fontsize=10)
    ax.set_ylabel('Jumlah Hari', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Insight suhu
    st.info("""
    **Insight Kategori Suhu:**
    - **Mild (333 hari)**: Suhu sedang mendominasi, ideal untuk bersepeda
    - **Warm (235 hari)**: Suhu hangat juga cukup sering terjadi
    - **Hot (129 hari)**: Cuaca panas tapi masih banyak penyewa
    - **Cold (34 hari)**: Suhu dingin paling jarang, sesuai dengan musim semi yang rendah
    """)

with col2:
    st.subheader("💧 Kategori Kelembaban")
    
    # Countplot kategori kelembaban - SESUAI NOTEBOOK
    hum_counts = filtered_df['hum_category'].value_counts().reset_index()
    hum_counts.columns = ['Kategori', 'Jumlah']
    
    hum_order = ['Low Humidity', 'Medium Humidity', 'High Humidity']
    hum_names = {'Low Humidity': 'Low', 'Medium Humidity': 'Medium', 'High Humidity': 'High'}
    
    hum_counts['Kategori'] = pd.Categorical(hum_counts['Kategori'], categories=hum_order, ordered=True)
    hum_counts = hum_counts.sort_values('Kategori')
    hum_counts['display'] = hum_counts['Kategori'].map(hum_names)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#87CEEB', '#4682B4', '#2E5984']
    bars = ax.bar(hum_counts['display'], hum_counts['Jumlah'], color=colors,
                  edgecolor='black', linewidth=0.5)
    
    ax.set_title('Distribusi Hari Berdasarkan Kategori Kelembaban', fontsize=12, pad=15)
    ax.set_xlabel('Kategori Kelembaban', fontsize=10)
    ax.set_ylabel('Jumlah Hari', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Insight kelembaban
    st.info("""
    **Insight Kategori Kelembaban:**
    - **Medium Humidity (415 hari)**: Kelembaban sedang paling dominan
    - **High Humidity (304 hari)**: Kelembaban tinggi cukup sering terjadi
    - **Low Humidity (12 hari)**: Kelembaban rendah sangat jarang
    """)

with col3:
    st.subheader("📊 Kategori Volume Penyewaan")
    
    # Countplot kategori volume - SESUAI NOTEBOOK
    rental_counts = filtered_df['rental_volume_category'].value_counts().reset_index()
    rental_counts.columns = ['Kategori', 'Jumlah']
    
    rental_order = ['Low Rentals', 'Medium Rentals', 'High Rentals', 'Very High Rentals']
    rental_names = {
        'Low Rentals': 'Rendah', 
        'Medium Rentals': 'Sedang', 
        'High Rentals': 'Tinggi', 
        'Very High Rentals': 'Sangat Tinggi'
    }
    
    rental_counts['Kategori'] = pd.Categorical(rental_counts['Kategori'], categories=rental_order, ordered=True)
    rental_counts = rental_counts.sort_values('Kategori')
    rental_counts['display'] = rental_counts['Kategori'].map(rental_names)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, 4))
    bars = ax.bar(rental_counts['display'], rental_counts['Jumlah'], color=colors,
                  edgecolor='black', linewidth=0.5)
    
    ax.set_title('Distribusi Hari Berdasarkan Kategori Volume Penyewaan', fontsize=12, pad=15)
    ax.set_xlabel('Kategori Volume', fontsize=10)
    ax.set_ylabel('Jumlah Hari', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Insight volume
    st.info("""
    **Insight Kategori Volume:**
    - Distribusi cukup merata di semua kategori
    - **Low Rentals (183 hari)**: Periode sepi penyewaan
    - **Very High Rentals (183 hari)**: Periode ramai penyewaan
    - Variasi signifikan mencerminkan adanya musim ramai dan sepi
    """)

st.markdown("---")

# ============================================================================
# KESIMPULAN
# ============================================================================
st.header("📝 Conclusion")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🍂 Kesimpulan 1")
    st.markdown("""
    - **Musim merupakan faktor utama** yang mempengaruhi jumlah penyewaan sepeda
    - **Musim Gugur (Fall)** adalah musim dengan permintaan tertinggi karena cuaca ideal
    - **Musim Semi (Spring)** merupakan titik terendah operasional
    - Ada kenaikan signifikan dari Spring ke Summer, puncak di Fall, dan menurun di Winter
    """)

with col2:
    st.subheader("☁️ Kesimpulan 2")
    st.markdown("""
    - **Pengguna Registered** adalah pengaruh terbesar dengan loyalitas tinggi - mereka tetap bersepeda meski cuaca kurang ideal
    - **Pengguna Casual** sangat dipengaruhi oleh kenyamanan - mereka hanya muncul saat cuaca cerah
    - **Cuaca buruk (hujan)** menyebabkan penurunan drastis pada pengguna casual
    - Pengguna Registered jauh lebih stabil di semua kondisi cuaca
    """)

st.success("""
**Rekomendasi Bisnis:**
1. **Terapkan promo/tarif lebih rendah** untuk menarik pengguna casual saat cuaca tidak 100% cerah
2. **Luncurkan program bundling/membership tahunan** saat musim gugur (puncak penyewaan)
3. **Distribusikan sepeda** sesuai kebutuhan: area perkantoran di hari kerja, area wisata di akhir pekan
4. **Berikan notifikasi real-time** tentang prediksi cuaca dan rute teraman untuk pengguna registered
5. **Optimalkan stok sepeda** berdasarkan kategori suhu dan kelembaban untuk efisiensi operasional
""")

st.markdown("---")

# Footer
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px; background-color: #f5f5f5; border-radius: 10px;'>
    <h4>🚲 Dashboard Analisis Bike Sharing</h4>
    <p><strong>Nama:</strong> Vania Rachmawati Dewi | <strong>Email:</strong> vaniardewi@gmail.com | <strong>ID Dicoding:</strong> vaniard</p>
    <p>© 2026 - Proyek Analisis Data</p>
</div>
""", unsafe_allow_html=True)