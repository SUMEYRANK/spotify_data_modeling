
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


# 1. Veri Okuma (Linkinizi kullanarak)
url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv"
df = pd.read_csv(url)
print("--- Orijinal Veri Başlangıcı ---")
print(df.head())
print("--- Boş Değer Kontrolü (Orijinal) ---")
print(df.isnull().sum())
print("-----------------------------------")


# Veri Temizleme ve Hazırlık
df1 = df.drop(["track_id", "playlist_id", "duration_ms", "key"], axis=1)

# NaN değerleri sadece "track_name" sütununda değil, modelde kullanacağımız
# 'track_popularity' veya 'artist_name' gibi sütunlarda da kontrol etmek faydalı.
# Ancak buradaki asıl amaç, boş değerleri temizlemek.
# Boş "track_name" satırlarını düşürmek genel bir yaklaşımdır.
df1 = df1.dropna() # Tüm NaN değerleri içeren satırları düşürdük

print("--- Boş Değer Kontrolü (Temizlenmiş) ---")
print(df1.isnull().sum())
print("-----------------------------------")

# 2. Hata Düzeltme: Feature (Özellik) Seçimi
# DataFrame'de birden fazla sütun seçerken **çift köşeli parantez** kullanılmalıdır.
# x, tahmin için kullanılacak bağımsız değişkenler (özellikler, features).
# y, tahmin edilmek istenen bağımlı değişken (hedef, target).
# Not: 'track_popularity' bir sayı (0-100) olduğu için Lineer Regresyon kullanmak doğrudur.

# Geliştirilmiş One-Hot Encoding Kodu (Multicollinearity'i Önler)
df_final = pd.get_dummies(
    df1,
    columns=["playlist_genre", "playlist_subgenre"],
    drop_first=True)  # İlk kategori sütununu sil

print("\n--- Yeni Sütunların Sayısı ve genre Sütun Adı ---")
print(f"Toplam Sütun Sayısı: {len(df_final.columns)}")
genre_cols = [col for col in df_final.columns if col.startswith("playlist_genre") or col.startswith("playlist_subgenre")]
print(genre_cols)

# print(df_final.columns[0:45].tolist())

df_final["artist_popularity"] = df_final.groupby("track_artist")["track_popularity"].transform("mean")
# df_final.groupby("track_artist")["track_popularity"].mean()
# Bu işlem, track_artist adlarını Index olarak kullanan ve sadece sanatçı sayısı kadar satırdan oluşan kısa bir Pandas Series oluşturur.
# index sayılarını uyumlu hale getirmek için transform kullandık, her sanatçı bulunan satıra eklendi
# kullanmadığımızda satırların çoğuna null değer atanır

df_final["release_date"] = pd.to_datetime(df_final["track_album_release_date"], errors="coerce")
# Veri setinde bazı satırlar hatalı veya eksik tarih bilgisine sahip olabilir.
# coerce parametresi, Pandas'a geçerli olmayan tüm tarihleri
# (örneğin "19XX-00-00" gibi) NaT (Not a Time) olarak ayarlamasını söyler.
# Bu, kodun çökmesini önler.

df_final["release_year"] = df_final["release_date"].dt.year
mode_year = df_final["release_year"].mode().iloc[0]
df_final["release_year"].fillna(mode_year, inplace=True)
# .mode() fonksiyonu, bir Series'deki en sık tekrar eden değeri/değerleri döndürür.
# Sonuç bir Series olduğu için, ilk elemanı almak için .iloc[0] kullanılır.


# .dt erişimcisi, bir datetime Series'indeki tarih/saat özelliklerine ulaşmamızı sağlar.
# .year özelliği, sadece o tarihin yıl bileşenini çıkarır.
# Sonuç bir tam sayıdır (örneğin 2018).

df_final['energy_dance'] = df_final['danceability'] * df_final['energy']     # yüksek enerji ve dans ilişkisi
#  df_final['acoustic_val'] = df_final['acousticness'] * df_final['valence']    # yüksek ses ve canlılık ilişkisi
# df_final['live_loud'] = df_final['liveness'] * df_final['loudness']          # akustik şarkı ve pozitif duygu ilişkisi

df_final["speech_pop"] = df_final['speechiness'] * df_final["artist_popularity"]

df_final["genre_popularity"] = df1.groupby("playlist_genre")["track_popularity"].transform("mean")
# df_final["subgenre_popularity"] = df1.groupby("playlist_subgenre")["track_popularity"].transform("mean")

df_final["acoustic_energy"] = df_final["acousticness"] * df_final["energy"]

features = [ "acousticness", "energy", "loudness", "tempo", "valence","artist_popularity", "release_year",'energy_dance',"genre_popularity","instrumentalness","liveness","acoustic_energy","speech_pop"]
# features.append(genre_cols)
# bu şekilde yapılınca hata vardi çünkü böyle string olarak ekliyor col adı olarak değil.
features.extend(genre_cols)
x = df_final[features]  # HATA Düzeltildi: Çift köşeli parantez kullanıldı: df1[["danceability", "acousticness", "energy"]]
y = df_final["track_popularity"]

# Veri Setini Ayırma
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) # random_state eklendi

# Model Oluşturma ve Eğitme
# model = LinearRegression()
model = RandomForestRegressor(random_state=42, n_jobs=-1)

param_grid = {
    "n_estimators" : [100,150],
    "max_depth" : [30,20],
    "min_samples_split" : [5,10]
}
grid = GridSearchCV(   # grid objesi tanımlandı
    estimator=model,    # Hangi modelin optimize edileceği
    param_grid=param_grid,    # Hangi parametrelerin deneneceği
    scoring= "r2",          # Değerlendirme metriği (R2 skoru)
    cv = 3,                 # Çapraz doğrulama (Cross-Validation) kat sayısı (3 kat)
    verbose = 2,           # Detaylı çıktı gösterme
    n_jobs = -1           # Tüm işlemci çekirdeklerini kullan
)
grid.fit(x_train, y_train)

# Tahmin Yapma
best_params = grid.best_params_
best_score = grid.best_score_
best_rf_model = grid.best_estimator_ # Optimize edilmiş modeli değişkene atama

print("\n--- Optimizasyon Sonuçları ---")
print(f"Grid Search En Yüksek R2 Skoru (CV): {best_score:.4f}")
print(f"En İyi Parametre Kombinasyonu: {best_params}")

# --- Tahmin ve Skorlama (Düzeltildi) ---

# Tahmin Yapma: Optimize Edilmiş Modeli kullan
y_pred = best_rf_model.predict(x_test)
print("--- Model Tahminleri (İlk 10) ---")
print(y_pred[:10])


# Model Performansını Değerlendirme
# Not: r2_score ve model.score(x_test, y_test) aynı sonucu verir.
r2_score_value = r2_score(y_test, y_pred)
model_score = best_rf_model.score(x_test, y_test) # Bu da R-kare değeridir (R2)

print("\n--- Model Performans Sonuçları ---")
print(f"R-Kare (r2_score): {r2_score_value:.4f}")
print(f"Model Skoru (model.score): {model_score:.4f}")

features_for_heatmap = ['track_popularity', 'danceability', 'energy', 'acousticness']
corr_matrix = df1[features_for_heatmap].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='vlag', fmt=".2f", linewidths=.5, linecolor='white')
plt.title('Ana Müzik Özellikleri ve Popülerlik Arasındaki Korelasyon', fontsize=14)
plt.show()

# 2. Ayrıntılı Dağılım Grafikleri (Tek Tek Etki)
features_to_plot_single = ['danceability', 'energy', 'acousticness']
fig, axes = plt.subplots(1, 3, figsize=(21, 6))

plt.suptitle('Popülerlik Skoruna Göre Tek Tek Özelliklerin Dağılımı ve Trendi', fontsize=16)

for i, feature in enumerate(features_to_plot_single):
    sns.regplot(
        x=feature, y='track_popularity', data=df1, ax=axes[i],
        scatter_kws={'alpha':0.05, 's':10}, line_kws={'color':'red', 'lw':2}, ci=None
    )
    axes[i].set_title(f'{feature.capitalize()} vs. Popülerlik', fontsize=14)
    axes[i].set_xlabel(feature.capitalize(), fontsize=12)
    axes[i].set_ylabel('Popülerlik Skoru (0-100)', fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

print("--- Grafik Oluşturma Tamamlandı ---")
