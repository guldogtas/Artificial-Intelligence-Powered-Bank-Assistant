# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:05:28 2024

@author: guldogtas
"""

import pandas as pd

# Function that classifies credit eligibility based on credit scores
def kredi_durum_siniflandirma(not_degeri):
    if not_degeri <= 699:
        return "Çok Riskli - Krediye Uygun Değil"
    elif 700 <= not_degeri <= 1099:
        return "Orta Riskli - Kefil ile Düşük Limitli Kredi"
    elif 1100 <= not_degeri <= 1499:
        return "Az Riskli - İhtiyaç Kredisi (Düşük-Orta Limit)"
    elif 1500 <= not_degeri <= 1699:
        return "İyi - İhtiyaç ve Konut Kredisi (Orta-Yüksek Limit)"
    else:  # 1700 - 1900 range
        return "Çok İyi - Tüm Kredi Türlerine Uygun"

# Sample DataFrame
data = pd.DataFrame({
    'Kredi Notu': [1405, 1549, 870, 1170, 1395]
})

# Adding the eligibility status
data['Kredi Uygunluk Durumu'] = data['Kredi Notu'].apply(kredi_durum_siniflandirma)

# Displaying the table
print(data)

def kredi_onerisi(kredi_notu, gelir, kredi_turu="ihtiyaç"):
    if kredi_notu >= 1500:
        if kredi_turu == "konut":
            return "Konut Kredisi - Uygun: Faiz %1.2, Limit: 500.000 TL"
        elif kredi_turu == "taşıt":
            return "Taşıt Kredisi - Uygun: Faiz %1.5, Limit: 150.000 TL"
        else:
            return "İhtiyaç Kredisi - Uygun: Faiz %1.8, Limit: 50.000 TL"
    elif 1100 <= kredi_notu < 1500:
        return "İhtiyaç Kredisi - Düşük Limit: Faiz %2.0, Limit: 20.000 TL"
    else:
        return "Krediye uygun değilsiniz. Daha yüksek kredi notuna ihtiyaç var."

# Example usage
print(kredi_onerisi(1600, 5000, kredi_turu="konut"))

