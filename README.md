# Melodi DNA — Tam Çalışan Web Sitesi

Bu sürüm önceki görsel demodan farklı olarak gerçek ses dosyalarını sunucu tarafında analiz eder.

## Özellikler
- WAV / MP3 / FLAC / OGG / M4A / AAC yükleme
- Gerçek perde (pitch) çıkarımı
- Nota aralığı benzerliği
- Melodik kontur benzerliği
- Ritmik yapı benzerliği
- Tekrarlanan motif benzerliği
- 0–100 genel skor
- Sonuç çubukları ve melodik perde grafiği
- Mobil uyumlu, yumuşak ve sade tasarım

## Model
DNA = 0.35 × Nota Aralığı + 0.25 × Melodik Kontur + 0.20 × Ritmik Yapı + 0.20 × Motif

Nota aralığı ve motif karşılaştırması, melodinin başka bir tona taşınmasından mümkün olduğunca az etkilenmesi için göreli/kuantize edilmiş temsil kullanır.

## Çalıştırma
Python 3.10+ önerilir.

    python -m venv .venv

Windows:
    .venv\Scripts\activate

macOS/Linux:
    source .venv/bin/activate

    pip install -r requirements.txt
    python app.py

Sonra tarayıcıdan:
    http://127.0.0.1:5000

## Not
MP3/M4A gibi bazı formatların okunması sistemdeki ses çözücü desteğine bağlı olabilir. En sorunsuz test formatı WAV'dır.

Bu proje bilimsel yarışma sürümünün temelidir. Gerçek TÜBİTAK raporunda algoritmanın doğruluğu ayrıca gerçek veri seti ve insan değerlendirmeleriyle ölçülmelidir.
