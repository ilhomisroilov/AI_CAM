"""
============================================================
run.py  —  AI_CAM ishga tushirish nuqtasi
============================================================
Ishlatish:
    python run.py
So'ng brauzerda oching:  http://localhost:8000/dashboard
"""
import uvicorn

from backend.config import SERVER

if __name__ == "__main__":
    # reload=False — fon threadlari (kamera/OCR) bilan ziddiyat bo'lmasligi uchun
    uvicorn.run(
        "backend.server:app",
        host=SERVER.host,
        port=SERVER.port,
        reload=False,
        log_level="info",
    )
