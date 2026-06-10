"""
============================================================
run.py  —  AI_CAM ishga tushirish nuqtasi (LAN ga ochiq)
============================================================
Ishlatish:
    python run.py

Server 0.0.0.0 ga bog'lanadi — ya'ni shu kompyuterning BARCHA tarmoq
interfeyslarida tinglaydi. Zavod tarmog'idagi boshqa kompyuterlar
brauzerda quyidagini ochib kirishadi:
    http://<SHU_KOMPYUTER_IP>:8000/dashboard

Ishga tushganda server aniq ulanish manzilini chop etadi.
Eslatma: boshqa kompyuterlar ulana olishi uchun Windows Firewall da
8000-port (TCP) uchun kiruvchi ruxsat ochilishi kerak (pastdagi izohga qarang).
"""
import socket

import uvicorn

from backend.config import SERVER

# LAN kirish uchun har doim barcha interfeyslarni tinglaymiz
HOST = "0.0.0.0"
PORT = SERVER.port


def get_lan_ip() -> str:
    """
    Shu kompyuterning asosiy LAN IP manzilini aniqlaydi (masalan 192.168.x.x).
    Internet TALAB QILINMAYDI — paket yuborilmaydi, faqat marshrut tanlanadi.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Tashqi manzilga "ulanish" — OS to'g'ri chiquvchi interfeysni tanlaydi
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"          # tarmoq yo'q bo'lsa zaxira
    finally:
        s.close()
    return ip


def _print_banner(lan_ip: str) -> None:
    line = "=" * 56
    print("\n" + line)
    print("  AI_CAM server ishga tushdi — LAN ga ochiq (host 0.0.0.0)")
    print(line)
    print(f"  Shu kompyuterda:     http://localhost:{PORT}/dashboard")
    print(f"  Tarmoqdagi boshqalar: http://{lan_ip}:{PORT}/dashboard   <-- ULASHING")
    print(line)
    print("  Eslatma: boshqa PC ulana olmasa, Windows Firewall da")
    print(f"  {PORT}-port (TCP, Inbound) uchun ruxsat oching (quyidagi izohga qarang).")
    print(line + "\n")


if __name__ == "__main__":
    _print_banner(get_lan_ip())
    # reload=False — fon threadlari (kamera/OCR) bilan ziddiyat bo'lmasligi uchun
    uvicorn.run(
        "backend.server:app",
        host=HOST,          # 0.0.0.0 — barcha interfeyslar (LAN kirish)
        port=PORT,
        reload=False,
        log_level="info",
    )
