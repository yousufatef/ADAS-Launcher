# launcher.py — ADAS Launcher بواجهة Grid أيقونات (تفتح تلقائياً)
import sys, subprocess, pathlib, traceback
from functools import partial
from PySide6.QtCore    import Qt, QSize, QPoint
from PySide6.QtGui     import QIcon, QPixmap, QColor, QPainter, QCursor
from PySide6.QtWidgets import (
    QApplication, QWidget, QToolButton, QGridLayout,
    QSystemTrayIcon, QMenu
)

BASE_DIR  = pathlib.Path(__file__).resolve().parent
ICON_PATH = BASE_DIR / "adas_icon.ico"

FEATURES = {
    "ACC": ("features/acc.py", "icons/acc.jpg"),
    "LKA": ("features/lka.py", "icons/lka.jpg"),
    "BSM": ("features/bsm.py", "icons/bsm.jpg"),
    "DMS": ("features/dms.py", "icons/dms.jpg"),
    "PA":  ("features/pa.py",  "icons/pa.jpg"),
}

# ---------- أيقونة Tray افتراضية ----------
def fallback_icon(size=64):
    pix = QPixmap(size, size)
    pix.fill(QColor("#0078d4"))
    p = QPainter(pix)
    p.setPen(QColor("white"))
    p.drawText(pix.rect(), Qt.AlignCenter, "A")
    p.end()
    return QIcon(pix)

def tray_icon():
    ico = QIcon(str(ICON_PATH)) if ICON_PATH.exists() else QIcon()
    return ico if not ico.isNull() else fallback_icon()
# -------------------------------------------

def run_feature(script_rel):
    path = BASE_DIR / script_rel
    if not path.exists():
        print(f"[!] الملف مش موجود: {path}")
        return
    subprocess.Popen(
        [sys.executable, str(path)],
        creationflags=subprocess.CREATE_NO_WINDOW   # يمنع ظهور الكونسول
    )

# ---------- نافذة اللوحة ----------
class FeatureWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ADAS Features")
        self.setFixedSize(360, 220)
        self.setWindowFlags(Qt.Tool | Qt.WindowStaysOnTopHint)

        grid = QGridLayout(self)
        grid.setSpacing(12)

        row = col = 0
        for label, (script, icon_rel) in FEATURES.items():
            btn = QToolButton()
            btn.setText(label)
            btn.setIcon(QIcon(str(BASE_DIR / icon_rel)) if (BASE_DIR / icon_rel).exists() else QIcon())
            btn.setIconSize(QSize(64, 64))
            btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)

            if (BASE_DIR / script).exists():
                btn.clicked.connect(partial(run_feature, script))
            else:
                btn.setDisabled(True)
                btn.setToolTip("⚠️ الملف غير موجود")

            grid.addWidget(btn, row, col)
            col += 1
            if col == 3:
                col = 0
                row += 1
# -------------------------------------------

def build_tray():
    app  = QApplication(sys.argv)
    tray = QSystemTrayIcon(tray_icon())
    tray.setToolTip("ADAS Launcher")

    panel = FeatureWindow()

    # قائمة يمين-كليك (Exit فقط)
    menu = QMenu()
    menu.addAction("Exit", app.quit)
    tray.setContextMenu(menu)

    # وظائف إظهار / إخفاء اللوحة
    def toggle_panel():
        if panel.isVisible():
            panel.hide()
        else:
            pos = QCursor.pos()
            if pos == QPoint(0, 0):  # fallback لمنتصف الشاشة
                screen_rect = QApplication.primaryScreen().geometry()
                pos = screen_rect.center() - panel.rect().center()
            panel.move(pos)
            panel.show()
            panel.raise_()
            panel.activateWindow()

    def on_activated(reason):
        if reason in (
            QSystemTrayIcon.Trigger,
            QSystemTrayIcon.DoubleClick,
            QSystemTrayIcon.Context,
        ):
            toggle_panel()

    tray.activated.connect(on_activated)

    # استثناءات تُعرض كرسالة
    def excepthook(exctype, value, tb):
        msg = "".join(traceback.format_exception(exctype, value, tb))
        tray.showMessage("خطأ فى ADAS", msg[:250], QSystemTrayIcon.Critical, 4000)
        sys.__excepthook__(exctype, value, tb)
    sys.excepthook = excepthook

    tray.show()
    tray.showMessage(
        "ADAS Launcher",
        "تم التشغيل - لوحة الميزات ستظهر الآن.",
        QSystemTrayIcon.Information,
        2500
    )

    # ✨ عرض اللوحة فوراً بعد بدء التطبيق
    toggle_panel()

    sys.exit(app.exec())

if __name__ == "__main__":
    build_tray()
