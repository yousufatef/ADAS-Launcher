from PySide6.QtWidgets import QApplication, QLabel

app = QApplication([])
label = QLabel("🎯 Object Tracking launched!")
label.resize(300, 100)
label.show()
app.exec()
