import sys
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QMainWindow, QTextEdit, QAction, QFileDialog, QApplication, QLabel, QPushButton)
import cv2


class Example(QMainWindow):

    def __init__(self):
        self.width = 800
        self.height = 600
        self.left = 200
        self.top = 200
        self.Lleft = 50
        self.Ltop = 50
        self.Lwidth = 500
        self.Lheight = 500
        self.Bleft = 600
        self.Btop = 50
        self.B1left = 600
        self.B1top = 150
        self.FlagB = True
        self.path = None


        super().__init__()
        self.initUI()

    def initUI(self):
        # ____________________main window__________________________
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setWindowTitle('File dialog')


        #__________________MENU___________________________________
        openFile = QAction('Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.triggered.connect(self.showDialog)


        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFile)


        #__________________________LAbel__________________
        self.label = QLabel(self)
        # self.label = QTextEdit(self)
        self.label.setGeometry(self.Lleft, self.Ltop, self.Lwidth, self.Lheight)

        #_______________________Button_______________________
        btn = QPushButton('Выделение', self)
        # btn.setToolTip('This is a <b>QPushButton</b> widget')
        # btn.resize(btn.sizeHint())
        btn.resize(100, 50)
        btn.move(self.Bleft, self.Btop)
        btn.clicked.connect(self.ButtonClick)

        #_____________________Button save________________
        # _______________________Button_______________________
        btn1 = QPushButton('Сохранить область', self)
        # btn.setToolTip('This is a <b>QPushButton</b> widget')
        # btn.resize(btn.sizeHint())
        btn1.resize(150, 50)
        btn1.move(self.B1left, self.B1top)

        #_____________________Индикатор_________________
        Indicator = QLabel(u'<span style="font-size: 32pt; color: red;">•</span>')
        Indicator.move(110,50)
        Indicator.show()

        self.show()

    def showDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', 'C:/')

        self.path = fname[0]

        img = cv2.imread(self.path)

        pixmap = QPixmap(self.path)
        # pixmap.size()
        pixmap = pixmap.scaledToHeight(self.Lheight)
        pixmap = pixmap.scaledToWidth(self.Lwidth)
        self.label.setPixmap(pixmap)
        print(self.path)

    def ButtonClick(self):
        if self.FlagB:
            print('red')
        else:
            print('green')

        self.FlagB = not self.FlagB

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
