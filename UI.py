import sys
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton, QMessageBox, QLineEdit, QLabel
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
import numpy as np
import lightgbm as lgb

class App(QMainWindow):
 
    def __init__(self):
        super().__init__()
        # Setting window properties
        self.title = 'Churn Prediction'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initUI()
 
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        # Adding textboxes, button(for giving prediction) and labels
        self.text1 = QLineEdit(self)
        self.text1.move(120, 20)
        self.text1.resize(180,20)

        self.text2 = QLineEdit(self)
        self.text2.move(120, 80)
        self.text2.resize(180,20)

        self.text3 = QLineEdit(self)
        self.text3.move(120, 140)
        self.text3.resize(180,20)

        self.text4 = QLineEdit(self)
        self.text4.move(120, 200)
        self.text4.resize(180,20)

        self.text5 = QLineEdit(self)
        self.text5.move(120, 260)
        self.text5.resize(180,20)

        self.text6 = QLineEdit(self)
        self.text6.move(120, 320)
        self.text6.resize(180,20)

        self.text7 = QLineEdit(self)
        self.text7.move(450, 20)
        self.text7.resize(180,20)

        self.text8 = QLineEdit(self)
        self.text8.move(450, 80)
        self.text8.resize(180,20)
        
        self.text9 = QLineEdit(self)
        self.text9.move(450, 140)
        self.text9.resize(180,20)


        button = QPushButton('Submit', self)
        button.setToolTip('Click to predict')
        button.move(100,400) 
        button.clicked.connect(self.on_click)

        self.label1 = QLabel(self)
        self.label1.setText("Name")
        self.label1.move(20,20)
        self.label1.resize(180,20)
        
        self.label2 = QLabel(self)
        self.label2.setText("Age")
        self.label2.move(20,80)
        self.label2.resize(180,20)

        self.label3 = QLabel(self)
        self.label3.setText("Credit Score")
        self.label3.move(20,140)
        self.label3.resize(180,20)

        self.label4 = QLabel(self)
        self.label4.setText("Nationality")
        self.label4.move(20,200)
        self.label4.resize(180,20)

        self.label5 = QLabel(self)
        self.label5.setText("Gender")
        self.label5.move(20,260)
        self.label5.resize(180,20)

        self.label6 = QLabel(self)
        self.label6.setText("Number of \n products")
        self.label6.move(20,320)
        self.label6.resize(180,35)

        self.label7 = QLabel(self)
        self.label7.setText("Has a \ncredit card?")
        self.label7.move(370,20)
        self.label7.resize(180,35)

        self.label8 = QLabel(self)
        self.label8.setText("Is an active \nmember")
        self.label8.move(370,80)
        self.label8.resize(180,35)
        
        self.label9 = QLabel(self)
        self.label9.setText("Balance")
        self.label9.move(370,140)
        self.label9.resize(180,20)

        self.show()

    @pyqtSlot()
    def on_click(self):
        textboxValue1 = self.text1.text()
        german, spanish = 0,0
        gender = 0
        p1, p2, p3 = 0,0,0
        c1, c2, c3, c4 = 0,0,0,0
        active, has_card = 0,0
        
        
        # Creating features for nationality(french, spanish or german only)
        nation = self.text4.text()
        #print(nation.lower())
        if nation.lower() == 'spanish':
                #print(nation.lower())
                spanish = 1
                german = 0
        elif nation.lower == 'germany':
                german = 1
                spanish = 0
        else:
                german = 0
                spanish = 0
        print(str(german)+'  '+str(spanish))
        
        # Creating feature for gender
        gen = self.text5.text()
        if gen.lower() == 'male':
                gender = 1
        else:
                gender = 0
        print(gender)
        
        # Creating features for number of products(products can only be 1 to 4)
        
        prod = self.text6.text()
        if prod == '1':
                p1 = 1
        elif prod == '2':
                p2 = 1
        elif prod == '3':
                p3 = 1
        print(p1)
        print(p2)
        print(p3)
        
        # Creating features for credit score
        credit_score = int(self.text3.text())
        if credit_score >= 580 and credit_score <= 669:
                c1 = 1
        elif credit_score >= 670 and credit_score <= 739:
                c2 = 1
        elif credit_score >= 740 and credit_score <= 799:
                c3 = 1
        elif credit_score >= 800:
                c4 = 1
        print(str(c1)+' '+str(c2)+' '+str(c3)+' '+str(c4))
        print(credit_score)
        
        # Creating feature for age
        age = int(self.text2.text())
        print(age)
        
        # Creating feature for balance
        balance = float(self.text9.text())
        print(balance)
        
        # Creating feature for has a card and is active member
        card = self.text7.text()
        if card.lower() == 'yes':
                has_card = 1 
        else:
                has_card = 0
        
        is_active = self.text8.text()
        if is_active.lower() == 'yes':
                active = 1 
        else:
                active = 0
        
        print(str(has_card)+' '+str(active))
        
        test = np.array([spanish, german, c1, c2, c3, c4, credit_score, gender, age, balance, has_card, active, p1, p2, p3]).reshape((15,1))
       
        test = np.transpose(test)
        
        model = lgb.Booster(model_file='model.txt')
        res = model.predict(test)
        # Using information box to show result        
        if res < 0.5:
                buttonReply2 = QMessageBox.information(self,'Result',str(textboxValue1+' '+'is not exiting'))
        else:
                buttonReply2 = QMessageBox.information(self,'Result',str(textboxValue1+' '+'is exiting'))
        
        
 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
