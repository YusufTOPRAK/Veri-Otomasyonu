# from PyQt5.QtCore import QAbstractTableModel, Qt
# from PyQt5.QtGui import QFont, QPixmap
# from PyQt5.QtCore import Qt, QSize
# from panel import*
# from data_frame_model import*
# from PyQt5 import QtWidgets
# from PyQt5.QtWidgets import*
# from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QVBoxLayout, QWidget
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# import matplotlib.pyplot as plt
# from matplotlib.figure import Figure
# import seaborn as sns
# import sys
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# # from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import OrdinalEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer  
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor 
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.cluster import KMeans, AgglomerativeClustering
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  
# import io
# import os
# import scipy.cluster.hierarchy as sch




# class main(QMainWindow, DataFrameModel):
#     def __init__(self):
#         super().__init__()
#         self.ui = Ui_MainWindow()
#         self.ui.setupUi(self)
        
#         ## Grafik işlemleri için
#         self.figure = Figure()
#         self.canvas = FigureCanvas(self.figure,)
#         self.scene = QGraphicsScene()
#         self.ui.graphicsView.setScene(self.scene)
#         self.scene.addWidget(self.canvas)

#         # dosya yolu
#         self.dosya_yolu = os.path.expanduser("~") # genel dizin
#         self.dosyalar = os.listdir(self.dosya_yolu) # ulaşılan dosyaları tutma
        
#         # veri setlerini temsil edecek değişkenler
#         self.df = None
#         self.df_copy = None


#         # textEdit'e yazılanların düzgün gözükmesi için
#         font = QFont("Courier New", 10) 
#         self.ui.textEdit.setFont(font) 
#         self.ui.textEdit_2.setFont(font)
#         self.ui.textEdit_3.setFont(font)
#         self.ui.textEdit_4.setFont(font)

#         # butons
#         self.ui.btn_oku_2.clicked.connect(self.dosya_goster)
#         self.ui.btn_oku.clicked.connect(self.read_df)
#         self.ui.btn_oku_3.clicked.connect(self.execute_code)
#         self.ui.btn_info.clicked.connect(self.info)
#         self.ui.btn_correlation_2.clicked.connect(self.mean)
#         self.ui.btn_correlation_3.clicked.connect(self.median)
#         self.ui.btn_correlation_4.clicked.connect(self.mode)
#         self.ui.btn_desc.clicked.connect(self.desc)
#         self.ui.btn_correlation.clicked.connect(self.correlation)
#         self.ui.btn_unique.clicked.connect(self.unique)
#         self.ui.pushButton_2.clicked.connect(self.show_missing_values)
#         self.ui.pushButton_3.clicked.connect(self.missing_value_change)
#         self.ui.btn_remove_col.clicked.connect(self.remove_col)
#         self.ui.btn_remove_col_2.clicked.connect(self.change_type)
#         self.ui.btn_outliers.clicked.connect(self.outliers)
#         self.ui.btn_outliers_index.clicked.connect(self.outliers_index)
#         self.ui.btn_outliers_change.clicked.connect(self.change_outliers)
#         self.ui.btn_label_encoding.clicked.connect(self.label_encoding)
#         self.ui.btn_onehot_encoding.clicked.connect(self.one_hot_encoding)
#         self.ui.btn_ordinal_encoding.clicked.connect(self.ordinal_encoding)
#         self.ui.pushButton.clicked.connect(self.add_categories_for_ordinal)
#         self.ui.btn_split.clicked.connect(self.train_test_split)
#         self.ui.btn_split_2.clicked.connect(self.scaler)
#         self.ui.btn_split_3.clicked.connect(self.poly)
#         self.ui.btn_model.clicked.connect(self.linear_model)
#         self.ui.btn_model_4.clicked.connect(self.decision_tree_model)
#         self.ui.btn_model_7.clicked.connect(self.random_forest_model)
#         self.ui.btn_model_2.clicked.connect(self.svr_model)
#         self.ui.btn_model_3.clicked.connect(self.logistic_regression)
#         self.ui.btn_model_6.clicked.connect(self.svc_model)
#         self.ui.btn_model_5.clicked.connect(self.k_neighbors_classifier)
#         self.ui.btn_model_8.clicked.connect(self.decision_tree_classifier)
#         self.ui.btn_model_9.clicked.connect(self.random_forest_classifier)
#         self.ui.btn_model_10.clicked.connect(self.naive_bayes_classifier)
#         self.ui.btn_model_11.clicked.connect(self.k_means_model)
#         self.ui.btn_model_12.clicked.connect(self.hierarchical_clustring)


#     # fonksiyonlar
#     def dosya_goster(self):
#         options = QFileDialog.Options()
#         options |= QFileDialog.ReadOnly
#         fileName, _ = QFileDialog.getOpenFileName(self, "Dosya Seç", self.dosya_yolu, "Tüm Dosyalar (*);;Excel Dosyaları (*.xlsx)", options=options)
        
#         # dosya türlerinin kontrolü
#         if fileName:
#             self.ui.lineEdit_2.setText(fileName)          
#             try:
#                 if fileName.endswith('.xlsx'):
#                     self.df = pd.read_excel(fileName)
#                 elif fileName.endswith('.csv'):
#                     self.df = pd.read_csv(fileName)
#                 elif fileName.endswith('.json'):
#                     self.df = pd.read_json(fileName)
#                 elif fileName.endswith('.html'):
#                     self.df = pd.read_html(fileName)
#                 elif fileName.endswith('.sav'):
#                     self.df = pd.read_spss(fileName)
#                 elif fileName.endswith('.sql'):
#                     self.df = pd.read_sql(fileName)   
#                 else:
#                     raise ValueError("Unsupported file format")
#                 self.show_dataframe()
#             except Exception as e:
#                 QMessageBox.critical(self, "File Upload Error", f"An error occurred while uploading the file: {str(e)}")
#                 self.ui.lineEdit_2.clear()

#     def show_dataframe(self):
#         if not self.check_file():
#                 return
        
#         self.model = DataFrameModel(self.df)
#         self.model_copy = DataFrameModel(self.df.copy())
#         self.ui.cmb_data.clear() 
#         self.ui.cmb_data.addItems(self.df.columns.tolist())
#         self.model2 = DataFrameModel(self.df.copy())

#     def check_file(self):
#         if self.df is None:
#             QMessageBox.critical(self, "File Selection Error", "Choose file")
#             return False
#         return True      
    
#     def check_col(self):
#         selected_col = self.ui.cmb_data.currentText()
#         if selected_col not in self.df.columns:
#             QMessageBox.critical(self, "Column Error", f"The selected column '{selected_col}' does not exist.")
#             return False
#         return True
    
#     def type_control(self):
#         selected_col = self.ui.cmb_data.currentText()
#         if not pd.api.types.is_numeric_dtype(self.df[selected_col]):
#             QMessageBox.critical(self, "Type Error", f"The '{selected_col}' column must be numeric.")
#             return False
#         return True

#     def read_df(self):
#         if not self.check_file():
#             return
                 
#         self.ui.tableView.setModel(self.model)
#         self.ui.tableView_2.setModel(self.model_copy)
 

#     def update_table_view(self):
#         # self.df üzerinde yapılan işlemlerden sonra tabloyu güncelle
#         self.model = DataFrameModel(self.df)  # Yeni bir model oluştur
#         self.ui.tableView.setModel(self.model)  # Modeli QTableView'e ata
#         self.model.layoutChanged.emit()  # QTableView'i güncelle

#     def execute_code(self):
#             if not self.check_file():
#                 return
            
#             # Kod metnini QTextEdit'ten al
#             code = self.ui.textEdit_3.toPlainText()
            
#             try:
#                 # Standart çıktı ve hata çıktısını yakalamak için yönlendir
#                 from io import StringIO
#                 import sys
#                 import traceback
                
#                 old_stdout = sys.stdout
#                 old_stderr = sys.stderr
#                 sys.stdout = StringIO()
#                 sys.stderr = StringIO()
                
#                 # local_vars içinde self'i geçiyoruz
#                 local_vars = {'self': self}
                
#                 # Kodu çalıştır
#                 exec(code, globals(), local_vars)
                
#                 # Yönlendirilen çıktıyı al ve QTextEdit'te göster
#                 output = sys.stdout.getvalue()
#                 error_output = sys.stderr.getvalue()
                
#                 # Eğer hata yoksa normal çıktıyı, varsa hata çıktısını göster
#                 if error_output:
#                     self.ui.textEdit_4.setPlainText(error_output)
#                 else:
#                     self.ui.textEdit_4.setPlainText(output)
#                     # self.canvas.draw()
                
#                 # Kod çalıştıktan sonra tabloyu güncelle
#                 self.update_table_view()
            
#             except Exception as e:
#                 # Hata mesajlarını yakala ve göster
#                 output = traceback.format_exc()
#                 self.ui.textEdit_4.setPlainText(output)
#             finally:
#                 # Standart çıktı ve hata çıktısını eski haline döndür
#                 sys.stdout = old_stdout
#                 sys.stderr = old_stderr

#     def info(self):
#         if not self.check_file():
#             return
                    
#         # Bilgi özetini bir StringIO nesnesine yönlendiriyoruz
#         # info
#         buffer = io.StringIO()
#         self.df.info(buf=buffer)
#         info_text = buffer.getvalue()  # StringIO'dan metni alıyoruz
#         # self.ui.textEdit.setText(info_text)  # QTextEdit'e yazdırıyoruz

#         # types
#         ty = self.df.dtypes
#         string_ty = ty.to_string()
#         # self.ui.textEdit.setText(string_ty)

#         # index            
#         index = self.df.index
#         string_index = str(index)
#         # self.ui.textEdit.setText(string_index)

#         # column
#         col_names = self.df.columns
#         string_col_names = "\n".join(map(str, col_names))

#         result = f"Info:\n{info_text}\nTypes:\n{string_ty}\n\nIndex:\n{string_index}\n\nColumns:\n{string_col_names}"
            
#         self.ui.textEdit.setText(result)            


#     def mean(self):
#         if not self.check_file():
#             return
        
#         if not self.check_col():
#             return
        
#         if not self.type_control():
#             return
        
#         selected_col = self.ui.cmb_data.currentText()

#         mean = self.df[selected_col].mean()
#         self.ui.textEdit.setText(f"{selected_col}: {str(mean)}")


#     def median(self):
#         if not self.check_file():
#             return

#         if not self.check_col():
#             return
        
#         if not self.type_control():
#             return
        
#         selected_col = self.ui.cmb_data.currentText()

#         median = self.df[selected_col].median()
#         self.ui.textEdit.setText(f"{selected_col}: {str(median)}")

   
#     def mode(self):
#         if not self.check_file():
#             return
        
#         if not self.check_col():
#             return
        
#         selected_col = self.ui.cmb_data.currentText()
        
#         mode = self.df[selected_col].mode().iloc[0]
#         self.ui.textEdit.setText(f"{selected_col}: {str(mode)}")

#     def desc(self):
#         if not self.check_file():
#             return 

#         describe = self.df.describe()
#         string_describe = describe.to_string()
#         self.ui.textEdit.setText(string_describe)               

#         # if self.df is not None:
#         #     describe = self.df.describe()
#         #     string_describe = describe.to_string()
#         #     self.ui.textEdit.setText(string_describe)
#         # else:
#         #     QMessageBox.critical(self, "Dosya Seçim Hatası", "Dosya seçiniz")            

#     def correlation(self):
#         if not self.check_file():
#             return         

#         corr = self.df.corr(numeric_only=True)
#         string_cor = corr.to_string()
#         self.ui.textEdit.setText(string_cor)

#         self.figure.clear() # başka grafik için komple temizler
#         ax = self.figure.add_subplot(111)
#         ax.clear()
#         sns.heatmap(corr, annot=True,ax=ax)
#         self.canvas.draw() 

#     def unique(self):
#         if not self.check_file():
#             return   
        
#         if not self.check_col():
#             return

#         selected_col = self.ui.cmb_data.currentText() # Seçilen kolonu ele alma
#         unique_values = self.df[selected_col].unique() # unique değerleri alma
#         n_unique_values = self.df[selected_col].nunique()
#         self.ui.textEdit.setText(f"There are {n_unique_values} unique values.\n\n{'\n'.join(map(str, unique_values))}")


#     def show_missing_values(self):
        
#         if not self.check_file():
#             return
        
#         if not self.check_col():
#             return

#         selected_col = self.ui.cmb_data.currentText()
#         genel_toplam = self.df.isnull().sum().sum()
#         kolon_bazlı_toplam = self.df.isnull().sum()
#         kolon_toplam = self.df[selected_col].isnull().sum()
#         index = self.df[self.df[selected_col].isna()].index
#         liste = []
#         for i in index:
#             liste.append(i)

#         result = f"Total Missing Data : {genel_toplam}\n\nMissing Data for '{selected_col}': {kolon_toplam}\nIndex: {liste}\n\nColumn-wise Total:\n{kolon_bazlı_toplam}"

#         self.ui.textEdit.setText(result)


#     def missing_value_change(self):
#         if not self.check_file():
#             return

#         if not self.check_col():
#             return

#         selected_col = self.ui.cmb_data.currentText()            
#         if not self.df[selected_col].isnull().any():
#             QMessageBox.information(self, "No Missing Data", f"There are no missing values in the '{selected_col}' column")
#             return  # İşlemi durdur

        
#         methods = ["Remove Observation Value","Fill with Mean", "Fill with Median", "Fill with Mode", "Fill with Fixed Value"]
#         method, ok = QInputDialog.getItem(self, "Choose Filling Method", "How would you like to fill the missing values?", methods, 0, False)
#         if ok and method:
#             if method == "Remove Observation Value":
#                 self.df.dropna(subset=[selected_col], inplace=True)
#                 self.df.reset_index(drop=True, inplace=True)
#                 # self.df.dropna(inplace=True)
#             if method == "Fill with Mean":
#                 if not self.type_control():
#                     return
#                 self.df[selected_col] = self.df[selected_col].fillna(self.df[selected_col].mean())
#                 self.ui.textEdit.setText(f"Mean: {str(self.df[selected_col].mean())}")
#             elif method == "Fill with Median":
#                 if not self.type_control():
#                     return
#                 self.df[selected_col] = self.df[selected_col].fillna(self.df[selected_col].median())
#                 self.ui.textEdit.setText(f"Median: {str(self.df[selected_col].median())}")
#             elif method == "Fill with Mode":
#                 self.df[selected_col] = self.df[selected_col].fillna(self.df[selected_col].mode()[0])
#                 self.ui.textEdit.setText(f"Mode: {str(self.df[selected_col].mode())}")
#             elif method == "Fill with Fixed Value":
#                 if pd.api.types.is_numeric_dtype(self.df[selected_col]):
#                     value, ok = QInputDialog.getDouble(self, "Enter Fixed Value", "Enter the fixed value to fill in:")
#                     if ok:
#                         # self.df[selected_col].fillna(value, inplace=True)
#                         self.df[selected_col] = self.df[selected_col].fillna(value=value)
#                 else:
#                     category = ""
#                     category, category_ok = QInputDialog.getText(self, "Enter Fixed Value", "Enter value:", QLineEdit.Normal, category)                                    
                
#                     if category_ok:
#                         self.df[selected_col] = self.df[selected_col].fillna(value=category)

#             self.model.layoutChanged.emit()
#         else:
#             QMessageBox.information(self, "Cancel", "Operation canceled.")
            




#     def change_type(self):
#         if not self.check_file():
#             return
        

#         if not self.check_col():
#             return

#         selected_col = self.ui.cmb_data.currentText()

#         tips = ["int8", "int16", "int32", "int64", "float16", "float32", "float64", "object", "string", "category", "bool", "time", "complex64", "complex128", "complex256"]


#         tips, ok = QInputDialog.getItem(self, "Select Data Type", f"What type would you like to convert the '{selected_col}' column to?", tips, 0, False)
#         if ok and tips:
#             if tips == "int8":
#                 try:
#                     self.df[selected_col] = self.df[selected_col].astype('int8')
#                 except Exception as e:
#                     QMessageBox.critical(self, "Conversion Error", f"Please review the error:\n{e}")
#             elif tips == "int16":
#                 try:
#                     self.df[selected_col] = self.df[selected_col].astype('int16')
#                 except Exception as e:
#                     QMessageBox.critical(self, "Conversion Error", f"Please review the error:\n{e}")
#             elif tips == "int32":
#                 try:
#                     self.df[selected_col] = self.df[selected_col].astype('int32') 
#                 except Exception as e:
#                     QMessageBox.critical(self, "Conversion Error", f"Please review the error:\n{e}")
#             elif tips == "int64":
#                 try:
#                     self.df[selected_col] = self.df[selected_col].astype('int64') 
#                 except Exception as e:
#                     QMessageBox.critical(self, "Conversion Error", f"Please review the error:\n{e}")
#             elif tips == "float16":
#                 try:
#                     self.df[selected_col] = self.df[selected_col].astype('float16') 
#                 except Exception as e:
#                     QMessageBox.critical(self, "Conversion Error", f"Please review the error:\n{e}")
#             elif tips == "float32":
#                 try:
#                     self.df[selected_col] = self.df[selected_col].astype('float32') 
#                 except Exception as e:
#                     QMessageBox.critical(self, "Conversion Error", f"Please review the error:\n{e}")
#             elif tips == "float64":
#                 try:
#                     self.df[selected_col] = self.df[selected_col].astype('float64') 
#                 except Exception as e:
#                     QMessageBox.critical(self, "Conversion Error", f"Please review the error:\n{e}")
#             elif tips == "object":
#                 try:
#                     self.df[selected_col] = self.df[selected_col].astype('object')
#                 except Exception as e:
#                     QMessageBox.critical(self, "Conversion Error", f"Please review the error:\n{e}")  
#             elif tips == "string":
#                 try:
#                     self.df[selected_col] = self.df[selected_col].astype('string') 
#                 except Exception as e:
#                     QMessageBox.critical(self, "Conversion Error", f"Please review the error:\n{e}")
#             elif tips == "category":
#                 try:
#                     self.df[selected_col] = self.df[selected_col].astype('category') 
#                 except Exception as e:
#                     QMessageBox.critical(self, "Conversion Error", f"Please review the error:\n{e}")
#             elif tips == "bool":
#                 try:
#                     self.df[selected_col] = self.df[selected_col].astype('bool')
#                 except Exception as e:
#                     QMessageBox.critical(self, "Conversion Error", f"Please review the error:\n{e}")
#             elif tips == "time":
#                 try:
#                     self.df[selected_col] = pd.to_datetime(self.df[selected_col])
#                 except Exception as e:
#                     QMessageBox.critical(self, "Conversion Error", f"Please review the error:\n{e}")
#             elif tips == "complex64":
#                 try:
#                     self.df[selected_col] = self.df[selected_col].astype('complex64')
#                 except Exception as e:
#                     QMessageBox.critical(self, "Conversion Error", f"Please review the error:\n{e}")
#             elif tips == "complex128":
#                 try:
#                     self.df[selected_col] = self.df[selected_col].astype('complex128') 
#                 except Exception as e:
#                     QMessageBox.critical(self, "Conversion Error", f"Please review the error:\n{e}")
#             elif tips == "complex256":
#                 try:
#                     self.df[selected_col] = self.df[selected_col].astype('complex256')
#                 except Exception as e:
#                     QMessageBox.critical(self, "Conversion Error", f"Please review the error:\n{e}")
#         else:
#             QMessageBox.information(self, "Cancel", "Operation canceled.")

                        

#     def remove_col(self):
#         if not self.check_file():
#             return
        
#         # if not self.check_col():
#         #     return

#         selected_col = self.ui.cmb_data.currentText() # Seçilen kolonu ele alma

#         if selected_col in self.df.columns:
#             self.df.drop(selected_col, axis=1 ,inplace=True) # silme işlemi
#             self.model.layoutChanged.emit() # değişikliğin tableView'e anında etki etmesi için
#         else:
#             QMessageBox.critical(self, "Column Error", f"The selected '{selected_col}' column has been deleted previously.")



#     def outliers(self, q1 = 0.25, q3 = 0.75):
#         if not self.check_file():
#             return
        
#         if not self.check_col():
#             return              
        
#         if not self.type_control():
#             return
        
#         selected_col = self.ui.cmb_data.currentText()

#         quantile1 = self.df[selected_col].quantile(q1)
#         quantile3 = self.df[selected_col].quantile(q3)

#         iqr = quantile3 - quantile1

#         up_limit = quantile3 + 1.5 * iqr
#         low_limit = quantile1 - 1.5 * iqr
    
#         string_up_limit = str(up_limit)
#         string_low_limit = str(low_limit)

#         self.figure.clear() # başka grafik için komple temizler
#         ax = self.figure.add_subplot(111)
#         ax.clear()
#         sns.boxplot(x = self.df[selected_col], ax=ax)      
#         self.canvas.draw() 

#         result = f"Low Limit: {string_low_limit}, Up Limit: {string_up_limit}"

#         self.ui.textEdit.setText(result)
#         return low_limit, up_limit


#     def outliers_index(self):
#         if not self.check_file():
#             return
        
#         if not self.check_col():
#             return   
        
#         if not self.type_control():
#             return
        
#         selected_col = self.ui.cmb_data.currentText()        

#         low_limit, up_limit = self.outliers()
#         index = self.df.loc[(self.df[selected_col] < low_limit) | (self.df[selected_col] > up_limit), selected_col]
#         outlier_texts = [f"Index: {index}, Value: {value}" for index, value in index.items()]
#         string_index = '\n'.join(map(str, outlier_texts))
#         self.ui.textEdit.setText(string_index)



#     def change_outliers(self):
#         if not self.check_file():
#            return

#         if not self.check_col():
#             return  
        
#         if not self.type_control():
#             return
        
#         selected_col = self.ui.cmb_data.currentText()

#         low_limit, up_limit = self.outliers()
#         low_limit = int(low_limit)
#         up_limit = int(up_limit)
#         self.df.loc[(self.df[selected_col] < low_limit), selected_col] = low_limit
#         self.df.loc[(self.df[selected_col] > up_limit), selected_col] = up_limit


        
#         self.figure.clear() # başka grafik için komple temizler
#         ax = self.figure.add_subplot(111)
#         ax.clear()
#         sns.boxplot(x = self.df[selected_col], ax=ax)      
#         self.canvas.draw() 


#     def label_encoding(self):

#         if not self.check_file():
#             return
        
#         if not self.check_col():
#             return

#         selected_col = self.ui.cmb_data.currentText()

#         # if not pd.api.types.is_numeric_dtype(self.df[selected_col]):
#         le = LabelEncoder()
#         self.df[selected_col] = le.fit_transform(self.df[selected_col])
#         strting_inverse = str(le.inverse_transform([0, 1]))
#         string_label_encoding = self.df[selected_col].to_string()
#         result = f"{strting_inverse}\n{string_label_encoding}"
#         self.ui.textEdit.setText(result)
#         self.model.layoutChanged.emit() # değişikliğin tableView'e anında etki etmesi için
#         # else:
#         #     QMessageBox.critical(self, "Encoding Hatası", "Sayısal verilere label encoding işlemi uygulanamaz.\nDaha önce bu kolona encoding işlemi uygulanmış olabilir.")


#     def one_hot_encoding(self):
#         if not self.check_file():
#             return

#         if not self.check_col():
#             return
        
#         selected_col = self.ui.cmb_data.currentText()
        
#         if self.df[selected_col].nunique() < 3:
#             QMessageBox.warning(self, "Encoding Error", "One hot encoding cannot be applied to columns with fewer than two unique values.")
#             return
        
#         one_hot_encoded = pd.get_dummies(self.df[selected_col], prefix=selected_col)
#         one_hot_encoded = one_hot_encoded.astype(int)
#         self.df = pd.concat([self.df, one_hot_encoded], axis=1)
#         self.df.drop(selected_col, axis=1, inplace=True)

#         # Modeli güncelleme
#         self.model = DataFrameModel(self.df)  
#         self.ui.tableView.setModel(self.model)
        
#         self.model.layoutChanged.emit() # değişikliğin tableView'e anında etki etmesi için



#     def ordinal_encoding(self):
#         if not self.check_file():
#             return
        
#         if not self.check_col():
#             return

#         selected_col = self.ui.cmb_data.currentText()

#         if self.ui.listWidget.count() != 0:
#             categories= []
#             for index in range(self.ui.listWidget.count()):
#                 item = self.ui.listWidget.item(index)
#                 categories.append(item.text())

#             unique_values = sorted(self.df[selected_col].unique())
#             if sorted(categories) != unique_values:
#                 QMessageBox.warning(self, "Category Error", f"The categories are missing or have been entered incorrectly.")
#                 return
            
#             categories = [categories]
        
    
#             ol = OrdinalEncoder(categories=categories)
            
#             self.df[selected_col] = ol.fit_transform(self.df[[selected_col]])
#             string_ordinal_encoding = self.df[selected_col].to_string()
#             self.ui.textEdit.setText(string_ordinal_encoding)
#             self.ui.listWidget.clear()
#             self.model.layoutChanged.emit()
#         else:
#             QMessageBox.warning(self, "Category Error", f"Enter a category.")




#     def add_categories_for_ordinal(self):
#         if not self.check_file():
#             return
    

#         text = self.ui.lineEdit.text().strip()
#         if text != "":
#             # Mevcut öğeleri kontrol et
#             existing_items = self.ui.listWidget.findItems(text, Qt.MatchExactly)
            
#             if not existing_items:  # Eğer mevcut değilse
#                 self.ui.listWidget.addItem(text)
#                 self.ui.lineEdit.clear()
#             else:
#                 QMessageBox.warning(self, "Category Error", f"The category already exists: {text}")
#         else:
#             QMessageBox.warning(self, "Category Error", "Enter a category.")


#     def train_test_split(self):
#         if not self.check_file():
#             return
        
#         if not self.check_col():
#             return
        
#         if not self.type_control():
#             return
        
#         selected_col = self.ui.cmb_data.currentText()
        
#         if pd.api.types.is_numeric_dtype(self.df[selected_col]):
#                 test_size, ok = QInputDialog.getDouble(self, "Cluster Distribution", "Select the percentage of data you want to allocate to the test set:", decimals=2)
#                 if ok:
#                     x = self.df.drop(selected_col, axis=1).values
#                     y = self.df[selected_col].values
#                     self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x,y, test_size=test_size, random_state=0)

#                     result = f"x_train:\n{str( self.x_train)}\n\nx_test:\n{str(self.x_test)}\n\ny_train:\n{str(self.y_train)}\n\ny_test:\n{str(self.y_test)}"
#                     self.ui.textEdit.setText(result)
#                     return self.x_train, self.x_test, self.y_train, self.y_test
#                 else:
#                     QMessageBox.information(self, "Cancel", "Operation canceled.")
#         else:
#             QMessageBox.warning(self, "Type Error", f"The data must be numeric for the train-test split.")


#     def scaler(self):
#         if not self.check_file():
#             return

#         standart_scaler = StandardScaler()
#         min_max_scaler = MinMaxScaler()
#         normalizer = Normalizer()
#         max_abs_scaler = MaxAbsScaler()
#         robuts_scaler = RobustScaler()


#         try:
#             if all(pd.api.types.is_numeric_dtype(self.df[col]) for col in self.df.columns):
#                 scalers = ["Standart Scaler", "Min-Max Scaler", "Normalizer", "Max Abs Scaler", "Robust Scaler"]
#                 sc, ok = QInputDialog.getItem(self, "Scaling Method", "Select Scaling Method", scalers, 0, False)
#                 if ok and sc:
#                     if sc == "Standart Scaler":
#                         self.x_train = standart_scaler.fit_transform(self.x_train)
#                         self.x_test = standart_scaler.transform(self.x_test)
#                         self.y_train = standart_scaler.fit_transform(self.y_train.reshape(-1, 1)).ravel()
#                         self.y_test = standart_scaler.transform(self.y_test.reshape(-1, 1)).ravel()
#                         result = f"x_train:\n{self.x_train}\n\nx_test:\n{self.x_test}\ny_train:\n{self.y_train}\n\ny_test:\n{self.y_test}"
#                         self.ui.textEdit.setText(result)

#                     elif sc == "Min-Max Scaler":
#                         self.x_train = min_max_scaler.fit_transform(self.x_train)
#                         self.x_test = min_max_scaler.transform(self.x_test)
#                         self.y_train = min_max_scaler.fit_transform(self.y_train.reshape(-1, 1)).ravel()
#                         self.y_test = min_max_scaler.transform(self.y_test.reshape(-1, 1)).ravel()
#                         result = f"x_train:\n{self.x_train}\n\nx_test:\n{self.x_test}\ny_train:\n{self.y_train}\n\ny_test:\n{self.y_test}"
#                         self.ui.textEdit.setText(result) 

#                     elif sc == "Normalizer":
#                         self.x_train = normalizer.fit_transform(self.x_train)
#                         self.x_test = normalizer.transform(self.x_test)
#                         self.y_train = normalizer.fit_transform(self.y_train.reshape(-1, 1)).ravel()
#                         self.y_test = normalizer.transform(self.y_test.reshape(-1, 1)).ravel()
#                         result = f"x_train:\n{self.x_train}\n\nx_test:\n{self.x_test}\ny_train:\n{self.y_train}\n\ny_test:\n{self.y_test}"
#                         self.ui.textEdit.setText(result)   

#                     elif sc == "Max Abs Scaler":
#                         self.x_train = max_abs_scaler.fit_transform(self.x_train)
#                         self.x_test = max_abs_scaler.transform(self.x_test)
#                         self.y_train = max_abs_scaler.fit_transform(self.y_train.reshape(-1, 1)).ravel()
#                         self.y_test = max_abs_scaler.transform(self.y_test.reshape(-1, 1)).ravel()
#                         result = f"x_train:\n{self.x_train}\n\nx_test:\n{self.x_test}\ny_train:\n{self.y_train}\n\ny_test:\n{self.y_test}"
#                         self.ui.textEdit.setText(result)   

#                     elif sc == "Robust Scaler":
#                         self.x_train = robuts_scaler.fit_transform(self.x_train)
#                         self.x_test = robuts_scaler.transform(self.x_test)
#                         self.y_train = robuts_scaler.fit_transform(self.y_train.reshape(-1, 1)).ravel()
#                         self.y_test = robuts_scaler.transform(self.y_test.reshape(-1, 1)).ravel()
#                         result = f"x_train:\n{self.x_train}\n\nx_test:\n{self.x_test}\ny_train:\n{self.y_train}\n\ny_test:\n{self.y_test}"
#                         self.ui.textEdit.setText(result)   

#                     else:
#                         QMessageBox.information(self, "Cancel", "Operation canceled.")
#                 else:
#                     QMessageBox.information(self, "Cancel", "Operation canceled.")  
#             else:
#                 QMessageBox.warning(self, "Column Error", f"Columns must be numeric.")
#         except Exception:
#                 QMessageBox.warning(self, "Conversion Error", f"The train-test split may not have been performed. Please review the process again.")


#     def poly(self):
#         if not self.check_file():
#             return
#         try:
#             value, ok = QInputDialog.getInt(self, "Selection of the 'degree' Parameter Value", "Enter the value:")
#             if ok:
#                 poly_reg = PolynomialFeatures(degree=value)
#                 self.x_train = poly_reg.fit_transform(self.x_train)
#                 self.x_test = poly_reg.fit_transform(self.x_test)
#                 result = f"x_train_poly:\n{self.x_train}\n\nx_test_poly:\n{self.x_test}"
#                 self.ui.textEdit.setText(result)
#             else:
#                 QMessageBox.information(self, "Cancel", "Operation canceled.")
#         except Exception:
#                 QMessageBox.warning(self, "Conversion Error", f"The train-test split may not have been performed. Please review the process again.")

  
#     def r2_performance(self, predict, y_test):
#         performance = r2_score(predict, y_test)
#         mae = mean_absolute_error(y_test, predict)
#         mse = mean_squared_error(y_test, predict)
        
#         self.ui.textEdit_2.clear()
#         result = f"MAE: {"{:.10f}".format(mae)}\nMSE: {"{:.9f}".format(mse)}"
#         self.ui.textEdit_2.setText(result)

#         self.figure.clear() # başka grafik için komple temizler
#         ax = self.figure.add_subplot(111)
#         ax.clear()
#         ax.scatter(y_test, predict, color='blue')
#         ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
#         ax.set_title(f'R² Score: {performance:.6f}')
#         ax.set_xlabel('Actual')
#         ax.set_ylabel('Predicted')            
#         self.canvas.draw() 


#     def linear_model(self):
#         if not self.check_file():
#             return

#         try:
#             lin_reg = LinearRegression()
#             lin_reg.fit(self.x_train, self.y_train)
#             lin_predict = lin_reg.predict(self.x_test)
#             self.r2_performance(lin_predict, self.y_test)
#             self.ui.textEdit.setText(f"Predicted Values:\n{str(lin_predict)}")

            
#         except Exception as e:
#             self.ui.textEdit_4.setText(str(e))
#             QMessageBox.warning(self, "Model Error", f"An unexpected error has occurred. Please review the processes again.") 


#     def decision_tree_model(self):
#         if not self.check_file():
#             return
                
#         try:
#             dt_reg = DecisionTreeRegressor(random_state=0)
#             dt_reg.fit(self.x_train, self.y_train)
#             dt_predict = dt_reg.predict(self.x_test)
#             self.r2_performance(dt_predict, self.y_test)

#             self.ui.textEdit.setText(f"Predicted Values:\n{str(dt_predict)}")
#         except Exception as e:
#             self.ui.textEdit_4.setText(str(e))
#             QMessageBox.warning(self, "Model Error", f"An unexpected error has occurred. Please review the processes again.") 

    
#     def random_forest_model(self):
#         if not self.check_file():
#             return   

#         try:
#             tree_count, ok = QInputDialog.getInt(self, "Selection of the Tree Parameter Value", "Enter the number of trees:")
#             if ok:
#                 rf_reg = RandomForestRegressor(n_estimators=tree_count)
#                 rf_reg.fit(self.x_train, self.y_train)
#                 rf_predict = rf_reg.predict(self.x_test)
#                 self.r2_performance(rf_predict, self.y_test)  
#                 self.ui.textEdit.setText(f"Predicted Values:\n{str(rf_predict)}")
#             else:
#                 QMessageBox.information(self, "Cancel", "Operation canceled.")
#         except Exception as e:
#             self.ui.textEdit_4.setText(str(e))
#             QMessageBox.warning(self, "Model Error", f"An unexpected error has occurred. Please review the processes again.") 


#     def svr_kurma(self, kernel):
#         svr_reg = SVR(kernel=kernel)
#         svr_reg.fit(self.x_train, self.y_train)
#         svr_predict = svr_reg.predict(self.x_test)
#         self.r2_performance(svr_predict, self.y_test)  
#         self.ui.textEdit.setText(f"Predicted Values:\n{str(svr_predict)}")
    
#     def svr_model(self):
#         if not self.check_file():
#             return
        
#         try:
#             kernels = ["linear", "poly", "rbf", "sigmoid"]
#             kernel, svr_ok = QInputDialog.getItem(self, "Selection Kernel", "Which kernel type would you like to choose?", kernels, 0, False)

#             if svr_ok and kernel:
#                 if kernel == "linear":
#                     self.svr_kurma(kernel)
                    
#                 elif kernel == "poly":
#                     self.svr_kurma(kernel)

#                 elif kernel == "rbf":
#                     self.svr_kurma(kernel)

#                 elif kernel == "sigmoid":
#                     self.svr_kurma(kernel)
                    
#                 else:
#                     QMessageBox.information(self, "Cancel", "Operation canceled.")
#         except Exception as e:
#             self.ui.textEdit_4.setText(str(e))
#             QMessageBox.warning(self, "Model Error", f"An unexpected error has occurred. Please review the processes again.") 

#     def confusion_performance(self, predict, y_test):
#         self.ui.textEdit_2.clear()
#         performance = confusion_matrix(y_test, predict)
#         # accuracy = accuracy_score(y_test, predict, zero_division=0)
#         # precision = precision_score(y_test, predict, zero_division=0)
#         # recall = recall_score(y_test, predict, zero_division=0)
#         # f1 = f1_score(y_test, predict, average='binary')
#         # self.ui.textEdit_2.setText(f"accuracy: {"{:.10f}".format(accuracy)}\nprecision: {"{:.10f}".format(precision)}\nrecall:{"{:.10f}".format(recall)}\nf1 score: {"{:.10f}".format(f1)} ")
#         self.figure.clear()
#         ax = self.figure.add_subplot(111)
#         ax.clear()
#         sns.heatmap(performance, annot=True, fmt='d', cmap='Blues', ax=ax)
#         ax.set_xlabel('Predicted')
#         ax.set_ylabel('Actual')
#         ax.set_title('Confusion Matrix')
#         self.canvas.draw()    


#     def logistic_regression(self):
#         if not self.check_file():
#             return   
        
#         try:
#             log_reg = LogisticRegression(max_iter=1000)
#             log_reg.fit(self.x_train, self.y_train)
#             log_reg_predict = log_reg.predict(self.x_test)
#             self.confusion_performance(self.y_test, log_reg_predict)

        
#             self.ui.textEdit.setText(f"Predicted Values:\n{str(log_reg_predict)}")
#         except Exception as e:
#             self.ui.textEdit_4.setText(str(e))
#             QMessageBox.warning(self, "Model Error", f"An unexpected error has occurred. Please review the processes again.") 

#     def svc_kurma(self, kernel):
#         svc_reg = SVC(kernel=kernel)
#         svc_reg.fit(self.x_train, self.y_train)
#         svc_predict = svc_reg.predict(self.x_test)
#         self.confusion_performance(self.y_test, svc_predict)
#         self.ui.textEdit.setText(f"Predicted Values:\n{str(svc_predict)}") 

#     def svc_model(self):
#         if not self.check_file():
#             return
        
#         try:
#             kernels = ["linear", "poly", "rbf", "sigmoid"]
#             kernel, svc_ok = QInputDialog.getItem(self, "Selection Kernel", "Which kernel type would you like to choose?", kernels, 0, False)

#             if svc_ok and kernel:
#                 if kernel == "linear":
#                     self.svc_kurma(kernel)

#                     # svc_reg = SVC(kernel=kernel)
#                     # svc_reg.fit(self.x_train, self.y_train)
#                     # svc_predict = svc_reg.predict(self.x_test)
#                     # self.confusion_performance(svc_predict, self.y_test)

#                 elif kernel == "poly":
#                     self.svc_kurma(kernel)

#                 elif kernel == "rbf":
#                     self.svc_kurma(kernel)

#                 elif kernel == "sigmoid":
#                     self.svc_kurma(kernel)
                 
#                 else:
#                     QMessageBox.information(self, "Cancel", "Operation canceled.")
#         except Exception as e:
#             self.ui.textEdit_4.setText(str(e))
#             QMessageBox.warning(self, "Model Error", f"An unexpected error has occurred. Please review the processes again.") 

#     def k_neighbors_kurma(self, n_neighbors, metric):
#         knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
#         knn.fit(self.x_train, self.y_train)
#         knn_predict = knn.predict(self.x_test)
#         self.confusion_performance(self.y_test, knn_predict)
#         self.ui.textEdit.setText(f"Predicted Values:\n{str(knn_predict)}")

#     def k_neighbors_classifier(self):
#         if not self.check_file():
#             return  

#         try:
#             kume, ok = QInputDialog.getInt(self, "Selection of the Cluster Parameter Value", "Enter the number of clusters:")
#             metrics = ["manhattan", "euclidean", "minkowski", "chebyshev", "canberra", "hamming"]
#             metrics, knn_ok = QInputDialog.getItem(self, "Metric Selection", "Which metric type would you like to choose?", metrics, 0, False)
            
#             if ok:
#                 if knn_ok and metrics:
#                     if metrics == "manhattan":
#                         self.k_neighbors_kurma(kume, metrics)
#                         # knn = KNeighborsClassifier(n_neighbors=kume, metric=metrics)
#                         # knn.fit(self.x_train, self.y_train)
#                         # knn_predict = knn.predict(self.x_test)
#                         # self.confusion_performance(knn_predict, self.y_test)

#                     elif  metrics == "euclidean":
#                         self.k_neighbors_kurma(kume, metrics)

#                     elif  metrics == "minkowski":
#                         self.k_neighbors_kurma(kume, metrics)

#                     elif  metrics == "chebyshev":
#                         self.k_neighbors_kurma(kume, metrics)

#                     elif  metrics == "canberra":
#                         self.k_neighbors_kurma(kume, metrics)

#                     elif  metrics == "hamming":
#                         self.k_neighbors_kurma(kume, metrics)

#                 else:
#                     QMessageBox.information(self, "Cancel", "Operation canceled.")
#             else:
#                 QMessageBox.information(self, "Cancel", "Operation canceled.")
#         except Exception as e:
#             self.ui.textEdit_4.setText(str(e))
#             QMessageBox.warning(self, "Model Error", f"An unexpected error has occurred. Please review the processes again.") 

#     def decision_tree_classifier(self):
#         if not self.check_file():
#             return 

#         try:
#             criterions  = ["entropy", "gini"]
#             criterions , ok = QInputDialog.getItem(self, "Select Criterion", "Select the feature by which the node will be split:", criterions, 0, False)                
#             if ok and criterions:
#                 if criterions == "entropy":
#                     dtr_classifier = DecisionTreeClassifier(criterion=criterions)
#                     dtr_classifier.fit(self.x_train, self.y_train)
#                     dtr_classifier_predict = dtr_classifier.predict(self.x_test)
#                     self.confusion_performance(self.y_test, dtr_classifier_predict)

#                     self.ui.textEdit.setText(f"Predicted Values:\n{str(dtr_classifier_predict)}")                
#                 else:
#                     dtr_classifier = RandomForestClassifier(criterion=criterions)
#                     dtr_classifier.fit(self.x_train, self.y_train)
#                     dtr_classifier_predict = dtr_classifier.predict(self.x_test)
#                     self.confusion_performance(self.y_test, dtr_classifier_predict)

#                     self.ui.textEdit.setText(f"Predicted Values:\n{str(dtr_classifier_predict)}")            
#             else:
#                 QMessageBox.information(self, "Cancel", "Operation canceled.")
#         except Exception as e:
#             self.ui.textEdit_4.setText(str(e))
#             QMessageBox.warning(self, "Model Error", f"An unexpected error has occurred. Please review the processes again.") 



#     def random_forest_classifier(self):
#         if not self.check_file():
#             return 
        
#         try:
#             tree_count, ok = QInputDialog.getInt(self, "Selection of the Tree Parameter Value", "Enter the number of trees:")
#             criterions  = ["entropy", "gini"]
#             criterions , rf_ok = QInputDialog.getItem(self, "Select Criterion", "Select the feature by which the node will be split:", criterions, 0, False)
#             if ok:
#                 if rf_ok and criterions:
#                     if criterions == "entropy":
#                         rf_classifier = RandomForestClassifier(n_estimators=tree_count, criterion=criterions)
#                         rf_classifier.fit(self.x_train, self.y_train)
#                         rf_classifier_predict = rf_classifier.predict(self.x_test)
#                         self.confusion_performance(self.y_test, rf_classifier_predict)

#                         self.ui.textEdit.setText(f"Predicted Values:\n{str(rf_classifier_predict)}")
#                     else:
#                         rf_classifier = RandomForestClassifier(n_estimators=tree_count, criterion=criterions)
#                         rf_classifier.fit(self.x_train, self.y_train)
#                         rf_classifier_predict = rf_classifier.predict(self.x_test)
#                         self.confusion_performance(self.y_test, rf_classifier_predict)

#                         self.ui.textEdit.setText(f"Predicted Values:\n{str(rf_classifier_predict)}")
#                 else:
#                     QMessageBox.information(self, "Cancel", "Operation canceled.")
#             else:
#                 QMessageBox.information(self, "Cancel", "Operation canceled.")
#         except Exception as e:
#             self.ui.textEdit_4.setText(str(e))
#             QMessageBox.warning(self, "Model Error", f"An unexpected error has occurred. Please review the processes again.") 

        

#     def naive_bayes_classifier(self):
#         if not self.check_file():
#             return  

#         try:
#             nb_classifier = GaussianNB()
#             nb_classifier.fit(self.x_train, self.y_train)
#             nb_classifier_predict = nb_classifier.predict(self.x_test)
#             self.confusion_performance( self.y_test, nb_classifier_predict)

#             self.ui.textEdit.setText(f"Predicted Values:\n{str(nb_classifier_predict)}")  
#         except Exception as e:
#             self.ui.textEdit_4.setText(str(e))
#             QMessageBox.warning(self, "Model Error", f"An unexpected error has occurred. Please review the processes again.") 

#     def k_means_model(self):
#         if not self.check_file():
#             return
  
#         try:
#             numeric_columns = self.df.select_dtypes(include=[np.number])

#             tree_count, ok = QInputDialog.getInt(self, "Selection of Class Quantity", "How many clusters would you like to create up to the amount you entered?:")
#             inits  = ["k-means++", "random"]
#             inits , k_ok = QInputDialog.getItem(self, "Selection of the init Parameter", "Select Method", inits, 0, False)
            
#             if ok:
#                 if k_ok and inits:
#                     if inits == "k-means++":
#                         results = []
#                         for i in range(1, tree_count+1):
#                             k_means = KMeans(n_clusters=i, init=inits, random_state=123 ) 
#                             k_means.fit(numeric_columns.values)
#                             results.append(k_means.inertia_)
                        
#                         self.figure.clear() # başka grafik için komple temizler
#                         ax = self.figure.add_subplot(111)
#                         ax.clear()
#                         ax.plot(range(1,tree_count+1),results)
#                         ax.set_title("K-Means")
#                         ax.set_xlabel("Number of Cluster")
#                         ax.set_ylabel("WCSS")
#                         self.canvas.draw()

#                         self.ui.textEdit.setText(f"Results:\n{str(results)}")
#                     elif inits == "random++":
#                         results = []
#                         for i in range(1, tree_count+1):
#                             k_means = KMeans(n_clusters=i, init=inits, random_state=123) 
#                             k_means.fit(numeric_columns.values)
#                             results.append(k_means.inertia_)
                        
#                         ax = self.figure.add_subplot(111)
#                         ax.clear()
#                         ax.plot(range(1,tree_count+1),results)
#                         self.canvas.draw()
#                         self.ui.textEdit.setText(f"Results:\n{str(results)}")
#                 else:
#                     QMessageBox.information(self, "Cancel", "Operation canceled.")
#             else:
#                 QMessageBox.information(self, "Cancel", "Operation canceled.")
#         except Exception as e:
#             self.ui.textEdit_4.setText(str(e))
#             QMessageBox.warning(self, "Model Error", f"An unexpected error has occurred. Please review the processes again.") 

#     def hierarchical_clustring_kurma(self, n_clusters, linkage):
#         self.numeric_columns = self.df.select_dtypes(include=[np.number])
#         ac = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
#         ac_predict = ac.fit_predict(self.numeric_columns)
#         self.ui.textEdit.setText(str(ac_predict))
        
        
#         colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'brown', 'pink', 'grey', 'cyan']
#         unique_clusters = np.unique(ac_predict)

#         self.figure.clear()
#         ax = self.figure.add_subplot(111)
#         ax.clear()
#         for i, cluster in enumerate(unique_clusters):
#             ax.scatter(self.numeric_columns.iloc[ac_predict == cluster, 0].values, 
#                         self.numeric_columns.iloc[ac_predict == cluster, 1].values, 
#                         s=100, c=colors[i % len(colors)], label=f"Cluster {cluster}")
#         ax.set_title('Hierarchical Clustering')
#         self.canvas.draw()  

#     def hierarchical_clustring(self):
#         if not self.check_file():
#             return 
#         try:
#             cluster_count, ok = QInputDialog.getInt(self, "Selection of Cluster Quantity", "Select how many clusters it will consist of:")
#             linkage = ["ward", "complete", "average", "single"]
#             linkage, linkage_ok = QInputDialog.getItem(self, "Selection of Linkage Parameter", "Which linkage type would you like to choose?", linkage, 0, False)
#             if ok:
#                 if linkage_ok and ok:
#                     if linkage == "ward":
#                         self.hierarchical_clustring_kurma(cluster_count, linkage)
#                         sch.dendrogram(sch.linkage(self.numeric_columns, method=linkage))
#                         plt.gca().set_xticks([])
#                         plt.show()
#                     elif linkage == "complete":
#                         self.hierarchical_clustring_kurma(cluster_count, linkage)
#                         sch.dendrogram(sch.linkage(self.numeric_columns, method=linkage))
#                         plt.gca().set_xticks([])
#                         plt.show()
#                     elif linkage == "average":
#                         self.hierarchical_clustring_kurma(cluster_count, linkage)
#                         sch.dendrogram(sch.linkage(self.numeric_columns, method=linkage))
#                         plt.gca().set_xticks([])
#                         plt.show()
#                     elif linkage == "single":
#                         self.hierarchical_clustring_kurma(cluster_count, linkage)
#                         sch.dendrogram(sch.linkage(self.numeric_columns, method=linkage))
#                         plt.gca().set_xticks([])
#                         plt.show()
#                 else:
#                     QMessageBox.information(self, "Cancel", "Operation canceled.")
#             else:
#                 QMessageBox.information(self, "Cancel", "Operation canceled.")
#         except Exception as e:
#             self.ui.textEdit_4.setText(str(e))
#             QMessageBox.warning(self, "Model Error", f"An unexpected error has occurred. Please review the processes again.") 

# def window():
#     app = QApplication(sys.argv) # Uygulama oluşturma
#     win = main() # Pencere Oluşturma. MyWindow class'ı QMainWindow'u parametre olarak aldı.
#     win.showMaximized() # Başlangıçta tam ekranda çalıştırma
#     win.show() # Ekranda Gösterme
#     sys.exit(app.exec_()) # Uygulamayı durdurma(Çarpı ikonuna basıldığında duracak.)
# window()