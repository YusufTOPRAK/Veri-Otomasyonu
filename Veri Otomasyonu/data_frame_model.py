# import pandas as pd
# from PyQt5.QtWidgets import QApplication, QMainWindow, QTableView
# from PyQt5.QtCore import QAbstractTableModel, Qt
# from PyQt5 import QtCore

# class DataFrameModel(QtCore.QAbstractTableModel):
#     def __init__(self, df=pd.DataFrame(), parent=None):
#         super().__init__(parent)
#         self._data = df

#     def rowCount(self, parent=None):
#         return self._data.shape[0]

#     def columnCount(self, parent=None):
#         return self._data.shape[1]

#     def data(self, index, role=Qt.DisplayRole):
#         if not index.isValid():
#             return None
#         if role == Qt.DisplayRole:
#             return str(self._data.iloc[index.row(), index.column()])
#         return None

#     def headerData(self, section, orientation, role=Qt.DisplayRole):
#         if role == Qt.DisplayRole:
#             if orientation == Qt.Horizontal:
#                 return str(self._data.columns[section])
#             elif orientation == Qt.Vertical:
#                 return str(self._data.index[section])
#         return None
    
#     def get_dataframe(self):
#         return self._data
