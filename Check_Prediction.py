from tkinter import *



######################
def Train():
    """GUI"""
    import tkinter as tk
    from tkinter import ttk
    import numpy as np
    import pandas as pd

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import LabelEncoder

    root = tk.Tk()

    root.geometry("1200x1000")
    root.title("Crop Prediction")
    root.configure(background="Burlywood")
    
    
#N - ratio of Nitrogen content in soil
# P - ratio of Phosphorous content in soil
# K - ratio of Potassium content in soil
# temperature - temperature in degree Celsius
# humidity - relative humidity in %
# ph - ph value of the soil
# rainfall - rainfall in mm

    Season = IntVar()
    Area = StringVar()
    PH = DoubleVar()
    Rainfall = DoubleVar()
    avg_temp= DoubleVar()
    soil_type= StringVar()
    

    # ===================================================================================================================

    def Detect():
        e1 = Season.get()
        print(e1)
        e2 = Area.get()
        print(e2)
        e3 = Rainfall.get()
        print(e3)
        e4 = avg_temp.get()
        print(e4)
        e5 = PH.get()
        print(e5)
        e6 = soil_type.get()
        print(e6)
        #########################################################################################

        from joblib import dump, load
        a1 = load("D:/Project/Crop Yield Prediction/DT_MODEL.joblib")
        v = a1.predict([[e1, e2, e3, e4, e5, e6]])
        print(v)
        if v[0]== 0:
            print("0")
            yes = tk.Label(root, text="Recommended Crop: RICE" + '\n' + str(v),
                       background="Chocolate", foreground="white", font=('times', 20, ' bold '), width=30)
            yes.place(x=400, y=570)

        elif v[0]== 1:
            print("1")
            no = tk.Label(root, text="Recommended Crop: BANANA", background="Chocolate", foreground="white",font=('times', 20, ' bold '),width=30)
            no.place(x=400, y=570)
            
         
        elif v[0]==2:
            print("2")
            no = tk.Label(root, text="Recommended Crop: COCONUT", background="Chocolate", foreground="white",font=('times', 20, ' bold '),width=30)
            no.place(x=400, y=570)   
            
        
        elif v[0]==3:
            print("3")
            no = tk.Label(root, text="Recommended Crop: SUGARCANE", background="Chocolate", foreground="white",font=('times', 20, ' bold '),width=30)
            no.place(x=400, y=570)    
            
        elif v[0]== 4:
              print("4")
              no = tk.Label(root, text="Recommended Crop: MAIZE", background="Chocolate", foreground="white",font=('times', 20, ' bold '),width=30)
              no.place(x=400, y=570)    
         
        elif v[0]== 5:
              print("5")
              no = tk.Label(root, text="Recommended Crop: GROUNDNUT", background="Chocolate", foreground="white",font=('times', 20, ' bold '),width=30)
              no.place(x=400, y=570)   
            
        elif v[0]== 6:
              print("6")
              no = tk.Label(root, text="Recommended Crop: JOWAR", background="Chocolate", foreground="white",font=('times', 20, ' bold '),width=30)
              no.place(x=400, y=570)     
              
        elif v[0]== 7:
              print("7")
              no = tk.Label(root, text="Recommended Crop: ONION", background="Chocolate", foreground="white",font=('times', 20, ' bold '),width=30)
              no.place(x=400, y=570) 

        elif v[0]== 8:
               print("8")
               no = tk.Label(root, text="Recommended Crop: POTATO", background="Chocolate", foreground="white",font=('times', 20, ' bold '),width=30)
               no.place(x=400, y=570)    
              
        elif v[0]== 9:
               print("9")
               no = tk.Label(root, text="Recommended Crop: WHEAT", background="Chocolate", foreground="white",font=('times', 20, ' bold '),width=30)
               no.place(x=400, y=570)
               
        elif v[0]== 10:
               print("10")
               no = tk.Label(root, text="Recommended Crop: BAJRA", background="Chocolate", foreground="white",font=('times', 20, ' bold '),width=30)
               no.place(x=400, y=570)
               
    

    



    l3 = tk.Label(root, text="Season", background="Bisque", font=(
        'times', 20, ' bold '), width=15)
    l3.place(x=150, y=250)
    tk.Radiobutton(root, text="Kharif", padx=5, width=10, bg="snow", font=("bold", 15), variable=Season, value=1).place(x=500,
                                                                                                                    y=250)
    tk.Radiobutton(root, text="Rabi", padx=5, width=10, bg="snow", font=("bold", 15), variable=Season, value=2).place(
        x=680, y=250)
    
    tk.Radiobutton(root, text="Whole Year", padx=10, width=10, bg="snow", font=("bold", 15), variable=Season, value=3).place(
        x=850, y=250)
    
    tk.Radiobutton(root, text="Autumn", padx=10, width=10, bg="snow", font=("bold", 15), variable=Season, value=4).place(
        x=1020, y=250)

    l4 = tk.Label(root, text="Area", background="Bisque",
                  font=('times', 20, ' bold '), width=15)
    l4.place(x=150, y=300)
    land = tk.Entry(root, bd=2, width=5, font=(
        "TkDefaultFont", 20), textvar=Area)
    land.place(x=500, y=300)
    
    l5 = tk.Label(root, text="Avg Temperature", background="Bisque",
                  font=('times', 20, ' bold '), width=15)
    l5.place(x=150, y=350)
    land = tk.Entry(root, bd=2, width=5, font=(
        "TkDefaultFont", 20), textvar=avg_temp)
    land.place(x=500, y=350)

    l6 = tk.Label(root, text="Rainfall", background="Bisque",
                  font=('times', 20, ' bold '), width=15)
    l6.place(x=150, y=400)
    urgent = tk.Entry(root, bd=2, width=5, font=(
        "TkDefaultFont", 20), textvar=Rainfall)
    urgent.place(x=500, y=400)
    

    l7 = tk.Label(root, text="PH", background="Bisque",
                  font=('times', 20, ' bold '), width=15)
    l7.place(x=150, y=450)
    hot = tk.Entry(root, bd=2, width=5, font=(
        "TkDefaultFont", 20), textvar=PH)
    hot.place(x=500, y=450)
    
    l8 = tk.Label(root, text="Soil Type", background="Bisque",
                  font=('times', 20, ' bold '), width=15)
    l8.place(x=150, y=500)
    tk.Radiobutton(root, text="Sandy soil", padx=5, width=10, bg="snow", font=("bold", 15), variable=soil_type, value=1).place(x=500,
                                                                                                                    y=500)
    tk.Radiobutton(root, text="Clayey soil", padx=5, width=10, bg="snow", font=("bold", 15), variable=soil_type, value=2).place(
        x=700, y=500)
    
    tk.Radiobutton(root, text="Loamy soil", padx=10, width=10, bg="snow", font=("bold", 15), variable=soil_type, value=3).place(
        x=900, y=500)


  

    button1 = tk.Button(root, text="Submit", command=Detect,
                        font=('times', 20, ' bold '), width=10)
    button1.place(x=500, y=650)

    root.mainloop()


Train()
