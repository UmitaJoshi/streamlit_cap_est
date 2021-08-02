import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score 
import scipy.stats as stats

@st.cache()
def prediction(Test_time, chemistry, cell):

    Test_time = Test_time
   
    # Making predictions
    if chemistry == 'NMC':
        #if cell == '0.5_a':
        #    df = pd.read_csv(r"/home/guest/users/Umita/NMC/time_cycle_summary_0-100_25C_0.5_0.5_a.csv")
        if cell == '0.5_b':
            df = pd.read_csv(r"/home/guest/users/Umita/NMC/time_cycle_summary_0-100_25C_0.5_0.5_b.csv")
        elif cell == '1_a':
            df = pd.read_csv(r"/home/guest/users/Umita/NMC/time_cycle_summary_0-100_25C_0.5_1_a.csv")
        elif cell == '1_b':
            df = pd.read_csv(r"/home/guest/users/Umita/NMC/time_cycle_summary_0-100_25C_0.5_1_b.csv")
        elif cell == '1_c':
            df = pd.read_csv(r"/home/guest/users/Umita/NMC/time_cycle_summary_0-100_25C_0.5_1_c.csv")
        elif cell == '1_d':
            df = pd.read_csv(r"/home/guest/users/Umita/NMC/time_cycle_summary_0-100_25C_0.5_1_d.csv")
        elif cell == '2_a':
            df = pd.read_csv(r"/home/guest/users/Umita/NMC/time_cycle_summary_0-100_25C_0.5_2_a.csv")
        elif cell == '2_b':
            df = pd.read_csv(r"/home/guest/users/Umita/NMC/time_cycle_summary_0-100_25C_0.5_2_b.csv")
        elif cell == '3_a':
            df = pd.read_csv(r"/home/guest/users/Umita/NMC/time_cycle_summary_0-100_25C_0.5_3_a.csv")
        elif cell == '3_b':
            df = pd.read_csv(r"/home/guest/users/Umita/NMC/time_cycle_summary_0-100_25C_0.5_3_b.csv")
        elif cell == '3_c':
            df = pd.read_csv(r"/home/guest/users/Umita/NMC/time_cycle_summary_0-100_25C_0.5_3_c.csv")
        elif cell == '3_d':
            df = pd.read_csv(r"/home/guest/users/Umita/NMC/time_cycle_summary_0-100_25C_0.5_3_d.csv")
    else:
        if cell == '0.5_a':
            df = pd.read_csv(r"/home/guest/users/Umita/LFP/time_cycle_summary_0-100_25C_0.5_0.5_a.csv")
        elif cell == '1_a':
            df = pd.read_csv(r"/home/guest/users/Umita/LFP/time_cycle_summary_0-100_25C_0.5_1_a.csv")
        elif cell == '1_b':
            df = pd.read_csv(r"/home/guest/users/Umita/LFP/time_cycle_summary_0-100_25C_0.5_1_b.csv")
        elif cell == '1_c':
            df = pd.read_csv(r"/home/guest/users/Umita/LFP/time_cycle_summary_0-100_25C_0.5_1_c.csv")
        elif cell == '1_d':
            df = pd.read_csv(r"/home/guest/users/Umita/LFP/time_cycle_summary_0-100_25C_0.5_1_d.csv")
        elif cell == '2_a':    
            df = pd.read_csv(r"/home/guest/users/Umita/LFP/time_cycle_summary_0-100_25C_0.5_2_a.csv")
        elif cell == '2_b':
            df = pd.read_csv(r"/home/guest/users/Umita/LFP/time_cycle_summary_0-100_25C_0.5_2_b.csv")
        elif cell == '3_a':
            df = pd.read_csv(r"/home/guest/users/Umita/LFP/time_cycle_summary_0-100_25C_0.5_3_a.csv")
        elif cell == '3_b':
            df = pd.read_csv(r"/home/guest/users/Umita/LFP/time_cycle_summary_0-100_25C_0.5_3_b.csv")
        elif cell == '3_c':
            df = pd.read_csv(r"/home/guest/users/Umita/LFP/time_cycle_summary_0-100_25C_0.5_3_c.csv")
        elif cell == '3_d':
            df = pd.read_csv(r"/home/guest/users/Umita/LFP/time_cycle_summary_0-100_25C_0.5_3_d.csv")

    df_final1=df.iloc[:65,:]
    #Linear regression model
    X_train = df_final1[['Test_Time (s)']] #independent variable array
    y_train = df_final1['Discharge_Capacity (Ah)']
    df_final2= df.iloc[66:,:]
    X_test= df_final2[['Test_Time (s)']]
    y_test= df_final2['Discharge_Capacity (Ah)']
    regressor = LinearRegression()
    regressor.fit(X_train,y_train)
    prediction = regressor.predict([[Test_time]])
    return prediction


# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Capacity estimation</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    chemistry = st.selectbox('Chemistry',("NMC", "LFP"))
    
    if chemistry == 'NMC':
        cell = st.selectbox('Cell',("0.5_b", "1_a", "1_b", "1_c", "1_d", "2_a", "2_b", "3_a", "3_b", "3_c", "3_d"))
        html_temp1 = """
        <h4 style ="color:black;text-align:center;">Initial 65 cycles used for Training  & Testing is done from 66th cycle to EOL</h4>
        </div>
        """
        st.markdown(html_temp1, unsafe_allow_html = True)

        
        

        if cell == '0.5_b':
            df11 = pd.read_csv(r"/home/guest/users/Umita/NMC/time_cycle_summary_0-100_25C_0.5_0.5_b.csv")
            
        elif cell == '1_a':
            df11 = pd.read_csv(r"/home/guest/users/Umita/NMC/time_cycle_summary_0-100_25C_0.5_1_a.csv")
            


        elif cell == '1_b':
            df11 = pd.read_csv(r"/home/guest/users/Umita/NMC/time_cycle_summary_0-100_25C_0.5_1_b.csv")
            

            
        elif cell == '1_c':
            df11 = pd.read_csv(r"/home/guest/users/Umita/NMC/time_cycle_summary_0-100_25C_0.5_1_c.csv")
            


        elif cell == '1_d':
            df11 = pd.read_csv(r"/home/guest/users/Umita/NMC/time_cycle_summary_0-100_25C_0.5_1_d.csv")
            


        elif cell == '2_a':
            df11 = pd.read_csv(r"/home/guest/users/Umita/NMC/time_cycle_summary_0-100_25C_0.5_2_a.csv")


        elif cell == '2_b':
            df11 = pd.read_csv(r"/home/guest/users/Umita/NMC/time_cycle_summary_0-100_25C_0.5_2_b.csv")
            


        elif cell == '3_a':
            df11 = pd.read_csv(r"/home/guest/users/Umita/NMC/time_cycle_summary_0-100_25C_0.5_3_a.csv")
            


        elif cell == '3_b':
            df11 = pd.read_csv(r"/home/guest/users/Umita/NMC/time_cycle_summary_0-100_25C_0.5_3_b.csv")
            


        elif cell == '3_c':
            df11 = pd.read_csv(r"/home/guest/users/Umita/NMC/time_cycle_summary_0-100_25C_0.5_3_c.csv")
            


        elif cell == '3_d':
            df11 = pd.read_csv(r"/home/guest/users/Umita/NMC/time_cycle_summary_0-100_25C_0.5_3_d.csv")
            
	


    elif chemistry == 'LFP':
        cell = st.selectbox('Cell',("0.5_a", "1_a", "1_b", "1_c", "1_d", "2_a", "2_b", "3_a", "3_b", "3_c", "3_d"))
        html_temp2 = """
        <h4 style ="color:black;text-align:center;">Initial 70 cycles used for Training & Testing is done from 71st cycle to EOL </h4>
        </div>
        """
        st.markdown(html_temp2, unsafe_allow_html = True)

        if cell == '0.5_a':
            df11 = pd.read_csv(r"/home/guest/users/Umita/LFP/time_cycle_summary_0-100_25C_0.5_0.5_a.csv")
            


        elif cell == '1_a':
            df11 = pd.read_csv(r"/home/guest/users/Umita/LFP/time_cycle_summary_0-100_25C_0.5_1_a.csv")
            


        elif cell == '1_b':
            df11 = pd.read_csv(r"/home/guest/users/Umita/LFP/time_cycle_summary_0-100_25C_0.5_1_b.csv")
            


        elif cell == '1_c':
            df11 = pd.read_csv(r"/home/guest/users/Umita/LFP/time_cycle_summary_0-100_25C_0.5_1_c.csv")
            


        elif cell == '1_d':
            df11 = pd.read_csv(r"/home/guest/users/Umita/LFP/time_cycle_summary_0-100_25C_0.5_1_d.csv")
            


        elif cell == '2_a':
            df11 = pd.read_csv(r"/home/guest/users/Umita/LFP/time_cycle_summary_0-100_25C_0.5_2_a.csv")
            


        elif cell == '2_b':
            df11 = pd.read_csv(r"/home/guest/users/Umita/LFP/time_cycle_summary_0-100_25C_0.5_2_b.csv")
            


        elif cell == '3_a':
            df11 = pd.read_csv(r"/home/guest/users/Umita/LFP/time_cycle_summary_0-100_25C_0.5_3_a.csv")
            


        elif cell == '3_b':
            df11 = pd.read_csv(r"/home/guest/users/Umita/LFP/time_cycle_summary_0-100_25C_0.5_3_b.csv")
            


        elif cell == '3_c':
            df11 = pd.read_csv(r"/home/guest/users/Umita/LFP/time_cycle_summary_0-100_25C_0.5_3_c.csv")
            


        elif cell == '3_d':
            df11 = pd.read_csv(r"/home/guest/users/Umita/LFP/time_cycle_summary_0-100_25C_0.5_3_d.csv")
            
		
    Q1 = df11.quantile(q=.25)
    Q3 = df11.quantile(q=.75)
    IQR = df11.apply(stats.iqr)

    #only keep rows in dataframe that have values within 1.5*IQR of Q1 and Q3
    df = df11[~((df11 < (Q1-1.5*IQR)) | (df11 > (Q3+1.5*IQR))).any(axis=1)]

    df_final1=df.iloc[:65,:]
    #Linear regression model
    X_train = df_final1[['Test_Time (s)']] #independent variable array
    y_train = df_final1['Discharge_Capacity (Ah)']
    x_t1 = df_final1[['Cycle_Index']]
    df_final2= df.iloc[65:,:]
    X_test= df_final2[['Test_Time (s)']]
    y_test= df_final2['Discharge_Capacity (Ah)']
    x_t= df_final2[['Cycle_Index']]
    #regressor = LinearRegression()
    lin_reg= linear_model.LinearRegression()
    lin_reg.fit(X_train,y_train)
    fig, ax = plt.subplots()
            
    ax.plot(x_t1, y_train, color='red',label="Actual")
    ax.plot(x_t, y_test, color='red')
    ax.plot(x_t, lin_reg.predict(X_test), color='blue',label="Predicted")
    ax.legend()
    ax.set_ylabel('Discharge_Capacity(Ah)')
    ax.set_xlabel('Cycle')
    st.pyplot(fig)

    fig, ax = plt.subplots()

    ax.plot(X_train, y_train, color='red',label="Actual")
    ax.plot(X_test, y_test, color='red')
    ax.plot(X_test, lin_reg.predict(X_test), color='blue',label="Predicted")
    ax.legend()
    ax.set_ylabel('Discharge_Capacity(Ah)')
    ax.set_xlabel('Time (sec)')
    st.pyplot(fig)

            



    Test_time = st.number_input("Time taken to discharge in seconds")
    result =""

    # when 'Predict' is clicked, make the prediction and store it
    if st.button("Predict"):
        result = prediction(Test_time, chemistry, cell)
        result = float(result)
       # result = round(result,3)
        st.success('Discharge Capacity is {} Ah'.format(result))


if __name__=='__main__':
    main()







