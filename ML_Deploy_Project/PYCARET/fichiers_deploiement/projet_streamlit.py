import pycaret
from pycaret.classification import load_model, predict_model
import joblib
from joblib import load
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
import missingno
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


#loading of model
model = load_model('ChurnModel')


#define the predict function
def predict(model, input_df ):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions



from PIL import Image
image = Image.open('logo.png')
streamlit_image = Image.open('streamlit.png')
image_telecoms = Image.open('telecoms.jpg')

col1, col2 = st.columns(2)
with col1:
   st.image(image,use_column_width=True)
with col2:
   st.image(streamlit_image,use_column_width=False)

   # ajout du sidebar (une sidebar est une colonne ( : bar) placée sur la droite ou la gauche de la page principale)
   st.sidebar.title("NAVIGATION BAR")
   add_selectbox = st.sidebar.selectbox(
       "How would you like to predict?",
       ("Online", "Batch"))


    #for online predictions
if add_selectbox == 'Online':

    st.sidebar.info('This app is created to predict if a telephonic subscriber will leave the operator or not')
    # st.sidebar.success('https://www.pycaret.org')
    st.sidebar.image(image_telecoms)
    st.sidebar.write("\n")
    st.sidebar.write("\n")
    st.sidebar.info("CONTACT")
    url = "https://lothairebazie1999.wixsite.com/my-siteweb-portfolio"
    st.sidebar.write("SITE WEB PORTFOLIO : ", url)
    url2 = "https://www.linkedin.com/in/dinin-ren%C3%A9-lothaire-bazie-83b5b01a2/"
    st.sidebar.write("LinkedIn : ", url2)
    st.title("Predicting subscribers churn")

    st.write("\n")
    st.subheader("LINE PREDICTION / PREDICTION EN LIGNE")
    st.success("Les scénarios de prédiction en ligne sont pour les cas où vous voulez générer des prédictions sur une base individuelle")


    #insertion de toutes les entrées requises pour la prédiction avec streamlit
    state = st.number_input('state')
    account_length = st.number_input('account_length')
    area_code = st.number_input('area_code')
    international_plan = st.number_input('international_plan')
    voice_mail_plan = st.number_input('voice_mail_plan')
    number_vmail_messages = st.number_input('number_vmail_messages')
    total_day_minutes = st.number_input('total_day_minutes')
    total_day_calls = st.number_input('total_day_calls')
    total_day_charge = st.number_input('total_day_charge')
    total_eve_minutes = st.number_input('total_eve_minutes')
    total_eve_calls = st.number_input('total_eve_calls')
    total_eve_charge = st.number_input('total_eve_charge')
    total_night_minutes = st.number_input('total_night_minutes')
    total_night_calls = st.number_input('total_night_calls')
    total_night_charge = st.number_input('total_night_charge')
    total_intl_minutes = st.number_input('total_intl_minutes')
    total_intl_calls = st.number_input('total_intl_calls')
    total_intl_charge = st.number_input('total_intl_charge')
    customer_service_calls = st.number_input('customer_service_calls')

    # salary = st.selectbox('salary', ['low', 'high','medium'])
    output=""
    input_dict={'state':state,'account_length':account_length,'area_code':area_code,'international_plan':international_plan,'voice_mail_plan': voice_mail_plan,'number_vmail_messages':number_vmail_messages,'total_day_minutes' : total_day_minutes, 'total_day_calls': total_day_calls, 'total_day_charge': total_day_charge, 'total_eve_minutes': total_eve_minutes, 'total_eve_calls': total_eve_calls, 'total_eve_charge': total_eve_charge, 'total_night_minutes': total_night_minutes, 'total_night_calls': total_night_calls, 'total_night_charge': total_night_charge, 'total_intl_minutes': total_intl_minutes, 'total_intl_calls': total_intl_calls, 'total_intl_charge': total_intl_charge, 'customer_service_calls': customer_service_calls}
    input_df = pd.DataFrame([input_dict])

    #appel de la fonction predict quand le bouton est cliqué
    st.write("\n")
    st.write("\n")
    if st.button("Predict"):
        output = predict(model=model, input_df=input_df)
        output = str(output)
    # st.success('The output is {}'.format(output))
        if output == "0" :
            st.success('The output is {}'.format(output))
            st.info("cet abonné présente des FAIBLES chances de se désabonner")
        else :
            st.success('The output is {}'.format(output))
            st.info("cet abonné présente des FORTES chances de se désabonner")


########################################################################################


    # for online predictions
if add_selectbox == 'Batch':
    image_telecoms2 = Image.open('telecom2.png')
    st.sidebar.image(image_telecoms2)
    st.sidebar.title("PROCESSING")

    add_selectbox2 = st.sidebar.selectbox("Datasciences",("Choose...","INFORMATIONS ON DATASET",'VISUALISATION' ,'DEPLOYMENT OF MODEL'))

    if add_selectbox2 == 'Choose...':
        st.subheader("BATCH PREDICTION / PREDICTION PAR LOT")
        st.success("Cette partie traite de la seconde fonctionnalité à savoir la prédiction par batch."
                 " Nous avons utilisé le widget file_uploader de streamlit pour télécharger un fichier csv,"
                 " puis appelé la fonction native predict_model() de PyCaret pour générer des prédictions qui sont affichées avec la fonction write() de streamlit.")
        lot_image = Image.open('lots.png')
        st.image(lot_image)
        st.success(
            "La prédiction par lots est utile lorsque vous souhaitez générer des prédictions pour un ensemble d'observations à la fois,"
            "puis entreprendre une action sur un certain pourcentage ou nombre d'observations.")

    if add_selectbox2 == 'INFORMATIONS ON DATASET':

        st.header('1. LOADING OF CSV FILE🔧')
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        # uploading of csv files for processing
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            st.dataframe(data)
            # predictions = predict_model(estimator=model,data=data)
            # st.write(predictions)

        st.subheader('DIMENSION OF DATASET')
        if st.button("Shape"):
            st.write(data.shape)

        st.subheader('SOME INFOS ON DATASET')
        if st.button("INFO"):
            st.write(data.info)

        st.subheader('STATISTICS OF DATASET')
        if st.button("Description"):
            st.write(data.describe())

        st.subheader('POSSIBLE Nan VALUES')
        if st.button("Null"):
            # m = missingno.matrix(data)
            val = data.isnull().sum()
            st.write(val)
            # st.pyplot(m)


    if add_selectbox2 == 'VISUALISATION':

        st.header('1. LOADING OF CSV FILE🔧')
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        # uploading of csv files for processing
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            st.dataframe(data)
            # predictions = predict_model(estimator=model,data=data)
            # st.write(predictions)


        st.subheader("VISUALIZATIONS")

        option = ["Line diagram", "scatter", "Histogram", "Box"]
        chart = st.radio("please, choose un chart", option, key=99)

        # Hide some warnings
        st.set_option('deprecation.showPyplotGlobalUse', False)

        if chart == "Line diagram":
            var_x = st.selectbox("Select the column on abscissa", data.columns, key=1)
            var_y = st.selectbox("Select the column on ordonate", data.columns, key=2)
            st.bar_chart(data=data, x=var_x, y=var_y)
            st.line_chart(data=data, x=var_x, y=var_y)

        if chart == "scatter":

            # As you must know, the scatter is good to correlations
            # that's why we have to fields to select columns
            x = st.selectbox("Select the X axis", data.columns)
            y = st.selectbox("Select the Y axis", data.columns)

            if x and y:
                # when the columns have values we prepare
                # the values and put on the chart
                x = data[x].values
                x = x.reshape(-1, 1)
                y = data[y].values
                y = y.reshape(-1, 1)
                plt.scatter(x, y)
                st.pyplot()

        if chart == 'Histogram':
            # if choose this, a column list is showed
            column = st.selectbox("Select a column", data.columns)
            # After the user select one, the chart is displayed
            plt.hist(data[column].values)
            st.pyplot()

        if chart == 'Box':
            # Field to select multiple columns
            columns = st.multiselect("Select the columns", data.columns)
            if len(columns) > 0:
                # Is there any column selected? Then the chart is plotted
                data.boxplot(column=columns)
                st.pyplot()

    if add_selectbox2 == 'DEPLOYMENT OF MODEL':

        st.header('1. LOADING OF CSV FILE🔧')
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        # uploading of csv files for processing
        if file_upload is not None:
            data = pd.read_csv(file_upload)
        st.dataframe(data)
        st.write("\n")
        st.write("\n")

        X_train = data.drop(["churn", "phone number"], axis=1)
        # st.write(X_train)
        X_train.columns = ['state', 'account_length', 'area_code', 'international_plan', 'voice_mail_plan',
                           'number_vmail_messages', 'total_day_minutes', 'total_day_calls', 'total_day_charge',
                           'total_eve_minutes', 'total_eve_calls', 'total_eve_charge', 'total_night_minutes',
                           'total_night_calls', 'total_night_charge', 'total_intl_minutes', 'total_intl_calls',
                           'total_intl_charge', 'customer_service_calls']


        st.header('2. LOADING OF PKL FILE🔧')
        file_upload2 = st.file_uploader("Upload pkl file for predictions", type=None)
        if file_upload2 is not None:
            modele = joblib.load(file_upload2)

            st.write("\n")
            st.write("\n")
            st.write("\n")

        st.header('3. PREDICTIONS🔧')
        if st.button("Predict"):
            predictions = predict_model(estimator=modele, data=X_train)
            st.success("done !")
            st.write(predictions)

            # predictions = predict_model(estimator=model, data=X_train)
            # st.write(predictions)
            # if st.button("Predict"):
            #     output = ""
            #     output = predict(model=model, input_df=X_train)
            #     output = str(output)
            #     # st.success('The output is {}'.format(output))
            #     for i in range(0,X_train.shape[0]+1):
            #         if output[i] == "0":
            #             st.success('The output is {}'.format(output))
            #             st.info("cet abonné présente des FAIBLES chances de se désabonner")
            #         else:
            #             st.success('The output is {}'.format(output))
            #             st.info("cet abonné présente des FORTES chances de se désabonner")
# FIN