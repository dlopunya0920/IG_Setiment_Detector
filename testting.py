import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

model_fraud = pickle.load(open('model_fraud_no_chi2.sav','rb'))

tfidf = TfidfVectorizer

loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open("feature_tf-idf.sav", "rb"))))
st.set_page_config(
    page_title="Instagram Sentiment Detector",
    page_icon=":smiley:",
    
)
import requests
from io import BytesIO

from PIL import Image
instagram_logo_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/95/Instagram_logo_2022.svg/2048px-Instagram_logo_2022.svg.png'
response = requests.get(instagram_logo_url)
if response.status_code == 200:
    instagram_logo = Image.open(BytesIO(response.content))

    # Resize the image to a smaller size
    smaller_logo = instagram_logo.resize((100, 100))  # Adjust the size as needed

    # Display the resized logo
    st.image(smaller_logo)
else:
    st.write("Gagal mengunduh logo Instagram")

def main():
    st.title("Instagram Sentiment Detector")

   
    message = st.text_area("Masukan")

    st.write("<style>.stButton {display: flex; justify-content: center;}</style>", unsafe_allow_html=True)

    if st.button("Deteksi Sentimen Masukan"):
        predict_fraud = model_fraud.predict(loaded_vec.fit_transform([message]))
    
        if (predict_fraud == 0):
            fraud_detection = 'Komentar positif'
        elif (predict_fraud == 1):
            fraud_detection = 'Komentar Negatif, get a life'
        else :
            fraud_detection = 'inputanmu salah'
        st.success(fraud_detection)

if __name__ == "__main__":
    main() 