# Import the necessary libraries
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer , PorterStemmer
import string
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
# Load your trained model
with open('modelchatbot.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
model_file.close()

# Load your encoder model
with open('encoderchatbot.pkl', 'rb') as model_file:
    model_le = pickle.load(model_file)
model_file.close()

with open('vect.pkl', 'rb') as model_file:
    vect = pickle.load(model_file)
model_file.close()

# Load the text file
def load_data():
    with open('nplchatbot.txt', 'r', encoding='utf-8') as f:
        data = f.read().replace('\n', ' ')
    return data

# Load your trained model
with open('modelchatbot.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
model_file.close()

# Load your encoder model
with open('encoderchatbot.pkl', 'rb') as model_file:
    model_le = pickle.load(model_file)
model_file.close()


def clean_sentence(sentence):
    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    # Remove punctuation and numbers
    sentence = sentence.translate(str.maketrans("", "", string.punctuation + string.digits))
    # Convert to lowercase
    sentence = sentence.lower()
    # Tokenize the sentence
    tokens = word_tokenize(sentence)
    # Remove stopwords
    tokens = [token for token in tokens if token not in stopwords.words("english")]
    # Lemmatize the tokens
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    # Join the lemmas back into a sentence
    sentence = " ".join(lemmas)
    return sentence


# Preprocess the data
def preprocess_data(data):
    # Split the data into sentences
    sentences = sent_tokenize(data)
    # Apply the clean_sentence function to each sentence
    clean_sentences = [clean_sentence(sentence) for sentence in sentences]
    return sentences, clean_sentences


# Create a chatbot
def create_chatbot(sentences, clean_sentences):
    # Initialize the vectorizer
    vectorizer = TfidfVectorizer()
    # Fit and transform the clean sentences
    tfidf_matrix = vectorizer.fit_transform(clean_sentences)
    # Define a function to generate a response
    def generate_response(query):
        # Clean and vectorize the query
        query = clean_sentence(query)
        query_vector = vectorizer.transform([query])
        # Compute the cosine similarity between the query and the sentences
        similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
        # Find the index of the most similar sentence
        max_index = similarity_scores.argmax()
        # Return the corresponding sentence
        response = sentences[max_index]
        return response
    return generate_response


# main program: Create a streamlit interface
def main():
    # Image URL or local file path
    image_url = "https://avatars.githubusercontent.com/u/91251811?s=280&v=4"

    image_width_percentage = 30

    # Display the image with the specified width as a percentage
    st.markdown(f'<img src="{image_url}" alt="Your Image Caption" width="{image_width_percentage}%">',
                unsafe_allow_html=True)

    # Set the title
    st.title("Hello. I'm SMARTBOT. How can I help you today?")
    # Ask the user to upload a text file
    # Load and preprocess the data
    data = load_data()
    sentences, clean_sentences = preprocess_data(data)
    # Create the chatbot
    chatbot = create_chatbot(sentences, clean_sentences)
    # Ask the user to enter a query
    query = st.text_input("Please enter your query")
    # If a query is entered
    if query:
        text_to_predict = [str(query)]
        text_to_predict_dtm = vect.transform(text_to_predict)
        y_predict = model.predict(text_to_predict_dtm)
        result = model_le.inverse_transform(y_predict)

        # Display the predicted label using st.write
        st.write("Your request is classified under:", str(result[0]))

        # Generate and display the response
        response = chatbot(query)
        st.write("Smartbot:", response)
    else:
         # Display a message
        st.write("Waiting for your query...")

# Run the main
if __name__ == "__main__":
    main()