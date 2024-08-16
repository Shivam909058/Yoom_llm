import os
import requests
import tempfile
import time
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import logging
from requests.exceptions import RequestException
import boto3

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET_NAME = "audiobucket123456"
OPENAI_API_KEY = "sk-i0ujYh3oHfs9c1JsSDKQEruEUR_rwXRSYc7UvNJ4liT3BlbkFJwF5NvWsVKbz0KzoVw7UYCSOYCITxiPiigT2VB6R0sA"
LANGCHAIN_API_KEY="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"
LANGCHAIN_PERSONAL_API_KEY="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"
LANGCHAIN_TRACING_V2="true"

s3 = boto3.client('s3',
                  aws_access_key_id=AWS_ACCESS_KEY_ID,
                  aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

app = Flask(__name__)
socketio = SocketIO(app)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

transcript_text = ''
vector_store = None
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, openai_api_key=OPENAI_API_KEY)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

@app.route('/')
def index():
    return render_template('index.html')

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def update_vector_store(new_text):
    global vector_store
    chunks = get_text_chunks(new_text)
    if not chunks:
        logger.warning("No new text chunks to add to vector store.")
        return

    if vector_store is None:
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    else:
        vector_store.add_texts(chunks)

    logger.info("Vector store updated with new text chunks.")

def get_conversational_chain():
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

@socketio.on('audio_data')
def handle_audio_data(data):
    global transcript_text, vector_store

    headers = {
        'authorization': "070c2b457b1c4d26bd31e64323e0546c",
        'content-type': 'application/json'
    }

    try:
        response = requests.post('https://api.assemblyai.com/v2/transcript',
                                 json={"audio_url": data['audio_url']},
                                 headers=headers,
                                 timeout=30)

        response.raise_for_status()

        transcript_id = response.json()['id']
        polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"

        max_retries = 15
        retry_delay = 10

        for _ in range(max_retries):
            try:
                polling_response = requests.get(polling_endpoint, headers=headers, timeout=30)
                polling_response.raise_for_status()
                polling_response = polling_response.json()

                if polling_response['status'] == 'completed':
                    new_transcript = polling_response['text']
                    transcript_text += new_transcript + '\n'
                    update_vector_store(new_transcript)
                    emit('transcription_update', new_transcript)
                    break
                elif polling_response['status'] == 'error':
                    logger.error(f"Transcription error: {polling_response['error']}")
                    break
                else:
                    time.sleep(retry_delay)
            except RequestException as e:
                logger.error(f"Error polling AssemblyAI API: {str(e)}")
                time.sleep(retry_delay)
        else:
            logger.error("Max retries reached for polling AssemblyAI API")

    except RequestException as e:
        logger.error(f"Error in AssemblyAI API call: {str(e)}")

    try:
        with open('transcription.txt', 'w') as file:
            file.write(transcript_text)
    except IOError as e:
        logger.error(f"Error writing transcription to file: {str(e)}")

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            audio_file.save(temp_file.name)

            s3.upload_file(temp_file.name, AWS_S3_BUCKET_NAME, os.path.basename(temp_file.name))

        audio_url = s3.generate_presigned_url('get_object',
                                              Params={'Bucket': AWS_S3_BUCKET_NAME,
                                                      'Key': os.path.basename(temp_file.name)},
                                              ExpiresIn=5200)

        return jsonify({"audio_url": audio_url})
    except Exception as e:
        logger.error(f"Error handling audio upload: {str(e)}")
        return jsonify({"error": "An error occurred while processing the audio"}), 500

@app.route('/end_meeting', methods=['POST'])
def end_meeting():
    global vector_store
    if vector_store:
        try:
            vector_store.save_local("faiss_index")
            logger.info("Vector store saved to disk.")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            return jsonify({"status": "error", "message": "Failed to save meeting data."}), 500
    return jsonify({"status": "success", "message": "Meeting ended and vector store saved."})

@app.route('/chat', methods=['POST'])
def chat():
    global vector_store
    user_question = request.json.get('question', '')

    try:
        if vector_store is None:
            response = llm.predict(user_question)
            return jsonify({"reply": response})

        chain = get_conversational_chain()
        response = chain({"question": user_question})
        return jsonify({"reply": response["answer"]})
    except Exception as e:
        logger.error(f"Error in chat function: {str(e)}")
        return jsonify({"reply": "An error occurred while processing your question. Please try again."})

if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=5000, debug=True)