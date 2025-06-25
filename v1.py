import streamlit as st
import openai
from PIL import Image
from dotenv import dotenv_values
import os
import base64
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import uuid

env = dotenv_values(".env")
### Secrets using Streamlit Cloud Mechanism
# https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
if 'QDRANT_URL' in st.secrets:
    env['QDRANT_URL'] = st.secrets['QDRANT_URL']
if 'QDRANT_API_KEY' in st.secrets:
    env['QDRANT_API_KEY'] = st.secrets['QDRANT_API_KEY']
###

# Zmienne
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072
QDRANT_COLLECTION_NAME = "notes"

# Tworzenie akordeonu w pasku bocznym
with st.sidebar.expander("Wprowadź klucz API OpenAI", expanded=True):
    api_key = st.text_input("Klucz API:", type="password")

    if api_key:
        st.success("Klucz jest OK")  

# Inicjalizacja klienta 
@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),  # Zmienione na os.getenv
        api_key=os.getenv("QDRANT_API_KEY")  # Zmienione na os.getenv
    )

qdrant_client = get_qdrant_client()

def assure_db_collection_exists():
    if not qdrant_client.collection_exists(QDRANT_COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )

def generate_image_description(client, uploaded_file):
    uploaded_file.seek(0)
    try:
        bytes_data = uploaded_file.getvalue()
        base64_image = base64.b64encode(bytes_data).decode('utf-8')
        file_type = uploaded_file.type.split('/')[-1]
        image_url = f"data:image/{file_type};base64,{base64_image}"

        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Stwórz opis obrazka w kilku słowach, co tam widzisz?"},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ]
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Wystąpił błąd przy generowaniu opisu: {str(e)}"

def generate_embeddings(client, description):
    try:
        result = client.embeddings.create(
            input=[description],
            model=EMBEDDING_MODEL,
        )
        embedding = result.data[0].embedding
        return embedding
    except Exception as e:
        return f"Wystąpił błąd przy generowaniu embeddingu: {str(e)}"

def add_note_to_db(note_text, uploaded_file, client):
    vector = generate_embeddings(client, note_text)

    uploaded_file.seek(0)
    bytes_data = uploaded_file.getvalue()
    base64_image = base64.b64encode(bytes_data).decode('utf-8')
    file_type = uploaded_file.type.split('/')[-1]
    image_url = f"data:image/{file_type};base64,{base64_image}"

    # Użycie UUID jako unikalnego identyfikatora
    note_id = str(uuid.uuid4())

    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=[
            PointStruct(
                id=note_id,
                vector=vector,
                payload={
                    "text": note_text,
                    "image": image_url
                },
            )
        ]
    )



def list_notes_from_db(query=None):
    try:
        if not query:
            notes = qdrant_client.scroll(collection_name=QDRANT_COLLECTION_NAME, limit=10)[0]
            return [
                {
                    "id": note.id,  
                    "text": note.payload["text"],
                    "image": note.payload.get("image"),
                } for note in notes
            ]
        else: 
            query_vector = generate_embeddings(client, query)
            notes = qdrant_client.search(
                collection_name=QDRANT_COLLECTION_NAME,
                query_vector=query_vector,
                limit=10,
            )
            return [
                {
                    "id": note.id,  
                    "text": note.payload["text"],
                    "image": note.payload.get("image"),
                } for note in notes
            ]
    except ValueError as e:
        print(f"Error: {e}")
        return []  # Możesz zwrócić pustą listę lub inne domyślne dane.
    
def delete_note_from_db(note_id):
    try:
        note_id_str = str(note_id)
        print(f"Próbuję usunąć notatkę o ID: {note_id_str}")

        # Sprawdzenie, czy notatka istnieje
        existing_notes = list_notes_from_db()
        existing_note_ids = [str(note["id"]) for note in existing_notes]

        if note_id_str not in existing_note_ids:
            st.warning(f"Notatka o ID {note_id_str} nie istnieje w bazie danych.")
            return

        # Użycie poprawnej metody usuwania
        qdrant_client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=[note_id_str]
        )
        st.success(f"Notatka o ID {note_id_str} została usunięta.")

        print(f"Notatka o ID {note_id_str} została pomyślnie usunięta.") 
    except Exception as e:
        print(f"Wystąpił błąd podczas usuwania notatki o ID {note_id}: {e}")
        st.error(f"Wystąpił błąd podczas usuwania notatki: {e}")

############################################################################################################################################
# Główna część aplikacji
############################################################################################################################################


if api_key:
    client = openai.OpenAI(api_key=api_key)
    assure_db_collection_exists()

    # Dodawanie nagłówka przed selectbox
    st.sidebar.markdown("# Wybierz opcję:")  

    # Tworzenie selectboxa
    selection = st.sidebar.selectbox("Galeria zdjęć:", ["Dodaj zdjęcie", "Wyszukiwarka zdjęć", "Moja Galeria"])

    # Resetowanie stanu sesji po zmianie zakładki
    if 'uploaded_files' not in st.session_state:
        st.session_state['uploaded_files'] = []

    # Resetujemy sesję przy zmianie zakładki
    if 'selected_option' not in st.session_state or st.session_state.selected_option != selection:
        st.session_state['uploaded_files'] = []
        st.session_state['selected_option'] = selection

############################################################################################################################################
        # Obsługa zakładki "Dodaj zdjęcie"
############################################################################################################################################
    if selection == "Dodaj zdjęcie":
        st.markdown("<h2 style='text-align: center; font-weight: bold;'>Dodaj zdjęcia do kolekcji</h2>", unsafe_allow_html=True) 
        st.markdown("<h5>Wczytaj zdjęcia (maks. 5)</h5>", unsafe_allow_html=True)  
        uploaded_files = st.file_uploader("", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        if uploaded_files:
            # Utworzymy kolumny dla przycisków
            col1, col2 = st.columns(2)

            # Przycisk generujący opisy dla wszystkich zdjęć
            with col1:
                if st.button("Generuj opisy dla wszystkich zdjęć"):
                    for uploaded_file in uploaded_files:
                        description = generate_image_description(client, uploaded_file)
                        if description and "Wystąpił błąd" not in description:
                            st.session_state[f"note_text_{uploaded_file.name}"] = description
                    st.success("Opisy zostały wygenerowane dla wszystkich zdjęć.")

            # Przycisk do zapisywania wszystkich zdjęć
            with col2:
                if st.button("Zapisz wszystkie zdjęcia"):
                    for uploaded_file in uploaded_files:
                        if f"note_text_{uploaded_file.name}" in st.session_state and st.session_state[f"note_text_{uploaded_file.name}"]:
                            add_note_to_db(note_text=st.session_state[f"note_text_{uploaded_file.name}"], uploaded_file=uploaded_file, client=client)  
                    st.success("Wszystkie notatki zostały zapisane.", icon="👍")
                    # Resetowanie przesłanych plików
                    uploaded_files = None  # Resetowanie, aby zdjęcia zniknęły

            # Wyświetlanie zdjęć i edytowanie notatek
            if uploaded_files:  # Sprawdzanie, czy są przesłane pliki
                for uploaded_file in uploaded_files:
                    # Wyświetlanie obrazu
                    image = Image.open(uploaded_file)
                    st.image(image, caption='Wczytane zdjęcie', use_container_width=True)

                    # Sprawdzanie i wyświetlanie opisu
                    note_key = f"note_text_{uploaded_file.name}"
                    if note_key in st.session_state:
                        description_text = st.session_state[note_key]
                        st.markdown(f"<div style='text-align: center; font-style: italic;'>{description_text}</div>", unsafe_allow_html=True)

                        # Dodawanie przerwy pomiędzy opisem a polem edycyjnym
                        st.markdown("<br>", unsafe_allow_html=True)

                        # Edytowanie notatki (jeśli istnieje)
                        st.session_state[note_key] = st.text_area("**Edytuj notatkę:**", value=st.session_state[note_key])

                    # Wydziel odstęp między zdjęciami
                    st.markdown("---")  # Oddzielenie zdjęć

                    

############################################################################################################################################
    # Obsługa zakładki "Wyszukiwarka zdjęć"
############################################################################################################################################
    elif selection == "Wyszukiwarka zdjęć":
        st.markdown("<h2 style='text-align: center; font-weight: bold;'>Dodaj zdjęcia do kolekcji</h2>", unsafe_allow_html=True) 
        query = st.text_input("Napisz czego szukasz:")
        if st.button("Szukaj"):
            notes = list_notes_from_db(query)
            if notes:
                cols = st.columns(3)
                for i, note in enumerate(notes):
                    with cols[i % 3]:
                        if note["image"]:
                            st.image(note["image"], caption="Miniaturka zdjęcia", use_container_width=True)
            else:
                st.write("Brak pasujących notatek.")

    
       
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################

        elif selection == "Moja Galeria":
        # Zainicjalizuj notes, jeśli nie jest ustawiony
            if 'notes' not in st.session_state:
                st.session_state.notes = []

            # Ładuj notatki, jeśli są puste (tylko raz)
            if not st.session_state.notes:
                st.session_state.notes = list_notes_from_db()

            st.markdown("<h2 style='text-align: center; font-weight: bold;'>Galeria zdjęć</h2>", unsafe_allow_html=True)

            # Używamy już załadowanych notatek w sesji
            if st.session_state.notes:
                # Grupa notatek w rzędy po trzy zdjęcia
                rows = []
                for i in range(0, len(st.session_state.notes), 3):
                    rows.append(st.session_state.notes[i:i + 3])  # Grupowanie zdjęć po trzy

                # Wyświetlanie zdjęć w tabeli (3 kolumny)
                for row in rows:
                    cols = st.columns(3)
                    for i, note in enumerate(row):
                        with cols[i]:
                            if note["image"]:
                                # Wyświetlanie zdjęcia o stałej szerokości kontenera
                                st.image(note["image"], use_container_width=True)  # Użycie szerokości kontenera
                                # Dodajemy margines oraz kontener dla przycisków
                                st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)  # Wyśrodkowanie

                                # Przycisk do usuwania zdjęcia
                                if st.button("Usuń", key=f"delete_{note['id']}"):
                                    print(f"Próbuję usunąć notatkę o ID: {note['id']}")
                                    delete_note_from_db(note['id'])  # Usunięcie notatki
                                    st.session_state.notes = list_notes_from_db()  # Odświeżanie notatek
                                    st.success("Zdjęcie zostało usunięte.")  # Informacja zwrotna o usunięciu

                                st.markdown("</div>", unsafe_allow_html=True)  # Zamknięcie kontenera
                    # Dodać odstęp po każdym rzędzie
                    st.markdown("<br>", unsafe_allow_html=True)