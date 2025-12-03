from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import json
import os

def create_mongo_client(uri: str):
    """
    Creates and returns a MongoDB client.
    :param uri: MongoDB connection string
    :return: MongoClient instance
    """
    try:
        client = MongoClient(uri, server_api=ServerApi('1'))
        client.admin.command('ping')  # Verify connection
        print("Connected to MongoDB successfully!")
        return client
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

def insert_data(client: MongoClient, database_name: str, collection_name: str, data: dict):
    """
    Inserts a document into a MongoDB collection.
    :param client: MongoDB client
    :param database_name: Name of the database
    :param collection_name: Name of the collection
    :param data: Dictionary containing the document to insert
    """
    try:
        db = client[database_name]
        collection = db[collection_name]
        result = collection.insert_one(data)
        print(f"Data inserted successfully with ID: {result.inserted_id}")
    except Exception as e:
        print(f"Error inserting data: {e}")

def get_data(client: MongoClient, database_name: str, collection_name: str, query: dict):
    """
    Retrieves documents from a MongoDB collection based on a query.
    :param client: MongoDB client
    :param database_name: Name of the database
    :param collection_name: Name of the collection
    :param query: Dictionary specifying the query filter
    :return: List of matching documents
    """
    try:
        db = client[database_name]
        collection = db[collection_name]
        documents = list(collection.find(query, {"_id": 0}))  # Excluding _id field
        return documents
    except Exception as e:
        print(f"Error retrieving data: {e}")
        return []
    
def get_user_ids(client, db, collection):
    # Fetch distinct user_ids from the chatHistory collection
    collection = db[collection]
    user_ids = collection.distinct("user_id")
    return user_ids

def get_last_10_conversations(client, db, collection_name, user_id):
    collection = db[collection_name]
    user_conversations = collection.find_one({"user_id": user_id})

    if user_conversations and "conversation" in user_conversations:
        # Extract the last 10 conversations
        chat_history = user_conversations["conversation"][-10:]
        
        # Ensure that each conversation has the required fields: role, content, timestamp
        formatted_history = []
        for convo in chat_history:
            formatted_convo = {}
            formatted_convo["role"] = convo.get("role", "unknown")  # Default to "unknown" if role is missing
            formatted_convo["content"] = convo.get("content", "")  # Default to empty string if content is missing
            formatted_convo["timestamp"] = convo.get("timestamp", "unknown")  # Default to "unknown" if timestamp is missing
            formatted_history.append(formatted_convo)

        return formatted_history
    else:
        return []  # If no conversations found, return an empty list


# Function to store a conversation (user's question + assistant's response)
def store_conversation(client, db, collection_name, user_id, chat_history, model_response, model_timestamp):
    collection = db[collection_name]

    # Create the assistant's message
    assistant_message = {
        "role": "assistant", 
        "content": model_response, 
        "timestamp": model_timestamp
    }
    
    # Append the new assistant message to the chat history
    chat_history.append(assistant_message)

    # Now update the conversation in the database
    # Use upsert=True to ensure the conversation is updated if it exists or created if it's a new user
    collection.update_one(
        {"user_id": user_id},  # Filter by user_id
        {"$set": {"conversation": chat_history}},  # Update the conversation field
        upsert=True  # Create a new document if the user doesn't exist
    )

    
def get_conversation(client: MongoClient, database_name: str, collection_name: str, user_id: str):
    """
    Retrieves the conversation history for a specific user.
    Returns an empty list if no conversation exists.
    """
    try:
        db = client[database_name]
        collection = db[collection_name]
        document = collection.find_one({"user_id": user_id}, {"_id": 0, "conversation": 1})
        return document["conversation"] if document else []
    except Exception as e:
        print(f"Error retrieving conversation: {e}")
        return []

    
def update_conversation(client: MongoClient, database_name: str, collection_name: str, user_id: str, chat_history: list):
    """
    Updates the conversation history for a user.
    If the user does not exist, it creates a new entry.
    """
    try:
        db = client[database_name]
        collection = db[collection_name]
        collection.update_one(
            {"user_id": user_id},
            {"$set": {"conversation": chat_history}},
            upsert=True  # Creates if not exists
        )
    except Exception as e:
        print(f"Error updating conversation: {e}")


def delete_data(client: MongoClient, database_name: str, collection_name: str, query: dict):
    """
    Deletes documents from a MongoDB collection based on a query.
    :param client: MongoDB client
    :param database_name: Name of the database
    :param collection_name: Name of the collection
    :param query: Dictionary specifying the query filter
    :return: Deletion result
    """
    try:
        db = client[database_name]
        collection = db[collection_name]
        result = collection.delete_many(query)
        print(f"Deleted {result.deleted_count} documents.")
        return result
    except Exception as e:
        print(f"Error deleting data: {e}")
        return None

def read_config(file_path: str):
    """
    Reads a JSON configuration file.
    :param file_path: Path to the JSON file
    :return: Dictionary containing the configuration data
    """
    try:
        conf_dict = {}
        with open(file_path, 'r') as file:
            config = json.load(file)
        for m in config:
            conf_dict[m['id']] = m
        return conf_dict
    except Exception as e:
        print(f"Error reading config file: {e}")
        return None

def mongo_creds_from_config():
    """
    Extracts MongoDB credentials from environment variables or configuration dictionary.
    :return: MongoDB connection string
    """
    try:
        # First check for environment variable (Docker usage)
        env_uri = os.getenv("MONGO_URI")
        if env_uri:
            return env_uri

        # Fallback to config file (Local usage)
        if os.path.exists("./config.json"):
            conf_dict = read_config("./config.json")
            if 'mongodb' in conf_dict and 'mongo_uri' in conf_dict['mongodb']:
                 return conf_dict['mongodb']['mongo_uri']
            
            # Legacy support (optional, or just remove)
            # mongo_cred = conf_dict['mongodb']
            # usrname = mongo_cred['mongo_usrname']
            # pwd = mongo_cred['mongo_pwd']
            # cluster = mongo_cred['cluster_name']
            # uri = f"mongodb+srv://{usrname}:{pwd}@{cluster}.mongodb.net/?retryWrites=true&w=majority"
            # return uri
            
            print("Config file found but missing 'mongo_uri'.")
            return None
        
        print("No MongoDB configuration found (MONGO_URI env var or config.json).")
        return None
    except Exception as e:
        print(f"Error extracting MongoDB credentials: {e}")
        return None
