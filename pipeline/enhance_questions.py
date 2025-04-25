from pipeline.pipeline_utils.db_connections import get_mysql_connection, get_mongo_connection,DBConfig, save_to_mongodb

def get_questions_from_mongo(subject=None):
    #Read from MongoDB. Filter by subject if provided, return list of questions
    mongo_client = get_mongo_connection(DBConfig.MONGO_DB_NAME)
    questions = mongo_client.find_many(collection_name="questions", query={"subject": subject})
    return questions
 

def load_template(file_path):
    """Load the content of a template file."""
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: Template file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading template file {file_path}: {e}")
        return None





def main(system_prompt_template_path, user_prompt_template_path):
    generation_system_prompt = load_template(system_prompt_template_path)
    generation_usr_prompt = load_template(user_prompt_template_path)

    questions = get_questions_from_mongo(subject=None)

    for question in questions:
        print(question)


if __name__ == "__main__":
    main('generation_system_prompt_template_mc.txt', 'generation_usr_prompt_template_mc.txt')