import os, openai, logging, configparser,json,re
from chain_of_density.chat_completion import make_chat_completion_request
from chain_of_density.msg_templates import create_system_message

# # Load config file and setup variables
# here = os.path.abspath(os.path.dirname(__file__))
# config = configparser.ConfigParser()
# config.read("config.ini")

def load_config():
    """Load config file and setup variables"""
    #here = os.path.abspath(os.path.dirname(__file__))
    config = configparser.ConfigParser()
    config.read("config.ini")
    return config
def perform_checks():
    #logger.info("being checks")
    # Check if OPENAI_API_KEY is set
    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")


    #logger.info("Checks passed")


def load_file(input_file_path):
    if os.path.exists(input_file_path):
        # Load input file
        with open(input_file_path, "r") as f:
            return f.read()
    elif os.path.exists(os.path.join(here, input_file_path)):
        # Load input file from main folder
        with open(f"{os.path.join(here, input_file_path)}", "r") as f:
            return f.read()
    else:
        raise FileNotFoundError(f"Input file not found at path: {input_file_path}")

def clean_json_string(json_string):
    pattern = r'^```json\s*(.*?)\s*```$'
    cleaned_string = re.sub(pattern, r'\1', json_string, flags=re.DOTALL)
    return cleaned_string.strip()
def get_final_summarize(json_data): 
    """Get the final summarize from the json data
    Json data is list of dictionaries
        "Missing_Entities":
        "Denser_Summary":
    Returns:
        str: final summarize (Combination of last Missing_Entities and Denser_Summary)
    """
    final_summarize = ""
    last_item = json_data[-1]
    # if "Missing_Entities" in last_item:
    #     final_summarize += last_item["Missing_Entities"]
    if "Denser_Summary" in last_item:
        final_summarize += last_item["Denser_Summary"]
    return final_summarize
def get_summarize(input_text):
    """Get the summarize from the given document"""
    config = load_config()
    msg = create_system_message(config)
    #input_text = load_file(doc)
    msg.append(
        {
            "role": "user",
            "content": f"Đây là văn bản đầu vào để bạn tóm tắt bằng cách sử dụng phương pháp 'Missing_Entities' và 'Denser_Summary':\n\n{input_text}",
        }
    )
    completion = make_chat_completion_request(config, msg, n=1)
    content = completion.choices[0].message.content
    content = clean_json_string(content)
    json_data = json.loads(content)
    print(json_data[-1])
    return get_final_summarize(json_data)
def main():
    #logger.info("main() starting")

    # Perform sense checks
    perform_checks()

    # Get system message
    msg = create_system_message(config)

    # Load input file
    input_text = load_file("corpus/benh-gout")

    msg.append(
        {
            "role": "user",
            "content": f"Đây là văn bản đầu vào để bạn tóm tắt bằng cách sử dụng phương pháp 'Missing_Entities' và 'Denser_Summary':\n\n{input_text}",
        }
    )

    completion = make_chat_completion_request(config, msg, n=1)
    # print(completion)
    # print()
    content = completion.choices[0].message.content
    content = clean_json_string(content)
    json_data = json.loads(content)
    with open("output.json", "w") as f:
        json.dump(json_data, f, indent=4)
    # Write the output to file
    content_final = get_final_summarize(json_data)
    with open(config["DEFAULT"]["OUTPUT_FILE"], "w") as f:
        f.write(content_final)

    return content_final


if __name__ == "__main__":
    #logger.info("triggering main() execution")
    config = load_config()
    #print(main())
    input_text = load_file("corpus/benh-gout")
    print(get_summarize(input_text))
